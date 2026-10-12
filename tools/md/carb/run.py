"""Carb MD run: short EM+NVT+NPT by default; 200 ns only if explicitly selected."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, TextIO

from tools.md.carb.commands import build_grompp_command, build_mdrun_command
from tools.md.carb.constants import (
    DEFAULT_PROTOCOL,
    INPUT_MD_SUBDIR,
    OUTPUT_SUBDIR,
    PRODUCTION_NSTEPS,
    PRODUCTION_PROTOCOL,
    PROTOCOL_ID,
    REPLICA_SEEDS,
)
from tools.md.carb.mdp import (
    MdpError,
    assert_carb_launch_nsteps,
    launch_nsteps_cap,
    materialize_carb_protocol,
    normalize_carb_protocol,
)
from tools.md.carb.prepare import CarbPrepareError, _input_md_dir
from tools.md.carb.scalars import default_n_replicas, mdp_only, normalize_carb_scalars
from tools.md.carb.status import mark, replica_record
from tools.md.commands import (
    GROMPP_TIMEOUT_SECONDS,
    MDRUN_TIMEOUT_SECONDS,
    MdCommandError,
    normalize_gmx_executable,
    normalize_nb,
)
from tools.md.environment import check_md_environment
from tools.md.prepare import MdPrepareError, resolve_structure_path, resolve_topology_path
from tools.md.run import MdError, _run_gmx, confirm_md_launch

RunFn = Callable[..., Any]

# Production mdrun is long; keep a high cap but tests mock gmx.
MDRUN_TIMEOUT_200NS_SECONDS = 7 * 24 * 3600


class CarbRunError(MdError):
    """Carb MD launch failed."""


def _output_dir(project: Path) -> Path:
    dest = (project / "output" / OUTPUT_SUBDIR).resolve()
    try:
        dest.relative_to(project.resolve())
    except ValueError as exc:
        raise CarbRunError(f"carb output escapes project: {dest}") from exc
    return dest


def _replica_dir(output_dir: Path, protocol: str, replica_id: int, n_replicas: int) -> Path:
    if protocol == PRODUCTION_PROTOCOL or n_replicas > 1:
        return output_dir / f"rep{replica_id}"
    return output_dir / "smoke"


def run_carb(
    project_dir: str | Path,
    *,
    protocol: str = DEFAULT_PROTOCOL,
    approve: bool = False,
    assume_yes: bool = False,
    nsteps: int | None = None,
    dt: float | None = None,
    ref_t: float | None = None,
    n_replicas: int | None = None,
    gpu: bool = False,
    structure: str | Path | None = None,
    topology: str | Path | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    run_fn: RunFn | None = None,
) -> dict[str, Any]:
    try:
        protocol = normalize_carb_protocol(protocol)
        scalars = normalize_carb_scalars(
            {
                k: v
                for k, v in {
                    "nsteps": nsteps,
                    "dt": dt,
                    "ref_t": ref_t,
                    "n_replicas": n_replicas,
                }.items()
                if v is not None
            }
        )
        nb = normalize_nb("gpu" if gpu else "cpu")
    except (MdpError, MdCommandError) as exc:
        raise CarbRunError(str(exc)) from exc

    replicas_n = default_n_replicas(protocol, scalars.get("n_replicas") if "n_replicas" in scalars else None)
    project = Path(project_dir).expanduser().resolve()
    if not project.is_dir():
        raise CarbRunError(f"Project directory does not exist: {project}")

    try:
        structure_path = resolve_structure_path(
            project, structure or "input/md/system.gro"
        )
        topology_path = resolve_topology_path(
            project, topology or "input/md/system.top"
        )
    except MdPrepareError as exc:
        # Fall back to unsolvated complex.gro written by prepare without gmx.
        try:
            input_dir = _input_md_dir(project)
            alt_gro = input_dir / "complex.gro"
            alt_top = input_dir / "system.top"
            if alt_gro.is_file() and alt_top.is_file() and structure is None:
                structure_path = alt_gro
                topology_path = alt_top
            else:
                raise CarbRunError(str(exc)) from exc
        except CarbPrepareError as inner:
            raise CarbRunError(str(exc) or str(inner)) from exc

    output_dir = _output_dir(project)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = check_md_environment()
    gmx_path = env.get("gmx_path")
    mdp_overrides = mdp_only(scalars)

    replica_rows = [
        replica_record(i + 1, seed=REPLICA_SEEDS[i % len(REPLICA_SEEDS)], status="pending")
        for i in range(replicas_n)
    ]
    planned = {
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol,
        "runnable": True,
        "executable": gmx_path or (env.get("gmx_binary") or "gmx"),
        "structure": str(structure_path),
        "topology": str(topology_path),
        "overrides": scalars,
        "output_dir": str(output_dir),
        "nb": nb,
        "gpu": bool(gpu),
        "n_replicas": replicas_n,
        "max_launch_nsteps": launch_nsteps_cap(protocol),
        "default_is_200ns": False,
    }

    result: dict[str, Any] = {
        "ok": False,
        "approved": bool(approve),
        "skipped": False,
        "success": False,
        "engine": "gmx",
        "package": "gromacs",
        "kind": "carb-md",
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol,
        "nb": nb,
        "gpu": bool(gpu),
        "n_replicas": replicas_n,
        "project_dir": str(project),
        "output_dir": str(output_dir),
        "environment": env,
        "planned": planned,
        "mdp": None,
        "replicas": replica_rows,
        "warnings": [],
        "errors": [],
        "message": "",
        "invented_trajectory": False,
        "launched_200ns": False,
    }

    # Materialize replica 0 mdp first so dry-run / no-approve still writes templates.
    try:
        mdp_meta = materialize_carb_protocol(
            protocol, output_dir / "mdp", overrides=mdp_overrides, replica_index=0
        )
    except MdpError as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        mark(
            output_dir,
            status="failed",
            protocol=protocol,
            replicas=replica_rows,
            message=str(exc),
        )
        return result
    result["mdp"] = mdp_meta
    planned["max_nsteps"] = mdp_meta.get("max_nsteps")

    is_200ns = (
        protocol == PRODUCTION_PROTOCOL
        or int(mdp_meta.get("max_nsteps") or 0) >= PRODUCTION_NSTEPS
    )
    result["would_launch_200ns"] = bool(is_200ns)

    if not approve:
        result["skipped"] = True
        result["ok"] = True
        result["message"] = (
            "Carb MD not launched (missing --approve-md). "
            "Environment, templates, and human mdp were checked; gmx was not run. "
            f"Default protocol is {DEFAULT_PROTOCOL}, not {PRODUCTION_PROTOCOL}."
        )
        mark(
            output_dir,
            status="skipped",
            protocol=protocol,
            replicas=replica_rows,
            message=result["message"],
            extra={"would_launch_200ns": is_200ns},
        )
        return result

    if not confirm_md_launch(
        planned, assume_yes=assume_yes, stdin=stdin, stdout=stdout
    ):
        result["skipped"] = True
        result["approved"] = False
        result["ok"] = True
        result["message"] = "Human declined or confirmation unavailable — carb MD will not run."
        mark(
            output_dir,
            status="skipped",
            protocol=protocol,
            replicas=replica_rows,
            message=result["message"],
        )
        return result

    try:
        assert_carb_launch_nsteps(mdp_meta.get("max_nsteps"), protocol)
    except MdpError as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        mark(
            output_dir,
            status="failed",
            protocol=protocol,
            replicas=replica_rows,
            message=str(exc),
        )
        return result

    if protocol != PRODUCTION_PROTOCOL and is_200ns:
        result["errors"].append(
            f"Refusing to launch 200 ns on protocol {protocol}. "
            f"Select --protocol {PRODUCTION_PROTOCOL} explicitly."
        )
        result["message"] = result["errors"][-1]
        mark(
            output_dir,
            status="failed",
            protocol=protocol,
            replicas=replica_rows,
            message=result["message"],
        )
        return result

    if not gmx_path:
        result["errors"].append(
            "gmx not found on PATH (carb doctor never auto-installs GROMACS)."
        )
        result["message"] = result["errors"][-1]
        mark(
            output_dir,
            status="failed",
            protocol=protocol,
            replicas=replica_rows,
            message=result["message"],
        )
        return result

    try:
        normalize_gmx_executable(str(gmx_path))
    except MdCommandError as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        mark(
            output_dir,
            status="failed",
            protocol=protocol,
            replicas=replica_rows,
            message=str(exc),
        )
        return result

    runner = run_fn or _run_gmx
    ndx = structure_path.parent / "index.ndx"
    mark(
        output_dir,
        status="running",
        protocol=protocol,
        replicas=replica_rows,
        message="carb MD running",
        extra={"launched_200ns": protocol == PRODUCTION_PROTOCOL},
    )

    replica_results: list[dict[str, Any]] = []
    failed = False
    for i in range(replicas_n):
        replica_id = i + 1
        seed = REPLICA_SEEDS[i % len(REPLICA_SEEDS)]
        rep_dir = _replica_dir(output_dir, protocol, replica_id, replicas_n)
        logs_dir = rep_dir / "logs"
        mdp_dir = rep_dir / "mdp"
        rep_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        replica_rows[i]["status"] = "running"
        replica_rows[i]["output_dir"] = str(rep_dir)
        mark(
            output_dir,
            status="running",
            protocol=protocol,
            replicas=replica_rows,
            message=f"replica {replica_id} running",
        )
        try:
            meta = materialize_carb_protocol(
                protocol, mdp_dir, overrides=mdp_overrides, replica_index=i
            )
            _run_carb_chain(
                executable=str(gmx_path),
                output_dir=rep_dir,
                logs_dir=logs_dir,
                structure_path=structure_path,
                topology_path=topology_path,
                mdp_meta=meta,
                nb=nb,
                ndx=ndx if ndx.is_file() else None,
                run_fn=runner,
                production=protocol == PRODUCTION_PROTOCOL,
            )
            replica_rows[i]["status"] = "completed"
            replica_rows[i]["message"] = "ok"
            replica_results.append({"id": replica_id, "ok": True, "dir": str(rep_dir), "seed": seed})
        except (MdError, MdCommandError, OSError, subprocess.TimeoutExpired) as exc:
            failed = True
            replica_rows[i]["status"] = "failed"
            replica_rows[i]["message"] = str(exc)
            replica_results.append({"id": replica_id, "ok": False, "error": str(exc), "seed": seed})
            result["errors"].append(f"replica {replica_id}: {exc}")
            # Do not invent a trajectory for this replica; continue others.
            continue

    result["replicas"] = replica_rows
    result["replica_results"] = replica_results
    result["launched_200ns"] = protocol == PRODUCTION_PROTOCOL
    if failed:
        result["ok"] = False
        result["success"] = False
        result["message"] = (
            "One or more replicas failed; trajectories were not invented. "
            + "; ".join(result["errors"])
        )
        mark(
            output_dir,
            status="failed",
            protocol=protocol,
            replicas=replica_rows,
            message=result["message"],
            extra={"launched_200ns": result["launched_200ns"]},
        )
        return result

    result["ok"] = True
    result["success"] = True
    result["message"] = (
        f"Carb MD finished ({protocol}, replicas={replicas_n}) under {output_dir}"
    )
    mark(
        output_dir,
        status="completed",
        protocol=protocol,
        replicas=replica_rows,
        message=result["message"],
        extra={"launched_200ns": result["launched_200ns"]},
    )
    return result


def _run_carb_chain(
    *,
    executable: str,
    output_dir: Path,
    logs_dir: Path,
    structure_path: Path,
    topology_path: Path,
    mdp_meta: dict[str, Any],
    nb: str,
    ndx: Path | None,
    run_fn: RunFn,
    production: bool,
) -> None:
    files = {item["template_id"]: Path(item["dest"]) for item in mdp_meta.get("files") or []}
    current_gro = structure_path
    current_cpt: Path | None = None
    stages = [("em", "em"), ("nvt", "nvt"), ("npt", "npt")]
    if production:
        stages.append(("md_200ns", "md"))
    for template_id, stage in stages:
        mdp = files.get(template_id)
        if mdp is None:
            continue
        tpr = output_dir / f"{stage}.tpr"
        deffnm = output_dir / stage
        timeout = (
            MDRUN_TIMEOUT_200NS_SECONDS if stage == "md" else MDRUN_TIMEOUT_SECONDS
        )
        grompp = build_grompp_command(
            executable=executable,
            mdp=mdp,
            gro=current_gro,
            top=topology_path,
            tpr=tpr,
            cpt=current_cpt if current_cpt and current_cpt.is_file() else None,
            maxwarn=2,
        )
        if ndx is not None:
            grompp.extend(["-n", str(ndx)])
        proc = run_fn(
            grompp,
            cwd=output_dir,
            log_path=logs_dir / f"{stage}_grompp.log",
            timeout=GROMPP_TIMEOUT_SECONDS,
        )
        if getattr(proc, "returncode", 1) != 0 or not tpr.is_file():
            raise CarbRunError(
                f"gmx grompp ({stage}) exited {getattr(proc, 'returncode', 'n/a')}; "
                f"see {logs_dir / f'{stage}_grompp.log'}"
            )
        mdrun = build_mdrun_command(
            executable=executable, tpr=tpr, deffnm=deffnm, nb=nb
        )
        proc = run_fn(
            mdrun,
            cwd=output_dir,
            log_path=logs_dir / f"{stage}_mdrun.log",
            timeout=timeout,
        )
        if getattr(proc, "returncode", 1) != 0:
            raise CarbRunError(
                f"gmx mdrun ({stage}) exited {getattr(proc, 'returncode', 'n/a')}; "
                f"see {logs_dir / f'{stage}_mdrun.log'}"
            )
        gro = output_dir / f"{stage}.gro"
        if gro.is_file():
            current_gro = gro
        cpt = output_dir / f"{stage}.cpt"
        if cpt.is_file():
            current_cpt = cpt
