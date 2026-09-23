"""MD pipeline + CLI — separate human approval from REINVENT.

``python -m tools.md --approve-md`` is the only way this module launches
GROMACS. The REINVENT executor never imports this package.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from tools import dumps_pretty, load_agent_config
from tools.md.analyze import parse_xvg, write_rmsd_csv, write_rmsf_csv, xy_stats
from tools.md.commands import (
    ANALYSIS_TIMEOUT_SECONDS,
    ALLOWED_ACPYPE_BINARIES,
    ALLOWED_GMX_SUBCOMMANDS,
    GROMPP_TIMEOUT_SECONDS,
    MDRUN_TIMEOUT_MD2NS_SECONDS,
    MDRUN_TIMEOUT_SECONDS,
    RMS_STDIN,
    RMSF_STDIN,
    MdCommandError,
    build_grompp_command,
    build_mdrun_command,
    build_rms_command,
    build_rmsf_command,
    normalize_gmx_executable,
    normalize_nb,
)
from tools.md.complex import ComplexPrepError, prepare_complex
from tools.md.environment import ALLOWED_GMX_BINARIES, check_md_environment
from tools.md.mdp import (
    ALLOWED_PROTOCOLS,
    DEFAULT_PROTOCOL,
    MAX_LAUNCH_NSTEPS,
    RUNNABLE_PROTOCOLS,
    MdpError,
    assert_launch_nsteps,
    launch_nsteps_cap,
    materialize_protocol,
    normalize_overrides,
    normalize_protocol,
)
from tools.md.prepare import MdPrepareError, resolve_structure_path, resolve_topology_path

OUTPUT_SUBDIR = "md"


class MdError(ValueError):
    """Sandbox, approval, mdp, or GROMACS failure."""


def run_md(
    project_dir: str | Path,
    *,
    structure: str | Path,
    topology: str | Path,
    protocol: str = DEFAULT_PROTOCOL,
    approve: bool = False,
    assume_yes: bool = False,
    nsteps: int | None = None,
    dt: float | None = None,
    ref_t: float | None = None,
    gpu: bool = False,
    agent_config: dict[str, Any] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> dict[str, Any]:
    """Validate, materialize mdp, optionally run short MD, write RMSD/RMSF."""
    stdout = sys.stdout if stdout is None else stdout
    cfg = (agent_config or {}).get("md") or {}
    try:
        protocol = normalize_protocol(protocol)
        overrides = _collect_overrides(nsteps=nsteps, dt=dt, ref_t=ref_t, cfg=cfg)
        nb = normalize_nb("gpu" if gpu else "cpu")
    except (MdpError, MdCommandError) as exc:
        raise MdError(str(exc)) from exc

    project = Path(project_dir).expanduser().resolve()
    if not project.is_dir():
        raise MdError(f"Project directory does not exist: {project}")

    try:
        structure_path = resolve_structure_path(project, structure)
        topology_path = resolve_topology_path(project, topology)
    except MdPrepareError as exc:
        raise MdError(str(exc)) from exc

    output_dir = _md_output_dir(project)
    output_dir.mkdir(parents=True, exist_ok=True)
    mdp_dir = output_dir / "mdp"
    logs_dir = output_dir / "logs"
    mdp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = check_md_environment()
    gmx_path = env.get("gmx_path")
    planned = {
        "protocol": protocol,
        "runnable": protocol in RUNNABLE_PROTOCOLS,
        "executable": gmx_path or (env.get("gmx_binary") or "gmx"),
        "structure": str(structure_path),
        "topology": str(topology_path),
        "overrides": overrides,
        "output_dir": str(output_dir),
        "nb": nb,
        "gpu": bool(gpu),
        "max_launch_nsteps": launch_nsteps_cap(protocol),
    }

    result: dict[str, Any] = {
        "ok": False,
        "approved": bool(approve),
        "skipped": False,
        "success": False,
        "engine": "gmx",
        "package": "gromacs",
        "modality": "molecular dynamics",
        "kind": "md simulation",
        "protocol": protocol,
        "nb": nb,
        "gpu": bool(gpu),
        "project_dir": str(project),
        "output_dir": str(output_dir),
        "table_present": False,
        "rmsd_csv": None,
        "rmsf_csv": None,
        "n_frames": 0,
        "rmsd": None,
        "rmsf": None,
        "environment": env,
        "planned": planned,
        "mdp": None,
        "command_templates": [],
        "warnings": [],
        "errors": [],
        "message": "",
    }

    try:
        mdp_meta = materialize_protocol(protocol, mdp_dir, overrides=overrides)
    except MdpError as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        _write_result_json(output_dir, result)
        return result
    result["mdp"] = mdp_meta
    planned["max_nsteps"] = mdp_meta.get("max_nsteps")
    planned["mdp_files"] = [item.get("dest") for item in mdp_meta.get("files") or []]

    try:
        result["command_templates"] = _command_templates(
            executable=str(planned["executable"]),
            output_dir=output_dir,
            structure_path=structure_path,
            topology_path=topology_path,
            mdp_meta=mdp_meta,
            nb=nb,
        )
    except (MdCommandError, MdError) as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        _write_result_json(output_dir, result)
        return result

    if not approve:
        result["skipped"] = True
        result["ok"] = True
        result["message"] = (
            "MD not launched (missing --approve-md). "
            "Environment, paths, and human mdp templates were checked; "
            "gmx was not run."
        )
        _write_result_json(output_dir, result)
        return result

    if not confirm_md_launch(
        planned, assume_yes=assume_yes, stdin=stdin, stdout=stdout
    ):
        result["skipped"] = True
        result["approved"] = False
        result["ok"] = True
        result["message"] = (
            "Human declined or confirmation unavailable — MD will not run."
        )
        _write_result_json(output_dir, result)
        return result

    if protocol not in RUNNABLE_PROTOCOLS:
        result["errors"].append(
            "Production MD (100 ns / experiments/md.mdp) is not launched by "
            "this module. Use --protocol em-nvt for the smoke-test chain. "
            "Production / MM-PBSA stay a later phase."
        )
        result["message"] = result["errors"][-1]
        _write_result_json(output_dir, result)
        return result

    try:
        assert_launch_nsteps(mdp_meta.get("max_nsteps"), protocol=protocol)
    except MdpError as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        _write_result_json(output_dir, result)
        return result

    if not gmx_path:
        result["errors"].append(
            "gmx not found on PATH (MD environment check is separate from "
            "REINVENT; this module does not auto-install)."
        )
        result["message"] = result["errors"][-1]
        _write_result_json(output_dir, result)
        return result

    try:
        normalize_gmx_executable(str(gmx_path))
    except MdCommandError as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        _write_result_json(output_dir, result)
        return result

    started = time.perf_counter()
    try:
        _run_protocol(
            result,
            executable=str(gmx_path),
            output_dir=output_dir,
            logs_dir=logs_dir,
            structure_path=structure_path,
            topology_path=topology_path,
            mdp_meta=mdp_meta,
            protocol=protocol,
            nb=nb,
        )
    except (MdError, MdCommandError, OSError, subprocess.TimeoutExpired) as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        _write_result_json(output_dir, result)
        return result
    runtime = time.perf_counter() - started
    result["runtime_seconds"] = round(runtime, 3)

    _write_analysis_tables(result, output_dir=output_dir)
    result["ok"] = True
    if result["table_present"]:
        result["success"] = True
        result["message"] = (
            f"Wrote MD tables under {output_dir} "
            f"(rmsd={result.get('rmsd_csv')}, rmsf={result.get('rmsf_csv')})"
        )
    else:
        result["message"] = (
            "MD chain finished but no RMSD/RMSF table was written "
            "(missing trajectory or gmx rms/rmsf output)."
        )
        result["errors"].append(result["message"])
    _write_result_json(output_dir, result)
    return result


def confirm_md_launch(
    planned: dict[str, Any],
    *,
    assume_yes: bool = False,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> bool:
    """Print the predefined MD job and require interactive confirmation."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    print("\n[Human approval required — MD / GROMACS]", file=stdout)
    print("The agent is ready to run this predefined MD job:", file=stdout)
    print(f"  protocol: {planned.get('protocol')} (runnable={planned.get('runnable')})", file=stdout)
    print(f"  executable: {planned.get('executable')}", file=stdout)
    print(f"  structure: {planned.get('structure')}", file=stdout)
    print(f"  topology:  {planned.get('topology')}", file=stdout)
    print(f"  overrides: {planned.get('overrides')}", file=stdout)
    print(f"  nsteps:    {planned.get('max_nsteps')} (launch cap {planned.get('max_launch_nsteps')})", file=stdout)
    print(f"  nb:        {planned.get('nb')} (gpu={planned.get('gpu')})", file=stdout)
    print(f"  output:    {planned.get('output_dir')}", file=stdout)
    print(
        "This is independent of REINVENT (--approve-run / --approve-dock do not apply here).",
        file=stdout,
    )
    print(
        "mdp files are human templates with allowlisted scalar overrides only.",
        file=stdout,
    )
    if assume_yes:
        print("Proceeding due to --yes.", file=stdout)
        return True
    if not hasattr(stdin, "isatty") or not stdin.isatty():
        print(
            "Non-interactive stdin: re-run with --approve-md --yes to launch.",
            file=stdout,
        )
        return False
    print("Proceed? [y/N]: ", end="", file=stdout, flush=True)
    reply = stdin.readline().strip().lower()
    return reply in ("y", "yes")


def _collect_overrides(
    *,
    nsteps: int | None,
    dt: float | None,
    ref_t: float | None,
    cfg: dict[str, Any],
) -> dict[str, int | float]:
    raw: dict[str, Any] = {}
    cfg_nsteps = cfg.get("default_nsteps")
    if nsteps is not None:
        raw["nsteps"] = nsteps
    elif cfg_nsteps not in (None, ""):
        raw["nsteps"] = cfg_nsteps
    if dt is not None:
        raw["dt"] = dt
    if ref_t is not None:
        raw["ref_t"] = ref_t
    return normalize_overrides(raw)


def _command_templates(
    *,
    executable: str,
    output_dir: Path,
    structure_path: Path,
    topology_path: Path,
    mdp_meta: dict[str, Any],
    nb: str = "cpu",
) -> list[list[str]]:
    files = {item["template_id"]: Path(item["dest"]) for item in mdp_meta.get("files") or []}
    templates: list[list[str]] = []
    if "minimization" in files:
        templates.append(
            build_grompp_command(
                executable=executable,
                mdp=files["minimization"],
                gro=structure_path,
                top=topology_path,
                tpr=output_dir / "em.tpr",
            )
        )
        templates.append(
            build_mdrun_command(
                executable=executable,
                tpr=output_dir / "em.tpr",
                deffnm=output_dir / "em",
                nb=nb,
            )
        )
    gro_for_nvt = output_dir / "em.gro" if "minimization" in files else structure_path
    nvt_mdp = files.get("nvt_eq") or files.get("nvt")
    if nvt_mdp is not None:
        templates.append(
            build_grompp_command(
                executable=executable,
                mdp=nvt_mdp,
                gro=gro_for_nvt,
                top=topology_path,
                tpr=output_dir / "nvt.tpr",
            )
        )
        templates.append(
            build_mdrun_command(
                executable=executable,
                tpr=output_dir / "nvt.tpr",
                deffnm=output_dir / "nvt",
                nb=nb,
            )
        )
    md_mdp = files.get("md2ns") or files.get("production")
    if md_mdp is not None:
        templates.append(
            build_grompp_command(
                executable=executable,
                mdp=md_mdp,
                gro=output_dir / "nvt.gro",
                top=topology_path,
                tpr=output_dir / "md.tpr",
                cpt=output_dir / "nvt.cpt",
            )
        )
        templates.append(
            build_mdrun_command(
                executable=executable,
                tpr=output_dir / "md.tpr",
                deffnm=output_dir / "md",
                nb=nb,
            )
        )
    traj, tpr = _preferred_traj(output_dir)
    templates.append(
        build_rms_command(
            executable=executable,
            tpr=tpr,
            traj=traj,
            xvg=output_dir / "rmsd.xvg",
        )
    )
    templates.append(
        build_rmsf_command(
            executable=executable,
            tpr=tpr,
            traj=traj,
            xvg=output_dir / "rmsf.xvg",
        )
    )
    return templates


def _run_protocol(
    result: dict[str, Any],
    *,
    executable: str,
    output_dir: Path,
    logs_dir: Path,
    structure_path: Path,
    topology_path: Path,
    mdp_meta: dict[str, Any],
    protocol: str,
    nb: str = "cpu",
) -> None:
    files = {item["template_id"]: Path(item["dest"]) for item in mdp_meta.get("files") or []}
    current_gro = structure_path
    if "minimization" in files:
        _grompp_and_mdrun(
            executable=executable,
            mdp=files["minimization"],
            gro=current_gro,
            top=topology_path,
            tpr=output_dir / "em.tpr",
            deffnm=output_dir / "em",
            logs_dir=logs_dir,
            cwd=output_dir,
            stage="em",
            nb=nb,
        )
        em_gro = output_dir / "em.gro"
        if em_gro.is_file():
            current_gro = em_gro
        elif protocol in {"em-nvt", "nvt", "em-nvt-md2ns"}:
            result.setdefault("warnings", []).append(
                "em.gro was not written; NVT will start from the input structure"
            )
    nvt_mdp = files.get("nvt_eq") or files.get("nvt")
    if nvt_mdp is not None:
        _grompp_and_mdrun(
            executable=executable,
            mdp=nvt_mdp,
            gro=current_gro,
            top=topology_path,
            tpr=output_dir / "nvt.tpr",
            deffnm=output_dir / "nvt",
            logs_dir=logs_dir,
            cwd=output_dir,
            stage="nvt",
            nb=nb,
        )
        nvt_gro = output_dir / "nvt.gro"
        if nvt_gro.is_file():
            current_gro = nvt_gro
    md_mdp = files.get("md2ns")
    if md_mdp is not None:
        nvt_cpt = output_dir / "nvt.cpt"
        _grompp_and_mdrun(
            executable=executable,
            mdp=md_mdp,
            gro=current_gro,
            top=topology_path,
            tpr=output_dir / "md.tpr",
            deffnm=output_dir / "md",
            logs_dir=logs_dir,
            cwd=output_dir,
            stage="md",
            nb=nb,
            cpt=nvt_cpt if nvt_cpt.is_file() else None,
            mdrun_timeout=MDRUN_TIMEOUT_MD2NS_SECONDS,
        )
    _run_rmsd_rmsf(executable=executable, output_dir=output_dir, logs_dir=logs_dir)


def _grompp_and_mdrun(
    *,
    executable: str,
    mdp: Path,
    gro: Path,
    top: Path,
    tpr: Path,
    deffnm: Path,
    logs_dir: Path,
    cwd: Path,
    stage: str,
    nb: str = "cpu",
    cpt: Path | None = None,
    mdrun_timeout: int | None = None,
) -> None:
    grompp = build_grompp_command(
        executable=executable, mdp=mdp, gro=gro, top=top, tpr=tpr, cpt=cpt
    )
    proc = _run_gmx(
        grompp,
        cwd=cwd,
        log_path=logs_dir / f"{stage}_grompp.log",
        timeout=GROMPP_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0 or not tpr.is_file():
        raise MdError(
            f"gmx grompp ({stage}) exited {proc.returncode}; see {logs_dir / f'{stage}_grompp.log'}"
        )
    mdrun = build_mdrun_command(
        executable=executable, tpr=tpr, deffnm=deffnm, nb=nb
    )
    proc = _run_gmx(
        mdrun,
        cwd=cwd,
        log_path=logs_dir / f"{stage}_mdrun.log",
        timeout=mdrun_timeout or MDRUN_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise MdError(
            f"gmx mdrun ({stage}) exited {proc.returncode}; see {logs_dir / f'{stage}_mdrun.log'}"
        )


def _run_rmsd_rmsf(*, executable: str, output_dir: Path, logs_dir: Path) -> None:
    traj, tpr = _existing_traj(output_dir)
    if traj is None or tpr is None:
        return
    rms = build_rms_command(
        executable=executable,
        tpr=tpr,
        traj=traj,
        xvg=output_dir / "rmsd.xvg",
    )
    _run_gmx(
        rms,
        cwd=output_dir,
        log_path=logs_dir / "rms.log",
        timeout=ANALYSIS_TIMEOUT_SECONDS,
        stdin_text=RMS_STDIN,
    )
    rmsf = build_rmsf_command(
        executable=executable,
        tpr=tpr,
        traj=traj,
        xvg=output_dir / "rmsf.xvg",
    )
    _run_gmx(
        rmsf,
        cwd=output_dir,
        log_path=logs_dir / "rmsf.log",
        timeout=ANALYSIS_TIMEOUT_SECONDS,
        stdin_text=RMSF_STDIN,
    )


def _preferred_traj(output_dir: Path) -> tuple[Path, Path]:
    """Placeholder paths used only to build dry-run command templates."""
    return output_dir / "nvt.xtc", output_dir / "nvt.tpr"


def _existing_traj(output_dir: Path) -> tuple[Path | None, Path | None]:
    candidates = (
        (output_dir / "md.xtc", output_dir / "md.tpr"),
        (output_dir / "md.trr", output_dir / "md.tpr"),
        (output_dir / "nvt.xtc", output_dir / "nvt.tpr"),
        (output_dir / "nvt.trr", output_dir / "nvt.tpr"),
        (output_dir / "em.xtc", output_dir / "em.tpr"),
        (output_dir / "em.trr", output_dir / "em.tpr"),
    )
    for traj, tpr in candidates:
        if traj.is_file() and tpr.is_file():
            return traj, tpr
    return None, None


def _write_analysis_tables(result: dict[str, Any], *, output_dir: Path) -> None:
    rmsd_rows = parse_xvg(output_dir / "rmsd.xvg")
    rmsf_rows = parse_xvg(output_dir / "rmsf.xvg")
    if rmsd_rows:
        rmsd_csv = write_rmsd_csv(output_dir / "rmsd.csv", rmsd_rows)
        result["rmsd_csv"] = str(rmsd_csv)
        result["rmsd"] = xy_stats(rmsd_rows)
        result["n_frames"] = len(rmsd_rows)
    if rmsf_rows:
        rmsf_csv = write_rmsf_csv(output_dir / "rmsf.csv", rmsf_rows)
        result["rmsf_csv"] = str(rmsf_csv)
        result["rmsf"] = xy_stats(rmsf_rows)
    result["table_present"] = bool(rmsd_rows or rmsf_rows)


def _run_gmx(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout: int,
    stdin_text: str | None = None,
    allowed_binaries: tuple[str, ...] | None = None,
    allowed_subcommands: tuple[str, ...] | None = ALLOWED_GMX_SUBCOMMANDS,
) -> subprocess.CompletedProcess[str]:
    if not command:
        raise MdError("empty command")
    binary = Path(command[0]).name
    allowed_bins = allowed_binaries or (ALLOWED_GMX_BINARIES + ALLOWED_ACPYPE_BINARIES)
    if binary not in allowed_bins:
        raise MdError(f"refusing to run non-allowlisted binary {binary!r}")
    if binary in ALLOWED_GMX_BINARIES:
        subcommands = allowed_subcommands or ALLOWED_GMX_SUBCOMMANDS
        if len(command) < 2 or command[1] not in subcommands:
            raise MdError(
                f"refusing gmx subcommand {command[1:]!r}; "
                f"allowed: {list(subcommands)}"
            )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cwd.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh:
        return subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            shell=False,
            input=stdin_text,
        )


def _md_output_dir(project: Path) -> Path:
    dest = (project / "output" / OUTPUT_SUBDIR).resolve()
    try:
        dest.relative_to(project.resolve())
    except ValueError as exc:
        raise MdError(f"MD output escapes project: {dest}") from exc
    return dest


def _write_result_json(output_dir: Path, result: dict[str, Any]) -> None:
    path = output_dir / "md_result.json"
    path.write_text(dumps_pretty(result), encoding="utf-8")
    result["result_json"] = str(path)


def _critic_payload(md: dict[str, Any]) -> dict[str, Any]:
    """Build a results dict the shared Critic can review (evidence only)."""
    return {
        "plan": {
            "approve_run": False,
            "skip_reinvent": True,
            "approve_md": md.get("approved"),
            "steps": ["md"],
        },
        "steps": {"md": md},
        "warnings": md.get("warnings") or [],
        "errors": md.get("errors") or [],
        "aborted": bool(md.get("errors")) and not md.get("skipped"),
        "abort_reason": md.get("message") if md.get("errors") else None,
        "project_dir": md.get("project_dir"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independent GROMACS MD module (smoke-test em-nvt, or 2 ns "
            "em-nvt-md2ns). Not part of the REINVENT executor. Requires "
            "--approve-md to launch. mdp files are human templates; only "
            "nsteps/dt/ref_t may be overridden. 100 ns is never launched."
        )
    )
    parser.add_argument("--project", required=True, help="Project directory")
    parser.add_argument(
        "--structure",
        default=None,
        help="Existing .gro or .pdb under <project>/input/",
    )
    parser.add_argument(
        "--topology",
        default=None,
        help="Existing .top under <project>/input/",
    )
    parser.add_argument(
        "--protocol",
        choices=ALLOWED_PROTOCOLS,
        default=DEFAULT_PROTOCOL,
        help=(
            "Human template chain. Default em-nvt (smoke-test). "
            "em-nvt-md2ns runs 2 ns after em+NVT. "
            "production materializes experiments/md.mdp but does not mdrun."
        ),
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=None,
        help=(
            "Allowlisted nsteps override "
            f"(smoke cap {MAX_LAUNCH_NSTEPS}; 2 ns protocol cap 1000000)"
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Allowlisted dt override in ps (scalar; rejected if not in template)",
    )
    parser.add_argument(
        "--ref-t",
        type=float,
        default=None,
        dest="ref_t",
        help="Allowlisted ref_t override in K",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Opt-in gmx mdrun -nb gpu (default is -nb cpu -nt 1)",
    )
    parser.add_argument(
        "--prepare-complex",
        action="store_true",
        help=(
            "Predefined pdb2gmx + ACPYPE + solvate/ions into input/md/system.gro. "
            "Requires --protein and --ligand-pose. Still needs --approve-md."
        ),
    )
    parser.add_argument(
        "--protein",
        default=None,
        help="Protein PDB under <project>/input/ (for --prepare-complex)",
    )
    parser.add_argument(
        "--ligand-pose",
        default=None,
        dest="ligand_pose",
        help="Docked ligand PDBQT/SDF under input/ or output/ (for --prepare-complex)",
    )
    parser.add_argument(
        "--approve-md",
        action="store_true",
        help="Human approval to launch the predefined gmx grompp/mdrun/rms chain",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive Proceed? prompt when used with --approve-md",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional agent.yaml (md defaults only; never TOML or mdp text)",
    )
    return parser


def run_prepare_complex_cli(
    args: argparse.Namespace,
    agent_config: dict[str, Any],
) -> dict[str, Any]:
    if not args.protein or not args.ligand_pose:
        raise MdError("--prepare-complex requires --protein and --ligand-pose")
    env = check_md_environment()
    gmx_path = env.get("gmx_path")
    acpype_path = shutil.which("acpype")
    obabel_path = shutil.which("obabel") or shutil.which("obabel3")
    planned = {
        "protocol": "prepare-complex",
        "runnable": True,
        "executable": gmx_path or "gmx",
        "structure": args.protein,
        "topology": args.ligand_pose,
        "overrides": {},
        "output_dir": str(Path(args.project).resolve() / "output" / "md" / "prep"),
        "max_launch_nsteps": 0,
        "nb": "cpu",
        "gpu": False,
    }
    result: dict[str, Any] = {
        "ok": False,
        "approved": bool(args.approve_md),
        "skipped": False,
        "success": False,
        "engine": "gmx",
        "package": "gromacs",
        "kind": "complex prep",
        "protocol": "prepare-complex",
        "project_dir": str(Path(args.project).resolve()),
        "table_present": False,
        "environment": env,
        "planned": planned,
        "warnings": [],
        "errors": [],
        "message": "",
    }
    if not args.approve_md:
        result["skipped"] = True
        result["ok"] = True
        result["message"] = (
            "Complex prep not launched (missing --approve-md). "
            "gmx/acpype were not run."
        )
        return result
    if not confirm_md_launch(
        planned, assume_yes=bool(args.yes)
    ):
        result["skipped"] = True
        result["approved"] = False
        result["ok"] = True
        result["message"] = "Human declined — complex prep will not run."
        return result
    if not gmx_path:
        result["errors"].append("gmx not found on PATH")
        result["message"] = result["errors"][-1]
        return result
    if not acpype_path:
        result["errors"].append(
            "acpype not found on PATH (needed to parameterize the ligand; "
            "this module does not auto-install)"
        )
        result["message"] = result["errors"][-1]
        return result
    try:
        prep = prepare_complex(
            args.project,
            protein=args.protein,
            ligand=args.ligand_pose,
            gmx_path=str(gmx_path),
            acpype_path=str(acpype_path),
            obabel_path=obabel_path,
            run_fn=_run_gmx,
        )
    except (ComplexPrepError, MdCommandError, OSError, subprocess.TimeoutExpired) as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        return result
    result["ok"] = True
    result["success"] = True
    result["prep"] = prep
    result["message"] = (
        f"Wrote parameterized complex {prep.get('structure')} / {prep.get('topology')}"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent_config: dict[str, Any] = {}
    if args.config:
        agent_config = load_agent_config(Path(args.config))
    else:
        try:
            agent_config = load_agent_config()
        except (OSError, ValueError):
            agent_config = {}

    try:
        if args.prepare_complex:
            result = run_prepare_complex_cli(args, agent_config)
        else:
            if not args.structure or not args.topology:
                raise MdError("--structure and --topology are required unless --prepare-complex")
            result = run_md(
                args.project,
                structure=args.structure,
                topology=args.topology,
                protocol=args.protocol,
                approve=bool(args.approve_md),
                assume_yes=bool(args.yes),
                nsteps=args.nsteps,
                dt=args.dt,
                ref_t=args.ref_t,
                gpu=bool(args.gpu),
                agent_config=agent_config,
            )
    except MdError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(json.dumps(_public_result(result), indent=2, ensure_ascii=False))

    from agents.critic import CriticAgent

    verdict = CriticAgent(agent_config=agent_config).review(_critic_payload(result))
    print("\n[Critic Agent]")
    print(f"Status: {verdict.get('status')}")
    for issue in verdict.get("issues") or []:
        print(f"  - {issue}")
    print(f"Recommendation: {verdict.get('recommendation')}")

    if str(verdict.get("status", "")).upper() == "FAIL":
        return 1
    return 0


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload.pop("command_templates", None)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
