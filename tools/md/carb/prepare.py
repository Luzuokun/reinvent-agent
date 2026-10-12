"""Assemble the amylose–fucoidan system from bundled templates.

Isolated from EGFR ``prepare_complex`` (no pdb2gmx, no ACPYPE, no ligand).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from tools.md.carb.commands import (
    build_carb_genion_command,
    build_editconf_command,
    build_grompp_command,
    build_solvate_command,
)
from tools.md.carb.constants import (
    DEFAULT_BOX_D,
    DEFAULT_SALT_MOL,
    DEFAULT_TEMPLATE_ID,
    INPUT_MD_SUBDIR,
    MDP_DIR,
    PROTOCOL_ID,
    TEMPLATES_DIR,
)
from tools.md.carb.forcefield import ForcefieldError, alias_ff_dir, ensure_forcefield
from tools.md.carb.scalars import normalize_carb_scalars
from tools.md.carb.templates import (
    CarbTemplateError,
    concatenate_complex,
    load_manifest,
    template_spec,
    validate_bundled_templates,
    write_index_ndx,
)
from tools.md.commands import GENION_STDIN, PREP_TIMEOUT_SECONDS, GROMPP_TIMEOUT_SECONDS
from tools.md.environment import check_md_environment
from tools.md.mdp import MdpError
from tools.md.run import MdError, _run_gmx

RunFn = Callable[..., Any]


class CarbPrepareError(ValueError):
    """Template assembly or solvation failed."""


def _input_md_dir(project: Path) -> Path:
    dest = project.joinpath(*INPUT_MD_SUBDIR).resolve()
    try:
        dest.relative_to(project.resolve())
    except ValueError as exc:
        raise CarbPrepareError(f"carb input escapes project: {dest}") from exc
    return dest


def _write_system_top(
    dest: Path,
    *,
    fucoidan_itp: str,
    fucoidan_molname: str,
    n_sol: int | None = None,
    n_na: int | None = None,
    n_cl: int | None = None,
) -> None:
    lines = [
        "; Amylose-26 × Fucoidan-12 assembled by tools.md.carb.prepare",
        f"; protocol_id = {PROTOCOL_ID}",
        "; CHARMM36 is the pinned MacKerell jul2022 port (not git-tracked).",
        '#include "charmm36.ff/forcefield.itp"',
        '#include "amylose_26.itp"',
        f'#include "{fucoidan_itp}"',
        '#include "charmm36.ff/tip3p.itp"',
        '#include "charmm36.ff/ions.itp"',
        "",
        "[ system ]",
        "Amylose-26 / Fucoidan-12 in water",
        "",
        "[ molecules ]",
        "; Compound        nmols",
        "Amylose26         1",
        f"{fucoidan_molname:<16s} 1",
    ]
    if n_sol:
        lines.append(f"SOL              {int(n_sol)}")
    if n_na:
        lines.append(f"NA               {int(n_na)}")
    if n_cl:
        lines.append(f"CL               {int(n_cl)}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_carb(
    project_dir: str | Path,
    *,
    template_id: str = DEFAULT_TEMPLATE_ID,
    box_d: float | None = None,
    salt_mol: float | None = None,
    dry_run: bool = False,
    fetch_forcefield: bool = True,
    run_fn: RunFn | None = None,
) -> dict[str, Any]:
    """Build ``<project>/input/md/`` from bundled oligomer templates."""
    try:
        scalars = normalize_carb_scalars(
            {
                k: v
                for k, v in {"box_d": box_d, "salt_mol": salt_mol}.items()
                if v is not None
            }
        )
    except MdpError as exc:
        raise CarbPrepareError(str(exc)) from exc
    box = float(scalars.get("box_d", DEFAULT_BOX_D))
    salt = float(scalars.get("salt_mol", DEFAULT_SALT_MOL))

    project = Path(project_dir).expanduser().resolve()
    if dry_run:
        spec = template_spec(template_id)
        validate_bundled_templates()
        planned = {
            "protocol_id": PROTOCOL_ID,
            "template_id": spec["template_id"],
            "alt_sulfation": spec["alt_sulfation"],
            "box_d": box,
            "salt_mol": salt,
            "dry_run": True,
            "isolated_from": "prepare-complex / tools.md.complex",
        }
        return {
            "ok": True,
            "kind": "carb-prepare",
            "protocol_id": PROTOCOL_ID,
            "template_id": spec["template_id"],
            "project_dir": str(project),
            "planned": planned,
            "dry_run": True,
            "skipped": True,
            "warnings": [],
            "errors": [],
            "structure": None,
            "topology": None,
            "message": (
                "prepare dry-run: templates and scalars checked; no files written, "
                "gmx not run."
            ),
        }
    if not project.exists():
        project.mkdir(parents=True, exist_ok=True)
    if not project.is_dir():
        raise CarbPrepareError(f"Project directory does not exist: {project}")

    spec = template_spec(template_id)
    manifest = load_manifest()
    validate_bundled_templates()

    planned = {
        "protocol_id": PROTOCOL_ID,
        "template_id": spec["template_id"],
        "alt_sulfation": spec["alt_sulfation"],
        "box_d": box,
        "salt_mol": salt,
        "dry_run": bool(dry_run),
        "isolated_from": "prepare-complex / tools.md.complex",
    }
    result: dict[str, Any] = {
        "ok": False,
        "kind": "carb-prepare",
        "protocol_id": PROTOCOL_ID,
        "template_id": spec["template_id"],
        "project_dir": str(project),
        "planned": planned,
        "dry_run": bool(dry_run),
        "warnings": [],
        "errors": [],
        "message": "",
        "structure": None,
        "topology": None,
    }

    input_dir = _input_md_dir(project)
    input_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = project / "output" / "carb" / "prep" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    amy_key = spec["amylose"]
    fuc_key = spec["fucoidan"]
    amy_files = manifest[amy_key]["files"]
    fuc_files = manifest[fuc_key]["files"]
    shutil.copy2(TEMPLATES_DIR / amy_files["itp"], input_dir / "amylose_26.itp")
    fuc_itp_name = Path(fuc_files["itp"]).name
    shutil.copy2(TEMPLATES_DIR / fuc_files["itp"], input_dir / fuc_itp_name)
    shutil.copy2(TEMPLATES_DIR / amy_files["pdb"], input_dir / Path(amy_files["pdb"]).name)
    shutil.copy2(TEMPLATES_DIR / fuc_files["pdb"], input_dir / Path(fuc_files["pdb"]).name)

    complex_gro = input_dir / "complex.gro"
    assembled = concatenate_complex(
        TEMPLATES_DIR / amy_files["gro"],
        TEMPLATES_DIR / fuc_files["gro"],
        complex_gro,
    )
    top_path = input_dir / "system.top"
    _write_system_top(
        top_path,
        fucoidan_itp=fuc_itp_name,
        fucoidan_molname=spec["fucoidan_molname"],
    )
    index_path = write_index_ndx(complex_gro, input_dir / "index.ndx")

    ff_meta: dict[str, Any] | None = None
    try:
        ff_meta = ensure_forcefield(fetch=fetch_forcefield)
        alias = Path(ff_meta["alias"] if ff_meta.get("alias") else alias_ff_dir())
        dest_ff = input_dir / "charmm36.ff"
        if alias.is_dir() and not dest_ff.exists():
            try:
                dest_ff.symlink_to(alias.resolve())
            except OSError:
                shutil.copytree(alias, dest_ff)
    except ForcefieldError as exc:
        result["warnings"].append(str(exc))
        result["forcefield"] = {"ok": False, "message": str(exc)}

    env = check_md_environment()
    gmx_path = env.get("gmx_path")
    result["environment"] = env
    result["forcefield"] = ff_meta
    result["assembled"] = assembled
    result["index"] = str(index_path)

    boxed = input_dir / "boxed.gro"
    solvated = input_dir / "solvated.gro"
    ionized = input_dir / "system.gro"
    runner = run_fn or _run_gmx

    if not gmx_path:
        shutil.copy2(complex_gro, ionized)
        result["structure"] = str(ionized)
        result["topology"] = str(top_path)
        result["ok"] = True
        result["solvated"] = False
        result["message"] = (
            "Wrote unsolvated complex under input/md/ (gmx not on PATH; "
            "solvation/genion skipped). doctor/prepare with gmx to solvate."
        )
        result["warnings"].append(result["message"])
        return result

    try:
        _solvate_and_ion(
            executable=str(gmx_path),
            input_dir=input_dir,
            logs_dir=logs_dir,
            complex_gro=complex_gro,
            boxed=boxed,
            solvated=solvated,
            ionized=ionized,
            top_path=top_path,
            box_d=box,
            salt_mol=salt,
            run_fn=runner,
        )
    except (MdError, CarbPrepareError, CarbTemplateError, OSError) as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        result["structure"] = str(complex_gro)
        result["topology"] = str(top_path)
        return result

    result["ok"] = True
    result["solvated"] = True
    result["structure"] = str(ionized)
    result["topology"] = str(top_path)
    result["message"] = (
        f"Wrote solvated carb system {ionized} / {top_path} "
        f"(template={spec['template_id']}, box_d={box}, salt_mol={salt})"
    )
    return result


def _solvate_and_ion(
    *,
    executable: str,
    input_dir: Path,
    logs_dir: Path,
    complex_gro: Path,
    boxed: Path,
    solvated: Path,
    ionized: Path,
    top_path: Path,
    box_d: float,
    salt_mol: float,
    run_fn: RunFn,
) -> None:
    editconf = build_editconf_command(
        executable=executable, gro=complex_gro, out=boxed, distance=box_d
    )
    proc = run_fn(
        editconf,
        cwd=input_dir,
        log_path=logs_dir / "editconf.log",
        timeout=PREP_TIMEOUT_SECONDS,
    )
    if getattr(proc, "returncode", 1) != 0 or not boxed.is_file():
        raise CarbPrepareError(
            f"gmx editconf failed; see {logs_dir / 'editconf.log'}"
        )

    solvate = build_solvate_command(
        executable=executable, gro=boxed, out=solvated, top=top_path
    )
    proc = run_fn(
        solvate,
        cwd=input_dir,
        log_path=logs_dir / "solvate.log",
        timeout=PREP_TIMEOUT_SECONDS,
    )
    if getattr(proc, "returncode", 1) != 0 or not solvated.is_file():
        raise CarbPrepareError(
            f"gmx solvate failed; see {logs_dir / 'solvate.log'}"
        )

    ions_mdp = MDP_DIR / "ions.mdp"
    ions_tpr = input_dir / "ions.tpr"
    grompp = build_grompp_command(
        executable=executable,
        mdp=ions_mdp,
        gro=solvated,
        top=top_path,
        tpr=ions_tpr,
        maxwarn=2,
    )
    proc = run_fn(
        grompp,
        cwd=input_dir,
        log_path=logs_dir / "ions_grompp.log",
        timeout=GROMPP_TIMEOUT_SECONDS,
    )
    if getattr(proc, "returncode", 1) != 0 or not ions_tpr.is_file():
        raise CarbPrepareError(
            f"gmx grompp (ions) failed; see {logs_dir / 'ions_grompp.log'}"
        )

    genion = build_carb_genion_command(
        executable=executable,
        tpr=ions_tpr,
        gro=ionized,
        top=top_path,
        conc=salt_mol,
    )
    proc = run_fn(
        genion,
        cwd=input_dir,
        log_path=logs_dir / "genion.log",
        timeout=PREP_TIMEOUT_SECONDS,
        stdin_text=GENION_STDIN,
    )
    if getattr(proc, "returncode", 1) != 0 or not ionized.is_file():
        raise CarbPrepareError(
            f"gmx genion failed; see {logs_dir / 'genion.log'}"
        )
