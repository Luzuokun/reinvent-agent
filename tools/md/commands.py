"""Predefined GROMACS argv — never interpolates model text."""

from __future__ import annotations

from pathlib import Path

from tools.md.environment import ALLOWED_GMX_BINARIES

ALLOWED_GMX_SUBCOMMANDS: tuple[str, ...] = (
    "grompp",
    "mdrun",
    "rms",
    "rmsf",
    "pdb2gmx",
    "editconf",
    "solvate",
    "genion",
)
ALLOWED_ACPYPE_BINARIES: tuple[str, ...] = ("acpype",)
ALLOWED_OBABEL_BINARIES: tuple[str, ...] = ("obabel", "obabel3")
ALLOWED_NB: tuple[str, ...] = ("cpu", "gpu")

# stdin group picks for rms / rmsf. Always "System" — never model-supplied.
RMS_STDIN = "System\nSystem\n"
RMSF_STDIN = "System\n"
GENION_STDIN = "SOL\n"

GROMPP_TIMEOUT_SECONDS = 120
MDRUN_TIMEOUT_SECONDS = 600
MDRUN_TIMEOUT_MD2NS_SECONDS = 14_400
ANALYSIS_TIMEOUT_SECONDS = 120
PREP_TIMEOUT_SECONDS = 300
ACPYPE_TIMEOUT_SECONDS = 900


class MdCommandError(ValueError):
    """Unknown binary, illegal subcommand, or path-less argv."""


def normalize_gmx_executable(executable: str) -> str:
    if not executable or not str(executable).strip():
        raise MdCommandError("gmx executable is required")
    base = Path(executable).name
    if base not in ALLOWED_GMX_BINARIES:
        raise MdCommandError(
            f"executable basename {base!r} is not an allowlisted GROMACS binary "
            f"{list(ALLOWED_GMX_BINARIES)}"
        )
    return str(executable)


def normalize_acpype_executable(executable: str) -> str:
    if not executable or not str(executable).strip():
        raise MdCommandError("acpype executable is required")
    base = Path(executable).name
    if base not in ALLOWED_ACPYPE_BINARIES:
        raise MdCommandError(
            f"executable basename {base!r} is not an allowlisted acpype binary "
            f"{list(ALLOWED_ACPYPE_BINARIES)}"
        )
    return str(executable)


def normalize_obabel_executable(executable: str) -> str:
    if not executable or not str(executable).strip():
        raise MdCommandError("obabel executable is required")
    base = Path(executable).name
    if base not in ALLOWED_OBABEL_BINARIES:
        raise MdCommandError(
            f"executable basename {base!r} is not an allowlisted obabel binary "
            f"{list(ALLOWED_OBABEL_BINARIES)}"
        )
    return str(executable)


def normalize_nb(nb: str) -> str:
    name = str(nb or "cpu").strip().lower()
    if name not in ALLOWED_NB:
        raise MdCommandError(f"mdrun -nb must be cpu or gpu; got {nb!r}")
    return name


def _require_file_arg(path: str | Path, *, what: str) -> str:
    if path is None or str(path).strip() == "":
        raise MdCommandError(f"{what} path is empty")
    return str(path)


def build_grompp_command(
    *,
    executable: str,
    mdp: str | Path,
    gro: str | Path,
    top: str | Path,
    tpr: str | Path,
    maxwarn: int = 1,
    cpt: str | Path | None = None,
) -> list[str]:
    """Predefined ``gmx grompp``. ``shell=False`` only."""
    gmx = normalize_gmx_executable(executable)
    if maxwarn < 0 or maxwarn > 2:
        raise MdCommandError("maxwarn must be 0, 1, or 2")
    cmd = [
        gmx,
        "grompp",
        "-f",
        _require_file_arg(mdp, what="mdp"),
        "-c",
        _require_file_arg(gro, what="gro"),
        "-p",
        _require_file_arg(top, what="top"),
        "-o",
        _require_file_arg(tpr, what="tpr"),
        "-maxwarn",
        str(int(maxwarn)),
    ]
    if cpt is not None:
        cmd.extend(["-t", _require_file_arg(cpt, what="cpt")])
    return cmd


def build_mdrun_command(
    *,
    executable: str,
    tpr: str | Path,
    deffnm: str | Path,
    nb: str = "cpu",
) -> list[str]:
    """Predefined ``gmx mdrun``. CPU is the default; GPU is opt-in via ``nb='gpu'``."""
    gmx = normalize_gmx_executable(executable)
    nb = normalize_nb(nb)
    cmd = [
        gmx,
        "mdrun",
        "-s",
        _require_file_arg(tpr, what="tpr"),
        "-deffnm",
        _require_file_arg(deffnm, what="deffnm"),
        "-nb",
        nb,
    ]
    if nb == "cpu":
        cmd.extend(["-nt", "1"])
    return cmd


def build_rms_command(
    *,
    executable: str,
    tpr: str | Path,
    traj: str | Path,
    xvg: str | Path,
) -> list[str]:
    gmx = normalize_gmx_executable(executable)
    return [
        gmx,
        "rms",
        "-s",
        _require_file_arg(tpr, what="tpr"),
        "-f",
        _require_file_arg(traj, what="traj"),
        "-o",
        _require_file_arg(xvg, what="rmsd xvg"),
        "-tu",
        "ps",
        "-xvg",
        "none",
    ]


def build_rmsf_command(
    *,
    executable: str,
    tpr: str | Path,
    traj: str | Path,
    xvg: str | Path,
) -> list[str]:
    gmx = normalize_gmx_executable(executable)
    return [
        gmx,
        "rmsf",
        "-s",
        _require_file_arg(tpr, what="tpr"),
        "-f",
        _require_file_arg(traj, what="traj"),
        "-o",
        _require_file_arg(xvg, what="rmsf xvg"),
        "-xvg",
        "none",
    ]


def build_pdb2gmx_command(
    *,
    executable: str,
    pdb: str | Path,
    gro: str | Path,
    top: str | Path,
    posre: str | Path,
) -> list[str]:
    gmx = normalize_gmx_executable(executable)
    return [
        gmx,
        "pdb2gmx",
        "-f",
        _require_file_arg(pdb, what="pdb"),
        "-o",
        _require_file_arg(gro, what="gro"),
        "-p",
        _require_file_arg(top, what="top"),
        "-i",
        _require_file_arg(posre, what="posre"),
        "-ff",
        "amber99sb-ildn",
        "-water",
        "tip3p",
        "-ignh",
    ]


def build_editconf_command(
    *,
    executable: str,
    gro: str | Path,
    out: str | Path,
    distance: float = 1.0,
) -> list[str]:
    gmx = normalize_gmx_executable(executable)
    if distance < 0.5 or distance > 2.0:
        raise MdCommandError("editconf -d must be between 0.5 and 2.0 nm")
    return [
        gmx,
        "editconf",
        "-f",
        _require_file_arg(gro, what="gro"),
        "-o",
        _require_file_arg(out, what="out"),
        "-c",
        "-d",
        f"{float(distance):.1f}",
        "-bt",
        "cubic",
    ]


def build_solvate_command(
    *,
    executable: str,
    gro: str | Path,
    out: str | Path,
    top: str | Path,
) -> list[str]:
    gmx = normalize_gmx_executable(executable)
    return [
        gmx,
        "solvate",
        "-cp",
        _require_file_arg(gro, what="gro"),
        "-cs",
        "spc216.gro",
        "-o",
        _require_file_arg(out, what="out"),
        "-p",
        _require_file_arg(top, what="top"),
    ]


def build_genion_command(
    *,
    executable: str,
    tpr: str | Path,
    gro: str | Path,
    top: str | Path,
) -> list[str]:
    gmx = normalize_gmx_executable(executable)
    return [
        gmx,
        "genion",
        "-s",
        _require_file_arg(tpr, what="tpr"),
        "-o",
        _require_file_arg(gro, what="gro"),
        "-p",
        _require_file_arg(top, what="top"),
        "-pname",
        "NA",
        "-nname",
        "CL",
        "-neutral",
    ]


def build_acpype_command(
    *,
    executable: str,
    ligand: str | Path,
    charge: int = 0,
) -> list[str]:
    acpype = normalize_acpype_executable(executable)
    if abs(int(charge)) > 4:
        raise MdCommandError(f"ligand net charge {charge} is outside ±4")
    return [
        acpype,
        "-i",
        _require_file_arg(ligand, what="ligand"),
        "-b",
        "ligand",
        "-a",
        "gaff2",
        "-c",
        "gas",
        "-n",
        str(int(charge)),
    ]


def build_obabel_sdf_command(
    *,
    executable: str,
    source: str | Path,
    dest: str | Path,
) -> list[str]:
    obabel = normalize_obabel_executable(executable)
    return [
        obabel,
        _require_file_arg(source, what="source"),
        "-O",
        _require_file_arg(dest, what="dest"),
        "-h",
    ]
