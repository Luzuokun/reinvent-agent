"""Predefined GROMACS argv — never interpolates model text."""

from __future__ import annotations

from pathlib import Path

from tools.md.environment import ALLOWED_GMX_BINARIES

ALLOWED_GMX_SUBCOMMANDS: tuple[str, ...] = ("grompp", "mdrun", "rms", "rmsf")

# stdin group picks for rms / rmsf. Always "System" — never model-supplied.
RMS_STDIN = "System\nSystem\n"
RMSF_STDIN = "System\n"

GROMPP_TIMEOUT_SECONDS = 60
MDRUN_TIMEOUT_SECONDS = 600
ANALYSIS_TIMEOUT_SECONDS = 60


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
) -> list[str]:
    """Predefined ``gmx grompp``. ``shell=False`` only."""
    gmx = normalize_gmx_executable(executable)
    if maxwarn < 0 or maxwarn > 2:
        raise MdCommandError("maxwarn must be 0, 1, or 2")
    return [
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


def build_mdrun_command(
    *,
    executable: str,
    tpr: str | Path,
    deffnm: str | Path,
) -> list[str]:
    """Predefined CPU ``gmx mdrun`` (no GPU flags, no free-form options)."""
    gmx = normalize_gmx_executable(executable)
    return [
        gmx,
        "mdrun",
        "-s",
        _require_file_arg(tpr, what="tpr"),
        "-deffnm",
        _require_file_arg(deffnm, what="deffnm"),
        "-nt",
        "1",
        "-nb",
        "cpu",
    ]


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
