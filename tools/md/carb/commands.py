"""Predefined GROMACS argv for the carb protocol. Never interpolates model text."""

from __future__ import annotations

from pathlib import Path

from tools.md.commands import (
    MdCommandError,
    build_editconf_command,
    build_genion_command,
    build_grompp_command,
    build_mdrun_command,
    build_rms_command,
    build_solvate_command,
    normalize_gmx_executable,
    _require_file_arg,
)

CARB_ANALYSIS_SUBCOMMANDS: tuple[str, ...] = (
    "rms",
    "gyrate",
    "hbond",
    "sasa",
    "distance",
    "energy",
    "mindist",
    "sham",
    "trjconv",
)

RMS_STDIN = "Amylose\nAmylose\n"
GYRATE_STDIN = "System\n"
HBOND_STDIN = "Amylose\nFucoidan\n"
SASA_STDIN = "Amylose\n"
DISTANCE_STDIN = "Amylose\nFucoidan\n"
MINDIST_STDIN = "Amylose\nFucoidan\n"
ENERGY_STDIN = (
    "Coulomb-SR:Amylose-Fucoidan\n"
    "LJ-SR:Amylose-Fucoidan\n"
    "Coulomb-SR:Amylose-Amylose\n"
    "LJ-SR:Amylose-Amylose\n"
    "\n"
)
TRJCONV_STDIN = "System\n"


def build_gyrate_command(
    *,
    executable: str,
    tpr: str | Path,
    traj: str | Path,
    xvg: str | Path,
) -> list[str]:
    gmx = normalize_gmx_executable(executable)
    return [
        gmx,
        "gyrate",
        "-s",
        _require_file_arg(tpr, what="tpr"),
        "-f",
        _require_file_arg(traj, what="traj"),
        "-o",
        _require_file_arg(xvg, what="rg xvg"),
        "-xvg",
        "none",
    ]


def build_hbond_command(
    *,
    executable: str,
    tpr: str | Path,
    traj: str | Path,
    xvg: str | Path,
) -> list[str]:
    gmx = normalize_gmx_executable(executable)
    return [
        gmx,
        "hbond",
        "-s",
        _require_file_arg(tpr, what="tpr"),
        "-f",
        _require_file_arg(traj, what="traj"),
        "-num",
        _require_file_arg(xvg, what="hbond xvg"),
        "-xvg",
        "none",
    ]


def build_sasa_command(
    *,
    executable: str,
    tpr: str | Path,
    traj: str | Path,
    xvg: str | Path,
) -> list[str]:
    gmx = normalize_gmx_executable(executable)
    return [
        gmx,
        "sasa",
        "-s",
        _require_file_arg(tpr, what="tpr"),
        "-f",
        _require_file_arg(traj, what="traj"),
        "-o",
        _require_file_arg(xvg, what="sasa xvg"),
        "-xvg",
        "none",
    ]


def build_distance_command(
    *,
    executable: str,
    tpr: str | Path,
    traj: str | Path,
    xvg: str | Path,
) -> list[str]:
    gmx = normalize_gmx_executable(executable)
    return [
        gmx,
        "distance",
        "-s",
        _require_file_arg(tpr, what="tpr"),
        "-f",
        _require_file_arg(traj, what="traj"),
        "-select",
        "com of group Amylose plus com of group Fucoidan",
        "-oall",
        _require_file_arg(xvg, what="com xvg"),
        "-xvg",
        "none",
    ]


def build_energy_command(
    *,
    executable: str,
    edr: str | Path,
    xvg: str | Path,
) -> list[str]:
    gmx = normalize_gmx_executable(executable)
    return [
        gmx,
        "energy",
        "-f",
        _require_file_arg(edr, what="edr"),
        "-o",
        _require_file_arg(xvg, what="energy xvg"),
        "-xvg",
        "none",
    ]


def build_mindist_command(
    *,
    executable: str,
    tpr: str | Path,
    traj: str | Path,
    xvg: str | Path,
) -> list[str]:
    gmx = normalize_gmx_executable(executable)
    return [
        gmx,
        "mindist",
        "-s",
        _require_file_arg(tpr, what="tpr"),
        "-f",
        _require_file_arg(traj, what="traj"),
        "-od",
        _require_file_arg(xvg, what="mindist xvg"),
        "-group",
        "-xvg",
        "none",
    ]


def build_carb_genion_command(
    *,
    executable: str,
    tpr: str | Path,
    gro: str | Path,
    top: str | Path,
    conc: float,
) -> list[str]:
    return build_genion_command(
        executable=executable,
        tpr=tpr,
        gro=gro,
        top=top,
        conc=conc,
        pname="NA",
        nname="CL",
    )


def build_trjconv_pbc_command(
    *,
    executable: str,
    tpr: str | Path,
    traj: str | Path,
    out: str | Path,
) -> list[str]:
    gmx = normalize_gmx_executable(executable)
    return [
        gmx,
        "trjconv",
        "-s",
        _require_file_arg(tpr, what="tpr"),
        "-f",
        _require_file_arg(traj, what="traj"),
        "-o",
        _require_file_arg(out, what="xtc"),
        "-pbc",
        "mol",
        "-ur",
        "compact",
    ]


# Re-export builders used by prepare/run.
__all__ = [
    "CARB_ANALYSIS_SUBCOMMANDS",
    "DISTANCE_STDIN",
    "ENERGY_STDIN",
    "GYRATE_STDIN",
    "HBOND_STDIN",
    "MINDIST_STDIN",
    "RMS_STDIN",
    "SASA_STDIN",
    "TRJCONV_STDIN",
    "MdCommandError",
    "build_carb_genion_command",
    "build_distance_command",
    "build_editconf_command",
    "build_energy_command",
    "build_grompp_command",
    "build_gyrate_command",
    "build_hbond_command",
    "build_mdrun_command",
    "build_mindist_command",
    "build_rms_command",
    "build_sasa_command",
    "build_solvate_command",
    "build_trjconv_pbc_command",
    "normalize_gmx_executable",
]
