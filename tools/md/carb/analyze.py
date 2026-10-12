"""Fixed carb analyses. Never invent FEL / contacts / energies / MM-PBSA numbers."""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path
from typing import Any, Callable

from tools.md.analyze import parse_xvg, write_xy_csv, xy_stats
from tools.md.carb.commands import (
    DISTANCE_STDIN,
    ENERGY_STDIN,
    GYRATE_STDIN,
    HBOND_STDIN,
    MINDIST_STDIN,
    RMS_STDIN,
    SASA_STDIN,
    build_distance_command,
    build_energy_command,
    build_gyrate_command,
    build_hbond_command,
    build_mindist_command,
    build_rms_command,
    build_sasa_command,
)
from tools.md.carb.constants import AMYLOSE_N_RESIDUES, FUCOIDAN_N_RESIDUES
from tools.md.commands import ANALYSIS_TIMEOUT_SECONDS
from tools.md.environment import ALLOWED_GMX_BINARIES
from tools.md.run import _run_gmx

R_KJ = 0.008314462618  # kJ mol-1 K-1
DEFAULT_T = 368.0

RunFn = Callable[..., Any]


def find_traj(run_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Prefer production md, then npt, then nvt. Missing files stay missing."""
    for stage in ("md", "npt", "nvt", "em"):
        tpr = run_dir / f"{stage}.tpr"
        edr = run_dir / f"{stage}.edr"
        for suffix in (".xtc", ".trr"):
            traj = run_dir / f"{stage}{suffix}"
            if traj.is_file() and tpr.is_file():
                return traj, tpr, edr if edr.is_file() else None
    return None, None, None


def collect_run_dirs(output_dir: Path) -> list[Path]:
    found: list[Path] = []
    for name in ("smoke", "rep1", "rep2", "rep3", "rep4", "rep5"):
        candidate = output_dir / name
        if candidate.is_dir():
            found.append(candidate)
    if not found and output_dir.is_dir():
        found.append(output_dir)
    return found


def write_fel_from_series(
    dest: Path,
    rmsd: list[tuple[float, float]],
    rg: list[tuple[float, float]],
    *,
    temperature: float = DEFAULT_T,
    bins: int = 12,
) -> dict[str, Any] | None:
    """2D FEL F = -RT ln(P/Pmax) from RMSD vs Rg. Empty inputs → no file."""
    n = min(len(rmsd), len(rg))
    if n < 3:
        return None
    xs = [rmsd[i][1] for i in range(n)]
    ys = [rg[i][1] for i in range(n)]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin:
        xmax = xmin + 1e-6
    if ymax <= ymin:
        ymax = ymin + 1e-6
    nx = max(4, min(bins, n // 2 or 4))
    ny = nx
    counts = [[0 for _ in range(ny)] for _ in range(nx)]
    for x, y in zip(xs, ys):
        ix = min(nx - 1, max(0, int((x - xmin) / (xmax - xmin) * nx)))
        iy = min(ny - 1, max(0, int((y - ymin) / (ymax - ymin) * ny)))
        counts[ix][iy] += 1
    peak = max(max(row) for row in counts)
    if peak <= 0:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    rt = R_KJ * float(temperature)
    rows: list[list[Any]] = []
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rmsd_bin_center_nm", "rg_bin_center_nm", "count", "deltaG_kJ_mol"])
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        for ix in range(nx):
            for iy in range(ny):
                c = counts[ix][iy]
                if c <= 0:
                    dg: float | str = ""
                else:
                    dg = round(-rt * math.log(c / peak), 6)
                cx = xmin + (ix + 0.5) * dx
                cy = ymin + (iy + 0.5) * dy
                writer.writerow([f"{cx:.6f}", f"{cy:.6f}", c, dg])
                rows.append([cx, cy, c, dg])
    return {
        "path": str(dest),
        "n_frames": n,
        "bins": nx,
        "temperature_K": temperature,
        "method": "-RT ln(P/Pmax) on RMSD vs Rg (no invented samples)",
    }


def write_contact_heatmap(
    dest: Path,
    mindist_rows: list[tuple[float, float]] | None = None,
    *,
    n_amy: int = AMYLOSE_N_RESIDUES,
    n_fuc: int = FUCOIDAN_N_RESIDUES,
) -> dict[str, Any] | None:
    """Residue–residue contact heatmap.

    Without a real per-residue matrix, a 1-D mindist series is recorded as-is
    and a placeholder grid is **not** filled with fake distances.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not mindist_rows:
        return None
    mean = sum(y for _x, y in mindist_rows) / len(mindist_rows)
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_ps", "min_dist_nm"])
        for x, y in mindist_rows:
            writer.writerow([x, y])
    meta_path = dest.with_name("contact_heatmap_meta.csv")
    with meta_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["amylose_residues", "fucoidan_residues", "n_frames", "mean_min_dist_nm"])
        writer.writerow([n_amy, n_fuc, len(mindist_rows), round(mean, 6)])
    return {
        "path": str(dest),
        "meta": str(meta_path),
        "n_frames": len(mindist_rows),
        "mean_min_dist_nm": round(mean, 6),
        "shape": [n_amy, n_fuc],
        "note": (
            "Per-frame Amylose–Fucoidan minimum distance. A full residue-pair "
            "grid requires a real trajectory; empty cells are not invented."
        ),
    }


def gmx_mmpbsa_status() -> dict[str, Any]:
    path = shutil.which("gmx_MMPBSA")
    if path:
        return {
            "status": "available",
            "binary": path,
            "message": "gmx_MMPBSA found; report may invoke it on an existing trajectory.",
        }
    return {
        "status": "skipped",
        "binary": None,
        "message": (
            "SKIPPED: gmx_MMPBSA not found on PATH. MM/PBSA was not run and "
            "no binding free energy was invented."
        ),
    }


def run_replica_analysis(
    run_dir: Path,
    *,
    executable: str | None,
    run_fn: RunFn | None = None,
    temperature: float = DEFAULT_T,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "tables": {},
        "skipped": [],
        "errors": [],
    }
    traj, tpr, edr = find_traj(run_dir)
    if traj is None or tpr is None:
        out["skipped"].append("no trajectory/tpr — analysis not invented")
        return out
    if not executable:
        out["skipped"].append("gmx missing — analysis not invented")
        return out
    runner = run_fn or _run_gmx
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("rmsd", build_rms_command, {"xvg": run_dir / "rmsd.xvg"}, RMS_STDIN, "rmsd_nm", "time_ps"),
        ("rg", build_gyrate_command, {"xvg": run_dir / "rg.xvg"}, GYRATE_STDIN, "rg_nm", "time_ps"),
        ("hbond", build_hbond_command, {"xvg": run_dir / "hbond.xvg"}, HBOND_STDIN, "hbonds", "time_ps"),
        ("sasa", build_sasa_command, {"xvg": run_dir / "sasa.xvg"}, SASA_STDIN, "sasa_nm2", "time_ps"),
        ("com", build_distance_command, {"xvg": run_dir / "com.xvg"}, DISTANCE_STDIN, "com_nm", "time_ps"),
        ("mindist", build_mindist_command, {"xvg": run_dir / "mindist.xvg"}, MINDIST_STDIN, "min_dist_nm", "time_ps"),
    ]
    for key, builder, extra, stdin, y_name, x_name in jobs:
        cmd = builder(executable=executable, tpr=tpr, traj=traj, **extra)
        try:
            runner(
                cmd,
                cwd=run_dir,
                log_path=logs / f"{key}.log",
                timeout=ANALYSIS_TIMEOUT_SECONDS,
                stdin_text=stdin,
                allowed_binaries=ALLOWED_GMX_BINARIES,
            )
        except Exception as exc:  # noqa: BLE001 — analysis skip must be loud, not a crash
            out["skipped"].append(f"{key}: {exc}")
            continue
        xvg = Path(extra["xvg"])
        rows = parse_xvg(xvg)
        if not rows:
            out["skipped"].append(f"{key}: empty xvg (not invented)")
            continue
        csv_path = write_xy_csv(run_dir / f"{key}.csv", rows, x_name=x_name, y_name=y_name)
        out["tables"][key] = {"csv": str(csv_path), "stats": xy_stats(rows), "n": len(rows)}

    if edr is not None and edr.is_file():
        energy_xvg = run_dir / "energy_groups.xvg"
        cmd = build_energy_command(executable=executable, edr=edr, xvg=energy_xvg)
        try:
            runner(
                cmd,
                cwd=run_dir,
                log_path=logs / "energy.log",
                timeout=ANALYSIS_TIMEOUT_SECONDS,
                stdin_text=ENERGY_STDIN,
                allowed_binaries=ALLOWED_GMX_BINARIES,
            )
        except Exception as exc:  # noqa: BLE001
            out["skipped"].append(f"energy: {exc}")
        else:
            rows = parse_xvg(energy_xvg)
            if rows:
                csv_path = write_xy_csv(
                    run_dir / "energy_groups.csv",
                    rows,
                    x_name="time_ps",
                    y_name="energy_kJ_mol",
                )
                out["tables"]["energy_groups"] = {
                    "csv": str(csv_path),
                    "stats": xy_stats(rows),
                    "n": len(rows),
                    "note": "Energy-group decomposition from gmx energy; terms follow the human mdp energygrps.",
                }
            else:
                out["skipped"].append("energy: empty xvg (not invented)")
    else:
        out["skipped"].append("energy: no edr (not invented)")

    rmsd_rows = parse_xvg(run_dir / "rmsd.xvg")
    rg_rows = parse_xvg(run_dir / "rg.xvg")
    fel = write_fel_from_series(
        run_dir / "fel.csv", rmsd_rows, rg_rows, temperature=temperature
    )
    if fel:
        out["tables"]["fel"] = fel
    else:
        out["skipped"].append("FEL: need RMSD and Rg series (not invented)")

    mindist_rows = parse_xvg(run_dir / "mindist.xvg")
    heatmap = write_contact_heatmap(run_dir / "contact_heatmap.csv", mindist_rows)
    if heatmap:
        out["tables"]["contact_heatmap"] = heatmap
    else:
        out["skipped"].append("contact heatmap: no mindist series (not invented)")

    out["mmpbsa"] = gmx_mmpbsa_status()
    if out["mmpbsa"]["status"] == "skipped":
        out["skipped"].append(out["mmpbsa"]["message"])
    return out
