"""Parse GROMACS XVG and write RMSD/RMSF CSV tables.

Never invent numbers: empty or unreadable XVG yields no table.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class MdAnalyzeError(ValueError):
    """Unreadable XVG."""


def parse_xvg(path: str | Path) -> list[tuple[float, float]]:
    """Return ``(x, y)`` pairs, skipping comments and @ directives."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return []
    rows: list[tuple[float, float]] = []
    text = file_path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("@"):
            continue
        if line.startswith("&"):
            break
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x_val = float(parts[0])
            y_val = float(parts[1])
        except ValueError:
            continue
        if x_val != x_val or y_val != y_val:  # NaN
            continue
        rows.append((x_val, y_val))
    return rows


def write_xy_csv(
    dest: str | Path,
    rows: list[tuple[float, float]],
    *,
    x_name: str,
    y_name: str,
) -> Path:
    dest_path = Path(dest).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([x_name, y_name])
        for x_val, y_val in rows:
            writer.writerow([_fmt(x_val), _fmt(y_val)])
    return dest_path


def xy_stats(rows: list[tuple[float, float]]) -> dict[str, Any] | None:
    if not rows:
        return None
    values = [y for _x, y in rows]
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 6),
        "max": round(max(values), 6),
        "last": round(values[-1], 6),
    }


def write_rmsd_csv(dest: str | Path, rows: list[tuple[float, float]]) -> Path:
    return write_xy_csv(dest, rows, x_name="time_ps", y_name="rmsd_nm")


def write_rmsf_csv(dest: str | Path, rows: list[tuple[float, float]]) -> Path:
    return write_xy_csv(dest, rows, x_name="atom_or_residue", y_name="rmsf_nm")


def _fmt(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")
