"""Predefined Vina / GNINA argv — never interpolates model text."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

ALLOWED_ENGINES: tuple[str, ...] = ("vina", "gnina")

# Hard caps so a typo cannot launch a runaway job.
MAX_EXHAUSTIVENESS = 32
MAX_NUM_MODES = 20
MAX_LIGANDS = 50
MAX_BOX_SIZE = 60.0
MIN_BOX_SIZE = 1.0

_MODE_LINE = re.compile(
    r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
)


class DockingEngineError(ValueError):
    """Unknown engine, illegal numeric bounds, or unparsable log."""


def normalize_engine(engine: str) -> str:
    if not isinstance(engine, str) or not engine.strip():
        raise DockingEngineError("engine is required (vina or gnina)")
    name = engine.strip().lower()
    if name not in ALLOWED_ENGINES:
        raise DockingEngineError(
            f"Unknown docking engine {engine!r}; allowed: {list(ALLOWED_ENGINES)}"
        )
    return name


def executable_name(engine: str) -> str:
    """Map an allowlisted engine id to a PATH executable name.

    Callers must still resolve the binary with ``shutil.which``. The engine
    string is never used as a free-form path or shell fragment.
    """
    return normalize_engine(engine)


def clamp_exhaustiveness(value: int | None, *, default: int = 8) -> int:
    raw = default if value is None else int(value)
    if raw < 1:
        raise DockingEngineError("exhaustiveness must be >= 1")
    if raw > MAX_EXHAUSTIVENESS:
        raise DockingEngineError(
            f"exhaustiveness {raw} exceeds cap ({MAX_EXHAUSTIVENESS})"
        )
    return raw


def clamp_num_modes(value: int | None, *, default: int = 1) -> int:
    raw = default if value is None else int(value)
    if raw < 1:
        raise DockingEngineError("num_modes must be >= 1")
    if raw > MAX_NUM_MODES:
        raise DockingEngineError(f"num_modes {raw} exceeds cap ({MAX_NUM_MODES})")
    return raw


def clamp_max_ligands(value: int | None, *, default: int = 20) -> int:
    raw = default if value is None else int(value)
    if raw < 1:
        raise DockingEngineError("max_ligands must be >= 1")
    if raw > MAX_LIGANDS:
        raise DockingEngineError(f"max_ligands {raw} exceeds cap ({MAX_LIGANDS})")
    return raw


def validate_box(
    center: Sequence[float],
    size: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    cx, cy, cz = _three_floats(center, what="center")
    sx, sy, sz = _three_floats(size, what="size")
    for axis, value in (("x", sx), ("y", sy), ("z", sz)):
        if value < MIN_BOX_SIZE:
            raise DockingEngineError(
                f"box size_{axis}={value} is below minimum {MIN_BOX_SIZE}"
            )
        if value > MAX_BOX_SIZE:
            raise DockingEngineError(
                f"box size_{axis}={value} exceeds cap ({MAX_BOX_SIZE})"
            )
    return (cx, cy, cz), (sx, sy, sz)


def build_docking_command(
    *,
    engine: str,
    executable: str,
    receptor: str | Path,
    ligand: str | Path,
    out: str | Path,
    center: Sequence[float],
    size: Sequence[float],
    exhaustiveness: int,
    num_modes: int,
) -> list[str]:
    """Build the predefined Vina/GNINA argv. ``shell=False`` only."""
    engine = normalize_engine(engine)
    if not executable or executable != executable_name(engine):
        # executable must be the which()'d basename or an absolute which() path
        # whose basename matches the engine. Reject /bin/bash etc.
        base = Path(executable).name
        if base != executable_name(engine):
            raise DockingEngineError(
                f"executable basename {base!r} does not match engine {engine!r}"
            )
    center_t, size_t = validate_box(center, size)
    exhaustiveness = clamp_exhaustiveness(exhaustiveness)
    num_modes = clamp_num_modes(num_modes)
    return [
        str(executable),
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
        "--out",
        str(out),
        "--center_x",
        _fmt(center_t[0]),
        "--center_y",
        _fmt(center_t[1]),
        "--center_z",
        _fmt(center_t[2]),
        "--size_x",
        _fmt(size_t[0]),
        "--size_y",
        _fmt(size_t[1]),
        "--size_z",
        _fmt(size_t[2]),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(num_modes),
    ]


def parse_engine_log(text: str, *, engine: str) -> list[dict[str, Any]]:
    """Parse Vina/GNINA pose table lines from a log or stdout dump."""
    normalize_engine(engine)
    if not isinstance(text, str) or not text.strip():
        return []
    modes: list[dict[str, Any]] = []
    in_table = False
    for line in text.splitlines():
        if "-----+" in line:
            in_table = True
            continue
        if not in_table:
            continue
        match = _MODE_LINE.match(line)
        if not match:
            continue
        mode = int(match.group(1))
        score = float(match.group(2))
        rest = line[match.end() :].split()
        row: dict[str, Any] = {"mode": mode, "score": score}
        if engine == "gnina" and rest:
            try:
                row["cnn_score"] = float(rest[0])
            except ValueError:
                row["cnn_score"] = None
        modes.append(row)
    return modes


def _three_floats(values: Sequence[float], *, what: str) -> tuple[float, float, float]:
    if values is None or len(values) != 3:
        raise DockingEngineError(f"{what} must be three numbers (x y z)")
    out: list[float] = []
    for item in values:
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise DockingEngineError(f"{what} values must be numbers") from exc
        if number != number:  # NaN
            raise DockingEngineError(f"{what} contains NaN")
        if number in (float("inf"), float("-inf")):
            raise DockingEngineError(f"{what} contains Infinity")
        out.append(number)
    return out[0], out[1], out[2]


def _fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") if value != 0 else "0"
