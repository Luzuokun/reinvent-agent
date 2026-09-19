"""Sandbox paths for MD structure / topology inputs.

Inputs must already exist under ``<project>/input/``. The module does not
run pdb2gmx, does not fetch remote structures, and never ``shell=True``.
"""

from __future__ import annotations

from pathlib import Path

STRUCTURE_SUFFIXES = {".gro", ".pdb"}
TOPOLOGY_SUFFIXES = {".top"}
INPUT_DIRNAME = "input"


class MdPrepareError(ValueError):
    """Unsafe path, missing file, or unsupported suffix."""


def resolve_structure_path(project_dir: str | Path, structure: str | Path) -> Path:
    """Structure must already exist under ``<project>/input/``."""
    project = Path(project_dir).expanduser().resolve()
    input_dir = (project / INPUT_DIRNAME).resolve()
    path = _resolve_existing(structure, project=project, allowed_roots=(input_dir,))
    if path.suffix.lower() not in STRUCTURE_SUFFIXES:
        raise MdPrepareError(
            f"Unsupported structure suffix {path.suffix!r}; use .gro or .pdb"
        )
    return path


def resolve_topology_path(project_dir: str | Path, topology: str | Path) -> Path:
    """Topology must already exist under ``<project>/input/``."""
    project = Path(project_dir).expanduser().resolve()
    input_dir = (project / INPUT_DIRNAME).resolve()
    path = _resolve_existing(topology, project=project, allowed_roots=(input_dir,))
    if path.suffix.lower() not in TOPOLOGY_SUFFIXES:
        raise MdPrepareError(
            f"Unsupported topology suffix {path.suffix!r}; use .top"
        )
    return path


def _resolve_existing(
    raw: str | Path,
    *,
    project: Path,
    allowed_roots: tuple[Path, ...],
) -> Path:
    if raw is None or str(raw).strip() == "":
        raise MdPrepareError("path is empty")
    path = Path(str(raw).strip()).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path.resolve())
    else:
        candidates.append((project / path).resolve())
        cwd_try = (Path.cwd() / path).resolve()
        if cwd_try not in candidates:
            candidates.append(cwd_try)

    existing = [c for c in candidates if c.is_file()]
    if not existing:
        raise MdPrepareError(f"File not found: {path}")
    candidate = existing[0]
    for root in allowed_roots:
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    allowed = ", ".join(str(r) for r in allowed_roots)
    raise MdPrepareError(
        f"Path escapes allowed directories ({allowed}): {candidate}"
    )
