"""Shared helpers for loading agent config and resolving repo paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AGENT_CONFIG = REPO_ROOT / "config" / "agent.yaml"


def load_agent_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_AGENT_CONFIG
    with cfg_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Agent config must be a mapping: {cfg_path}")
    return data


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion for JSON / structured logging."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def dumps_pretty(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False)


def resolve_under_root(
    root: Path,
    *parts: str | Path,
    follow_symlinks: bool = True,
) -> Path:
    """Resolve a path and ensure the declared location stays under ``root``.

    Output paths should keep ``follow_symlinks=True`` so a symlink cannot
    redirect writes outside the project. Input paths (priors, training
    SMILES) may use ``follow_symlinks=False`` so a project-local symlink
    can point at a shared prior.
    """
    root = root.resolve()
    if not parts:
        return root
    joined = root / Path(*parts)
    if follow_symlinks:
        candidate = joined.resolve()
    else:
        candidate = Path(os.path.normpath(str(joined)))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"Path escapes project/repo root: {candidate} (root={root})"
        ) from exc
    return candidate
