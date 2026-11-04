"""Local campaign run index — not CRM.

Append-only JSON list under ``<project>/output/portfolio/run_index.json``.
Records paths and preset ids for local recall. No accounts, no cloud sync,
no customer fields.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INDEX_FILENAME = "run_index.json"


def load_run_index(output_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(output_dir) / INDEX_FILENAME
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, dict) and isinstance(raw.get("runs"), list):
        return [r for r in raw["runs"] if isinstance(r, dict)]
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    return []


def append_run_index(
    output_dir: str | Path,
    entry: dict[str, Any],
) -> Path:
    """Append one local index entry. Never deletes prior runs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_run_index(out)
    payload = dict(entry)
    payload.setdefault(
        "recorded_at_utc",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    payload.setdefault("kind", "portfolio_shortlist")
    payload.setdefault("crm", False)
    runs.append(payload)
    path = out / INDEX_FILENAME
    doc = {
        "kind": "local_portfolio_run_index",
        "crm": False,
        "note": "Local campaign recall only — not a CRM.",
        "runs": runs,
    }
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
