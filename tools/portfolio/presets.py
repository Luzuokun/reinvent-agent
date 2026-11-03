"""Human-written portfolio weight presets — no LLM scorer.

Each preset is a fixed weight vector over optional cross-modal fields
(QED / SA / PAINS / dock / MD summary). Agents may only name a ``preset_id``
from this table. Missing input columns are skipped (never invented).
"""

from __future__ import annotations

from typing import Any

# Weights apply only when that column is present on a row.
# Direction (documented, fixed):
#   QED  — higher better → positive weight
#   SA   — lower better → negative weight
#   PAINS — clean=0 / hit=1 → negative weight (penalty)
#   dock_score — more negative (Vina) better → negative weight
#   md_summary — higher better when present (e.g. stability proxy) → positive
#                OR boolean in_md (True→1 / False→0) when md_summary absent
PREDEFINED_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "id": "balanced",
        "description": (
            "Balanced cross-modal shortlist: QED up, SA down, PAINS penalty, "
            "dock_score (Vina-style, more negative better), optional MD summary."
        ),
        "weights": {
            "qed": 1.0,
            "sa": -0.15,
            "pains": -5.0,
            "dock": -1.0,
            "md": 0.5,
        },
        "drop_pains": True,
        "top_n": 10,
    },
    "ligand-prop": {
        "id": "ligand-prop",
        "description": (
            "Ligand-property only (QED / SA / PAINS). Ignores dock and MD "
            "even when those columns exist."
        ),
        "weights": {
            "qed": 1.0,
            "sa": -0.2,
            "pains": -5.0,
            "dock": 0.0,
            "md": 0.0,
        },
        "drop_pains": True,
        "top_n": 20,
    },
    "dock-first": {
        "id": "dock-first",
        "description": (
            "Docking-first shortlist: strong dock_score weight, modest QED, "
            "SA/PAINS soft filters. Requires a dock_score (or alias) column."
        ),
        "weights": {
            "qed": 0.3,
            "sa": -0.05,
            "pains": -5.0,
            "dock": -2.0,
            "md": 0.25,
        },
        "drop_pains": True,
        "top_n": 10,
        "require_fields": ("dock",),
    },
}

ALLOWED_PRESET_IDS: frozenset[str] = frozenset(PREDEFINED_PRESETS)

# Canonical modality → accepted CSV header aliases (case-insensitive).
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "smiles": ("smiles", "canonical_smiles", "smiles_canonical"),
    "qed": ("qed",),
    "sa": ("sa", "sa_score", "sascore"),
    "pains": ("pains", "pains_hit", "pains_flag"),
    "dock": ("dock_score", "score", "docking_score", "vina_score", "affinity"),
    "md": ("md_summary", "md_score", "rmsd_nm", "in_md"),
    "id": ("ligand_id", "id", "name", "molecule_id"),
}


class PortfolioPresetError(ValueError):
    """Unknown or malformed weight preset id."""


def list_preset_ids() -> list[str]:
    return sorted(PREDEFINED_PRESETS)


def resolve_preset(preset_id: str) -> dict[str, Any]:
    """Return a copy of a predefined weight preset. Never invents weights."""
    if not isinstance(preset_id, str):
        raise PortfolioPresetError("preset_id must be a string")
    key = preset_id.strip()
    if not key:
        raise PortfolioPresetError("preset_id is empty")
    if key not in PREDEFINED_PRESETS:
        allowed = ", ".join(list_preset_ids())
        raise PortfolioPresetError(
            f"Unknown portfolio preset_id {key!r}. Allowed: {allowed}. "
            "Do not invent weight vectors; add a human PR to presets.py."
        )
    spec = dict(PREDEFINED_PRESETS[key])
    spec["weights"] = dict(spec["weights"])
    if "require_fields" in spec:
        spec["require_fields"] = tuple(spec["require_fields"])
    return spec


def preset_catalog() -> list[dict[str, Any]]:
    """Public catalog for CLI / docs (no scoring)."""
    rows: list[dict[str, Any]] = []
    for pid in list_preset_ids():
        spec = PREDEFINED_PRESETS[pid]
        rows.append(
            {
                "id": spec["id"],
                "description": spec["description"],
                "weights": dict(spec["weights"]),
                "drop_pains": bool(spec.get("drop_pains", True)),
                "top_n": int(spec.get("top_n", 10)),
                "require_fields": list(spec.get("require_fields") or ()),
            }
        )
    return rows
