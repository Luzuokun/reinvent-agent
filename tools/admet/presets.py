"""Human-written ADMET rule presets — no ML models, no invented assay numbers.

Each preset is a fixed threshold set over RDKit descriptors (MW / logP / HBD /
HBA / TPSA / rotatable bonds / QED). Agents may only name a ``preset_id`` from
this table. Missing RDKit is a loud failure; scores are never fabricated.
"""

from __future__ import annotations

from typing import Any

# Rule keys map 1:1 onto descriptors computed in ``score.py``.
# Direction (documented, fixed):
#   *_max — value must be <= threshold to pass that rule
#   *_min — value must be >= threshold to pass that rule
PREDEFINED_PRESETS: dict[str, dict[str, Any]] = {
    "lipinski": {
        "id": "lipinski",
        "description": (
            "Lipinski Rule-of-Five coarse filter: MW≤500, logP≤5, HBD≤5, HBA≤10. "
            "Offline RDKit descriptors only — not an assay or docking score."
        ),
        "rules": {
            "mw_max": 500.0,
            "logp_max": 5.0,
            "hbd_max": 5.0,
            "hba_max": 10.0,
        },
    },
    "veber": {
        "id": "veber",
        "description": (
            "Veber oral-bioavailability proxy: TPSA≤140, rotatable bonds≤10. "
            "Rule-based RDKit only; not a clinical ADME claim."
        ),
        "rules": {
            "tpsa_max": 140.0,
            "rotatable_bonds_max": 10.0,
        },
    },
    "leadlike": {
        "id": "leadlike",
        "description": (
            "Lead-like coarse filter (stricter than Lipinski): MW≤350, logP≤3.5, "
            "HBD≤3, HBA≤8, rotatable bonds≤7, TPSA≤90. Offline RDKit only."
        ),
        "rules": {
            "mw_max": 350.0,
            "logp_max": 3.5,
            "hbd_max": 3.0,
            "hba_max": 8.0,
            "rotatable_bonds_max": 7.0,
            "tpsa_max": 90.0,
        },
    },
    "qed-floor": {
        "id": "qed-floor",
        "description": (
            "QED floor filter: keep molecules with RDKit QED ≥ 0.5. "
            "Not a wet-lab druglikeness claim."
        ),
        "rules": {
            "qed_min": 0.5,
        },
    },
}

ALLOWED_PRESET_IDS: frozenset[str] = frozenset(PREDEFINED_PRESETS)

# Accepted CSV header aliases (case-insensitive) for the input molecules file.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "smiles": ("smiles", "canonical_smiles", "smiles_canonical"),
    "id": ("ligand_id", "id", "name", "molecule_id"),
}

KNOWN_RULE_KEYS: frozenset[str] = frozenset(
    {
        "mw_max",
        "logp_max",
        "hbd_max",
        "hba_max",
        "tpsa_max",
        "rotatable_bonds_max",
        "qed_min",
    }
)


class AdmetPresetError(ValueError):
    """Unknown or malformed ADMET rule preset id."""


def list_preset_ids() -> list[str]:
    return sorted(PREDEFINED_PRESETS)


def resolve_preset(preset_id: str) -> dict[str, Any]:
    """Return a copy of a predefined rule preset. Never invents thresholds."""
    if not isinstance(preset_id, str):
        raise AdmetPresetError("preset_id must be a string")
    key = preset_id.strip()
    if not key:
        raise AdmetPresetError("preset_id is empty")
    if key not in PREDEFINED_PRESETS:
        allowed = ", ".join(list_preset_ids())
        raise AdmetPresetError(
            f"Unknown ADMET preset_id {key!r}. Allowed: {allowed}. "
            "Do not invent rule thresholds; add a human PR to presets.py."
        )
    spec = dict(PREDEFINED_PRESETS[key])
    rules = dict(spec.get("rules") or {})
    unknown = sorted(set(rules) - KNOWN_RULE_KEYS)
    if unknown:
        raise AdmetPresetError(
            f"Preset {key!r} has unsupported rule keys: {unknown}. "
            "Fix presets.py (human edit only)."
        )
    spec["rules"] = rules
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
                "rules": dict(spec["rules"]),
            }
        )
    return rows
