"""Human-written ChEMBL query allowlist — no LLM SQL / curl.

Each query is a fixed ``target_chembl_id`` + ``standard_type`` (+ optional
assay filter). Agents may only name a ``query_id`` from this table.
"""

from __future__ import annotations

from typing import Any, Mapping

# Keep this list short and scientific. New targets need a human PR.
PREDEFINED_QUERIES: dict[str, dict[str, Any]] = {
    "egfr-ic50": {
        "id": "egfr-ic50",
        "description": (
            "EGFR kinase (CHEMBL203) binding IC50 activities with canonical SMILES"
        ),
        "target_chembl_id": "CHEMBL203",
        "standard_type": "IC50",
        "assay_type": "B",
        "default_limit": 50,
    },
    "braf-ic50": {
        "id": "braf-ic50",
        "description": (
            "BRAF (CHEMBL5145) binding IC50 activities with canonical SMILES"
        ),
        "target_chembl_id": "CHEMBL5145",
        "standard_type": "IC50",
        "assay_type": "B",
        "default_limit": 50,
    },
}

ALLOWED_QUERY_IDS: frozenset[str] = frozenset(PREDEFINED_QUERIES)


class ChemblQueryError(ValueError):
    """Unknown or malformed predefined query id."""


def list_query_ids() -> list[str]:
    return sorted(PREDEFINED_QUERIES)


def resolve_query(query_id: str) -> dict[str, Any]:
    """Return a copy of a predefined query. Never invents a new target filter."""
    if not isinstance(query_id, str):
        raise ChemblQueryError("query_id must be a string")
    key = query_id.strip()
    if not key:
        raise ChemblQueryError("query_id is empty")
    if key not in PREDEFINED_QUERIES:
        allowed = ", ".join(list_query_ids())
        raise ChemblQueryError(
            f"Unknown ChEMBL query_id {key!r}. Allowed: {allowed}. "
            "Do not invent SQL or free-form ChEMBL filters."
        )
    return dict(PREDEFINED_QUERIES[key])


def query_catalog() -> list[dict[str, Any]]:
    """Public catalog for CLI / docs (no network)."""
    rows: list[dict[str, Any]] = []
    for qid in list_query_ids():
        spec = PREDEFINED_QUERIES[qid]
        rows.append(
            {
                "id": spec["id"],
                "description": spec["description"],
                "target_chembl_id": spec["target_chembl_id"],
                "standard_type": spec["standard_type"],
                "assay_type": spec.get("assay_type"),
                "default_limit": int(spec.get("default_limit", 50)),
            }
        )
    return rows


def activity_filter_params(spec: Mapping[str, Any]) -> dict[str, str]:
    """Map a predefined query to ChEMBL ``/activity`` GET params."""
    params = {
        "target_chembl_id": str(spec["target_chembl_id"]),
        "standard_type": str(spec["standard_type"]),
        "format": "json",
    }
    assay_type = spec.get("assay_type")
    if assay_type:
        params["assay_type"] = str(assay_type)
    return params
