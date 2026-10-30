"""Fixed local assay CSV schema for ChEMBL downloads."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

# Stable column order for TL / score fuel. Do not invent extra science columns.
ASSAY_CSV_FIELDS: tuple[str, ...] = (
    "molecule_chembl_id",
    "canonical_smiles",
    "target_chembl_id",
    "standard_type",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_chembl_id",
    "activity_id",
    "query_id",
    "origin",
)

SMILES_CSV_FIELDS: tuple[str, ...] = ("canonical_smiles", "molecule_chembl_id")


def write_activities_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(ASSAY_CSV_FIELDS), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ASSAY_CSV_FIELDS})
    return path


def write_smiles_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write a two-column CSV that ``smiles_prep`` can canonicalize locally."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(SMILES_CSV_FIELDS), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            smiles = str(row.get("canonical_smiles") or "").strip()
            if not smiles:
                continue
            writer.writerow(
                {
                    "canonical_smiles": smiles,
                    "molecule_chembl_id": str(row.get("molecule_chembl_id") or ""),
                }
            )
    return path
