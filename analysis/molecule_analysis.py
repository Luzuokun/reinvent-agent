"""Molecule CSV analysis with optional RDKit descriptors."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

SMILES_CANDIDATES = ("SMILES", "smiles", "Smiles", "canonical_smiles", "CANONICAL_SMILES")


def _detect_smiles_column(columns: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for name in SMILES_CANDIDATES:
        if name in columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    for col in columns:
        if "smiles" in col.lower():
            return col
    return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = _mean(values)
    assert m is not None
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def analyze_molecules(csv_path: str | Path, *, max_rows: int | None = None) -> dict[str, Any]:
    """Analyze a REINVENT (or similar) molecule CSV.

    If RDKit is unavailable, returns counts/duplicates and a warning without
    crashing the caller.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    warnings: list[str] = []
    errors: list[str] = []

    if not csv_path.is_file():
        return {
            "ok": False,
            "csv_path": str(csv_path),
            "errors": [f"CSV not found: {csv_path}"],
            "warnings": warnings,
        }

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "csv_path": str(csv_path),
            "errors": [f"Failed to read CSV: {exc}"],
            "warnings": warnings,
        }

    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
        warnings.append(f"Truncated analysis to first {max_rows} rows")

    smiles_col = _detect_smiles_column(list(df.columns))
    if smiles_col is None:
        return {
            "ok": False,
            "csv_path": str(csv_path),
            "columns": list(df.columns),
            "errors": ["Could not detect a SMILES column"],
            "warnings": warnings,
        }

    smiles_series = df[smiles_col].astype(str).fillna("")
    smiles_list = [s.strip() for s in smiles_series.tolist() if s.strip()]
    total = len(smiles_list)
    unique = len(set(smiles_list))
    duplicates = total - unique
    duplicate_fraction = (duplicates / total) if total else 0.0

    state_col = None
    for candidate in ("SMILES_state", "smiles_state", "VALID"):
        if candidate in df.columns:
            state_col = candidate
            break

    reinvent_valid_count: int | None = None
    if state_col is not None:
        try:
            reinvent_valid_count = int((pd.to_numeric(df[state_col], errors="coerce") == 1).sum())
        except Exception:  # noqa: BLE001
            reinvent_valid_count = None

    rdkit_available = False
    rdkit_error: str | None = None
    valid = 0
    invalid = 0
    mw_vals: list[float] = []
    logp_vals: list[float] = []
    qed_vals: list[float] = []

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED

        rdkit_available = True
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid += 1
                continue
            valid += 1
            try:
                mw_vals.append(float(Descriptors.MolWt(mol)))
                logp_vals.append(float(Descriptors.MolLogP(mol)))
                qed_vals.append(float(QED.qed(mol)))
            except Exception:  # noqa: BLE001
                warnings.append("Descriptor calculation failed for at least one molecule")
    except Exception as exc:  # noqa: BLE001
        rdkit_error = str(exc)
        warnings.append(
            "RDKit unavailable; validity/MW/logP/QED skipped. "
            f"Detail: {rdkit_error}"
        )

    valid_fraction = (valid / total) if total and rdkit_available else None

    result: dict[str, Any] = {
        "ok": True,
        "csv_path": str(csv_path),
        "columns": list(df.columns),
        "smiles_column": smiles_col,
        "total_molecules": total,
        "unique_molecules": unique,
        "duplicate_molecules": duplicates,
        "duplicate_fraction": round(duplicate_fraction, 4),
        "reinvent_smiles_state_valid": reinvent_valid_count,
        "rdkit": {
            "available": rdkit_available,
            "error": rdkit_error,
            "valid_molecules": valid if rdkit_available else None,
            "invalid_molecules": invalid if rdkit_available else None,
            "valid_fraction": round(valid_fraction, 4) if valid_fraction is not None else None,
            "mw": {
                "mean": _round(_mean(mw_vals)),
                "stdev": _round(_stdev(mw_vals)),
                "n": len(mw_vals),
            },
            "logp": {
                "mean": _round(_mean(logp_vals)),
                "stdev": _round(_stdev(logp_vals)),
                "n": len(logp_vals),
            },
            "qed": {
                "mean": _round(_mean(qed_vals)),
                "stdev": _round(_stdev(qed_vals)),
                "n": len(qed_vals),
            },
        },
        "warnings": warnings,
        "errors": errors,
    }
    return result


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, ndigits)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze molecule CSV")
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_molecules(args.csv), indent=2))


if __name__ == "__main__":
    main()
