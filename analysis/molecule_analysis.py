"""Molecule CSV analysis with optional RDKit descriptors."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd

SMILES_CANDIDATES = ("SMILES", "smiles", "Smiles", "canonical_smiles", "CANONICAL_SMILES")
DEFAULT_QED_PASS_THRESHOLD = 0.5
HISTOGRAM_BINS = 10
LIPINSKI_RULES = "MW<=500, LogP<=5, HBD<=5, HBA<=10"


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


def _load_smiles_frame(path: Path) -> tuple[Any, str | None]:
    """Load a CSV or whitespace ``.smi`` into a DataFrame plus SMILES column name."""
    suffix = path.suffix.lower()
    if suffix in {".smi", ".smiles"}:
        rows: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            token = text.split()[0].strip()
            if token:
                rows.append(token)
        df = pd.DataFrame({"SMILES": rows})
        return df, "SMILES"
    df = pd.read_csv(path)
    return df, _detect_smiles_column(list(df.columns))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = _mean(values)
    assert m is not None
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def _round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, ndigits)


def _stat_dict(values: list[float]) -> dict[str, Any]:
    return {
        "mean": _round(_mean(values)),
        "stdev": _round(_stdev(values)),
        "n": len(values),
        "min": _round(min(values)) if values else None,
        "max": _round(max(values)) if values else None,
    }


def _histogram(
    values: list[float],
    *,
    n_bins: int = HISTOGRAM_BINS,
    range_min: float | None = None,
    range_max: float | None = None,
) -> dict[str, Any]:
    """Equal-width histogram. Empty input yields empty bins (no invented counts)."""
    if not values:
        return {"bin_edges": [], "counts": [], "n": 0, "min": None, "max": None}

    vmin = min(values) if range_min is None else range_min
    vmax = max(values) if range_max is None else range_max
    data_min = min(values)
    data_max = max(values)
    n = len(values)
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if vmin == vmax:
        return {
            "bin_edges": [_round(vmin), _round(vmax)],
            "counts": [n],
            "n": n,
            "min": _round(data_min),
            "max": _round(data_max),
        }

    n_bins = max(1, int(n_bins))
    width = (vmax - vmin) / n_bins
    edges = [vmin + i * width for i in range(n_bins + 1)]
    edges[-1] = vmax
    counts = [0] * n_bins
    for x in values:
        if x >= vmax:
            idx = n_bins - 1
        elif x <= vmin:
            idx = 0
        else:
            idx = int((x - vmin) / width)
            idx = min(max(idx, 0), n_bins - 1)
        counts[idx] += 1
    return {
        "bin_edges": [_round(e) for e in edges],
        "counts": counts,
        "n": n,
        "min": _round(data_min),
        "max": _round(data_max),
    }


def _load_sa_scorer() -> tuple[Callable[[Any], float] | None, str | None]:
    try:
        from rdkit.Contrib.SA_Score import sascorer

        fn = sascorer.calculateScore
        if not callable(fn):
            return None, "SA Score calculateScore is not callable"
        return fn, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _load_pains_catalog() -> tuple[Any | None, str | None]:
    try:
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        return FilterCatalog(params), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _import_rdkit_modules() -> tuple[Any, Any, Any]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED

    return Chem, Descriptors, QED


def _empty_rdkit_block(
    *,
    rdkit_error: str | None,
    qed_pass_threshold: float,
) -> dict[str, Any]:
    empty_stats = {"mean": None, "stdev": None, "n": 0, "min": None, "max": None}
    empty_hist = {"bin_edges": [], "counts": [], "n": 0, "min": None, "max": None}
    return {
        "available": False,
        "error": rdkit_error,
        "valid_molecules": None,
        "invalid_molecules": None,
        "valid_fraction": None,
        "mw": dict(empty_stats),
        "logp": dict(empty_stats),
        "qed": dict(empty_stats),
        "tpsa": dict(empty_stats),
        "hbd": dict(empty_stats),
        "hba": dict(empty_stats),
        "rotatable_bonds": dict(empty_stats),
        "sa_score": {**empty_stats, "available": False},
        "pains": {
            "available": False,
            "molecules_with_hits": None,
            "total_hits": None,
            "hit_fraction": None,
        },
        "lipinski": {
            "rules": LIPINSKI_RULES,
            "pass": None,
            "fail": None,
            "fraction": None,
        },
        "filters": {
            "qed_pass_threshold": qed_pass_threshold,
            "qed_pass_count": None,
            "pains_free_count": None,
            "qed_pass_and_pains_free_count": None,
        },
        "histograms": {
            "qed": dict(empty_hist),
            "mw": dict(empty_hist),
            "logp": dict(empty_hist),
        },
    }


def analyze_molecules(
    csv_path: str | Path,
    *,
    max_rows: int | None = None,
    qed_pass_threshold: float = DEFAULT_QED_PASS_THRESHOLD,
) -> dict[str, Any]:
    """Analyze a REINVENT (or similar) molecule CSV or training ``.smi``.

    If RDKit is unavailable, returns counts/duplicates and a warning without
    crashing the caller. SA Score / PAINS / histograms are omitted (nulls),
    never fabricated.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    warnings: list[str] = []
    errors: list[str] = []
    qed_pass_threshold = float(qed_pass_threshold)

    if not csv_path.is_file():
        return {
            "ok": False,
            "csv_path": str(csv_path),
            "errors": [f"CSV not found: {csv_path}"],
            "warnings": warnings,
        }

    try:
        df, smiles_col = _load_smiles_frame(csv_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "csv_path": str(csv_path),
            "errors": [f"Failed to read molecule file: {exc}"],
            "warnings": warnings,
        }

    if smiles_col is None:
        return {
            "ok": False,
            "csv_path": str(csv_path),
            "columns": list(df.columns) if df is not None else [],
            "errors": ["Could not detect a SMILES column"],
            "warnings": warnings,
        }

    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
        warnings.append(f"Truncated analysis to first {max_rows} rows")

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
    tpsa_vals: list[float] = []
    hbd_vals: list[float] = []
    hba_vals: list[float] = []
    rotb_vals: list[float] = []
    lipinski_pass = 0
    lipinski_fail = 0
    sa_vals: list[float] = []
    sa_available = False
    pains_available = False
    pains_mols = 0
    pains_hits = 0
    qed_pass = 0
    pains_free = 0
    qed_and_pains_free = 0
    scored_for_filters = 0

    try:
        Chem, Descriptors, QED = _import_rdkit_modules()

        rdkit_available = True
        sa_fn, sa_error = _load_sa_scorer()
        if sa_fn is None:
            warnings.append(
                "SA Score unavailable; mean SA skipped. "
                f"Detail: {sa_error}"
            )
        else:
            sa_available = True

        pains_catalog, pains_error = _load_pains_catalog()
        if pains_catalog is None:
            warnings.append(
                "PAINS catalog unavailable; PAINS counts skipped. "
                f"Detail: {pains_error}"
            )
        else:
            pains_available = True

        sa_failed = False
        pains_failed = False
        descriptor_failed = False
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid += 1
                continue
            valid += 1
            qed: float | None = None
            try:
                mw = float(Descriptors.MolWt(mol))
                logp = float(Descriptors.MolLogP(mol))
                qed = float(QED.qed(mol))
                tpsa = float(Descriptors.TPSA(mol))
                hbd = float(Descriptors.NumHDonors(mol))
                hba = float(Descriptors.NumHAcceptors(mol))
                rotb = float(Descriptors.NumRotatableBonds(mol))
            except Exception:  # noqa: BLE001
                descriptor_failed = True
            else:
                mw_vals.append(mw)
                logp_vals.append(logp)
                qed_vals.append(qed)
                tpsa_vals.append(tpsa)
                hbd_vals.append(hbd)
                hba_vals.append(hba)
                rotb_vals.append(rotb)
                if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
                    lipinski_pass += 1
                else:
                    lipinski_fail += 1

            if sa_fn is not None:
                try:
                    sa_vals.append(float(sa_fn(mol)))
                except Exception:  # noqa: BLE001
                    sa_failed = True

            pains_hit: bool | None = None
            if pains_catalog is not None:
                try:
                    n_hits = len(pains_catalog.GetMatches(mol))
                    pains_hit = n_hits > 0
                    if n_hits:
                        pains_mols += 1
                        pains_hits += n_hits
                except Exception:  # noqa: BLE001
                    pains_failed = True

            if qed is not None:
                scored_for_filters += 1
                passes_qed = qed >= qed_pass_threshold
                if passes_qed:
                    qed_pass += 1
                if pains_hit is False:
                    pains_free += 1
                    if passes_qed:
                        qed_and_pains_free += 1

        if descriptor_failed:
            warnings.append("Descriptor calculation failed for at least one molecule")
        if sa_failed:
            warnings.append("SA Score calculation failed for at least one molecule")
        if pains_failed:
            warnings.append("PAINS matching failed for at least one molecule")
    except Exception as exc:  # noqa: BLE001
        rdkit_error = str(exc)
        warnings.append(
            "RDKit unavailable; validity/MW/logP/QED/SA/PAINS skipped. "
            f"Detail: {rdkit_error}"
        )

    valid_fraction = (valid / total) if total and rdkit_available else None
    if rdkit_available:
        sa_stats = _stat_dict(sa_vals)
        sa_stats["available"] = sa_available
        pains_block: dict[str, Any] = {
            "available": pains_available,
            "molecules_with_hits": pains_mols if pains_available else None,
            "total_hits": pains_hits if pains_available else None,
            "hit_fraction": (
                _round(pains_mols / valid) if pains_available and valid else None
            ),
        }
        scored_lipinski = lipinski_pass + lipinski_fail
        filters_block = {
            "qed_pass_threshold": qed_pass_threshold,
            "qed_pass_count": qed_pass if scored_for_filters else None,
            "pains_free_count": pains_free if pains_available else None,
            "qed_pass_and_pains_free_count": (
                qed_and_pains_free if pains_available and scored_for_filters else None
            ),
        }
        rdkit_block: dict[str, Any] = {
            "available": True,
            "error": rdkit_error,
            "valid_molecules": valid,
            "invalid_molecules": invalid,
            "valid_fraction": round(valid_fraction, 4) if valid_fraction is not None else None,
            "mw": _stat_dict(mw_vals),
            "logp": _stat_dict(logp_vals),
            "qed": _stat_dict(qed_vals),
            "tpsa": _stat_dict(tpsa_vals),
            "hbd": _stat_dict(hbd_vals),
            "hba": _stat_dict(hba_vals),
            "rotatable_bonds": _stat_dict(rotb_vals),
            "sa_score": sa_stats,
            "pains": pains_block,
            "lipinski": {
                "rules": LIPINSKI_RULES,
                "pass": lipinski_pass,
                "fail": lipinski_fail,
                "fraction": (
                    round(lipinski_pass / scored_lipinski, 4) if scored_lipinski else None
                ),
            },
            "filters": filters_block,
            "histograms": {
                "qed": _histogram(qed_vals, range_min=0.0, range_max=1.0),
                "mw": _histogram(mw_vals),
                "logp": _histogram(logp_vals),
            },
        }
    else:
        rdkit_block = _empty_rdkit_block(
            rdkit_error=rdkit_error, qed_pass_threshold=qed_pass_threshold
        )

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
        "rdkit": rdkit_block,
        "warnings": warnings,
        "errors": errors,
    }
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze molecule CSV")
    parser.add_argument("--csv", required=True)
    parser.add_argument(
        "--qed-pass-threshold",
        type=float,
        default=DEFAULT_QED_PASS_THRESHOLD,
        help="Count molecules with QED at or above this value (analysis only)",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_molecules(args.csv, qed_pass_threshold=args.qed_pass_threshold),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
