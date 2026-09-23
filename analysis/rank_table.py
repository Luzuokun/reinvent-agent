"""Per-molecule QED / SA / PAINS ranking from an existing SMILES CSV.

Uses the same RDKit helpers as ``analysis.molecule_analysis``. Drops invalid
SMILES and PAINS hits, then ranks remaining rows by QED (descending).
Does not invent scores.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from analysis.molecule_analysis import (
    _detect_smiles_column,
    _load_pains_catalog,
    _load_sa_scorer,
    _load_smiles_frame,
)


class RankError(ValueError):
    """Unreadable CSV or missing RDKit."""


def score_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise RankError(f"CSV not found: {path}")
    df, smiles_col = _load_smiles_frame(path)
    if smiles_col is None:
        raise RankError(f"No SMILES column in {path}")
    try:
        from rdkit import Chem
        from rdkit.Chem import QED
    except ImportError as exc:
        raise RankError("RDKit is required to rank molecules") from exc

    sa_fn, _sa_err = _load_sa_scorer()
    pains_catalog, pains_err = _load_pains_catalog()
    rows: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        smiles = str(row[smiles_col]).strip()
        record: dict[str, Any] = {
            "source_row": int(index) if isinstance(index, int) else len(rows),
            "SMILES": smiles,
            "valid": False,
            "QED": None,
            "SA": None,
            "PAINS": None,
            "pains_hits": None,
        }
        if not smiles or smiles.lower() == "nan":
            rows.append(record)
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rows.append(record)
            continue
        record["valid"] = True
        try:
            record["QED"] = round(float(QED.qed(mol)), 4)
        except Exception:  # noqa: BLE001
            record["QED"] = None
        if sa_fn is not None:
            try:
                record["SA"] = round(float(sa_fn(mol)), 4)
            except Exception:  # noqa: BLE001
                record["SA"] = None
        if pains_catalog is not None:
            try:
                n_hits = len(pains_catalog.GetMatches(mol))
                record["pains_hits"] = int(n_hits)
                record["PAINS"] = n_hits > 0
            except Exception:  # noqa: BLE001
                record["PAINS"] = None
                record["pains_hits"] = None
        else:
            record["PAINS"] = None
            record["note"] = pains_err
        rows.append(record)
    return rows


def rank_pains_clean(
    rows: list[dict[str, Any]],
    *,
    top_n: int | None = 50,
) -> list[dict[str, Any]]:
    """Keep valid, PAINS-clean molecules with a QED, sorted by QED descending."""
    kept: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("valid"):
            continue
        if row.get("PAINS") is True:
            continue
        if row.get("QED") is None:
            continue
        kept.append(dict(row))
    kept.sort(key=lambda item: (-float(item["QED"]), str(item.get("SMILES") or "")))
    if top_n is not None:
        return kept[: int(top_n)]
    return kept


def write_rank_csv(dest: str | Path, rows: list[dict[str, Any]]) -> Path:
    dest_path = Path(dest).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "SMILES",
        "QED",
        "SA",
        "PAINS",
        "pains_hits",
        "valid",
        "source_row",
    ]
    with dest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            payload = dict(row)
            payload["rank"] = index
            payload["PAINS"] = (
                "hit"
                if payload.get("PAINS") is True
                else ("clean" if payload.get("PAINS") is False else "")
            )
            writer.writerow(payload)
    return dest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Drop invalid/PAINS SMILES and rank remaining by QED."
    )
    parser.add_argument("--csv", required=True, help="Sampled SMILES CSV")
    parser.add_argument("--out", required=True, help="Ranked CSV destination")
    parser.add_argument("--top", type=int, default=50, help="How many to keep (0 = all)")
    parser.add_argument(
        "--keep-pains",
        action="store_true",
        help="Do not drop PAINS hits (still rank by QED)",
    )
    args = parser.parse_args(argv)
    rows = score_rows(args.csv)
    if args.keep_pains:
        ranked = [
            row
            for row in rows
            if row.get("valid") and row.get("QED") is not None
        ]
        ranked.sort(key=lambda item: (-float(item["QED"]), str(item.get("SMILES") or "")))
        if args.top > 0:
            ranked = ranked[: args.top]
    else:
        ranked = rank_pains_clean(rows, top_n=None if args.top <= 0 else args.top)
    path = write_rank_csv(args.out, ranked)
    n_valid = sum(1 for row in rows if row.get("valid"))
    n_pains = sum(1 for row in rows if row.get("PAINS") is True)
    print(
        f"scored={len(rows)} valid={n_valid} pains_hits={n_pains} "
        f"ranked={len(ranked)} wrote={path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
