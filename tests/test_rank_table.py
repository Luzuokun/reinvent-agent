"""Rank-table helper: drop invalid/PAINS, sort by QED. No invented scores."""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.rank_table import rank_pains_clean, score_rows, write_rank_csv


def test_rank_drops_invalid_and_pains_then_sorts_by_qed(tmp_path: Path):
    pytest.importorskip("rdkit")
    csv_path = tmp_path / "mols.csv"
    csv_path.write_text(
        "SMILES\n"
        "not-a-smiles\n"
        "CCO\n"
        "O=C1C=CC(=O)C=C1\n"
        "c1ccccc1\n",
        encoding="utf-8",
    )
    rows = score_rows(csv_path)
    ranked = rank_pains_clean(rows, top_n=50)
    smiles = [row["SMILES"] for row in ranked]
    assert "not-a-smiles" not in smiles
    assert "O=C1C=CC(=O)C=C1" not in smiles
    assert "CCO" in smiles
    assert "c1ccccc1" in smiles
    qeds = [row["QED"] for row in ranked]
    assert qeds == sorted(qeds, reverse=True)
    out = write_rank_csv(tmp_path / "ranked.csv", ranked)
    text = out.read_text(encoding="utf-8")
    assert "QED" in text
    assert "PAINS" in text
