"""Local SMILES prep — no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.plan_schema import ALLOWED_STEPS
from tools.smiles_prep import SmilesPrepError, prepare_training_smiles

REPO = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO / "projects" / "demo_project" / "output" / "sampled-sample.csv"


def test_smiles_prep_rejects_source_outside_repo(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "input").mkdir()
    outside = tmp_path / "escape.smi"
    outside.write_text("CCO\n", encoding="utf-8")
    with pytest.raises(SmilesPrepError, match="escapes"):
        prepare_training_smiles(project, outside, repo_root=REPO)


def test_smiles_prep_rejects_dest_outside_project(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    source = REPO / "projects" / "demo_project" / "output" / "sampled-sample.csv"
    if not source.is_file():
        pytest.skip("bundled sample CSV missing")
    with pytest.raises((SmilesPrepError, PermissionError)):
        prepare_training_smiles(
            project,
            source,
            dest_relative="../outside.smi",
            repo_root=REPO,
        )


def test_smiles_prep_canonicalizes_and_deduplicates(tmp_path: Path):
    pytest.importorskip("rdkit")
    project = tmp_path / "proj"
    (project / "input").mkdir(parents=True)
    source = project / "raw.smi"
    source.write_text("CCO\nOCC\nnot-a-molecule\nCCO\n", encoding="utf-8")
    result = prepare_training_smiles(
        project,
        source,
        dest_relative="input/tl_train.smi",
        repo_root=tmp_path,
    )
    assert result["ok"] is True
    assert result["total"] == 4
    assert result["invalid"] == 1
    assert result["valid_unique"] == 1
    dest = Path(result["dest"])
    lines = [ln for ln in dest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines == ["CCO"]


def test_smiles_prep_is_not_a_planner_step():
    assert "smiles_prep" not in ALLOWED_STEPS
    assert "prepare_training_smiles" not in ALLOWED_STEPS
