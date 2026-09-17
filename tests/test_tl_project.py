"""Transfer-learning project validation and critic — no REINVENT launch."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.critic import CriticAgent
from agents.executor import ExecutionAgent
from agents.planner import PlannerAgent
from analysis.report import generate_html_report
from tools import resolve_under_root
from tools.files import validate_project
from tools.reinvent import find_output_files

REPO = Path(__file__).resolve().parent.parent
DEMO_TL = REPO / "projects" / "demo_tl"
TRAIN_SMI = DEMO_TL / "input" / "tl_train.smi"


def test_demo_tl_toml_is_transfer_learning():
    result = validate_project(DEMO_TL)
    details = result["details"]
    assert details["run_type"] == "transfer_learning"
    assert details["smiles_file"] == "input/tl_train.smi"
    assert TRAIN_SMI.is_file()
    assert details.get("smiles_path", "").endswith("tl_train.smi")
    assert "parameters.model_file is missing" not in result["errors"]
    prior = DEMO_TL / "priors" / "reinvent.prior"
    if prior.exists():
        assert result["ok"] is True
        assert not any("escapes" in err for err in result["errors"])


def test_find_output_records_model_checkpoints(tmp_path: Path):
    output_dir = tmp_path / "output"
    models_dir = tmp_path / "models"
    output_dir.mkdir()
    models_dir.mkdir()
    (output_dir / ".gitkeep").write_text("", encoding="utf-8")
    model = models_dir / "demo_tl.model"
    model.write_bytes(b"fake")
    inventory = find_output_files(
        output_dir, project_dir=tmp_path, extra_dirs=[models_dir]
    )
    assert inventory["ok"] is True
    assert inventory["csv_count"] == 0
    assert inventory["model_count"] == 1
    assert inventory["model_files"][0].endswith("demo_tl.model")


def test_critic_tl_success_without_csv_is_not_fail():
    plan = PlannerAgent().create_plan(
        "tl",
        project_dir=str(DEMO_TL),
        approve_run=True,
        skip_reinvent=False,
    )
    results = {
        "plan": plan,
        "steps": {
            "validate_project": {
                "ok": True,
                "details": {"run_type": "transfer_learning"},
            },
            "run_reinvent": {"skipped": False, "success": True, "exit_code": 0},
            "analyze_molecules": {
                "ok": True,
                "total_molecules": 15,
                "duplicate_fraction": 0.0,
                "analysis_source": "tl_training_set",
                "rdkit": {"available": True, "valid_fraction": 1.0},
            },
            "find_output": {"ok": True, "csv_count": 0, "model_count": 1},
        },
        "warnings": [],
        "errors": [],
    }
    verdict = CriticAgent().review(results)
    assert verdict["status"] == "PASS"
    assert any("training SMILES" in issue for issue in verdict["issues"])


def test_critic_tl_success_missing_model_fails():
    results = {
        "plan": {"approve_run": True, "skip_reinvent": False},
        "steps": {
            "validate_project": {
                "ok": True,
                "details": {"run_type": "transfer_learning"},
            },
            "run_reinvent": {"skipped": False, "success": True, "exit_code": 0},
            "analyze_molecules": {
                "ok": True,
                "total_molecules": 15,
                "analysis_source": "tl_training_set",
                "rdkit": {"available": True, "valid_fraction": 1.0},
            },
            "find_output": {"ok": True, "csv_count": 0, "model_count": 0},
        },
        "warnings": [],
        "errors": [],
    }
    verdict = CriticAgent().review(results)
    assert verdict["status"] == "FAIL"
    assert any("no model checkpoint" in issue.lower() for issue in verdict["issues"])


def test_executor_tl_dry_run_analyzes_training_smiles():
    plan = PlannerAgent().create_plan(
        "tl dry",
        project_dir=str(DEMO_TL),
        approve_run=False,
        skip_reinvent=False,
    )
    results = ExecutionAgent().execute(
        plan,
        project_dir=DEMO_TL,
        approve_run=False,
    )
    analysis = results["steps"]["analyze_molecules"]
    assert analysis.get("ok") is True
    assert analysis.get("analysis_source") == "tl_training_set"
    assert analysis.get("from_fresh_reinvent") is False
    assert analysis.get("total_molecules", 0) >= 1
    assert results["steps"]["run_reinvent"]["skipped"] is True


def test_html_report_tl_training_banner(tmp_path: Path):
    meta = generate_html_report(
        {
            "goal": "tl",
            "project": {"project_dir": str(DEMO_TL)},
            "environment": {},
            "validation": {"ok": True, "details": {"run_type": "transfer_learning"}},
            "execution": {"success": True, "skipped": False},
            "inventory": {"model_count": 1, "csv_count": 0},
            "analysis": {
                "ok": True,
                "csv_path": str(TRAIN_SMI),
                "analysis_source": "tl_training_set",
                "artefact_path": str(DEMO_TL / "models" / "demo_tl.model"),
                "total_molecules": 15,
            },
            "critic": {"status": "PASS", "issues": []},
            "warnings": [],
            "errors": [],
            "analysis_source": "tl_training_set",
        },
        reports_dir=tmp_path,
        filename="tl_banner.html",
    )
    html = Path(meta["report_path"]).read_text(encoding="utf-8")
    assert "training SMILES" in html
    assert "not a fresh sample" in html


def test_resolve_under_root_allows_project_local_symlink(tmp_path: Path):
    project = tmp_path / "proj"
    shared = tmp_path / "shared"
    project.mkdir()
    shared.mkdir()
    target = shared / "reinvent.prior"
    target.write_bytes(b"prior")
    link = project / "priors"
    link.mkdir()
    (link / "reinvent.prior").symlink_to(target)
    resolved = resolve_under_root(
        project, "priors/reinvent.prior", follow_symlinks=False
    )
    assert resolved.is_file()
    assert resolved.parent == link


def test_resolve_under_root_rejects_parent_escape(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    with pytest.raises(PermissionError, match="escapes"):
        resolve_under_root(project, "../secret.prior", follow_symlinks=False)


def test_validate_tl_rejects_input_model_outside_project(tmp_path: Path):
    project = tmp_path / "demo_tl"
    (project / "input").mkdir(parents=True)
    (project / "models").mkdir()
    (project / "priors").mkdir()
    (project / "output").mkdir()
    (project / "input" / "tl_train.smi").write_text("CCO\n", encoding="utf-8")
    (project / "reinvent.toml").write_text(
        "\n".join(
            [
                'run_type = "transfer_learning"',
                "device = \"cpu\"",
                "[parameters]",
                'input_model_file = "../outside.prior"',
                'smiles_file = "input/tl_train.smi"',
                'output_model_file = "models/demo_tl.model"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = validate_project(project)
    assert result["ok"] is False
    assert any("escapes" in err for err in result["errors"])
