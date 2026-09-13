"""Basic offline tests (no REINVENT launch)."""

from __future__ import annotations

from pathlib import Path

from agents.critic import CriticAgent
from agents.planner import PlannerAgent
from analysis.molecule_analysis import analyze_molecules
from tools.environment import check_environment
from tools.files import validate_project

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "projects" / "demo_project"
SAMPLE_CSV = DEMO / "output" / "sampled-sample.csv"


def test_check_environment_returns_dict():
    result = check_environment()
    assert "python" in result
    assert "reinvent" in result
    assert "rdkit" in result
    assert "gpu" in result


def test_validate_project_demo_structure():
    result = validate_project(DEMO)
    assert result["details"]["config_path"].endswith("reinvent.toml")
    # Prior may be absent in CI/dev clones — ok flag depends on local setup
    assert "errors" in result
    assert "warnings" in result


def test_analyze_sample_csv():
    assert SAMPLE_CSV.is_file()
    result = analyze_molecules(SAMPLE_CSV)
    assert result["ok"] is True
    assert result["total_molecules"] > 0
    assert result["smiles_column"] == "SMILES"


def test_planner_and_critic_dry_run():
    plan = PlannerAgent().create_plan(
        "test",
        project_dir=str(DEMO),
        approve_run=False,
        skip_reinvent=True,
    )
    assert "analyze_molecules" in plan["steps"]
    assert "run_reinvent" not in plan["steps"]

    results = {
        "plan": plan,
        "steps": {
            "analyze_molecules": {
                "ok": True,
                "total_molecules": 10,
                "duplicate_fraction": 0.0,
                "rdkit": {"available": True, "valid_fraction": 1.0},
            },
            "find_output": {"ok": True, "csv_count": 1},
            "validate_project": {"ok": True},
        },
        "warnings": [],
        "errors": [],
    }
    verdict = CriticAgent().review(results)
    assert verdict["status"] == "PASS"
