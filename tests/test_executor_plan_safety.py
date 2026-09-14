"""Executor safety around LLM-shaped plans (no REINVENT launch)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from agents.executor import ExecutionAgent
from agents.planner import PlannerAgent

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "projects" / "demo_project"
SAMPLE = DEMO / "output" / "sampled-sample.csv"


def test_executor_uses_validated_plan_csv_path():
    plan = {
        "goal": "analyze specific csv",
        "project_dir": str(DEMO),
        "approve_run": False,
        "skip_reinvent": True,
        "steps": [
            "find_output",
            "analyze_molecules",
        ],
        "notes": [],
        "step_params": {"analyze_molecules": {"csv_path": str(SAMPLE.resolve())}},
        "csv_path": str(SAMPLE.resolve()),
    }
    results = ExecutionAgent().execute(
        plan,
        project_dir=DEMO,
        approve_run=False,
    )
    analysis = results["steps"]["analyze_molecules"]
    assert analysis.get("ok") is True
    assert Path(analysis["csv_path"]) == SAMPLE.resolve()


def test_executor_ignores_escaping_plan_csv_path():
    plan = {
        "goal": "escape",
        "project_dir": str(DEMO),
        "approve_run": False,
        "skip_reinvent": True,
        "steps": ["analyze_molecules"],
        "notes": [],
        "csv_path": str(REPO / "README.md"),
    }
    results = ExecutionAgent().execute(
        plan,
        project_dir=DEMO,
        approve_run=False,
    )
    warnings = " ".join(results.get("warnings") or [])
    assert "Ignoring unsafe or invalid plan csv_path" in warnings
    analysis = results["steps"]["analyze_molecules"]
    # Falls back to inventory / sample CSV under output — never README.md
    if analysis.get("csv_path"):
        assert Path(analysis["csv_path"]).resolve() != (REPO / "README.md").resolve()
        assert "output" in Path(analysis["csv_path"]).parts


def test_skip_reinvent_does_not_launch_even_if_plan_lists_run():
    plan = PlannerAgent().create_plan(
        "offline",
        project_dir=str(DEMO),
        approve_run=True,
        skip_reinvent=True,
    )
    # Simulate a bad/unsanitized LLM plan that still lists run_reinvent.
    plan["steps"] = ["run_reinvent", "find_output"]
    plan["skip_reinvent"] = True

    with patch("agents.executor.run_reinvent") as mocked:
        results = ExecutionAgent().execute(
            plan,
            project_dir=DEMO,
            approve_run=True,
        )
    mocked.assert_not_called()
    run = results["steps"]["run_reinvent"]
    assert run["skipped"] is True
    assert run["command"] == []
