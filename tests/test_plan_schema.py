"""Strict plan validator tests (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.plan_schema import PlanValidationError, validate_plan
from agents.planner import PlannerAgent

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "projects" / "demo_project"
OUTPUT = DEMO / "output"
SAMPLE_CSV = OUTPUT / "sampled-sample.csv"

VALID_STEPS = [
    "check_environment",
    "validate_project",
    "prepare_execution",
    "run_reinvent",
    "find_output",
    "analyze_molecules",
    "generate_report",
    "critic_review",
]


def _validate(raw, **kwargs):
    defaults = {
        "project_dir": DEMO,
        "output_dir": OUTPUT,
        "source": "llm",
    }
    defaults.update(kwargs)
    return validate_plan(raw, **defaults)


def test_validate_plan_accepts_allowlisted_steps():
    result = _validate({"steps": VALID_STEPS, "notes": [], "csv_path": None})
    assert result["steps"] == VALID_STEPS
    assert result["step_params"] == {}


def test_validate_plan_rejects_unknown_step():
    with pytest.raises(PlanValidationError, match="Unknown plan step"):
        _validate({"steps": ["check_environment", "rm_rf"]})


def test_validate_plan_rejects_unknown_top_level_key():
    with pytest.raises(PlanValidationError, match="Unknown plan keys"):
        _validate({"steps": ["check_environment"], "shell": "rm -rf /"})


def test_validate_plan_rejects_model_setting_approve_run():
    with pytest.raises(PlanValidationError, match="Unknown plan keys"):
        _validate(
            {
                "steps": ["run_reinvent"],
                "approve_run": True,
            },
            source="llm",
        )


def test_validate_plan_rejects_unknown_step_param():
    with pytest.raises(PlanValidationError, match="Unknown params"):
        _validate(
            {
                "steps": [
                    {
                        "name": "run_reinvent",
                        "params": {"config_path": "/tmp/evil.toml"},
                    }
                ]
            }
        )


def test_validate_plan_rejects_csv_path_escape():
    with pytest.raises(PlanValidationError, match="escapes"):
        _validate(
            {
                "steps": ["analyze_molecules"],
                "csv_path": "../demo_project/reinvent.toml",
            }
        )


def test_validate_plan_rejects_absolute_csv_outside_output(tmp_path: Path):
    outsider = tmp_path / "outside.csv"
    outsider.write_text("SMILES\nC\n", encoding="utf-8")
    with pytest.raises(PlanValidationError, match="escapes"):
        _validate(
            {
                "steps": ["analyze_molecules"],
                "csv_path": str(outsider),
            }
        )


def test_validate_plan_rejects_dotdot_escape_under_output_name():
    with pytest.raises(PlanValidationError, match="escapes"):
        _validate(
            {
                "steps": [
                    {
                        "name": "analyze_molecules",
                        "params": {"csv_path": "output/../../README.md"},
                    }
                ]
            }
        )


def test_validate_plan_accepts_existing_csv_under_output():
    result = _validate(
        {
            "steps": ["analyze_molecules"],
            "csv_path": "output/sampled-sample.csv",
        }
    )
    assert Path(result["csv_path"]) == SAMPLE_CSV.resolve()
    assert result["step_params"]["analyze_molecules"]["csv_path"] == str(
        SAMPLE_CSV.resolve()
    )


def test_validate_plan_rejects_missing_csv():
    with pytest.raises(PlanValidationError, match="does not exist"):
        _validate(
            {
                "steps": ["analyze_molecules"],
                "csv_path": "output/does-not-exist.csv",
            }
        )


def test_deterministic_plan_steps_unchanged():
    plan = PlannerAgent().create_plan(
        "test",
        project_dir=str(DEMO),
        approve_run=False,
        skip_reinvent=False,
    )
    assert plan["steps"] == VALID_STEPS
    assert plan["planner"] == "deterministic"
    assert plan["planner_fallback"] is False

    skipped = PlannerAgent().create_plan(
        "offline",
        project_dir=str(DEMO),
        approve_run=False,
        skip_reinvent=True,
    )
    assert skipped["steps"] == [
        "check_environment",
        "validate_project",
        "find_output",
        "analyze_molecules",
        "generate_report",
        "critic_review",
    ]
    assert "run_reinvent" not in skipped["steps"]
