"""LLM planner unit tests — mocked client only, no network."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.llm_planner import LLMPlannerAgent, LLMPlannerError
from agents.planner import PlannerAgent
from main import build_parser

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "projects" / "demo_project"

FULL_STEPS = list(PlannerAgent.DEFAULT_STEPS)
SKIP_STEPS = [
    "check_environment",
    "validate_project",
    "find_output",
    "analyze_molecules",
    "generate_report",
    "critic_review",
]


def _agent(complete_fn=None, client=None, **planner_cfg):
    cfg = {
        "project": {"output_dirname": "output"},
        "planner": {
            "mode": "llm",
            "fallback_on_error": True,
            "max_steps": 8,
            **planner_cfg,
        },
    }
    return LLMPlannerAgent(cfg, client=client, complete_fn=complete_fn)


def test_llm_planner_uses_mocked_valid_plan():
    payload = {"steps": SKIP_STEPS, "notes": ["offline analysis"], "csv_path": None}

    def complete(_messages):
        return json.dumps(payload)

    plan = _agent(complete_fn=complete).create_plan(
        "Analyze existing molecules",
        project_dir=str(DEMO),
        approve_run=False,
        skip_reinvent=True,
    )
    assert plan["planner"] == "llm"
    assert plan["planner_fallback"] is False
    assert plan["steps"] == SKIP_STEPS
    assert "offline analysis" in plan["notes"]


def test_llm_planner_mocked_chat_client():
    content = json.dumps(
        {"steps": SKIP_STEPS, "notes": [], "csv_path": "output/sampled-sample.csv"}
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    plan = _agent(client=client, api="chat_completions").create_plan(
        "Analyze sample CSV",
        project_dir=str(DEMO),
        approve_run=False,
        skip_reinvent=True,
    )
    assert plan["planner"] == "llm"
    assert plan["steps"] == SKIP_STEPS
    assert Path(plan["csv_path"]).name == "sampled-sample.csv"


def test_llm_planner_mocked_responses_client():
    content = json.dumps({"steps": SKIP_STEPS, "notes": [], "csv_path": None})
    response = SimpleNamespace(output_text=content, output=[])
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_kwargs: response)
    )
    plan = _agent(client=client, api="responses").create_plan(
        "Analyze existing molecules",
        project_dir=str(DEMO),
        approve_run=False,
        skip_reinvent=True,
    )
    assert plan["planner"] == "llm"
    assert plan["steps"] == SKIP_STEPS


def test_llm_planner_invalid_plan_falls_back(caplog):
    def complete(_messages):
        return json.dumps({"steps": ["hack_the_planet"], "notes": [], "csv_path": None})

    with caplog.at_level("WARNING"):
        plan = _agent(complete_fn=complete).create_plan(
            "goal",
            project_dir=str(DEMO),
            approve_run=False,
            skip_reinvent=False,
        )
    assert plan["planner"] == "deterministic"
    assert plan["planner_fallback"] is True
    assert plan["steps"] == FULL_STEPS
    assert any("falling back" in rec.message.lower() for rec in caplog.records)
    assert plan["planner_warnings"]


def test_llm_planner_missing_api_key_falls_back(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    plan = _agent().create_plan(
        "goal",
        project_dir=str(DEMO),
        approve_run=False,
        skip_reinvent=True,
    )
    assert plan["planner_fallback"] is True
    assert plan["steps"] == SKIP_STEPS
    assert any("API key" in w or "Missing" in w for w in plan["planner_warnings"])


def test_llm_planner_provider_error_falls_back():
    def complete(_messages):
        raise RuntimeError("provider 503")

    plan = _agent(complete_fn=complete).create_plan(
        "goal",
        project_dir=str(DEMO),
        approve_run=True,
        skip_reinvent=False,
    )
    assert plan["planner_fallback"] is True
    assert plan["steps"] == FULL_STEPS
    assert any("503" in w for w in plan["planner_warnings"])


def test_llm_planner_can_fail_hard_when_fallback_disabled():
    def complete(_messages):
        return json.dumps({"steps": ["not_a_step"], "notes": [], "csv_path": None})

    agent = _agent(complete_fn=complete, fallback_on_error=False)
    with pytest.raises(LLMPlannerError, match="Unknown plan step"):
        agent.create_plan(
            "goal",
            project_dir=str(DEMO),
            approve_run=False,
            skip_reinvent=False,
        )


def test_cli_planner_flag_defaults_and_choices():
    parser = build_parser()
    assert parser.parse_args([]).planner is None
    assert parser.parse_args(["--planner", "deterministic"]).planner == "deterministic"
    assert parser.parse_args(["--planner", "llm"]).planner == "llm"
    with pytest.raises(SystemExit):
        parser.parse_args(["--planner", "crewai"])
