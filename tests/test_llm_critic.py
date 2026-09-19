"""LLM critic unit tests — mocked client only, no network."""

from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agents.critic import CriticAgent
from agents.llm_critic import LLMCriticAgent, LLMCriticError
from main import _review_with_critic, build_parser

PASSING_RESULTS = {
    "plan": {
        "approve_run": False,
        "skip_reinvent": True,
        "steps": ["analyze_molecules", "critic_review"],
        "planner": "deterministic",
        "planner_fallback": False,
    },
    "steps": {
        "analyze_molecules": {
            "ok": True,
            "total_molecules": 10,
            "duplicate_fraction": 0.0,
            "analysis_source": "existing_csv",
            "rdkit": {"available": True, "valid_fraction": 1.0},
        },
        "find_output": {"ok": True, "csv_count": 1},
        "validate_project": {"ok": True},
    },
    "warnings": [],
    "errors": [],
}

FAILING_RESULTS = {
    "plan": {"approve_run": True, "skip_reinvent": False, "steps": ["run_reinvent"]},
    "aborted": True,
    "abort_reason": "REINVENT run failed",
    "steps": {
        "run_reinvent": {"skipped": False, "success": False, "exit_code": 1, "message": "boom"},
        "validate_project": {"ok": True},
    },
    "warnings": [],
    "errors": ["REINVENT failed"],
}


def _agent(complete_fn=None, client=None, **critic_cfg):
    cfg = {
        "critic": {
            "mode": "llm",
            "fallback_on_error": True,
            "min_valid_fraction": 0.80,
            "max_duplicate_fraction": 0.25,
            "min_molecules": 1,
            **critic_cfg,
        }
    }
    return LLMCriticAgent(cfg, client=client, complete_fn=complete_fn)


def test_llm_critic_uses_mocked_valid_verdict():
    payload = {
        "status": "PASS",
        "issues": [],
        "recommendation": "Offline analysis evidence is consistent.",
    }

    def complete(_messages):
        return json.dumps(payload)

    verdict = _agent(complete_fn=complete).review(PASSING_RESULTS)
    assert verdict["critic"] == "llm"
    assert verdict["critic_fallback"] is False
    assert verdict["status"] == "PASS"
    assert verdict["recommendation"] == payload["recommendation"]
    assert "min_valid_fraction" in verdict["thresholds"]


def test_llm_critic_mocked_chat_client():
    content = json.dumps(
        {
            "status": "WARNING",
            "issues": ["Dry-run used an existing CSV."],
            "recommendation": "Human review recommended.",
        }
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    verdict = _agent(client=client, api="chat_completions").review(PASSING_RESULTS)
    assert verdict["critic"] == "llm"
    assert verdict["status"] == "WARNING"


def test_llm_critic_mocked_responses_client():
    content = json.dumps(
        {
            "status": "FAIL",
            "issues": ["REINVENT run failed"],
            "recommendation": "Fix errors and re-run.",
        }
    )
    response = SimpleNamespace(output_text=content, output=[])
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_kwargs: response)
    )
    verdict = _agent(client=client, api="responses").review(FAILING_RESULTS)
    assert verdict["critic"] == "llm"
    assert verdict["status"] == "FAIL"


def test_llm_critic_gemini_mocked_client_uses_preset_model():
    captured: dict = {}
    content = json.dumps(
        {
            "status": "PASS",
            "issues": [],
            "recommendation": "Offline analysis evidence is consistent.",
        }
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )

    def create(**kwargs):
        captured.update(kwargs)
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    verdict = _agent(client=client, provider="gemini").review(PASSING_RESULTS)
    assert verdict["critic"] == "llm"
    assert captured["model"] == "gemini-2.5-flash"


def test_llm_critic_messages_are_evidence_only():
    captured: list[list[dict[str, str]]] = []

    def complete(messages):
        captured.append(messages)
        return json.dumps(
            {
                "status": "PASS",
                "issues": [],
                "recommendation": "ok",
            }
        )

    results = {
        **PASSING_RESULTS,
        "steps": {
            **PASSING_RESULTS["steps"],
            "analyze_molecules": {
                **PASSING_RESULTS["steps"]["analyze_molecules"],
                "smiles": ["CCO", "CCN"],
            },
        },
    }
    _agent(complete_fn=complete).review(results)
    assert captured
    blob = json.dumps(captured[0])
    assert "CCO" not in blob
    assert "CCN" not in blob
    assert "structured workflow evidence" in captured[0][1]["content"]
    assert "do not invent" in captured[0][0]["content"].lower() or "Do not invent" in captured[0][0]["content"]


def test_llm_critic_does_not_call_tools():
    def complete(_messages):
        return json.dumps(
            {
                "status": "PASS",
                "issues": [],
                "recommendation": "ok",
            }
        )

    with patch("tools.environment.check_environment") as env, patch(
        "tools.reinvent.run_reinvent"
    ) as run:
        _agent(complete_fn=complete).review(PASSING_RESULTS)
    env.assert_not_called()
    run.assert_not_called()


def test_llm_critic_invalid_status_falls_back(caplog):
    def complete(_messages):
        return json.dumps(
            {"status": "SUPERB", "issues": [], "recommendation": "ship it"}
        )

    with caplog.at_level("WARNING"):
        verdict = _agent(complete_fn=complete).review(PASSING_RESULTS)
    assert verdict["critic"] == "deterministic"
    assert verdict["critic_fallback"] is True
    assert verdict["status"] == "PASS"
    assert any("falling back" in rec.message.lower() for rec in caplog.records)
    assert verdict["critic_warnings"]


def test_llm_critic_docking_scores_allowed_when_table_present():
    results = {
        "plan": {"approve_run": False, "skip_reinvent": True, "steps": ["docking"]},
        "steps": {
            "docking": {
                "ok": True,
                "approved": True,
                "skipped": False,
                "engine": "vina",
                "table_present": True,
                "scores_csv": "output/docking/scores.csv",
                "n_scored": 2,
                "score": {"best": -7.2, "mean": -5.1, "n": 2},
            }
        },
        "warnings": [],
        "errors": [],
    }

    def complete(_messages):
        return json.dumps(
            {
                "status": "PASS",
                "issues": ["Best vina docking score is -7.2 (n=2)."],
                "recommendation": "Human review of the docking table is recommended.",
            }
        )

    verdict = _agent(complete_fn=complete).review(results)
    assert verdict["critic"] == "llm"
    assert verdict["critic_fallback"] is False
    assert verdict["status"] == "PASS"


def test_llm_critic_md_metrics_allowed_when_table_present():
    results = {
        "plan": {"approve_run": False, "skip_reinvent": True, "steps": ["md"]},
        "steps": {
            "md": {
                "ok": True,
                "approved": True,
                "skipped": False,
                "engine": "gmx",
                "protocol": "em-nvt",
                "table_present": True,
                "rmsd_csv": "output/md/rmsd.csv",
                "rmsf_csv": "output/md/rmsf.csv",
                "n_frames": 3,
                "rmsd": {"mean": 0.012, "max": 0.02, "last": 0.02, "n": 3},
                "rmsf": {"mean": 0.012, "max": 0.015, "n": 3},
            }
        },
        "warnings": [],
        "errors": [],
    }

    def complete(_messages):
        return json.dumps(
            {
                "status": "PASS",
                "issues": ["GROMACS RMSF mean is 0.012 nm (n=3)."],
                "recommendation": "Human review of the MD table is recommended.",
            }
        )

    verdict = _agent(complete_fn=complete).review(results)
    assert verdict["critic"] == "llm"
    assert verdict["critic_fallback"] is False
    assert verdict["status"] == "PASS"


def test_llm_critic_invented_md_claim_falls_back():
    def complete(_messages):
        return json.dumps(
            {
                "status": "PASS",
                "issues": ["GROMACS RMSD is stable over 100 ns."],
                "recommendation": "MD confirms the pose.",
            }
        )

    verdict = _agent(complete_fn=complete).review(PASSING_RESULTS)
    assert verdict["critic_fallback"] is True
    assert any("invented" in w or "GROMACS" in w or "out-of-scope" in w for w in verdict["critic_warnings"])


def _clear_llm_keys(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_llm_critic_missing_api_key_falls_back(monkeypatch):
    _clear_llm_keys(monkeypatch)
    verdict = _agent().review(PASSING_RESULTS)
    assert verdict["critic_fallback"] is True
    assert any("API key" in w or "Missing" in w for w in verdict["critic_warnings"])
    expected = CriticAgent().review(PASSING_RESULTS)
    assert verdict["status"] == expected["status"]
    assert verdict["issues"] == expected["issues"]


def test_llm_critic_provider_error_falls_back():
    def complete(_messages):
        raise RuntimeError("provider 503")

    verdict = _agent(complete_fn=complete).review(FAILING_RESULTS)
    assert verdict["critic_fallback"] is True
    assert verdict["status"] == "FAIL"
    assert any("503" in w for w in verdict["critic_warnings"])


def test_llm_critic_can_fail_hard_when_fallback_disabled():
    def complete(_messages):
        return json.dumps({"status": "NOPE", "issues": [], "recommendation": "x"})

    agent = _agent(complete_fn=complete, fallback_on_error=False)
    with pytest.raises(LLMCriticError, match="status must be one of"):
        agent.review(PASSING_RESULTS)


def test_deterministic_critic_path_unchanged():
    verdict = CriticAgent().review(PASSING_RESULTS)
    assert verdict["status"] == "PASS"
    assert set(verdict) >= {"status", "issues", "recommendation", "thresholds"}
    assert "critic_fallback" not in verdict


def test_cli_critic_flag_defaults_and_choices():
    parser = build_parser()
    assert parser.parse_args([]).critic is None
    assert parser.parse_args(["--critic", "deterministic"]).critic == "deterministic"
    assert parser.parse_args(["--critic", "llm"]).critic == "llm"
    with pytest.raises(SystemExit):
        parser.parse_args(["--critic", "langchain"])


def test_review_with_critic_default_is_deterministic():
    args = Namespace(critic=None)
    verdict = _review_with_critic(args, {}, PASSING_RESULTS)
    assert verdict["status"] == "PASS"
    assert "critic_fallback" not in verdict


def test_review_with_critic_llm_falls_back_without_key(monkeypatch):
    _clear_llm_keys(monkeypatch)
    args = Namespace(critic="llm", provider=None)
    verdict = _review_with_critic(
        args,
        {"critic": {"mode": "deterministic", "fallback_on_error": True}},
        PASSING_RESULTS,
    )
    assert verdict["critic_fallback"] is True
    assert verdict["status"] == CriticAgent().review(PASSING_RESULTS)["status"]


def test_llm_critic_xai_missing_key_mentions_xai_env(monkeypatch):
    _clear_llm_keys(monkeypatch)
    verdict = _agent(provider="xai").review(PASSING_RESULTS)
    assert verdict["critic_fallback"] is True
    assert any("XAI_API_KEY" in w for w in verdict["critic_warnings"])


def test_review_with_critic_provider_flag_uses_xai_env(monkeypatch):
    _clear_llm_keys(monkeypatch)
    args = Namespace(critic="llm", provider="xai")
    verdict = _review_with_critic(
        args,
        {"critic": {"mode": "deterministic", "fallback_on_error": True}},
        PASSING_RESULTS,
    )
    assert verdict["critic_fallback"] is True
    assert any("XAI_API_KEY" in w for w in verdict["critic_warnings"])
