"""Critic schema tests — no network."""

from __future__ import annotations

import pytest

from agents.critic_schema import (
    CriticValidationError,
    build_evidence_summary,
    parse_json_object,
    validate_critic_verdict,
)


def test_validate_accepts_minimal_pass():
    result = validate_critic_verdict(
        {
            "status": "PASS",
            "issues": [],
            "recommendation": "Evidence looks consistent.",
        },
        evidence={},
        source="llm",
    )
    assert result["status"] == "PASS"
    assert result["issues"] == []


def test_validate_rejects_unknown_status():
    with pytest.raises(CriticValidationError, match="status must be one of"):
        validate_critic_verdict(
            {
                "status": "MAYBE",
                "issues": [],
                "recommendation": "n/a",
            },
            source="llm",
        )


def test_validate_rejects_unknown_keys():
    with pytest.raises(CriticValidationError, match="Unknown critic keys"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": [],
                "recommendation": "ok",
                "argv": ["rm", "-rf"],
            },
            source="llm",
        )


def test_validate_rejects_llm_setting_thresholds():
    with pytest.raises(CriticValidationError, match="Unknown critic keys"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": [],
                "recommendation": "ok",
                "thresholds": {"min_molecules": 0},
            },
            source="llm",
        )


def test_validate_rejects_invented_docking_claim():
    with pytest.raises(CriticValidationError, match="invented out-of-scope"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["Docking scores look excellent against EGFR."],
                "recommendation": "Proceed to wet lab.",
            },
            evidence={"analysis": {"total_molecules": 10}},
            source="llm",
        )


def test_validate_rejects_shellish_recommendation():
    with pytest.raises(CriticValidationError, match="shell/command"):
        validate_critic_verdict(
            {
                "status": "FAIL",
                "issues": [],
                "recommendation": "Run: sudo rm -rf /tmp/output",
            },
            source="llm",
        )


def test_validate_rejects_empty_recommendation():
    with pytest.raises(CriticValidationError, match="recommendation"):
        validate_critic_verdict(
            {"status": "PASS", "issues": [], "recommendation": "  "},
            source="llm",
        )


def test_parse_json_object_strips_fences():
    data = parse_json_object('```json\n{"status": "FAIL", "issues": [], "recommendation": "x"}\n```')
    assert data["status"] == "FAIL"


def test_build_evidence_summary_omits_smiles_and_html():
    results = {
        "plan": {"approve_run": False, "skip_reinvent": True, "steps": ["analyze_molecules"]},
        "steps": {
            "analyze_molecules": {
                "ok": True,
                "csv_path": "/tmp/sampled.csv",
                "total_molecules": 2,
                "smiles": ["CCO", "c1ccccc1"],
                "html": "<html>secret</html>",
                "rdkit": {"available": True, "valid_fraction": 1.0, "qed": {"mean": 0.5, "n": 2}},
            }
        },
        "errors": [],
        "warnings": [],
    }
    evidence = build_evidence_summary(results, thresholds={"min_molecules": 1})
    blob = str(evidence)
    assert "CCO" not in blob
    assert "c1ccccc1" not in blob
    assert "<html>" not in blob
    assert evidence["analysis"]["total_molecules"] == 2
    assert evidence["analysis"]["rdkit"]["qed"]["mean"] == 0.5
    assert evidence["thresholds"]["min_molecules"] == 1
