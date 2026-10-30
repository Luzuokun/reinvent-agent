"""Smoke: dry-run / --skip-reinvent claim gate is a soft Critic WARNING.

No GPU, no network, no API keys. Missing ``--approve-claim`` downgrades PASS
to WARNING and does not FAIL. Present ``--approve-claim`` clears that issue.
"""

from __future__ import annotations

from agents.critic import CLAIM_UNAPPROVED_ISSUE, CriticAgent
from main import _exit_code_for_critic


def _analysis() -> dict:
    return {
        "ok": True,
        "total_molecules": 10,
        "duplicate_fraction": 0.0,
        "analysis_source": "existing_csv",
        "rdkit": {"available": True, "valid_fraction": 1.0, "qed": {"mean": 0.6, "n": 10}},
    }


def _results(
    *,
    approve_run: bool,
    skip_reinvent: bool,
    approve_claim: bool,
    approve_config: bool = False,
    include_analysis: bool = True,
) -> dict:
    steps: dict = {
        "validate_project": {"ok": True},
        "find_output": {"ok": True, "csv_count": 1},
        "run_reinvent": {
            "skipped": not (approve_run and not skip_reinvent),
            "success": bool(approve_run and not skip_reinvent),
        },
    }
    plan_steps = ["analyze_molecules"]
    if include_analysis:
        steps["analyze_molecules"] = _analysis()
    return {
        "plan": {
            "approve_run": approve_run,
            "approve_config": approve_config,
            "approve_claim": approve_claim,
            "skip_reinvent": skip_reinvent,
            "steps": plan_steps,
        },
        "steps": steps,
        "warnings": [],
        "errors": [],
    }


def _review(**kwargs) -> dict:
    return CriticAgent().review(_results(**kwargs))


def test_dry_run_missing_approve_claim_is_soft_warning():
    verdict = _review(approve_run=False, skip_reinvent=False, approve_claim=False)
    assert verdict["status"] == "WARNING"
    assert CLAIM_UNAPPROVED_ISSUE in verdict["issues"]
    assert any("Dry-run only" in item for item in verdict["issues"])
    assert not any("--approve-config" in item for item in verdict["issues"])
    assert _exit_code_for_critic(verdict["status"]) == 0


def test_dry_run_with_approve_claim_omits_claim_issue():
    verdict = _review(approve_run=False, skip_reinvent=False, approve_claim=True)
    assert verdict["status"] == "WARNING"
    assert CLAIM_UNAPPROVED_ISSUE not in verdict["issues"]
    assert any("Dry-run only" in item for item in verdict["issues"])
    assert _exit_code_for_critic(verdict["status"]) == 0


def test_skip_reinvent_missing_approve_claim_is_soft_warning():
    verdict = _review(approve_run=False, skip_reinvent=True, approve_claim=False)
    assert verdict["status"] == "WARNING"
    assert verdict["issues"] == [CLAIM_UNAPPROVED_ISSUE]
    assert _exit_code_for_critic(verdict["status"]) == 0


def test_skip_reinvent_with_approve_claim_can_pass():
    verdict = _review(approve_run=False, skip_reinvent=True, approve_claim=True)
    assert verdict["status"] == "PASS"
    assert CLAIM_UNAPPROVED_ISSUE not in verdict["issues"]
    assert _exit_code_for_critic(verdict["status"]) == 0


def test_skip_reinvent_and_approve_run_still_warns_without_claim():
    """Both flags: skip wins for launch, claim gate still warns."""
    verdict = _review(approve_run=True, skip_reinvent=True, approve_claim=False)
    assert verdict["status"] == "WARNING"
    assert CLAIM_UNAPPROVED_ISSUE in verdict["issues"]
    assert not any("supposed to run" in item for item in verdict["issues"])
    assert _exit_code_for_critic(verdict["status"]) == 0


def test_skip_reinvent_and_approve_run_with_claim_omits_claim_issue():
    verdict = _review(approve_run=True, skip_reinvent=True, approve_claim=True)
    assert verdict["status"] == "PASS"
    assert CLAIM_UNAPPROVED_ISSUE not in verdict["issues"]


def test_missing_claim_does_not_hide_a_hard_fail():
    results = _results(approve_run=False, skip_reinvent=True, approve_claim=False)
    results["steps"]["analyze_molecules"] = {
        "ok": False,
        "errors": ["Molecule analysis failed"],
        "total_molecules": 0,
    }
    verdict = CriticAgent().review(results)
    assert verdict["status"] == "FAIL"
    assert CLAIM_UNAPPROVED_ISSUE in verdict["issues"]
    assert _exit_code_for_critic(verdict["status"]) == 1


def test_approved_launch_without_claim_stays_soft_warning():
    """Non-dry launch path is unchanged: missing claim is WARNING, not FAIL."""
    verdict = _review(
        approve_run=True,
        skip_reinvent=False,
        approve_claim=False,
        approve_config=True,
    )
    assert verdict["status"] == "WARNING"
    assert CLAIM_UNAPPROVED_ISSUE in verdict["issues"]
    assert not any("--approve-config" in item for item in verdict["issues"])
    assert _exit_code_for_critic(verdict["status"]) == 0


def test_approved_launch_with_claim_and_config_can_pass():
    verdict = _review(
        approve_run=True,
        skip_reinvent=False,
        approve_claim=True,
        approve_config=True,
    )
    assert verdict["status"] == "PASS"
    assert CLAIM_UNAPPROVED_ISSUE not in verdict["issues"]


def test_docking_only_skip_does_not_take_the_claim_warning():
    """Independent modules are not a reinvent dry-run / skip-reinvent claim review."""
    verdict = CriticAgent().review(
        {
            "plan": {"approve_run": False, "skip_reinvent": True, "approve_claim": False},
            "steps": {
                "docking": {
                    "ok": True,
                    "approved": True,
                    "skipped": False,
                    "engine": "vina",
                    "table_present": True,
                    "n_scored": 1,
                    "score": {"best": -8.0, "mean": -8.0, "n": 1},
                }
            },
            "warnings": [],
            "errors": [],
        }
    )
    assert CLAIM_UNAPPROVED_ISSUE not in verdict["issues"]
    assert verdict["status"] == "PASS"
