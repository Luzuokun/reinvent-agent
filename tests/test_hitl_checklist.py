"""HITL three-gate checklist: config → run → claim (no GPU / network)."""

from __future__ import annotations

from pathlib import Path

from agents.critic import CriticAgent
from agents.executor import ExecutionAgent
from agents.planner import PlannerAgent
from analysis.report import generate_html_report
from tools.from_run import NEVER_COPY_FIELDS, explicit_cli_flags
from tools.hitl_checklist import (
    attach_hitl_to_plan,
    claim_banner_text,
    evaluate_hitl_gates,
)

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "projects" / "demo_project"


def test_evaluate_gates_default_all_pending():
    checklist = evaluate_hitl_gates({})
    assert checklist["schema"] == "reinvent-hitl-gates-v1"
    assert checklist["approved_count"] == 0
    assert checklist["missing"] == ["config", "run", "claim"]
    assert checklist["claims_human_approved"] is False
    assert checklist["soft_policy"] is True
    assert claim_banner_text(checklist)


def test_evaluate_gates_all_approved():
    checklist = evaluate_hitl_gates(
        {
            "approve_config": True,
            "approve_run": True,
            "approve_claim": True,
        }
    )
    assert checklist["all_approved"] is True
    assert checklist["missing"] == []
    assert claim_banner_text(checklist) is None


def test_warn_run_without_config_soft_policy():
    checklist = evaluate_hitl_gates(
        approve_config=False,
        approve_run=True,
        approve_claim=False,
        skip_reinvent=False,
    )
    assert checklist["warn_run_without_config"] is True
    assert checklist["hard_block_run_without_config"] is False


def test_planner_defaults_do_not_invent_hitl_approvals():
    plan = PlannerAgent().create_plan(
        "demo",
        project_dir=str(DEMO),
        approve_run=False,
        skip_reinvent=True,
    )
    assert plan["approve_config"] is False
    assert plan["approve_claim"] is False
    assert plan["approve_run"] is False
    assert any("HITL gates" in n for n in plan["notes"])


def test_attach_hitl_to_plan_mutates():
    plan = {
        "approve_config": True,
        "approve_run": False,
        "approve_claim": False,
        "skip_reinvent": True,
    }
    attach_hitl_to_plan(plan)
    assert plan["hitl"]["approved_count"] == 1
    assert plan["hitl"]["missing"] == ["run", "claim"]


def test_critic_warns_on_missing_claim_and_config_after_approved_run():
    results = {
        "plan": {
            "approve_run": True,
            "approve_config": False,
            "approve_claim": False,
            "skip_reinvent": False,
            "steps": ["run_reinvent", "analyze_molecules"],
        },
        "steps": {
            "run_reinvent": {"success": True, "skipped": False},
            "analyze_molecules": {
                "ok": True,
                "total_molecules": 2,
                "duplicate_fraction": 0.0,
                "rdkit": {
                    "available": True,
                    "valid_fraction": 1.0,
                    "qed": {"mean": 0.7, "n": 2},
                },
            },
            "find_output": {"ok": True, "csv_count": 1},
            "validate_project": {"ok": True},
        },
        "warnings": [],
        "errors": [],
    }
    verdict = CriticAgent().review(results)
    assert verdict["status"] == "WARNING"
    joined = " ".join(verdict["issues"])
    assert "--approve-config" in joined
    assert "--approve-claim" in joined


def test_critic_offline_pass_without_claim_still_ok():
    """Offline dry-run: claim gate is checklist/banner-only unless a run launched."""
    results = {
        "plan": {
            "approve_run": False,
            "approve_config": False,
            "approve_claim": False,
            "skip_reinvent": True,
            "steps": ["analyze_molecules"],
        },
        "steps": {
            "analyze_molecules": {
                "ok": True,
                "total_molecules": 10,
                "duplicate_fraction": 0.0,
                "rdkit": {"available": True, "valid_fraction": 1.0, "qed": {"mean": 0.6}},
            },
            "find_output": {"ok": True, "csv_count": 1},
            "validate_project": {"ok": True},
        },
        "warnings": [],
        "errors": [],
    }
    verdict = CriticAgent().review(results)
    assert verdict["status"] == "PASS"
    assert not any("approve-claim" in i for i in verdict["issues"])


def test_html_report_includes_hitl_section(tmp_path: Path):
    checklist = evaluate_hitl_gates(
        approve_config=True,
        approve_run=False,
        approve_claim=False,
        skip_reinvent=True,
    )
    meta = generate_html_report(
        {
            "goal": "HITL stub",
            "project": {
                "project_dir": str(DEMO),
                "approve_config": True,
                "approve_run": False,
                "approve_claim": False,
            },
            "hitl": checklist,
            "environment": {},
            "validation": {"ok": True},
            "execution": {"skipped": True},
            "inventory": {},
            "analysis": {"ok": True, "analysis_source": "existing_csv"},
            "critic": {"status": "WARNING", "issues": ["claims"]},
            "warnings": [],
            "errors": [],
            "dry_run": True,
            "run_id": "20261029_000000",
        },
        reports_dir=tmp_path,
        filename="hitl_stub.html",
    )
    html = Path(meta["report_path"]).read_text(encoding="utf-8")
    assert "HITL checklist" in html
    assert "--approve-config" in html
    assert "--approve-claim" in html
    assert "Claims unapproved" in html
    assert "gate approved" in html
    assert "gate pending" in html


def test_executor_write_report_embeds_hitl(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(REPO)
    results = {
        "plan": {
            "steps": ["analyze_molecules", "generate_report"],
            "approve_run": False,
            "approve_config": True,
            "approve_claim": False,
            "skip_reinvent": True,
        },
        "project_dir": str(DEMO),
        "reinvent_config_path": str(DEMO / "reinvent.toml"),
        "steps": {
            "check_environment": {"reinvent": False},
            "validate_project": {"ok": True},
            "run_reinvent": {"skipped": True},
            "analyze_molecules": {
                "ok": True,
                "analysis_source": "existing_csv",
                "csv_path": str(DEMO / "output" / "sampled-sample.csv"),
            },
            "find_output": {"ok": True},
        },
        "warnings": [],
        "errors": [],
        "run_id": "20261029_010101",
    }
    import agents.executor as executor_mod

    real_generate = executor_mod.generate_html_report

    def _generate(payload, **kwargs):
        kwargs.setdefault("reports_dir", tmp_path)
        return real_generate(payload, **kwargs)

    monkeypatch.setattr(executor_mod, "generate_html_report", _generate)
    meta = ExecutionAgent().write_report(
        results,
        goal="offline",
        critic={"status": "WARNING", "issues": ["claim"]},
    )
    html = Path(meta["report_path"]).read_text(encoding="utf-8")
    assert "HITL checklist" in html
    assert results["plan"]["approve_config"] is True


def test_from_run_never_copies_hitl_flags():
    assert "approve_config" in NEVER_COPY_FIELDS
    assert "approve_run" in NEVER_COPY_FIELDS
    assert "approve_claim" in NEVER_COPY_FIELDS
    assert "hitl" in NEVER_COPY_FIELDS
    found = explicit_cli_flags(
        [
            "--from-run",
            "last",
            "--approve-config",
            "--approve-run",
            "--approve-claim",
            "--yes",
        ]
    )
    assert found == {
        "from_run",
        "approve_config",
        "approve_run",
        "approve_claim",
        "yes",
    }
