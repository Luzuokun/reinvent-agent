"""Independent portfolio tool tests — fixture CSV only, no LLM / GPU / network."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from agents.critic import CriticAgent
from agents.critic_schema import (
    CriticValidationError,
    build_evidence_summary,
    validate_critic_verdict,
)
from agents.executor import ExecutionAgent
from agents.plan_schema import PlanValidationError, validate_plan
from tools.handbook_map import INDEPENDENT_TOOL_IDS, STABLE_TOOL_IDS
from tools.mcp_allowlist import McpAllowlistError, invoke_mcp_tool
from tools.portfolio.presets import (
    PortfolioPresetError,
    list_preset_ids,
    resolve_preset,
)
from tools.portfolio.rank import load_molecule_rows, rank_molecules
from tools.portfolio.run import main as portfolio_main, run_portfolio

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "portfolio_molecules.csv"
DEMO = REPO / "projects" / "demo_project"


def _project_with_csv(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "proj"
    (project / "input" / "portfolio").mkdir(parents=True)
    (project / "output").mkdir(parents=True)
    # Minimal project marker like other demos (toml optional for portfolio)
    (project / "reinvent.toml").write_text("# fixture project\n", encoding="utf-8")
    csv_path = project / "input" / "portfolio" / "molecules.csv"
    shutil.copy2(FIXTURE, csv_path)
    return project, csv_path


def test_preset_allowlist_rejects_unknown_and_freeform_weights():
    assert "balanced" in list_preset_ids()
    with pytest.raises(PortfolioPresetError, match="Unknown portfolio preset_id"):
        resolve_preset('{"qed": 99}')
    with pytest.raises(PortfolioPresetError, match="empty"):
        resolve_preset("   ")
    spec = resolve_preset("dock-first")
    assert spec["weights"]["dock"] < 0
    assert "dock" in spec["require_fields"]


def test_rank_uses_present_fields_only_and_drops_pains(tmp_path: Path):
    project, csv_path = _project_with_csv(tmp_path)
    rows, columns = load_molecule_rows(csv_path)
    assert "qed" in columns and "dock" in columns and "md" in columns
    preset = resolve_preset("balanced")
    ranked = rank_molecules(rows, preset)
    smiles = [r["SMILES"] for r in ranked]
    assert "O=C1C=CC(=O)C=C1" not in smiles  # PAINS hit dropped
    assert all(r["composite_score"] is not None for r in ranked)
    # Ibuprofen has high QED + strong dock → should rank near top
    assert ranked[0]["SMILES"].startswith("CC(C)Cc1ccc")
    # fields_used never invents missing modalities
    for row in ranked:
        used = set(row["fields_used"].split(",")) if row["fields_used"] else set()
        if "qed" in used:
            assert row["QED"] is not None
        if "dock" in used:
            assert row["dock_score"] is not None


def test_missing_modality_not_invented(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    csv_path = project / "mols.csv"
    csv_path.write_text(
        "SMILES,QED,SA,PAINS\nCCO,0.5,1.0,clean\nc1ccccc1,0.4,1.1,clean\n",
        encoding="utf-8",
    )
    rows, columns = load_molecule_rows(csv_path)
    assert "dock" not in columns
    ranked = rank_molecules(rows, resolve_preset("ligand-prop"))
    assert len(ranked) == 2
    assert all(r["dock_score"] is None or r["dock_score"] == "" for r in ranked)
    assert all("dock" not in (r["fields_used"] or "") for r in ranked)

    with pytest.raises(Exception, match="requires a 'dock' column|require"):
        from tools.portfolio.rank import assert_required_columns

        assert_required_columns(columns, resolve_preset("dock-first"))


def test_dry_run_skips_shortlist_write(tmp_path: Path):
    project, csv_path = _project_with_csv(tmp_path)
    result = run_portfolio(
        project,
        molecules_csv=csv_path,
        preset_id="balanced",
        approve=False,
    )
    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["approved"] is False
    assert result["shortlist_csv"] is None
    assert result["one_pager_html"] is None
    assert result["invents_scores"] is False
    assert result["llm_scorer"] is False
    out = project / "output" / "portfolio"
    assert (out / "portfolio_result.json").is_file()
    assert not (out / "shortlist.csv").exists()
    assert not (out / "campaign_one_pager.html").exists()
    assert "approve-portfolio" in result["message"]


def test_approve_writes_shortlist_one_pager_and_index(tmp_path: Path):
    project, csv_path = _project_with_csv(tmp_path)
    result = run_portfolio(
        project,
        molecules_csv=csv_path,
        preset_id="balanced",
        approve=True,
        assume_yes=True,
        campaign={
            "target": "Demo target",
            "protocols": "fixture protocol",
            "red_flags": "fixture only",
            "next_steps": "PI review",
            "campaign_id": "demo-fixture",
        },
    )
    assert result["success"] is True
    assert result["table_present"] is True
    shortlist = Path(result["shortlist_csv"])
    assert shortlist.is_file()
    assert shortlist.name == "shortlist.csv"
    assert "output/portfolio" in shortlist.as_posix()
    with shortlist.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert "composite_score" in rows[0]
    assert all(r.get("PAINS") != "hit" for r in rows)

    one_pager = Path(result["one_pager_html"])
    html = one_pager.read_text(encoding="utf-8")
    assert "靶点" in html and "批准协议" in html and "红旗" in html
    assert "Top 分子" in html and "下一步" in html
    assert "Demo target" in html
    assert "PI-approved" in html

    index_path = Path(result["run_index"])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["crm"] is False
    assert index["runs"][-1]["preset_id"] == "balanced"
    assert index["runs"][-1]["approved"] is True


def test_csv_outside_project_rejected(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("SMILES,QED\nCCO,0.5\n", encoding="utf-8")
    with pytest.raises(Exception, match="under the project directory"):
        run_portfolio(
            project,
            molecules_csv=outside,
            preset_id="ligand-prop",
            approve=False,
        )


def test_executor_skips_portfolio_step_names():
    plan = {
        "goal": "sneak portfolio",
        "project_dir": str(DEMO),
        "approve_run": False,
        "skip_reinvent": True,
        "steps": ["check_environment", "portfolio", "shortlist", "run_portfolio"],
    }
    results = ExecutionAgent().execute(plan, project_dir=DEMO, approve_run=False)
    warnings = " ".join(results.get("warnings") or [])
    assert "independent module" in warnings
    assert "python -m tools.portfolio" in warnings
    assert "approve-portfolio" in warnings
    assert "portfolio" not in results.get("steps", {})
    assert "shortlist" not in results.get("steps", {})


def test_plan_schema_rejects_portfolio_step():
    with pytest.raises(PlanValidationError, match="Unknown plan step"):
        validate_plan(
            {"steps": ["check_environment", "portfolio"]},
            project_dir=DEMO,
            output_dir=DEMO / "output",
            source="llm",
        )


def test_mcp_rejects_portfolio_tools():
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("portfolio", {"preset_id": "balanced"})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("run_portfolio", {"project_dir": str(DEMO)})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("shortlist", {"preset_id": "balanced"})


def test_handbook_lists_portfolio_as_independent():
    assert "portfolio" in STABLE_TOOL_IDS
    assert "portfolio" in INDEPENDENT_TOOL_IDS


def test_critic_rejects_shortlist_claim_without_table():
    with pytest.raises(CriticValidationError, match="invented out-of-scope"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["The portfolio shortlist is ready for synthesis."],
                "recommendation": "Buy compounds.",
            },
            evidence={"analysis": {"total_molecules": 10}},
            source="llm",
        )


def test_critic_allows_shortlist_when_table_present():
    evidence = build_evidence_summary(
        {
            "plan": {"skip_reinvent": True, "steps": ["portfolio"]},
            "steps": {
                "portfolio": {
                    "ok": True,
                    "approved": True,
                    "skipped": False,
                    "success": True,
                    "table_present": True,
                    "n_shortlisted": 3,
                    "shortlist_csv": "/tmp/shortlist.csv",
                    "preset_id": "balanced",
                    "message": "ok",
                    "invents_scores": False,
                }
            },
        }
    )
    assert evidence["portfolio"]["table_present"] is True
    verdict = validate_critic_verdict(
        {
            "status": "PASS",
            "issues": ["Portfolio shortlist table present for balanced preset."],
            "recommendation": "Human-check shortlist.csv before synthesis talk.",
        },
        evidence=evidence,
        source="llm",
    )
    assert verdict["status"] == "PASS"


def test_deterministic_critic_warns_when_portfolio_skipped():
    verdict = CriticAgent().review(
        {
            "plan": {"approve_run": False, "skip_reinvent": True, "steps": ["portfolio"]},
            "steps": {
                "portfolio": {
                    "ok": True,
                    "approved": False,
                    "skipped": True,
                    "table_present": False,
                    "n_shortlisted": 0,
                    "message": "skipped",
                }
            },
            "warnings": [],
            "errors": [],
        }
    )
    assert verdict["status"] == "WARNING"
    assert any("approve-portfolio" in i for i in verdict["issues"])


def test_cli_list_presets_and_dry_run(tmp_path: Path, capsys):
    assert portfolio_main(["--list-presets"]) == 0
    out = capsys.readouterr().out
    assert "balanced" in out

    project, csv_path = _project_with_csv(tmp_path)
    code = portfolio_main(
        [
            "--project",
            str(project),
            "--molecules-csv",
            str(csv_path),
            "--preset-id",
            "balanced",
        ]
    )
    assert code == 0
    captured = capsys.readouterr().out
    assert "WARNING" in captured or "approve-portfolio" in captured
