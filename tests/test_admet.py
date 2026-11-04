"""Independent ADMET tool tests — fixture SMILES only, no LLM / GPU / network."""

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
from tools.admet.presets import (
    AdmetPresetError,
    list_preset_ids,
    resolve_preset,
)
from tools.admet.run import main as admet_main, run_admet
from tools.admet.score import load_molecule_rows, score_molecules
from tools.handbook_map import INDEPENDENT_TOOL_IDS, STABLE_TOOL_IDS
from tools.mcp_allowlist import McpAllowlistError, invoke_mcp_tool

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "admet_molecules.csv"
DEMO = REPO / "projects" / "demo_project"

pytest.importorskip("rdkit")


def _project_with_csv(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "proj"
    (project / "input" / "admet").mkdir(parents=True)
    (project / "output").mkdir(parents=True)
    (project / "reinvent.toml").write_text("# fixture project\n", encoding="utf-8")
    csv_path = project / "input" / "admet" / "molecules.csv"
    shutil.copy2(FIXTURE, csv_path)
    return project, csv_path


def test_preset_allowlist_rejects_unknown_and_freeform_rules():
    assert "lipinski" in list_preset_ids()
    assert "veber" in list_preset_ids()
    with pytest.raises(AdmetPresetError, match="Unknown ADMET preset_id"):
        resolve_preset('{"mw_max": 9999}')
    with pytest.raises(AdmetPresetError, match="empty"):
        resolve_preset("   ")
    spec = resolve_preset("leadlike")
    assert spec["rules"]["mw_max"] == 350.0


def test_score_applies_lipinski_and_flags_high_logp(tmp_path: Path):
    project, csv_path = _project_with_csv(tmp_path)
    rows, columns = load_molecule_rows(csv_path)
    assert "smiles" in columns
    scored = score_molecules(rows, resolve_preset("lipinski"))
    by_id = {r["ligand_id"]: r for r in scored}
    assert by_id["ethanol_001"]["preset_pass"] is True
    assert by_id["ibuprofen_006"]["preset_pass"] is True
    # Stearic acid fixture: high MW/logP → Lipinski fail
    assert by_id["stearic_007"]["preset_pass"] is False
    assert "logp>" in by_id["stearic_007"]["fail_reasons"] or "mw>" in by_id[
        "stearic_007"
    ]["fail_reasons"]
    # Never invent docking / MD / assay fields
    for row in scored:
        assert "dock_score" not in row
        assert "ic50" not in row
        assert "rmsd" not in row


def test_qed_floor_preset(tmp_path: Path):
    project, csv_path = _project_with_csv(tmp_path)
    rows, _ = load_molecule_rows(csv_path)
    scored = score_molecules(rows, resolve_preset("qed-floor"))
    assert all(r["valid"] for r in scored)
    assert any(r["preset_pass"] for r in scored)
    assert any(not r["preset_pass"] for r in scored) or all(
        float(r["qed"]) >= 0.5 for r in scored
    )


def test_dry_run_skips_scores_csv_write(tmp_path: Path):
    project, csv_path = _project_with_csv(tmp_path)
    result = run_admet(
        project,
        molecules_csv=csv_path,
        preset_id="lipinski",
        approve=False,
    )
    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["approved"] is False
    assert result["scores_csv"] is None
    assert result["invents_scores"] is False
    assert result["llm_scorer"] is False
    assert result["engine"] == "rdkit_rules"
    out = project / "output" / "admet"
    assert (out / "admet_result.json").is_file()
    assert not (out / "admet_scores.csv").exists()
    assert "approve-admet" in result["message"]
    preview = result["scores_preview"]
    assert isinstance(preview, list) and preview


def test_approve_writes_scores_csv(tmp_path: Path):
    project, csv_path = _project_with_csv(tmp_path)
    result = run_admet(
        project,
        molecules_csv=csv_path,
        preset_id="lipinski",
        approve=True,
        assume_yes=True,
    )
    assert result["success"] is True
    assert result["table_present"] is True
    scores = Path(result["scores_csv"])
    assert scores.is_file()
    assert scores.name == "admet_scores.csv"
    assert "output/admet" in scores.as_posix()
    with scores.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert "preset_pass" in rows[0]
    assert "mw" in rows[0] and "qed" in rows[0]
    assert "dock_score" not in rows[0]
    assert "ic50" not in rows[0]
    # Stearic fixture should be present with preset_pass=false
    stearic = [r for r in rows if r.get("ligand_id") == "stearic_007"]
    assert stearic and stearic[0]["preset_pass"] == "false"


def test_csv_outside_project_rejected(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("SMILES\nCCO\n", encoding="utf-8")
    with pytest.raises(Exception, match="under the project directory"):
        run_admet(
            project,
            molecules_csv=outside,
            preset_id="lipinski",
            approve=False,
        )


def test_executor_skips_admet_step_names():
    plan = {
        "goal": "sneak admet",
        "project_dir": str(DEMO),
        "approve_run": False,
        "skip_reinvent": True,
        "steps": ["check_environment", "admet", "run_admet", "admet_filter"],
    }
    results = ExecutionAgent().execute(plan, project_dir=DEMO, approve_run=False)
    warnings = " ".join(results.get("warnings") or [])
    assert "independent module" in warnings
    assert "python -m tools.admet" in warnings
    assert "approve-admet" in warnings
    assert "admet" not in results.get("steps", {})
    assert "run_admet" not in results.get("steps", {})


def test_plan_schema_rejects_admet_step():
    with pytest.raises(PlanValidationError, match="Unknown plan step"):
        validate_plan(
            {"steps": ["check_environment", "admet"]},
            project_dir=DEMO,
            output_dir=DEMO / "output",
            source="llm",
        )


def test_mcp_rejects_admet_tools():
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("admet", {"preset_id": "lipinski"})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("run_admet", {"project_dir": str(DEMO)})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("admet_filter", {"preset_id": "lipinski"})


def test_handbook_lists_admet_as_independent():
    assert "admet" in STABLE_TOOL_IDS
    assert "admet" in INDEPENDENT_TOOL_IDS


def test_critic_rejects_admet_claim_without_table():
    with pytest.raises(CriticValidationError, match="invented out-of-scope"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["The ADMET druglikeness screen proves oral bioavailability."],
                "recommendation": "Advance to clinic.",
            },
            evidence={"analysis": {"total_molecules": 10}},
            source="llm",
        )


def test_critic_allows_admet_when_table_present():
    evidence = build_evidence_summary(
        {
            "plan": {"skip_reinvent": True, "steps": ["admet"]},
            "steps": {
                "admet": {
                    "ok": True,
                    "approved": True,
                    "skipped": False,
                    "success": True,
                    "table_present": True,
                    "n_scored": 6,
                    "n_pass": 5,
                    "n_valid": 6,
                    "scores_csv": "/tmp/admet_scores.csv",
                    "preset_id": "lipinski",
                    "message": "ok",
                    "invents_scores": False,
                    "engine": "rdkit_rules",
                }
            },
        }
    )
    assert evidence["admet"]["table_present"] is True
    verdict = validate_critic_verdict(
        {
            "status": "PASS",
            "issues": ["ADMET scores table present for lipinski preset."],
            "recommendation": "Human-check admet_scores.csv before spending compute on structure work.",
        },
        evidence=evidence,
        source="llm",
    )
    assert verdict["status"] == "PASS"


def test_deterministic_critic_warns_when_admet_skipped():
    verdict = CriticAgent().review(
        {
            "plan": {"approve_run": False, "skip_reinvent": True, "steps": ["admet"]},
            "steps": {
                "admet": {
                    "ok": True,
                    "approved": False,
                    "skipped": True,
                    "table_present": False,
                    "n_scored": 0,
                    "message": "skipped",
                }
            },
            "warnings": [],
            "errors": [],
        }
    )
    assert verdict["status"] == "WARNING"
    assert any("approve-admet" in i for i in verdict["issues"])


def test_cli_list_presets_and_dry_run(tmp_path: Path, capsys):
    assert admet_main(["--list-presets"]) == 0
    out = capsys.readouterr().out
    assert "lipinski" in out
    catalog = json.loads(out)
    assert any(item["id"] == "veber" for item in catalog)

    project, csv_path = _project_with_csv(tmp_path)
    code = admet_main(
        [
            "--project",
            str(project),
            "--molecules-csv",
            str(csv_path),
            "--preset-id",
            "lipinski",
        ]
    )
    assert code == 0
    captured = capsys.readouterr().out
    assert "WARNING" in captured or "approve-admet" in captured


def test_missing_rdkit_is_loud_failure(tmp_path: Path, monkeypatch):
    from tools.admet.score import AdmetScoreError
    import tools.admet.score as score_mod

    project, csv_path = _project_with_csv(tmp_path)

    def _boom():
        raise AdmetScoreError(
            "RDKit is required for ADMET coarse screen and is not importable. "
            "Refusing to invent MW/logP/QED/rule outcomes."
        )

    monkeypatch.setattr(score_mod, "_import_rdkit", _boom)
    with pytest.raises(Exception, match="RDKit is required|Refusing to invent"):
        run_admet(
            project,
            molecules_csv=csv_path,
            preset_id="lipinski",
            approve=False,
        )
