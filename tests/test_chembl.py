"""Independent ChEMBL tool tests — mocked REST only, no network.

The fetch function is injected. Tests never open a socket and never invent
assay rows when the mock raises.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.critic import CriticAgent
from agents.critic_schema import (
    CriticValidationError,
    build_evidence_summary,
    validate_critic_verdict,
)
from agents.executor import ExecutionAgent
from agents.plan_schema import PlanValidationError, validate_plan
from tools.chembl.client import (
    ChemblNetworkError,
    assert_chembl_url,
    fetch_activities,
    normalize_activity_row,
    parse_activity_payload,
)
from tools.chembl.queries import ChemblQueryError, list_query_ids, resolve_query
from tools.chembl.run import main as chembl_main, run_chembl
from tools.environment import check_environment
from tools.handbook_map import INDEPENDENT_TOOL_IDS, STABLE_TOOL_IDS
from tools.mcp_allowlist import McpAllowlistError, invoke_mcp_tool

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "projects" / "demo_project"


def _activity_bytes(
    *,
    mol_id: str = "CHEMBL25",
    smiles: str = "CC(=O)Oc1ccccc1C(=O)O",
    target: str = "CHEMBL203",
    standard_type: str = "IC50",
    standard_value: float = 12.0,
    include_row: bool = True,
) -> bytes:
    activities = []
    if include_row:
        activities.append(
            {
                "activity_id": 31863,
                "molecule_chembl_id": mol_id,
                "canonical_smiles": smiles,
                "target_chembl_id": target,
                "assay_chembl_id": "CHEMBL617995",
                "standard_type": standard_type,
                "standard_value": standard_value,
                "standard_units": "nM",
                "pchembl_value": 7.9,
            }
        )
    return json.dumps({"activities": activities, "page_meta": {"total_count": len(activities)}}).encode(
        "utf-8"
    )


def _fake_get_ok(url: str) -> bytes:
    assert_chembl_url(url)
    assert "activity" in url
    return _activity_bytes()


def test_reinvent_env_check_does_not_probe_chembl():
    result = check_environment()
    blob = json.dumps(result)
    assert "chembl" not in blob.lower()
    assert "python" in result


def test_query_allowlist_rejects_unknown_and_sqlish():
    assert "egfr-ic50" in list_query_ids()
    with pytest.raises(ChemblQueryError, match="Unknown ChEMBL query_id"):
        resolve_query("SELECT * FROM activities")
    with pytest.raises(ChemblQueryError, match="empty"):
        resolve_query("   ")
    spec = resolve_query("egfr-ic50")
    assert spec["target_chembl_id"] == "CHEMBL203"
    assert spec["standard_type"] == "IC50"


def test_assert_chembl_url_rejects_other_hosts():
    with pytest.raises(ChemblNetworkError, match="NETWORK FAILURE"):
        assert_chembl_url("https://example.com/chembl/api/data/activity")
    with pytest.raises(ChemblNetworkError, match="NETWORK FAILURE"):
        assert_chembl_url("http://www.ebi.ac.uk/chembl/api/data/activity")
    with pytest.raises(ChemblNetworkError, match="NETWORK FAILURE"):
        assert_chembl_url("https://www.ebi.ac.uk/chembl/api/other/activity")


def test_normalize_drops_rows_without_molecule_chembl_id():
    assert normalize_activity_row({"canonical_smiles": "CCO"}, query_id="egfr-ic50") is None
    row = normalize_activity_row(
        {
            "molecule_chembl_id": "CHEMBL25",
            "canonical_smiles": "CCO",
            "target_chembl_id": "CHEMBL203",
            "standard_type": "IC50",
            "standard_value": 1.0,
            "standard_units": "nM",
            "assay_chembl_id": "CHEMBL1",
            "activity_id": 9,
        },
        query_id="egfr-ic50",
    )
    assert row is not None
    assert row["molecule_chembl_id"] == "CHEMBL25"
    assert row["query_id"] == "egfr-ic50"
    assert row["origin"] == "chembl"


def test_parse_activity_payload_empty_list_is_not_invented():
    rows = parse_activity_payload({"activities": []}, query_id="egfr-ic50")
    assert rows == []


def test_fetch_activities_uses_injected_get_fn():
    seen: list[str] = []

    def get_fn(url: str) -> bytes:
        seen.append(url)
        return _fake_get_ok(url)

    rows = fetch_activities(resolve_query("egfr-ic50"), limit=5, get_fn=get_fn)
    assert len(rows) == 1
    assert rows[0]["molecule_chembl_id"] == "CHEMBL25"
    assert rows[0]["canonical_smiles"]
    assert all("www.ebi.ac.uk" in url for url in seen)
    assert all("target_chembl_id=CHEMBL203" in url for url in seen)


def test_run_chembl_requires_approval(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    with patch("tools.chembl.client.http_get") as mocked:
        result = run_chembl(proj, query_id="egfr-ic50", approve=False)
    mocked.assert_not_called()
    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["table_present"] is False
    assert result["n_activities"] == 0
    assert "approve-chembl" in result["message"]
    assert (proj / "output" / "chembl" / "chembl_result.json").is_file()


def test_run_chembl_mocked_writes_assay_table(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    result = run_chembl(
        proj,
        query_id="egfr-ic50",
        approve=True,
        assume_yes=True,
        get_fn=_fake_get_ok,
    )
    assert result["ok"] is True
    assert result["success"] is True
    assert result["table_present"] is True
    assert result["n_activities"] == 1
    assert result["n_with_smiles"] == 1
    activities_csv = Path(result["activities_csv"])
    assert activities_csv.is_file()
    text = activities_csv.read_text(encoding="utf-8")
    assert "molecule_chembl_id" in text
    assert "CHEMBL25" in text
    assert Path(result["smiles_csv"]).is_file()


def test_run_chembl_network_failure_is_loud_not_invented(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()

    def boom(_url: str) -> bytes:
        raise ChemblNetworkError("NETWORK FAILURE: cannot reach ChEMBL (timeout)")

    result = run_chembl(
        proj,
        query_id="egfr-ic50",
        approve=True,
        assume_yes=True,
        get_fn=boom,
    )
    assert result["ok"] is False
    assert result["table_present"] is False
    assert result["n_activities"] == 0
    assert any("NETWORK FAILURE" in e for e in result["errors"])
    assert not (proj / "output" / "chembl" / "activities.csv").exists()


def test_run_chembl_empty_hits_are_not_fabricated(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()

    def empty(url: str) -> bytes:
        assert_chembl_url(url)
        return _activity_bytes(include_row=False)

    result = run_chembl(
        proj,
        query_id="braf-ic50",
        approve=True,
        assume_yes=True,
        get_fn=empty,
    )
    assert result["ok"] is True
    assert result["success"] is True
    assert result["table_present"] is False
    assert result["n_activities"] == 0
    assert "No assay values were invented" in result["message"]


def test_cli_list_queries_and_dry_run(tmp_path: Path):
    code = chembl_main(["--list-queries"])
    assert code == 0
    proj = tmp_path / "proj"
    proj.mkdir()
    with patch("tools.chembl.client.http_get") as mocked:
        code = chembl_main(
            ["--project", str(proj), "--query-id", "egfr-ic50"]
        )
    mocked.assert_not_called()
    assert code == 0


def test_executor_does_not_run_chembl():
    plan = {
        "goal": "sneak chembl",
        "project_dir": str(DEMO),
        "approve_run": False,
        "skip_reinvent": True,
        "steps": ["chembl", "download_chembl", "assay"],
        "notes": [],
    }
    with patch("subprocess.run") as mocked:
        results = ExecutionAgent().execute(
            plan, project_dir=DEMO, approve_run=False
        )
    mocked.assert_not_called()
    warnings = " ".join(results.get("warnings") or [])
    assert "independent module" in warnings
    assert "python -m tools.chembl" in warnings
    assert "chembl" not in results.get("steps", {})
    assert "assay" not in results.get("steps", {})


def test_plan_schema_rejects_chembl_step():
    with pytest.raises(PlanValidationError, match="Unknown plan step"):
        validate_plan(
            {"steps": ["check_environment", "chembl"]},
            project_dir=DEMO,
            output_dir=DEMO / "output",
            source="llm",
        )
    with pytest.raises(PlanValidationError, match="Unknown plan step"):
        validate_plan(
            {"steps": ["download_chembl"]},
            project_dir=DEMO,
            output_dir=DEMO / "output",
            source="llm",
        )


def test_mcp_rejects_chembl_tools():
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("chembl", {"query_id": "egfr-ic50"})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("run_chembl", {"project_dir": str(DEMO)})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("download_chembl", {"query_id": "egfr-ic50"})


def test_handbook_lists_chembl_as_independent():
    assert "chembl" in STABLE_TOOL_IDS
    assert "chembl" in INDEPENDENT_TOOL_IDS


def test_critic_rejects_invented_ic50_without_chembl_table():
    with pytest.raises(CriticValidationError, match="invented out-of-scope"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["ChEMBL IC50 values show these are potent EGFR inhibitors."],
                "recommendation": "Proceed to wet lab.",
            },
            evidence={"analysis": {"total_molecules": 10}},
            source="llm",
        )


def test_critic_rejects_chembl_claim_if_only_plan_names_the_step():
    evidence = build_evidence_summary(
        {
            "plan": {"steps": ["chembl", "download_chembl"]},
            "steps": {},
        }
    )
    assert "chembl" not in evidence or evidence.get("chembl") in (None, {})
    with pytest.raises(CriticValidationError, match="invented out-of-scope"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["ChEMBL download supports this scaffold IC50 claim."],
                "recommendation": "Cite the assay table.",
            },
            evidence=evidence,
            source="llm",
        )


def test_critic_allows_ic50_when_chembl_table_present():
    evidence = build_evidence_summary(
        {
            "plan": {"skip_reinvent": True, "steps": ["chembl"]},
            "steps": {
                "chembl": {
                    "ok": True,
                    "approved": True,
                    "skipped": False,
                    "success": True,
                    "table_present": True,
                    "n_activities": 2,
                    "n_with_smiles": 2,
                    "activities_csv": "/tmp/activities.csv",
                    "query_id": "egfr-ic50",
                    "query": {
                        "id": "egfr-ic50",
                        "target_chembl_id": "CHEMBL203",
                        "standard_type": "IC50",
                    },
                    "message": "ok",
                }
            },
        }
    )
    assert evidence["chembl"]["table_present"] is True
    verdict = validate_critic_verdict(
        {
            "status": "PASS",
            "issues": ["ChEMBL activity table present for egfr-ic50 IC50 rows."],
            "recommendation": "Human-check the assay CSV before TL.",
        },
        evidence=evidence,
        source="llm",
    )
    assert verdict["status"] == "PASS"


def test_deterministic_critic_warns_when_chembl_skipped():
    verdict = CriticAgent().review(
        {
            "plan": {"approve_run": False, "skip_reinvent": True, "steps": ["chembl"]},
            "steps": {
                "chembl": {
                    "ok": True,
                    "approved": False,
                    "skipped": True,
                    "table_present": False,
                    "n_activities": 0,
                    "message": "skipped",
                }
            },
            "warnings": [],
            "errors": [],
        }
    )
    assert verdict["status"] == "WARNING"
    assert any("approve-chembl" in i for i in verdict["issues"])


def test_deterministic_critic_fails_on_chembl_network_error():
    verdict = CriticAgent().review(
        {
            "plan": {"approve_run": False, "skip_reinvent": True, "steps": ["chembl"]},
            "steps": {
                "chembl": {
                    "ok": False,
                    "approved": True,
                    "skipped": False,
                    "table_present": False,
                    "n_activities": 0,
                    "errors": ["NETWORK FAILURE: cannot reach ChEMBL"],
                    "message": "NETWORK FAILURE: cannot reach ChEMBL",
                }
            },
            "warnings": [],
            "errors": [],
        }
    )
    assert verdict["status"] == "FAIL"
    assert any("NETWORK FAILURE" in i for i in verdict["issues"])
    assert any("invented" in i.lower() or "Refusing" in i for i in verdict["issues"])


def test_build_evidence_omits_chembl_key_without_module():
    evidence = build_evidence_summary(
        {
            "plan": {"approve_run": False, "skip_reinvent": True},
            "steps": {
                "analyze_molecules": {
                    "ok": True,
                    "total_molecules": 3,
                    "rdkit": {"available": True},
                }
            },
        }
    )
    assert "chembl" not in evidence
