"""Independent literature tool tests — mocked NCBI only, no network.

The fetch function is injected. Tests never open a socket and never invent
papers when the mock raises.
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
from tools.environment import check_environment
from tools.literature.entries import filter_sourced
from tools.literature.pubmed import (
    LiteratureNetworkError,
    assert_eutils_url,
    normalize_query,
    parse_esummary,
    search_pubmed,
)
from tools.literature.run import main as literature_main, run_literature
from tools.mcp_allowlist import McpAllowlistError, invoke_mcp_tool

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "projects" / "demo_project"

PMID = "12345678"
DOI = "10.1000/test.doi"
TITLE = "A sourced test paper"


def _esearch_bytes(ids: list[str]) -> bytes:
    return json.dumps({"esearchresult": {"idlist": ids}}).encode("utf-8")


def _esummary_bytes(
    pmid: str = PMID,
    *,
    title: str = TITLE,
    doi: str | None = DOI,
    include_record: bool = True,
) -> bytes:
    articleids = [{"idtype": "doi", "value": doi}] if doi else []
    result: dict = {"uids": [pmid]}
    if include_record:
        result[pmid] = {
            "title": title,
            "source": "J Fake Chem",
            "fulljournalname": "Journal of Fake Chemistry",
            "pubdate": "2020 Jan",
            "articleids": articleids,
        }
    return json.dumps({"result": result}).encode("utf-8")


def _fake_get_ok(url: str) -> bytes:
    assert_eutils_url(url)
    if "esearch.fcgi" in url:
        return _esearch_bytes([PMID])
    if "esummary.fcgi" in url:
        return _esummary_bytes()
    raise AssertionError(f"unexpected URL {url}")


def test_reinvent_env_check_does_not_probe_pubmed():
    result = check_environment()
    blob = json.dumps(result)
    assert "pubmed" not in blob
    assert "literature" not in blob
    assert "python" in result


def test_normalize_query_rejects_url_and_shellish():
    with pytest.raises(ValueError, match="not a URL"):
        normalize_query("https://evil.example/papers")
    with pytest.raises(ValueError, match="disallowed"):
        normalize_query("EGFR; rm -rf /")
    with pytest.raises(ValueError, match="empty"):
        normalize_query("   ")
    assert normalize_query("  EGFR tyrosine kinase  ") == "EGFR tyrosine kinase"


def test_assert_eutils_url_rejects_other_hosts():
    with pytest.raises(LiteratureNetworkError, match="NETWORK FAILURE"):
        assert_eutils_url("https://example.com/entrez/eutils/esearch.fcgi")
    with pytest.raises(LiteratureNetworkError, match="NETWORK FAILURE"):
        assert_eutils_url("http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")


def test_filter_sourced_drops_records_without_url_pmid_or_doi():
    kept = filter_sourced(
        [
            {"title": "unsourced guess", "pmid": "", "doi": "", "url": ""},
            {
                "pmid": PMID,
                "doi": DOI,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
                "title": TITLE,
                "journal": "J",
                "pubdate": "2020",
                "origin": "pubmed",
            },
        ]
    )
    assert len(kept) == 1
    assert kept[0]["pmid"] == PMID


def test_parse_esummary_skips_missing_records_instead_of_inventing():
    payload = json.loads(_esummary_bytes(include_record=False))
    entries = parse_esummary(payload, expected_ids=[PMID, "99999"])
    assert entries == []


def test_search_pubmed_uses_injected_get_fn(tmp_path: Path):
    seen: list[str] = []

    def get_fn(url: str) -> bytes:
        seen.append(url)
        return _fake_get_ok(url)

    entries = search_pubmed("EGFR inhibitor", retmax=5, get_fn=get_fn)
    assert len(entries) == 1
    assert entries[0]["pmid"] == PMID
    assert entries[0]["doi"] == DOI
    assert entries[0]["url"].endswith(f"{PMID}/")
    assert all("eutils.ncbi.nlm.nih.gov" in url for url in seen)
    assert tmp_path.exists()  # keep tmp_path used; no files invented here


def test_run_literature_requires_approval(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    with patch("tools.literature.pubmed.http_get") as mocked:
        result = run_literature(
            proj,
            query="EGFR inhibitor",
            approve=False,
        )
    mocked.assert_not_called()
    assert result["skipped"] is True
    assert result["table_present"] is False
    assert result["entries"] == []
    assert result["n_entries"] == 0
    assert "invented" in result["message"].lower() or "skipped" in result["message"].lower()
    assert not (proj / "output" / "literature" / "entries.csv").is_file()


def test_run_literature_mocked_pubmed_writes_sourced_table(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    result = run_literature(
        proj,
        query="EGFR inhibitor",
        approve=True,
        assume_yes=True,
        get_fn=_fake_get_ok,
    )
    assert result["ok"] is True
    assert result["success"] is True
    assert result["table_present"] is True
    assert result["n_entries"] == 1
    assert result["entries"][0]["pmid"] == PMID
    csv_path = Path(result["entries_csv"])
    assert csv_path.is_file()
    text = csv_path.read_text(encoding="utf-8")
    assert PMID in text
    assert DOI in text
    assert "https://pubmed.ncbi.nlm.nih.gov/" in text


def test_run_literature_network_failure_is_loud_not_invented(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()

    def boom(_url: str) -> bytes:
        raise LiteratureNetworkError("NETWORK FAILURE: cannot reach PubMed (timed out)")

    result = run_literature(
        proj,
        query="EGFR inhibitor",
        approve=True,
        assume_yes=True,
        get_fn=boom,
    )
    assert result["ok"] is False
    assert result["success"] is False
    assert result["table_present"] is False
    assert result["entries"] == []
    assert result["n_entries"] == 0
    errors = " ".join(result["errors"])
    assert "NETWORK FAILURE" in errors
    assert not (proj / "output" / "literature" / "entries.csv").is_file()


def test_run_literature_empty_hits_are_not_fabricated(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()

    def empty(url: str) -> bytes:
        assert_eutils_url(url)
        return _esearch_bytes([])

    result = run_literature(
        proj,
        query="EGFR inhibitor",
        approve=True,
        assume_yes=True,
        get_fn=empty,
    )
    assert result["ok"] is True
    assert result["success"] is True
    assert result["table_present"] is False
    assert result["entries"] == []
    assert "invented" in result["message"].lower() or "no sourced" in result["message"].lower()


def test_cli_dry_run_does_not_fetch(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    with patch("tools.literature.pubmed.http_get") as mocked:
        code = literature_main(
            ["--project", str(proj), "--query", "EGFR inhibitor"]
        )
    mocked.assert_not_called()
    assert code == 0


def test_executor_does_not_run_literature():
    plan = {
        "goal": "sneak literature",
        "project_dir": str(DEMO),
        "approve_run": False,
        "skip_reinvent": True,
        "steps": ["literature", "pubmed", "write_paper"],
        "notes": [],
    }
    with patch("subprocess.run") as mocked:
        results = ExecutionAgent().execute(
            plan, project_dir=DEMO, approve_run=False
        )
    mocked.assert_not_called()
    warnings = " ".join(results.get("warnings") or [])
    assert "independent module" in warnings
    assert "python -m tools.literature" in warnings
    assert "literature" not in results.get("steps", {})
    assert "pubmed" not in results.get("steps", {})


def test_plan_schema_rejects_literature_step():
    with pytest.raises(PlanValidationError, match="Unknown plan step"):
        validate_plan(
            {"steps": ["check_environment", "literature"]},
            project_dir=DEMO,
            output_dir=DEMO / "output",
            source="llm",
        )
    with pytest.raises(PlanValidationError, match="Unknown plan step"):
        validate_plan(
            {"steps": ["pubmed"]},
            project_dir=DEMO,
            output_dir=DEMO / "output",
            source="llm",
        )


def test_mcp_rejects_literature_and_write_paper_tools():
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("literature", {"query": "EGFR"})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("pubmed", {"query": "EGFR"})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("write_paper", {"title": "A discovery"})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("run_literature", {"project_dir": str(DEMO)})


def test_critic_rejects_unsourced_literature_shows():
    with pytest.raises(CriticValidationError, match="invented out-of-scope"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["Literature shows that these molecules bind EGFR."],
                "recommendation": "Proceed to wet lab.",
            },
            evidence={"analysis": {"total_molecules": 10}},
            source="llm",
        )


def test_critic_rejects_literature_claim_if_only_plan_names_the_step():
    evidence = build_evidence_summary(
        {
            "plan": {"steps": ["literature", "pubmed"]},
            "steps": {},
        }
    )
    assert "literature" not in evidence or evidence.get("literature") in (None, {})
    with pytest.raises(CriticValidationError, match="invented out-of-scope"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["PubMed literature review supports this scaffold."],
                "recommendation": "Cite the papers.",
            },
            evidence=evidence,
            source="llm",
        )


def test_critic_allows_sourced_pubmed_when_entries_present():
    evidence = build_evidence_summary(
        {
            "plan": {"skip_reinvent": True, "steps": ["literature"]},
            "steps": {
                "literature": {
                    "ok": True,
                    "approved": True,
                    "skipped": False,
                    "source": "pubmed",
                    "table_present": True,
                    "entries_csv": "output/literature/entries.csv",
                    "n_entries": 1,
                    "query": "EGFR inhibitor",
                    "entries": [
                        {
                            "pmid": PMID,
                            "doi": DOI,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
                            "title": TITLE,
                        }
                    ],
                }
            },
        }
    )
    assert evidence["literature"]["table_present"] is True
    result = validate_critic_verdict(
        {
            "status": "PASS",
            "issues": [f"Literature table lists PMID {PMID} from PubMed."],
            "recommendation": "Human review of the sourced entries is recommended.",
        },
        evidence=evidence,
        source="llm",
    )
    assert result["status"] == "PASS"


def test_critic_rejects_invented_pmid_even_when_table_present():
    evidence = build_evidence_summary(
        {
            "steps": {
                "literature": {
                    "ok": True,
                    "table_present": True,
                    "n_entries": 1,
                    "entries": [
                        {
                            "pmid": PMID,
                            "doi": DOI,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
                        }
                    ],
                }
            }
        }
    )
    with pytest.raises(CriticValidationError, match="invented citations"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["Literature shows PMID 11111111 is a key paper."],
                "recommendation": "Cite the invented paper.",
            },
            evidence=evidence,
            source="llm",
        )


def test_critic_still_rejects_docking_when_only_literature_table_present():
    evidence = build_evidence_summary(
        {
            "steps": {
                "literature": {
                    "ok": True,
                    "table_present": True,
                    "n_entries": 1,
                    "entries": [
                        {
                            "pmid": PMID,
                            "doi": DOI,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
                        }
                    ],
                }
            }
        }
    )
    with pytest.raises(CriticValidationError, match="vina"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["Vina docking score is -12.0."],
                "recommendation": "Great binders.",
            },
            evidence=evidence,
            source="llm",
        )


def test_deterministic_critic_warns_when_literature_skipped():
    verdict = CriticAgent().review(
        {
            "plan": {"approve_run": False, "skip_reinvent": True},
            "steps": {
                "literature": {
                    "ok": True,
                    "approved": False,
                    "skipped": True,
                    "table_present": False,
                    "n_entries": 0,
                    "entries": [],
                }
            },
            "warnings": [],
            "errors": [],
        }
    )
    assert verdict["status"] == "WARNING"
    assert any("approve-literature" in item for item in verdict["issues"])


def test_deterministic_critic_fails_loud_on_network_error():
    verdict = CriticAgent().review(
        {
            "plan": {"approve_run": False, "skip_reinvent": True},
            "steps": {
                "literature": {
                    "ok": False,
                    "approved": True,
                    "skipped": False,
                    "table_present": False,
                    "n_entries": 0,
                    "entries": [],
                    "errors": ["NETWORK FAILURE: cannot reach PubMed (timed out)"],
                    "message": "NETWORK FAILURE: cannot reach PubMed (timed out)",
                }
            },
            "warnings": [],
            "errors": [],
        }
    )
    assert verdict["status"] == "FAIL"
    assert any("NETWORK FAILURE" in item for item in verdict["issues"])


def test_deterministic_critic_reads_sourced_table():
    verdict = CriticAgent().review(
        {
            "plan": {"approve_run": False, "skip_reinvent": True},
            "steps": {
                "literature": {
                    "ok": True,
                    "approved": True,
                    "skipped": False,
                    "table_present": True,
                    "n_entries": 2,
                    "entries": [
                        {
                            "pmid": PMID,
                            "doi": DOI,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
                        }
                    ],
                }
            },
            "warnings": [],
            "errors": [],
        }
    )
    assert verdict["status"] == "PASS"
    assert any("sourced PubMed" in item for item in verdict["issues"])


def test_build_evidence_omits_literature_key_without_module():
    evidence = build_evidence_summary(
        {
            "plan": {"steps": ["analyze_molecules"]},
            "steps": {
                "analyze_molecules": {"ok": True, "total_molecules": 2},
            },
        }
    )
    assert "literature" not in evidence
    blob = json.dumps(evidence)
    assert '"literature"' not in blob
    assert "pubmed" not in blob
