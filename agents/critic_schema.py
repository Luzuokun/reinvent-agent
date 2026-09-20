"""Strict critic schema — evidence-only verdicts, no invented science.

The model may emit only ``status``, ``issues``, and ``recommendation``.
It never calls tools. Runtime-owned fields (thresholds, critic backend
metadata) are attached after validation.
"""

from __future__ import annotations

import json
import re
from typing import Any

ALLOWED_STATUS: tuple[str, ...] = ("PASS", "WARNING", "FAIL")

LLM_CRITIC_KEYS = frozenset({"status", "issues", "recommendation"})
SYSTEM_CRITIC_KEYS = frozenset(
    {
        "critic",
        "critic_fallback",
        "critic_warnings",
        "thresholds",
    }
)
ALLOWED_CRITIC_KEYS = LLM_CRITIC_KEYS | SYSTEM_CRITIC_KEYS

MAX_ISSUES = 16
MAX_ISSUE_CHARS = 400
MAX_RECOMMENDATION_CHARS = 800

# Claims that this MVP never produces as evidence. If the model mentions
# them and they are absent from the evidence JSON, the verdict is rejected.
FORBIDDEN_CLAIM_TERMS: tuple[str, ...] = (
    "docking",
    "autodock",
    "vina",
    "gnina",
    "glide",
    "gromacs",
    "namd",
    "openmm",
    "molecular dynamics",
    "md simulation",
    "pubmed",
    "literature review",
    "literature shows",
    "literature",
    "pmid",
    "binding affinity",
    "ic50",
    "crystal structure",
    "free energy perturbation",
    "mm-gbsa",
    "mmgbsa",
    "rmsf",
)

# These may appear in a critic verdict only when evidence.docking.table_present.
DOCKING_CLAIM_TERMS: frozenset[str] = frozenset(
    {"docking", "autodock", "vina", "gnina"}
)

# These may appear only when evidence.md.table_present (independent MD module).
MD_CLAIM_TERMS: frozenset[str] = frozenset(
    {
        "gromacs",
        "molecular dynamics",
        "md simulation",
        "rmsf",
    }
)

# These may appear only when evidence.literature has sourced URL/PMID/DOI entries.
LITERATURE_CLAIM_TERMS: frozenset[str] = frozenset(
    {
        "pubmed",
        "literature review",
        "literature shows",
        "literature",
        "pmid",
    }
)

_PMID_MENTION = re.compile(r"\bpmid[:\s#]*(\d{5,9})\b", re.IGNORECASE)
_PUBMED_URL_PMID = re.compile(
    r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{5,9})",
    re.IGNORECASE,
)
_DOI_MENTION = re.compile(r"\b(10\.\d{4,9}/[^\s,;]+)", re.IGNORECASE)

_SHELLISH = re.compile(
    r"(sudo\s|rm\s+-|chmod\s|curl\s|wget\s|bash\s+-|/bin/sh|subprocess|shell=True)",
    re.IGNORECASE,
)

LLM_CRITIC_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {
            "type": "string",
            "enum": list(ALLOWED_STATUS),
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
        },
        "recommendation": {"type": "string"},
    },
    "required": ["status", "issues", "recommendation"],
}


class CriticValidationError(ValueError):
    """Raised when an LLM critic payload is unsafe or does not match the schema."""


def parse_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise CriticValidationError("LLM returned empty critic text")
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.strip("`")
        if payload.startswith("json"):
            payload = payload[4:]
        payload = payload.strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CriticValidationError(f"LLM critic is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CriticValidationError("LLM critic JSON must be an object")
    return data


def build_evidence_summary(
    results: dict[str, Any],
    *,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact structured evidence for the LLM critic.

    Omits SMILES lists, CSV contents, HTML, and stdout. Paths are included
    as strings only. This is the *only* scientific input the model sees.
    """
    plan = results.get("plan") if isinstance(results.get("plan"), dict) else {}
    steps = results.get("steps") if isinstance(results.get("steps"), dict) else {}
    env = steps.get("check_environment") if isinstance(steps.get("check_environment"), dict) else {}
    validation = steps.get("validate_project") if isinstance(steps.get("validate_project"), dict) else {}
    run = steps.get("run_reinvent") if isinstance(steps.get("run_reinvent"), dict) else {}
    inventory = steps.get("find_output") if isinstance(steps.get("find_output"), dict) else {}
    analysis = steps.get("analyze_molecules") if isinstance(steps.get("analyze_molecules"), dict) else {}
    rdkit = analysis.get("rdkit") if isinstance(analysis.get("rdkit"), dict) else {}
    docking = steps.get("docking") if isinstance(steps.get("docking"), dict) else {}
    if not docking and isinstance(results.get("docking"), dict):
        docking = results["docking"]
    md = steps.get("md") if isinstance(steps.get("md"), dict) else {}
    if not md and isinstance(results.get("md"), dict):
        md = results["md"]
    literature = steps.get("literature") if isinstance(steps.get("literature"), dict) else {}
    if not literature and isinstance(results.get("literature"), dict):
        literature = results["literature"]

    command = run.get("command")
    if not isinstance(command, list):
        command = None

    summary = {
        "plan": {
            "approve_run": plan.get("approve_run"),
            "skip_reinvent": plan.get("skip_reinvent"),
            "steps": plan.get("steps"),
            "planner": plan.get("planner"),
            "planner_fallback": plan.get("planner_fallback"),
        },
        "aborted": results.get("aborted"),
        "abort_reason": _truncate_text(results.get("abort_reason")),
        "errors": _truncate_str_list(results.get("errors")),
        "warnings": _truncate_str_list(results.get("warnings")),
        "environment": {
            "reinvent": env.get("reinvent"),
            "rdkit": env.get("rdkit"),
            "gpu": env.get("gpu"),
        },
        "validation": {
            "ok": validation.get("ok"),
            "errors": _truncate_str_list(validation.get("errors")),
            "run_type": (validation.get("details") or {}).get("run_type")
            if isinstance(validation.get("details"), dict)
            else None,
        },
        "run_reinvent": {
            "skipped": run.get("skipped"),
            "success": run.get("success"),
            "exit_code": run.get("exit_code"),
            "runtime_seconds": run.get("runtime_seconds"),
            "message": _truncate_text(run.get("message")),
            "command": command,
        },
        "inventory": {
            "ok": inventory.get("ok"),
            "csv_count": inventory.get("csv_count"),
            "model_count": inventory.get("model_count"),
        },
        "analysis": {
            "ok": analysis.get("ok"),
            "csv_path": analysis.get("csv_path"),
            "analysis_source": analysis.get("analysis_source"),
            "from_fresh_reinvent": analysis.get("from_fresh_reinvent"),
            "run_type": analysis.get("run_type"),
            "artefact_kind": analysis.get("artefact_kind"),
            "artefact_path": analysis.get("artefact_path"),
            "total_molecules": analysis.get("total_molecules"),
            "unique_molecules": analysis.get("unique_molecules"),
            "duplicate_molecules": analysis.get("duplicate_molecules"),
            "duplicate_fraction": analysis.get("duplicate_fraction"),
            "errors": _truncate_str_list(analysis.get("errors")),
            "warnings": _truncate_str_list(analysis.get("warnings")),
            "rdkit": {
                "available": rdkit.get("available"),
                "valid_molecules": rdkit.get("valid_molecules"),
                "invalid_molecules": rdkit.get("invalid_molecules"),
                "valid_fraction": rdkit.get("valid_fraction"),
                "mw": _stat_block(rdkit.get("mw")),
                "logp": _stat_block(rdkit.get("logp")),
                "qed": _stat_block(rdkit.get("qed")),
                "tpsa": _stat_block(rdkit.get("tpsa")),
                "hbd": _stat_block(rdkit.get("hbd")),
                "hba": _stat_block(rdkit.get("hba")),
                "rotatable_bonds": _stat_block(rdkit.get("rotatable_bonds")),
                "sa_score": _stat_block(rdkit.get("sa_score")),
                "pains": _pains_block(rdkit.get("pains")),
                "filters": _filters_block(rdkit.get("filters")),
                "lipinski": _lipinski_block(rdkit.get("lipinski")),
            },
        },
        "thresholds": dict(thresholds or {}),
    }
    docking_block = _docking_block(docking)
    if docking_block:
        summary["docking"] = docking_block
    md_block = _md_block(md)
    if md_block:
        summary["md"] = md_block
    literature_block = _literature_block(literature)
    if literature_block:
        summary["literature"] = literature_block
    return summary


def validate_critic_verdict(
    raw: Any,
    *,
    evidence: dict[str, Any] | None = None,
    source: str = "llm",
) -> dict[str, Any]:
    """Validate and normalize a critic JSON object.

    ``source="llm"`` rejects runtime-only keys so the model cannot set
    ``thresholds`` or ``critic_fallback``.
    """
    if not isinstance(raw, dict):
        raise CriticValidationError("critic verdict must be a JSON object")

    if source == "llm":
        allowed = LLM_CRITIC_KEYS
    elif source == "any":
        allowed = ALLOWED_CRITIC_KEYS
    else:
        raise CriticValidationError(f"Unknown validation source: {source}")

    extra = sorted(set(raw) - allowed)
    if extra:
        raise CriticValidationError(f"Unknown critic keys: {extra}")

    status = raw.get("status")
    if not isinstance(status, str) or status not in ALLOWED_STATUS:
        raise CriticValidationError(
            f"status must be one of {list(ALLOWED_STATUS)}; got {status!r}"
        )

    issues = _parse_issues(raw.get("issues"))
    recommendation = raw.get("recommendation")
    if not isinstance(recommendation, str) or not recommendation.strip():
        raise CriticValidationError("recommendation must be a non-empty string")
    recommendation = recommendation.strip()
    if len(recommendation) > MAX_RECOMMENDATION_CHARS:
        raise CriticValidationError(
            f"recommendation exceeds {MAX_RECOMMENDATION_CHARS} characters"
        )

    _reject_shellish(issues, recommendation)
    _reject_ungrounded_claims(issues, recommendation, evidence or {})
    _reject_invented_citations(issues, recommendation, evidence or {})

    seen: set[str] = set()
    unique_issues: list[str] = []
    for item in issues:
        if item not in seen:
            seen.add(item)
            unique_issues.append(item)

    return {
        "status": status,
        "issues": unique_issues,
        "recommendation": recommendation,
    }


def _parse_issues(issues: Any) -> list[str]:
    if issues is None:
        return []
    if not isinstance(issues, list):
        raise CriticValidationError("issues must be a list of strings")
    if len(issues) > MAX_ISSUES:
        raise CriticValidationError(f"issues exceed limit ({len(issues)} > {MAX_ISSUES})")
    cleaned: list[str] = []
    for item in issues:
        if not isinstance(item, str):
            raise CriticValidationError("issues must be a list of strings")
        text = item.strip()
        if not text:
            continue
        if len(text) > MAX_ISSUE_CHARS:
            raise CriticValidationError(
                f"issue exceeds {MAX_ISSUE_CHARS} characters (never used as a command)"
            )
        cleaned.append(text)
    return cleaned


def _reject_shellish(issues: list[str], recommendation: str) -> None:
    blob = " ".join(issues) + " " + recommendation
    if _SHELLISH.search(blob):
        raise CriticValidationError(
            "critic text looks like a shell/command; rejected (never executed)"
        )


def _reject_ungrounded_claims(
    issues: list[str],
    recommendation: str,
    evidence: dict[str, Any],
) -> None:
    evidence_text = json.dumps(evidence, ensure_ascii=False, default=str).lower()
    blob = (" ".join(issues) + " " + recommendation).lower()
    docking_ok = _has_docking_table(evidence)
    md_ok = _has_md_table(evidence)
    literature_ok = _has_sourced_literature(evidence)
    invented: list[str] = []
    for term in FORBIDDEN_CLAIM_TERMS:
        if term not in blob:
            continue
        if term in DOCKING_CLAIM_TERMS:
            # Plan step names must not unlock docking talk; need a score table.
            if not docking_ok or term not in evidence_text:
                invented.append(term)
            continue
        if term in MD_CLAIM_TERMS:
            if not md_ok or term not in evidence_text:
                invented.append(term)
            continue
        if term in LITERATURE_CLAIM_TERMS:
            # Need sourced URL/PMID/DOI entries, not just a literature step name.
            if not literature_ok or term not in evidence_text:
                invented.append(term)
            continue
        if term not in evidence_text:
            invented.append(term)
    if invented:
        raise CriticValidationError(
            "critic invented out-of-scope claims not present in evidence: "
            + ", ".join(invented)
        )


def _has_docking_table(evidence: dict[str, Any]) -> bool:
    docking = evidence.get("docking") if isinstance(evidence, dict) else None
    if not isinstance(docking, dict) or not docking:
        return False
    return bool(
        docking.get("table_present")
        or docking.get("scores_csv")
        or (isinstance(docking.get("score"), dict) and docking.get("n_scored"))
    )


def _has_md_table(evidence: dict[str, Any]) -> bool:
    md = evidence.get("md") if isinstance(evidence, dict) else None
    if not isinstance(md, dict) or not md:
        return False
    return bool(
        md.get("table_present")
        or md.get("rmsd_csv")
        or md.get("rmsf_csv")
        or (isinstance(md.get("rmsd"), dict) and md.get("n_frames"))
    )


def _has_sourced_literature(evidence: dict[str, Any]) -> bool:
    literature = evidence.get("literature") if isinstance(evidence, dict) else None
    if not isinstance(literature, dict) or not literature:
        return False
    return bool(_sourced_literature_entries(literature))


def _sourced_literature_entries(literature: dict[str, Any]) -> list[dict[str, str]]:
    raw = literature.get("entries") if isinstance(literature, dict) else None
    if not isinstance(raw, list):
        return []
    sourced: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pmid = str(item.get("pmid") or "").strip()
        doi = str(item.get("doi") or "").strip()
        url = str(item.get("url") or "").strip()
        if not (pmid.isdigit() or doi or url.startswith("http://") or url.startswith("https://")):
            continue
        sourced.append({"pmid": pmid, "doi": doi.lower().rstrip("."), "url": url})
    return sourced


def _reject_invented_citations(
    issues: list[str],
    recommendation: str,
    evidence: dict[str, Any],
) -> None:
    """Reject PMIDs / DOIs in critic text that are not in sourced evidence."""
    blob = " ".join(issues) + " " + recommendation
    mentioned_pmids = set(_PMID_MENTION.findall(blob)) | set(_PUBMED_URL_PMID.findall(blob))
    mentioned_dois = {item.rstrip(").,;").lower() for item in _DOI_MENTION.findall(blob)}
    if not mentioned_pmids and not mentioned_dois:
        return
    literature = evidence.get("literature") if isinstance(evidence, dict) else None
    sourced = _sourced_literature_entries(literature if isinstance(literature, dict) else {})
    allowed_pmids = {row["pmid"] for row in sourced if row["pmid"]}
    allowed_dois = {row["doi"] for row in sourced if row["doi"]}
    extra_pmids = sorted(mentioned_pmids - allowed_pmids)
    extra_dois = sorted(doi for doi in mentioned_dois if doi not in allowed_dois)
    invented: list[str] = []
    if extra_pmids:
        invented.append("pmid " + ", ".join(extra_pmids))
    if extra_dois:
        invented.append("doi " + ", ".join(extra_dois))
    if invented:
        raise CriticValidationError(
            "critic invented citations not present in sourced literature evidence: "
            + "; ".join(invented)
        )


def _truncate_text(value: Any, limit: int = 400) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _truncate_str_list(value: Any, *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:limit]:
        text = _truncate_text(item)
        if text:
            out.append(text)
    return out


def _stat_block(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "mean": value.get("mean"),
        "stdev": value.get("stdev"),
        "n": value.get("n"),
    }


def _md_stat_block(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "mean": value.get("mean"),
        "max": value.get("max"),
        "last": value.get("last"),
        "n": value.get("n"),
    }


def _pains_block(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "available": value.get("available"),
        "molecules_with_hits": value.get("molecules_with_hits"),
        "total_hits": value.get("total_hits"),
        "hit_fraction": value.get("hit_fraction"),
    }


def _filters_block(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "qed_pass_threshold": value.get("qed_pass_threshold"),
        "qed_pass_count": value.get("qed_pass_count"),
        "pains_free_count": value.get("pains_free_count"),
        "qed_pass_and_pains_free_count": value.get("qed_pass_and_pains_free_count"),
    }


def _lipinski_block(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "rules": value.get("rules"),
        "pass": value.get("pass"),
        "fail": value.get("fail"),
        "fraction": value.get("fraction"),
    }


def _docking_block(value: Any) -> dict[str, Any] | None:
    """Include a docking table summary only when the docking module attached one.

    Omitting this block (or leaving it empty) keeps docking/vina claims forbidden.
    SMILES lists, pose coordinates, and raw logs are never copied here.
    """
    if not isinstance(value, dict) or not value:
        return None
    engine = value.get("engine")
    table_present = bool(
        value.get("table_present")
        or value.get("scores_csv")
        or (isinstance(value.get("score"), dict) and value.get("score"))
    )
    return {
        "ok": value.get("ok"),
        "approved": value.get("approved"),
        "skipped": value.get("skipped"),
        "success": value.get("success"),
        "engine": engine,
        "table_present": table_present,
        "scores_csv": value.get("scores_csv"),
        "n_ligands": value.get("n_ligands"),
        "n_scored": value.get("n_scored"),
        "score": _stat_block(value.get("score"))
        if isinstance(value.get("score"), dict)
        else None,
        "best_score": (value.get("score") or {}).get("best")
        if isinstance(value.get("score"), dict)
        else value.get("best_score"),
        "message": _truncate_text(value.get("message")),
        "errors": _truncate_str_list(value.get("errors")),
    }


def _md_block(value: Any) -> dict[str, Any] | None:
    """Include an MD table summary only when the MD module attached one.

    Omitting this block keeps GROMACS / RMSF / MD-simulation claims forbidden.
    Full mdp text, trajectories, and stdout are never copied here.
    """
    if not isinstance(value, dict) or not value:
        return None
    table_present = bool(
        value.get("table_present")
        or value.get("rmsd_csv")
        or value.get("rmsf_csv")
        or (isinstance(value.get("rmsd"), dict) and value.get("rmsd"))
    )
    return {
        "ok": value.get("ok"),
        "approved": value.get("approved"),
        "skipped": value.get("skipped"),
        "success": value.get("success"),
        "engine": value.get("engine") or "gmx",
        "package": "gromacs",
        "modality": "molecular dynamics",
        "kind": "md simulation",
        "protocol": value.get("protocol"),
        "table_present": table_present,
        "rmsd_csv": value.get("rmsd_csv"),
        "rmsf_csv": value.get("rmsf_csv"),
        "n_frames": value.get("n_frames"),
        "rmsd": _md_stat_block(value.get("rmsd"))
        if isinstance(value.get("rmsd"), dict)
        else None,
        "rmsf": _md_stat_block(value.get("rmsf"))
        if isinstance(value.get("rmsf"), dict)
        else None,
        "message": _truncate_text(value.get("message")),
        "errors": _truncate_str_list(value.get("errors")),
    }


def _literature_block(value: Any) -> dict[str, Any] | None:
    """Include a literature summary only when the literature module attached one.

    Omitting this block keeps PubMed / literature-shows / PMID claims forbidden.
    Abstracts and full paper text are never copied here. Entries without URL,
    PMID, or DOI are dropped so the critic cannot treat unsourced rows as evidence.
    """
    if not isinstance(value, dict) or not value:
        return None
    sourced = _sourced_literature_entries(value)
    compact = []
    for item in sourced[:8]:
        compact.append(
            {
                "pmid": item.get("pmid") or None,
                "doi": item.get("doi") or None,
                "url": item.get("url") or None,
            }
        )
    table_present = bool(compact)
    n_entries = len(sourced) if sourced else 0
    return {
        "ok": value.get("ok"),
        "approved": value.get("approved"),
        "skipped": value.get("skipped"),
        "success": value.get("success"),
        "source": "pubmed",
        "package": "pubmed",
        "modality": "literature",
        "kind": "literature review",
        "phrase": "literature shows",
        "table_present": table_present,
        "n_entries": n_entries if table_present else 0,
        "entries_csv": value.get("entries_csv") if table_present else None,
        "query": _truncate_text(value.get("query"), limit=200),
        "entries": compact,
        "message": _truncate_text(value.get("message")),
        "errors": _truncate_str_list(value.get("errors")),
        "writes_papers": False,
    }
