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
    "glide",
    "gromacs",
    "namd",
    "openmm",
    "molecular dynamics",
    "md simulation",
    "pubmed",
    "literature review",
    "binding affinity",
    "ic50",
    "crystal structure",
    "free energy perturbation",
    "mm-gbsa",
    "mmgbsa",
)

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

    command = run.get("command")
    if not isinstance(command, list):
        command = None

    return {
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
        },
        "analysis": {
            "ok": analysis.get("ok"),
            "csv_path": analysis.get("csv_path"),
            "analysis_source": analysis.get("analysis_source"),
            "from_fresh_reinvent": analysis.get("from_fresh_reinvent"),
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
                "sa_score": _stat_block(rdkit.get("sa_score")),
                "pains": _pains_block(rdkit.get("pains")),
                "filters": _filters_block(rdkit.get("filters")),
            },
        },
        "thresholds": dict(thresholds or {}),
    }


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
    invented = [
        term
        for term in FORBIDDEN_CLAIM_TERMS
        if term in blob and term not in evidence_text
    ]
    if invented:
        raise CriticValidationError(
            "critic invented out-of-scope claims not present in evidence: "
            + ", ".join(invented)
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
