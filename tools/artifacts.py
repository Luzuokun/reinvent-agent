"""Persist structured run artefacts under logs/runs/<timestamp>/."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools import REPO_ROOT, dumps_pretty, load_agent_config


def create_run_dir(logs_dir: str | Path | None = None) -> Path:
    """Create and return ``logs/runs/<UTC timestamp>/``."""
    if logs_dir is None:
        cfg = load_agent_config()
        logs_dir = REPO_ROOT / cfg.get("logging", {}).get("logs_dirname", "logs")
    logs_dir = Path(logs_dir).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = logs_dir / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_result(
    run_dir: str | Path,
    *,
    goal: str,
    plan: dict[str, Any],
    results: dict[str, Any],
    critic: dict[str, Any],
    report: dict[str, Any] | None = None,
    exit_code: int,
    invocation: dict[str, Any] | None = None,
    from_run: dict[str, Any] | None = None,
) -> Path:
    """Write ``result.json`` with the full structured workflow payload."""
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = results.get("steps") or {}
    analysis = steps.get("analyze_molecules") or {}
    reinvent = steps.get("run_reinvent") or {}
    payload = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "goal": goal,
        "exit_code": exit_code,
        "critic_status": critic.get("status"),
        "plan": plan,
        "invocation": invocation or results.get("invocation"),
        "from_run": from_run or results.get("from_run"),
        "environment": steps.get("check_environment"),
        "validation": steps.get("validate_project"),
        "prepare_execution": steps.get("prepare_execution"),
        "reinvent": reinvent,
        "inventory": steps.get("find_output"),
        "analysis": analysis,
        "analysis_source": analysis.get("analysis_source"),
        "from_fresh_reinvent": analysis.get("from_fresh_reinvent"),
        "dry_run": bool(reinvent.get("skipped")),
        "critic": critic,
        "report": report,
        "warnings": results.get("warnings") or [],
        "errors": results.get("errors") or [],
        "project_dir": results.get("project_dir"),
        "aborted": results.get("aborted", False),
        "abort_reason": results.get("abort_reason"),
    }
    out_path = run_dir / "result.json"
    out_path.write_text(dumps_pretty(payload), encoding="utf-8")
    return out_path
