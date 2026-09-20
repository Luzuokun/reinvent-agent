"""Literature pipeline + CLI — separate from the REINVENT executor.

``python -m tools.literature --approve-literature`` is the only way this
module hits the network. The REINVENT executor never imports this package.
It never writes papers. Missing network is a loud failure, not fake citations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from tools import dumps_pretty, load_agent_config
from tools.literature.entries import filter_sourced, write_entries_csv
from tools.literature.pubmed import (
    DEFAULT_RETMAX,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_QUERY_CHARS,
    MAX_RETMAX,
    LiteratureNetworkError,
    clamp_retmax,
    normalize_query,
    search_pubmed,
)

OUTPUT_SUBDIR = "literature"


class LiteratureError(ValueError):
    """Sandbox, query, or approval failure (not a fabricated citation)."""


def run_literature(
    project_dir: str | Path,
    *,
    query: str,
    approve: bool = False,
    assume_yes: bool = False,
    retmax: int | None = None,
    timeout: float | None = None,
    agent_config: dict[str, Any] | None = None,
    get_fn: Any | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> dict[str, Any]:
    """Validate the query and optionally fetch sourced PubMed entries."""
    stdout = sys.stdout if stdout is None else stdout
    cfg = (agent_config or {}).get("literature") or {}
    try:
        query_norm = normalize_query(query)
    except ValueError as exc:
        raise LiteratureError(str(exc)) from exc
    retmax_n = clamp_retmax(
        retmax, default=int(cfg.get("default_retmax", DEFAULT_RETMAX))
    )
    timeout_s = float(
        timeout if timeout is not None else cfg.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    )

    project = Path(project_dir).expanduser().resolve()
    if not project.is_dir():
        raise LiteratureError(f"Project directory does not exist: {project}")

    output_dir = _literature_output_dir(project)
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = {
        "host": "eutils.ncbi.nlm.nih.gov",
        "db": "pubmed",
        "query": query_norm,
        "retmax": retmax_n,
        "timeout_seconds": timeout_s,
        "output_dir": str(output_dir),
        "writes_papers": False,
    }

    result: dict[str, Any] = {
        "ok": False,
        "approved": bool(approve),
        "skipped": False,
        "success": False,
        "source": "pubmed",
        "modality": "literature",
        "kind": "literature review",
        "project_dir": str(project),
        "output_dir": str(output_dir),
        "query": query_norm,
        "retmax": retmax_n,
        "entries_csv": None,
        "table_present": False,
        "n_entries": 0,
        "entries": [],
        "planned": planned,
        "warnings": [],
        "errors": [],
        "message": "",
        "writes_papers": False,
    }

    if not approve:
        result["ok"] = True
        result["skipped"] = True
        result["message"] = (
            "Literature search skipped (pass --approve-literature to query PubMed). "
            "No citations were invented."
        )
        result["warnings"].append(result["message"])
        _write_result_json(output_dir, result)
        return result

    if not confirm_literature_fetch(
        planned, assume_yes=assume_yes, stdin=stdin, stdout=stdout
    ):
        result["ok"] = True
        result["skipped"] = True
        result["approved"] = False
        result["message"] = (
            "Literature search not confirmed. No network call; no citations invented."
        )
        result["warnings"].append(result["message"])
        _write_result_json(output_dir, result)
        return result

    try:
        raw_entries = search_pubmed(
            query_norm,
            retmax=retmax_n,
            timeout=timeout_s,
            get_fn=get_fn,
        )
    except LiteratureNetworkError as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        result["warnings"].append(
            "PubMed was not reached; refusing to invent papers or PMIDs."
        )
        _write_result_json(output_dir, result)
        return result

    sourced = filter_sourced(raw_entries)
    dropped = len(raw_entries) - len(sourced)
    if dropped:
        result["warnings"].append(
            f"Dropped {dropped} PubMed record(s) that lacked URL, PMID, and DOI."
        )
    result["entries"] = sourced
    result["n_entries"] = len(sourced)
    result["ok"] = True
    if sourced:
        csv_path = write_entries_csv(output_dir / "entries.csv", sourced)
        result["entries_csv"] = str(csv_path)
        result["table_present"] = True
        result["success"] = True
        result["message"] = (
            f"Wrote {len(sourced)} sourced PubMed entries under {output_dir}"
        )
    else:
        result["success"] = True
        result["table_present"] = False
        result["message"] = (
            "PubMed returned no sourced hits (URL/PMID/DOI required). "
            "No citations were invented."
        )
        result["warnings"].append(result["message"])
    _write_result_json(output_dir, result)
    return result


def confirm_literature_fetch(
    planned: dict[str, Any],
    *,
    assume_yes: bool = False,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> bool:
    """Print the predefined PubMed query and require interactive confirmation."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    print("\n[Human approval required — literature / PubMed]", file=stdout)
    print("The agent is ready to run this predefined PubMed query:", file=stdout)
    print(f"  host:    {planned.get('host')}", file=stdout)
    print(f"  db:      {planned.get('db')}", file=stdout)
    print(f"  query:   {planned.get('query')}", file=stdout)
    print(f"  retmax:  {planned.get('retmax')}", file=stdout)
    print(f"  output:  {planned.get('output_dir')}", file=stdout)
    print(
        "Only entries with URL and/or PMID and/or DOI are kept. "
        "This module does not write papers.",
        file=stdout,
    )
    print(
        "Independent of REINVENT (--approve-run / --approve-dock / --approve-md "
        "do not apply here).",
        file=stdout,
    )
    if assume_yes:
        print("Proceeding due to --yes.", file=stdout)
        return True
    if not hasattr(stdin, "isatty") or not stdin.isatty():
        print(
            "Non-interactive stdin: re-run with --approve-literature --yes to fetch.",
            file=stdout,
        )
        return False
    print("Proceed? [y/N]: ", end="", file=stdout, flush=True)
    reply = stdin.readline().strip().lower()
    return reply in ("y", "yes")


def _literature_output_dir(project: Path) -> Path:
    dest = (project / "output" / OUTPUT_SUBDIR).resolve()
    try:
        dest.relative_to(project.resolve())
    except ValueError as exc:
        raise LiteratureError(f"Literature output escapes project: {dest}") from exc
    return dest


def _write_result_json(output_dir: Path, result: dict[str, Any]) -> None:
    path = output_dir / "literature_result.json"
    path.write_text(dumps_pretty(result), encoding="utf-8")
    result["result_json"] = str(path)


def _critic_payload(literature: dict[str, Any]) -> dict[str, Any]:
    """Build a results dict the shared Critic can review (evidence only)."""
    return {
        "plan": {
            "approve_run": False,
            "skip_reinvent": True,
            "approve_literature": literature.get("approved"),
            "steps": ["literature"],
        },
        "steps": {"literature": literature},
        "warnings": literature.get("warnings") or [],
        "errors": literature.get("errors") or [],
        "aborted": bool(literature.get("errors")) and not literature.get("skipped"),
        "abort_reason": literature.get("message") if literature.get("errors") else None,
        "project_dir": literature.get("project_dir"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independent PubMed literature tool. Returns only sourced entries "
            "(URL / PMID / DOI). Not part of the REINVENT executor. Requires "
            "--approve-literature to hit the network. Never invents papers."
        )
    )
    parser.add_argument("--project", required=True, help="Project directory")
    parser.add_argument(
        "--query",
        required=True,
        help=(
            f"PubMed search string (max {MAX_QUERY_CHARS} chars). "
            "Not a URL and not a shell command."
        ),
    )
    parser.add_argument(
        "--retmax",
        type=int,
        default=None,
        help=f"Max hits to request (cap {MAX_RETMAX}; default {DEFAULT_RETMAX})",
    )
    parser.add_argument(
        "--approve-literature",
        action="store_true",
        help="Human approval to query NCBI PubMed E-utilities",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive Proceed? prompt when used with --approve-literature",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional agent.yaml (literature defaults only; never TOML)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent_config: dict[str, Any] = {}
    if args.config:
        agent_config = load_agent_config(Path(args.config))
    else:
        try:
            agent_config = load_agent_config()
        except (OSError, ValueError):
            agent_config = {}

    try:
        result = run_literature(
            args.project,
            query=args.query,
            approve=bool(args.approve_literature),
            assume_yes=bool(args.yes),
            retmax=args.retmax,
            agent_config=agent_config,
        )
    except LiteratureError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(json.dumps(_public_result(result), indent=2, ensure_ascii=False))

    from agents.critic import CriticAgent

    verdict = CriticAgent(agent_config=agent_config).review(_critic_payload(result))
    print("\n[Critic Agent]")
    print(f"Status: {verdict.get('status')}")
    for issue in verdict.get("issues") or []:
        print(f"  - {issue}")
    print(f"Recommendation: {verdict.get('recommendation')}")

    if str(verdict.get("status", "")).upper() == "FAIL":
        return 1
    return 0


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    return payload
