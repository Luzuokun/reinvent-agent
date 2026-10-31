"""ChEMBL pipeline + CLI — separate from the REINVENT executor.

``python -m tools.chembl --approve-chembl`` is the only way this module hits
the network. The REINVENT executor never imports this package. Missing
network is a loud failure, not fake assay rows. Optional ``--prep-smiles``
runs local ``smiles_prep`` on the downloaded CSV (still no invented SMILES).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from tools import dumps_pretty, load_agent_config
from tools.chembl.client import (
    CHEMBL_HOST,
    DEFAULT_LIMIT,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_LIMIT,
    ChemblNetworkError,
    clamp_limit,
    fetch_activities,
    rows_with_smiles,
)
from tools.chembl.entries import write_activities_csv, write_smiles_csv
from tools.chembl.queries import (
    ChemblQueryError,
    list_query_ids,
    query_catalog,
    resolve_query,
)

OUTPUT_SUBDIR = "chembl"


class ChemblError(ValueError):
    """Sandbox, query, or approval failure (not a fabricated assay row)."""


def run_chembl(
    project_dir: str | Path,
    *,
    query_id: str,
    approve: bool = False,
    assume_yes: bool = False,
    limit: int | None = None,
    timeout: float | None = None,
    prep_smiles: bool = False,
    agent_config: dict[str, Any] | None = None,
    get_fn: Any | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> dict[str, Any]:
    """Validate a predefined query id and optionally fetch ChEMBL activities."""
    stdout = sys.stdout if stdout is None else stdout
    cfg = (agent_config or {}).get("chembl") or {}
    try:
        spec = resolve_query(query_id)
    except ChemblQueryError as exc:
        raise ChemblError(str(exc)) from exc

    limit_n = clamp_limit(
        limit,
        default=int(cfg.get("default_limit", spec.get("default_limit", DEFAULT_LIMIT))),
    )
    timeout_s = float(
        timeout if timeout is not None else cfg.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    )

    project = Path(project_dir).expanduser().resolve()
    if not project.is_dir():
        raise ChemblError(f"Project directory does not exist: {project}")

    output_dir = _chembl_output_dir(project)
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = {
        "host": CHEMBL_HOST,
        "api_prefix": "/chembl/api/data",
        "resource": "activity",
        "query_id": spec["id"],
        "target_chembl_id": spec["target_chembl_id"],
        "standard_type": spec["standard_type"],
        "assay_type": spec.get("assay_type"),
        "limit": limit_n,
        "timeout_seconds": timeout_s,
        "output_dir": str(output_dir),
        "prep_smiles": bool(prep_smiles),
        "invents_rows": False,
    }

    result: dict[str, Any] = {
        "ok": False,
        "approved": bool(approve),
        "skipped": False,
        "success": False,
        "source": "chembl",
        "modality": "assay",
        "kind": "chembl download",
        "project_dir": str(project),
        "output_dir": str(output_dir),
        "query_id": spec["id"],
        "query": {
            "id": spec["id"],
            "description": spec["description"],
            "target_chembl_id": spec["target_chembl_id"],
            "standard_type": spec["standard_type"],
            "assay_type": spec.get("assay_type"),
        },
        "limit": limit_n,
        "activities_csv": None,
        "smiles_csv": None,
        "smiles_prep": None,
        "table_present": False,
        "n_activities": 0,
        "n_with_smiles": 0,
        "activities": [],
        "planned": planned,
        "warnings": [],
        "errors": [],
        "message": "",
        "invents_rows": False,
    }

    if not approve:
        result["ok"] = True
        result["skipped"] = True
        result["message"] = (
            "ChEMBL download skipped (pass --approve-chembl to query "
            f"{CHEMBL_HOST}). No assay rows were invented."
        )
        result["warnings"].append(result["message"])
        _write_result_json(output_dir, result)
        return result

    if not confirm_chembl_fetch(
        planned, assume_yes=assume_yes, stdin=stdin, stdout=stdout
    ):
        result["ok"] = True
        result["skipped"] = True
        result["approved"] = False
        result["message"] = (
            "ChEMBL download not confirmed. No network call; no assay rows invented."
        )
        result["warnings"].append(result["message"])
        _write_result_json(output_dir, result)
        return result

    try:
        rows = fetch_activities(
            spec,
            limit=limit_n,
            timeout=timeout_s,
            get_fn=get_fn,
        )
    except ChemblNetworkError as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        result["warnings"].append(
            "ChEMBL was not reached; refusing to invent SMILES or IC50 values."
        )
        _write_result_json(output_dir, result)
        return result

    smiled = rows_with_smiles(rows)
    result["activities"] = rows
    result["n_activities"] = len(rows)
    result["n_with_smiles"] = len(smiled)
    result["ok"] = True
    result["success"] = True

    if rows:
        csv_path = write_activities_csv(output_dir / "activities.csv", rows)
        result["activities_csv"] = str(csv_path)
        result["table_present"] = True
        if smiled:
            smiles_path = write_smiles_csv(output_dir / "smiles.csv", smiled)
            result["smiles_csv"] = str(smiles_path)
        else:
            result["warnings"].append(
                "Activity table has no canonical_smiles values; "
                "smiles.csv was not written and --prep-smiles cannot invent them."
            )
        result["message"] = (
            f"Wrote {len(rows)} ChEMBL activities "
            f"({len(smiled)} with SMILES) under {output_dir}"
        )
    else:
        result["table_present"] = False
        result["message"] = (
            "ChEMBL returned no activity rows for this predefined query. "
            "No assay values were invented."
        )
        result["warnings"].append(result["message"])

    if prep_smiles:
        result["smiles_prep"] = _maybe_prep_smiles(
            project=project,
            smiles_csv=result.get("smiles_csv"),
            query_id=spec["id"],
            warnings=result["warnings"],
        )

    _write_result_json(output_dir, result)
    return result


def confirm_chembl_fetch(
    planned: dict[str, Any],
    *,
    assume_yes: bool = False,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> bool:
    """Print the predefined ChEMBL query and require interactive confirmation."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    print("\n[Human approval required — ChEMBL assay download]", file=stdout)
    print("The agent is ready to run this predefined ChEMBL query:", file=stdout)
    print(f"  host:              {planned.get('host')}", file=stdout)
    print(f"  resource:          {planned.get('resource')}", file=stdout)
    print(f"  query_id:          {planned.get('query_id')}", file=stdout)
    print(f"  target_chembl_id:  {planned.get('target_chembl_id')}", file=stdout)
    print(f"  standard_type:     {planned.get('standard_type')}", file=stdout)
    print(f"  assay_type:        {planned.get('assay_type')}", file=stdout)
    print(f"  limit:             {planned.get('limit')}", file=stdout)
    print(f"  output:            {planned.get('output_dir')}", file=stdout)
    print(
        "Only predefined query ids are allowed. This module does not invent "
        "SQL, curl, or missing SMILES/IC50 values.",
        file=stdout,
    )
    print(
        "Independent of REINVENT (--approve-run / --approve-dock / --approve-md "
        "/ --approve-literature do not apply here).",
        file=stdout,
    )
    if assume_yes:
        print("Proceeding due to --yes.", file=stdout)
        return True
    if not hasattr(stdin, "isatty") or not stdin.isatty():
        print(
            "Non-interactive stdin: re-run with --approve-chembl --yes to fetch.",
            file=stdout,
        )
        return False
    print("Proceed? [y/N]: ", end="", file=stdout, flush=True)
    reply = stdin.readline().strip().lower()
    return reply in ("y", "yes")


def _maybe_prep_smiles(
    *,
    project: Path,
    smiles_csv: str | None,
    query_id: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    if not smiles_csv:
        warnings.append(
            "--prep-smiles requested but no smiles.csv was written; "
            "refusing to invent training SMILES."
        )
        return {
            "ok": False,
            "skipped": True,
            "message": "No SMILES CSV available for smiles_prep",
        }
    try:
        from tools.smiles_prep import SmilesPrepError, prepare_training_smiles
    except ImportError as exc:  # pragma: no cover - package always present in repo
        warnings.append(f"smiles_prep unavailable: {exc}")
        return {"ok": False, "skipped": True, "message": str(exc)}

    dest = Path("input") / f"chembl_{query_id}.smi"
    try:
        prep = prepare_training_smiles(
            project,
            smiles_csv,
            dest_relative=dest,
        )
    except SmilesPrepError as exc:
        warnings.append(f"smiles_prep failed: {exc}")
        return {"ok": False, "skipped": False, "message": str(exc)}
    return prep


def _chembl_output_dir(project: Path) -> Path:
    dest = (project / "output" / OUTPUT_SUBDIR).resolve()
    try:
        dest.relative_to(project.resolve())
    except ValueError as exc:
        raise ChemblError(f"ChEMBL output escapes project: {dest}") from exc
    return dest


def _write_result_json(output_dir: Path, result: dict[str, Any]) -> None:
    path = output_dir / "chembl_result.json"
    # Do not dump full activity arrays into the sidecar when large — keep a
    # compact copy for critics / audits (counts + paths + first few ids).
    compact = dict(result)
    activities = compact.get("activities")
    if isinstance(activities, list) and len(activities) > 20:
        compact["activities_preview"] = activities[:20]
        compact["activities"] = []
        compact["activities_omitted_from_json"] = len(activities) - 20
    path.write_text(dumps_pretty(compact), encoding="utf-8")
    result["result_json"] = str(path)


def _critic_payload(chembl: dict[str, Any]) -> dict[str, Any]:
    """Build a results dict the shared Critic can review (evidence only)."""
    return {
        "plan": {
            "approve_run": False,
            "skip_reinvent": True,
            "approve_chembl": chembl.get("approved"),
            "steps": ["chembl"],
        },
        "steps": {"chembl": chembl},
        "warnings": chembl.get("warnings") or [],
        "errors": chembl.get("errors") or [],
        "aborted": bool(chembl.get("errors")) and not chembl.get("skipped"),
        "abort_reason": chembl.get("message") if chembl.get("errors") else None,
        "project_dir": chembl.get("project_dir"),
    }


def build_parser() -> argparse.ArgumentParser:
    allowed = ", ".join(list_query_ids())
    parser = argparse.ArgumentParser(
        description=(
            "Independent ChEMBL assay download tool. Uses predefined query "
            "ids only (no LLM SQL). Not part of the REINVENT executor. "
            "Requires --approve-chembl to hit the network. Never invents rows."
        )
    )
    parser.add_argument(
        "--project",
        required=False,
        default=None,
        help="Project directory (required unless --list-queries)",
    )
    parser.add_argument(
        "--query-id",
        required=False,
        default=None,
        help=f"Predefined ChEMBL query id. Allowed: {allowed}",
    )
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="Print the predefined query catalog as JSON and exit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"Max activities to request (cap {MAX_LIMIT}; default {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--approve-chembl",
        action="store_true",
        help="Human approval to query the ChEMBL REST API",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive Proceed? prompt when used with --approve-chembl",
    )
    parser.add_argument(
        "--prep-smiles",
        action="store_true",
        help=(
            "After a successful download, run local smiles_prep on smiles.csv "
            "into input/chembl_<query_id>.smi (requires RDKit; no network)"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional agent.yaml (chembl defaults only; never TOML)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_queries:
        print(json.dumps(query_catalog(), indent=2, ensure_ascii=False))
        return 0
    if not args.project:
        print("ERROR: --project is required (or pass --list-queries).")
        return 2
    if not args.query_id:
        print(
            "ERROR: --query-id is required (or pass --list-queries). "
            f"Allowed: {', '.join(list_query_ids())}"
        )
        return 2

    agent_config: dict[str, Any] = {}
    if args.config:
        agent_config = load_agent_config(Path(args.config))
    else:
        try:
            agent_config = load_agent_config()
        except (OSError, ValueError):
            agent_config = {}

    try:
        result = run_chembl(
            args.project,
            query_id=args.query_id,
            approve=bool(args.approve_chembl),
            assume_yes=bool(args.yes),
            limit=args.limit,
            prep_smiles=bool(args.prep_smiles),
            agent_config=agent_config,
        )
    except ChemblError as exc:
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
    activities = payload.get("activities")
    if isinstance(activities, list) and len(activities) > 5:
        payload["activities_preview"] = activities[:5]
        payload["activities"] = f"<{len(activities)} rows; see activities.csv>"
    return payload
