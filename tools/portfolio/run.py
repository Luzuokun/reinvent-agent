"""Portfolio shortlist + campaign one-pager CLI — separate from REINVENT.

``python -m tools.portfolio --approve-portfolio`` signs the shortlist.
Without that flag the module validates the weight preset and input CSV,
computes a dry-run preview in ``portfolio_result.json``, and does **not**
write ``shortlist.csv`` / the one-pager / the run index. No LLM scorer;
missing modality columns are skipped, never invented.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from tools import dumps_pretty, load_agent_config
from tools.portfolio.index import append_run_index
from tools.portfolio.one_pager import write_campaign_one_pager
from tools.portfolio.presets import (
    PortfolioPresetError,
    list_preset_ids,
    preset_catalog,
    resolve_preset,
)
from tools.portfolio.rank import (
    PortfolioRankError,
    assert_required_columns,
    load_molecule_rows,
    rank_molecules,
    write_shortlist_csv,
)

OUTPUT_SUBDIR = "portfolio"


class PortfolioError(ValueError):
    """Sandbox, preset, or approval failure (not invented scores)."""


def run_portfolio(
    project_dir: str | Path,
    *,
    molecules_csv: str | Path,
    preset_id: str,
    approve: bool = False,
    assume_yes: bool = False,
    top_n: int | None = None,
    campaign: dict[str, Any] | None = None,
    agent_config: dict[str, Any] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> dict[str, Any]:
    """Validate preset + CSV; optionally write shortlist + one-pager + index."""
    stdout = sys.stdout if stdout is None else stdout
    _ = agent_config  # reserved for future portfolio defaults; unused in MVP

    try:
        preset = resolve_preset(preset_id)
    except PortfolioPresetError as exc:
        raise PortfolioError(str(exc)) from exc

    if top_n is not None:
        if int(top_n) < 0:
            raise PortfolioError("--top must be >= 0 (0 = all)")
        preset["top_n"] = int(top_n)

    project = Path(project_dir).expanduser().resolve()
    if not project.is_dir():
        raise PortfolioError(f"Project directory does not exist: {project}")

    csv_path = Path(molecules_csv).expanduser().resolve()
    try:
        csv_path.relative_to(project)
    except ValueError as exc:
        raise PortfolioError(
            f"Molecules CSV must stay under the project directory: {csv_path}"
        ) from exc

    output_dir = _portfolio_output_dir(project)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows, columns = load_molecule_rows(csv_path)
        assert_required_columns(columns, preset)
    except PortfolioRankError as exc:
        raise PortfolioError(str(exc)) from exc

    planned = {
        "preset_id": preset["id"],
        "weights": dict(preset["weights"]),
        "drop_pains": bool(preset.get("drop_pains", True)),
        "top_n": preset.get("top_n"),
        "require_fields": list(preset.get("require_fields") or ()),
        "molecules_csv": str(csv_path),
        "columns_mapped": columns,
        "n_input_rows": len(rows),
        "output_dir": str(output_dir),
        "invents_scores": False,
        "llm_scorer": False,
    }

    # Preview ranking even on dry-run (in memory / result JSON only).
    ranked = rank_molecules(rows, preset)
    preview = ranked[:5]

    result: dict[str, Any] = {
        "ok": False,
        "approved": bool(approve),
        "skipped": False,
        "success": False,
        "source": "portfolio",
        "modality": "decision",
        "kind": "portfolio shortlist",
        "project_dir": str(project),
        "output_dir": str(output_dir),
        "preset_id": preset["id"],
        "preset": {
            "id": preset["id"],
            "description": preset.get("description"),
            "weights": dict(preset["weights"]),
            "drop_pains": bool(preset.get("drop_pains", True)),
            "top_n": preset.get("top_n"),
        },
        "molecules_csv": str(csv_path),
        "columns_mapped": columns,
        "n_input_rows": len(rows),
        "n_shortlisted": len(ranked),
        "shortlist_csv": None,
        "one_pager_html": None,
        "run_index": None,
        "table_present": False,
        "shortlist_preview": preview,
        "planned": planned,
        "campaign": dict(campaign or {}),
        "warnings": [],
        "errors": [],
        "message": "",
        "invents_scores": False,
        "llm_scorer": False,
        "crm": False,
    }

    if not approve:
        result["ok"] = True
        result["skipped"] = True
        result["message"] = (
            "Portfolio shortlist skipped (pass --approve-portfolio to sign "
            f"and write shortlist.csv under {output_dir}). "
            "No scores were invented; dry-run preview is in portfolio_result.json only."
        )
        result["warnings"].append(result["message"])
        _write_result_json(output_dir, result)
        return result

    if not confirm_portfolio_sign(
        planned,
        n_shortlisted=len(ranked),
        assume_yes=assume_yes,
        stdin=stdin,
        stdout=stdout,
    ):
        result["ok"] = True
        result["skipped"] = True
        result["approved"] = False
        result["message"] = (
            "Portfolio shortlist not confirmed. No shortlist.csv / one-pager written."
        )
        result["warnings"].append(result["message"])
        _write_result_json(output_dir, result)
        return result

    shortlist_path = write_shortlist_csv(output_dir / "shortlist.csv", ranked)
    one_pager = write_campaign_one_pager(
        output_dir / "campaign_one_pager.html",
        project_dir=project,
        preset_id=preset["id"],
        shortlist_rows=ranked,
        shortlist_csv=shortlist_path,
        campaign=campaign,
        approved=True,
        warnings=result["warnings"],
    )
    index_path = append_run_index(
        output_dir,
        {
            "project_dir": str(project),
            "preset_id": preset["id"],
            "molecules_csv": str(csv_path),
            "shortlist_csv": str(shortlist_path),
            "one_pager_html": str(one_pager),
            "n_shortlisted": len(ranked),
            "approved": True,
            "campaign_id": (campaign or {}).get("campaign_id")
            or (campaign or {}).get("id"),
            "target": (campaign or {}).get("target"),
        },
    )

    result["ok"] = True
    result["success"] = True
    result["skipped"] = False
    result["approved"] = True
    result["table_present"] = True
    result["shortlist_csv"] = str(shortlist_path)
    result["one_pager_html"] = str(one_pager)
    result["run_index"] = str(index_path)
    result["n_shortlisted"] = len(ranked)
    result["message"] = (
        f"Wrote shortlist ({len(ranked)} rows) to {shortlist_path} "
        f"and campaign one-pager to {one_pager}"
    )
    if not ranked:
        result["warnings"].append(
            "Shortlist is empty after filters (PAINS drop / missing required "
            "fields / no scorable rows). No numbers were invented."
        )
    _write_result_json(output_dir, result)
    return result


def confirm_portfolio_sign(
    planned: dict[str, Any],
    *,
    n_shortlisted: int,
    assume_yes: bool = False,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> bool:
    """Print the weight preset and require interactive confirmation."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    print("\n[Human approval required — portfolio shortlist]", file=stdout)
    print("The agent is ready to sign this cross-modal shortlist:", file=stdout)
    print(f"  preset_id:      {planned.get('preset_id')}", file=stdout)
    print(f"  weights:        {planned.get('weights')}", file=stdout)
    print(f"  drop_pains:     {planned.get('drop_pains')}", file=stdout)
    print(f"  top_n:          {planned.get('top_n')}", file=stdout)
    print(f"  molecules_csv:  {planned.get('molecules_csv')}", file=stdout)
    print(f"  columns:        {planned.get('columns_mapped')}", file=stdout)
    print(f"  n_input_rows:   {planned.get('n_input_rows')}", file=stdout)
    print(f"  n_shortlisted:  {n_shortlisted}", file=stdout)
    print(f"  output:         {planned.get('output_dir')}", file=stdout)
    print(
        "Deterministic weighted sum only. Missing modality cells are skipped; "
        "scores are never invented. No LLM scorer.",
        file=stdout,
    )
    print(
        "Independent of REINVENT (--approve-run / --approve-dock / --approve-md "
        "/ --approve-literature / --approve-chembl do not apply here).",
        file=stdout,
    )
    if assume_yes:
        print("Proceeding due to --yes.", file=stdout)
        return True
    if not hasattr(stdin, "isatty") or not stdin.isatty():
        print(
            "Non-interactive stdin: re-run with --approve-portfolio --yes to sign.",
            file=stdout,
        )
        return False
    print("Sign shortlist? [y/N]: ", end="", file=stdout, flush=True)
    reply = stdin.readline().strip().lower()
    return reply in ("y", "yes")


def _portfolio_output_dir(project: Path) -> Path:
    dest = (project / "output" / OUTPUT_SUBDIR).resolve()
    try:
        dest.relative_to(project.resolve())
    except ValueError as exc:
        raise PortfolioError(f"Portfolio output escapes project: {dest}") from exc
    return dest


def _write_result_json(output_dir: Path, result: dict[str, Any]) -> None:
    path = output_dir / "portfolio_result.json"
    compact = dict(result)
    preview = compact.get("shortlist_preview")
    if isinstance(preview, list) and len(preview) > 10:
        compact["shortlist_preview"] = preview[:10]
    path.write_text(dumps_pretty(compact), encoding="utf-8")
    result["result_json"] = str(path)


def _critic_payload(portfolio: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": {
            "approve_run": False,
            "skip_reinvent": True,
            "approve_portfolio": portfolio.get("approved"),
            "steps": ["portfolio"],
        },
        "steps": {"portfolio": portfolio},
        "warnings": portfolio.get("warnings") or [],
        "errors": portfolio.get("errors") or [],
        "aborted": bool(portfolio.get("errors")) and not portfolio.get("skipped"),
        "abort_reason": portfolio.get("message") if portfolio.get("errors") else None,
        "project_dir": portfolio.get("project_dir"),
    }


def _parse_campaign_args(args: argparse.Namespace) -> dict[str, Any]:
    campaign: dict[str, Any] = {}
    if args.campaign_json:
        path = Path(args.campaign_json).expanduser()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PortfolioError(f"Cannot read --campaign-json: {exc}") from exc
        if not isinstance(raw, dict):
            raise PortfolioError("--campaign-json must be a JSON object")
        campaign.update(raw)
    if args.target:
        campaign["target"] = args.target
    if args.protocols:
        campaign["protocols"] = args.protocols
    if args.red_flags:
        campaign["red_flags"] = args.red_flags
    if args.next_steps:
        campaign["next_steps"] = args.next_steps
    if args.campaign_id:
        campaign["campaign_id"] = args.campaign_id
    return campaign


def build_parser() -> argparse.ArgumentParser:
    allowed = ", ".join(list_preset_ids())
    parser = argparse.ArgumentParser(
        description=(
            "Independent portfolio shortlist + campaign one-pager. "
            "Human-written weight preset ids only (no LLM scorer). "
            "Not part of the REINVENT executor. Requires --approve-portfolio "
            "to write shortlist.csv. Never invents QED/SA/dock/MD numbers."
        )
    )
    parser.add_argument(
        "--project",
        required=False,
        default=None,
        help="Project directory (required unless --list-presets)",
    )
    parser.add_argument(
        "--molecules-csv",
        required=False,
        default=None,
        help=(
            "Existing molecules CSV under the project (QED/SA/PAINS/dock/MD "
            "columns used when present)"
        ),
    )
    parser.add_argument(
        "--preset-id",
        required=False,
        default=None,
        help=f"Human-written weight preset id. Allowed: {allowed}",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print the weight preset catalog as JSON and exit",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Override preset top_n (0 = keep all after filters)",
    )
    parser.add_argument(
        "--approve-portfolio",
        action="store_true",
        help="Human approval to sign and write shortlist.csv + one-pager",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive Sign shortlist? prompt when used with --approve-portfolio",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Campaign one-pager: 靶点 / target (optional; not invented if omitted)",
    )
    parser.add_argument(
        "--protocols",
        default=None,
        help="Campaign one-pager: 批准协议 / approved protocols",
    )
    parser.add_argument(
        "--red-flags",
        default=None,
        help="Campaign one-pager: 红旗 / red flags",
    )
    parser.add_argument(
        "--next-steps",
        default=None,
        help="Campaign one-pager: 下一步 / next steps",
    )
    parser.add_argument(
        "--campaign-id",
        default=None,
        help="Optional local campaign id for the run index (not CRM)",
    )
    parser.add_argument(
        "--campaign-json",
        default=None,
        help="Optional JSON object with target/protocols/red_flags/next_steps",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional agent.yaml (unused for scoring; never TOML)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_presets:
        print(json.dumps(preset_catalog(), indent=2, ensure_ascii=False))
        return 0
    if not args.project:
        print("ERROR: --project is required (or pass --list-presets).")
        return 2
    if not args.molecules_csv:
        print("ERROR: --molecules-csv is required (or pass --list-presets).")
        return 2
    if not args.preset_id:
        print(
            "ERROR: --preset-id is required (or pass --list-presets). "
            f"Allowed: {', '.join(list_preset_ids())}"
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
        campaign = _parse_campaign_args(args)
        result = run_portfolio(
            args.project,
            molecules_csv=args.molecules_csv,
            preset_id=args.preset_id,
            approve=bool(args.approve_portfolio),
            assume_yes=bool(args.yes),
            top_n=args.top,
            campaign=campaign,
            agent_config=agent_config,
        )
    except PortfolioError as exc:
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
    preview = payload.get("shortlist_preview")
    if isinstance(preview, list) and len(preview) > 5:
        payload["shortlist_preview"] = preview[:5]
        payload["shortlist_preview_omitted"] = len(preview) - 5
    return payload
