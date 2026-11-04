"""ADMET coarse-screen CLI — separate from the REINVENT executor.

``python -m tools.admet --approve-admet`` signs the rule table and writes
``admet_scores.csv``. Without that flag the module validates the preset and
input CSV, computes a dry-run preview in ``admet_result.json``, and does
**not** write the scores CSV. Offline RDKit / rule-based only; never invents
docking, MD, or assay numbers. Missing RDKit is a loud failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from tools import dumps_pretty, load_agent_config
from tools.admet.presets import (
    AdmetPresetError,
    list_preset_ids,
    preset_catalog,
    resolve_preset,
)
from tools.admet.score import (
    AdmetScoreError,
    load_molecule_rows,
    score_molecules,
    write_scores_csv,
)

OUTPUT_SUBDIR = "admet"


class AdmetError(ValueError):
    """Sandbox, preset, or approval failure (not invented scores)."""


def run_admet(
    project_dir: str | Path,
    *,
    molecules_csv: str | Path,
    preset_id: str,
    approve: bool = False,
    assume_yes: bool = False,
    agent_config: dict[str, Any] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> dict[str, Any]:
    """Validate preset + CSV; optionally write ADMET scores under output/admet/."""
    stdout = sys.stdout if stdout is None else stdout
    _ = agent_config  # reserved for future admet defaults; unused in MVP

    try:
        preset = resolve_preset(preset_id)
    except AdmetPresetError as exc:
        raise AdmetError(str(exc)) from exc

    project = Path(project_dir).expanduser().resolve()
    if not project.is_dir():
        raise AdmetError(f"Project directory does not exist: {project}")

    csv_path = Path(molecules_csv).expanduser().resolve()
    try:
        csv_path.relative_to(project)
    except ValueError as exc:
        raise AdmetError(
            f"Molecules CSV must stay under the project directory: {csv_path}"
        ) from exc

    output_dir = _admet_output_dir(project)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows, columns = load_molecule_rows(csv_path)
        scored = score_molecules(rows, preset)
    except AdmetScoreError as exc:
        raise AdmetError(str(exc)) from exc

    n_valid = sum(1 for r in scored if r.get("valid"))
    n_pass = sum(1 for r in scored if r.get("preset_pass"))
    preview = scored[:5]

    planned = {
        "preset_id": preset["id"],
        "rules": dict(preset["rules"]),
        "molecules_csv": str(csv_path),
        "columns_mapped": columns,
        "n_input_rows": len(rows),
        "n_valid": n_valid,
        "n_pass": n_pass,
        "output_dir": str(output_dir),
        "invents_scores": False,
        "llm_scorer": False,
        "docking": False,
        "md": False,
        "assay": False,
        "engine": "rdkit_rules",
    }

    result: dict[str, Any] = {
        "ok": False,
        "approved": bool(approve),
        "skipped": False,
        "success": False,
        "source": "admet",
        "modality": "admet",
        "kind": "admet coarse screen",
        "project_dir": str(project),
        "output_dir": str(output_dir),
        "preset_id": preset["id"],
        "preset": {
            "id": preset["id"],
            "description": preset.get("description"),
            "rules": dict(preset["rules"]),
        },
        "molecules_csv": str(csv_path),
        "columns_mapped": columns,
        "n_input_rows": len(rows),
        "n_valid": n_valid,
        "n_pass": n_pass,
        "n_scored": len(scored),
        "scores_csv": None,
        "table_present": False,
        "scores_preview": preview,
        "planned": planned,
        "warnings": [],
        "errors": [],
        "message": "",
        "invents_scores": False,
        "llm_scorer": False,
        "engine": "rdkit_rules",
    }

    if not approve:
        result["ok"] = True
        result["skipped"] = True
        result["message"] = (
            "ADMET coarse screen skipped (pass --approve-admet to sign "
            f"and write admet_scores.csv under {output_dir}). "
            "No docking/MD/assay numbers were invented; dry-run preview is "
            "in admet_result.json only."
        )
        result["warnings"].append(result["message"])
        _write_result_json(output_dir, result)
        return result

    if not confirm_admet_sign(
        planned,
        n_scored=len(scored),
        assume_yes=assume_yes,
        stdin=stdin,
        stdout=stdout,
    ):
        result["ok"] = True
        result["skipped"] = True
        result["approved"] = False
        result["message"] = (
            "ADMET coarse screen not confirmed. No admet_scores.csv written."
        )
        result["warnings"].append(result["message"])
        _write_result_json(output_dir, result)
        return result

    scores_path = write_scores_csv(output_dir / "admet_scores.csv", scored)
    result["ok"] = True
    result["success"] = True
    result["skipped"] = False
    result["approved"] = True
    result["table_present"] = True
    result["scores_csv"] = str(scores_path)
    result["message"] = (
        f"Wrote ADMET scores ({len(scored)} rows, {n_pass} pass / {n_valid} valid) "
        f"to {scores_path}"
    )
    if n_valid == 0:
        result["warnings"].append(
            "No valid SMILES after RDKit parse. No descriptor values were invented."
        )
    elif n_pass == 0:
        result["warnings"].append(
            "No molecules passed the preset rules. Empty pass set is not a drug claim."
        )
    _write_result_json(output_dir, result)
    return result


def confirm_admet_sign(
    planned: dict[str, Any],
    *,
    n_scored: int,
    assume_yes: bool = False,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> bool:
    """Print the rule preset and require interactive confirmation."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    print("\n[Human approval required — ADMET coarse screen]", file=stdout)
    print("The agent is ready to sign this offline rule-based ADMET table:", file=stdout)
    print(f"  preset_id:      {planned.get('preset_id')}", file=stdout)
    print(f"  rules:          {planned.get('rules')}", file=stdout)
    print(f"  molecules_csv:  {planned.get('molecules_csv')}", file=stdout)
    print(f"  columns:        {planned.get('columns_mapped')}", file=stdout)
    print(f"  n_input_rows:   {planned.get('n_input_rows')}", file=stdout)
    print(f"  n_valid:        {planned.get('n_valid')}", file=stdout)
    print(f"  n_pass:         {planned.get('n_pass')}", file=stdout)
    print(f"  n_scored:       {n_scored}", file=stdout)
    print(f"  output:         {planned.get('output_dir')}", file=stdout)
    print(
        "RDKit descriptors + human rule thresholds only. Never invents docking, "
        "MD, IC50, or assay values. Not a wet-lab druglikeness claim.",
        file=stdout,
    )
    print(
        "Independent of REINVENT (--approve-run / --approve-dock / --approve-md "
        "/ --approve-literature / --approve-chembl / --approve-portfolio do not "
        "apply here).",
        file=stdout,
    )
    if assume_yes:
        print("Proceeding due to --yes.", file=stdout)
        return True
    if not hasattr(stdin, "isatty") or not stdin.isatty():
        print(
            "Non-interactive stdin: re-run with --approve-admet --yes to sign.",
            file=stdout,
        )
        return False
    print("Sign ADMET scores? [y/N]: ", end="", file=stdout, flush=True)
    reply = stdin.readline().strip().lower()
    return reply in ("y", "yes")


def _admet_output_dir(project: Path) -> Path:
    dest = (project / "output" / OUTPUT_SUBDIR).resolve()
    try:
        dest.relative_to(project.resolve())
    except ValueError as exc:
        raise AdmetError(f"ADMET output escapes project: {dest}") from exc
    return dest


def _write_result_json(output_dir: Path, result: dict[str, Any]) -> None:
    path = output_dir / "admet_result.json"
    compact = dict(result)
    preview = compact.get("scores_preview")
    if isinstance(preview, list) and len(preview) > 10:
        compact["scores_preview"] = preview[:10]
    path.write_text(dumps_pretty(compact), encoding="utf-8")
    result["result_json"] = str(path)


def _critic_payload(admet: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": {
            "approve_run": False,
            "skip_reinvent": True,
            "approve_admet": admet.get("approved"),
            "steps": ["admet"],
        },
        "steps": {"admet": admet},
        "warnings": admet.get("warnings") or [],
        "errors": admet.get("errors") or [],
        "aborted": bool(admet.get("errors")) and not admet.get("skipped"),
        "abort_reason": admet.get("message") if admet.get("errors") else None,
        "project_dir": admet.get("project_dir"),
    }


def build_parser() -> argparse.ArgumentParser:
    allowed = ", ".join(list_preset_ids())
    parser = argparse.ArgumentParser(
        description=(
            "Independent ADMET coarse screen. Human-written rule preset ids "
            "only (offline RDKit descriptors). Not part of the REINVENT "
            "executor. Requires --approve-admet to write admet_scores.csv. "
            "Never invents docking/MD/assay numbers."
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
        help="Existing molecules CSV under the project (needs a SMILES column)",
    )
    parser.add_argument(
        "--preset-id",
        required=False,
        default=None,
        help=f"Human-written rule preset id. Allowed: {allowed}",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print the rule preset catalog as JSON and exit",
    )
    parser.add_argument(
        "--approve-admet",
        action="store_true",
        help="Human approval to sign and write admet_scores.csv",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive Sign ADMET scores? prompt when used with --approve-admet",
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
        result = run_admet(
            args.project,
            molecules_csv=args.molecules_csv,
            preset_id=args.preset_id,
            approve=bool(args.approve_admet),
            assume_yes=bool(args.yes),
            agent_config=agent_config,
        )
    except AdmetError as exc:
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
    preview = payload.get("scores_preview")
    if isinstance(preview, list) and len(preview) > 5:
        payload["scores_preview"] = preview[:5]
        payload["scores_preview_omitted"] = len(preview) - 5
    return payload
