"""REINVENT MVP human-in-the-loop gates: config → run → claim.

Charter gates (2026-10-29):
1. ``approve_config`` — human signed off on the project/preset TOML identity
2. ``approve_run`` — human allowed the predefined ``reinvent`` argv to launch
3. ``approve_claim`` — human allowed report language beyond mechanical counts

This module only evaluates and renders checklist state. It does not launch
REINVENT, write TOML, or invent scientific claims. Approval flags are never
copied by ``--from-run``.
"""

from __future__ import annotations

from typing import Any, Mapping

GATE_IDS: tuple[str, ...] = ("config", "run", "claim")

GATE_SPECS: tuple[dict[str, str], ...] = (
    {
        "id": "config",
        "flag": "approve_config",
        "cli": "--approve-config",
        "title": "Config approve",
        "meaning": (
            "PI signed off on the human-written project/preset reinvent.toml "
            "identity before launch. Agents may only select a preset ID."
        ),
    },
    {
        "id": "run",
        "flag": "approve_run",
        "cli": "--approve-run",
        "title": "Run approve",
        "meaning": (
            "PI allowed the predefined reinvent command to launch "
            "(interactive confirm or --yes)."
        ),
    },
    {
        "id": "claim",
        "flag": "approve_claim",
        "cli": "--approve-claim",
        "title": "Claim approve",
        "meaning": (
            "PI allowed interpretive / outbound language about this run. "
            "Without it, counts are mechanical only — not a discovery claim."
        ),
    },
)


def evaluate_hitl_gates(
    plan: Mapping[str, Any] | None = None,
    *,
    approve_config: bool | None = None,
    approve_run: bool | None = None,
    approve_claim: bool | None = None,
    skip_reinvent: bool | None = None,
) -> dict[str, Any]:
    """Return a structured checklist for CLI / HTML / result.json."""
    plan = plan or {}
    if approve_config is None:
        approve_config = bool(plan.get("approve_config"))
    if approve_run is None:
        approve_run = bool(plan.get("approve_run"))
    if approve_claim is None:
        approve_claim = bool(plan.get("approve_claim"))
    if skip_reinvent is None:
        skip_reinvent = bool(plan.get("skip_reinvent"))

    flags = {
        "config": bool(approve_config),
        "run": bool(approve_run),
        "claim": bool(approve_claim),
    }

    gates: list[dict[str, Any]] = []
    for spec in GATE_SPECS:
        gid = spec["id"]
        approved = flags[gid]
        note = None
        if gid == "run" and skip_reinvent:
            note = "Offline / --skip-reinvent: run gate not required for launch."
        elif gid == "config" and skip_reinvent and not approved:
            note = (
                "Offline analysis: config gate still recommended when citing "
                "which TOML / preset the CSV came from."
            )
        elif gid == "claim" and not approved:
            note = "Report must not be treated as a discovery or wet-lab claim."
        gates.append(
            {
                "id": gid,
                "flag": spec["flag"],
                "cli": spec["cli"],
                "title": spec["title"],
                "meaning": spec["meaning"],
                "approved": approved,
                "status": "approved" if approved else "pending",
                "note": note,
            }
        )

    missing = [g["id"] for g in gates if not g["approved"]]
    # Soft MVP policy: run still only hard-requires --approve-run (existing).
    # Config/claim are recorded + warned until PI hardens them.
    hard_block_run = bool(approve_run) and not skip_reinvent and not approve_config
    return {
        "schema": "reinvent-hitl-gates-v1",
        "gates": gates,
        "approved_count": sum(1 for g in gates if g["approved"]),
        "total": len(gates),
        "all_approved": not missing,
        "missing": missing,
        "skip_reinvent": bool(skip_reinvent),
        "soft_policy": True,
        "hard_block_run_without_config": False,
        "warn_run_without_config": hard_block_run,
        "claims_human_approved": bool(approve_claim),
        "summary": (
            "All three HITL gates approved."
            if not missing
            else f"Pending HITL gates: {', '.join(missing)}."
        ),
    }


def attach_hitl_to_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Mutate plan with a fresh checklist derived from its approval flags."""
    checklist = evaluate_hitl_gates(plan)
    plan["hitl"] = checklist
    return plan


def claim_banner_text(checklist: Mapping[str, Any] | None) -> str | None:
    """HTML/CLI banner when claim gate is still pending."""
    if not checklist:
        return (
            "Claim gate unknown: treat molecule counts as mechanical only; "
            "not a discovery or wet-lab recommendation."
        )
    if checklist.get("claims_human_approved"):
        return None
    return (
        "Claims unapproved (--approve-claim missing): this report is an internal "
        "run artefact. Do not cite as a drug discovery, wet-lab advice, or "
        "publication-ready conclusion."
    )
