"""Scientific critic — evaluates structured evidence only."""

from __future__ import annotations

from typing import Any


class CriticAgent:
    """Review workflow results and return PASS / WARNING / FAIL."""

    def __init__(self, agent_config: dict[str, Any] | None = None) -> None:
        critic_cfg = (agent_config or {}).get("critic") or {}
        self.min_valid_fraction = float(critic_cfg.get("min_valid_fraction", 0.80))
        self.max_duplicate_fraction = float(critic_cfg.get("max_duplicate_fraction", 0.25))
        self.min_molecules = int(critic_cfg.get("min_molecules", 1))

    def review(self, results: dict[str, Any]) -> dict[str, Any]:
        issues: list[str] = []
        status = "PASS"

        plan = results.get("plan") or {}
        steps = results.get("steps") or {}
        run = steps.get("run_reinvent") or {}
        analysis = steps.get("analyze_molecules") or {}
        inventory = steps.get("find_output") or {}
        validation = steps.get("validate_project") or {}

        # Hard failures
        if results.get("aborted") and not plan.get("skip_reinvent"):
            status = "FAIL"
            issues.append(results.get("abort_reason") or "Workflow aborted")

        if validation and validation.get("ok") is False and (
            plan.get("approve_run") and not plan.get("skip_reinvent")
        ):
            status = "FAIL"
            issues.extend(validation.get("errors") or ["Project validation failed"])

        if plan.get("approve_run") and not plan.get("skip_reinvent"):
            if run.get("skipped"):
                status = "FAIL"
                issues.append("REINVENT was supposed to run but was skipped")
            elif run.get("success") is False:
                status = "FAIL"
                issues.append(run.get("message") or "REINVENT did not succeed")

        if analysis.get("ok") is False:
            status = "FAIL"
            issues.extend(analysis.get("errors") or ["Molecule analysis failed"])

        total = int(analysis.get("total_molecules") or 0)
        if analysis.get("ok") and total < self.min_molecules:
            status = "FAIL"
            issues.append(
                f"Too few molecules detected ({total} < {self.min_molecules})"
            )

        if inventory.get("ok") and inventory.get("csv_count", 0) == 0 and not analysis.get("ok"):
            status = "FAIL"
            issues.append("No CSV output files found")

        # Warnings (do not invent science — only quantitative thresholds)
        dup_frac = analysis.get("duplicate_fraction")
        if dup_frac is not None and dup_frac > self.max_duplicate_fraction:
            if status == "PASS":
                status = "WARNING"
            issues.append(
                f"Duplicate fraction {dup_frac:.2%} exceeds threshold "
                f"{self.max_duplicate_fraction:.0%}"
            )

        rdkit = analysis.get("rdkit") or {}
        valid_frac = rdkit.get("valid_fraction")
        if valid_frac is not None and valid_frac < self.min_valid_fraction:
            if status == "PASS":
                status = "WARNING"
            issues.append(
                f"Valid molecule fraction {valid_frac:.2%} below threshold "
                f"{self.min_valid_fraction:.0%}"
            )

        if not rdkit.get("available") and analysis.get("ok"):
            if status == "PASS":
                status = "WARNING"
            issues.append("RDKit unavailable; chemical validity not verified")

        if not plan.get("approve_run") and not plan.get("skip_reinvent"):
            if status == "PASS":
                status = "WARNING"
            issues.append(
                "Dry-run only: REINVENT was not executed (pass --approve-run to run)"
            )

        recommendation = {
            "PASS": "Workflow evidence looks consistent; human spot-check recommended.",
            "WARNING": "Human review recommended before trusting downstream conclusions.",
            "FAIL": "Do not treat outputs as successful; fix errors and re-run.",
        }[status]

        # Deduplicate issues while preserving order
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
            "thresholds": {
                "min_valid_fraction": self.min_valid_fraction,
                "max_duplicate_fraction": self.max_duplicate_fraction,
                "min_molecules": self.min_molecules,
            },
        }
