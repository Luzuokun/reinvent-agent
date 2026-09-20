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
        raw_qed = critic_cfg.get("min_mean_qed")
        self.min_mean_qed: float | None
        if raw_qed is None or raw_qed == "":
            self.min_mean_qed = None
        else:
            self.min_mean_qed = float(raw_qed)

    def review(self, results: dict[str, Any]) -> dict[str, Any]:
        issues: list[str] = []
        status = "PASS"

        plan = results.get("plan") or {}
        steps = results.get("steps") or {}
        run = steps.get("run_reinvent") or {}
        analysis = steps.get("analyze_molecules") or {}
        inventory = steps.get("find_output") or {}
        validation = steps.get("validate_project") or {}
        details = validation.get("details") if isinstance(validation.get("details"), dict) else {}
        run_type = str(details.get("run_type") or "").strip().lower()
        is_tl = run_type == "transfer_learning"

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

        model_count = int(inventory.get("model_count") or 0)
        csv_count = int(inventory.get("csv_count") or 0)
        if is_tl:
            if (
                plan.get("approve_run")
                and not plan.get("skip_reinvent")
                and run.get("success")
                and model_count == 0
            ):
                status = "FAIL"
                issues.append("Transfer learning produced no model checkpoint")
        elif inventory.get("ok") and csv_count == 0 and not analysis.get("ok"):
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

        qed_stats = rdkit.get("qed") if isinstance(rdkit.get("qed"), dict) else {}
        qed_mean = qed_stats.get("mean") if isinstance(qed_stats, dict) else None
        if self.min_mean_qed is not None:
            if qed_mean is None:
                if status == "PASS":
                    status = "WARNING"
                issues.append(
                    "min_mean_qed is configured but QED mean is unavailable; "
                    "threshold not applied (no value invented)"
                )
            elif qed_mean < self.min_mean_qed:
                if status == "PASS":
                    status = "WARNING"
                issues.append(
                    f"Mean QED {qed_mean:.4f} below threshold {self.min_mean_qed:.4f}"
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

        if is_tl and analysis.get("ok") and analysis.get("analysis_source") == "tl_training_set":
            successful_tl = bool(
                plan.get("approve_run")
                and not plan.get("skip_reinvent")
                and run.get("success")
            )
            if not successful_tl and status == "PASS":
                status = "WARNING"
            issues.append(
                "Molecule stats are from the TL training SMILES, not molecules sampled from the new model"
            )

        docking = steps.get("docking") if isinstance(steps.get("docking"), dict) else None
        if docking is None and isinstance(results.get("docking"), dict):
            docking = results["docking"]
        if docking:
            status = _review_docking(docking, issues=issues, status=status)

        md = steps.get("md") if isinstance(steps.get("md"), dict) else None
        if md is None and isinstance(results.get("md"), dict):
            md = results["md"]
        if md:
            status = _review_md(md, issues=issues, status=status)

        literature = (
            steps.get("literature") if isinstance(steps.get("literature"), dict) else None
        )
        if literature is None and isinstance(results.get("literature"), dict):
            literature = results["literature"]
        if literature:
            status = _review_literature(literature, issues=issues, status=status)

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
                "min_mean_qed": self.min_mean_qed,
            },
        }


def _review_docking(
    docking: dict[str, Any],
    *,
    issues: list[str],
    status: str,
) -> str:
    """Evaluate docking evidence when a score table (or failure) is attached.

    Called only if the docking module put a ``docking`` object on the results.
    REINVENT-only runs never hit this branch, so docking claims stay forbidden
    unless that evidence is present.
    """
    approved = bool(docking.get("approved"))
    skipped = bool(docking.get("skipped"))
    table_present = bool(
        docking.get("table_present")
        or docking.get("scores_csv")
        or (isinstance(docking.get("score"), dict) and docking.get("score"))
    )
    n_scored = int(docking.get("n_scored") or 0)
    errors = docking.get("errors") or []

    if skipped or not approved:
        if status == "PASS":
            status = "WARNING"
        issues.append(
            "Docking was not executed (pass --approve-dock to run the independent module)"
        )
        return status

    if docking.get("ok") is False or (errors and not table_present):
        status = "FAIL"
        if errors:
            issues.extend(str(e) for e in errors if e)
        else:
            issues.append(docking.get("message") or "Docking failed")
        return status

    if not table_present:
        status = "FAIL"
        issues.append("Docking ran but no score table was written under output/docking/")
        return status

    if n_scored <= 0:
        status = "FAIL"
        issues.append("Docking score table is present but contains no scores")
        return status

    best = None
    score_block = docking.get("score")
    if isinstance(score_block, dict):
        best = score_block.get("best")
    if best is not None:
        issues.append(
            f"Docking score table present (n={n_scored}, best={best}, "
            f"engine={docking.get('engine')})"
        )
    return status


def _review_md(
    md: dict[str, Any],
    *,
    issues: list[str],
    status: str,
) -> str:
    """Evaluate MD evidence when an RMSD/RMSF table (or failure) is attached.

    Called only if the MD module put an ``md`` object on the results.
    REINVENT-only runs never hit this branch, so GROMACS claims stay forbidden
    unless that evidence is present.
    """
    approved = bool(md.get("approved"))
    skipped = bool(md.get("skipped"))
    table_present = bool(
        md.get("table_present")
        or md.get("rmsd_csv")
        or md.get("rmsf_csv")
        or (isinstance(md.get("rmsd"), dict) and md.get("rmsd"))
    )
    n_frames = int(md.get("n_frames") or 0)
    errors = md.get("errors") or []

    if skipped or not approved:
        if status == "PASS":
            status = "WARNING"
        issues.append(
            "MD was not executed (pass --approve-md to run the independent module)"
        )
        return status

    if md.get("ok") is False or (errors and not table_present):
        status = "FAIL"
        if errors:
            issues.extend(str(e) for e in errors if e)
        else:
            issues.append(md.get("message") or "MD failed")
        return status

    if not table_present:
        status = "FAIL"
        issues.append("MD ran but no RMSD/RMSF table was written under output/md/")
        return status

    rmsd_block = md.get("rmsd") if isinstance(md.get("rmsd"), dict) else {}
    mean = rmsd_block.get("mean") if isinstance(rmsd_block, dict) else None
    detail = f"n_frames={n_frames}"
    if mean is not None:
        detail += f", rmsd_mean={mean}"
    issues.append(
        f"MD table present ({detail}, protocol={md.get('protocol')}, engine=gmx)"
    )
    return status


def _review_literature(
    literature: dict[str, Any],
    *,
    issues: list[str],
    status: str,
) -> str:
    """Evaluate literature evidence when the independent research tool attached one.

    Called only if the literature module put a ``literature`` object on the
    results. REINVENT-only runs never hit this branch, so "literature shows"
    stays forbidden unless sourced URL/PMID/DOI entries are present.
    """
    approved = bool(literature.get("approved"))
    skipped = bool(literature.get("skipped"))
    entries = literature.get("entries") if isinstance(literature.get("entries"), list) else []
    sourced_n = 0
    for item in entries:
        if not isinstance(item, dict):
            continue
        pmid = str(item.get("pmid") or "").strip()
        doi = str(item.get("doi") or "").strip()
        url = str(item.get("url") or "").strip()
        if pmid.isdigit() or doi or url.startswith("http://") or url.startswith("https://"):
            sourced_n += 1
    table_present = bool(literature.get("table_present") and sourced_n > 0)
    errors = literature.get("errors") or []

    if skipped or not approved:
        if status == "PASS":
            status = "WARNING"
        issues.append(
            "Literature was not fetched (pass --approve-literature to query PubMed). "
            "No citations were invented."
        )
        return status

    if literature.get("ok") is False or errors:
        status = "FAIL"
        if errors:
            issues.extend(str(e) for e in errors if e)
        else:
            issues.append(literature.get("message") or "Literature search failed")
        issues.append("Refusing unsourced literature claims; network failure is not a citation.")
        return status

    if not table_present:
        if status == "PASS":
            status = "WARNING"
        issues.append(
            "PubMed returned no sourced hits (URL/PMID/DOI required). "
            "Do not treat this as a literature review."
        )
        return status

    issues.append(
        f"Literature table present (n={sourced_n} sourced PubMed entries with URL/PMID/DOI)"
    )
    return status
