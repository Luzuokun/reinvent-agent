"""Deterministic execution agent — calls predefined tools only."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from analysis.molecule_analysis import analyze_molecules
from analysis.report import generate_html_report
from agents.plan_schema import PlanValidationError, resolve_plan_csv_path
from tools.environment import check_environment
from tools.files import validate_project
from tools.reinvent import find_output_files, run_reinvent

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """Execute an approved plan using validated tools."""

    def __init__(self, agent_config: dict[str, Any] | None = None) -> None:
        self.agent_config = agent_config or {}

    def execute(
        self,
        plan: dict[str, Any],
        *,
        project_dir: str | Path,
        approve_run: bool,
        csv_override: str | Path | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        project_dir = Path(project_dir).expanduser().resolve()
        config_name = self.agent_config.get("project", {}).get("config_name", "reinvent.toml")
        output_dirname = self.agent_config.get("project", {}).get("output_dirname", "output")
        config_path = project_dir / config_name
        output_dir = project_dir / output_dirname

        results: dict[str, Any] = {
            "plan": plan,
            "project_dir": str(project_dir),
            "steps": {},
            "warnings": [],
            "errors": [],
        }

        for raw_step in plan.get("steps", []):
            step = raw_step["name"] if isinstance(raw_step, dict) else raw_step
            if not isinstance(step, str):
                results["warnings"].append(f"Invalid plan step skipped: {raw_step!r}")
                continue
            logger.info("Executing step: %s", step)
            if step == "check_environment":
                env = check_environment()
                results["steps"]["check_environment"] = env
                if not env.get("reinvent") and not plan.get("skip_reinvent"):
                    results["warnings"].append("reinvent executable not found on PATH")
                if not env.get("rdkit"):
                    results["warnings"].append("RDKit not available in current Python")

            elif step == "validate_project":
                validation = validate_project(
                    project_dir, agent_config=self.agent_config
                )
                results["steps"]["validate_project"] = validation
                if not validation.get("ok"):
                    results["errors"].extend(validation.get("errors") or [])
                    # Still continue for dry-run visibility unless running reinvent
                    if approve_run and not plan.get("skip_reinvent"):
                        results["aborted"] = True
                        results["abort_reason"] = "Project validation failed"
                        break

            elif step == "prepare_execution":
                results["steps"]["prepare_execution"] = {
                    "config_path": str(config_path),
                    "output_dir": str(output_dir),
                    "approve_run": approve_run,
                    "ready": config_path.is_file(),
                }

            elif step == "run_reinvent":
                if plan.get("skip_reinvent"):
                    results["steps"]["run_reinvent"] = {
                        "approved": False,
                        "skipped": True,
                        "success": False,
                        "exit_code": None,
                        "runtime_seconds": 0.0,
                        "command": [],
                        "message": "REINVENT skipped (--skip-reinvent); plan step ignored.",
                    }
                    continue
                run_result = run_reinvent(
                    config_path,
                    approve=approve_run,
                    seed=seed,
                    project_dir=project_dir,
                    agent_config=self.agent_config,
                )
                results["steps"]["run_reinvent"] = run_result
                if approve_run and not run_result.get("success"):
                    results["errors"].append(run_result.get("message", "REINVENT failed"))
                    results["aborted"] = True
                    results["abort_reason"] = "REINVENT run failed"
                    break

            elif step == "find_output":
                extra_dirs: list[Path] = []
                models_dir = project_dir / "models"
                if models_dir.is_dir():
                    extra_dirs.append(models_dir)
                inventory = find_output_files(
                    output_dir,
                    project_dir=project_dir,
                    extra_dirs=extra_dirs,
                )
                results["steps"]["find_output"] = inventory

            elif step == "analyze_molecules":
                self._analyze_molecules_step(
                    results,
                    plan=plan,
                    csv_override=csv_override,
                    config_path=config_path,
                    project_dir=project_dir,
                    output_dir=output_dir,
                )

            elif step == "generate_report":
                # Filled after critic in main, or here with partial payload
                results["steps"]["generate_report"] = {"pending": True}

            elif step == "critic_review":
                results["steps"]["critic_review"] = {"pending": True}

            else:
                results["warnings"].append(f"Unknown plan step skipped: {step}")

        return results

    def write_report(
        self,
        results: dict[str, Any],
        *,
        goal: str,
        critic: dict[str, Any],
    ) -> dict[str, Any]:
        analysis = results.get("steps", {}).get("analyze_molecules", {}) or {}
        execution = results.get("steps", {}).get("run_reinvent", {}) or {}
        dry_run = bool(execution.get("skipped"))
        payload = {
            "goal": goal,
            "project": {
                "project_dir": results.get("project_dir"),
                "plan_steps": (results.get("plan") or {}).get("steps"),
            },
            "environment": results.get("steps", {}).get("check_environment", {}),
            "validation": results.get("steps", {}).get("validate_project", {}),
            "execution": execution,
            "inventory": results.get("steps", {}).get("find_output", {}),
            "analysis": analysis,
            "critic": critic,
            "warnings": results.get("warnings", []),
            "errors": results.get("errors", []),
            "dry_run": dry_run,
            "analysis_source": analysis.get("analysis_source"),
            "run_type": _run_type(results),
        }
        report_meta = generate_html_report(payload)
        results["steps"]["generate_report"] = report_meta
        return report_meta

    def _analyze_molecules_step(
        self,
        results: dict[str, Any],
        *,
        plan: dict[str, Any],
        csv_override: str | Path | None,
        config_path: Path,
        project_dir: Path,
        output_dir: Path,
    ) -> None:
        molecule_path = self._resolve_molecule_file(
            results,
            csv_override=csv_override,
            config_path=config_path,
            project_dir=project_dir,
            output_dir=output_dir,
        )
        source = self._analysis_source(results, plan)
        if molecule_path is None:
            msg = (
                "No training SMILES file found for transfer learning"
                if source == "tl_training_set"
                else "No CSV found for analysis"
            )
            results["errors"].append(msg)
            results["steps"]["analyze_molecules"] = {
                "ok": False,
                "errors": [msg],
                "analysis_source": source,
                "from_fresh_reinvent": False,
                "run_type": _run_type(results),
            }
            return
        analysis_cfg = self.agent_config.get("analysis") or {}
        raw_thr = analysis_cfg.get("qed_pass_threshold", 0.5)
        qed_pass_threshold = 0.5 if raw_thr is None else float(raw_thr)
        analysis = analyze_molecules(
            molecule_path, qed_pass_threshold=qed_pass_threshold
        )
        analysis["analysis_source"] = source
        analysis["from_fresh_reinvent"] = source == "fresh_run"
        analysis["run_type"] = _run_type(results)
        details = _validation_details(results)
        artefact = details.get("resolved_output_model")
        inventory = results.get("steps", {}).get("find_output") or {}
        model_files = inventory.get("model_files") or []
        if artefact:
            analysis["artefact_kind"] = "model"
            analysis["artefact_path"] = artefact
        elif model_files:
            analysis["artefact_kind"] = "model"
            analysis["artefact_path"] = model_files[-1]
        results["steps"]["analyze_molecules"] = analysis
        results["warnings"].extend(analysis.get("warnings") or [])
        if not analysis.get("ok"):
            results["errors"].extend(analysis.get("errors") or [])

    @staticmethod
    def _analysis_source(results: dict[str, Any], plan: dict[str, Any]) -> str:
        """Label whether molecule stats came from sampling, TL training, or a stale CSV."""
        if _run_type(results) == "transfer_learning":
            return "tl_training_set"
        run = results.get("steps", {}).get("run_reinvent") or {}
        if run.get("success") and not run.get("skipped"):
            return "fresh_run"
        return "existing_csv"

    def _resolve_molecule_file(
        self,
        results: dict[str, Any],
        *,
        csv_override: str | Path | None,
        config_path: Path,
        project_dir: Path,
        output_dir: Path | None = None,
    ) -> Path | None:
        if _run_type(results) == "transfer_learning" and csv_override is None:
            details = _validation_details(results)
            smiles_path = details.get("smiles_path")
            if smiles_path and Path(smiles_path).is_file():
                return Path(smiles_path)
        return self._resolve_csv(
            results,
            csv_override=csv_override,
            config_path=config_path,
            project_dir=project_dir,
            output_dir=output_dir,
        )

    @staticmethod
    def _resolve_csv(
        results: dict[str, Any],
        *,
        csv_override: str | Path | None,
        config_path: Path,
        project_dir: Path,
        output_dir: Path | None = None,
    ) -> Path | None:
        if csv_override is not None:
            return Path(csv_override).expanduser().resolve()

        if output_dir is None:
            output_dir = project_dir / "output"
        planned_csv = _planned_csv_path(results.get("plan") or {})
        if planned_csv:
            try:
                return resolve_plan_csv_path(
                    planned_csv, project_dir=project_dir, output_dir=output_dir
                )
            except PlanValidationError as exc:
                results.setdefault("warnings", []).append(
                    f"Ignoring unsafe or invalid plan csv_path: {exc}"
                )

        inventory = results.get("steps", {}).get("find_output") or {}
        csv_files = inventory.get("csv_files") or []
        # Prefer sampled.csv / newest non-sample if present
        preferred = [p for p in csv_files if p.endswith("sampled.csv")]
        if preferred:
            return Path(preferred[0])
        non_sample = [p for p in csv_files if "sampled-sample.csv" not in p]
        if non_sample:
            return Path(sorted(non_sample)[-1])
        if csv_files:
            return Path(csv_files[0])

        # Fall back to TOML output_file if present on disk
        validation = results.get("steps", {}).get("validate_project") or {}
        details = validation.get("details") or {}
        resolved = details.get("resolved_output_file")
        if resolved and Path(resolved).is_file():
            return Path(resolved)
        return None


def _run_type(results: dict[str, Any]) -> str:
    return str(_validation_details(results).get("run_type") or "").strip().lower()


def _validation_details(results: dict[str, Any]) -> dict[str, Any]:
    validation = results.get("steps", {}).get("validate_project") or {}
    details = validation.get("details") or {}
    return details if isinstance(details, dict) else {}


def _planned_csv_path(plan: dict[str, Any]) -> str | None:
    """Read the optional validated csv_path from a plan (never from argv)."""
    step_params = plan.get("step_params") or {}
    nested = (step_params.get("analyze_molecules") or {}).get("csv_path")
    if isinstance(nested, str) and nested.strip():
        return nested
    top = plan.get("csv_path")
    if isinstance(top, str) and top.strip():
        return top
    return None
