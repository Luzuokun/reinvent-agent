"""Deterministic execution agent — calls predefined tools only."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from analysis.molecule_analysis import analyze_molecules
from analysis.report import generate_html_report
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

        for step in plan.get("steps", []):
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
                inventory = find_output_files(output_dir, project_dir=project_dir)
                results["steps"]["find_output"] = inventory

            elif step == "analyze_molecules":
                csv_path = self._resolve_csv(
                    results,
                    csv_override=csv_override,
                    config_path=config_path,
                    project_dir=project_dir,
                )
                if csv_path is None:
                    msg = "No CSV found for analysis"
                    results["errors"].append(msg)
                    results["steps"]["analyze_molecules"] = {
                        "ok": False,
                        "errors": [msg],
                    }
                else:
                    analysis = analyze_molecules(csv_path)
                    results["steps"]["analyze_molecules"] = analysis
                    results["warnings"].extend(analysis.get("warnings") or [])
                    if not analysis.get("ok"):
                        results["errors"].extend(analysis.get("errors") or [])

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
        payload = {
            "goal": goal,
            "project": {
                "project_dir": results.get("project_dir"),
                "plan_steps": (results.get("plan") or {}).get("steps"),
            },
            "environment": results.get("steps", {}).get("check_environment", {}),
            "validation": results.get("steps", {}).get("validate_project", {}),
            "execution": results.get("steps", {}).get("run_reinvent", {}),
            "inventory": results.get("steps", {}).get("find_output", {}),
            "analysis": results.get("steps", {}).get("analyze_molecules", {}),
            "critic": critic,
            "warnings": results.get("warnings", []),
            "errors": results.get("errors", []),
        }
        report_meta = generate_html_report(payload)
        results["steps"]["generate_report"] = report_meta
        return report_meta

    @staticmethod
    def _resolve_csv(
        results: dict[str, Any],
        *,
        csv_override: str | Path | None,
        config_path: Path,
        project_dir: Path,
    ) -> Path | None:
        if csv_override is not None:
            return Path(csv_override).expanduser().resolve()

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
