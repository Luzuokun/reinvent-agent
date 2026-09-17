"""Deterministic planning agent (no LLM). Default Phase 1 / Phase 2 planner."""

from __future__ import annotations

from typing import Any

from agents.experiment_presets import PresetError, validate_preset_selection
from agents.plan_schema import ALLOWED_STEPS, PlanValidationError


class PlannerAgent:
    """Creates a fixed execution plan from a user goal."""

    DEFAULT_STEPS = list(ALLOWED_STEPS)

    def create_plan(
        self,
        goal: str,
        *,
        project_dir: str,
        approve_run: bool,
        skip_reinvent: bool = False,
        preset_id: str | None = None,
        scaffold_path: str | None = None,
    ) -> dict[str, Any]:
        steps = list(self.DEFAULT_STEPS)
        if skip_reinvent:
            steps = [
                "check_environment",
                "validate_project",
                "find_output",
                "analyze_molecules",
                "generate_report",
                "critic_review",
            ]

        notes: list[str] = []
        if not approve_run and not skip_reinvent:
            notes.append(
                "run_reinvent will prepare the command but will not launch "
                "without --approve-run."
            )
        if skip_reinvent:
            notes.append("REINVENT execution skipped (--skip-reinvent).")

        try:
            preset = validate_preset_selection(
                preset_id,
                project_dir=project_dir,
                scaffold_path=scaffold_path,
            )
        except PresetError as exc:
            raise PlanValidationError(str(exc)) from exc
        if preset:
            notes.append(
                f"Experiment preset: {preset['preset_id']} ({preset['description']})"
            )

        plan: dict[str, Any] = {
            "goal": goal,
            "project_dir": project_dir,
            "approve_run": approve_run,
            "skip_reinvent": skip_reinvent,
            "steps": steps,
            "notes": notes,
            "planner": "deterministic",
            "planner_fallback": False,
            "planner_warnings": [],
        }
        if preset.get("preset_id"):
            plan["preset_id"] = preset["preset_id"]
            plan["preset"] = {
                "preset_id": preset["preset_id"],
                "description": preset["description"],
                "source_toml": preset["source_toml"],
            }
        if preset.get("scaffold_path"):
            plan["scaffold_path"] = preset["scaffold_path"]
        return plan
