"""Deterministic planning agent (no LLM). Default Phase 1 / Phase 2 planner."""

from __future__ import annotations

from typing import Any

from agents.plan_schema import ALLOWED_STEPS


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

        return {
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
