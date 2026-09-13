"""Deterministic planning agent (no LLM)."""

from __future__ import annotations

from typing import Any


class PlannerAgent:
    """Creates a fixed execution plan from a user goal."""

    DEFAULT_STEPS = [
        "check_environment",
        "validate_project",
        "prepare_execution",
        "run_reinvent",
        "find_output",
        "analyze_molecules",
        "generate_report",
        "critic_review",
    ]

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
        }
