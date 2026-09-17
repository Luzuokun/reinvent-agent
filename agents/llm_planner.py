"""Optional LLM planner — structured JSON only, never executed as a command.

Calls an OpenAI-compatible Chat Completions or Responses API and returns a
Plan dict. Model text is parsed as JSON, then passed through
``validate_plan``. It is never interpolated into argv or a shell.

Client construction is centralized in ``agents.llm_client`` (openai / xai /
gemini / openai_compatible).

On missing API key, invalid plan, or provider error the default is to fall
back to the deterministic planner with a loud warning (resilience). Set
``planner.fallback_on_error: false`` to fail instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from agents.llm_client import (
    LLMClientError,
    LLMSettings,
    build_client,
    complete_structured,
    resolve_llm_settings,
)
from agents.plan_schema import (
    ALLOWED_STEPS,
    LLM_PLAN_JSON_SCHEMA,
    PlanValidationError,
    resolve_allowlist,
    validate_plan,
)
from agents.planner import PlannerAgent

logger = logging.getLogger(__name__)

CompleteFn = Callable[[list[dict[str, str]]], str]


class LLMPlannerError(RuntimeError):
    """Fatal LLM planner failure (only raised when fallback is disabled)."""


SYSTEM_PROMPT = """You are the planning component of a REINVENT4 research agent.

Return ONLY a JSON object. You do not execute tools, shell, or Python.
You never invent tools, argv, or REINVENT config/model paths.

Allowed step names (use only these):
- check_environment
- validate_project
- prepare_execution
- run_reinvent
- find_output
- analyze_molecules
- generate_report
- critic_review

JSON keys allowed: steps, notes, csv_path.
- steps: array of allowed step name strings, no duplicates.
- notes: short human comments only — never commands.
- csv_path: optional. If set, it must already exist under the project output directory.
  Prefer null unless the user named a specific existing CSV.

Rules:
- Prefer the standard full pipeline unless skip_reinvent is true or the goal is offline analysis.
- If skip_reinvent is true, omit run_reinvent and prepare_execution.
- run_reinvent has no parameters. The runtime uses the project's reinvent.toml and still requires --approve-run.
- If the project is transfer_learning, molecule analysis is the training SMILES file, not a fresh generation. Do not treat training molecules as sampled output.
- Do not include any other keys.
"""


class LLMPlannerAgent:
    """LLM-backed planner with schema validation and deterministic fallback."""

    def __init__(
        self,
        agent_config: dict[str, Any] | None = None,
        *,
        client: Any | None = None,
        complete_fn: CompleteFn | None = None,
        provider_override: str | None = None,
    ) -> None:
        self.agent_config = agent_config or {}
        planner_cfg = self.agent_config.get("planner") or {}
        self._settings_error: BaseException | None = None
        self.llm: LLMSettings | None
        try:
            self.llm = resolve_llm_settings(
                planner_cfg, provider_override=provider_override
            )
        except LLMClientError as exc:
            self.llm = None
            self._settings_error = exc
        self.provider = (
            self.llm.provider if self.llm is not None else str(planner_cfg.get("provider") or "openai")
        )
        self.api = self.llm.api if self.llm is not None else str(planner_cfg.get("api") or "chat_completions").lower()
        self.model = self.llm.model if self.llm is not None else str(planner_cfg.get("model") or "gpt-4o-mini")
        self.temperature = (
            self.llm.temperature if self.llm is not None else float(planner_cfg.get("temperature", 0.0))
        )
        self.max_steps = int(planner_cfg.get("max_steps") or len(ALLOWED_STEPS))
        self.allowlist = resolve_allowlist(planner_cfg.get("allowlist"))
        self.api_key_env = (
            self.llm.api_key_env if self.llm is not None else str(planner_cfg.get("api_key_env") or "OPENAI_API_KEY")
        )
        self.base_url = self.llm.base_url if self.llm is not None else (planner_cfg.get("base_url") or None)
        self.fallback_on_error = bool(planner_cfg.get("fallback_on_error", True))
        self._client = client
        self._complete_fn = complete_fn

    def create_plan(
        self,
        goal: str,
        *,
        project_dir: str,
        approve_run: bool,
        skip_reinvent: bool = False,
    ) -> dict[str, Any]:
        project_dir = str(project_dir)
        output_dirname = self.agent_config.get("project", {}).get("output_dirname", "output")
        output_dir = Path(project_dir).expanduser().resolve() / output_dirname

        try:
            raw_text = self._complete(self._messages(goal, project_dir, approve_run, skip_reinvent, output_dir))
            raw = _parse_json_object(raw_text)
            validated = validate_plan(
                raw,
                project_dir=project_dir,
                output_dir=output_dir,
                max_steps=self.max_steps,
                allowlist=self.allowlist,
                skip_reinvent=skip_reinvent,
                source="llm",
            )
        except Exception as exc:  # noqa: BLE001 — any planner failure is non-fatal by default
            return self._fallback_or_raise(exc, goal, project_dir, approve_run, skip_reinvent)

        notes = list(validated.get("notes") or [])
        notes.extend(validated.get("warnings") or [])
        plan: dict[str, Any] = {
            "goal": goal,
            "project_dir": project_dir,
            "approve_run": approve_run,
            "skip_reinvent": skip_reinvent,
            "steps": list(validated["steps"]),
            "notes": notes,
            "planner": "llm",
            "planner_fallback": False,
            "planner_warnings": [],
            "step_params": validated.get("step_params") or {},
        }
        if validated.get("csv_path"):
            plan["csv_path"] = validated["csv_path"]
        return plan

    def _fallback_or_raise(
        self,
        exc: BaseException,
        goal: str,
        project_dir: str,
        approve_run: bool,
        skip_reinvent: bool,
    ) -> dict[str, Any]:
        message = f"LLM planner failed ({type(exc).__name__}: {exc})"
        if not self.fallback_on_error:
            raise LLMPlannerError(message) from exc
        logger.warning("%s — falling back to deterministic plan.", message)
        plan = PlannerAgent().create_plan(
            goal,
            project_dir=project_dir,
            approve_run=approve_run,
            skip_reinvent=skip_reinvent,
        )
        plan["planner"] = "deterministic"
        plan["planner_fallback"] = True
        plan["planner_warnings"] = [message]
        return plan

    def _messages(
        self,
        goal: str,
        project_dir: str,
        approve_run: bool,
        skip_reinvent: bool,
        output_dir: Path,
    ) -> list[dict[str, str]]:
        user = (
            f"Goal: {goal}\n"
            f"Project directory: {project_dir}\n"
            f"Project output directory: {output_dir}\n"
            f"approve_run: {str(approve_run).lower()}\n"
            f"skip_reinvent: {str(skip_reinvent).lower()}\n"
            "Return the JSON plan object."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def _complete(self, messages: list[dict[str, str]]) -> str:
        if self._complete_fn is not None:
            return self._complete_fn(messages)
        settings = self._require_settings()
        client = self._ensure_client()
        try:
            return complete_structured(
                client,
                settings,
                messages,
                schema_name="reinvent_agent_plan",
                json_schema=LLM_PLAN_JSON_SCHEMA,
            )
        except LLMClientError as exc:
            raise LLMPlannerError(str(exc)) from exc

    def _require_settings(self) -> LLMSettings:
        if self._settings_error is not None:
            raise LLMPlannerError(str(self._settings_error)) from self._settings_error
        if self.llm is None:
            raise LLMPlannerError("LLM planner settings are not resolved")
        return self.llm

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        settings = self._require_settings()
        try:
            return build_client(settings)
        except LLMClientError as exc:
            raise LLMPlannerError(str(exc)) from exc


def _parse_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise PlanValidationError("LLM returned empty plan text")
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.strip("`")
        if payload.startswith("json"):
            payload = payload[4:]
        payload = payload.strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PlanValidationError(f"LLM plan is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PlanValidationError("LLM plan JSON must be an object")
    return data
