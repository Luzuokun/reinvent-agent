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
from agents.experiment_presets import ALLOWED_PRESET_IDS, PRESET_SPECS
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


def _preset_prompt_lines() -> str:
    lines = []
    for preset_id in ALLOWED_PRESET_IDS:
        spec = PRESET_SPECS[preset_id]
        extra = ""
        if "scaffold_path" in spec.fields:
            extra = " Requires scaffold_path (existing file under project input/)."
        lines.append(f"- {preset_id}: {spec.description}{extra}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are the planning component of a REINVENT4 research agent.

Return ONLY a JSON object. You do not execute tools, shell, or Python.
You never invent tools, argv, or REINVENT config/model paths.
You never generate TOML text, scoring components, or config file contents.

Allowed step names (use only these):
- check_environment
- validate_project
- prepare_execution
- run_reinvent
- find_output
- analyze_molecules
- generate_report
- critic_review

Allowed experiment preset IDs (or null to use the project's reinvent.toml):
{_preset_prompt_lines()}

JSON keys allowed: steps, notes, csv_path, preset_id, scaffold_path.
- steps: array of allowed step name strings, no duplicates.
- notes: short human comments only — never commands and never TOML.
- csv_path: optional. If set, it must already exist under the project output directory.
  Prefer null unless the user named a specific existing CSV.
- preset_id: one of the IDs above, or null. This is the only way to choose an experiment.
- scaffold_path: only when preset_id is sampling-cpu-scaffold. Must already exist under
  the project input directory. Prefer null otherwise.

Rules:
- Prefer the standard full pipeline unless skip_reinvent is true or the goal is offline analysis.
- If skip_reinvent is true, omit run_reinvent and prepare_execution.
- For "generate N molecules" goals, pick sampling-cpu-100 or sampling-cpu-1000. Do not write TOML.
- For "from this scaffold / optimize QED" goals, pick sampling-cpu-scaffold and set scaffold_path
  to an existing input file. QED is measured after sampling; you cannot add a scoring section.
- run_reinvent has no parameters. Launch still requires --approve-run (you cannot set it).
- If the project is transfer_learning and preset_id is null, molecule analysis is the training
  SMILES file, not a fresh generation. Do not treat training molecules as sampled output.
- Do not include any other keys (no toml, config, argv, command, shell).
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
        preset_id: str | None = None,
        scaffold_path: str | None = None,
    ) -> dict[str, Any]:
        project_dir = str(project_dir)
        output_dirname = self.agent_config.get("project", {}).get("output_dirname", "output")
        output_dir = Path(project_dir).expanduser().resolve() / output_dirname

        try:
            raw_text = self._complete(
                self._messages(
                    goal,
                    project_dir,
                    approve_run,
                    skip_reinvent,
                    output_dir,
                    preset_id=preset_id,
                    scaffold_path=scaffold_path,
                )
            )
            raw = _parse_json_object(raw_text)
            # CLI --preset / --scaffold are caller-owned, like approve_run.
            if preset_id:
                raw["preset_id"] = preset_id
            if scaffold_path:
                raw["scaffold_path"] = scaffold_path
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
            return self._fallback_or_raise(
                exc,
                goal,
                project_dir,
                approve_run,
                skip_reinvent,
                preset_id=preset_id,
                scaffold_path=scaffold_path,
            )

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
        if validated.get("preset_id"):
            plan["preset_id"] = validated["preset_id"]
        if validated.get("preset"):
            plan["preset"] = validated["preset"]
        if validated.get("scaffold_path"):
            plan["scaffold_path"] = validated["scaffold_path"]
        return plan

    def _fallback_or_raise(
        self,
        exc: BaseException,
        goal: str,
        project_dir: str,
        approve_run: bool,
        skip_reinvent: bool,
        preset_id: str | None = None,
        scaffold_path: str | None = None,
    ) -> dict[str, Any]:
        message = f"LLM planner failed ({type(exc).__name__}: {exc})"
        if not self.fallback_on_error:
            raise LLMPlannerError(message) from exc
        logger.warning("%s — falling back to deterministic plan.", message)
        # Do not carry a rejected model preset into the fallback; CLI-owned
        # --preset / --scaffold are still applied (they already passed argparse).
        try:
            plan = PlannerAgent().create_plan(
                goal,
                project_dir=project_dir,
                approve_run=approve_run,
                skip_reinvent=skip_reinvent,
                preset_id=preset_id,
                scaffold_path=scaffold_path,
            )
        except PlanValidationError:
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
        *,
        preset_id: str | None = None,
        scaffold_path: str | None = None,
    ) -> list[dict[str, str]]:
        input_dir = Path(project_dir).expanduser().resolve() / "input"
        user = (
            f"Goal: {goal}\n"
            f"Project directory: {project_dir}\n"
            f"Project output directory: {output_dir}\n"
            f"Project input directory: {input_dir}\n"
            f"approve_run: {str(approve_run).lower()}\n"
            f"skip_reinvent: {str(skip_reinvent).lower()}\n"
            f"cli_preset_id: {preset_id or 'null'}\n"
            f"cli_scaffold_path: {scaffold_path or 'null'}\n"
            "Return the JSON plan object. preset_id must be an allowlisted ID or null; "
            "never return TOML."
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
