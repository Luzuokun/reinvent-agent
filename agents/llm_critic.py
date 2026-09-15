"""Optional LLM critic — structured JSON only, never executed as a command.

Calls an OpenAI-compatible Chat Completions or Responses API and returns a
critic verdict dict (PASS | WARNING | FAIL). Model text is parsed as JSON,
then passed through ``validate_critic_verdict``. It is never interpolated
into argv, a shell, or a filesystem write.

The model sees a compact evidence summary only. It does not call tools,
run REINVENT, or invent docking / MD / literature claims.

On missing API key, invalid verdict, or provider error the default is to
fall back to the deterministic critic with a loud warning (resilience).
Set ``critic.fallback_on_error: false`` to fail instead.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

from agents.critic import CriticAgent
from agents.critic_schema import (
    LLM_CRITIC_JSON_SCHEMA,
    CriticValidationError,
    build_evidence_summary,
    parse_json_object,
    validate_critic_verdict,
)

logger = logging.getLogger(__name__)

CompleteFn = Callable[[list[dict[str, str]]], str]


class LLMCriticError(RuntimeError):
    """Fatal LLM critic failure (only raised when fallback is disabled)."""


SYSTEM_PROMPT = """You are the critic component of a REINVENT4 research agent.

Return ONLY a JSON object. You do not execute tools, shell, or Python.
You do not modify files. You do not call REINVENT.

You review structured workflow evidence only. Do not invent experiments
that are not in the evidence: no docking, no molecular dynamics / GROMACS,
no PubMed / literature claims, no binding affinities, no IC50/Kd.

JSON keys allowed:
- status: exactly one of PASS, WARNING, FAIL
- issues: array of short evidence-grounded strings (may be empty)
- recommendation: one short human-facing sentence

Rules:
- PASS only if the evidence is internally consistent and no hard failure is present.
- FAIL if the workflow aborted, validation failed when a run was approved,
  REINVENT was supposed to run but failed/skipped, analysis failed, or
  molecule count is below the given min_molecules threshold.
- WARNING for dry-run (approve_run false and skip_reinvent false),
  high duplicate fraction, low valid-molecule fraction, or missing RDKit —
  using the numeric thresholds provided in the evidence.
- analysis_source existing_csv means stats are from an existing file, not a
  fresh generation; mention that if relevant (usually WARNING when dry-run).
- Do not include any other keys.
- Do not emit shell commands.
"""


class LLMCriticAgent:
    """LLM-backed critic with schema validation and deterministic fallback."""

    def __init__(
        self,
        agent_config: dict[str, Any] | None = None,
        *,
        client: Any | None = None,
        complete_fn: CompleteFn | None = None,
    ) -> None:
        self.agent_config = agent_config or {}
        critic_cfg = self.agent_config.get("critic") or {}
        planner_cfg = self.agent_config.get("planner") or {}
        self.provider = str(critic_cfg.get("provider") or planner_cfg.get("provider") or "openai")
        self.api = str(critic_cfg.get("api") or planner_cfg.get("api") or "chat_completions").lower()
        self.model = str(critic_cfg.get("model") or planner_cfg.get("model") or "gpt-4o-mini")
        self.temperature = float(critic_cfg.get("temperature", planner_cfg.get("temperature", 0.0)))
        self.api_key_env = str(
            critic_cfg.get("api_key_env") or planner_cfg.get("api_key_env") or "OPENAI_API_KEY"
        )
        self.base_url = critic_cfg.get("base_url") or planner_cfg.get("base_url") or None
        self.fallback_on_error = bool(critic_cfg.get("fallback_on_error", True))
        self._deterministic = CriticAgent(agent_config=self.agent_config)
        self._client = client
        self._complete_fn = complete_fn

    def review(self, results: dict[str, Any]) -> dict[str, Any]:
        evidence = build_evidence_summary(
            results,
            thresholds=self._deterministic_thresholds(),
        )
        try:
            raw_text = self._complete(self._messages(evidence))
            raw = parse_json_object(raw_text)
            validated = validate_critic_verdict(raw, evidence=evidence, source="llm")
        except Exception as exc:  # noqa: BLE001 — any critic failure is non-fatal by default
            return self._fallback_or_raise(exc, results)

        return {
            "status": validated["status"],
            "issues": list(validated["issues"]),
            "recommendation": validated["recommendation"],
            "thresholds": self._deterministic_thresholds(),
            "critic": "llm",
            "critic_fallback": False,
            "critic_warnings": [],
        }

    def _deterministic_thresholds(self) -> dict[str, Any]:
        return {
            "min_valid_fraction": self._deterministic.min_valid_fraction,
            "max_duplicate_fraction": self._deterministic.max_duplicate_fraction,
            "min_molecules": self._deterministic.min_molecules,
        }

    def _fallback_or_raise(
        self,
        exc: BaseException,
        results: dict[str, Any],
    ) -> dict[str, Any]:
        message = f"LLM critic failed ({type(exc).__name__}: {exc})"
        if not self.fallback_on_error:
            raise LLMCriticError(message) from exc
        logger.warning("%s — falling back to deterministic critic.", message)
        verdict = self._deterministic.review(results)
        verdict["critic"] = "deterministic"
        verdict["critic_fallback"] = True
        verdict["critic_warnings"] = [message]
        return verdict

    def _messages(self, evidence: dict[str, Any]) -> list[dict[str, str]]:
        user = (
            "Review this structured workflow evidence and return the JSON "
            "verdict object.\n\n"
            + json.dumps(evidence, indent=2, ensure_ascii=False, default=str)
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def _complete(self, messages: list[dict[str, str]]) -> str:
        if self._complete_fn is not None:
            return self._complete_fn(messages)
        client = self._ensure_client()
        if self.api in ("responses", "response"):
            return self._complete_responses(client, messages)
        return self._complete_chat(client, messages)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise LLMCriticError(
                f"Missing API key: set environment variable {self.api_key_env}"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise LLMCriticError(
                "The openai package is not installed. "
                "Install it with: pip install 'openai>=1.40'"
            ) from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = str(self.base_url)
        return OpenAI(**kwargs)

    def _complete_chat(self, client: Any, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            "max_tokens": 1024,
        }
        schema_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "reinvent_agent_critic",
                "strict": True,
                "schema": LLM_CRITIC_JSON_SCHEMA,
            },
        }
        try:
            response = client.chat.completions.create(
                **kwargs, response_format=schema_format
            )
        except Exception as exc:  # noqa: BLE001
            hint = str(exc).lower()
            if any(token in hint for token in ("response_format", "json_schema", "structured")):
                logger.warning("Structured json_schema not accepted; retrying as json_object.")
                response = client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"}
                )
            else:
                raise
        content = _chat_content(response)
        if not content:
            raise LLMCriticError("Empty Chat Completions response")
        return content

    def _complete_responses(self, client: Any, messages: list[dict[str, str]]) -> str:
        input_items = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "input": input_items,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "reinvent_agent_critic",
                    "strict": True,
                    "schema": LLM_CRITIC_JSON_SCHEMA,
                }
            },
        }
        response = client.responses.create(**kwargs)
        text = getattr(response, "output_text", None)
        if not text:
            text = _responses_text(response)
        if not text:
            raise LLMCriticError("Empty Responses API output")
        return text


def _chat_content(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    return content if isinstance(content, str) else ""


def _responses_text(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)
