"""Shared OpenAI-compatible LLM client for planner and critic.

Provider presets (openai / xai / gemini / openai_compatible) pick the
base URL, API-key env var, and default model. Callers never read keys
from files — only from environment variables named in config or presets.

The optional ``openai`` package is imported lazily so CI and deterministic
runs do not need it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("openai", "xai", "gemini", "openai_compatible")

# Official OpenAI-compatible endpoints (override with yaml ``base_url``).
XAI_DEFAULT_BASE_URL = "https://api.x.ai/v1"
GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# xAI docs currently list grok-4.6 as the chat model; grok-4 is the
# configurable family default (override in agent.yaml).
XAI_DEFAULT_MODEL = "grok-4"
# Gemini OpenAI-compat docs use gemini-3.8-flash; 2.5-flash is a stable default.
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


class LLMClientError(RuntimeError):
    """Missing key, unknown provider, or SDK/client construction failure."""


@dataclass(frozen=True)
class ProviderPreset:
    """Built-in defaults for a named provider."""

    api_key_env: str
    api_key_aliases: tuple[str, ...] = ()
    base_url: str | None = None
    default_model: str = OPENAI_DEFAULT_MODEL


PROVIDERS: dict[str, ProviderPreset] = {
    "openai": ProviderPreset(
        api_key_env="OPENAI_API_KEY",
        default_model=OPENAI_DEFAULT_MODEL,
    ),
    "xai": ProviderPreset(
        api_key_env="XAI_API_KEY",
        api_key_aliases=("GROK_API_KEY",),
        base_url=XAI_DEFAULT_BASE_URL,
        default_model=XAI_DEFAULT_MODEL,
    ),
    "gemini": ProviderPreset(
        api_key_env="GEMINI_API_KEY",
        api_key_aliases=("GOOGLE_API_KEY",),
        base_url=GEMINI_DEFAULT_BASE_URL,
        default_model=GEMINI_DEFAULT_MODEL,
    ),
    "openai_compatible": ProviderPreset(
        api_key_env="",
        default_model="",
    ),
}


@dataclass(frozen=True)
class LLMSettings:
    """Resolved client settings (no secrets)."""

    provider: str
    api: str
    model: str
    temperature: float
    api_key_env: str
    api_key_aliases: tuple[str, ...]
    base_url: str | None


def overlay_provider(
    agent_config: Mapping[str, Any] | None,
    provider: str | None,
) -> dict[str, Any]:
    """Return a copy of agent_config with planner/critic provider overridden."""
    cfg = dict(agent_config or {})
    if not provider:
        return cfg
    name = str(provider).strip().lower()
    if not name:
        return cfg
    for section_name in ("planner", "critic"):
        section = dict(cfg.get(section_name) or {})
        section["provider"] = name
        cfg[section_name] = section
    return cfg


def resolve_llm_settings(
    section: Mapping[str, Any] | None = None,
    *,
    fallback: Mapping[str, Any] | None = None,
    provider_override: str | None = None,
) -> LLMSettings:
    """Resolve provider, model, base_url, and key env names from config.

    ``section`` is planner or critic yaml. ``fallback`` is used for omitted
    critic fields (inherit planner). ``provider_override`` is the CLI flag.
    """
    section = section or {}
    fallback = fallback or {}

    provider = _optional_str(provider_override) or _pick_str(
        section, fallback, "provider", default="openai"
    )
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise LLMClientError(
            f"Unknown LLM provider {provider!r}; "
            f"use {', '.join(SUPPORTED_PROVIDERS)}"
        )

    preset = PROVIDERS[provider]
    api = (_pick_str(section, fallback, "api", default="chat_completions") or "chat_completions").lower()
    temperature = _pick_float(section, fallback, "temperature", default=0.0)

    model = _pick_str(section, fallback, "model") or preset.default_model
    base_url = _pick_str(section, fallback, "base_url") or preset.base_url
    api_key_env = _pick_str(section, fallback, "api_key_env") or preset.api_key_env

    aliases: tuple[str, ...] = ()
    if api_key_env == preset.api_key_env:
        aliases = preset.api_key_aliases

    if provider == "openai_compatible":
        if not api_key_env:
            raise LLMClientError(
                "openai_compatible provider requires api_key_env in config"
            )
        if not base_url:
            raise LLMClientError(
                "openai_compatible provider requires base_url in config"
            )
        if not model:
            raise LLMClientError(
                "openai_compatible provider requires model in config"
            )

    if not model:
        raise LLMClientError(
            f"LLM provider {provider!r} requires a model (set planner/critic.model)"
        )
    if not api_key_env:
        raise LLMClientError(
            f"LLM provider {provider!r} requires api_key_env "
            "(set planner/critic.api_key_env or use a named provider)"
        )

    return LLMSettings(
        provider=provider,
        api=api,
        model=model,
        temperature=temperature,
        api_key_env=api_key_env,
        api_key_aliases=aliases,
        base_url=base_url,
    )


def api_key_candidates(settings: LLMSettings) -> tuple[str, ...]:
    names = [settings.api_key_env]
    for alias in settings.api_key_aliases:
        if alias not in names:
            names.append(alias)
    return tuple(names)


def read_api_key(
    settings: LLMSettings,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Return ``(api_key, env_var_name)``. Never logs the key value."""
    env = os.environ if environ is None else environ
    candidates = api_key_candidates(settings)
    for name in candidates:
        value = env.get(name)
        if value and str(value).strip():
            return str(value).strip(), name
    listed = " or ".join(candidates)
    raise LLMClientError(f"Missing API key: set environment variable {listed}")


def build_client(
    settings: LLMSettings,
    *,
    openai_cls: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Construct an OpenAI SDK client for the resolved provider.

    ``openai_cls`` is a test seam; production uses ``openai.OpenAI``.
    """
    api_key, source_env = read_api_key(settings, environ=environ)
    cls = openai_cls
    if cls is None:
        try:
            from openai import OpenAI as cls
        except ImportError as exc:  # pragma: no cover - optional extra
            raise LLMClientError(
                "The openai package is not installed. "
                "Install it with: pip install 'openai>=1.40'"
            ) from exc
    kwargs: dict[str, Any] = {"api_key": api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    logger.info(
        "LLM client provider=%s model=%s api=%s base_url=%s key_env=%s",
        settings.provider,
        settings.model,
        settings.api,
        settings.base_url or "default",
        source_env,
    )
    return cls(**kwargs)


def complete_structured(
    client: Any,
    settings: LLMSettings,
    messages: list[dict[str, str]],
    *,
    schema_name: str,
    json_schema: dict[str, Any],
    max_tokens: int = 1024,
) -> str:
    """Chat Completions or Responses call; returns assistant text."""
    if settings.api in ("responses", "response"):
        return _complete_responses(
            client,
            settings,
            messages,
            schema_name=schema_name,
            json_schema=json_schema,
        )
    return _complete_chat(
        client,
        settings,
        messages,
        schema_name=schema_name,
        json_schema=json_schema,
        max_tokens=max_tokens,
    )


def _complete_chat(
    client: Any,
    settings: LLMSettings,
    messages: list[dict[str, str]],
    *,
    schema_name: str,
    json_schema: dict[str, Any],
    max_tokens: int,
) -> str:
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "temperature": settings.temperature,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    schema_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": json_schema,
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
    content = chat_content(response)
    if not content:
        raise LLMClientError("Empty Chat Completions response")
    return content


def _complete_responses(
    client: Any,
    settings: LLMSettings,
    messages: list[dict[str, str]],
    *,
    schema_name: str,
    json_schema: dict[str, Any],
) -> str:
    input_items = [{"role": m["role"], "content": m["content"]} for m in messages]
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "temperature": settings.temperature,
        "input": input_items,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": json_schema,
            }
        },
    }
    response = client.responses.create(**kwargs)
    text = getattr(response, "output_text", None)
    if not text:
        text = responses_text(response)
    if not text:
        raise LLMClientError("Empty Responses API output")
    return text


def chat_content(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    return content if isinstance(content, str) else ""


def responses_text(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pick_str(
    section: Mapping[str, Any],
    fallback: Mapping[str, Any],
    key: str,
    *,
    default: str | None = None,
) -> str | None:
    picked = _optional_str(section.get(key))
    if picked is not None:
        return picked
    picked = _optional_str(fallback.get(key))
    if picked is not None:
        return picked
    return default


def _pick_float(
    section: Mapping[str, Any],
    fallback: Mapping[str, Any],
    key: str,
    *,
    default: float,
) -> float:
    if key in section and section.get(key) is not None:
        return float(section[key])
    if key in fallback and fallback.get(key) is not None:
        return float(fallback[key])
    return float(default)
