"""LLM provider resolution — mocked client only, no network."""

from __future__ import annotations

import pytest

from agents.llm_client import (
    GEMINI_DEFAULT_BASE_URL,
    GEMINI_DEFAULT_MODEL,
    OPENAI_DEFAULT_MODEL,
    XAI_DEFAULT_BASE_URL,
    XAI_DEFAULT_MODEL,
    LLMClientError,
    build_client,
    overlay_provider,
    read_api_key,
    resolve_llm_settings,
)
from main import build_parser
from tools import load_agent_config


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_openai_preset_base_url_and_env():
    settings = resolve_llm_settings({"provider": "openai"})
    assert settings.provider == "openai"
    assert settings.base_url is None
    assert settings.api_key_env == "OPENAI_API_KEY"
    assert settings.api_key_aliases == ()
    assert settings.model == OPENAI_DEFAULT_MODEL


def test_xai_preset_base_url_and_env():
    settings = resolve_llm_settings({"provider": "xai"})
    assert settings.provider == "xai"
    assert settings.base_url == XAI_DEFAULT_BASE_URL == "https://api.x.ai/v1"
    assert settings.api_key_env == "XAI_API_KEY"
    assert settings.api_key_aliases == ("GROK_API_KEY",)
    assert settings.model == XAI_DEFAULT_MODEL == "grok-4"


def test_gemini_preset_base_url_and_env():
    settings = resolve_llm_settings({"provider": "gemini"})
    assert settings.provider == "gemini"
    assert settings.base_url == GEMINI_DEFAULT_BASE_URL
    assert settings.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert settings.api_key_env == "GEMINI_API_KEY"
    assert settings.api_key_aliases == ("GOOGLE_API_KEY",)
    assert settings.model == GEMINI_DEFAULT_MODEL


def test_yaml_overrides_model_base_url_and_key_env():
    settings = resolve_llm_settings(
        {
            "provider": "xai",
            "model": "grok-4.6",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "MY_XAI_KEY",
        }
    )
    assert settings.model == "grok-4.6"
    assert settings.base_url == "https://example.invalid/v1"
    assert settings.api_key_env == "MY_XAI_KEY"
    assert settings.api_key_aliases == ()


def test_openai_compatible_uses_config_only():
    settings = resolve_llm_settings(
        {
            "provider": "openai_compatible",
            "model": "local-model",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key_env": "LOCAL_API_KEY",
        }
    )
    assert settings.provider == "openai_compatible"
    assert settings.model == "local-model"
    assert settings.base_url == "http://127.0.0.1:8000/v1"
    assert settings.api_key_env == "LOCAL_API_KEY"
    assert settings.api_key_aliases == ()


@pytest.mark.parametrize(
    "cfg",
    [
        {"provider": "openai_compatible", "base_url": "http://x", "model": "m"},
        {"provider": "openai_compatible", "api_key_env": "K", "model": "m"},
        {"provider": "openai_compatible", "api_key_env": "K", "base_url": "http://x"},
    ],
)
def test_openai_compatible_requires_key_url_model(cfg):
    with pytest.raises(LLMClientError, match="openai_compatible"):
        resolve_llm_settings(cfg)


def test_unknown_provider_raises():
    with pytest.raises(LLMClientError, match="Unknown LLM provider"):
        resolve_llm_settings({"provider": "anthropic"})


def test_critic_inherits_planner_provider_when_omitted():
    settings = resolve_llm_settings(
        {},
        fallback={"provider": "xai", "model": "grok-4.6"},
    )
    assert settings.provider == "xai"
    assert settings.model == "grok-4.6"
    assert settings.base_url == XAI_DEFAULT_BASE_URL


def test_xai_reads_grok_api_key_alias(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("GROK_API_KEY", "dummy-grok")
    settings = resolve_llm_settings({"provider": "xai"})
    key, source = read_api_key(settings)
    assert key == "dummy-grok"
    assert source == "GROK_API_KEY"


def test_gemini_reads_google_api_key_alias(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-google")
    settings = resolve_llm_settings({"provider": "gemini"})
    key, source = read_api_key(settings)
    assert key == "dummy-google"
    assert source == "GOOGLE_API_KEY"


def test_xai_prefers_primary_env_over_alias(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "primary")
    monkeypatch.setenv("GROK_API_KEY", "alias")
    settings = resolve_llm_settings({"provider": "xai"})
    key, source = read_api_key(settings)
    assert key == "primary"
    assert source == "XAI_API_KEY"


def test_build_client_openai_omits_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = resolve_llm_settings({"provider": "openai"})
    client = build_client(settings, openai_cls=_FakeOpenAI)
    assert client.kwargs == {"api_key": "sk-test"}


def test_build_client_xai_kwargs(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    settings = resolve_llm_settings({"provider": "xai"})
    client = build_client(settings, openai_cls=_FakeOpenAI)
    assert client.kwargs["api_key"] == "xai-test"
    assert client.kwargs["base_url"] == "https://api.x.ai/v1"


def test_build_client_gemini_kwargs(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
    settings = resolve_llm_settings({"provider": "gemini"})
    client = build_client(settings, openai_cls=_FakeOpenAI)
    assert client.kwargs["api_key"] == "gem-test"
    assert (
        client.kwargs["base_url"]
        == "https://generativelanguage.googleapis.com/v1beta/openai/"
    )


def test_missing_key_lists_aliases(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    settings = resolve_llm_settings({"provider": "xai"})
    with pytest.raises(LLMClientError, match="XAI_API_KEY or GROK_API_KEY"):
        read_api_key(settings)


def test_overlay_provider_sets_planner_and_critic():
    cfg = overlay_provider(
        {"planner": {"mode": "llm"}, "critic": {"mode": "llm"}},
        "xai",
    )
    assert cfg["planner"]["provider"] == "xai"
    assert cfg["critic"]["provider"] == "xai"


def test_cli_provider_flag_defaults_and_choices():
    parser = build_parser()
    assert parser.parse_args([]).provider is None
    assert parser.parse_args(["--provider", "xai"]).provider == "xai"
    assert parser.parse_args(["--provider", "gemini"]).provider == "gemini"
    assert parser.parse_args(["--provider", "openai_compatible"]).provider == "openai_compatible"
    with pytest.raises(SystemExit):
        parser.parse_args(["--provider", "anthropic"])


def test_default_agent_yaml_resolves_openai_preset():
    cfg = load_agent_config()
    planner = resolve_llm_settings(cfg.get("planner") or {})
    critic = resolve_llm_settings(cfg.get("critic") or {}, fallback=cfg.get("planner") or {})
    assert planner.provider == "openai"
    assert planner.base_url is None
    assert planner.api_key_env == "OPENAI_API_KEY"
    assert planner.model == OPENAI_DEFAULT_MODEL
    assert critic.provider == "openai"
    assert critic.api_key_env == "OPENAI_API_KEY"


def test_overlay_default_yaml_to_xai_uses_xai_preset():
    cfg = overlay_provider(load_agent_config(), "xai")
    planner = resolve_llm_settings(cfg.get("planner") or {})
    critic = resolve_llm_settings(cfg.get("critic") or {}, fallback=cfg.get("planner") or {})
    assert planner.provider == "xai"
    assert planner.base_url == XAI_DEFAULT_BASE_URL
    assert planner.api_key_env == "XAI_API_KEY"
    assert planner.model == XAI_DEFAULT_MODEL
    assert critic.provider == "xai"
    assert critic.api_key_env == "XAI_API_KEY"
    assert critic.model == XAI_DEFAULT_MODEL
