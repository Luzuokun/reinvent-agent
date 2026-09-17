"""Repo-root .env loader — no secrets, no network."""

from __future__ import annotations

from pathlib import Path

from tools.envfile import apply_dotenv, load_repo_dotenv, parse_dotenv


def test_parse_dotenv_ignores_comments_and_export():
    parsed = parse_dotenv(
        """
# comment
export XAI_API_KEY='xai-test'
OPENAI_API_KEY="sk-test"
EMPTY=
not a line
"""
    )
    assert parsed["XAI_API_KEY"] == "xai-test"
    assert parsed["OPENAI_API_KEY"] == "sk-test"
    assert parsed["EMPTY"] == ""
    assert "not a line" not in parsed


def test_apply_dotenv_does_not_override_existing():
    environ = {"XAI_API_KEY": "already-set"}
    applied = apply_dotenv({"XAI_API_KEY": "from-file", "GROK_API_KEY": "alias"}, environ)
    assert environ["XAI_API_KEY"] == "already-set"
    assert environ["GROK_API_KEY"] == "alias"
    assert applied == ["GROK_API_KEY"]


def test_apply_dotenv_fills_empty_existing():
    environ = {"XAI_API_KEY": "  "}
    applied = apply_dotenv({"XAI_API_KEY": "from-file"}, environ)
    assert environ["XAI_API_KEY"] == "from-file"
    assert applied == ["XAI_API_KEY"]


def test_load_repo_dotenv_reads_file(tmp_path: Path):
    (tmp_path / ".env").write_text("XAI_API_KEY=from-disk\n", encoding="utf-8")
    environ: dict[str, str] = {}
    applied = load_repo_dotenv(tmp_path, environ=environ)
    assert applied == ["XAI_API_KEY"]
    assert environ["XAI_API_KEY"] == "from-disk"


def test_load_repo_dotenv_missing_file(tmp_path: Path):
    assert load_repo_dotenv(tmp_path, environ={}) == []
