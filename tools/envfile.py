"""Load a gitignored repo-root ``.env`` into the process environment.

Existing non-empty environment variables win. Values are never logged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, MutableMapping

from tools import REPO_ROOT

_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines. Comments and blank lines are ignored."""
    parsed: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not _ENV_KEY.match(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        parsed[key] = value
    return parsed


def apply_dotenv(
    values: Mapping[str, str],
    environ: MutableMapping[str, str],
) -> list[str]:
    """Set keys that are missing or empty. Returns names that were applied."""
    applied: list[str] = []
    for key, value in values.items():
        current = environ.get(key)
        if current is not None and str(current).strip():
            continue
        environ[key] = value
        applied.append(key)
    return applied


def load_repo_dotenv(
    root: Path | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> list[str]:
    """Load ``<root>/.env`` if present. Never overrides a non-empty env var."""
    import os

    env: MutableMapping[str, str] = os.environ if environ is None else environ
    path = (root or REPO_ROOT) / ".env"
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    if text.startswith("\ufeff"):
        text = text[1:]
    return apply_dotenv(parse_dotenv(text), env)
