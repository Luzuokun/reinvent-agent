"""Human-in-the-loop rerun: copy CLI config from a previous run.

The executor stays single-shot. After a human reads the HTML report they
re-invoke the CLI with ``--from-run <id>``. This module reads
``logs/runs/<id>/result.json`` (or ``last``) and copies experiment identity
(project, goal, preset, scaffold, seed, planner/critic/provider, config).

It never copies ``--approve-run``, ``--yes``, or ``--skip-reinvent``.
It never copies TOML, argv, or shell. Launch still requires ``--approve-run``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from agents.experiment_presets import ALLOWED_PRESET_IDS, PRESET_SPECS
from agents.llm_client import SUPPORTED_PROVIDERS
from tools import REPO_ROOT

RUN_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}$")
LAST_ALIAS = "last"
RESULT_NAME = "result.json"

# CLI experiment identity that a human may reuse. Approval flags are not here.
COPYABLE_FIELDS: tuple[str, ...] = (
    "project",
    "goal",
    "preset",
    "scaffold",
    "seed",
    "planner",
    "critic",
    "provider",
    "config",
)

# Never applied from a previous run, even if present in result.json.
NEVER_COPY_FIELDS: frozenset[str] = frozenset(
    {
        "approve_run",
        "yes",
        "skip_reinvent",
        "command",
        "argv",
        "toml",
        "config_text",
        "shell",
        "steps",
        "notes",
    }
)

_FLAG_TO_FIELD: dict[str, str] = {
    "--project": "project",
    "--goal": "goal",
    "--preset": "preset",
    "--scaffold": "scaffold",
    "--seed": "seed",
    "--config": "config",
    "--planner": "planner",
    "--critic": "critic",
    "--provider": "provider",
    "--csv": "csv",
    "--skip-reinvent": "skip_reinvent",
    "--approve-run": "approve_run",
    "--yes": "yes",
    "--from-run": "from_run",
}

_PLANNER_CHOICES = frozenset({"deterministic", "llm"})
_CRITIC_CHOICES = frozenset({"deterministic", "llm"})


class FromRunError(ValueError):
    """Unknown run id, sandbox violation, or unusable previous result."""


def explicit_cli_flags(argv: Iterable[str] | None) -> set[str]:
    """Return argparse dest names the caller actually passed on the CLI."""
    found: set[str] = set()
    if argv is None:
        return found
    for token in argv:
        if not isinstance(token, str) or not token.startswith("--"):
            continue
        name = token.split("=", 1)[0]
        field = _FLAG_TO_FIELD.get(name)
        if field:
            found.add(field)
    return found


def snapshot_invocation(args: argparse.Namespace) -> dict[str, Any]:
    """Record copyable CLI identity for the next human rerun (no approval flags)."""
    skip_reinvent = bool(getattr(args, "skip_reinvent", False))
    snapshot: dict[str, Any] = {
        "project": _repo_relative_or_none(getattr(args, "project", None)),
        "goal": _optional_str(getattr(args, "goal", None)),
        "preset": _optional_str(getattr(args, "preset", None)),
        "scaffold": _repo_relative_or_none(getattr(args, "scaffold", None)),
        "seed": _optional_int(getattr(args, "seed", None)),
        "planner": _optional_str(getattr(args, "planner", None)),
        "critic": _optional_str(getattr(args, "critic", None)),
        "provider": _optional_str(getattr(args, "provider", None)),
        "config": _nondefault_config(getattr(args, "config", None)),
        # Audit-only: applied later only when this invocation also skips REINVENT.
        "csv": _repo_relative_or_none(getattr(args, "csv", None)) if skip_reinvent else None,
        "skip_reinvent": skip_reinvent,
    }
    return snapshot


def resolve_from_run_ref(
    ref: str,
    *,
    logs_dir: str | Path,
) -> tuple[str, Path]:
    """Return ``(run_id, result.json path)`` sandboxed under ``logs_dir/runs``."""
    raw = str(ref).strip()
    if not raw:
        raise FromRunError("--from-run requires a run id")
    if "\x00" in raw:
        raise FromRunError("Invalid --from-run id")

    logs_dir = Path(logs_dir).expanduser().resolve()
    runs_root = (logs_dir / "runs").resolve()

    if raw.lower() == LAST_ALIAS:
        return _resolve_last(logs_dir=logs_dir, runs_root=runs_root)

    if RUN_ID_RE.fullmatch(raw):
        result = _must_be_under(runs_root / raw / RESULT_NAME, runs_root)
        if not result.is_file():
            raise FromRunError(f"No {RESULT_NAME} for run {raw}")
        return raw, result

    return _resolve_path_ref(raw, runs_root=runs_root)


def load_run_result(result_path: str | Path, *, logs_dir: str | Path) -> dict[str, Any]:
    """Load and type-check ``result.json``; path must stay under ``logs/runs``."""
    logs_dir = Path(logs_dir).expanduser().resolve()
    runs_root = (logs_dir / "runs").resolve()
    path = _must_be_under(Path(result_path), runs_root)
    if not path.is_file() or path.name != RESULT_NAME:
        raise FromRunError(f"Expected {RESULT_NAME} under logs/runs/<id>/: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FromRunError(f"Cannot read previous run result: {exc}") from exc
    if not isinstance(payload, dict):
        raise FromRunError("Previous result.json must be a JSON object")
    return payload


def extract_invocation(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Pull copyable CLI identity from a result payload (new or reconstructed)."""
    raw = payload.get("invocation")
    if isinstance(raw, dict) and raw:
        return _sanitize_invocation(raw)

    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
    analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
    critic = payload.get("critic") if isinstance(payload.get("critic"), dict) else {}
    reconstructed = {
        "project": payload.get("project_dir") or plan.get("project_dir"),
        "goal": payload.get("goal") or plan.get("goal"),
        "preset": plan.get("preset_id"),
        "scaffold": plan.get("scaffold_path"),
        "seed": None,
        "planner": plan.get("planner"),
        "critic": critic.get("critic"),
        "provider": None,
        "config": None,
        "csv": analysis.get("csv_path"),
        "skip_reinvent": bool(plan.get("skip_reinvent")),
    }
    return _sanitize_invocation(reconstructed)


def apply_from_run(
    args: argparse.Namespace,
    *,
    logs_dir: str | Path,
    explicit: set[str],
) -> dict[str, Any]:
    """Copy previous-run identity onto ``args``. Explicit CLI flags always win.

    Does not copy ``approve_run``, ``yes``, or ``skip_reinvent``. ``csv`` is
    applied only when this invocation also passed ``--skip-reinvent``.
    """
    ref = getattr(args, "from_run", None)
    if not ref:
        raise FromRunError("--from-run requires a run id")

    run_id, result_path = resolve_from_run_ref(str(ref), logs_dir=logs_dir)
    payload = load_run_result(result_path, logs_dir=logs_dir)
    invocation = extract_invocation(payload)

    copied: list[str] = []
    overridden: list[str] = []
    warnings: list[str] = []

    for field in COPYABLE_FIELDS:
        value = invocation.get(field)
        if value is None or value == "":
            continue
        if field in explicit:
            overridden.append(field)
            continue
        try:
            applied = _apply_field(args, field, value)
        except FromRunError as exc:
            raise FromRunError(f"Previous run {run_id}: {exc}") from exc
        if applied:
            copied.append(field)

    # CSV is not experiment identity; reuse it only for another offline analysis.
    csv_value = invocation.get("csv")
    if csv_value and "csv" not in explicit and bool(getattr(args, "skip_reinvent", False)):
        if _apply_field(args, "csv", csv_value):
            copied.append("csv")
    elif csv_value and "csv" not in explicit:
        warnings.append(
            "Previous csv was not copied (pass --skip-reinvent to re-analyze "
            "the same file, or --csv to override)."
        )

    if "scaffold" not in explicit:
        _drop_scaffold_if_preset_forbids(args)

    meta = {
        "run_id": run_id,
        "result_json": str(result_path),
        "copied": copied,
        "overridden": overridden,
        "warnings": warnings,
        "copied_approve_run": False,
        "copied_skip_reinvent": False,
        "copied_yes": False,
        "copied_toml": False,
    }
    return meta


def _apply_field(args: argparse.Namespace, field: str, value: Any) -> bool:
    if field in NEVER_COPY_FIELDS:
        return False
    if field == "project":
        setattr(args, "project", _require_repo_path(value, what="project"))
        return True
    if field == "goal":
        text = _optional_str(value)
        if not text:
            return False
        args.goal = text
        return True
    if field == "preset":
        pid = _optional_str(value)
        if not pid:
            return False
        if pid not in ALLOWED_PRESET_IDS:
            raise FromRunError(
                f"previous preset_id {pid!r} is not on the allowlist"
            )
        args.preset = pid
        return True
    if field == "scaffold":
        args.scaffold = _require_repo_path(value, what="scaffold")
        return True
    if field == "seed":
        seed = _optional_int(value)
        if seed is None:
            return False
        args.seed = seed
        return True
    if field == "planner":
        mode = _optional_str(value)
        if mode not in _PLANNER_CHOICES:
            return False
        args.planner = mode
        return True
    if field == "critic":
        mode = _optional_str(value)
        if mode not in _CRITIC_CHOICES:
            return False
        args.critic = mode
        return True
    if field == "provider":
        name = _optional_str(value)
        if name not in SUPPORTED_PROVIDERS:
            return False
        args.provider = name
        return True
    if field == "config":
        args.config = _require_repo_path(value, what="config")
        path = Path(args.config)
        if not path.is_file():
            raise FromRunError(f"previous --config does not exist: {path}")
        return True
    if field == "csv":
        args.csv = _require_repo_path(value, what="csv")
        return True
    return False


def _drop_scaffold_if_preset_forbids(args: argparse.Namespace) -> None:
    preset = getattr(args, "preset", None)
    if not preset:
        return
    spec = PRESET_SPECS.get(str(preset))
    if spec is None:
        return
    if "scaffold_path" not in spec.fields:
        args.scaffold = None


def _sanitize_invocation(raw: Mapping[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "skip_reinvent":
            # Audit / reconstruction only; apply_from_run never copies this flag.
            clean[key] = bool(value)
            continue
        if key in NEVER_COPY_FIELDS:
            continue
        if key not in COPYABLE_FIELDS and key not in {"csv", "skip_reinvent"}:
            continue
        if key == "preset":
            pid = _optional_str(value)
            if pid is None:
                continue
            if pid not in ALLOWED_PRESET_IDS:
                raise FromRunError(
                    f"previous preset_id {pid!r} is not on the allowlist"
                )
            clean[key] = pid
            continue
        if key == "seed":
            seed = _optional_int(value)
            if seed is not None:
                clean[key] = seed
            continue
        if key in {"planner", "critic", "provider", "goal"}:
            text = _optional_str(value)
            if text:
                clean[key] = text
            continue
        if key in {"project", "scaffold", "config", "csv"}:
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            rel = _repo_relative_or_none(value)
            if not rel:
                raise FromRunError(
                    f"previous {key} path is missing or outside the repo"
                )
            clean[key] = rel
            continue
    return clean


def _resolve_last(*, logs_dir: Path, runs_root: Path) -> tuple[str, Path]:
    summary_path = logs_dir / "last_run_summary.json"
    if not summary_path.is_file():
        raise FromRunError("No last_run_summary.json; pass an explicit run id")
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FromRunError(f"Cannot read last_run_summary.json: {exc}") from exc
    if not isinstance(data, dict):
        raise FromRunError("last_run_summary.json must be a JSON object")

    result_json = data.get("result_json")
    run_dir = data.get("run_dir")
    run_id = data.get("run_id")

    if isinstance(result_json, str) and result_json.strip():
        result = _must_be_under(Path(result_json), runs_root)
        if not result.is_file():
            raise FromRunError(f"last_run_summary.json points at missing {RESULT_NAME}")
        return _run_id_from_result(result, run_id=run_id), result

    if isinstance(run_dir, str) and run_dir.strip():
        result = _must_be_under(Path(run_dir) / RESULT_NAME, runs_root)
        if not result.is_file():
            raise FromRunError(f"last_run_summary.json run_dir has no {RESULT_NAME}")
        return _run_id_from_result(result, run_id=run_id), result

    raise FromRunError("last_run_summary.json has no result_json / run_dir")


def _resolve_path_ref(raw: str, *, runs_root: Path) -> tuple[str, Path]:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        cwd_try = (Path.cwd() / candidate).resolve()
        runs_try = (runs_root / candidate).resolve()
        if _is_under(cwd_try, runs_root):
            candidate = cwd_try
        elif _is_under(runs_try, runs_root):
            candidate = runs_try
        else:
            raise FromRunError(
                "--from-run path must stay under logs/runs/<id>/result.json"
            )
    else:
        candidate = candidate.resolve()

    if candidate.is_dir():
        candidate = candidate / RESULT_NAME
    candidate = _must_be_under(candidate, runs_root)
    if candidate.name != RESULT_NAME or not candidate.is_file():
        raise FromRunError(
            f"--from-run must point at {RESULT_NAME} under logs/runs/<id>/"
        )
    return candidate.parent.name, candidate


def _run_id_from_result(result: Path, *, run_id: Any) -> str:
    if isinstance(run_id, str) and RUN_ID_RE.fullmatch(run_id.strip()):
        return run_id.strip()
    name = result.parent.name
    if RUN_ID_RE.fullmatch(name):
        return name
    return name


def _must_be_under(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise FromRunError(
            "Run artefact path escapes logs/runs/ (refused)"
        ) from exc
    return resolved


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _repo_relative_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        # Keep relative paths as given if they do not escape the repo when joined.
        joined = (REPO_ROOT / path).resolve()
        try:
            rel = joined.relative_to(REPO_ROOT)
        except ValueError:
            return None
        return str(rel)
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return None


def _require_repo_path(value: Any, *, what: str) -> str:
    rel = _repo_relative_or_none(value)
    if not rel:
        raise FromRunError(f"previous {what} path is missing or outside the repo")
    return rel


def _nondefault_config(value: Any) -> str | None:
    rel = _repo_relative_or_none(value)
    if not rel:
        return None
    default = (REPO_ROOT / "config" / "agent.yaml").resolve()
    try:
        if (REPO_ROOT / rel).resolve() == default:
            return None
    except OSError:
        return rel
    return rel
