"""Strict plan schema and validator for REINVENT4 Agent planners.

The model may only name steps from ``ALLOWED_STEPS`` and a preset ID from
``ALLOWED_PRESET_IDS``. It cannot introduce commands, argv fragments,
TOML text, or unrestricted paths. ``run_reinvent`` has no model-controlled
parameters; experiment configuration is a human-written preset, not generated
TOML. ``analyze_molecules`` may optionally name an existing CSV that already
lives under the project output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from agents.experiment_presets import PresetError, validate_preset_selection

ALLOWED_STEPS: tuple[str, ...] = (
    "check_environment",
    "validate_project",
    "prepare_execution",
    "run_reinvent",
    "find_output",
    "analyze_molecules",
    "generate_report",
    "critic_review",
)

DEFAULT_MAX_STEPS = len(ALLOWED_STEPS)

# Keys an LLM is allowed to emit. approve_run / skip_reinvent are never
# taken from model text — the CLI / caller owns those flags.
# Experiment config is preset_id only; TOML / argv / config_text are forbidden.
LLM_PLAN_KEYS = frozenset(
    {"steps", "notes", "csv_path", "preset_id", "scaffold_path"}
)

SYSTEM_PLAN_KEYS = frozenset(
    {
        "goal",
        "project_dir",
        "approve_run",
        "skip_reinvent",
        "planner",
        "planner_fallback",
        "planner_warnings",
        "step_params",
        "preset",
    }
)

ALLOWED_PLAN_KEYS = LLM_PLAN_KEYS | SYSTEM_PLAN_KEYS
ALLOWED_STEP_OBJECT_KEYS = frozenset({"name", "params"})
ALLOWED_STEP_PARAMS: dict[str, frozenset[str]] = {
    "analyze_molecules": frozenset({"csv_path"}),
}

MAX_NOTES = 8
MAX_NOTE_CHARS = 400

# OpenAI structured-output JSON Schema (strict: every property is required).
LLM_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": DEFAULT_MAX_STEPS,
            "items": {
                "type": "string",
                "enum": list(ALLOWED_STEPS),
            },
        },
        "notes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "csv_path": {
            "type": ["string", "null"],
        },
        "preset_id": {
            "type": ["string", "null"],
            "description": (
                "Allowlisted experiment preset ID "
                "(sampling-cpu-100, sampling-cpu-1000, sampling-cpu-scaffold), "
                "or null to use the project's reinvent.toml. Never TOML text."
            ),
        },
        "scaffold_path": {
            "type": ["string", "null"],
        },
    },
    "required": ["steps", "notes", "csv_path", "preset_id", "scaffold_path"],
}


class PlanValidationError(ValueError):
    """Raised when a plan is not safe or does not match the schema."""


def resolve_allowlist(configured: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return the effective step allowlist.

    Config may only *narrow* the hardcoded set; unknown names are rejected
    so a typo cannot introduce a new tool.
    """
    if not configured:
        return ALLOWED_STEPS
    names = [str(item) for item in configured]
    unknown = [name for name in names if name not in ALLOWED_STEPS]
    if unknown:
        raise PlanValidationError(
            f"Config planner.allowlist contains unknown steps: {unknown}"
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for name in ALLOWED_STEPS:
        if name in names and name not in seen:
            seen.add(name)
            ordered.append(name)
    if not ordered:
        raise PlanValidationError("planner.allowlist is empty after filtering")
    return tuple(ordered)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_plan_csv_path(
    csv_path: str | Path,
    *,
    project_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Resolve a model/plan ``csv_path`` and require it stay under output_dir.

    Relative paths are tried against ``project_dir`` first (e.g.
    ``output/sampled.csv``), then against ``output_dir`` (e.g. ``sampled.csv``).
    Absolute paths are accepted only if they already resolve inside output_dir.
    The file must exist. Symlink escapes are rejected via ``Path.resolve``.
    """
    if csv_path is None or str(csv_path).strip() == "":
        raise PlanValidationError("csv_path is empty")

    raw = Path(str(csv_path).strip())
    project_dir = Path(project_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if raw.is_absolute():
        candidates = [raw.expanduser().resolve()]
    else:
        candidates = [
            (project_dir / raw).resolve(),
            (output_dir / raw).resolve(),
        ]

    under_output = [c for c in candidates if _is_under(c, output_dir)]
    if not under_output:
        raise PlanValidationError(
            f"csv_path escapes project output dir: {csv_path} (output_dir={output_dir})"
        )
    resolved = under_output[0]
    if not resolved.is_file():
        raise PlanValidationError(f"csv_path does not exist or is not a file: {resolved}")
    return resolved


def _reject_unknown_keys(mapping: dict[str, Any], allowed: frozenset[str], *, where: str) -> None:
    extra = sorted(set(mapping) - allowed)
    if extra:
        raise PlanValidationError(f"Unknown {where} keys: {extra}")


def _parse_steps(
    steps: Any,
    *,
    allowlist: Sequence[str],
    max_steps: int,
) -> tuple[list[str], str | None]:
    if not isinstance(steps, list) or not steps:
        raise PlanValidationError("steps must be a non-empty list")
    if len(steps) > max_steps:
        raise PlanValidationError(f"steps exceed max_steps ({len(steps)} > {max_steps})")

    allow = set(allowlist)
    names: list[str] = []
    csv_from_params: str | None = None

    for index, item in enumerate(steps):
        params: Any = None
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            _reject_unknown_keys(item, ALLOWED_STEP_OBJECT_KEYS, where="step")
            name = item.get("name")
            params = item.get("params")
        else:
            raise PlanValidationError(
                f"steps[{index}] must be a step name string or {{name, params?}} object"
            )

        if not isinstance(name, str) or not name:
            raise PlanValidationError(f"steps[{index}] has an invalid name")
        if name not in allow:
            raise PlanValidationError(f"Unknown plan step: {name}")
        if name in names:
            raise PlanValidationError(f"Duplicate plan step: {name}")

        if params is not None:
            if not isinstance(params, dict):
                raise PlanValidationError(f"params for {name} must be an object")
            allowed_params = ALLOWED_STEP_PARAMS.get(name, frozenset())
            extra = sorted(set(params) - allowed_params)
            if extra:
                raise PlanValidationError(f"Unknown params for step {name}: {extra}")
            if name == "analyze_molecules" and "csv_path" in params:
                value = params["csv_path"]
                if value is not None and not isinstance(value, str):
                    raise PlanValidationError("csv_path must be a string")
                if isinstance(value, str) and value.strip():
                    csv_from_params = value

        names.append(name)

    return names, csv_from_params


def _parse_notes(notes: Any) -> list[str]:
    if notes is None:
        return []
    if not isinstance(notes, list):
        raise PlanValidationError("notes must be a list of strings")
    if len(notes) > MAX_NOTES:
        raise PlanValidationError(f"notes exceed limit ({len(notes)} > {MAX_NOTES})")
    cleaned: list[str] = []
    for item in notes:
        if not isinstance(item, str):
            raise PlanValidationError("notes must be a list of strings")
        text = item.strip()
        if not text:
            continue
        if len(text) > MAX_NOTE_CHARS:
            raise PlanValidationError(
                f"note exceeds {MAX_NOTE_CHARS} characters (never used as a command)"
            )
        cleaned.append(text)
    return cleaned


def validate_plan(
    raw: Any,
    *,
    project_dir: str | Path,
    output_dir: str | Path | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    allowlist: Sequence[str] | None = None,
    skip_reinvent: bool = False,
    source: str = "any",
) -> dict[str, Any]:
    """Validate and normalize a plan dict.

    ``source="llm"`` rejects runtime-only keys so the model cannot set
    ``approve_run`` or inject extra fields. ``source="any"`` also accepts
    system keys (used for already-merged plans).

    Returns a normalized fragment: ``steps``, ``notes``, optional
    ``csv_path`` / ``step_params``, plus sanitization ``warnings``.
    """
    if not isinstance(raw, dict):
        raise PlanValidationError("plan must be a JSON object")

    if source == "llm":
        allowed_keys = LLM_PLAN_KEYS
    elif source == "any":
        allowed_keys = ALLOWED_PLAN_KEYS
    else:
        raise PlanValidationError(f"Unknown validation source: {source}")

    _reject_unknown_keys(raw, allowed_keys, where="plan")

    effective_allowlist = resolve_allowlist(allowlist)
    names, csv_from_step = _parse_steps(
        raw.get("steps"),
        allowlist=effective_allowlist,
        max_steps=max_steps,
    )

    top_csv = raw.get("csv_path")
    if top_csv is not None and not isinstance(top_csv, str):
        raise PlanValidationError("csv_path must be a string or null")
    if isinstance(top_csv, str) and not top_csv.strip():
        top_csv = None

    if csv_from_step and top_csv and csv_from_step.strip() != top_csv.strip():
        raise PlanValidationError("conflicting csv_path values at top level and step params")

    csv_raw = csv_from_step or top_csv
    warnings: list[str] = []

    if skip_reinvent:
        stripped = [s for s in names if s in ("run_reinvent", "prepare_execution")]
        if stripped:
            names = [s for s in names if s not in ("run_reinvent", "prepare_execution")]
            warnings.append(
                "Removed "
                + ", ".join(stripped)
                + " because skip_reinvent is set (CLI owns this flag)."
            )
        if not names:
            raise PlanValidationError("plan has no steps left after applying skip_reinvent")

    if csv_raw and "analyze_molecules" not in names:
        raise PlanValidationError("csv_path was provided but analyze_molecules is not in steps")

    normalized: dict[str, Any] = {
        "steps": names,
        "notes": _parse_notes(raw.get("notes")),
        "warnings": warnings,
        "step_params": {},
    }

    if csv_raw:
        if output_dir is None:
            output_dir = Path(project_dir) / "output"
        resolved = resolve_plan_csv_path(
            csv_raw, project_dir=project_dir, output_dir=output_dir
        )
        normalized["csv_path"] = str(resolved)
        normalized["step_params"] = {"analyze_molecules": {"csv_path": str(resolved)}}

    preset_id = raw.get("preset_id")
    if preset_id is not None and not isinstance(preset_id, str):
        raise PlanValidationError("preset_id must be a string or null")
    scaffold_raw = raw.get("scaffold_path")
    if scaffold_raw is not None and not isinstance(scaffold_raw, str):
        raise PlanValidationError("scaffold_path must be a string or null")

    try:
        preset = validate_preset_selection(
            preset_id,
            project_dir=project_dir,
            scaffold_path=scaffold_raw,
        )
    except PresetError as exc:
        raise PlanValidationError(str(exc)) from exc
    if preset:
        normalized["preset_id"] = preset["preset_id"]
        if preset.get("scaffold_path"):
            normalized["scaffold_path"] = preset["scaffold_path"]
        normalized["preset"] = {
            "preset_id": preset["preset_id"],
            "description": preset["description"],
            "source_toml": preset["source_toml"],
        }

    return normalized
