"""Allowlisted MCP wrappers around existing REINVENT4 Agent tools.

This module is the security boundary for the MCP entry point. It does **not**
add capabilities: every tool name is an existing planner step, every call
forwards to an existing Python function, and path/argv rules match the
executor / plan schema (no shell, no TOML edits, no extra steps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agents.critic import CriticAgent
from agents.plan_schema import ALLOWED_STEPS, PlanValidationError, resolve_plan_csv_path
from analysis.molecule_analysis import analyze_molecules
from analysis.report import generate_html_report
from tools import REPO_ROOT, load_agent_config, to_jsonable
from tools.environment import check_environment
from tools.files import validate_project
from tools.reinvent import find_output_files, run_reinvent

MCP_SERVER_NAME = "reinvent-agent"
MCP_SERVER_VERSION = "0.6.0"

# Identical to the planner allowlist — MCP must not grow a parallel tool set.
MCP_TOOL_NAMES: tuple[str, ...] = ALLOWED_STEPS

FORBIDDEN_TOOL_NAMES: tuple[str, ...] = (
    "shell",
    "run_shell",
    "bash",
    "sh",
    "execute",
    "execute_command",
    "run_command",
    "subprocess",
    "popen",
)


class McpAllowlistError(ValueError):
    """Raised when an MCP call names an unknown tool or breaks sandbox rules."""


def _schema(
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


_PROJECT_DIR_PROP = {
    "type": "string",
    "description": (
        "REINVENT project directory. Relative paths resolve from the repo root. "
        "Must stay under the repository (sandbox); escapes are rejected."
    ),
}

MCP_TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "check_environment",
        "description": (
            "Inspect local Python, REINVENT, RDKit, and GPU. "
            "Never installs packages and never runs arbitrary shell."
        ),
        "input_schema": _schema({}),
        "required": (),
        "optional": (),
    },
    {
        "name": "validate_project",
        "description": (
            "Read-only validation of a REINVENT project (toml present, prior path, "
            "output dir). Does not modify reinvent.toml."
        ),
        "input_schema": _schema(
            {"project_dir": _PROJECT_DIR_PROP},
            required=["project_dir"],
        ),
        "required": ("project_dir",),
        "optional": (),
    },
    {
        "name": "prepare_execution",
        "description": (
            "Show the predefined REINVENT config and output paths for a project. "
            "Does not launch REINVENT."
        ),
        "input_schema": _schema(
            {"project_dir": _PROJECT_DIR_PROP},
            required=["project_dir"],
        ),
        "required": ("project_dir",),
        "optional": (),
    },
    {
        "name": "run_reinvent",
        "description": (
            "Launch the predefined command "
            "`reinvent -l <log> -s <seed> <project>/reinvent.toml` only. "
            "Requires approve_run=true (same as CLI --approve-run). "
            "Does not accept shell, argv, or executable overrides. "
            "Default approve_run=false skips launch."
        ),
        "input_schema": _schema(
            {
                "project_dir": _PROJECT_DIR_PROP,
                "approve_run": {
                    "type": "boolean",
                    "description": (
                        "Human approval to launch the predefined REINVENT command. "
                        "Default false (dry-run / skip)."
                    ),
                },
            },
            required=["project_dir"],
        ),
        "required": ("project_dir",),
        "optional": ("approve_run",),
    },
    {
        "name": "find_output",
        "description": (
            "Inventory files under <project>/output/ only. Cannot list arbitrary paths."
        ),
        "input_schema": _schema(
            {"project_dir": _PROJECT_DIR_PROP},
            required=["project_dir"],
        ),
        "required": ("project_dir",),
        "optional": (),
    },
    {
        "name": "analyze_molecules",
        "description": (
            "Analyze an existing molecule CSV. csv_path must already exist under "
            "<project>/output/ (same rule as the LLM planner). "
            "Computes counts, RDKit descriptors, SA/PAINS, histograms."
        ),
        "input_schema": _schema(
            {
                "project_dir": _PROJECT_DIR_PROP,
                "csv_path": {
                    "type": "string",
                    "description": (
                        "Existing CSV under the project output directory "
                        "(e.g. output/sampled-sample.csv)."
                    ),
                },
            },
            required=["project_dir", "csv_path"],
        ),
        "required": ("project_dir", "csv_path"),
        "optional": (),
    },
    {
        "name": "generate_report",
        "description": (
            "Write the self-contained HTML report from a structured payload "
            "(same helper as the CLI). Does not run shell or edit TOML."
        ),
        "input_schema": _schema(
            {
                "payload": {
                    "type": "object",
                    "description": (
                        "Report payload: goal, analysis, critic, environment, etc. "
                        "Same structure the CLI passes to generate_html_report."
                    ),
                },
            },
            required=["payload"],
        ),
        "required": ("payload",),
        "optional": (),
    },
    {
        "name": "critic_review",
        "description": (
            "Deterministic scientific critic over structured execution results. "
            "Evidence only; no docking/MD/literature claims, no tools, no shell."
        ),
        "input_schema": _schema(
            {
                "results": {
                    "type": "object",
                    "description": (
                        "Executor-style results dict (plan + steps + warnings/errors)."
                    ),
                },
            },
            required=["results"],
        ),
        "required": ("results",),
        "optional": (),
    },
)

MCP_TOOL_SPEC_BY_NAME: dict[str, dict[str, Any]] = {
    spec["name"]: spec for spec in MCP_TOOL_SPECS
}

ALLOWED_ARGS: dict[str, frozenset[str]] = {
    spec["name"]: frozenset(spec["required"]) | frozenset(spec["optional"])
    for spec in MCP_TOOL_SPECS
}


def list_mcp_tools() -> list[dict[str, Any]]:
    """Public tool catalog for MCP ``tools/list`` (name, description, schema)."""
    return [
        {
            "name": spec["name"],
            "description": spec["description"],
            "inputSchema": spec["input_schema"],
        }
        for spec in MCP_TOOL_SPECS
    ]


def resolve_mcp_project_dir(project_dir: str | Path) -> Path:
    """Resolve ``project_dir`` and require it stay under the repo root."""
    if not isinstance(project_dir, (str, Path)) or str(project_dir).strip() == "":
        raise McpAllowlistError("project_dir is required")
    raw = Path(str(project_dir).strip()).expanduser()
    candidate = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise McpAllowlistError(
            f"project_dir escapes repo root: {candidate} (root={REPO_ROOT})"
        ) from exc
    if not candidate.is_dir():
        raise McpAllowlistError(
            f"project_dir does not exist or is not a directory: {candidate}"
        )
    return candidate


def _require_args(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise McpAllowlistError(f"arguments for {name} must be an object")
    allowed = ALLOWED_ARGS[name]
    extra = sorted(set(arguments) - allowed)
    if extra:
        raise McpAllowlistError(f"Unknown arguments for {name}: {extra}")
    missing = [key for key in MCP_TOOL_SPEC_BY_NAME[name]["required"] if key not in arguments]
    if missing:
        raise McpAllowlistError(f"Missing required arguments for {name}: {missing}")
    return arguments


def _load_config() -> dict[str, Any]:
    return load_agent_config()


def _output_dir(project: Path, cfg: dict[str, Any]) -> Path:
    output_dirname = cfg.get("project", {}).get("output_dirname", "output")
    return project / output_dirname


def _config_path(project: Path, cfg: dict[str, Any]) -> Path:
    config_name = cfg.get("project", {}).get("config_name", "reinvent.toml")
    return project / config_name


def _handle_check_environment(_arguments: dict[str, Any]) -> dict[str, Any]:
    return check_environment()


def _handle_validate_project(arguments: dict[str, Any]) -> dict[str, Any]:
    project = resolve_mcp_project_dir(arguments["project_dir"])
    return validate_project(project, agent_config=_load_config())


def _handle_prepare_execution(arguments: dict[str, Any]) -> dict[str, Any]:
    project = resolve_mcp_project_dir(arguments["project_dir"])
    cfg = _load_config()
    config_path = _config_path(project, cfg)
    output_dir = _output_dir(project, cfg)
    return {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "approve_run": False,
        "ready": config_path.is_file(),
    }


def _handle_run_reinvent(arguments: dict[str, Any]) -> dict[str, Any]:
    approve = arguments.get("approve_run", False)
    if "approve_run" in arguments and not isinstance(approve, bool):
        raise McpAllowlistError("approve_run must be a boolean")
    project = resolve_mcp_project_dir(arguments["project_dir"])
    cfg = _load_config()
    return run_reinvent(
        _config_path(project, cfg),
        approve=bool(approve),
        project_dir=project,
        agent_config=cfg,
    )


def _handle_find_output(arguments: dict[str, Any]) -> dict[str, Any]:
    project = resolve_mcp_project_dir(arguments["project_dir"])
    cfg = _load_config()
    return find_output_files(_output_dir(project, cfg), project_dir=project)


def _handle_analyze_molecules(arguments: dict[str, Any]) -> dict[str, Any]:
    csv_path = arguments["csv_path"]
    if not isinstance(csv_path, str) or not csv_path.strip():
        raise McpAllowlistError("csv_path must be a non-empty string")
    project = resolve_mcp_project_dir(arguments["project_dir"])
    cfg = _load_config()
    output_dir = _output_dir(project, cfg)
    try:
        resolved = resolve_plan_csv_path(
            csv_path, project_dir=project, output_dir=output_dir
        )
    except PlanValidationError as exc:
        raise McpAllowlistError(str(exc)) from exc
    analysis_cfg = cfg.get("analysis") or {}
    raw_thr = analysis_cfg.get("qed_pass_threshold", 0.5)
    qed_pass_threshold = 0.5 if raw_thr is None else float(raw_thr)
    return analyze_molecules(resolved, qed_pass_threshold=qed_pass_threshold)


def _handle_generate_report(arguments: dict[str, Any]) -> dict[str, Any]:
    payload = arguments["payload"]
    if isinstance(payload, str):
        import json

        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise McpAllowlistError("payload must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise McpAllowlistError("payload must be an object")
    return generate_html_report(payload)


def _handle_critic_review(arguments: dict[str, Any]) -> dict[str, Any]:
    results = arguments["results"]
    if isinstance(results, str):
        import json

        try:
            results = json.loads(results)
        except json.JSONDecodeError as exc:
            raise McpAllowlistError("results must be a JSON object") from exc
    if not isinstance(results, dict):
        raise McpAllowlistError("results must be an object")
    return CriticAgent(agent_config=_load_config()).review(results)


_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "check_environment": _handle_check_environment,
    "validate_project": _handle_validate_project,
    "prepare_execution": _handle_prepare_execution,
    "run_reinvent": _handle_run_reinvent,
    "find_output": _handle_find_output,
    "analyze_molecules": _handle_analyze_molecules,
    "generate_report": _handle_generate_report,
    "critic_review": _handle_critic_review,
}


def invoke_mcp_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch one allowlisted tool. Unknown names and extra keys are errors."""
    if not isinstance(name, str) or not name:
        raise McpAllowlistError("tool name is required")
    if name not in _HANDLERS:
        raise McpAllowlistError(
            f"Unknown tool {name!r}. Allowlist: {list(MCP_TOOL_NAMES)}"
        )
    args = _require_args(name, arguments or {})
    result = _HANDLERS[name](args)
    if not isinstance(result, dict):
        raise McpAllowlistError(f"internal error: {name} did not return an object")
    return to_jsonable(result)
