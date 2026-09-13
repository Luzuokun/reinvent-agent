"""Project / config validation — read-only."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from tools import load_agent_config, resolve_under_root


def validate_project(
    project_dir: str | Path,
    *,
    config_name: str | None = None,
    agent_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a REINVENT project directory without modifying files."""
    cfg = agent_config or load_agent_config()
    project = Path(project_dir).expanduser().resolve()
    config_name = config_name or cfg.get("project", {}).get("config_name", "reinvent.toml")
    input_dirname = cfg.get("project", {}).get("input_dirname", "input")
    output_dirname = cfg.get("project", {}).get("output_dirname", "output")

    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {
        "project_dir": str(project),
        "config_name": config_name,
    }

    if not project.is_dir():
        return {
            "ok": False,
            "errors": [f"Project directory does not exist: {project}"],
            "warnings": warnings,
            "details": details,
        }

    config_path = project / config_name
    details["config_path"] = str(config_path)
    if not config_path.is_file():
        errors.append(f"REINVENT config missing: {config_path}")
    else:
        try:
            with config_path.open("rb") as fh:
                toml_data = tomllib.load(fh)
            details["run_type"] = toml_data.get("run_type")
            details["device"] = toml_data.get("device")
            params = toml_data.get("parameters") or {}
            if not isinstance(params, dict):
                errors.append("[parameters] section must be a table")
                params = {}
            model_file = params.get("model_file")
            output_file = params.get("output_file")
            details["model_file"] = model_file
            details["output_file"] = output_file
            details["num_smiles"] = params.get("num_smiles")

            if not model_file:
                errors.append("parameters.model_file is missing")
            else:
                model_path = resolve_under_root(project, model_file)
                details["model_path"] = str(model_path)
                if not model_path.is_file():
                    errors.append(f"model_file not found: {model_path}")

            if not output_file:
                warnings.append("parameters.output_file is missing")
            else:
                output_path = resolve_under_root(project, output_file)
                details["resolved_output_file"] = str(output_path)
                out_parent = output_path.parent
                if not out_parent.is_dir():
                    warnings.append(
                        f"output parent directory does not exist yet: {out_parent}"
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Failed to parse TOML: {exc}")

    input_dir = project / input_dirname
    output_dir = project / output_dirname
    details["input_dir"] = str(input_dir)
    details["output_dir"] = str(output_dir)
    if not input_dir.is_dir():
        warnings.append(f"input directory missing: {input_dir}")
    if not output_dir.is_dir():
        warnings.append(f"output directory missing: {output_dir}")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a REINVENT project")
    parser.add_argument("--project", required=True, help="Project directory")
    args = parser.parse_args()
    print(json.dumps(validate_project(args.project), indent=2))


if __name__ == "__main__":
    main()
