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
            run_type = str(toml_data.get("run_type") or "").strip().lower()
            if run_type == "transfer_learning":
                _validate_transfer_learning(project, params, details, errors, warnings)
            else:
                _validate_sampling(project, params, details, errors, warnings)
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


def _validate_sampling(
    project: Path,
    params: dict[str, Any],
    details: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    model_file = params.get("model_file")
    output_file = params.get("output_file")
    details["model_file"] = model_file
    details["output_file"] = output_file
    details["num_smiles"] = params.get("num_smiles")

    if not model_file:
        errors.append("parameters.model_file is missing")
    else:
        try:
            model_path = resolve_under_root(
                project, model_file, follow_symlinks=False
            )
        except PermissionError as exc:
            errors.append(str(exc))
            return
        details["model_path"] = str(model_path)
        if not model_path.is_file():
            errors.append(f"model_file not found: {model_path}")

    if not output_file:
        warnings.append("parameters.output_file is missing")
    else:
        try:
            output_path = resolve_under_root(project, output_file)
        except PermissionError as exc:
            errors.append(str(exc))
            return
        details["resolved_output_file"] = str(output_path)
        out_parent = output_path.parent
        if not out_parent.is_dir():
            warnings.append(
                f"output parent directory does not exist yet: {out_parent}"
            )


def _validate_transfer_learning(
    project: Path,
    params: dict[str, Any],
    details: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    input_model = params.get("input_model_file")
    smiles_file = params.get("smiles_file")
    output_model = params.get("output_model_file")
    validation_smiles = params.get("validation_smiles_file")
    details["input_model_file"] = input_model
    details["smiles_file"] = smiles_file
    details["output_model_file"] = output_model
    details["validation_smiles_file"] = validation_smiles
    details["num_epochs"] = params.get("num_epochs")

    if not input_model:
        errors.append("parameters.input_model_file is missing")
    else:
        try:
            input_path = resolve_under_root(
                project, input_model, follow_symlinks=False
            )
        except PermissionError as exc:
            errors.append(str(exc))
        else:
            details["input_model_path"] = str(input_path)
            if not input_path.is_file():
                errors.append(f"input_model_file not found: {input_path}")

    if not smiles_file:
        errors.append("parameters.smiles_file is missing")
    else:
        try:
            smiles_path = resolve_under_root(
                project, smiles_file, follow_symlinks=False
            )
        except PermissionError as exc:
            errors.append(str(exc))
        else:
            details["smiles_path"] = str(smiles_path)
            if not smiles_path.is_file():
                errors.append(f"smiles_file not found: {smiles_path}")

    if not output_model:
        errors.append("parameters.output_model_file is missing")
    else:
        try:
            output_path = resolve_under_root(project, output_model)
        except PermissionError as exc:
            errors.append(str(exc))
        else:
            details["resolved_output_model"] = str(output_path)
            if not output_path.parent.is_dir():
                warnings.append(
                    f"output model directory does not exist yet: {output_path.parent}"
                )

    if validation_smiles:
        try:
            val_path = resolve_under_root(
                project, validation_smiles, follow_symlinks=False
            )
        except PermissionError as exc:
            errors.append(str(exc))
        else:
            details["validation_smiles_path"] = str(val_path)
            if not val_path.is_file():
                errors.append(f"validation_smiles_file not found: {val_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a REINVENT project")
    parser.add_argument("--project", required=True, help="Project directory")
    args = parser.parse_args()
    print(json.dumps(validate_project(args.project), indent=2))


if __name__ == "__main__":
    main()
