"""Human-written experiment presets — IDs only, never generated TOML."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.experiment_presets import (
    ALLOWED_PRESET_IDS,
    PresetError,
    materialize_preset,
    relative_scaffold_for_toml,
    resolve_scaffold_path,
    validate_preset_selection,
)
from agents.executor import ExecutionAgent
from agents.llm_planner import LLMPlannerAgent
from agents.plan_schema import LLM_PLAN_JSON_SCHEMA, PlanValidationError, validate_plan
from agents.planner import PlannerAgent
from main import build_parser

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "projects" / "demo_project"
OUTPUT = DEMO / "output"
INPUT = DEMO / "input"
SCAFFOLD = INPUT / "scaffold.smi"
EXPERIMENTS = REPO / "experiments"

FULL_STEPS = list(PlannerAgent.DEFAULT_STEPS)
SKIP_STEPS = [
    "check_environment",
    "validate_project",
    "find_output",
    "analyze_molecules",
    "generate_report",
    "critic_review",
]


def _validate(raw, **kwargs):
    defaults = {
        "project_dir": DEMO,
        "output_dir": OUTPUT,
        "source": "llm",
    }
    defaults.update(kwargs)
    return validate_plan(raw, **defaults)


def _llm_payload(**overrides):
    payload = {
        "steps": SKIP_STEPS,
        "notes": [],
        "csv_path": None,
        "preset_id": None,
        "scaffold_path": None,
    }
    payload.update(overrides)
    return payload


def test_human_written_presets_exist_and_differ():
    assert ALLOWED_PRESET_IDS == (
        "sampling-cpu-100",
        "sampling-cpu-1000",
        "sampling-cpu-scaffold",
    )
    with (EXPERIMENTS / "sampling-cpu-100.toml").open("rb") as fh:
        n100 = tomllib.load(fh)
    with (EXPERIMENTS / "sampling-cpu-1000.toml").open("rb") as fh:
        n1000 = tomllib.load(fh)
    assert n100["parameters"]["num_smiles"] == 100
    assert n1000["parameters"]["num_smiles"] == 1000
    assert n100["run_type"] == "sampling"
    assert n100["device"] == "cpu"
    scaffold_text = (EXPERIMENTS / "sampling-cpu-scaffold.toml").read_text(
        encoding="utf-8"
    )
    assert "{{scaffold_path}}" in scaffold_text
    assert SCAFFOLD.is_file()


def test_validate_plan_accepts_preset_id():
    result = _validate(_llm_payload(preset_id="sampling-cpu-1000", steps=FULL_STEPS))
    assert result["preset_id"] == "sampling-cpu-1000"
    assert result["preset"]["source_toml"].endswith("sampling-cpu-1000.toml")


def test_validate_plan_rejects_unknown_preset():
    with pytest.raises(PlanValidationError, match="Unknown preset_id"):
        _validate(_llm_payload(preset_id="gpu-docking-secret", steps=FULL_STEPS))


def test_validate_plan_rejects_toml_text_key():
    with pytest.raises(PlanValidationError, match="Unknown plan keys"):
        _validate(
            {
                "steps": FULL_STEPS,
                "notes": [],
                "csv_path": None,
                "toml": 'run_type = "sampling"\nnum_smiles = 9999\n',
            }
        )


def test_validate_plan_rejects_config_text_key():
    with pytest.raises(PlanValidationError, match="Unknown plan keys"):
        _validate(
            {
                "steps": ["run_reinvent"],
                "config_text": "run_type = 'sampling'",
            }
        )


def test_validate_plan_rejects_toml_string_as_preset_id():
    with pytest.raises(PlanValidationError, match="Unknown preset_id"):
        _validate(
            _llm_payload(
                preset_id='run_type = "sampling"\n[parameters]\nnum_smiles = 50\n',
                steps=FULL_STEPS,
            )
        )


def test_validate_plan_accepts_existing_scaffold_under_input():
    result = _validate(
        _llm_payload(
            preset_id="sampling-cpu-scaffold",
            scaffold_path="input/scaffold.smi",
            steps=FULL_STEPS,
        )
    )
    assert result["preset_id"] == "sampling-cpu-scaffold"
    assert Path(result["scaffold_path"]) == SCAFFOLD.resolve()


def test_validate_plan_accepts_repo_relative_scaffold_from_cwd():
    result = _validate(
        _llm_payload(
            preset_id="sampling-cpu-scaffold",
            scaffold_path="projects/demo_project/input/scaffold.smi",
            steps=FULL_STEPS,
        )
    )
    assert Path(result["scaffold_path"]) == SCAFFOLD.resolve()


def test_validate_plan_rejects_scaffold_escape(tmp_path: Path):
    outsider = tmp_path / "outside.smi"
    outsider.write_text("C\n", encoding="utf-8")
    with pytest.raises(PlanValidationError, match="escapes"):
        _validate(
            _llm_payload(
                preset_id="sampling-cpu-scaffold",
                scaffold_path=str(outsider),
                steps=FULL_STEPS,
            )
        )


def test_validate_plan_rejects_scaffold_dotdot():
    with pytest.raises(PlanValidationError, match="escapes"):
        _validate(
            _llm_payload(
                preset_id="sampling-cpu-scaffold",
                scaffold_path="../reinvent.toml",
                steps=FULL_STEPS,
            )
        )


def test_validate_plan_rejects_scaffold_for_non_scaffold_preset():
    with pytest.raises(PlanValidationError, match="not an allowlisted field"):
        _validate(
            _llm_payload(
                preset_id="sampling-cpu-100",
                scaffold_path="input/scaffold.smi",
                steps=FULL_STEPS,
            )
        )


def test_validate_plan_rejects_scaffold_preset_without_path():
    with pytest.raises(PlanValidationError, match="requires scaffold_path"):
        _validate(
            _llm_payload(preset_id="sampling-cpu-scaffold", steps=FULL_STEPS)
        )


def test_llm_plan_schema_forbids_toml_and_requires_preset_id():
    dumped = json.dumps(LLM_PLAN_JSON_SCHEMA)
    schema = json.loads(dumped)
    assert schema["additionalProperties"] is False
    props = schema["properties"]
    assert "preset_id" in props
    assert "scaffold_path" in props
    for banned in ("toml", "config", "config_text", "argv", "command", "shell"):
        assert banned not in props
    assert "preset_id" in schema["required"]


def test_deterministic_planner_records_preset():
    plan = PlannerAgent().create_plan(
        "Sample 1000 molecules",
        project_dir=str(DEMO),
        approve_run=False,
        preset_id="sampling-cpu-1000",
    )
    assert plan["preset_id"] == "sampling-cpu-1000"
    assert plan["planner"] == "deterministic"
    assert any("sampling-cpu-1000" in n for n in plan["notes"])


def test_deterministic_planner_rejects_illegal_preset():
    with pytest.raises(PlanValidationError, match="Unknown preset_id"):
        PlannerAgent().create_plan(
            "evil",
            project_dir=str(DEMO),
            approve_run=False,
            preset_id="not-a-preset",
        )


def _agent(complete_fn=None, **planner_cfg):
    cfg = {
        "project": {"output_dirname": "output", "input_dirname": "input"},
        "planner": {
            "mode": "llm",
            "fallback_on_error": True,
            "max_steps": 8,
            **planner_cfg,
        },
    }
    return LLMPlannerAgent(cfg, complete_fn=complete_fn)


def test_llm_planner_valid_preset_id():
    def complete(_messages):
        return json.dumps(_llm_payload(preset_id="sampling-cpu-100", steps=FULL_STEPS))

    plan = _agent(complete_fn=complete).create_plan(
        "Generate 100 molecules",
        project_dir=str(DEMO),
        approve_run=False,
    )
    assert plan["planner"] == "llm"
    assert plan["planner_fallback"] is False
    assert plan["preset_id"] == "sampling-cpu-100"


def test_llm_planner_unknown_preset_falls_back(caplog):
    def complete(_messages):
        return json.dumps(_llm_payload(preset_id="md-gromacs", steps=FULL_STEPS))

    with caplog.at_level("WARNING"):
        plan = _agent(complete_fn=complete).create_plan(
            "goal",
            project_dir=str(DEMO),
            approve_run=False,
        )
    assert plan["planner"] == "deterministic"
    assert plan["planner_fallback"] is True
    assert plan.get("preset_id") is None
    assert any("Unknown preset_id" in w for w in plan["planner_warnings"])
    assert any("falling back" in rec.message.lower() for rec in caplog.records)


def test_llm_planner_toml_key_falls_back(caplog):
    def complete(_messages):
        return json.dumps(
            {
                "steps": FULL_STEPS,
                "notes": [],
                "csv_path": None,
                "toml": "run_type = \"sampling\"\n",
            }
        )

    plan = _agent(complete_fn=complete).create_plan(
        "goal",
        project_dir=str(DEMO),
        approve_run=False,
    )
    assert plan["planner_fallback"] is True
    assert plan.get("preset_id") is None
    assert any("Unknown plan keys" in w for w in plan["planner_warnings"])


def test_llm_planner_scaffold_escape_falls_back(tmp_path: Path):
    outsider = tmp_path / "pwn.smi"
    outsider.write_text("C\n", encoding="utf-8")

    def complete(_messages):
        return json.dumps(
            _llm_payload(
                preset_id="sampling-cpu-scaffold",
                scaffold_path=str(outsider),
                steps=FULL_STEPS,
            )
        )

    plan = _agent(complete_fn=complete).create_plan(
        "from this scaffold optimize QED",
        project_dir=str(DEMO),
        approve_run=False,
    )
    assert plan["planner_fallback"] is True
    assert plan.get("preset_id") is None
    assert any("escapes" in w for w in plan["planner_warnings"])


def test_llm_planner_valid_scaffold_preset():
    def complete(_messages):
        return json.dumps(
            _llm_payload(
                preset_id="sampling-cpu-scaffold",
                scaffold_path="input/scaffold.smi",
                steps=FULL_STEPS,
            )
        )

    plan = _agent(complete_fn=complete).create_plan(
        "Generate from scaffold and report QED",
        project_dir=str(DEMO),
        approve_run=False,
    )
    assert plan["planner"] == "llm"
    assert plan["preset_id"] == "sampling-cpu-scaffold"
    assert Path(plan["scaffold_path"]) == SCAFFOLD.resolve()


def test_materialize_static_preset(tmp_path: Path):
    meta = materialize_preset("sampling-cpu-1000", project_dir=tmp_path)
    dest = Path(meta["config_path"])
    assert dest.is_file()
    assert dest.parent.name == ".agent"
    with dest.open("rb") as fh:
        data = tomllib.load(fh)
    assert data["parameters"]["num_smiles"] == 1000
    assert "{{" not in dest.read_text(encoding="utf-8")


def test_materialize_scaffold_fills_only_allowlisted_path(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    smiles = input_dir / "scaffold.smi"
    smiles.write_text("c1ccccc1\n", encoding="utf-8")
    meta = materialize_preset(
        "sampling-cpu-scaffold",
        project_dir=tmp_path,
        scaffold_path="input/scaffold.smi",
    )
    text = Path(meta["config_path"]).read_text(encoding="utf-8")
    assert 'smiles_file = "input/scaffold.smi"' in text
    assert "{{scaffold_path}}" not in text
    with Path(meta["config_path"]).open("rb") as fh:
        data = tomllib.load(fh)
    assert data["parameters"]["smiles_file"] == "input/scaffold.smi"


def test_materialize_rejects_path_injection(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with pytest.raises(PresetError):
        materialize_preset(
            "sampling-cpu-scaffold",
            project_dir=tmp_path,
            scaffold_path='input/foo"\nnum_smiles = 1',
        )


def test_resolve_scaffold_rejects_symlink_escape(tmp_path: Path):
    project = tmp_path / "proj"
    input_dir = project / "input"
    input_dir.mkdir(parents=True)
    outside = tmp_path / "secret.smi"
    outside.write_text("C\n", encoding="utf-8")
    link = input_dir / "link.smi"
    link.symlink_to(outside)
    for named in ("link.smi", "input/link.smi"):
        with pytest.raises(PresetError, match="escapes") as excinfo:
            resolve_scaffold_path(named, project_dir=project, input_dir=input_dir)
        assert str(outside.resolve()) not in str(excinfo.value)


def test_relative_scaffold_rejects_unsafe_chars(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    weird = project / "input"
    weird.mkdir()
    # A resolved path outside relative_to would already fail; this covers the regex.
    with pytest.raises(PresetError, match="not a safe relative path"):
        relative_scaffold_for_toml(project / "x y.smi", project_dir=project)


def test_executor_preset_uses_materialized_toml_without_approve():
    plan = PlannerAgent().create_plan(
        "Sample 1000",
        project_dir=str(DEMO),
        approve_run=False,
        preset_id="sampling-cpu-1000",
    )
    with patch("agents.executor.run_reinvent") as mocked:
        mocked.return_value = {
            "approved": False,
            "skipped": True,
            "success": False,
            "command": [],
            "message": "REINVENT not launched (missing --approve-run).",
        }
        results = ExecutionAgent().execute(
            plan, project_dir=DEMO, approve_run=False
        )
    mocked.assert_called_once()
    config_arg = Path(mocked.call_args.args[0])
    assert config_arg.name == "sampling-cpu-1000.toml"
    assert config_arg.parent.name == ".agent"
    assert mocked.call_args.kwargs.get("approve") is False
    assert results["preset"]["preset_id"] == "sampling-cpu-1000"
    run = results["steps"]["run_reinvent"]
    assert run["skipped"] is True


def test_executor_rejects_out_of_sandbox_scaffold_at_runtime():
    plan = {
        "goal": "pwn",
        "project_dir": str(DEMO),
        "approve_run": True,
        "skip_reinvent": False,
        "steps": ["run_reinvent"],
        "preset_id": "sampling-cpu-scaffold",
        "scaffold_path": str(REPO / "README.md"),
    }
    results = ExecutionAgent().execute(plan, project_dir=DEMO, approve_run=True)
    assert results.get("aborted") is True
    assert any("preset" in e.lower() for e in results["errors"])


def test_cli_preset_choices():
    parser = build_parser()
    assert parser.parse_args([]).preset is None
    assert parser.parse_args(["--preset", "sampling-cpu-100"]).preset == "sampling-cpu-100"
    assert parser.parse_args(["--preset", "sampling-cpu-1000"]).preset == "sampling-cpu-1000"
    with pytest.raises(SystemExit):
        parser.parse_args(["--preset", "write-toml-please"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--preset", 'run_type = "sampling"'])


def test_validate_preset_selection_null_is_noop():
    assert validate_preset_selection(None, project_dir=DEMO) == {}
    assert validate_preset_selection("", project_dir=DEMO) == {}
