"""HITL --from-run: copy CLI config from a previous result.json, never loop."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from analysis.report import generate_html_report
from main import build_parser, main
from tools.artifacts import create_run_dir, write_run_result
from tools.from_run import (
    COPYABLE_FIELDS,
    NEVER_COPY_FIELDS,
    FromRunError,
    apply_from_run,
    explicit_cli_flags,
    extract_invocation,
    resolve_from_run_ref,
    snapshot_invocation,
)

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "projects" / "demo_project"
SAMPLE_CSV = DEMO / "output" / "sampled-sample.csv"


def _write_result(logs_dir: Path, run_id: str, payload: dict) -> Path:
    run_dir = logs_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    path = run_dir / "result.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_explicit_cli_flags_detects_passed_names():
    argv = [
        "--from-run",
        "20260101_120000",
        "--preset",
        "sampling-cpu-1000",
        "--approve-run",
        "--yes",
    ]
    found = explicit_cli_flags(argv)
    assert found == {"from_run", "preset", "approve_run", "yes"}
    assert "project" not in found
    assert "skip_reinvent" not in found


def test_resolve_run_id_and_reject_escape(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    run_id = "20260101_120000"
    result = _write_result(logs_dir, run_id, {"goal": "x", "invocation": {}})
    got_id, got_path = resolve_from_run_ref(run_id, logs_dir=logs_dir)
    assert got_id == run_id
    assert got_path == result.resolve()

    with pytest.raises(FromRunError, match="must stay under"):
        resolve_from_run_ref("../secret", logs_dir=logs_dir)
    with pytest.raises(FromRunError, match="escapes"):
        resolve_from_run_ref(str(tmp_path / "outside.json"), logs_dir=logs_dir)
    with pytest.raises(FromRunError, match="No result.json"):
        resolve_from_run_ref("19990101_000000", logs_dir=logs_dir)


def test_resolve_last_and_path_forms(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    run_id = "20260102_010203"
    result = _write_result(
        logs_dir,
        run_id,
        {"goal": "offline", "invocation": {"project": "projects/demo_project"}},
    )
    (logs_dir / "last_run_summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(result.parent),
                "result_json": str(result),
            }
        ),
        encoding="utf-8",
    )
    last_id, last_path = resolve_from_run_ref("last", logs_dir=logs_dir)
    assert last_id == run_id
    assert last_path == result.resolve()

    via_dir, via_dir_path = resolve_from_run_ref(str(result.parent), logs_dir=logs_dir)
    assert via_dir == run_id
    assert via_dir_path == result.resolve()


def test_extract_invocation_ignores_toml_and_approval():
    payload = {
        "goal": "Sample 1000",
        "project_dir": str(DEMO),
        "invocation": {
            "project": "projects/demo_project",
            "goal": "Sample 1000",
            "preset": "sampling-cpu-1000",
            "seed": 7,
            "approve_run": True,
            "yes": True,
            "skip_reinvent": True,
            "toml": 'run_type = "sampling"',
            "command": ["reinvent", "-s", "7", "evil.toml"],
            "argv": "--approve-run",
        },
    }
    inv = extract_invocation(payload)
    assert inv["project"] == "projects/demo_project"
    assert inv["preset"] == "sampling-cpu-1000"
    assert inv["seed"] == 7
    assert "approve_run" not in inv
    assert "yes" not in inv
    assert "toml" not in inv
    assert "command" not in inv
    assert inv.get("skip_reinvent") is True  # audit only


def test_extract_invocation_rejects_unknown_preset():
    with pytest.raises(FromRunError, match="allowlist"):
        extract_invocation(
            {
                "invocation": {
                    "project": "projects/demo_project",
                    "preset": 'run_type = "sampling"\nnum_smiles = 9',
                }
            }
        )


def test_extract_invocation_reconstructs_old_result_json():
    payload = {
        "goal": "Dry run",
        "project_dir": str(DEMO),
        "plan": {
            "goal": "Dry run",
            "preset_id": "sampling-cpu-100",
            "scaffold_path": None,
            "planner": "deterministic",
            "skip_reinvent": False,
        },
        "analysis": {"csv_path": str(SAMPLE_CSV)},
        "critic": {"critic": "deterministic", "status": "WARNING"},
    }
    inv = extract_invocation(payload)
    assert inv["goal"] == "Dry run"
    assert inv["preset"] == "sampling-cpu-100"
    assert inv["project"] == "projects/demo_project"
    assert inv["planner"] == "deterministic"


def test_apply_from_run_copies_identity_not_approval(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    run_id = "20260303_030303"
    _write_result(
        logs_dir,
        run_id,
        {
            "invocation": {
                "project": "projects/demo_project",
                "goal": "Generate 1000 molecules.",
                "preset": "sampling-cpu-1000",
                "seed": 99,
                "planner": "deterministic",
                "csv": "projects/demo_project/output/sampled-sample.csv",
                "skip_reinvent": True,
                "approve_run": True,
                "yes": True,
            }
        },
    )
    parser = build_parser()
    argv = ["--from-run", run_id]
    args = parser.parse_args(argv)
    assert args.approve_run is False
    assert args.skip_reinvent is False
    meta = apply_from_run(
        args, logs_dir=logs_dir, explicit=explicit_cli_flags(argv)
    )
    assert args.project == "projects/demo_project"
    assert args.goal == "Generate 1000 molecules."
    assert args.preset == "sampling-cpu-1000"
    assert args.seed == 99
    assert args.approve_run is False
    assert args.yes is False
    assert args.skip_reinvent is False
    assert args.csv is None
    assert "csv" not in meta["copied"]
    assert meta["copied_approve_run"] is False
    assert meta["copied_toml"] is False
    assert "preset" in meta["copied"]


def test_apply_from_run_cli_override_wins_and_drops_scaffold(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    run_id = "20260404_040404"
    _write_result(
        logs_dir,
        run_id,
        {
            "invocation": {
                "project": "projects/demo_project",
                "goal": "From scaffold",
                "preset": "sampling-cpu-scaffold",
                "scaffold": "projects/demo_project/input/scaffold.smi",
            }
        },
    )
    parser = build_parser()
    argv = [
        "--from-run",
        run_id,
        "--preset",
        "sampling-cpu-1000",
        "--approve-run",
        "--yes",
    ]
    args = parser.parse_args(argv)
    meta = apply_from_run(
        args, logs_dir=logs_dir, explicit=explicit_cli_flags(argv)
    )
    assert args.preset == "sampling-cpu-1000"
    assert args.scaffold is None
    assert args.approve_run is True
    assert args.yes is True
    assert "preset" in meta["overridden"]
    assert "goal" in meta["copied"]


def test_apply_from_run_copies_csv_only_with_skip_reinvent(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    run_id = "20260505_050505"
    _write_result(
        logs_dir,
        run_id,
        {
            "invocation": {
                "project": "projects/demo_project",
                "goal": "Offline",
                "csv": "projects/demo_project/output/sampled-sample.csv",
                "skip_reinvent": True,
            }
        },
    )
    parser = build_parser()
    argv = ["--from-run", run_id, "--skip-reinvent"]
    args = parser.parse_args(argv)
    apply_from_run(args, logs_dir=logs_dir, explicit=explicit_cli_flags(argv))
    assert args.csv == "projects/demo_project/output/sampled-sample.csv"
    assert args.skip_reinvent is True


def test_apply_from_run_rejects_project_outside_repo(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    run_id = "20260606_060606"
    _write_result(
        logs_dir,
        run_id,
        {"invocation": {"project": "/etc", "goal": "nope"}},
    )
    parser = build_parser()
    argv = ["--from-run", run_id]
    args = parser.parse_args(argv)
    with pytest.raises(FromRunError, match="outside the repo"):
        apply_from_run(args, logs_dir=logs_dir, explicit=explicit_cli_flags(argv))


def test_snapshot_omits_approval_and_default_config():
    args = build_parser().parse_args(
        ["--project", "projects/demo_project", "--preset", "sampling-cpu-100"]
    )
    snap = snapshot_invocation(args)
    assert snap["preset"] == "sampling-cpu-100"
    assert snap["config"] is None
    assert "approve_run" not in snap
    assert "yes" not in snap
    for field in NEVER_COPY_FIELDS:
        assert field not in snap or field == "skip_reinvent"
    for field in COPYABLE_FIELDS:
        assert field in snap


def test_write_run_result_stores_invocation_not_approval(tmp_path: Path):
    run_dir = create_run_dir(tmp_path)
    out = write_run_result(
        run_dir,
        goal="test",
        plan={"steps": ["check_environment"], "preset_id": "sampling-cpu-100"},
        results={
            "project_dir": str(DEMO),
            "steps": {
                "run_reinvent": {"skipped": True},
                "analyze_molecules": {"ok": True, "analysis_source": "existing_csv"},
            },
            "warnings": [],
            "errors": [],
        },
        critic={"status": "WARNING"},
        report={"report_path": "/tmp/r.html"},
        exit_code=0,
        invocation={
            "project": "projects/demo_project",
            "preset": "sampling-cpu-100",
            "goal": "test",
        },
        from_run=None,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_dir.name
    assert payload["invocation"]["preset"] == "sampling-cpu-100"
    assert "approve_run" not in (payload["invocation"] or {})


def test_html_report_includes_from_run_hint(tmp_path: Path):
    meta = generate_html_report(
        {
            "goal": "rerun",
            "project": {"project_dir": str(DEMO)},
            "environment": {},
            "validation": {"ok": True},
            "execution": {"skipped": True},
            "inventory": {},
            "analysis": {"ok": True, "analysis_source": "existing_csv"},
            "critic": {"status": "WARNING", "issues": []},
            "warnings": [],
            "errors": [],
            "dry_run": True,
            "run_id": "20260707_070707",
            "from_run": {
                "run_id": "20260101_120000",
                "copied": ["project", "goal", "preset"],
                "overridden": ["preset"],
            },
        },
        reports_dir=tmp_path,
        filename="from_run.html",
    )
    html = Path(meta["report_path"]).read_text(encoding="utf-8")
    assert "Human rerun" in html
    assert "python main.py --from-run 20260707_070707 --approve-run" in html
    assert "will not edit TOML or loop" in html
    assert "single-shot" in html


def test_cli_from_run_unknown_id_exits_2(capsys):
    argv = ["--from-run", "19990101_000000", "--skip-reinvent"]
    code = main(argv)
    assert code == 2
    err = capsys.readouterr().out
    assert "--from-run failed" in err


def test_main_from_run_offline_copies_preset():
    logs_dir = REPO / "logs"
    run_id = "19700101_000042"
    run_dir = logs_dir / "runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "goal": "Copy me then stay offline",
        "invocation": {
            "project": "projects/demo_project",
            "goal": "Copy me then stay offline",
            "preset": "sampling-cpu-1000",
            "seed": 42,
        },
        "plan": {"preset_id": "sampling-cpu-1000", "skip_reinvent": False},
    }
    (run_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    try:
        with patch("agents.executor.run_reinvent") as mocked:
            mocked.return_value = {
                "approved": False,
                "skipped": True,
                "success": False,
                "command": [],
                "message": "REINVENT not launched (missing --approve-run).",
            }
            code = main(
                [
                    "--from-run",
                    run_id,
                    "--skip-reinvent",
                    "--csv",
                    str(SAMPLE_CSV),
                ]
            )
        assert code == 0
        mocked.assert_not_called()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_main_from_run_does_not_launch_without_approve():
    logs_dir = REPO / "logs"
    run_id = "19700101_000043"
    run_dir = logs_dir / "runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    payload = {
        "invocation": {
            "project": "projects/demo_project",
            "goal": "Need a human to approve again",
            "preset": "sampling-cpu-100",
        }
    }
    (run_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    try:
        with patch("agents.executor.run_reinvent") as mocked:
            mocked.return_value = {
                "approved": False,
                "skipped": True,
                "success": False,
                "command": [],
                "message": "REINVENT not launched (missing --approve-run).",
            }
            code = main(["--from-run", run_id])
        assert code == 0
        mocked.assert_called_once()
        assert mocked.call_args.kwargs.get("approve") is False
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_cli_from_run_flag_exists():
    parser = build_parser()
    assert parser.parse_args([]).from_run is None
    assert parser.parse_args(["--from-run", "20260101_120000"]).from_run == (
        "20260101_120000"
    )
    assert parser.parse_args(["--from-run", "last"]).from_run == "last"
