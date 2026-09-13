"""Tests for run artefacts and human-approval helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path

from tools.artifacts import create_run_dir, write_run_result
from tools.reinvent import confirm_reinvent_launch, prepare_reinvent_command

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "projects" / "demo_project" / "reinvent.toml"


def test_write_run_result(tmp_path: Path):
    run_dir = create_run_dir(tmp_path)
    out = write_run_result(
        run_dir,
        goal="test goal",
        plan={"steps": ["check_environment"]},
        results={
            "project_dir": str(REPO),
            "steps": {
                "check_environment": {"reinvent": True},
                "validate_project": {"ok": True},
                "run_reinvent": {"skipped": True},
                "analyze_molecules": {"ok": True, "total_molecules": 1},
            },
            "warnings": [],
            "errors": [],
        },
        critic={"status": "WARNING", "issues": ["dry-run"]},
        report={"report_path": "/tmp/report.html"},
        exit_code=0,
    )
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["goal"] == "test goal"
    assert payload["critic_status"] == "WARNING"
    assert payload["exit_code"] == 0
    assert payload["plan"]["steps"] == ["check_environment"]


def test_prepare_reinvent_command_shape():
    prepared = prepare_reinvent_command(CONFIG, seed=42)
    assert prepared["ok"] is True
    assert prepared["command"][1] == "-l"
    assert prepared["command"][3] == "-s"
    assert prepared["command"][4] == "42"
    assert prepared["command"][5].endswith("reinvent.toml")


def test_confirm_assumes_yes():
    prepared = {"command": ["reinvent", "-l", "x.log", "-s", "42", "c.toml"], "cwd": "/tmp"}
    assert confirm_reinvent_launch(prepared, assume_yes=True, stdout=io.StringIO()) is True


def test_confirm_noninteractive_requires_yes():
    prepared = {"command": ["reinvent", "-l", "x.log", "-s", "42", "c.toml"], "cwd": "/tmp"}
    stdin = io.StringIO("y\n")  # not a TTY
    assert confirm_reinvent_launch(prepared, assume_yes=False, stdin=stdin, stdout=io.StringIO()) is False


def test_write_run_result_marks_existing_csv_source(tmp_path: Path):
    run_dir = create_run_dir(tmp_path)
    out = write_run_result(
        run_dir,
        goal="dry-run labeling",
        plan={"steps": ["run_reinvent", "analyze_molecules"], "approve_run": False},
        results={
            "project_dir": str(REPO),
            "steps": {
                "run_reinvent": {"skipped": True, "success": False},
                "analyze_molecules": {
                    "ok": True,
                    "csv_path": "/tmp/sampled.csv",
                    "total_molecules": 3,
                    "analysis_source": "existing_csv",
                    "from_fresh_reinvent": False,
                },
            },
            "warnings": [],
            "errors": [],
        },
        critic={"status": "WARNING", "issues": ["Dry-run only"]},
        report={"report_path": "/tmp/report.html"},
        exit_code=0,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["analysis_source"] == "existing_csv"
    assert payload["from_fresh_reinvent"] is False


def test_html_report_dry_run_banner(tmp_path: Path):
    from analysis.report import generate_html_report

    meta = generate_html_report(
        {
            "goal": "dry run",
            "project": {"project_dir": str(REPO)},
            "environment": {},
            "validation": {"ok": True},
            "execution": {"skipped": True, "message": "missing --approve-run"},
            "inventory": {},
            "analysis": {
                "ok": True,
                "csv_path": str(REPO / "projects/demo_project/output/sampled.csv"),
                "analysis_source": "existing_csv",
                "total_molecules": 1,
            },
            "critic": {"status": "WARNING", "issues": ["Dry-run only"]},
            "warnings": [],
            "errors": [],
            "dry_run": True,
            "analysis_source": "existing_csv",
        },
        reports_dir=tmp_path,
        filename="dry_run_test.html",
    )
    html = Path(meta["report_path"]).read_text(encoding="utf-8")
    assert "Dry-run: REINVENT was not executed" in html
    assert "existing output file" in html
    assert "sampled.csv" in html


def test_executor_marks_existing_csv_on_dry_run():
    from agents.executor import ExecutionAgent
    from agents.planner import PlannerAgent

    plan = PlannerAgent().create_plan(
        "dry",
        project_dir=str(REPO / "projects" / "demo_project"),
        approve_run=False,
        skip_reinvent=False,
    )
    results = ExecutionAgent().execute(
        plan,
        project_dir=REPO / "projects" / "demo_project",
        approve_run=False,
    )
    analysis = results["steps"]["analyze_molecules"]
    assert analysis.get("ok") is True
    assert analysis.get("analysis_source") == "existing_csv"
    assert analysis.get("from_fresh_reinvent") is False
    assert analysis.get("csv_path")
