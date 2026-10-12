"""Local carb UI: same Python functions as the CLI, no shell child process, HITL approve off."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from unittest.mock import patch

from tools.md.carb.constants import DEFAULT_PROTOCOL, PRODUCTION_PROTOCOL, PROTOCOL_ID
from tools.md.carb.ui import (
    INSTALL_HINT,
    approve_default_for_protocol,
    effective_approve,
    main as ui_main,
    public_payload,
    ui_doctor,
    ui_prepare,
    ui_report,
    ui_run,
)


def test_ui_source_has_no_shell_true():
    import tools.md.carb.ui as ui_mod

    tree = ast.parse(Path(ui_mod.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "subprocess" not in imported
    assert "os" not in imported
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert "system" not in called
    assert "Popen" not in called
    assert "check_call" not in called
    assert "check_output" not in called


def test_ui_handlers_call_same_functions_as_cli():
    import tools.md.carb.cli as cli
    import tools.md.carb.ui as ui_mod

    cli_src = inspect.getsource(cli)
    ui_src = inspect.getsource(ui_mod)
    for name in ("doctor(", "prepare_carb(", "run_carb(", "report_carb("):
        assert name in cli_src
        assert name in ui_src
    assert "from tools.md.carb.doctor import doctor" in ui_src
    assert "from tools.md.carb.prepare import" in ui_src
    assert "from tools.md.carb.run import" in ui_src
    assert "from tools.md.carb.report import report_carb" in ui_src


def test_approve_default_off_including_production():
    assert approve_default_for_protocol(DEFAULT_PROTOCOL) is False
    assert approve_default_for_protocol(PRODUCTION_PROTOCOL) is False
    assert effective_approve(DEFAULT_PROTOCOL, False, False) is False
    assert effective_approve(PRODUCTION_PROTOCOL, False, False) is False
    assert effective_approve(PRODUCTION_PROTOCOL, True, False) is False
    assert effective_approve(PRODUCTION_PROTOCOL, True, True) is True
    assert effective_approve(DEFAULT_PROTOCOL, True, False) is True


def test_ui_run_without_approve_does_not_launch():
    with patch("tools.md.carb.ui.run_carb") as mocked:
        mocked.return_value = {
            "ok": True,
            "skipped": True,
            "approved": False,
            "launched_200ns": False,
            "message": "Carb MD not launched (missing --approve-md).",
        }
        text = ui_run(
            "projects/amylose_fucoidan",
            DEFAULT_PROTOCOL,
            False,
            False,
            False,
            None,
            None,
            None,
            None,
        )
    mocked.assert_called_once()
    kwargs = mocked.call_args.kwargs
    assert kwargs["approve"] is False
    assert kwargs["assume_yes"] is True
    assert kwargs["protocol"] == DEFAULT_PROTOCOL
    assert kwargs["gpu"] is False
    payload = json.loads(text)
    assert payload["skipped"] is True
    assert payload["launched_200ns"] is False
    assert payload["protocol_id"] == PROTOCOL_ID


def test_ui_production_stays_off_without_both_checkboxes():
    with patch("tools.md.carb.ui.run_carb") as mocked:
        mocked.return_value = {
            "ok": True,
            "skipped": True,
            "approved": False,
            "launched_200ns": False,
            "message": "Carb MD not launched (missing --approve-md).",
        }
        text = ui_run(
            "projects/amylose_fucoidan",
            PRODUCTION_PROTOCOL,
            True,
            False,
            False,
            None,
            None,
            None,
            None,
        )
    kwargs = mocked.call_args.kwargs
    assert kwargs["approve"] is False
    assert kwargs["protocol"] == PRODUCTION_PROTOCOL
    payload = json.loads(text)
    assert "Confirm 200 ns" in payload["message"]
    assert payload.get("launched_200ns") is False


def test_ui_doctor_prepare_report_delegate():
    with patch("tools.md.carb.ui.doctor", return_value={"ok": True, "kind": "carb-md-doctor"}) as d:
        payload = json.loads(ui_doctor())
    d.assert_called_once_with(fetch_forcefield=True)
    assert payload["ok"] is True
    assert payload["protocol_id"] == PROTOCOL_ID

    with patch(
        "tools.md.carb.ui.prepare_carb",
        return_value={"ok": True, "dry_run": True, "kind": "carb-prepare"},
    ) as p:
        json.loads(ui_prepare("projects/amylose_fucoidan", "amylose26_fucoidan12", 1.2, 0.15, True))
    p.assert_called_once()
    assert p.call_args.kwargs["dry_run"] is True
    assert p.call_args.kwargs["fetch_forcefield"] is True

    with patch(
        "tools.md.carb.ui.report_carb",
        return_value={"ok": True, "kind": "carb-report", "invented_trajectory": False},
    ) as r:
        payload = json.loads(ui_report("projects/amylose_fucoidan"))
    r.assert_called_once()
    assert payload["invented_trajectory"] is False


def test_ui_main_without_gradio_exits_2(capsys):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "gradio" or str(name).startswith("gradio."):
            raise ImportError("No module named 'gradio'")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", fake_import):
        code = ui_main([])
    assert code == 2
    err = capsys.readouterr().err
    assert "Gradio" in err
    assert "python -m tools.md.carb" in err


def test_install_hint_mentions_cli():
    assert "python -m tools.md.carb" in INSTALL_HINT
    assert ".[ui]" in INSTALL_HINT or "requirements-ui.txt" in INSTALL_HINT


def test_public_payload_adds_protocol_id():
    out = public_payload({"ok": True})
    assert out["protocol_id"] == PROTOCOL_ID
    assert "command_templates" not in out
