"""Tests for the predefined REINVENT runner (mocked subprocess)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.reinvent import run_reinvent

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "projects" / "demo_project" / "reinvent.toml"


def test_run_reinvent_requires_approval():
    result = run_reinvent(CONFIG, approve=False)
    assert result["skipped"] is True
    assert result["success"] is False
    assert result["exit_code"] is None
    assert "--approve-run" in result["message"]


def test_run_reinvent_approve_uses_predefined_argv():
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    with patch("tools.reinvent.shutil.which", return_value="/fake/bin/reinvent"):
        with patch("tools.reinvent.subprocess.run", return_value=mock_proc) as mocked:
            result = run_reinvent(CONFIG, approve=True, seed=42)
    assert result["success"] is True
    assert result["exit_code"] == 0
    assert result["skipped"] is False
    cmd = mocked.call_args.args[0]
    assert cmd[0] == "/fake/bin/reinvent"
    assert cmd[1] == "-l"
    assert cmd[3] == "-s"
    assert cmd[4] == "42"
    assert cmd[5].endswith("reinvent.toml")
    assert mocked.call_args.kwargs.get("shell") is False
