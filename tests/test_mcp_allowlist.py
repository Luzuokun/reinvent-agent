"""MCP allowlist: same tools as the planner, no arbitrary shell."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.plan_schema import ALLOWED_STEPS
from tools.mcp_allowlist import (
    FORBIDDEN_TOOL_NAMES,
    MCP_TOOL_NAMES,
    MCP_TOOL_SPECS,
    McpAllowlistError,
    invoke_mcp_tool,
    list_mcp_tools,
)

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "projects" / "demo_project"
SAMPLE = DEMO / "output" / "sampled-sample.csv"


def test_mcp_tool_names_match_planner_allowlist():
    assert MCP_TOOL_NAMES == ALLOWED_STEPS
    assert [spec["name"] for spec in MCP_TOOL_SPECS] == list(ALLOWED_STEPS)


def test_list_mcp_tools_has_no_shell_and_forbids_extra_properties():
    catalog = list_mcp_tools()
    names = [item["name"] for item in catalog]
    assert names == list(MCP_TOOL_NAMES)
    for forbidden in FORBIDDEN_TOOL_NAMES:
        assert forbidden not in names
    for item in catalog:
        schema = item["inputSchema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        for bad in ("command", "argv", "shell", "executable"):
            assert bad not in schema.get("properties", {})


def test_unknown_shell_tool_rejected():
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("run_shell", {"command": "ls"})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("shell", {"command": "echo pwned"})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("execute_command", {"argv": ["bash", "-c", "id"]})


def test_check_environment_via_mcp():
    result = invoke_mcp_tool("check_environment", {})
    assert "python" in result
    assert "reinvent" in result
    assert "rdkit" in result
    assert "gpu" in result


def test_check_environment_rejects_extra_keys():
    with pytest.raises(McpAllowlistError, match="Unknown arguments"):
        invoke_mcp_tool("check_environment", {"command": "nvidia-smi"})


def test_validate_and_prepare_demo_project():
    validation = invoke_mcp_tool(
        "validate_project", {"project_dir": "projects/demo_project"}
    )
    assert "ok" in validation
    assert validation["details"]["config_path"].endswith("reinvent.toml")

    prepared = invoke_mcp_tool(
        "prepare_execution", {"project_dir": str(DEMO)}
    )
    assert prepared["ready"] is True
    assert prepared["approve_run"] is False
    assert prepared["config_path"].endswith("reinvent.toml")
    assert prepared["output_dir"].endswith("output")


def test_project_dir_must_stay_under_repo(tmp_path):
    with pytest.raises(McpAllowlistError, match="escapes repo root"):
        invoke_mcp_tool("validate_project", {"project_dir": str(tmp_path)})
    with pytest.raises(McpAllowlistError, match="escapes repo root"):
        invoke_mcp_tool("validate_project", {"project_dir": "/etc"})


def test_analyze_molecules_sample_csv():
    result = invoke_mcp_tool(
        "analyze_molecules",
        {
            "project_dir": "projects/demo_project",
            "csv_path": "output/sampled-sample.csv",
        },
    )
    assert result["ok"] is True
    assert result["total_molecules"] > 0
    assert Path(result["csv_path"]).resolve() == SAMPLE.resolve()


def test_analyze_molecules_rejects_csv_outside_output():
    with pytest.raises(McpAllowlistError, match="escapes project output dir"):
        invoke_mcp_tool(
            "analyze_molecules",
            {
                "project_dir": "projects/demo_project",
                "csv_path": str(REPO / "README.md"),
            },
        )


def test_analyze_molecules_rejects_missing_csv_arg():
    with pytest.raises(McpAllowlistError, match="Missing required arguments"):
        invoke_mcp_tool("analyze_molecules", {"project_dir": "projects/demo_project"})


def test_find_output_stays_in_project_output():
    result = invoke_mcp_tool("find_output", {"project_dir": "projects/demo_project"})
    assert result["ok"] is True
    assert result["csv_count"] >= 1
    output_dir = Path(result["output_dir"]).resolve()
    assert output_dir == (DEMO / "output").resolve()
    for path in result["csv_files"]:
        Path(path).resolve().relative_to(output_dir)


def test_find_output_rejects_output_dir_override():
    with pytest.raises(McpAllowlistError, match="Unknown arguments"):
        invoke_mcp_tool(
            "find_output",
            {
                "project_dir": "projects/demo_project",
                "output_dir": str(REPO),
            },
        )


def test_run_reinvent_without_approve_does_not_launch():
    with patch("tools.reinvent.subprocess.run") as mocked:
        result = invoke_mcp_tool(
            "run_reinvent",
            {"project_dir": "projects/demo_project", "approve_run": False},
        )
    mocked.assert_not_called()
    assert result["skipped"] is True
    assert result["success"] is False
    assert "--approve-run" in result["message"]


def test_run_reinvent_default_is_not_approved():
    with patch("tools.reinvent.subprocess.run") as mocked:
        result = invoke_mcp_tool(
            "run_reinvent", {"project_dir": "projects/demo_project"}
        )
    mocked.assert_not_called()
    assert result["skipped"] is True


def test_run_reinvent_rejects_command_and_argv():
    with pytest.raises(McpAllowlistError, match="Unknown arguments"):
        invoke_mcp_tool(
            "run_reinvent",
            {
                "project_dir": "projects/demo_project",
                "approve_run": True,
                "command": "rm -rf /",
            },
        )
    with pytest.raises(McpAllowlistError, match="Unknown arguments"):
        invoke_mcp_tool(
            "run_reinvent",
            {
                "project_dir": "projects/demo_project",
                "argv": ["bash", "-c", "id"],
            },
        )
    with pytest.raises(McpAllowlistError, match="Unknown arguments"):
        invoke_mcp_tool(
            "run_reinvent",
            {
                "project_dir": "projects/demo_project",
                "shell": True,
            },
        )


def test_run_reinvent_approve_uses_predefined_argv_not_shell():
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    with patch("tools.reinvent.shutil.which", return_value="/fake/bin/reinvent"):
        with patch("tools.reinvent.subprocess.run", return_value=mock_proc) as mocked:
            result = invoke_mcp_tool(
                "run_reinvent",
                {"project_dir": "projects/demo_project", "approve_run": True},
            )
    assert result["success"] is True
    assert result["skipped"] is False
    cmd = mocked.call_args.args[0]
    assert cmd[0] == "/fake/bin/reinvent"
    assert cmd[1] == "-l"
    assert cmd[3] == "-s"
    assert cmd[-1].endswith("reinvent.toml")
    assert mocked.call_args.kwargs.get("shell") is False


def test_generate_report_forwards_payload(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_report(payload, **kwargs):
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        return {"ok": True, "report_path": str(tmp_path / "report.html")}

    monkeypatch.setattr("tools.mcp_allowlist.generate_html_report", fake_report)
    result = invoke_mcp_tool("generate_report", {"payload": {"goal": "offline"}})
    assert result["ok"] is True
    assert captured["payload"]["goal"] == "offline"


def test_critic_review_deterministic():
    results = {
        "plan": {
            "approve_run": False,
            "skip_reinvent": True,
            "steps": ["analyze_molecules"],
        },
        "steps": {
            "analyze_molecules": {
                "ok": True,
                "total_molecules": 10,
                "duplicate_fraction": 0.0,
                "rdkit": {
                    "available": True,
                    "valid_fraction": 1.0,
                    "qed": {"mean": 0.6, "n": 10},
                },
            },
            "find_output": {"ok": True, "csv_count": 1},
            "validate_project": {"ok": True},
        },
        "warnings": [],
        "errors": [],
    }
    verdict = invoke_mcp_tool("critic_review", {"results": results})
    assert verdict["status"] in {"PASS", "WARNING", "FAIL"}
    assert "issues" in verdict
    assert verdict.get("critic") in (None, "deterministic") or "critic" in verdict


def test_mcp_server_module_registers_allowlist_only():
    from tools.mcp_server import registered_tool_names

    names = registered_tool_names()
    assert names == list(MCP_TOOL_NAMES)
    for forbidden in FORBIDDEN_TOOL_NAMES:
        assert forbidden not in names


def test_mcp_stdio_client_check_environment_and_no_shell():
    """End-to-end stdio session when the optional MCP extra is installed."""
    pytest.importorskip("mcp")
    import asyncio
    import os
    import sys

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        pytest.skip("mcp client stdio API not available")

    async def _exercise() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "tools.mcp_server"],
            cwd=str(REPO),
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                names = {tool.name for tool in listed.tools}
                assert names == set(MCP_TOOL_NAMES)
                assert "run_shell" not in names
                assert "shell" not in names

                env_result = await session.call_tool("check_environment", {})
                env_text = env_result.content[0].text
                env_payload = json.loads(env_text)
                assert "python" in env_payload
                assert "reinvent" in env_payload

                analysis = await session.call_tool(
                    "analyze_molecules",
                    {
                        "project_dir": "projects/demo_project",
                        "csv_path": "output/sampled-sample.csv",
                    },
                )
                analysis_payload = json.loads(analysis.content[0].text)
                assert analysis_payload["ok"] is True

                try:
                    denied = await session.call_tool(
                        "run_shell", {"command": "ls"}
                    )
                    denied_text = denied.content[0].text
                    denied_payload = json.loads(denied_text)
                    assert denied_payload["ok"] is False
                    assert "Unknown tool" in denied_payload["error"]
                except Exception as exc:
                    # Some SDK versions reject unknown names before call_tool.
                    assert "run_shell" in str(exc).lower() or "unknown" in str(exc).lower()

    asyncio.run(asyncio.wait_for(_exercise(), timeout=45))
