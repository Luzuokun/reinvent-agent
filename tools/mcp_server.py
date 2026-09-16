"""MCP stdio server — allowlisted wrappers, no arbitrary shell.

Run from the repo root (after ``pip install '.[mcp]'`` or ``pip install 'mcp>=1.9,<2'``):

    python -m tools.mcp_server

Cursor / Claude example (``~/.cursor/mcp.json``):

    {
      "mcpServers": {
        "reinvent-agent": {
          "command": "python",
          "args": ["-m", "tools.mcp_server"],
          "cwd": "/path/to/reinvent-agent"
        }
      }
    }

The server only registers names from ``tools.mcp_allowlist.MCP_TOOL_NAMES``.
There is no shell / bash / execute tool. Internal CLI still calls the same
Python functions; this is a new entry point, not a new capability.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from tools import dumps_pretty
from tools.mcp_allowlist import (
    MCP_SERVER_NAME,
    MCP_SERVER_VERSION,
    McpAllowlistError,
    invoke_mcp_tool,
    list_mcp_tools,
)

_INSTALL_HINT = (
    "MCP SDK is required for the stdio server.\n"
    "  pip install 'mcp>=1.9.0,<2'\n"
    "  # or from the repo: pip install '.[mcp]'\n"
    "Then: python -m tools.mcp_server"
)


def _import_mcp():
    try:
        from mcp.server.lowlevel import NotificationOptions, Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except ImportError:
        try:
            from mcp.server import Server
            from mcp.server.lowlevel import NotificationOptions
            from mcp.server.stdio import stdio_server
            from mcp.types import TextContent, Tool
        except ImportError as exc:
            raise SystemExit(_INSTALL_HINT) from exc
    return NotificationOptions, Server, stdio_server, TextContent, Tool


def build_server():
    """Construct the low-level MCP ``Server`` with allowlisted tools only."""
    _NotificationOptions, Server, _stdio_server, TextContent, Tool = _import_mcp()

    app = Server(MCP_SERVER_NAME)

    @app.list_tools()
    async def _list_tools() -> list[Any]:
        return [
            Tool(
                name=item["name"],
                description=item["description"],
                inputSchema=item["inputSchema"],
            )
            for item in list_mcp_tools()
        ]

    @app.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[Any]:
        try:
            result = invoke_mcp_tool(name, arguments or {})
            text = dumps_pretty(result)
        except McpAllowlistError as exc:
            # Still a tool result (not a crashed session): the client sees why.
            text = dumps_pretty({"ok": False, "error": str(exc), "tool": name})
        return [TextContent(type="text", text=text)]

    return app


def registered_tool_names() -> list[str]:
    """Tool names this process will advertise (always the allowlist)."""
    return [item["name"] for item in list_mcp_tools()]


def _initialization_options(app: Any, notification_options_cls: Any) -> Any:
    try:
        return app.create_initialization_options(
            notification_options=notification_options_cls()
        )
    except TypeError:
        return app.create_initialization_options()


async def _run_stdio() -> None:
    notification_options_cls, _server_cls, stdio_server, _text, _tool = _import_mcp()
    app = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            _initialization_options(app, notification_options_cls),
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
        force=True,
    )
    # Import early so a missing extra prints the hint instead of an asyncio trace.
    _import_mcp()
    logging.getLogger("mcp_server").info(
        "Starting %s MCP server v%s (tools: %s)",
        MCP_SERVER_NAME,
        MCP_SERVER_VERSION,
        ", ".join(registered_tool_names()),
    )
    asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()
