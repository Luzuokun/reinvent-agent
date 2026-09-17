# MCP packaging of existing allowlist tools

This adds a **stdio MCP server** that exposes the same eight planner steps
already used by the CLI executor. It is a new entry point (Cursor, Claude,
or any MCP client), not a new agent framework and not a new capability.

Internal execution still calls the same Python functions
(`check_environment`, `validate_project`, `run_reinvent`, `analyze_molecules`,
…). There is no shell tool.

## Architecture

```text
MCP client (Cursor / Claude / inspector)
        │  stdio  (python -m tools.mcp_server)
        ▼
tools/mcp_server.py     protocol only (list_tools / call_tool)
        │
        ▼
tools/mcp_allowlist.py  names + args + path sandbox
        │
        ▼
existing tools / analysis / CriticAgent
```

The CLI (`main.py` → Planner → Executor → Critic) is unchanged. MCP does not
replace it and does not introduce CrewAI / LangGraph / Agents SDK. Experiment
presets (`--preset`) are CLI / planner only; this server does not grow a
preset tool.

| MCP tool | Existing function | Restrictions (same as CLI / planner) |
|----------|-------------------|--------------------------------------|
| `check_environment` | `tools.environment.check_environment` | No args; never installs |
| `validate_project` | `tools.files.validate_project` | `project_dir` under repo root; read-only |
| `prepare_execution` | executor preview of config/output paths | Does not launch REINVENT |
| `run_reinvent` | `tools.reinvent.run_reinvent` | Predefined argv only; `approve_run` default false; `shell=False` |
| `find_output` | `tools.reinvent.find_output_files` | `<project>/output/` only |
| `analyze_molecules` | `analysis.molecule_analysis.analyze_molecules` | `csv_path` must exist under `<project>/output/` |
| `generate_report` | `analysis.report.generate_html_report` | Structured payload; writes repo `reports/` |
| `critic_review` | `agents.critic.CriticAgent.review` | Deterministic, evidence-only |

Unknown names (`run_shell`, `execute_command`, …) are rejected. Extra keys
such as `command` / `argv` / `shell` on any tool are rejected
(`additionalProperties: false`).

## How to run

```bash
conda activate reinvent4
cd /path/to/reinvent-agent
pip install -r requirements.txt
pip install 'mcp>=1.9.0,<2'    # or: pip install '.[mcp]'

python -m tools.mcp_server
```

The process speaks MCP over stdin/stdout. Logs go to stderr.

Cursor example (`~/.cursor/mcp.json` or project MCP settings):

```json
{
  "mcpServers": {
    "reinvent-agent": {
      "command": "python",
      "args": ["-m", "tools.mcp_server"],
      "cwd": "/path/to/reinvent-agent"
    }
  }
}
```

Use the same interpreter that has this repo and (optionally) RDKit on
`PYTHONPATH`. The client can then call `check_environment` and
`analyze_molecules`; it cannot call arbitrary shell.

`run_reinvent` still requires `approve_run: true` (same as CLI
`--approve-run`). Without it the predefined command is prepared and skipped.
The client cannot pass an executable, argv, or `shell=True`.

## Tests

```bash
python -m pytest tests/ -q
```

Allowlist tests do not need the MCP SDK, GPU, network, or API keys. If `mcp`
is installed, `test_mcp_stdio_client_check_environment_and_no_shell` also
opens a real stdio session and checks `tools/list` + `check_environment` /
`analyze_molecules`, and that `run_shell` is denied.

## Out of scope

Docking, MD, literature search, experiment preset IDs, LLM-written
`reinvent.toml`, and any Agent framework. Those remain later phases.
The executor is not rewritten to speak MCP internally.
