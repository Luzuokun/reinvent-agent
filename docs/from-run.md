# Human-in-the-loop rerun — `--from-run`

After a human reads the HTML report they re-invoke the CLI. `--from-run <id>`
copies experiment identity from `logs/runs/<id>/result.json`. The executor
stays **single-shot**. The agent does not edit TOML, does not tweak parameters,
and does not loop.

This is the HITL version of “find issue → change params → rerun”.

## Architecture

```text
run N:  Planner → Executor (once) → Critic → report.html + result.json
                │
                ▼  human reads the report
                │
run N+1: python main.py --from-run <id> [--preset …] --approve-run
                │
                ▼
        copy CLI config from result.json
        (project / goal / preset / scaffold / seed / planner / critic / provider)
                │
                ▼
        Planner → Executor (once) → Critic
        --approve-run is required again to launch REINVENT
```

| Component | Role |
|-----------|------|
| `tools/from_run.py` | Resolve run id, sandbox `logs/runs/`, copy allowlisted CLI fields |
| `tools/artifacts.py` | Persist `run_id` + `invocation` on every `result.json` |
| `main.py` | `--from-run`; explicit flags override copied values |
| HTML report | Shows the exact `--from-run <id> --approve-run` command |
| Executor | Unchanged: one pass over the plan; no retry loop |
| MCP | Unchanged. Rerun is CLI / human first. |

No docking, MD, or literature is copied into a rerun. Those stay
independent CLIs. No new agent framework, no arbitrary shell.

## What is copied

From `result.json` `invocation` (or reconstructed from older payloads):

- `--project`
- `--goal`
- `--preset`
- `--scaffold`
- `--seed`
- `--planner` / `--critic` / `--provider`
- `--config` (only if it was not the default `config/agent.yaml`)

A later `--csv` is copied **only** when this invocation also passed
`--skip-reinvent` (offline re-analysis of the same file).

Explicit CLI flags always win.

## What is never copied

- `--approve-run`
- `--yes`
- `--skip-reinvent`
- TOML text, argv, command, shell, plan steps, critic notes

Preset IDs must still be on the human-written allowlist
(`sampling-cpu-100` / `sampling-cpu-1000` / `sampling-cpu-scaffold`). Paths must
stay under the repo. `--from-run` only reads `logs/runs/<id>/result.json` (or
the `last` alias via `logs/last_run_summary.json`).

If you switch `--preset` away from `sampling-cpu-scaffold`, a copied
`--scaffold` is dropped unless you pass `--scaffold` again.

## How to run

```bash
conda activate reinvent4
cd /path/to/reinvent-agent

# 1. First run (dry-run or approved). Note the printed run id.
python main.py \
  --project projects/demo_project \
  --goal "Generate 100 molecules." \
  --preset sampling-cpu-100

# 2. Read reports/report_*.html

# 3. Same experiment, now launch REINVENT (human approval)
python main.py --from-run 20260917_120000 --approve-run --yes

# 4. Human changed the experiment (100 → 1000). Still one shot.
python main.py \
  --from-run 20260917_120000 \
  --preset sampling-cpu-1000 \
  --approve-run --yes

# Convenience: most recent result.json
python main.py --from-run last --approve-run
```

Without `--approve-run`, REINVENT is not launched. `--from-run` never supplies
that flag.

## Tests

```bash
python -m pytest tests/ -q
```

Unknown ids, path escapes, illegal preset IDs, approval-flag leakage, and
CLI overrides are covered. Tests do not need GPU or network.

## Out of scope (still)

Agent-owned parameter-tweak loops, model-written TOML, PubMed, docking, MD,
CrewAI / LangGraph / LlamaIndex, MCP tools for `--from-run`.
