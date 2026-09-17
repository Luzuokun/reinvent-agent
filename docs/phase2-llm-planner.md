# Phase 2 — Optional LLM Planner

Phase 2 adds an **optional** LLM planning step on top of the existing
Planning → Execution → Tools → Critic pipeline. The deterministic planner
remains the default. Nothing in Phase 1 is rewritten.

## Architecture

```text
CLI / config
    │
    ▼
Planner  ── deterministic (default) ──► Plan dict
    │
    └── llm (optional) ── JSON schema call ──► validate_plan ──► Plan dict
                              │
                              └── invalid / no key / provider error
                                    └── WARNING + deterministic fallback
    │
    ▼
Executor  (predefined tools only; never shell=True)
    │
    ▼
Critic    (structured evidence only; deterministic default — Phase 3 adds optional LLM)
```

The LLM is a **planner only**. It does not call tools, does not write TOML,
and does not talk to other agents. Its output is JSON that must pass
`agents/plan_schema.py` before the executor sees it. Experiment configuration
is a **preset ID** from a human-written allowlist — see
[experiment-presets.md](experiment-presets.md).

| Component | Role in Phase 2 |
|-----------|-----------------|
| `agents/planner.py` | Unchanged algorithm; still the default |
| `agents/plan_schema.py` | Allowlist + strict validator |
| `agents/llm_planner.py` | LLM planner; client via `agents/llm_client.py` |
| `agents/llm_client.py` | Shared OpenAI-compatible client (openai / xai / gemini / openai_compatible) |
| `agents/executor.py` | Same tools; optional validated `csv_path`; ignores `run_reinvent` when `--skip-reinvent` |
| `agents/critic.py` | Unchanged |

## Plan schema

The model may emit **only** these keys:

```json
{
  "steps": ["check_environment", "validate_project", "..."],
  "notes": ["optional human comments"],
  "csv_path": null,
  "preset_id": null,
  "scaffold_path": null
}
```

`steps` may also be written as objects for validator tests / future use:

```json
{"name": "analyze_molecules", "params": {"csv_path": "output/sampled-sample.csv"}}
```

Allowed step names (hard ceiling — config can only narrow this set):

1. `check_environment`
2. `validate_project`
3. `prepare_execution`
4. `run_reinvent`
5. `find_output`
6. `analyze_molecules`
7. `generate_report`
8. `critic_review`

The runtime then merges caller-owned fields that the model **cannot** set:

- `goal`, `project_dir`
- `approve_run`, `skip_reinvent` (CLI flags)
- `planner`, `planner_fallback`, `planner_warnings`

Unknown keys, unknown step names, duplicate steps, extra step params, and
`csv_path` values that escape the project output directory are rejected.

### Parameters the model may control

Extremely narrow on purpose:

| Step | Model-controlled params |
|------|-------------------------|
| `run_reinvent` | **none**. Config path, cwd, seed, and argv come from a human-written preset ID or the project TOML + `agent.yaml` + CLI. Launch still requires `--approve-run`. The model cannot emit TOML. |
| `analyze_molecules` | optional `csv_path` that **already exists** and resolves **under** `<project>/output/` |
| all other steps | none |
| plan-level | optional `preset_id` (`sampling-cpu-100` / `sampling-cpu-1000` / `sampling-cpu-scaffold`); optional `scaffold_path` only for the scaffold preset, must exist under `<project>/input/` |

If `csv_path` is omitted, the executor keeps the Phase 1 discovery rules
(inventory / TOML `output_file` / CLI `--csv`).

## Safety

These invariants are unchanged and are enforced again on the LLM path:

1. **Never** `subprocess.run(..., shell=True)`.
2. **Never** interpolate model text into argv or a shell. Notes/goal/csv_path
   are not commands. `csv_path` is only used as a filesystem path after
   `Path.resolve()` + `relative_to(output_dir)`.
3. `run_reinvent` still requires `--approve-run` (plus interactive confirm or `--yes`).
4. File operations stay under the project / repo root. Plan `csv_path` is
   stricter: it must live under the project **output** directory.
5. `--skip-reinvent` is owned by the CLI. If an LLM plan still lists
   `run_reinvent`, the validator strips it and the executor will not launch.
6. No auto-install, no TOML rewriting, no PubMed / docking / MD.
   Optional LLM critic is Phase 3 — see [phase3-llm-critic.md](phase3-llm-critic.md).

On invalid LLM plan, missing API key, missing `openai` package, or provider
error the agent **falls back to the deterministic plan** and prints a loud
`WARNING` (default). Set `planner.fallback_on_error: false` to exit instead
(process exit code `2`).

## Configuration

`config/agent.yaml`:

```yaml
planner:
  mode: deterministic          # or llm
  provider: openai             # openai | xai | gemini | openai_compatible
  api: chat_completions        # or responses
  model: null                  # null → provider default
  temperature: 0.0
  max_steps: 8
  api_key_env: null            # null → OPENAI_API_KEY / XAI_API_KEY / GEMINI_API_KEY
  base_url: null               # null → provider default endpoint
  fallback_on_error: true
  allowlist:                   # must be a subset of the hardcoded list
    - check_environment
    - validate_project
    - prepare_execution
    - run_reinvent
    - find_output
    - analyze_molecules
    - generate_report
    - critic_review
```

`--planner {deterministic,llm}` overrides `planner.mode`. Both default to
`deterministic`. `--provider {openai,xai,gemini,openai_compatible}` overrides
`planner.provider` and `critic.provider` for whichever LLM path is active.

Keys are environment variables only (never files in the repo). `.env` is
gitignored. See README for xAI / Gemini export examples.

## How to run

Install the optional SDK only if you want the LLM planner (CI does not need it):

```bash
conda activate reinvent4
pip install -r requirements.txt
pip install 'openai>=1.40'    # or: pip install '.[llm]'
export OPENAI_API_KEY=sk-...  # or XAI_API_KEY / GEMINI_API_KEY
```

Deterministic (Phase 1, unchanged):

```bash
python main.py \
  --project projects/demo_project \
  --goal "Run the REINVENT4 workflow and analyze generated molecules."
```

LLM planner (falls back if the key is missing or the plan is invalid):

```bash
python main.py \
  --project projects/demo_project \
  --goal "Analyze existing sample molecules." \
  --planner llm \
  --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv
```

OpenAI-compatible local server:

```yaml
planner:
  mode: llm
  provider: openai_compatible
  api: chat_completions
  model: local-model
  base_url: http://127.0.0.1:8000/v1
  api_key_env: LOCAL_API_KEY
```

xAI:

```yaml
planner:
  mode: llm
  provider: xai
  # model/base_url/api_key_env omitted → grok-4, https://api.x.ai/v1, XAI_API_KEY
```

```bash
export XAI_API_KEY=...
python main.py --project projects/demo_project --goal "Offline" \
  --planner llm --provider xai --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv
```

Responses API (official OpenAI models that support it):

```yaml
planner:
  mode: llm
  api: responses
  model: gpt-4o-mini
```

## Tests

```bash
python -m pytest tests/ -q
```

LLM tests inject a fake client / `complete_fn`. They do not open a network
socket. Deterministic planner steps stay the Phase 1 list.

## Out of scope (still)

PubMed, docking, MD, auto-install, rewriting TOML, CrewAI /
LangGraph / LlamaIndex / multi-agent chat frameworks.

Phase 3 (optional LLM critic) is documented in
[phase3-llm-critic.md](phase3-llm-critic.md).
