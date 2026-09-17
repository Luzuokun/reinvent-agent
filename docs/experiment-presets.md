# Experiment presets — understand the goal without writing TOML

This adds **human-written experiment presets**. The planner (deterministic or
optional LLM) may only name a preset ID. The model never generates TOML,
argv, scoring sections, or config file contents. Launch still requires
`--approve-run`.

This is the safe translation of “understand the goal / generate a REINVENT
config”: pick a template a human already wrote.

## Architecture

```text
CLI --preset / --scaffold   (caller-owned)
        │
        ▼
Planner  ── deterministic (default) ──► Plan dict + optional preset_id
        │
        └── llm (optional) ── JSON schema ──► validate_plan
                              │                 (steps + preset_id only)
                              └── illegal ID / escaping path / TOML key
                                    └── WARNING + deterministic fallback
        │
        ▼
materialize_preset  ── copy experiments/<id>.toml → <project>/.agent/
        │              (scaffold preset: fill {{scaffold_path}} only)
        ▼
Executor  run_reinvent(<project>/.agent/<id>.toml)   still needs --approve-run
```

| Component | Role |
|-----------|------|
| `experiments/*.toml` | Human-authored REINVENT configs (git-tracked) |
| `agents/experiment_presets.py` | Allowlist, path sandbox, materialize |
| `agents/plan_schema.py` | `preset_id` / `scaffold_path`; rejects `toml` / `config_text` / argv |
| `agents/planner.py` | Optional `--preset` on the deterministic plan |
| `agents/llm_planner.py` | May name an ID; never emits TOML |
| `agents/executor.py` | Uses the materialized preset TOML when `preset_id` is set |
| MCP | **Unchanged.** Presets are CLI / planner first. |

No docking, MD, literature, new agent framework, or arbitrary shell.

## Allowed preset IDs

| ID | What it is | Fillable fields |
|----|------------|-----------------|
| `sampling-cpu-100` | CPU sampling, `num_smiles = 100` (demo-sized) | none |
| `sampling-cpu-1000` | CPU sampling, `num_smiles = 1000` | none |
| `sampling-cpu-scaffold` | Same CPU sampling, plus `smiles_file` | `scaffold_path` only |

`scaffold_path` must **already exist** under `<project>/input/`. Symlink
escapes and `..` are rejected. The runtime copies the human template into
`<project>/.agent/` and replaces the `{{scaffold_path}}` placeholder with a
regex-checked relative path (`input/scaffold.smi`). That is the only
substitution. QED / predicted affinity are **not** added as REINVENT scoring
sections; QED is still measured in post-run analysis.

Omitting `preset_id` keeps today’s behavior: `projects/<name>/reinvent.toml`.

## Plan schema additions

The model may emit **only** these keys (steps are unchanged from Phase 2):

```json
{
  "steps": ["check_environment", "validate_project", "..."],
  "notes": ["optional human comments — never TOML"],
  "csv_path": null,
  "preset_id": "sampling-cpu-1000",
  "scaffold_path": null
}
```

`preset_id` must be one of the IDs above or `null`. Unknown IDs, TOML text
passed as an ID, extra keys (`toml`, `config_text`, `argv`, `command`,
`shell`), a scaffold path outside `<project>/input/`, or `scaffold_path` on a
preset that does not allow it are **schema-rejected**. The LLM planner then
falls back to the deterministic plan (no model-chosen preset) and prints a
loud `WARNING`.

`approve_run` remains CLI-owned. `--preset` / `--scaffold` also override
whatever the model returned.

## Safety

1. **Never** interpolate model text into TOML, argv, or a shell.
2. Preset files in `experiments/` are written by humans and reviewed in git.
3. `run_reinvent` still requires `--approve-run` (plus confirm or `--yes`).
   The command shown to the human is the **materialized preset** path, not a
   model-authored config.
4. `scaffold_path` uses the same resolve + `relative_to(input_dir)` rule as
   plan `csv_path` under `output/`.
5. No MCP expansion for presets (prefer CLI / planner).
6. No docking, MD, literature, or new agent framework.

## How to run

```bash
conda activate reinvent4
cd /path/to/reinvent-agent

# Explicit human-written 1000-molecule CPU sampling (still needs approval)
python main.py \
  --project projects/demo_project \
  --goal "Generate 1000 molecules." \
  --preset sampling-cpu-1000 \
  --approve-run --yes

# Scaffold-style goal: select the template and point at an existing file
python main.py \
  --project projects/demo_project \
  --goal "Generate molecules from this scaffold and report QED." \
  --preset sampling-cpu-scaffold \
  --scaffold projects/demo_project/input/scaffold.smi \
  --approve-run --yes

# Optional LLM planner: it may only return a preset_id (or fall back)
python main.py \
  --project projects/demo_project \
  --goal "Generate 1000 molecules on CPU." \
  --planner llm \
  --approve-run --yes
```

Without `--approve-run` the preset is still recorded and the predefined
command is prepared, but REINVENT does not launch.

## Tests

```bash
python -m pytest tests/ -q
```

Illegal preset IDs, TOML keys, and out-of-sandbox scaffold paths are covered
and must fall back (LLM) or raise (CLI / validator). Tests do not need GPU
or network.

## Out of scope (still)

PubMed, docking, MD, auto-install, rewriting `reinvent.toml` from model
text, CrewAI / LangGraph / LlamaIndex, MCP tools for preset IDs.
