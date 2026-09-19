# Phase 3 — Optional LLM Critic

Phase 3 adds an **optional** LLM critic on top of the existing
Planning → Execution → Tools → Critic pipeline. The deterministic critic
remains the default. Execution, tools, and the Phase 1/2 planner paths
are unchanged.

## Architecture

```text
CLI / config
    │
    ▼
Planner  (deterministic default, or optional LLM — Phase 2)
    │
    ▼
Executor  (predefined tools only; never shell=True)
    │
    ▼
Critic  ── deterministic (default) ──► {status, issues, recommendation}
    │
    └── llm (optional) ── JSON schema call ──► validate_critic_verdict
                              │
                              └── invalid / no key / provider error
                                    └── WARNING + deterministic fallback
```

The LLM is a **critic only**. It does not call tools, does not write files,
does not talk to other agents, and does not invent docking / MD / literature
results unless the matching evidence table is already in the JSON. Its
output is JSON that must pass `agents/critic_schema.py` before
the HTML report or process exit code sees it. MD metrics are allowed only
when `evidence.md.table_present` is true (independent MD module).

| Component | Role in Phase 3 |
|-----------|-----------------|
| `agents/critic.py` | Unchanged algorithm; still the default |
| `agents/critic_schema.py` | Evidence summary + strict verdict validator |
| `agents/llm_critic.py` | LLM critic; client via `agents/llm_client.py` |
| `agents/llm_client.py` | Shared OpenAI-compatible client (openai / xai / gemini / openai_compatible) |
| `agents/executor.py` | Unchanged |
| `agents/planner.py` / `llm_planner.py` | Unchanged |

## Verdict schema

The model may emit **only** these keys:

```json
{
  "status": "PASS",
  "issues": ["optional evidence-grounded comments"],
  "recommendation": "one short human-facing sentence"
}
```

`status` must be exactly `PASS`, `WARNING`, or `FAIL` — the same contract
the HTML report and CLI exit codes already use.

The runtime then attaches fields the model **cannot** set:

- `thresholds` (from `config/agent.yaml` / `CriticAgent`)
- `critic`, `critic_fallback`, `critic_warnings`

Unknown keys, empty recommendations, shell-like text, and claims about
docking / GROMACS / PubMed / binding affinities that are **not** in the
evidence JSON are rejected. Docking score comments are allowed only when
`evidence.docking.table_present` is true (independent docking module).
MD / RMSF comments are allowed only when `evidence.md.table_present` is
true (independent MD module).

### What the model is allowed to see

`build_evidence_summary()` sends a compact object:

- plan flags (`approve_run`, `skip_reinvent`, step names)
- abort / error / warning strings (truncated)
- environment booleans (reinvent / rdkit / gpu)
- validation ok / errors
- REINVENT skipped / success / exit_code / message / argv
- inventory csv_count
- analysis counts, duplicate fraction, RDKit summary stats (incl. SA / PAINS / filters when present), `analysis_source`
- optional `docking` block (engine, score table summary) **only** when the docking module attached one
- optional `md` block (protocol, RMSD/RMSF table summary) **only** when the MD module attached one
- critic numeric thresholds

It does **not** send SMILES lists, CSV contents, HTML, or stdout dumps.

## Safety

These invariants are unchanged and are enforced again on the LLM path:

1. **Never** `subprocess.run(..., shell=True)`.
2. **Never** interpolate model text into argv, a shell, or a file write.
   Issues / recommendation are report strings only.
3. The critic does **not** call tools. No REINVENT, no analysis re-run.
4. File operations stay under the project / repo root (unchanged executor).
5. No auto-install, no TOML rewriting, no PubMed. Docking / MD comments
   require their own evidence tables.

On invalid LLM verdict, missing API key, missing `openai` package, or
provider error the agent **falls back to the deterministic critic** and
prints a loud `WARNING` (default). Set `critic.fallback_on_error: false`
to exit instead (process exit code `2`).

## Configuration

`config/agent.yaml`:

```yaml
critic:
  mode: deterministic          # or llm
  provider: openai             # openai | xai | gemini | openai_compatible
  api: chat_completions        # or responses
  model: null                  # null → provider default (or planner.model)
  temperature: 0.0
  api_key_env: null            # null → provider default env var
  base_url: null               # null → provider default endpoint
  fallback_on_error: true
    min_valid_fraction: 0.80
    max_duplicate_fraction: 0.25
    min_molecules: 1
    min_mean_qed: null           # optional; skip if QED mean is missing
```

`--critic {deterministic,llm}` overrides `critic.mode`. Both default to
`deterministic`. `--provider` overrides planner and critic provider together.

Keys are environment variables only (never files in the repo). `.env` is
gitignored.

## How to run

Install the optional SDK only if you want the LLM critic or planner
(CI does not need it):

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

LLM critic (falls back if the key is missing or the verdict is invalid):

```bash
python main.py \
  --project projects/demo_project \
  --goal "Analyze existing sample molecules." \
  --critic llm \
  --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv
```

Planner + critic together:

```bash
python main.py \
  --project projects/demo_project \
  --goal "Analyze existing sample molecules." \
  --planner llm \
  --critic llm \
  --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv
```

OpenAI-compatible local server (same pattern as the planner):

```yaml
critic:
  mode: llm
  provider: openai_compatible
  api: chat_completions
  model: local-model
  base_url: http://127.0.0.1:8000/v1
  api_key_env: LOCAL_API_KEY
```

xAI:

```bash
export XAI_API_KEY=...
python main.py \
  --project projects/demo_project \
  --goal "Analyze existing sample molecules." \
  --planner llm --critic llm --provider xai \
  --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv
```

## Tests

```bash
python -m pytest tests/ -q
```

LLM tests inject a fake client / `complete_fn`. They do not open a network
socket. The deterministic critic still returns `PASS` on the same offline
fixture as Phase 1.

## Out of scope (still)

PubMed, MD, auto-install, rewriting TOML, CrewAI / LangGraph /
LlamaIndex / multi-agent chat frameworks, MCP servers. Docking is an
independent module (`python -m tools.docking`); the LLM critic may quote
those scores only when the docking table is in evidence.
