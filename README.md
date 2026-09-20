# REINVENT4 Agent MVP

Minimal, safe research automation around an existing REINVENT4 molecular
generation workflow.

**v0.11 scope:** same pipeline as v0.10, plus an **independent PubMed
literature tool** (`python -m tools.literature`) that returns only entries
with URL and/or PMID (or DOI). Missing network is a loud failure, not fake
citations. Critic still refuses unsourced “literature shows…”. This repo
adds stable tool ids and a small Handbook tutorial-section ↔ tool-id table
(`python -m tools.handbook_map`); it does **not** copy
[AI-Drug-Discovery-Lab](https://github.com/Luzuokun/ai-drug-discovery-lab).
See [docs/literature.md](docs/literature.md) and
[docs/handbook-tools.md](docs/handbook-tools.md).
**v0.10** added an **independent GROMACS MD
module** (`python -m tools.md`). Minimization and/or short NVT, RMSD/RMSF
tables under `output/md/`. Separate `--approve-md` and MD env checks — `gmx`
is **not** a REINVENT executor step. Human mdp templates only; allowlisted
scalars are `nsteps` / `dt` / `ref_t`. See [docs/md.md](docs/md.md).
**v0.9** added an **independent docking module** (`python -m tools.docking`).
Prepare receptor/ligand, run Vina or GNINA, write `output/docking/scores.csv`.
Separate `--approve-dock` and docking env checks — vina is **not** a REINVENT
executor step. See [docs/docking.md](docs/docking.md).
**v0.8** added **human-in-the-loop rerun**.
`--from-run <id>` copies experiment CLI config from
`logs/runs/<id>/result.json`. The executor stays single-shot; the agent does
not edit TOML or loop. Launch still requires `--approve-run`. See
[docs/from-run.md](docs/from-run.md).
v0.7 added **human-written experiment presets**.
The planner may name a preset ID (`sampling-cpu-100`, `sampling-cpu-1000`,
`sampling-cpu-scaffold`); it cannot generate TOML. Launch still requires
`--approve-run`. See [docs/experiment-presets.md](docs/experiment-presets.md).
v0.6 added an optional **stdio MCP server** wrapping the existing allowlisted
tools. MCP is unchanged here (CLI / planner first). Deterministic planning and
critic remain the default. See
[docs/mcp-tools.md](docs/mcp-tools.md),
[docs/csv-analysis-report.md](docs/csv-analysis-report.md),
[docs/phase2-llm-planner.md](docs/phase2-llm-planner.md) and
[docs/phase3-llm-critic.md](docs/phase3-llm-critic.md).
Project intent and stage notes: [AI_CONTEXT.md](AI_CONTEXT.md).

No LLM tool-calling. The planner may only return a JSON plan of allowlisted step
names plus an optional preset ID. The critic may only return `PASS`/`WARNING`/`FAIL`
JSON from structured evidence. Agents never execute arbitrary shell; they only
call predefined tools. The MCP server advertises those same tools only.

## Requirements

- Linux workstation with an existing REINVENT4 install
- Conda env `reinvent4` (verified with REINVENT 4.8.x)
- Python 3.11+ inside that env
- RDKit (already provided by a typical REINVENT4 install)
- A prior model file (not committed to git)

Agent-side packages (pandas, PyYAML) are listed in `requirements.txt`. Install
them into the same `reinvent4` env if missing:

```bash
conda activate reinvent4
pip install -r requirements.txt
# optional, only for --planner llm / --critic llm (same SDK for xAI / Gemini):
pip install 'openai>=1.40'
# optional, only for the MCP stdio server (Cursor / other MCP clients):
pip install 'mcp>=1.9,<2'
```

## Place the prior (required for real runs)

Priors are large and gitignored. From the repo root:

```bash
mkdir -p projects/demo_project/priors
ln -s /path/to/reinvent.prior projects/demo_project/priors/reinvent.prior
# or: cp /path/to/reinvent.prior projects/demo_project/priors/reinvent.prior
```

Example on this workstation (kinase project prior):

```bash
ln -sf \
  ~/Projects/kinase-denovo-design/generation/reinvent/priors/reinvent.prior \
  projects/demo_project/priors/reinvent.prior
```

## Demo project

`projects/demo_project/reinvent.toml` is a Tutorial-01-style CPU sampling job
(`num_smiles = 100`). It does not wrap the large EGFR GPU workflow.

`projects/demo_tl/reinvent.toml` is a **CPU transfer-learning** demo (1 epoch,
bundled `input/tl_train.smi`). The artefact is a model checkpoint, not a new
molecule CSV.

**Research vs Chemistry:** literature search is the independent tool
`python -m tools.literature` (PubMed; sourced URL/PMID/DOI only; not a
Planner step). Local SMILES cleaning is
`python -m tools.smiles_prep` (no network; not a Planner step). Running the
predefined TL TOML is the existing Chemistry / REINVENT executor.

Place the same prior used for sampling:

```bash
mkdir -p projects/demo_tl/priors
ln -sf "$(pwd)/projects/demo_project/priors/reinvent.prior" \
  projects/demo_tl/priors/reinvent.prior
```

## CLI

Dry-run (inspect, validate, plan — does **not** launch REINVENT).
Molecule analysis still runs against an **existing** project CSV (e.g. a
previous sample or prior run); the CLI, HTML report, and `result.json` all
label that the stats are from an existing/stale file, not a fresh generation.
Critic returns `WARNING`.

```bash
conda activate reinvent4
cd /path/to/reinvent-agent

python main.py \
  --project projects/demo_project \
  --goal "Run the REINVENT4 workflow and analyze generated molecules."
```

Approved run (prints the predefined command, then asks `Proceed? [y/N]`):

```bash
python main.py \
  --project projects/demo_project \
  --goal "Run the REINVENT4 workflow and analyze generated molecules." \
  --approve-run
```

Optional LLM planner and/or critic (each falls back to its deterministic
implementation with a loud warning if the API key is missing, the provider
errors, or the JSON fails validation). Keys are read from the process
environment. A gitignored repo-root `.env` is loaded at startup; existing
environment values are kept. Never put keys in yaml or git.

```bash
export OPENAI_API_KEY=sk-...
python main.py \
  --project projects/demo_project \
  --goal "Analyze existing sample molecules." \
  --planner llm \
  --critic llm \
  --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv
```

### LLM providers

`config/agent.yaml` `planner.provider` / `critic.provider`, or `--provider`.
The same OpenAI Python SDK is used for every vendor (`pip install 'openai>=1.40'`).

| provider | default `base_url` | API key env (aliases) | default model |
|----------|--------------------|------------------------|---------------|
| `openai` | SDK default (`api.openai.com`) | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `xai` | `https://api.x.ai/v1` | `XAI_API_KEY` (`GROK_API_KEY`) | `grok-4` |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY` (`GOOGLE_API_KEY`) | `gemini-2.5-flash` |
| `openai_compatible` | required in yaml | required `api_key_env` | required `model` |

Leave `model` / `base_url` / `api_key_env` as `null` in yaml to use the
provider preset. Override them when you need a specific model (e.g. `grok-4.6`
or `gemini-3.8-flash`).

xAI (no OpenAI quota):

```bash
# XAI_API_KEY=... in a gitignored .env, or:
export XAI_API_KEY=...          # or GROK_API_KEY
python main.py \
  --project projects/demo_project \
  --goal "Analyze existing sample molecules." \
  --planner llm --critic llm --provider xai \
  --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv
```

Gemini (OpenAI-compatible endpoint):

```bash
export GEMINI_API_KEY=...       # or GOOGLE_API_KEY
python main.py \
  --project projects/demo_project \
  --goal "Analyze existing sample molecules." \
  --planner llm --critic llm --provider gemini \
  --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv
```

Generic OpenAI-compatible server:

```yaml
planner:
  mode: llm
  provider: openai_compatible
  model: local-model
  base_url: http://127.0.0.1:8000/v1
  api_key_env: LOCAL_API_KEY
```

Non-interactive approved run (scripts / CI):

```bash
python main.py \
  --project projects/demo_project \
  --goal "Run the REINVENT4 workflow and analyze generated molecules." \
  --approve-run --yes
```

### Experiment presets

Human-written TOML lives in `experiments/`. The planner/LLM may only name an ID;
it cannot write config text. `--preset` is the human override. Launch still
needs `--approve-run`. Details: [docs/experiment-presets.md](docs/experiment-presets.md).

```bash
python main.py \
  --project projects/demo_project \
  --goal "Generate 1000 molecules." \
  --preset sampling-cpu-1000 \
  --approve-run --yes

python main.py \
  --project projects/demo_project \
  --goal "Generate from this scaffold and report QED." \
  --preset sampling-cpu-scaffold \
  --scaffold projects/demo_project/input/scaffold.smi \
  --approve-run --yes
```

Illegal preset IDs are rejected by argparse / schema. Out-of-sandbox scaffold
paths are schema-rejected; the LLM planner falls back to the deterministic plan.

### Human-in-the-loop rerun (`--from-run`)

The executor does not loop. After you read `reports/report_*.html`, re-invoke
the CLI. `--from-run <id>` copies project / goal / preset / scaffold / seed
from `logs/runs/<id>/result.json`. It never copies `--approve-run`, `--yes`,
or `--skip-reinvent`. Change parameters yourself on the new command.
Details: [docs/from-run.md](docs/from-run.md).

```bash
# After reading the report from run 20260917_120000:
python main.py --from-run 20260917_120000 --approve-run --yes

# Same identity, human switches 100 → 1000 molecules:
python main.py \
  --from-run 20260917_120000 \
  --preset sampling-cpu-1000 \
  --approve-run --yes

# Most recent result.json
python main.py --from-run last --approve-run
```

### Independent docking (`python -m tools.docking`)

Not a Planner/Executor/MCP step. Separate `--approve-dock` and a docking-only
environment check (Vina / GNINA / Open Babel / Meeko). Scores go to
`<project>/output/docking/scores.csv`. The REINVENT critic may quote those
scores only when that table is in the evidence JSON. Details:
[docs/docking.md](docs/docking.md).

```bash
python -m tools.docking \
  --project projects/demo_project \
  --receptor projects/demo_project/input/docking/receptor.pdbqt \
  --ligands projects/demo_project/input/docking/ligand.pdbqt \
  --engine vina \
  --center 0 0 0 \
  --size 20 20 20 \
  --approve-dock --yes
```

Without `--approve-dock` the module checks paths and docking tools and does
**not** launch Vina/GNINA. `--approve-run` never starts docking.

### Independent MD (`python -m tools.md`)

Not a Planner/Executor/MCP step. Separate `--approve-md` and an MD-only
environment check (`gmx` / `gmx_mpi`). Human-written mdp templates live in
`experiments/` (`minimization.mdp`, `nvt.mdp`, production `md.mdp`). The
default protocol is short **em-nvt**, not 100 ns production. Allowlisted
overrides: `--nsteps`, `--dt`, `--ref-t`. RMSD/RMSF tables go to
`<project>/output/md/`. The REINVENT critic may quote those metrics only
when that table is in the evidence JSON. Details: [docs/md.md](docs/md.md).

```bash
python -m tools.md \
  --project projects/demo_project \
  --structure projects/demo_project/input/md/system.gro \
  --topology projects/demo_project/input/md/system.top \
  --protocol em-nvt \
  --nsteps 50 \
  --approve-md --yes
```

Without `--approve-md` the module checks paths, env, and materializes mdp
files and does **not** launch `gmx`. `--approve-run` / `--approve-dock`
never start MD.

### Independent literature (`python -m tools.literature`)

Not a Planner/Executor/MCP step. Separate `--approve-literature`. Queries
NCBI PubMed E-utilities and keeps **only** entries with URL and/or PMID
(or DOI). Missing network is a loud `NETWORK FAILURE`, not fabricated
citations. The critic may mention PubMed / “literature shows” only when
those sourced entries are in the evidence JSON. This module does **not**
write papers. Details: [docs/literature.md](docs/literature.md).

```bash
python -m tools.literature \
  --project projects/demo_project \
  --query "EGFR tyrosine kinase inhibitor" \
  --retmax 10 \
  --approve-literature --yes
```

Without `--approve-literature` the module checks the query and does **not**
hit NCBI. `--approve-run` never starts a literature search.

### Handbook tool-id mapping (`python -m tools.handbook_map`)

Stable tool ids plus a small tutorial-section ↔ tool-id table. Tutorial
prose stays in the Handbook repo. Details:
[docs/handbook-tools.md](docs/handbook-tools.md).

```bash
python -m tools.handbook_map
```

### MCP server (optional)

Same eight allowlisted tools as the planner, over stdio. No shell tool.
Requires `pip install 'mcp>=1.9,<2'` (or `pip install '.[mcp]'`).

```bash
python -m tools.mcp_server
```

Cursor (`~/.cursor/mcp.json` or project MCP settings), with `cwd` set to this
repo:

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

Clients can call `check_environment` and `analyze_molecules` (CSV must already
live under `<project>/output/`). They cannot call arbitrary shell.
`run_reinvent` still requires `approve_run: true` and uses the predefined
argv only. Details: [docs/mcp-tools.md](docs/mcp-tools.md).

### Offline analysis

Uses the bundled sample CSV, no REINVENT:

```bash
python main.py \
  --project projects/demo_project \
  --goal "Analyze existing sample molecules." \
  --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv
```

CPU transfer learning (writes `projects/demo_tl/models/demo_tl.model`;
analysis is the **training** SMILES, not a new sample). Rebuild the bundled
`.smi` from a local CSV if needed:

```bash
python -m tools.smiles_prep --project projects/demo_tl \
  --source projects/demo_project/output/sampled-sample.csv
python main.py \
  --project projects/demo_tl \
  --goal "Fine-tune prior on bundled SMILES via transfer learning." \
  --approve-run --yes
```

### Exit codes (aligned with Critic)

| Critic status | Exit code | Meaning |
|---------------|-----------|---------|
| `PASS` | `0` | Evidence looks consistent |
| `WARNING` | `0` | Completed, but human review recommended |
| `FAIL` | `1` | Do not trust outputs; fix and re-run |

### Run artefacts

Every invocation writes:

- `logs/runs/<UTC_timestamp>/result.json` — full structured payload (plan, env,
  validation, reinvent, analysis, critic, report path, `run_id`, `invocation`)
- `logs/last_run_summary.json` — pointer (`run_id`) + copy of the latest summary
- `reports/report_<timestamp>.html` — self-contained HTML report (includes the
  `--from-run <id> --approve-run` command)

## Acceptance checklist (8 scenarios)

Run from the repo root with `conda activate reinvent4`.

1. **Unit tests**
   ```bash
   python -m pytest tests/ -q
   ```
   Expect: all tests pass.

2. **Dry-run (no `--approve-run`)**
   ```bash
   python main.py --project projects/demo_project --goal "Dry run"
   ```
   Expect: Critic `WARNING` (REINVENT not executed); CLI says analysis used an
   **existing** CSV (path printed); HTML report dry-run banner; `result.json`
   has `analysis_source: existing_csv` / `dry_run: true`; exit `0`.

3. **Missing prior + `--approve-run --yes`**
   ```bash
   mv projects/demo_project/priors/reinvent.prior /tmp/reinvent.prior.bak
   python main.py --project projects/demo_project --goal "Missing prior" --approve-run --yes
   # restore:
   mv /tmp/reinvent.prior.bak projects/demo_project/priors/reinvent.prior
   ```
   Expect: validation fails; Critic `FAIL`; exit `1`; REINVENT not launched.

4. **Offline analysis (`--skip-reinvent`)**
   ```bash
   python main.py --project projects/demo_project --goal "Offline" \
     --skip-reinvent --csv projects/demo_project/output/sampled-sample.csv
   ```
   Expect: Critic `PASS` (or `WARNING` only if RDKit missing); exit `0`.
   HTML report includes QED / MW / logP SVG histograms and a PAINS count
   when RDKit is available.

5. **Full approved sampling**
   ```bash
   python main.py --project projects/demo_project \
     --goal "Run sampling and analyze" --approve-run --yes
   ```
   Expect: REINVENT `success=True`, molecule stats, Critic `PASS`/`WARNING`,
   new HTML under `reports/`, and `logs/runs/*/result.json`.

6. **HITL rerun (`--from-run`)**
   ```bash
   python main.py --project projects/demo_project --goal "First look" \
     --preset sampling-cpu-100 --skip-reinvent \
     --csv projects/demo_project/output/sampled-sample.csv
   # note the printed Run id, then:
   python main.py --from-run <that-id> --preset sampling-cpu-1000
   ```
   Expect: second invocation copies project/goal; CLI override changes preset;
   REINVENT is **not** launched without `--approve-run`; HTML shows a
   `--from-run … --approve-run` command; exit `0` (dry-run `WARNING`).

7. **Docking dry-run (no `--approve-dock`)**
   ```bash
   python -m tools.docking \
     --project projects/demo_project \
     --receptor projects/demo_project/input/docking/receptor.pdbqt \
     --ligands projects/demo_project/input/docking/ligand.pdbqt \
     --engine vina --center 0 0 0 --size 20 20 20
   ```
   Expect: env/path check only; Vina **not** launched; no `scores.csv`;
   Critic `WARNING`; exit `0`. `python main.py` still does not call vina.

8. **MD dry-run (no `--approve-md`)**
   ```bash
   python -m tools.md \
     --project projects/demo_project \
     --structure projects/demo_project/input/md/system.gro \
     --topology projects/demo_project/input/md/system.top
   ```
   Expect: env/path/mdp check only; `gmx` **not** launched; no `rmsd.csv`;
   Critic `WARNING`; exit `0`. `python main.py` still does not call gmx.

9. **Literature dry-run (no `--approve-literature`)**
   ```bash
   python -m tools.literature \
     --project projects/demo_project \
     --query "EGFR tyrosine kinase inhibitor"
   ```
   Expect: query check only; **no** NCBI call; no `entries.csv`; Critic
   `WARNING`; exit `0`. No invented PMIDs. `python main.py` still does not
   search PubMed.

## Safety

1. Never execute LLM-generated shell strings.
2. Never use `subprocess.run(..., shell=True)` with untrusted text.
3. All executable commands are predefined in tools.
4. File operations stay under the project / repo root.
5. Never delete files; never auto-install packages.
6. REINVENT requires `--approve-run`, plus interactive confirm or `--yes`.
7. Every action is logged under `logs/` (including per-run `result.json`).
8. Optional LLM critic is evidence-only: no tools, no shell, no invented
   literature. Docking scores may be mentioned only when a docking table is
   in the evidence JSON. MD metrics only when an MD table is in the evidence.
   PubMed / “literature shows” only when sourced URL/PMID/DOI entries are
   in the evidence.
9. MCP exposes the same allowlisted tools only; no shell / argv / TOML-writing tool.
10. Experiment config is a human-written preset ID only. The model cannot emit TOML.
11. `--from-run` copies CLI identity only. It never copies `--approve-run` /
    `--yes`, never edits `reinvent.toml`, and never starts an agent retry loop.
12. Docking is `python -m tools.docking` with `--approve-dock`. It is not a
    REINVENT executor step; `--approve-run` never launches vina/gnina/gmx.
13. MD is `python -m tools.md` with `--approve-md`. mdp files are human
    templates; the model cannot emit a full mdp. Production 100 ns is not
    the default and is not launched by this module.
14. Literature is `python -m tools.literature` with `--approve-literature`.
    It returns only sourced entries and does not write papers. Missing
    network is a loud failure. Handbook mapping is
    `python -m tools.handbook_map` (ids only; no copied tutorials).

## Relation to other projects

| Project | Role |
|---------|------|
| [AI-Drug-Discovery-Lab](https://github.com/Luzuokun/ai-drug-discovery-lab) | Handbook / MkDocs tutorials (not modified by this repo) |
| kinase-denovo-design | Real research pipeline; optional local prior source |
| **reinvent-agent** (this repo) | Thin, safe automation MVP around REINVENT4 |

## Layout

```text
agents/          Planner + Critic (deterministic default), optional LLM layers, shared llm_client
tools/           Validated environment / reinvent / files / smiles_prep / docking / md / literature / handbook_map / analysis / artefacts / MCP allowlist
analysis/        Molecule stats + HTML report
experiments/     Human-written REINVENT presets and GROMACS mdp templates (never model-authored)
projects/        Sandboxed REINVENT projects (demo_project sampling, demo_tl TL)
docs/            Phase design notes (literature.md, handbook-tools.md, md.md, docking.md, …)
AI_CONTEXT.md    Living project context
logs/runs/       Per-run result.json artefacts
config/agent.yaml
main.py
```

## License

See repository license when published. Scientific software (REINVENT4, RDKit)
retains their upstream licenses.
