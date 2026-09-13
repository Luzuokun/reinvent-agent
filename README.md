# REINVENT4 Agent MVP

Minimal, safe research automation around an existing REINVENT4 molecular
generation workflow.

**v0.2 scope:** check environment → validate project → (optionally) run
REINVENT4 sampling → analyze molecules → HTML report → critic verdict →
structured `logs/runs/<timestamp>/result.json`.

No LLM tool-calling yet. Planner / Executor / Critic are deterministic Python
classes. Agents never execute arbitrary shell; they only call predefined tools.

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

Non-interactive approved run (scripts / CI):

```bash
python main.py \
  --project projects/demo_project \
  --goal "Run the REINVENT4 workflow and analyze generated molecules." \
  --approve-run --yes
```

Offline analysis only (uses the bundled sample CSV, no REINVENT):

```bash
python main.py \
  --project projects/demo_project \
  --goal "Analyze existing sample molecules." \
  --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv
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
  validation, reinvent, analysis, critic, report path)
- `logs/last_run_summary.json` — pointer + copy of the latest summary
- `reports/report_<timestamp>.html` — self-contained HTML report

## Acceptance checklist (5 scenarios)

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

5. **Full approved sampling**
   ```bash
   python main.py --project projects/demo_project \
     --goal "Run sampling and analyze" --approve-run --yes
   ```
   Expect: REINVENT `success=True`, molecule stats, Critic `PASS`/`WARNING`,
   new HTML under `reports/`, and `logs/runs/*/result.json`.

## Safety

1. Never execute LLM-generated shell strings.
2. Never use `subprocess.run(..., shell=True)` with untrusted text.
3. All executable commands are predefined in tools.
4. File operations stay under the project / repo root.
5. Never delete files; never auto-install packages.
6. REINVENT requires `--approve-run`, plus interactive confirm or `--yes`.
7. Every action is logged under `logs/` (including per-run `result.json`).

## Relation to other projects

| Project | Role |
|---------|------|
| [AI-Drug-Discovery-Lab](https://github.com/Luzuokun/ai-drug-discovery-lab) | Handbook / MkDocs tutorials (not modified by this repo) |
| kinase-denovo-design | Real research pipeline; optional local prior source |
| **reinvent-agent** (this repo) | Thin, safe automation MVP around REINVENT4 |

## Layout

```text
agents/          Planner, Executor, Critic (deterministic)
tools/           Validated environment / reinvent / files / analysis / artefacts
analysis/        Molecule stats + HTML report
projects/        Sandboxed REINVENT projects (demo_project)
logs/runs/       Per-run result.json artefacts
config/agent.yaml
main.py
```

## License

See repository license when published. Scientific software (REINVENT4, RDKit)
retains their upstream licenses.
