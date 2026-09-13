# REINVENT4 Agent MVP

Minimal, safe research automation around an existing REINVENT4 molecular
generation workflow.

**v0.1 scope:** check environment → validate project → (optionally) run
REINVENT4 sampling → analyze molecules → HTML report → critic verdict.

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

Dry-run (inspect, validate, plan — does **not** launch REINVENT):

```bash
conda activate reinvent4
cd /path/to/reinvent-agent

python main.py \
  --project projects/demo_project \
  --goal "Run the REINVENT4 workflow and analyze generated molecules."
```

Approved run (executes the predefined `reinvent` command):

```bash
python main.py \
  --project projects/demo_project \
  --goal "Run the REINVENT4 workflow and analyze generated molecules." \
  --approve-run
```

Offline analysis only (uses the bundled sample CSV, no REINVENT):

```bash
python main.py \
  --project projects/demo_project \
  --goal "Analyze existing sample molecules." \
  --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv
```

## Safety

1. Never execute LLM-generated shell strings.
2. Never use `subprocess.run(..., shell=True)` with untrusted text.
3. All executable commands are predefined in tools.
4. File operations stay under the project / repo root.
5. Never delete files; never auto-install packages.
6. REINVENT requires explicit `--approve-run`.
7. Every action is logged under `logs/`.

## Relation to other projects

| Project | Role |
|---------|------|
| [AI-Drug-Discovery-Lab](https://github.com/Luzuokun/ai-drug-discovery-lab) | Handbook / MkDocs tutorials (not modified by this repo) |
| kinase-denovo-design | Real research pipeline; optional local prior source |
| **reinvent-agent** (this repo) | Thin, safe automation MVP around REINVENT4 |

## Layout

```text
agents/          Planner, Executor, Critic (deterministic)
tools/           Validated environment / reinvent / files / analysis tools
analysis/        Molecule stats + HTML report
projects/        Sandboxed REINVENT projects (demo_project)
config/agent.yaml
main.py
```

## License

See repository license when published. Scientific software (REINVENT4, RDKit)
retains their upstream licenses.
