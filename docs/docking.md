# Independent docking module (Vina / GNINA)

Phase 5 adds a **directory-level docking tool**. It prepares a receptor and
ligands, runs AutoDock Vina or GNINA with a predefined argv, and writes a
score table under `<project>/output/docking/`.

This is **not** a Planner step and **not** part of the REINVENT Executor.
Human approval (`--approve-dock`) and environment checks (`vina` / `gnina` /
Open Babel / Meeko) are separate from `--approve-run` and
`tools.environment.check_environment`.

## Architecture

```text
python -m tools.docking --approve-dock --yes
        │
        ▼
check_docking_environment   (vina / gnina / obabel / meeko / RDKit)
        │
        ▼
sandbox paths   receptor ∈ <project>/input/
                ligands  ∈ <project>/input/ or <project>/output/
        │
        ▼
[Human approval]  --approve-dock + confirm / --yes
        │
        ▼
prepare receptor/ligand → PDBQT
        │
        ▼
predefined vina|gnina argv   (shell=False)
        │
        ▼
output/docking/scores.csv + docking_result.json
        │
        ▼
Critic  (only if that docking object is in the evidence JSON)
```

| Component | Role |
|-----------|------|
| `tools/docking/` | Independent package: env, prep, engines, CLI |
| `python -m tools.docking` | Only entry that may launch Vina/GNINA |
| `tools/environment.py` | Unchanged: REINVENT / RDKit / GPU only |
| Executor / planner allowlist / MCP | Unchanged: no `run_vina`, no `gmx` |
| Critic | May mention docking **scores** only when `evidence.docking.table_present` |

No LLM-written TOML, no arbitrary shell. Literature search is a
separate module (`python -m tools.literature`); see
[docs/literature.md](literature.md). MD is a
separate module (`python -m tools.md`, `--approve-md`); see
[docs/md.md](md.md).

## How to run

Vina or GNINA must already be on `PATH`. This module never `pip`/`conda`
installs them.

```bash
conda activate reinvent4
cd /path/to/reinvent-agent

# Inspect docking tools only (also: python -m tools.docking.environment)
python -m tools.docking.environment

# Dry-run: sandbox + env check; does not launch Vina
python -m tools.docking \
  --project projects/demo_project \
  --receptor projects/demo_project/input/docking/receptor.pdbqt \
  --ligands projects/demo_project/input/docking/ligand.pdbqt \
  --engine vina \
  --center 0 0 0 \
  --size 20 20 20

# Human-approved launch (prints the predefined job, then Proceed? or --yes)
python -m tools.docking \
  --project projects/demo_project \
  --receptor input/docking/receptor.pdbqt \
  --ligands input/docking/ligand.pdbqt \
  --engine vina \
  --center 0 0 0 \
  --size 20 20 20 \
  --approve-dock --yes
```

`--engine {vina,gnina}` selects a PATH executable of that name. It is not a
free-form path; `/bin/bash` is rejected.

SMILES / CSV ligands need RDKit plus Open Babel (`obabel`) or Meeko to write
PDBQT. Passing an existing `.pdbqt` ligand skips conversion. Receptor `.pdb`
also needs `obabel`; prefer `.pdbqt`.

Outputs (gitignored except `.gitkeep`):

- `<project>/output/docking/scores.csv` — one row per ligand
- `<project>/output/docking/docking_result.json`
- `prepared/`, `poses/`, `logs/` under the same directory

## Critic

The REINVENT CLI does **not** load docking scores. The docking CLI builds a
results payload with a `docking` object and runs the deterministic critic.

- If that object is absent, LLM critic text mentioning docking / vina / gnina
  is schema-rejected (same as before).
- If `evidence.docking.table_present` is true, the critic may quote the
  numeric scores and the engine named in that block.
- MD / PubMed / IC50 / binding-affinity claims stay forbidden unless their
  own evidence tables (sourced literature entries for PubMed) are present.

## Safety

1. Never `subprocess.run(..., shell=True)`.
2. Engine argv is predefined; the model cannot supply command strings.
3. Receptor stays under `<project>/input/`. Ligands stay under `input/` or
   `output/`. Scores are written only under `output/docking/`.
4. `--approve-dock` is independent of `--approve-run`. Neither flag launches
   the other tool.
5. Exhaustiveness, pose count, ligand count, and box size are capped.
6. No auto-install. Missing `vina`/`gnina` is a loud failure, not invented
   scores.

## Tests

```bash
python -m pytest tests/test_docking.py tests/ -q
```

Tests mock Vina. They do not need GPU, network, or a docking binary. If
`vina` / `gnina` are on PATH, a small probe test runs; otherwise it skips.

## Out of scope

Literature search, MCP docking tools, LLM-written box/TOML, rewriting
`reinvent.toml`, adding vina to the REINVENT Executor loop. GROMACS is
`python -m tools.md`. PubMed is `python -m tools.literature`.
