# Independent MD module (GROMACS)

Phase 6 adds a **directory-level MD tool**. It materializes human-written
mdp templates, runs a short minimization and/or NVT chain, and writes
RMSD/RMSF tables under `<project>/output/md/`.

This is **not** a Planner step and **not** part of the REINVENT Executor.
Human approval (`--approve-md`) and environment checks (`gmx` / `gmx_mpi`)
are separate from `--approve-run`, `--approve-dock`, and
`tools.environment.check_environment`.

mdp handling matches the REINVENT TOML policy: **human-written templates
only**. The model may change a small allowlist of scalars (`nsteps`, `dt`,
`ref_t`). It never emits a full mdp, and it cannot change integrator,
cutoffs, or constraints.

## Architecture

```text
python -m tools.md --approve-md --yes
        │
        ▼
check_md_environment   (gmx / gmx_mpi only; never pip/conda install)
        │
        ▼
sandbox paths   structure + topology ∈ <project>/input/
        │
        ▼
materialize mdp from experiments/*.mdp
        │         allowlisted overrides: nsteps, dt, ref_t
        ▼
[Human approval]  --approve-md + confirm / --yes
        │
        ▼
predefined gmx grompp / mdrun / rms / rmsf   (shell=False, CPU)
        │
        ▼
output/md/rmsd.csv + rmsf.csv + md_result.json
        │
        ▼
Critic  (only if that md object is in the evidence JSON)
```

| Component | Role |
|-----------|------|
| `tools/md/` | Independent package: env, mdp, commands, prepare, CLI |
| `python -m tools.md` | Only entry that may launch `gmx` |
| `tools/environment.py` | Unchanged: REINVENT / RDKit / GPU only |
| Executor / planner allowlist / MCP | Unchanged: no `gmx`, no `run_md` |
| Critic | May mention MD **metrics** only when `evidence.md.table_present` |

No LLM-written TOML, no arbitrary shell, no MM-PBSA. Production 100 ns is
checked in as a human template, not launched. Literature search is
`python -m tools.literature` (Phase 7).

## Templates

| ID | File | Default role |
|----|------|----------------|
| `minimization` | `experiments/minimization.mdp` | Smoke-test EM (`nsteps=500`) |
| `nvt` | `experiments/nvt.mdp` | Smoke-test short NVT (`nsteps=500`) |
| `production` | `experiments/md.mdp` | 100 ns production after NPT (`nsteps=50000000`). **Not** the default. |

`--protocol` selects a chain:

- `em-nvt` (default): minimization then short NVT, then RMSD/RMSF
- `em` / `nvt`: one stage only
- `production`: materializes `experiments/md.mdp` for inspection; **does not** `mdrun`

Launch cap: `nsteps ≤ 10000`. A 100 ns override is rejected. Tests never
start production MD.

## How to run

`gmx` (or `gmx_mpi`) must already be on `PATH`. This module never
`pip`/`conda` installs it. Topology and coordinates must already exist
under `<project>/input/` (no `pdb2gmx`).

```bash
conda activate reinvent4
cd /path/to/reinvent-agent

# Inspect GROMACS only (also: python -m tools.md.environment)
python -m tools.md.environment

# Dry-run: sandbox + env + materialize mdp; does not launch gmx
python -m tools.md \
  --project projects/demo_project \
  --structure projects/demo_project/input/md/system.gro \
  --topology projects/demo_project/input/md/system.top

# Human-approved smoke-test (prints the predefined job, then Proceed? or --yes)
python -m tools.md \
  --project projects/demo_project \
  --structure input/md/system.gro \
  --topology input/md/system.top \
  --protocol em-nvt \
  --nsteps 50 \
  --approve-md --yes
```

Allowlisted overrides (optional scalars): `--nsteps`, `--dt`, `--ref-t`.
There is no `--mdp-text`, `--integrator`, or free-form mdp path.

Outputs (gitignored except `.gitkeep`):

- `<project>/output/md/rmsd.csv`, `rmsf.csv`
- `<project>/output/md/md_result.json`
- materialized `mdp/`, `logs/`, and GROMACS `-deffnm` files under the same directory

The demo `input/md/system.gro` + `system.top` is a **tiny water fixture**,
not a scientific protein–ligand system.

## Critic

The REINVENT CLI does **not** load MD tables. The MD CLI builds a results
payload with an `md` object and runs the deterministic critic.

- If that object is absent, LLM critic text mentioning GROMACS / molecular
  dynamics / RMSF is schema-rejected (same as before).
- If `evidence.md.table_present` is true, the critic may quote the numeric
  RMSD/RMSF values in that block.
- Docking / PubMed / IC50 / MM-PBSA / binding-affinity claims stay
  forbidden unless their own evidence tables (sourced literature entries
  for PubMed) are present.

## Safety

1. Never `subprocess.run(..., shell=True)`.
2. `gmx` argv is predefined (`grompp` / `mdrun` / `rms` / `rmsf` only).
   The model cannot supply command strings. `/bin/bash` is rejected.
3. Structure and topology stay under `<project>/input/`. Tables are written
   only under `output/md/`.
4. `--approve-md` is independent of `--approve-run` and `--approve-dock`.
   None of these flags launches the other tools.
5. mdp files are human templates. Allowlisted overrides are scalars
   (`nsteps`, `dt`, `ref_t`). Integrator / cutoff / constraint changes are
   rejected. A full mdp cannot be passed in.
6. No auto-install. Missing `gmx` is a loud failure, not invented RMSD.
7. CPU-only `mdrun` (`-nb cpu -nt 1`). No GPU flag. Launch cap keeps this
   phase off 100 ns production.

## Tests

```bash
python -m pytest tests/test_md.py tests/ -q
```

Tests mock `gmx`. They do not need GPU, network, or a GROMACS binary, and
they do not launch 100 ns. If `gmx` / `gmx_mpi` are on PATH, a small
`--version` probe runs; otherwise it skips.

## Out of scope

Production-length MD, NPT continuation as a runnable protocol, MM-PBSA,
`pdb2gmx` / solvation / ions, MCP MD tools, LLM-written
mdp/TOML, adding `gmx` to the REINVENT Executor loop.
Literature search is `python -m tools.literature`, not this module.
