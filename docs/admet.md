# Independent ADMET coarse-screen tool

Phase C MVP: a **directory-level ADMET tool**. It scores an existing SMILES
CSV with **human-written rule preset ids** (offline RDKit descriptors —
MW / logP / HBD / HBA / TPSA / rotatable bonds / QED) and writes
`admet_scores.csv` under `<project>/output/admet/`.

This is **not** a Planner step and **not** part of the REINVENT Executor.
Human approval (`--approve-admet`) is separate from `--approve-run`,
`--approve-dock`, `--approve-md`, `--approve-literature`,
`--approve-chembl`, and `--approve-portfolio`.

There is **no LLM scorer** and **no docking / MD / assay invention**.
Missing RDKit is a **loud failure**, not a fabricated descriptor table.

## Architecture

```text
python -m tools.admet --preset-id lipinski --approve-admet --yes
        │
        ▼
sandbox paths   CSV ∈ <project>/… ; output ∈ <project>/output/admet/
        │
        ▼
[Human approval]  --approve-admet + confirm / --yes
        │
        ▼
RDKit descriptors + human rule thresholds (preset only)
        │
        ▼
output/admet/admet_scores.csv
output/admet/admet_result.json
        │
        ▼
Critic  (WARNING if scores not signed; FAIL only on hard errors)
```

| Component | Role |
|-----------|------|
| `tools/admet/` | Independent package: rule presets, RDKit scorer, CLI |
| `python -m tools.admet` | Only entry that may sign an ADMET scores table |
| Executor / planner allowlist / MCP | Unchanged: no `admet` step |
| Critic | Soft WARNING without `--approve-admet`; may mention ADMET only when the scores table is in evidence |

## Rule presets

List them without scoring:

```bash
python -m tools.admet --list-presets
```

| preset_id | Intent |
|-----------|--------|
| `lipinski` | Rule-of-Five: MW≤500, logP≤5, HBD≤5, HBA≤10 |
| `veber` | TPSA≤140, rotatable bonds≤10 |
| `leadlike` | Stricter lead-like cutoffs (MW≤350, logP≤3.5, …) |
| `qed-floor` | Keep molecules with RDKit QED ≥ 0.5 |

New thresholds require a human PR that adds a row to
`tools/admet/presets.py`. There is no free-form rule JSON flag.

## How to run

```bash
conda activate reinvent4
cd /path/to/reinvent-agent

# Dry-run: validate preset + CSV; preview in admet_result.json only
python -m tools.admet \
  --project projects/demo_project \
  --molecules-csv projects/demo_project/input/admet/molecules.csv \
  --preset-id lipinski

# Human-approved scores table
python -m tools.admet \
  --project projects/demo_project \
  --molecules-csv projects/demo_project/input/admet/molecules.csv \
  --preset-id lipinski \
  --approve-admet --yes
```

`--molecules-csv` must stay under the project directory. Demo ships
`projects/demo_project/input/admet/molecules.csv` (fixture SMILES only).

Outputs:

- `output/admet/admet_scores.csv` — descriptor + pass/fail table (**only** with `--approve-admet`)
- `output/admet/admet_result.json` — machine-readable result (+ Critic input)

## Scores CSV schema

Fixed columns (order stable):

`SMILES`, `ligand_id`, `valid`, `mw`, `logp`, `hbd`, `hba`, `tpsa`,
`rotatable_bonds`, `qed`, `preset_pass`, `fail_reasons`, `preset_id`,
`source_row`

## Safety invariants

- `--approve-admet` ≠ `--approve-run` / dock / md / literature / chembl / portfolio
- Predefined `preset_id` allowlist only
- Offline RDKit / rules only; no invented docking / MD / IC50 / assay values
- Missing RDKit → loud failure (no fabricated table)
- Not in Planner / MCP allowlist; Executor warns and skips admet step names
- Soft WARNING when scores are not signed; no hard-block of other modules
- Not a wet-lab druglikeness or clinical ADME claim

## Tests

`tests/test_admet.py` — fixture SMILES CSV only; no GPU, no network, no LLM.
No docking / MD / assay numbers invented.
