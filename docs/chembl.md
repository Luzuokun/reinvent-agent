# Independent ChEMBL assay download tool

Phase D MVP scaffold: a **directory-level ChEMBL tool**. It downloads
bioactivity rows for a **predefined query id** from the ChEMBL REST API and
writes a fixed assay CSV under `<project>/output/chembl/`.

This is **not** a Planner step and **not** part of the REINVENT Executor.
Human approval (`--approve-chembl`) is separate from `--approve-run`,
`--approve-dock`, `--approve-md`, and `--approve-literature`.

Missing network is a **loud failure**. The tool does not invent SMILES,
IC50 values, or activity rows to fill the gap. Agents must not invent SQL,
curl, or free-form ChEMBL filters — only human-written query ids.

## Architecture

```text
python -m tools.chembl --query-id egfr-ic50 --approve-chembl --yes
        │
        ▼
sandbox paths   output ∈ <project>/output/chembl/
        │
        ▼
[Human approval]  --approve-chembl + confirm / --yes
        │
        ▼
HTTPS GET www.ebi.ac.uk/chembl/api/data/activity  (predefined filters only)
        │
        ▼
keep rows with molecule_chembl_id; never invent missing SMILES
        │
        ▼
output/chembl/activities.csv + smiles.csv + chembl_result.json
        │         optional: --prep-smiles → local smiles_prep
        ▼
Critic  (only if that chembl object is in the evidence JSON
         *and* an activity table is present)
```

| Component | Role |
|-----------|------|
| `tools/chembl/` | Independent package: query allowlist, REST client, assay CSV, CLI |
| `python -m tools.chembl` | Only entry that may hit ChEMBL |
| `tools.smiles_prep` | Optional local canonicalize of `smiles.csv` (no network) |
| Executor / planner allowlist / MCP | Unchanged: no `chembl`, no free-form download step |
| Critic | May mention ChEMBL / IC50 only when an activity table is in evidence |

No LLM-written TOML/mdp, no arbitrary shell, no SaaS/Web UI, no invented
assay values. Handbook tutorials stay in
[AI-Drug-Discovery-Lab](https://github.com/Luzuokun/ai-drug-discovery-lab);
this repo only records [stable tool ids](handbook-tools.md).

## Predefined queries

List them without network:

```bash
python -m tools.chembl --list-queries
```

| query_id | Target | Filter |
|----------|--------|--------|
| `egfr-ic50` | CHEMBL203 (EGFR) | IC50, assay_type=B |
| `braf-ic50` | CHEMBL5145 (BRAF) | IC50, assay_type=B |

New targets require a human PR that adds a row to
`tools/chembl/queries.py`. There is no free-text SQL / SPARQL / curl flag.

## How to run

No extra packages. The client uses the Python standard library
(`urllib.request`) against `www.ebi.ac.uk` only. This module never
`pip`/`conda` installs anything and never follows redirects off that host.

```bash
conda activate reinvent4
cd /path/to/reinvent-agent

# Dry-run: validate the query id; does not hit the network; invents no rows
python -m tools.chembl \
  --project projects/demo_project \
  --query-id egfr-ic50

# Human-approved fetch (prints the predefined filters, then Proceed? or --yes)
python -m tools.chembl \
  --project projects/demo_project \
  --query-id egfr-ic50 \
  --limit 50 \
  --approve-chembl --yes

# Optional: canonicalize SMILES locally into input/chembl_egfr-ic50.smi
python -m tools.chembl \
  --project projects/demo_project \
  --query-id egfr-ic50 \
  --approve-chembl --yes \
  --prep-smiles
```

`--limit` is capped (200). There is no `--sql`, `--url`, `--curl`, or
`--endpoint` flag.

Outputs (gitignored except `.gitkeep`):

- `output/chembl/activities.csv` — fixed assay schema
- `output/chembl/smiles.csv` — rows that already had `canonical_smiles`
- `output/chembl/chembl_result.json` — machine-readable result (+ Critic input)

## Assay CSV schema

Fixed columns (order stable):

`molecule_chembl_id`, `canonical_smiles`, `target_chembl_id`,
`standard_type`, `standard_value`, `standard_units`, `pchembl_value`,
`assay_chembl_id`, `activity_id`, `query_id`, `origin`

Missing SMILES stay empty; the tool does not look them up from a second
endpoint in this MVP scaffold.

## Safety invariants

- `--approve-chembl` ≠ `--approve-run` / dock / md / literature
- Predefined `query_id` allowlist only
- HTTPS host must be `www.ebi.ac.uk`; path must start `/chembl/api/data`
- Network failure → loud `NETWORK FAILURE`; refuse invented rows
- Not in Planner / MCP allowlist; Executor warns and skips chembl step names
- Optional `--prep-smiles` is local RDKit only (same as `tools.smiles_prep`)

## Tests

`tests/test_chembl.py` — mocked ChEMBL JSON only; no live network, no GPU,
no API keys.
