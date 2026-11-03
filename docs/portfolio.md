# Independent portfolio shortlist + campaign one-pager

Phase B Decision MVP: a **directory-level portfolio tool**. It ranks an
existing molecules CSV with **human-written weight preset ids** (QED / SA /
PAINS / dock / MD summary columns used only when present) and writes
`shortlist.csv` plus a fixed-field campaign one-pager under
`<project>/output/portfolio/`.

This is **not** a Planner step and **not** part of the REINVENT Executor.
Human approval (`--approve-portfolio`) is separate from `--approve-run`,
`--approve-dock`, `--approve-md`, `--approve-literature`, and
`--approve-chembl`.

There is **no LLM scorer**. Missing modality cells are skipped; values are
never invented to fill gaps. The local run index is **not** CRM.

## Architecture

```text
python -m tools.portfolio --preset-id balanced --approve-portfolio --yes
        │
        ▼
sandbox paths   CSV ∈ <project>/… ; output ∈ <project>/output/portfolio/
        │
        ▼
[Human approval]  --approve-portfolio + confirm / --yes
        │
        ▼
deterministic weighted sum (preset weights only; missing fields skipped)
        │
        ▼
output/portfolio/shortlist.csv
output/portfolio/campaign_one_pager.html   (靶点 / 批准协议 / 红旗 / top 分子 / 下一步)
output/portfolio/run_index.json            (local recall — not CRM)
output/portfolio/portfolio_result.json
        │
        ▼
Critic  (WARNING if shortlist not signed; FAIL only on hard errors)
```

| Component | Role |
|-----------|------|
| `tools/portfolio/` | Independent package: weight presets, ranker, one-pager, run index, CLI |
| `python -m tools.portfolio` | Only entry that may sign a shortlist |
| Executor / planner allowlist / MCP | Unchanged: no `portfolio` step |
| Critic | Soft WARNING without `--approve-portfolio`; may mention shortlist only when the table is in evidence |

## Weight presets

List them without scoring:

```bash
python -m tools.portfolio --list-presets
```

| preset_id | Intent |
|-----------|--------|
| `balanced` | QED↑ SA↓ PAINS penalty dock↓(Vina) + optional MD |
| `ligand-prop` | QED / SA / PAINS only (dock/MD weights zero) |
| `dock-first` | Strong dock weight; **requires** a dock_score column |

New weight vectors require a human PR that adds a row to
`tools/portfolio/presets.py`. There is no free-form weight JSON flag.

### Score formula (fixed direction)

For each row, only modalities **present** on that row contribute:

```text
composite = w_qed * QED + w_sa * SA + w_pains * (1 if PAINS hit else 0)
          + w_dock * dock_score + w_md * md_summary
```

- QED: higher better → typically `w_qed > 0`
- SA: lower better → typically `w_sa < 0`
- PAINS: hit penalty → typically `w_pains < 0` (and `drop_pains: true` removes hits)
- dock_score: more negative (Vina) better → typically `w_dock < 0`
- md_summary / `in_md`: higher better when present → typically `w_md > 0`

Empty cells stay empty in `shortlist.csv`.

## How to run

```bash
conda activate reinvent4
cd /path/to/reinvent-agent

# Dry-run: validate preset + CSV; preview in portfolio_result.json only
python -m tools.portfolio \
  --project projects/demo_project \
  --molecules-csv projects/demo_project/input/portfolio/molecules.csv \
  --preset-id balanced

# Human-approved shortlist (writes shortlist.csv + one-pager + run index)
python -m tools.portfolio \
  --project projects/demo_project \
  --molecules-csv projects/demo_project/input/portfolio/molecules.csv \
  --preset-id balanced \
  --target "EGFR (demo fixture)" \
  --protocols "sampling preset; dock vina (human-approved)" \
  --red-flags "fixture data only — not a discovery claim" \
  --next-steps "PI review shortlist; no wet-lab claim yet" \
  --approve-portfolio --yes
```

`--molecules-csv` must stay under the project directory. Demo ships
`projects/demo_project/input/portfolio/molecules.csv` (fixture numbers only).

Outputs:

- `output/portfolio/shortlist.csv` — ranked shortlist (**only** with `--approve-portfolio`)
- `output/portfolio/campaign_one_pager.html` — fixed fields (靶点 / 批准协议 / 红旗 / top 分子 / 下一步)
- `output/portfolio/run_index.json` — append-only local index (`crm: false`)
- `output/portfolio/portfolio_result.json` — machine-readable result (+ Critic input)

## Shortlist CSV schema

Fixed columns (order stable):

`rank`, `SMILES`, `ligand_id`, `QED`, `SA`, `PAINS`, `dock_score`,
`md_summary`, `composite_score`, `fields_used`, `preset_id`, `source_row`

## Safety invariants

- `--approve-portfolio` ≠ `--approve-run` / dock / md / literature / chembl
- Predefined `preset_id` allowlist only
- No LLM scorer; no invented QED / SA / dock / MD values
- Not in Planner / MCP allowlist; Executor warns and skips portfolio step names
- Soft WARNING when shortlist is not signed; no hard-block of other modules
- Local run index is not CRM

## Tests

`tests/test_portfolio.py` — fixture CSV only; no GPU, no network, no LLM.
