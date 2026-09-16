# CSV analysis, histograms, and optional QED critic threshold

This adds SA Score, PAINS counts, simple QED/PAINS filters, and embedded SVG
histograms on the existing offline CSV path. Planner, Executor, REINVENT
launch, and TOML handling are unchanged. No new Agent class.

## What changed

```text
analyze_molecules(CSV)
  → counts / duplicates (unchanged)
  → RDKit: validity, MW, logP, QED (unchanged)
  → SA Score (rdkit.Contrib.SA_Score; skip with warning if missing)
  → PAINS hits (RDKit FilterCatalog; skip with warning if missing)
  → filter counts: QED ≥ threshold, PAINS-free, both
  → histogram bins for QED / MW / logP

generate_html_report
  → summary tables + 3 inline SVG histograms (no JS build)
  → raw analysis JSON stays behind a <details> disclosure

CriticAgent
  → optional min_mean_qed; if QED mean is missing the check is skipped
    (WARNING that the threshold was not applied; no invented number)
```

## Configuration

`config/agent.yaml`:

```yaml
critic:
  min_mean_qed: null          # optional; skip if unset or QED mean is missing

analysis:
  qed_pass_threshold: 0.5     # count-only; not a critic fail
```

Missing RDKit / SA / PAINS never fabricates zeros that look like measured
counts. Those fields stay `null` / `available: false`.

## How to run

```bash
python main.py --project projects/demo_project \
  --goal "Analyze existing sample molecules." \
  --skip-reinvent \
  --csv projects/demo_project/output/sampled-sample.csv
```

Open the new `reports/report_*.html`. Expect QED / MW / logP charts and a
PAINS count. No API key, GPU, or TOML edit is required.

## Tests

```bash
python -m pytest tests/ -q
```

RDKit-dependent assertions use `pytest.importorskip("rdkit")`. Histogram SVG
rendering is tested with a synthetic payload so CI without RDKit still checks
charts. LLM critic tests remain mocked.

## Out of scope

MCP, docking, MD, literature, experiment preset IDs, rewriting `reinvent.toml`,
and any Agent framework (Agents SDK / LangGraph / CrewAI).
