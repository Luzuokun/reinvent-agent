# Independent literature / research tool (PubMed)

Phase 7 adds a **directory-level literature tool**. It queries NCBI PubMed
E-utilities with a human-supplied search string and writes **only sourced
entries** (URL and/or PMID and/or DOI) under
`<project>/output/literature/`.

This is **not** a Planner step and **not** part of the REINVENT Executor.
Human approval (`--approve-literature`) is separate from `--approve-run`,
`--approve-dock`, and `--approve-md`. The module never writes papers.

Missing network is a **loud failure**. The tool does not invent citations
to fill the gap.

## Architecture

```text
python -m tools.literature --approve-literature --yes
        │
        ▼
sandbox paths   output ∈ <project>/output/literature/
        │
        ▼
[Human approval]  --approve-literature + confirm / --yes
        │
        ▼
HTTPS GET eutils.ncbi.nlm.nih.gov  (esearch + esummary only)
        │
        ▼
keep entries that have URL and/or PMID and/or DOI
        │         drop the rest; invent nothing
        ▼
output/literature/entries.csv + literature_result.json
        │
        ▼
Critic  (only if that literature object is in the evidence JSON
         *and* sourced entries are present)
```

| Component | Role |
|-----------|------|
| `tools/literature/` | Independent package: PubMed client, sourced-entry filter, CLI |
| `python -m tools.literature` | Only entry that may hit NCBI |
| `tools.environment.check_environment` | Unchanged: REINVENT / RDKit / GPU only |
| Executor / planner allowlist / MCP | Unchanged: no `pubmed`, no `literature`, no `write_paper` |
| Critic | May mention PubMed / literature / PMID only when sourced entries are in evidence |

No LLM-written TOML/mdp, no arbitrary shell, no SaaS/Web UI, no manuscript
generator. Handbook tutorials stay in
[AI-Drug-Discovery-Lab](https://github.com/Luzuokun/ai-drug-discovery-lab);
this repo only records [stable tool ids](handbook-tools.md).

## How to run

No extra packages. The client uses the Python standard library
(`urllib.request`) against NCBI E-utilities only. This module never
`pip`/`conda` installs anything and never follows redirects off
`eutils.ncbi.nlm.nih.gov`.

```bash
conda activate reinvent4
cd /path/to/reinvent-agent

# Dry-run: validate the query; does not hit the network; invents no papers
python -m tools.literature \
  --project projects/demo_project \
  --query "EGFR tyrosine kinase inhibitor"

# Human-approved fetch (prints the predefined query, then Proceed? or --yes)
python -m tools.literature \
  --project projects/demo_project \
  --query "EGFR tyrosine kinase inhibitor" \
  --retmax 10 \
  --approve-literature --yes
```

`--query` is a PubMed search string, not a URL and not a shell command.
`--retmax` is capped (50). There is no `--url`, `--html`, or `--write-paper`.

Outputs (gitignored except `.gitkeep`):

- `<project>/output/literature/entries.csv` — only when sourced hits exist
- `<project>/output/literature/literature_result.json`

If NCBI is unreachable, `literature_result.json` records
`NETWORK FAILURE: …`, `entries` is empty, and no CSV is written.

## Critic

The REINVENT CLI does **not** load literature tables. The literature CLI
builds a results payload with a `literature` object and runs the
deterministic critic.

- If that object is absent, LLM critic text mentioning PubMed / literature /
  PMID / “literature shows” is schema-rejected (same as before).
- If `evidence.literature.table_present` is true **and** the block lists
  sourced URL/PMID/DOI entries, the critic may quote **those** identifiers.
- Inventing an extra PMID or DOI still fails validation.
- Docking / MD / IC50 / binding-affinity claims stay forbidden unless their
  own evidence tables are present.

## Safety

1. Never `subprocess.run(..., shell=True)`.
2. The only host is `eutils.ncbi.nlm.nih.gov` over HTTPS. The query is
   encoded; it is never interpolated into a shell.
3. Tables are written only under `output/literature/`.
4. `--approve-literature` is independent of `--approve-run` /
   `--approve-dock` / `--approve-md`. None of these flags launches the
   others, and none writes a paper.
5. No auto-install. Missing network is a loud failure, not invented PMIDs.
6. Records without URL, PMID, *and* DOI are dropped.

## Tests

```bash
python -m pytest tests/test_literature.py tests/ -q
```

Tests inject a fake GET function. They do not need GPU, API keys, or a live
NCBI connection.

## Out of scope

ChEMBL download, arbitrary HTTP, MCP literature tools, auto-writing papers,
adding `pubmed` to the REINVENT Executor loop, copying the Handbook into
this repo. Docking remains `python -m tools.docking`; MD remains
`python -m tools.md`.
