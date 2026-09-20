# Handbook tutorial-section ↔ tool-id mapping

The Handbook is a **separate** repository:
[Luzuokun/ai-drug-discovery-lab](https://github.com/Luzuokun/ai-drug-discovery-lab)
([live site](https://luzuokun.github.io/ai-drug-discovery-lab/)).

This repo does **not** copy tutorial prose, figures, or MkDocs pages. It
only publishes:

1. **Stable tool ids** used by reinvent-agent.
2. A **small mapping table** from Handbook MkDocs page paths to those ids.

Dump the machine-readable catalog:

```bash
python -m tools.handbook_map
```

## Stable tool ids

Planner / MCP allowlist (REINVENT executor — unchanged):

| tool id | Role |
|---------|------|
| `check_environment` | REINVENT / RDKit / GPU probe (never installs) |
| `validate_project` | Read-only project sandbox check |
| `prepare_execution` | Show predefined TOML + output paths |
| `run_reinvent` | Predefined `reinvent` argv; needs `--approve-run` |
| `find_output` | Inventory `<project>/output/` |
| `analyze_molecules` | Existing CSV descriptors |
| `generate_report` | HTML report |
| `critic_review` | Evidence-only PASS/WARNING/FAIL |

Independent CLIs (not planner/MCP steps; not silent Executor steps):

| tool id | CLI | Approval |
|---------|-----|----------|
| `smiles_prep` | `python -m tools.smiles_prep` | local files only |
| `docking` | `python -m tools.docking` | `--approve-dock` |
| `md` | `python -m tools.md` | `--approve-md` |
| `literature` | `python -m tools.literature` | `--approve-literature` |

There is no `write_paper`, `shell`, or `gmx` tool id.

## Mapping table

Page paths match the Handbook `mkdocs.yml` nav (files under `docs/`).
URLs are derived; bodies are not stored here. Sections without a matching
tool (Protein AI, Amber, DrugEx, …) are omitted rather than faked.

| Handbook section | tool id |
|------------------|---------|
| `getting-started/conda.md` | `check_environment` |
| `getting-started/cuda.md` | `check_environment` |
| `molecular-generation/reinvent4/01-installation-first-molecule.md` | `run_reinvent` |
| `molecular-generation/reinvent4/02-priors-in-practice.md` | `run_reinvent` |
| `molecular-generation/reinvent4/03-scoring-function.md` | `analyze_molecules` |
| `molecular-generation/reinvent4/07-transfer-learning.md` | `smiles_prep`, `run_reinvent` |
| `molecular-generation/reinvent4/08-docking-guided-design.md` | `docking` |
| `docking/autodock-vina.md` | `docking` |
| `docking/gnina.md` | `docking` |
| `molecular-dynamics/gromacs.md` | `md` |
| `rdkit/descriptor.md` | `analyze_molecules` |
| `rdkit/scaffold.md` | `analyze_molecules` |
| `papers/index.md` | `literature` |
| `papers/braf-project.md` | `literature` |

Source of truth: `tools/handbook_map.py`.

## Out of scope

Copying the Handbook into this repo, generating tutorial Markdown with an
LLM, adding 10 chatting agents, exposing literature/docking/MD through MCP.
