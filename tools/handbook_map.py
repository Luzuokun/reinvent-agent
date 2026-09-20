"""Stable tool ids and Handbook tutorial-section ↔ tool-id mapping.

The Handbook lives in a **separate** repository
(https://github.com/Luzuokun/ai-drug-discovery-lab). This module does **not**
copy tutorial text. It only records canonical tool ids used by this repo and
a small table of MkDocs page paths that correspond to those ids.

``python -m tools.handbook_map`` prints the table as JSON.
"""

from __future__ import annotations

import json
from typing import Any

from agents.plan_schema import ALLOWED_STEPS

HANDBOOK_REPO = "https://github.com/Luzuokun/ai-drug-discovery-lab"
HANDBOOK_SITE = "https://luzuokun.github.io/ai-drug-discovery-lab"

# Canonical ids. Planner/MCP allowlist is a subset. Independent CLIs are not
# planner steps and must never be silently inserted into the REINVENT loop.
STABLE_TOOL_IDS: tuple[str, ...] = (
    "check_environment",
    "validate_project",
    "prepare_execution",
    "run_reinvent",
    "find_output",
    "analyze_molecules",
    "generate_report",
    "critic_review",
    "smiles_prep",
    "docking",
    "md",
    "literature",
)

INDEPENDENT_TOOL_IDS: frozenset[str] = frozenset(
    {"smiles_prep", "docking", "md", "literature"}
)

# MkDocs nav paths (relative to docs/) in AI-Drug-Discovery-Lab. One row per
# (section, tool_id). Sections without a matching tool are omitted — not faked.
HANDBOOK_TOOL_MAP: tuple[dict[str, str], ...] = (
    {
        "tutorial_section": "getting-started/conda.md",
        "tool_id": "check_environment",
    },
    {
        "tutorial_section": "getting-started/cuda.md",
        "tool_id": "check_environment",
    },
    {
        "tutorial_section": "molecular-generation/reinvent4/01-installation-first-molecule.md",
        "tool_id": "run_reinvent",
    },
    {
        "tutorial_section": "molecular-generation/reinvent4/02-priors-in-practice.md",
        "tool_id": "run_reinvent",
    },
    {
        "tutorial_section": "molecular-generation/reinvent4/03-scoring-function.md",
        "tool_id": "analyze_molecules",
    },
    {
        "tutorial_section": "molecular-generation/reinvent4/07-transfer-learning.md",
        "tool_id": "smiles_prep",
    },
    {
        "tutorial_section": "molecular-generation/reinvent4/07-transfer-learning.md",
        "tool_id": "run_reinvent",
    },
    {
        "tutorial_section": "molecular-generation/reinvent4/08-docking-guided-design.md",
        "tool_id": "docking",
    },
    {
        "tutorial_section": "docking/autodock-vina.md",
        "tool_id": "docking",
    },
    {
        "tutorial_section": "docking/gnina.md",
        "tool_id": "docking",
    },
    {
        "tutorial_section": "molecular-dynamics/gromacs.md",
        "tool_id": "md",
    },
    {
        "tutorial_section": "rdkit/descriptor.md",
        "tool_id": "analyze_molecules",
    },
    {
        "tutorial_section": "rdkit/scaffold.md",
        "tool_id": "analyze_molecules",
    },
    {
        "tutorial_section": "papers/index.md",
        "tool_id": "literature",
    },
    {
        "tutorial_section": "papers/braf-project.md",
        "tool_id": "literature",
    },
)


def handbook_page_url(tutorial_section: str) -> str:
    """Site URL for a MkDocs page path. Does not fetch or copy the page."""
    slug = tutorial_section.strip().removesuffix(".md").strip("/")
    return f"{HANDBOOK_SITE}/{slug}/"


def list_handbook_mappings() -> list[dict[str, str]]:
    """Return the mapping table with derived Handbook URLs (no page bodies)."""
    rows: list[dict[str, str]] = []
    for item in HANDBOOK_TOOL_MAP:
        section = item["tutorial_section"]
        tool_id = item["tool_id"]
        rows.append(
            {
                "tutorial_section": section,
                "tool_id": tool_id,
                "handbook_url": handbook_page_url(section),
            }
        )
    return rows


def mapping_catalog() -> dict[str, Any]:
    return {
        "handbook_repo": HANDBOOK_REPO,
        "handbook_site": HANDBOOK_SITE,
        "note": (
            "Mapping only. Tutorial prose lives in AI-Drug-Discovery-Lab; "
            "this repo does not copy the Handbook."
        ),
        "stable_tool_ids": list(STABLE_TOOL_IDS),
        "independent_tool_ids": sorted(INDEPENDENT_TOOL_IDS),
        "planner_allowlist": list(ALLOWED_STEPS),
        "mappings": list_handbook_mappings(),
    }


def main() -> int:
    print(json.dumps(mapping_catalog(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
