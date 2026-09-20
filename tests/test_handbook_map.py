"""Handbook mapping is ids only — no copied tutorial prose."""

from __future__ import annotations

from pathlib import Path

from agents.plan_schema import ALLOWED_STEPS
from tools.handbook_map import (
    HANDBOOK_REPO,
    HANDBOOK_SITE,
    HANDBOOK_TOOL_MAP,
    INDEPENDENT_TOOL_IDS,
    STABLE_TOOL_IDS,
    list_handbook_mappings,
    mapping_catalog,
)

REPO = Path(__file__).resolve().parent.parent
MAP_FILE = REPO / "tools" / "handbook_map.py"


def test_stable_tool_ids_cover_planner_allowlist_and_independent_clis():
    assert tuple(ALLOWED_STEPS) == STABLE_TOOL_IDS[: len(ALLOWED_STEPS)]
    assert set(ALLOWED_STEPS).isdisjoint(INDEPENDENT_TOOL_IDS)
    assert INDEPENDENT_TOOL_IDS <= set(STABLE_TOOL_IDS)
    assert "literature" in STABLE_TOOL_IDS
    assert "docking" in STABLE_TOOL_IDS
    assert "md" in STABLE_TOOL_IDS
    assert "run_reinvent" in STABLE_TOOL_IDS
    assert "write_paper" not in STABLE_TOOL_IDS
    assert "shell" not in STABLE_TOOL_IDS


def test_mapping_rows_use_only_stable_ids_and_md_paths():
    rows = list_handbook_mappings()
    assert rows
    ids = {row["tool_id"] for row in rows}
    assert ids <= set(STABLE_TOOL_IDS)
    for row in rows:
        assert row["tutorial_section"].endswith(".md")
        assert "/" in row["tutorial_section"]
        assert row["handbook_url"].startswith(HANDBOOK_SITE)
        assert ".md" not in row["handbook_url"]
        # No copied chapter body — values are identifiers / URLs only.
        for value in row.values():
            assert len(value) < 220
            assert "Every tutorial is reproducible" not in value
            assert "staged_learning" not in value


def test_mapping_catalog_points_at_separate_handbook_repo():
    catalog = mapping_catalog()
    assert catalog["handbook_repo"] == HANDBOOK_REPO
    assert "does not copy the Handbook" in catalog["note"]
    assert catalog["planner_allowlist"] == list(ALLOWED_STEPS)
    assert "literature" in catalog["independent_tool_ids"]


def test_handbook_map_module_does_not_embed_tutorial_prose():
    text = MAP_FILE.read_text(encoding="utf-8")
    assert "AI-Drug-Discovery-Lab" in text or "ai-drug-discovery-lab" in text
    assert "Real workflows. Real code. Real papers." not in text
    assert "12 章战役" not in text
    assert len(HANDBOOK_TOOL_MAP) < 40
    assert len(text) < 12_000
