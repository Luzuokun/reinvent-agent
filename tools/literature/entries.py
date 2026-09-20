"""Sourced-entry helpers — drop records that lack URL / PMID / DOI."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tools.literature.pubmed import is_sourced

CSV_FIELDS = ("pmid", "doi", "url", "title", "journal", "pubdate", "origin")


def filter_sourced(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only records with PMID and/or DOI and/or http(s) URL."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        if not is_sourced(raw):
            continue
        row = {field: str(raw.get(field) or "").strip() for field in CSV_FIELDS}
        key = (row["pmid"], row["doi"].lower(), row["url"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_entries_csv(path: Path, entries: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in entries:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    return path
