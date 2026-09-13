"""Thin wrappers that re-export analysis helpers as tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analysis.molecule_analysis import analyze_molecules
from analysis.report import generate_html_report


def analyze_tool(csv_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    return analyze_molecules(csv_path, **kwargs)


def report_tool(
    report_payload: dict[str, Any],
    *,
    reports_dir: str | Path | None = None,
) -> dict[str, Any]:
    return generate_html_report(report_payload, reports_dir=reports_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Molecule analysis tool")
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_molecules(args.csv), indent=2, default=str))


if __name__ == "__main__":
    main()
