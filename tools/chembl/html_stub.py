"""Optional short HTML stub for ChEMBL assay evidence.

Written only when an activities CSV is present. Never invents assay rows or
IC50 values — the stub quotes counts and paths from the real download.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def write_chembl_html_stub(
    output_dir: Path,
    *,
    result: dict[str, Any],
) -> Path | None:
    """Write ``chembl_stub.html`` when ``table_present`` and a CSV path exist.

    Returns the stub path, or ``None`` when there is no assay table to quote.
    Dry-run / network failure / empty hits must not write a stub that looks
    like a successful assay download.
    """
    if not result.get("table_present"):
        return None
    activities_csv = result.get("activities_csv")
    if not activities_csv:
        return None
    csv_path = Path(str(activities_csv))
    if not csv_path.is_file():
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "chembl_stub.html"

    query = result.get("query") if isinstance(result.get("query"), dict) else {}
    query_id = result.get("query_id") or query.get("id") or "unknown"
    n_activities = int(result.get("n_activities") or 0)
    n_with_smiles = int(result.get("n_with_smiles") or 0)
    target = query.get("target_chembl_id") or ""
    standard_type = query.get("standard_type") or ""
    smiles_csv = result.get("smiles_csv") or ""
    message = result.get("message") or ""

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>ChEMBL assay stub — {_esc(query_id)}</title>
  <style>
    body {{
      margin: 1.5rem;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: #1c1a16;
      background: #f7f4ef;
      line-height: 1.5;
      max-width: 42rem;
    }}
    h1 {{ font-size: 1.25rem; margin: 0 0 0.5rem; }}
    .muted {{ color: #5c574e; font-size: 0.95rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{
      border: 1px solid #d9d2c5;
      padding: 0.4rem 0.6rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #efe9df; width: 11rem; }}
    code {{ font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>ChEMBL assay evidence</h1>
  <p class="muted">
    Short stub only. Critic may quote ChEMBL / IC50 only when this table is in
    the evidence JSON. Rows are never invented. Not a Planner / MCP / Executor
    step.
  </p>
  <table>
    <tr><th>query_id</th><td><code>{_esc(query_id)}</code></td></tr>
    <tr><th>target_chembl_id</th><td><code>{_esc(target)}</code></td></tr>
    <tr><th>standard_type</th><td>{_esc(standard_type)}</td></tr>
    <tr><th>n_activities</th><td>{n_activities}</td></tr>
    <tr><th>n_with_smiles</th><td>{n_with_smiles}</td></tr>
    <tr><th>activities.csv</th><td><code>{_esc(str(csv_path))}</code></td></tr>
    <tr><th>smiles.csv</th><td><code>{_esc(str(smiles_csv) if smiles_csv else "(none)")}</code></td></tr>
    <tr><th>message</th><td>{_esc(message)}</td></tr>
  </table>
</body>
</html>
"""
    dest.write_text(body, encoding="utf-8")
    return dest


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))
