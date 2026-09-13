"""Self-contained HTML scientific report generator."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools import REPO_ROOT, load_agent_config


def generate_html_report(
    payload: dict[str, Any],
    *,
    reports_dir: str | Path | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """Write a self-contained HTML report and return its metadata."""
    cfg = load_agent_config()
    if reports_dir is None:
        reports_dir = REPO_ROOT / cfg.get("logging", {}).get("reports_dirname", "reports")
    reports_dir = Path(reports_dir).expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_name = filename or f"report_{stamp}.html"
    out_path = reports_dir / out_name

    body = _render_html(payload)
    out_path.write_text(body, encoding="utf-8")
    return {
        "ok": True,
        "report_path": str(out_path),
        "created_at_utc": stamp,
    }


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _section(title: str, content: str) -> str:
    return f"<section><h2>{_esc(title)}</h2>{content}</section>"


def _kv_table(data: dict[str, Any]) -> str:
    rows = []
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            rendered = f"<pre>{_esc(json.dumps(value, indent=2, default=str))}</pre>"
        else:
            rendered = _esc(value)
        rows.append(f"<tr><th>{_esc(key)}</th><td>{rendered}</td></tr>")
    return "<table class='kv'>" + "".join(rows) + "</table>"


def _list_block(items: list[Any], empty: str = "None") -> str:
    if not items:
        return f"<p class='muted'>{_esc(empty)}</p>"
    return "<ul>" + "".join(f"<li>{_esc(item)}</li>" for item in items) + "</ul>"


def _render_html(payload: dict[str, Any]) -> str:
    goal = payload.get("goal", "")
    project = payload.get("project", {})
    environment = payload.get("environment", {})
    validation = payload.get("validation", {})
    execution = payload.get("execution", {})
    inventory = payload.get("inventory", {})
    analysis = payload.get("analysis", {})
    critic = payload.get("critic", {})
    warnings = payload.get("warnings", []) or []
    errors = payload.get("errors", []) or []

    critic_status = (critic or {}).get("status", "N/A")
    status_class = {
        "PASS": "pass",
        "WARNING": "warn",
        "FAIL": "fail",
    }.get(str(critic_status).upper(), "muted")

    dry_run = bool(payload.get("dry_run") or (isinstance(execution, dict) and execution.get("skipped")))
    analysis_source = payload.get("analysis_source") or (
        (analysis or {}).get("analysis_source") if isinstance(analysis, dict) else None
    )
    csv_path = (analysis or {}).get("csv_path") if isinstance(analysis, dict) else None

    banner = ""
    if dry_run or analysis_source == "existing_csv":
        csv_note = _esc(csv_path) if csv_path else "(path unavailable)"
        if dry_run:
            banner_text = (
                "Dry-run: REINVENT was not executed. "
                f"Analysis below is from existing output file: {csv_note}"
            )
        else:
            banner_text = (
                "Analysis below is from an existing output file "
                f"(not a fresh REINVENT generation): {csv_note}"
            )
        banner = (
            f"<div class='banner warn-banner' role='status'>"
            f"<strong>Notice:</strong> {banner_text}</div>"
        )

    parts = [
        _section("Project", _kv_table({"goal": goal, **(project if isinstance(project, dict) else {"project": project})})),
        _section("Environment", _kv_table(environment if isinstance(environment, dict) else {})),
        _section("Validation", _kv_table(validation if isinstance(validation, dict) else {})),
        _section("REINVENT execution", _kv_table(execution if isinstance(execution, dict) else {})),
        _section("Output inventory", _kv_table(inventory if isinstance(inventory, dict) else {})),
        _section("Molecule analysis", _kv_table(analysis if isinstance(analysis, dict) else {})),
        _section(
            "Critic",
            f"<p class='status {status_class}'>Status: {_esc(critic_status)}</p>"
            + _kv_table({k: v for k, v in (critic or {}).items() if k != "status"}),
        ),
        _section("Warnings", _list_block(list(warnings), empty="No warnings")),
        _section("Errors", _list_block(list(errors), empty="No errors")),
    ]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>REINVENT4 Agent Report</title>
  <style>
    :root {{
      --bg: #f7f4ef;
      --ink: #1c1a16;
      --muted: #5c574e;
      --line: #d9d2c5;
      --pass: #1f6b3a;
      --warn: #8a5a00;
      --fail: #8b1e1e;
      --card: #fffdf8;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #efe6d6 0, transparent 40%),
        linear-gradient(180deg, #f7f4ef, #efe9df);
      line-height: 1.5;
    }}
    main {{
      max-width: 920px;
      margin: 0 auto;
      padding: 2.5rem 1.25rem 4rem;
    }}
    h1 {{
      font-family: "IBM Plex Serif", Georgia, serif;
      font-weight: 600;
      font-size: 2rem;
      margin: 0 0 0.35rem;
    }}
    .subtitle {{ color: var(--muted); margin-bottom: 2rem; }}
    section {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 1rem 1.15rem 1.15rem;
      margin-bottom: 1rem;
    }}
    h2 {{
      margin: 0 0 0.75rem;
      font-size: 1.1rem;
      letter-spacing: 0.01em;
    }}
    table.kv {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    table.kv th {{
      text-align: left;
      vertical-align: top;
      width: 32%;
      padding: 0.35rem 0.5rem 0.35rem 0;
      color: var(--muted);
      font-weight: 600;
    }}
    table.kv td {{
      padding: 0.35rem 0;
      border-top: 1px solid var(--line);
      word-break: break-word;
    }}
    table.kv tr:first-child td {{ border-top: none; }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-size: 0.85rem;
    }}
    .status {{ font-weight: 700; font-size: 1.05rem; }}
    .status.pass {{ color: var(--pass); }}
    .status.warn {{ color: var(--warn); }}
    .status.fail {{ color: var(--fail); }}
    .muted {{ color: var(--muted); }}
    .banner {{
      border-radius: 10px;
      padding: 0.85rem 1rem;
      margin-bottom: 1rem;
      border: 1px solid var(--line);
      background: #fff6e0;
      color: var(--warn);
      font-size: 0.98rem;
    }}
    .banner.warn-banner {{
      border-color: #e0c48a;
    }}
    ul {{ margin: 0.25rem 0 0 1.1rem; }}
  </style>
</head>
<body>
  <main>
    <h1>REINVENT4 Agent Report</h1>
    <p class="subtitle">Deterministic MVP workflow summary — generated locally.</p>
    {banner}
    {''.join(parts)}
  </main>
</body>
</html>
"""
