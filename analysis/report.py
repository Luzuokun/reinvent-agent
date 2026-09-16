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


def _fmt_num(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _descriptor_table(rdkit: dict[str, Any]) -> str:
    rows = [
        ("MW", rdkit.get("mw") or {}),
        ("logP", rdkit.get("logp") or {}),
        ("QED", rdkit.get("qed") or {}),
        ("SA Score", rdkit.get("sa_score") or {}),
    ]
    body = []
    for name, stats in rows:
        if not isinstance(stats, dict):
            stats = {}
        available = stats.get("available")
        note = ""
        if name == "SA Score" and available is False:
            note = " <span class='muted'>(unavailable)</span>"
        body.append(
            "<tr>"
            f"<th>{_esc(name)}{note}</th>"
            f"<td>{_esc(_fmt_num(stats.get('n')))}</td>"
            f"<td>{_esc(_fmt_num(stats.get('mean')))}</td>"
            f"<td>{_esc(_fmt_num(stats.get('stdev')))}</td>"
            f"<td>{_esc(_fmt_num(stats.get('min')))}</td>"
            f"<td>{_esc(_fmt_num(stats.get('max')))}</td>"
            "</tr>"
        )
    return (
        "<table class='desc'>"
        "<thead><tr>"
        "<th>Descriptor</th><th>n</th><th>mean</th><th>stdev</th><th>min</th><th>max</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _svg_histogram(
    title: str,
    histogram: dict[str, Any] | None,
    *,
    xlabel: str,
    color: str,
) -> str:
    """Self-contained SVG bar chart. No JS, no frontend build."""
    hist = histogram if isinstance(histogram, dict) else {}
    counts = hist.get("counts") or []
    edges = hist.get("bin_edges") or []
    n = hist.get("n") or 0
    if not counts or not isinstance(counts, list) or n == 0:
        return (
            f"<figure class='chart'>"
            f"<figcaption>{_esc(title)}</figcaption>"
            f"<p class='muted'>No {_esc(xlabel)} values to plot "
            f"(RDKit unavailable or no valid molecules).</p>"
            f"</figure>"
        )

    width, height = 640, 252
    pad_l, pad_r, pad_t, pad_b = 48, 16, 28, 52
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    max_count = max(int(c) for c in counts) if counts else 1
    max_count = max(max_count, 1)
    n_bars = len(counts)
    gap = 3
    bar_w = max((plot_w - gap * max(n_bars - 1, 0)) / n_bars, 2)

    bars = []
    for i, raw in enumerate(counts):
        count = int(raw)
        h = 0 if max_count == 0 else (count / max_count) * plot_h
        x = pad_l + i * (bar_w + gap)
        y = pad_t + plot_h - h
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" '
            f'fill="{_esc(color)}" rx="2">'
            f"<title>{_esc(xlabel)} bin {i + 1}: count {count}</title></rect>"
        )

    y_ticks = [0, max_count]
    if max_count > 1:
        mid = max_count // 2
        if mid not in y_ticks:
            y_ticks.insert(1, mid)
    tick_els = []
    for tick in y_ticks:
        ty = pad_t + plot_h - (tick / max_count) * plot_h
        tick_els.append(
            f'<line x1="{pad_l}" y1="{ty:.2f}" x2="{width - pad_r}" y2="{ty:.2f}" '
            f'class="grid"/>'
            f'<text x="{pad_l - 8}" y="{ty + 4:.2f}" class="tick" text-anchor="end">'
            f"{tick}</text>"
        )

    x_min = edges[0] if edges else hist.get("min")
    x_max = edges[-1] if len(edges) > 1 else hist.get("max")
    axis = (
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" '
        f'y2="{pad_t + plot_h}" class="axis"/>'
        f'<text x="{pad_l}" y="{height - 14}" class="tick">{_esc(_fmt_num(x_min))}</text>'
        f'<text x="{width - pad_r}" y="{height - 14}" class="tick" text-anchor="end">'
        f"{_esc(_fmt_num(x_max))}</text>"
        f'<text x="{pad_l + plot_w / 2:.2f}" y="{height - 2}" class="xlabel" '
        f'text-anchor="middle">{_esc(xlabel)} (n={int(n)})</text>'
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{_esc(title)} histogram">'
        f"<title>{_esc(title)}</title>"
        "<style>"
        "text{font-family:'IBM Plex Sans',system-ui,sans-serif;fill:#1c1a16}"
        ".tick{font-size:11px;fill:#5c574e}"
        ".xlabel{font-size:12px;fill:#5c574e}"
        ".title{font-size:14px;font-weight:600}"
        ".grid{stroke:#eadfcd;stroke-width:1}"
        ".axis{stroke:#1c1a16;stroke-width:1.2}"
        "</style>"
        f'<text x="{pad_l}" y="18" class="title">{_esc(title)}</text>'
        f"{''.join(tick_els)}{''.join(bars)}{axis}"
        "</svg>"
    )
    return f"<figure class='chart'>{svg}</figure>"


def _render_analysis(analysis: dict[str, Any]) -> str:
    """Human-readable analysis: counts, descriptors, filters, SVG histograms."""
    if not isinstance(analysis, dict) or not analysis:
        return "<p class='muted'>No analysis payload.</p>"

    rdkit = analysis.get("rdkit") if isinstance(analysis.get("rdkit"), dict) else {}
    pains = rdkit.get("pains") if isinstance(rdkit.get("pains"), dict) else {}
    filters = rdkit.get("filters") if isinstance(rdkit.get("filters"), dict) else {}
    histograms = rdkit.get("histograms") if isinstance(rdkit.get("histograms"), dict) else {}

    summary = _kv_table(
        {
            "csv_path": analysis.get("csv_path"),
            "analysis_source": analysis.get("analysis_source"),
            "from_fresh_reinvent": analysis.get("from_fresh_reinvent"),
            "smiles_column": analysis.get("smiles_column"),
            "total_molecules": analysis.get("total_molecules"),
            "unique_molecules": analysis.get("unique_molecules"),
            "duplicate_molecules": analysis.get("duplicate_molecules"),
            "duplicate_fraction": analysis.get("duplicate_fraction"),
            "reinvent_smiles_state_valid": analysis.get("reinvent_smiles_state_valid"),
            "rdkit_available": rdkit.get("available"),
            "valid_molecules": rdkit.get("valid_molecules"),
            "invalid_molecules": rdkit.get("invalid_molecules"),
            "valid_fraction": rdkit.get("valid_fraction"),
        }
    )

    pains_rows = {
        "PAINS catalog available": pains.get("available"),
        "molecules with PAINS hits": pains.get("molecules_with_hits"),
        "total PAINS hits": pains.get("total_hits"),
        "PAINS hit fraction": pains.get("hit_fraction"),
        "QED pass threshold": filters.get("qed_pass_threshold"),
        "molecules with QED ≥ threshold": filters.get("qed_pass_count"),
        "PAINS-free molecules": filters.get("pains_free_count"),
        "QED ≥ threshold and PAINS-free": filters.get("qed_pass_and_pains_free_count"),
    }

    charts = (
        "<div class='charts'>"
        + _svg_histogram(
            "QED distribution",
            histograms.get("qed"),
            xlabel="QED",
            color="#1f6b3a",
        )
        + _svg_histogram(
            "Molecular weight distribution",
            histograms.get("mw"),
            xlabel="MW",
            color="#3d4f6b",
        )
        + _svg_histogram(
            "logP distribution",
            histograms.get("logp"),
            xlabel="logP",
            color="#8a5a00",
        )
        + "</div>"
    )

    raw = (
        "<details class='raw'><summary>Raw analysis JSON</summary>"
        f"<pre>{_esc(json.dumps(analysis, indent=2, default=str))}</pre></details>"
    )

    rdkit_note = ""
    if rdkit.get("available") is False:
        detail = rdkit.get("error") or "RDKit was not importable in this Python"
        rdkit_note = f"<p class='muted'>Descriptor charts skipped: {_esc(detail)}</p>"

    return (
        summary
        + "<h3>Descriptors</h3>"
        + _descriptor_table(rdkit)
        + "<h3>PAINS and simple filters</h3>"
        + _kv_table(pains_rows)
        + "<h3>Distributions</h3>"
        + rdkit_note
        + charts
        + raw
    )


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
        _section("Molecule analysis", _render_analysis(analysis if isinstance(analysis, dict) else {})),
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
    h3 {{
      margin: 1rem 0 0.5rem;
      font-size: 0.98rem;
      font-weight: 600;
    }}
    table.desc {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    table.desc th, table.desc td {{
      text-align: left;
      padding: 0.3rem 0.45rem 0.3rem 0;
      border-top: 1px solid var(--line);
    }}
    table.desc thead th {{
      color: var(--muted);
      font-weight: 600;
      border-top: none;
    }}
    .charts {{
      display: grid;
      gap: 0.75rem;
    }}
    .chart {{
      margin: 0;
      overflow: hidden;
    }}
    .chart svg {{
      width: 100%;
      max-width: 100%;
      height: auto;
      display: block;
      box-sizing: border-box;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    details.raw {{
      margin-top: 1rem;
      color: var(--muted);
    }}
    details.raw summary {{
      cursor: pointer;
      font-size: 0.9rem;
    }}
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
