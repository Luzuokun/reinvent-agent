"""Carb report pack: RMSD/Rg/COM/hbond/SASA + FEL/heatmap/energy + optional MM/PBSA."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tools import dumps_pretty
from tools.md.carb.analyze import (
    collect_run_dirs,
    gmx_mmpbsa_status,
    run_replica_analysis,
)
from tools.md.carb.constants import DEFAULT_PROTOCOL, DEFAULT_REF_T, OUTPUT_SUBDIR, PROTOCOL_ID
from tools.md.carb.status import read_status
from tools.md.environment import check_md_environment

RunFn = Callable[..., Any]

DISCLAIMER = (
    "These are protocol / smoke-test outputs for workflow wiring. "
    "They do not prove a binding mechanism, a bound pose, or a paper-level "
    "free-energy claim. Demo numbers must not be written up as science."
)


def report_carb(
    project_dir: str | Path,
    *,
    run_fn: RunFn | None = None,
    temperature: float = DEFAULT_REF_T,
) -> dict[str, Any]:
    project = Path(project_dir).expanduser().resolve()
    output_dir = (project / "output" / OUTPUT_SUBDIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = check_md_environment()
    gmx_path = env.get("gmx_path")
    job = read_status(output_dir)
    run_dirs = collect_run_dirs(output_dir)

    replica_reports: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        replica_reports.append(
            run_replica_analysis(
                run_dir,
                executable=str(gmx_path) if gmx_path else None,
                run_fn=run_fn,
                temperature=temperature,
            )
        )

    mmpbsa = gmx_mmpbsa_status()
    figures = _figure_list(replica_reports, mmpbsa)
    payload = {
        "ok": True,
        "kind": "carb-report",
        "protocol_id": PROTOCOL_ID,
        "default_protocol": DEFAULT_PROTOCOL,
        "project_dir": str(project),
        "output_dir": str(output_dir),
        "disclaimer": DISCLAIMER,
        "job_status": job,
        "environment": {"gmx": env.get("gmx"), "gmx_path": gmx_path},
        "replicas": replica_reports,
        "mmpbsa": mmpbsa,
        "figures": figures,
        "invented_trajectory": False,
        "message": "",
    }
    if not run_dirs:
        payload["ok"] = False
        payload["message"] = (
            "No carb run directory under output/carb/. Run prepare+run first. "
            "No analysis tables were invented."
        )
    elif all(not (item.get("tables") or {}) for item in replica_reports):
        payload["message"] = (
            "Report wrote the pack, but analysis tables are empty or skipped "
            "(missing trajectory or gmx). Numbers were not invented. " + DISCLAIMER
        )
    else:
        payload["message"] = (
            f"Wrote carb report pack under {output_dir}. " + DISCLAIMER
        )

    csv_index = output_dir / "report_tables.csv"
    _write_table_index(csv_index, replica_reports)
    html_path = output_dir / "report.html"
    html_path.write_text(_render_html(payload), encoding="utf-8")
    json_path = output_dir / "report.json"
    json_path.write_text(dumps_pretty(payload), encoding="utf-8")
    payload["html"] = str(html_path)
    payload["json"] = str(json_path)
    payload["csv_index"] = str(csv_index)
    return payload


def _figure_list(replica_reports: list[dict[str, Any]], mmpbsa: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = [
        ("rmsd", "RMSD vs time"),
        ("rg", "Radius of gyration"),
        ("com", "Amylose–Fucoidan COM distance"),
        ("hbond", "Hydrogen bonds"),
        ("sasa", "SASA"),
        ("fel", "Free-energy landscape (RMSD vs Rg)"),
        ("contact_heatmap", "Contact / minimum-distance heatmap"),
        ("energy_groups", "Energy-group decomposition"),
    ]
    figures: list[dict[str, Any]] = []
    have: set[str] = set()
    for rep in replica_reports:
        have.update((rep.get("tables") or {}).keys())
    for key, title in wanted:
        figures.append(
            {
                "id": key,
                "title": title,
                "present": key in have,
                "note": None if key in have else "skipped (no data; not invented)",
            }
        )
    figures.append(
        {
            "id": "mmpbsa",
            "title": "MM/PBSA (optional)",
            "present": mmpbsa.get("status") == "available",
            "note": mmpbsa.get("message"),
        }
    )
    return figures


def _write_table_index(dest: Path, replica_reports: list[dict[str, Any]]) -> None:
    import csv

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["replica_dir", "table", "path", "n"])
        for rep in replica_reports:
            for key, meta in (rep.get("tables") or {}).items():
                writer.writerow(
                    [
                        rep.get("run_dir"),
                        key,
                        meta.get("csv") or meta.get("path"),
                        meta.get("n") or meta.get("n_frames") or "",
                    ]
                )


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _render_html(payload: dict[str, Any]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    figures = "".join(
        f"<li><strong>{_esc(item['title'])}</strong> — "
        f"{'present' if item.get('present') else 'skipped'}"
        f"{(' (' + _esc(item.get('note')) + ')') if item.get('note') else ''}</li>"
        for item in payload.get("figures") or []
    )
    skipped = []
    for rep in payload.get("replicas") or []:
        skipped.extend(rep.get("skipped") or [])
    skip_html = (
        "<ul>" + "".join(f"<li>{_esc(s)}</li>" for s in skipped) + "</ul>"
        if skipped
        else "<p>None.</p>"
    )
    mmpbsa = payload.get("mmpbsa") or {}
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Amylose–fucoidan MD report</title>
  <style>
    body {{ font-family: sans-serif; max-width: 880px; margin: 2rem auto; color: #222; }}
    .banner {{ background: #fff3cd; border: 1px solid #e6c35c; padding: 0.8rem 1rem; }}
    pre {{ background: #f6f6f6; padding: 0.8rem; overflow: auto; }}
    h1 {{ margin-bottom: 0.2rem; }}
  </style>
</head>
<body>
  <h1>Amylose–fucoidan MD report</h1>
  <p class="muted">protocol_id={_esc(PROTOCOL_ID)} · { _esc(stamp) }</p>
  <p class="banner"><strong>Disclaimer:</strong> {_esc(payload.get("disclaimer"))}</p>
  <h2>Job status</h2>
  <pre>{_esc(json.dumps(payload.get("job_status"), indent=2, default=str))}</pre>
  <h2>Figures (scheme §11 subset)</h2>
  <ul>{figures}</ul>
  <h2>MM/PBSA</h2>
  <p>{_esc(mmpbsa.get("status"))}: {_esc(mmpbsa.get("message"))}</p>
  <h2>Skipped / missing (not invented)</h2>
  {skip_html}
  <h2>Message</h2>
  <p>{_esc(payload.get("message"))}</p>
</body>
</html>
"""
