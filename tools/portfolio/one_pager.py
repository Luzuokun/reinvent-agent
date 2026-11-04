"""Campaign one-pager HTML — fixed fields, no auto-written paper.

Template slots (Chinese labels match the company Decision MVP):
靶点 / 批准协议 / 红旗 / top 分子 / 下一步

Numbers and SMILES come only from the shortlist rows already on disk.
Missing campaign meta stays empty — never invented prose.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_campaign_one_pager(
    dest: str | Path,
    *,
    project_dir: str | Path,
    preset_id: str,
    shortlist_rows: list[dict[str, Any]],
    shortlist_csv: str | Path | None,
    campaign: dict[str, Any] | None = None,
    approved: bool = False,
    warnings: list[str] | None = None,
) -> Path:
    """Write a fixed-field campaign card. Does not invent science claims."""
    dest_path = Path(dest).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(campaign or {})
    target = str(meta.get("target") or meta.get("靶点") or "").strip()
    protocols = str(meta.get("protocols") or meta.get("批准协议") or "").strip()
    red_flags = str(meta.get("red_flags") or meta.get("红旗") or "").strip()
    next_steps = str(meta.get("next_steps") or meta.get("下一步") or "").strip()
    campaign_id = str(meta.get("campaign_id") or meta.get("id") or "").strip()
    warn_list = [str(w) for w in (warnings or []) if w]

    rows_html = _rows_table(shortlist_rows)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    approval_banner = (
        "PI-approved shortlist (--approve-portfolio)"
        if approved
        else "DRAFT — not PI-approved (pass --approve-portfolio to sign)"
    )
    body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>Campaign one-pager — portfolio</title>
<style>
  body {{ font-family: Georgia, "Times New Roman", serif; margin: 2rem; color: #1a1a1a;
         background: linear-gradient(#f7f5f0, #ebe6dc); min-height: 100vh; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; }}
  .banner {{ padding: 0.6rem 0.8rem; margin: 1rem 0; border-left: 4px solid #8b4513;
             background: #fff8f0; }}
  .banner.draft {{ border-color: #a67c00; background: #fffbeb; }}
  section {{ margin: 1.4rem 0; }}
  h2 {{ font-size: 1.1rem; border-bottom: 1px solid #ccc; padding-bottom: 0.2rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; background: #fff; }}
  th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
  th {{ background: #f0ebe3; }}
  .empty {{ color: #666; font-style: italic; }}
  .meta {{ color: #444; font-size: 0.85rem; }}
  ul.warn {{ color: #6b3a00; }}
  code {{ font-family: ui-monospace, monospace; font-size: 0.85em; }}
</style>
</head>
<body>
  <h1>战役一页卡 / Campaign one-pager</h1>
  <p class="meta">project=<code>{_esc(project_dir)}</code>
     · preset=<code>{_esc(preset_id)}</code>
     · { _esc(stamp) }
     {f'· campaign_id=<code>{_esc(campaign_id)}</code>' if campaign_id else ''}</p>
  <div class="banner{' draft' if not approved else ''}">{_esc(approval_banner)}</div>

  <section>
    <h2>靶点 / Target</h2>
    {_para_or_empty(target)}
  </section>
  <section>
    <h2>批准协议 / Approved protocols</h2>
    {_para_or_empty(protocols)}
  </section>
  <section>
    <h2>红旗 / Red flags</h2>
    {_para_or_empty(red_flags)}
    {_warn_list(warn_list)}
  </section>
  <section>
    <h2>Top 分子 / Top molecules</h2>
    <p class="meta">Source: <code>{_esc(shortlist_csv or "(in-memory)")}</code>
       — values copied from input CSV; empty cells were not invented.</p>
    {rows_html}
  </section>
  <section>
    <h2>下一步 / Next steps</h2>
    {_para_or_empty(next_steps)}
  </section>
  <p class="meta">Local Decision MVP card — not a paper, not CRM.</p>
</body>
</html>
"""
    dest_path.write_text(body, encoding="utf-8")
    return dest_path


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _para_or_empty(text: str) -> str:
    if not text:
        return '<p class="empty">（未填写 — not provided; not invented）</p>'
    return f"<p>{_esc(text)}</p>"


def _warn_list(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<li>{_esc(w)}</li>" for w in warnings)
    return f'<ul class="warn">{items}</ul>'


def _rows_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No shortlisted molecules.</p>'
    headers = [
        "rank",
        "SMILES",
        "ligand_id",
        "QED",
        "SA",
        "PAINS",
        "dock_score",
        "md_summary",
        "composite_score",
        "fields_used",
    ]
    thead = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body_rows = []
    for index, row in enumerate(rows, start=1):
        cells = []
        for h in headers:
            if h == "rank":
                val = row.get("rank", index)
            elif h == "PAINS":
                pains = row.get("PAINS")
                if pains is True:
                    val = "hit"
                elif pains is False:
                    val = "clean"
                else:
                    val = pains if pains not in (None, "") else ""
            else:
                val = row.get(h)
                if val is None:
                    val = ""
            cells.append(f"<td>{_esc(val)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
