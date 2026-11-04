"""Deterministic cross-modal ranking — no LLM, no invented numbers.

Reads an existing molecules CSV. Uses QED / SA / PAINS / dock / MD summary
columns **only when present**. Missing values contribute nothing (weight
skipped for that modality on that row); empty cells stay empty in the
shortlist — they are never filled with made-up scores.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tools.portfolio.presets import FIELD_ALIASES, PortfolioPresetError

# Fixed shortlist schema (stable column order).
SHORTLIST_FIELDS: tuple[str, ...] = (
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
    "preset_id",
    "source_row",
)


class PortfolioRankError(ValueError):
    """Unreadable CSV or missing required modality for a preset."""


def _norm_header(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def map_columns(fieldnames: list[str] | None) -> dict[str, str]:
    """Map canonical modality → actual CSV header (first alias wins)."""
    if not fieldnames:
        return {}
    by_norm = {_norm_header(h): h for h in fieldnames if h is not None}
    mapping: dict[str, str] = {}
    for canon, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias in by_norm:
                mapping[canon] = by_norm[alias]
                break
    return mapping


def _parse_float(raw: Any) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_pains(raw: Any) -> bool | None:
    """Return True=hit, False=clean, None=unknown/missing."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text or text in {"nan", "none", "null", "na", "-"}:
        return None
    if text in {"hit", "true", "1", "yes", "pains", "flag"}:
        return True
    if text in {"clean", "false", "0", "no", "ok"}:
        return False
    # Numeric count: >0 means hit
    as_float = _parse_float(text)
    if as_float is not None:
        return as_float > 0
    return None


def _parse_md(raw: Any, header_norm: str) -> float | None:
    """MD summary: numeric as-is; boolean in_md → 1.0 / 0.0; never invent."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "-"}:
        return None
    if header_norm == "in_md" or text.lower() in {"true", "false", "yes", "no"}:
        low = text.lower()
        if low in {"true", "1", "yes"}:
            return 1.0
        if low in {"false", "0", "no"}:
            return 0.0
    return _parse_float(text)


def load_molecule_rows(csv_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise PortfolioRankError(f"Molecules CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = map_columns(list(reader.fieldnames or []))
        if "smiles" not in columns:
            raise PortfolioRankError(
                f"No SMILES column in {path}. Expected one of: "
                f"{', '.join(FIELD_ALIASES['smiles'])}"
            )
        rows: list[dict[str, Any]] = []
        for index, raw in enumerate(reader):
            smiles_h = columns["smiles"]
            smiles = str(raw.get(smiles_h) or "").strip()
            record: dict[str, Any] = {
                "source_row": index,
                "SMILES": smiles,
                "ligand_id": "",
                "QED": None,
                "SA": None,
                "PAINS": None,
                "dock_score": None,
                "md_summary": None,
                "present": set(),
            }
            if "id" in columns:
                lid = str(raw.get(columns["id"]) or "").strip()
                record["ligand_id"] = lid
            if "qed" in columns:
                val = _parse_float(raw.get(columns["qed"]))
                if val is not None:
                    record["QED"] = val
                    record["present"].add("qed")
            if "sa" in columns:
                val = _parse_float(raw.get(columns["sa"]))
                if val is not None:
                    record["SA"] = val
                    record["present"].add("sa")
            if "pains" in columns:
                pains = _parse_pains(raw.get(columns["pains"]))
                if pains is not None:
                    record["PAINS"] = pains
                    record["present"].add("pains")
            if "dock" in columns:
                val = _parse_float(raw.get(columns["dock"]))
                if val is not None:
                    record["dock_score"] = val
                    record["present"].add("dock")
            if "md" in columns:
                md_h = columns["md"]
                val = _parse_md(raw.get(md_h), _norm_header(md_h))
                if val is not None:
                    record["md_summary"] = val
                    record["present"].add("md")
            rows.append(record)
    return rows, columns


def composite_score(
    row: dict[str, Any],
    weights: dict[str, float],
) -> tuple[float | None, list[str]]:
    """Weighted sum over modalities present on the row. No invented inputs."""
    used: list[str] = []
    total = 0.0
    present: set[str] = set(row.get("present") or ())
    contrib = 0

    if "qed" in present and weights.get("qed"):
        total += float(weights["qed"]) * float(row["QED"])
        used.append("qed")
        contrib += 1
    if "sa" in present and weights.get("sa"):
        total += float(weights["sa"]) * float(row["SA"])
        used.append("sa")
        contrib += 1
    if "pains" in present and weights.get("pains"):
        pains_val = 1.0 if row["PAINS"] is True else 0.0
        total += float(weights["pains"]) * pains_val
        used.append("pains")
        contrib += 1
    if "dock" in present and weights.get("dock"):
        total += float(weights["dock"]) * float(row["dock_score"])
        used.append("dock")
        contrib += 1
    if "md" in present and weights.get("md"):
        total += float(weights["md"]) * float(row["md_summary"])
        used.append("md")
        contrib += 1

    if contrib == 0:
        return None, []
    return round(total, 6), used


def rank_molecules(
    rows: list[dict[str, Any]],
    preset: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply a human weight preset. Drops empty SMILES; optional PAINS drop."""
    weights = dict(preset.get("weights") or {})
    drop_pains = bool(preset.get("drop_pains", True))
    require = tuple(preset.get("require_fields") or ())
    for field in require:
        # Preset may require a modality to exist on *some* input column set;
        # per-row missing still skips that term (no invent). Enforce at call
        # site via ``assert_required_columns``.
        if field not in ("qed", "sa", "pains", "dock", "md"):
            raise PortfolioPresetError(f"Invalid require_fields entry: {field!r}")

    scored: list[dict[str, Any]] = []
    for row in rows:
        smiles = str(row.get("SMILES") or "").strip()
        if not smiles:
            continue
        if drop_pains and row.get("PAINS") is True:
            continue
        # If preset requires dock (etc.) on the row, skip rows lacking it —
        # do not invent a dock score.
        skip = False
        for field in require:
            if field not in (row.get("present") or ()):
                skip = True
                break
        if skip:
            continue
        score, used = composite_score(row, weights)
        if score is None:
            continue
        out = {
            "SMILES": smiles,
            "ligand_id": row.get("ligand_id") or "",
            "QED": row.get("QED"),
            "SA": row.get("SA"),
            "PAINS": row.get("PAINS"),
            "dock_score": row.get("dock_score"),
            "md_summary": row.get("md_summary"),
            "composite_score": score,
            "fields_used": ",".join(used),
            "preset_id": preset.get("id"),
            "source_row": row.get("source_row"),
        }
        scored.append(out)

    scored.sort(
        key=lambda item: (
            -float(item["composite_score"]),
            str(item.get("SMILES") or ""),
        )
    )
    top_n = preset.get("top_n")
    if top_n is not None and int(top_n) > 0:
        return scored[: int(top_n)]
    return scored


def assert_required_columns(
    columns: dict[str, str],
    preset: dict[str, Any],
) -> None:
    """Fail loudly if the CSV lacks a modality the preset requires."""
    for field in preset.get("require_fields") or ():
        if field not in columns:
            aliases = FIELD_ALIASES.get(field, ())
            raise PortfolioRankError(
                f"Preset {preset.get('id')!r} requires a {field!r} column "
                f"(aliases: {', '.join(aliases)}). Refusing to invent values."
            )


def write_shortlist_csv(dest: str | Path, rows: list[dict[str, Any]]) -> Path:
    dest_path = Path(dest).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SHORTLIST_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            payload = dict(row)
            payload["rank"] = index
            pains = payload.get("PAINS")
            if pains is True:
                payload["PAINS"] = "hit"
            elif pains is False:
                payload["PAINS"] = "clean"
            else:
                payload["PAINS"] = ""
            for key in ("QED", "SA", "dock_score", "md_summary", "composite_score"):
                if payload.get(key) is None:
                    payload[key] = ""
            writer.writerow(payload)
    return dest_path
