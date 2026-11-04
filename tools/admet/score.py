"""RDKit / rule-based ADMET coarse screen — offline, deterministic.

Computes standard descriptors and evaluates a human-written preset's
thresholds. Never invents docking scores, MD metrics, IC50, or assay
values. Missing RDKit is a loud failure (no fabricated descriptor table).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tools.admet.presets import FIELD_ALIASES

SCORE_COLUMNS: tuple[str, ...] = (
    "SMILES",
    "ligand_id",
    "valid",
    "mw",
    "logp",
    "hbd",
    "hba",
    "tpsa",
    "rotatable_bonds",
    "qed",
    "preset_pass",
    "fail_reasons",
    "preset_id",
    "source_row",
)


class AdmetScoreError(ValueError):
    """Input CSV / RDKit / scoring failure (not an invented score)."""


def _import_rdkit() -> tuple[Any, Any, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED
    except ImportError as exc:
        raise AdmetScoreError(
            "RDKit is required for ADMET coarse screen and is not importable. "
            "Refusing to invent MW/logP/QED/rule outcomes. "
            f"Detail: {exc}"
        ) from exc
    return Chem, Descriptors, QED


def load_molecule_rows(csv_path: str | Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Load SMILES (+ optional id) from a CSV under the project."""
    path = Path(csv_path)
    if not path.is_file():
        raise AdmetScoreError(f"Molecules CSV not found: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise AdmetScoreError(f"Molecules CSV has no header: {path}")
        columns = _map_columns(list(reader.fieldnames))
        if "smiles" not in columns:
            raise AdmetScoreError(
                "Molecules CSV needs a SMILES column "
                f"(aliases: {FIELD_ALIASES['smiles']}). Refusing to invent SMILES."
            )
        rows: list[dict[str, str]] = []
        for i, raw in enumerate(reader):
            smiles = (raw.get(columns["smiles"]) or "").strip()
            if not smiles:
                continue
            ligand_id = ""
            if "id" in columns:
                ligand_id = (raw.get(columns["id"]) or "").strip()
            rows.append(
                {
                    "SMILES": smiles,
                    "ligand_id": ligand_id,
                    "source_row": str(i),
                }
            )
    if not rows:
        raise AdmetScoreError(
            f"No SMILES rows found in {path}. Refusing to invent molecules."
        )
    return rows, columns


def _map_columns(fieldnames: list[str]) -> dict[str, str]:
    lower = {name.strip().lower(): name for name in fieldnames if name}
    mapped: dict[str, str] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                mapped[canonical] = lower[alias]
                break
    return mapped


def score_molecules(
    rows: list[dict[str, str]],
    preset: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute descriptors and apply preset rules. Loud failure if RDKit missing."""
    Chem, Descriptors, QED = _import_rdkit()
    rules = dict(preset.get("rules") or {})
    preset_id = str(preset.get("id") or "")
    scored: list[dict[str, Any]] = []

    for row in rows:
        smiles = row["SMILES"]
        mol = Chem.MolFromSmiles(smiles)
        record: dict[str, Any] = {
            "SMILES": smiles,
            "ligand_id": row.get("ligand_id") or "",
            "valid": False,
            "mw": None,
            "logp": None,
            "hbd": None,
            "hba": None,
            "tpsa": None,
            "rotatable_bonds": None,
            "qed": None,
            "preset_pass": False,
            "fail_reasons": "invalid_smiles",
            "preset_id": preset_id,
            "source_row": row.get("source_row") or "",
        }
        if mol is None:
            scored.append(record)
            continue

        try:
            mw = float(Descriptors.MolWt(mol))
            logp = float(Descriptors.MolLogP(mol))
            hbd = float(Descriptors.NumHDonors(mol))
            hba = float(Descriptors.NumHAcceptors(mol))
            tpsa = float(Descriptors.TPSA(mol))
            rotb = float(Descriptors.NumRotatableBonds(mol))
            qed = float(QED.qed(mol))
        except Exception as exc:  # noqa: BLE001
            record["fail_reasons"] = f"descriptor_error:{exc}"
            scored.append(record)
            continue

        record.update(
            {
                "valid": True,
                "mw": round(mw, 4),
                "logp": round(logp, 4),
                "hbd": round(hbd, 4),
                "hba": round(hba, 4),
                "tpsa": round(tpsa, 4),
                "rotatable_bonds": round(rotb, 4),
                "qed": round(qed, 4),
            }
        )
        fails = _rule_failures(
            {
                "mw": mw,
                "logp": logp,
                "hbd": hbd,
                "hba": hba,
                "tpsa": tpsa,
                "rotatable_bonds": rotb,
                "qed": qed,
            },
            rules,
        )
        record["preset_pass"] = len(fails) == 0
        record["fail_reasons"] = ";".join(fails) if fails else ""
        scored.append(record)
    return scored


def _rule_failures(values: dict[str, float], rules: dict[str, Any]) -> list[str]:
    fails: list[str] = []
    checks: tuple[tuple[str, str, str], ...] = (
        ("mw_max", "mw", "max"),
        ("logp_max", "logp", "max"),
        ("hbd_max", "hbd", "max"),
        ("hba_max", "hba", "max"),
        ("tpsa_max", "tpsa", "max"),
        ("rotatable_bonds_max", "rotatable_bonds", "max"),
        ("qed_min", "qed", "min"),
    )
    for rule_key, value_key, direction in checks:
        if rule_key not in rules:
            continue
        threshold = float(rules[rule_key])
        value = float(values[value_key])
        if direction == "max" and value > threshold:
            fails.append(f"{value_key}>{threshold:g}")
        elif direction == "min" and value < threshold:
            fails.append(f"{value_key}<{threshold:g}")
    return fails


def write_scores_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SCORE_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key, "") for key in SCORE_COLUMNS}
            for key in ("preset_pass", "valid"):
                if payload[key] is True:
                    payload[key] = "true"
                elif payload[key] is False:
                    payload[key] = "false"
                elif payload[key] is None:
                    payload[key] = ""
            writer.writerow(payload)
    return dest
