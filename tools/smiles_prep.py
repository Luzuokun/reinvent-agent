"""Local SMILES prep for transfer learning — no network, no literature.

Reads an existing CSV or .smi already on disk, RDKit-canonicalizes, drops
invalid rows, deduplicates, and writes a training ``.smi`` under the project.
The source path must stay inside this repo (or the project). The destination
must stay under the project directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analysis.molecule_analysis import SMILES_CANDIDATES, _detect_smiles_column
from tools import REPO_ROOT, resolve_under_root

DEFAULT_OUTPUT_RELATIVE = Path("input") / "tl_train.smi"


class SmilesPrepError(ValueError):
    """Unsafe path or unreadable source."""


def prepare_training_smiles(
    project_dir: str | Path,
    source: str | Path,
    *,
    dest_relative: str | Path = DEFAULT_OUTPUT_RELATIVE,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write a cleaned ``.smi`` under ``project_dir``. Never fetches remote data."""
    project = Path(project_dir).expanduser().resolve()
    root = Path(repo_root or REPO_ROOT).expanduser().resolve()
    if not project.is_dir():
        raise SmilesPrepError(f"Project directory does not exist: {project}")

    source_path = _resolve_source(source, project=project, repo_root=root)
    dest_path = resolve_under_root(project, dest_relative)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _read_smiles_lines(source_path)
    total = len(raw)
    if total == 0:
        raise SmilesPrepError(f"No SMILES found in {source_path}")

    try:
        from rdkit import Chem
    except ImportError as exc:
        raise SmilesPrepError(
            "RDKit is required to canonicalize training SMILES"
        ) from exc

    unique: list[str] = []
    seen: set[str] = set()
    invalid = 0
    for line in raw:
        mol = Chem.MolFromSmiles(line)
        if mol is None:
            invalid += 1
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen:
            continue
        seen.add(canonical)
        unique.append(canonical)

    if not unique:
        raise SmilesPrepError(
            f"No valid SMILES after RDKit parse ({invalid} invalid of {total})"
        )

    dest_path.write_text("\n".join(unique) + "\n", encoding="utf-8")
    dropped = total - len(unique)
    return {
        "ok": True,
        "source": str(source_path),
        "dest": str(dest_path),
        "total": total,
        "valid_unique": len(unique),
        "invalid": invalid,
        "dropped": dropped,
        "duplicates_or_invalid": dropped,
    }


def _resolve_source(
    source: str | Path,
    *,
    project: Path,
    repo_root: Path,
) -> Path:
    raw = Path(source).expanduser()
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        candidate = None
        for base in (project, Path.cwd(), repo_root):
            trial = (base / raw).resolve()
            if trial.is_file():
                candidate = trial
                break
        if candidate is None:
            candidate = (Path.cwd() / raw).resolve()
    if not candidate.is_file():
        raise SmilesPrepError(f"Source file not found: {candidate}")
    for allowed in (repo_root, project):
        try:
            candidate.relative_to(allowed.resolve())
            return candidate
        except ValueError:
            continue
    raise SmilesPrepError(
        f"Source path escapes repo and project: {candidate} "
        f"(repo={repo_root}, project={project})"
    )


def _read_smiles_lines(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".smi", ".smiles"}:
        lines: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            token = text.split()[0].strip()
            if token:
                lines.append(token)
        return lines

    import pandas as pd

    df = pd.read_csv(path)
    col = _detect_smiles_column(list(df.columns))
    if col is None:
        # Headerless single-column file.
        df = pd.read_csv(path, header=None)
        if df.empty:
            return []
        col = df.columns[0]
        first = str(df.iloc[0, 0]).strip()
        if first.lower() in {name.lower() for name in SMILES_CANDIDATES}:
            df = df.iloc[1:]
    series = df[col].astype(str).fillna("")
    return [s.strip() for s in series.tolist() if s.strip() and s.strip().lower() != "nan"]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Canonicalize and deduplicate local SMILES into a TL training file"
    )
    parser.add_argument("--project", required=True, help="REINVENT project directory")
    parser.add_argument("--source", required=True, help="Existing CSV or .smi (inside repo)")
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_OUTPUT_RELATIVE),
        help="Destination relative to the project (default: input/tl_train.smi)",
    )
    args = parser.parse_args()
    result = prepare_training_smiles(args.project, args.source, dest_relative=args.dest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
