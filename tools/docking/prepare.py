"""Prepare receptor / ligand files for Vina or GNINA.

Inputs must already exist under the project sandbox. Conversion uses RDKit
(3D embed) plus Meeko or Open Babel when the source is not already PDBQT.
Never fetches remote structures. Never ``shell=True``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from analysis.molecule_analysis import _load_smiles_frame
from tools.docking.engines import clamp_max_ligands

RECEPTOR_SUFFIXES = {".pdbqt", ".pdb"}
LIGAND_FILE_SUFFIXES = {".pdbqt", ".sdf", ".mol", ".smi", ".smiles", ".csv"}
INPUT_DIRNAME = "input"
OUTPUT_DIRNAME = "output"


class DockingPrepareError(ValueError):
    """Unsafe path, missing file, or conversion failure."""


def resolve_receptor_path(project_dir: str | Path, receptor: str | Path) -> Path:
    """Receptor must already exist under ``<project>/input/``."""
    project = Path(project_dir).expanduser().resolve()
    input_dir = (project / INPUT_DIRNAME).resolve()
    return _resolve_existing(receptor, project=project, allowed_roots=(input_dir,))


def resolve_ligands_path(project_dir: str | Path, ligands: str | Path) -> Path:
    """Ligand file must already exist under ``input/`` or ``output/``."""
    project = Path(project_dir).expanduser().resolve()
    input_dir = (project / INPUT_DIRNAME).resolve()
    output_dir = (project / OUTPUT_DIRNAME).resolve()
    return _resolve_existing(
        ligands, project=project, allowed_roots=(input_dir, output_dir)
    )


def prepare_receptor(
    receptor: str | Path,
    dest_dir: str | Path,
    *,
    obabel_path: str | None = None,
) -> dict[str, Any]:
    """Copy or convert a receptor to ``dest_dir/receptor.pdbqt``."""
    src = Path(receptor).expanduser().resolve()
    dest_dir = Path(dest_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "receptor.pdbqt"
    suffix = src.suffix.lower()
    if suffix not in RECEPTOR_SUFFIXES:
        raise DockingPrepareError(
            f"Unsupported receptor suffix {src.suffix!r}; use .pdbqt or .pdb"
        )
    if suffix == ".pdbqt":
        shutil.copy2(src, dest)
        return {
            "ok": True,
            "source": str(src),
            "pdbqt": str(dest),
            "converted": False,
        }
    if not obabel_path:
        raise DockingPrepareError(
            "Receptor is PDB; Open Babel (obabel) is required to write PDBQT, "
            "or pass a .pdbqt receptor"
        )
    _run_obabel(
        [
            obabel_path,
            "-ipdb",
            str(src),
            "-opdbqt",
            "-xr",
            "-O",
            str(dest),
        ]
    )
    if not dest.is_file():
        raise DockingPrepareError(f"obabel did not write receptor PDBQT: {dest}")
    return {
        "ok": True,
        "source": str(src),
        "pdbqt": str(dest),
        "converted": True,
        "converter": "obabel",
    }


def load_ligand_records(
    ligands_path: str | Path,
    *,
    max_ligands: int = 20,
) -> list[dict[str, Any]]:
    """Read ligand identities from PDBQT/SDF/SMILES/CSV. No docking yet."""
    path = Path(ligands_path).expanduser().resolve()
    limit = clamp_max_ligands(max_ligands)
    suffix = path.suffix.lower()
    if suffix not in LIGAND_FILE_SUFFIXES:
        raise DockingPrepareError(
            f"Unsupported ligands suffix {path.suffix!r}; "
            "use .pdbqt, .sdf, .smi, or .csv"
        )
    if suffix == ".pdbqt":
        return [{"ligand_id": "lig_000", "smiles": None, "source": str(path), "kind": "pdbqt"}]
    if suffix in {".sdf", ".mol"}:
        return [{"ligand_id": "lig_000", "smiles": None, "source": str(path), "kind": "sdf"}]

    df, smiles_col = _load_smiles_frame(path)
    if smiles_col is None:
        raise DockingPrepareError(f"No SMILES column in {path}")
    name_col = _detect_name_column(list(df.columns), smiles_col)
    records: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        if len(records) >= limit:
            break
        smiles = str(row[smiles_col]).strip()
        if not smiles or smiles.lower() == "nan":
            continue
        ligand_id = f"lig_{len(records):03d}"
        if name_col is not None:
            raw_name = str(row[name_col]).strip()
            slug = _slug(raw_name)
            if slug:
                ligand_id = f"{slug}_{len(records):03d}"
        records.append(
            {
                "ligand_id": ligand_id,
                "smiles": smiles,
                "source": str(path),
                "kind": "smiles",
                "row": int(index) if isinstance(index, int) else len(records),
            }
        )
    if not records:
        raise DockingPrepareError(f"No SMILES found in {path}")
    return records


def prepare_ligand(
    record: dict[str, Any],
    dest_dir: str | Path,
    *,
    obabel_path: str | None = None,
    meeko_available: bool = False,
) -> dict[str, Any]:
    """Write one ligand PDBQT under ``dest_dir``."""
    dest_dir = Path(dest_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    ligand_id = str(record.get("ligand_id") or "lig_000")
    dest = dest_dir / f"{ligand_id}.pdbqt"
    kind = record.get("kind")
    source = Path(str(record["source"])).expanduser().resolve()

    if kind == "pdbqt":
        shutil.copy2(source, dest)
        return {
            "ok": True,
            "ligand_id": ligand_id,
            "smiles": record.get("smiles"),
            "pdbqt": str(dest),
            "converted": False,
        }

    sdf_path = dest_dir / f"{ligand_id}.sdf"
    if kind == "sdf":
        shutil.copy2(source, sdf_path)
    elif kind == "smiles":
        smiles = record.get("smiles")
        if not smiles:
            raise DockingPrepareError(f"{ligand_id}: missing SMILES")
        _smiles_to_sdf(str(smiles), sdf_path)
    else:
        raise DockingPrepareError(f"{ligand_id}: unknown ligand kind {kind!r}")

    if meeko_available:
        _sdf_to_pdbqt_meeko(sdf_path, dest)
        converter = "meeko"
    elif obabel_path:
        _run_obabel([obabel_path, "-isdf", str(sdf_path), "-opdbqt", "-O", str(dest)])
        converter = "obabel"
    else:
        raise DockingPrepareError(
            "Cannot convert ligand to PDBQT: install Open Babel (obabel) or "
            "Meeko, or pass a .pdbqt ligands file"
        )
    if not dest.is_file():
        raise DockingPrepareError(f"Conversion did not write {dest}")
    return {
        "ok": True,
        "ligand_id": ligand_id,
        "smiles": record.get("smiles"),
        "pdbqt": str(dest),
        "converted": True,
        "converter": converter,
    }


def _resolve_existing(
    raw: str | Path,
    *,
    project: Path,
    allowed_roots: tuple[Path, ...],
) -> Path:
    if raw is None or str(raw).strip() == "":
        raise DockingPrepareError("path is empty")
    path = Path(str(raw).strip()).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path.resolve())
    else:
        candidates.append((project / path).resolve())
        cwd_try = (Path.cwd() / path).resolve()
        if cwd_try not in candidates:
            candidates.append(cwd_try)

    existing = [c for c in candidates if c.is_file()]
    if not existing:
        raise DockingPrepareError(f"File not found: {path}")
    candidate = existing[0]
    for root in allowed_roots:
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    allowed = ", ".join(str(r) for r in allowed_roots)
    raise DockingPrepareError(
        f"Path escapes allowed directories ({allowed}): {candidate}"
    )


def _detect_name_column(columns: list[str], smiles_col: str) -> str | None:
    for name in ("name", "Name", "id", "ID", "ligand", "title"):
        if name in columns and name != smiles_col:
            return name
    return None


def _slug(text: str, *, max_len: int = 24) -> str:
    cleaned = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        elif char.isspace():
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug[:max_len]


def _smiles_to_sdf(smiles: str, dest: Path) -> None:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise DockingPrepareError(
            "RDKit is required to embed SMILES ligands in 3D"
        ) from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DockingPrepareError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if status != 0:
        # Fall back to a 2D-coords molecule; conversion tools may still accept it.
        AllChem.Compute2DCoords(mol)
    else:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:  # noqa: BLE001 — keep the embedded conformer
            pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(dest))
    writer.write(mol)
    writer.close()


def _sdf_to_pdbqt_meeko(sdf_path: Path, dest: Path) -> None:
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem
    except ImportError as exc:
        raise DockingPrepareError("Meeko/RDKit import failed") from exc
    mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
    if mol is None:
        raise DockingPrepareError(f"Meeko could not read SDF: {sdf_path}")
    prep = MoleculePreparation()
    prep.prepare(mol)
    pdbqt_string, is_ok, error = PDBQTWriterLegacy.write_string(prep.setup)
    if not is_ok:
        raise DockingPrepareError(f"Meeko PDBQT write failed: {error}")
    dest.write_text(pdbqt_string, encoding="utf-8")


def _run_obabel(command: list[str]) -> None:
    if not command or Path(command[0]).name not in {"obabel", "obabel3"}:
        raise DockingPrepareError("refusing to run a non-obabel converter")
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DockingPrepareError(f"obabel failed: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise DockingPrepareError(
            f"obabel exited {proc.returncode}" + (f": {detail}" if detail else "")
        )
