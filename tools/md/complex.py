"""Predefined protein–ligand GROMACS prep (pdb2gmx + ACPYPE + solvate/ions).

Writes ``<project>/input/md/system.gro`` and ``system.top``. Does not invent
RMSD. Missing acpype / gmx / ligand parameters is a loud failure.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.md.commands import (
    ACPYPE_TIMEOUT_SECONDS,
    ALLOWED_ACPYPE_BINARIES,
    ALLOWED_GMX_SUBCOMMANDS,
    ALLOWED_OBABEL_BINARIES,
    GENION_STDIN,
    GROMPP_TIMEOUT_SECONDS,
    PREP_TIMEOUT_SECONDS,
    build_acpype_command,
    build_editconf_command,
    build_genion_command,
    build_grompp_command,
    build_obabel_sdf_command,
    build_pdb2gmx_command,
    build_solvate_command,
)
from tools.md.environment import ALLOWED_GMX_BINARIES
from tools.md.prepare import (
    INPUT_DIRNAME,
    MdPrepareError,
    resolve_structure_path,
)

WORK_SUBDIR = ("output", "md", "prep")


class ComplexPrepError(ValueError):
    """Protein–ligand parameterization or solvation failed."""


def resolve_protein_pdb(project_dir: str | Path, protein: str | Path) -> Path:
    """Protein PDB must already exist under ``<project>/input/``."""
    try:
        path = resolve_structure_path(project_dir, protein)
    except MdPrepareError as exc:
        raise ComplexPrepError(str(exc)) from exc
    if path.suffix.lower() != ".pdb":
        raise ComplexPrepError(f"Protein for pdb2gmx must be .pdb; got {path.suffix}")
    return path


def resolve_ligand_pose(project_dir: str | Path, ligand: str | Path) -> Path:
    """Ligand pose must already exist under ``input/`` or ``output/``."""
    project = Path(project_dir).expanduser().resolve()
    input_dir = (project / INPUT_DIRNAME).resolve()
    output_dir = (project / "output").resolve()
    path = Path(str(ligand).strip()).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path.resolve())
    else:
        candidates.append((project / path).resolve())
    existing = [c for c in candidates if c.is_file()]
    if not existing:
        raise ComplexPrepError(f"Ligand pose not found: {path}")
    candidate = existing[0]
    for root in (input_dir, output_dir):
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise ComplexPrepError(
        f"Ligand pose escapes project input/output: {candidate}"
    )


def ligand_formal_charge(sdf_path: Path) -> int:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ComplexPrepError("RDKit is required to read ligand net charge") from exc
    mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
    if mol is None:
        raise ComplexPrepError(f"RDKit could not read ligand SDF: {sdf_path}")
    charge = int(Chem.GetFormalCharge(mol))
    if abs(charge) > 4:
        raise ComplexPrepError(f"ligand formal charge {charge} is outside ±4")
    return charge


def extract_first_model(text: str) -> str:
    """Keep MODEL 1 of a Vina/GNINA multi-model PDBQT/PDB, otherwise the whole file."""
    if "MODEL" not in text:
        return text
    chunks: list[str] = []
    keep = False
    seen = False
    for line in text.splitlines(keepends=True):
        if line.startswith("MODEL"):
            if seen:
                break
            seen = True
            keep = True
            continue
        if line.startswith("ENDMDL"):
            if keep:
                break
            continue
        if keep or not seen:
            chunks.append(line)
    return "".join(chunks) if chunks else text


def protein_atom_pdb(src: Path, dest: Path) -> Path:
    """Write ATOM/TER records only (no HETATM / waters / co-crystal ligand)."""
    lines: list[str] = []
    for raw in src.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("ATOM") or raw.startswith("TER"):
            lines.append(raw)
    if not any(line.startswith("ATOM") for line in lines):
        raise ComplexPrepError(f"No ATOM records in protein PDB: {src}")
    if not lines[-1].startswith("END"):
        lines.append("END")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def merge_coordinates(protein_gro: Path, lig_gro: Path, dest: Path) -> Path:
    def read_gro(path: Path) -> tuple[int, list[str], str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 3:
            raise ComplexPrepError(f"GRO too short: {path}")
        natoms = int(lines[1].strip())
        atoms = lines[2 : 2 + natoms]
        if len(atoms) != natoms:
            raise ComplexPrepError(f"GRO atom count mismatch: {path}")
        box = lines[2 + natoms]
        return natoms, atoms, box

    n_prot, prot_atoms, box = read_gro(protein_gro)
    n_lig, lig_atoms, _ = read_gro(lig_gro)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        fh.write("Protein-ligand complex\n")
        fh.write(f"{n_prot + n_lig}\n")
        for atom in prot_atoms:
            fh.write(atom + "\n")
        for atom in lig_atoms:
            fh.write(atom + "\n")
        fh.write(box + "\n")
    return dest


def merge_topologies(protein_top: Path, lig_itp: Path, dest: Path) -> Path:
    top_text = protein_top.read_text(encoding="utf-8")
    raw_itp = lig_itp.read_text(encoding="utf-8").replace(
        " ligand           3", "ligand           3"
    )
    atomtypes_block = ""
    mol_block = raw_itp
    if "[ atomtypes ]" in raw_itp:
        parts = raw_itp.split("[ moleculetype ]", 1)
        atomtypes_block = parts[0].strip() + "\n\n"
        if len(parts) > 1:
            mol_block = "[ moleculetype ]" + parts[1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    (dest.parent / "ligand_atomtypes.itp").write_text(atomtypes_block, encoding="utf-8")
    (dest.parent / "ligand.itp").write_text(mol_block, encoding="utf-8")

    for stale in ('#include "ligand.itp"\n', '#include "ligand_atomtypes.itp"\n'):
        top_text = top_text.replace(stale, "")
    if '#include "ligand_atomtypes.itp"' not in top_text:
        top_text = top_text.replace(
            '#include "amber99sb-ildn.ff/forcefield.itp"',
            '#include "amber99sb-ildn.ff/forcefield.itp"\n#include "ligand_atomtypes.itp"\n',
            1,
        )
    if '#include "ligand.itp"' not in top_text:
        inserted = False
        for needle in (
            '#include "amber99sb-ildn.ff/tip3p.itp"',
            '#include "amber99sb-ildn.ff/spce.itp"',
        ):
            if needle in top_text:
                top_text = top_text.replace(
                    needle, '#include "ligand.itp"\n' + needle, 1
                )
                inserted = True
                break
        if not inserted:
            raise ComplexPrepError(
                "Could not insert ligand.itp include into protein topology"
            )
    if "ligand               1" not in top_text:
        replaced = False
        for needle in ("Protein_chain_A     1", "Protein             1"):
            if needle in top_text:
                top_text = top_text.replace(
                    needle, needle + "\nligand               1", 1
                )
                replaced = True
                break
        if not replaced:
            raise ComplexPrepError(
                "Could not append ligand molecule count to protein topology"
            )
    dest.write_text(top_text, encoding="utf-8")
    return dest


def prepare_complex(
    project_dir: str | Path,
    *,
    protein: str | Path,
    ligand: str | Path,
    gmx_path: str,
    acpype_path: str,
    obabel_path: str | None,
    run_fn,
) -> dict[str, Any]:
    """pdb2gmx protein + ACPYPE ligand + box/solvate/genion → input/md/system.*"""
    project = Path(project_dir).expanduser().resolve()
    protein_src = resolve_protein_pdb(project, protein)
    ligand_src = resolve_ligand_pose(project, ligand)
    work = project.joinpath(*WORK_SUBDIR)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    logs = work / "logs"
    logs.mkdir()
    md_input = project / INPUT_DIRNAME / "md"
    md_input.mkdir(parents=True, exist_ok=True)

    protein_pdb = protein_atom_pdb(protein_src, work / "protein_clean.pdb")
    ligand_in = work / f"ligand_pose{ligand_src.suffix.lower()}"
    ligand_in.write_text(
        extract_first_model(ligand_src.read_text(encoding="utf-8", errors="replace")),
        encoding="utf-8",
    )
    ligand_sdf = work / "ligand.sdf"
    if ligand_in.suffix.lower() in {".sdf", ".mol"}:
        shutil.copy2(ligand_in, ligand_sdf)
    else:
        if not obabel_path:
            raise ComplexPrepError(
                "obabel is required to convert a docked PDBQT/PDB pose to SDF "
                "for ACPYPE"
            )
        obabel_cmd = build_obabel_sdf_command(
            executable=obabel_path, source=ligand_in, dest=ligand_sdf
        )
        _run_obabel(obabel_cmd, log_path=logs / "obabel.log")
        if not ligand_sdf.is_file():
            raise ComplexPrepError("obabel did not write ligand.sdf")

    charge = ligand_formal_charge(ligand_sdf)
    acpype_cmd = build_acpype_command(
        executable=acpype_path, ligand=ligand_sdf, charge=charge
    )
    proc = run_fn(
        acpype_cmd,
        cwd=work,
        log_path=logs / "acpype.log",
        timeout=ACPYPE_TIMEOUT_SECONDS,
        stdin_text=None,
        allowed_binaries=ALLOWED_ACPYPE_BINARIES,
        allowed_subcommands=None,
    )
    if proc.returncode != 0:
        raise ComplexPrepError(
            f"acpype exited {proc.returncode}; see {logs / 'acpype.log'}"
        )
    lig_gro = work / "ligand.acpype" / "ligand_GMX.gro"
    lig_itp = work / "ligand.acpype" / "ligand_GMX.itp"
    if not lig_gro.is_file() or not lig_itp.is_file():
        raise ComplexPrepError(
            f"ACPYPE did not write GMX ligand files under {work / 'ligand.acpype'}"
        )

    protein_gro = work / "protein.gro"
    protein_top = work / "topol_protein.top"
    posre = work / "posre.itp"
    pdb2gmx = build_pdb2gmx_command(
        executable=gmx_path,
        pdb=protein_pdb,
        gro=protein_gro,
        top=protein_top,
        posre=posre,
    )
    proc = run_fn(
        pdb2gmx,
        cwd=work,
        log_path=logs / "pdb2gmx.log",
        timeout=PREP_TIMEOUT_SECONDS,
        stdin_text=None,
    )
    if proc.returncode != 0 or not protein_gro.is_file() or not protein_top.is_file():
        raise ComplexPrepError(f"gmx pdb2gmx failed; see {logs / 'pdb2gmx.log'}")

    complex_gro = merge_coordinates(protein_gro, lig_gro, work / "complex.gro")
    topol = merge_topologies(protein_top, lig_itp, work / "topol.top")

    editconf = build_editconf_command(
        executable=gmx_path, gro=complex_gro, out=work / "box.gro"
    )
    proc = run_fn(
        editconf,
        cwd=work,
        log_path=logs / "editconf.log",
        timeout=PREP_TIMEOUT_SECONDS,
        stdin_text=None,
    )
    if proc.returncode != 0 or not (work / "box.gro").is_file():
        raise ComplexPrepError(f"gmx editconf failed; see {logs / 'editconf.log'}")

    solvate = build_solvate_command(
        executable=gmx_path,
        gro=work / "box.gro",
        out=work / "solv.gro",
        top=topol,
    )
    proc = run_fn(
        solvate,
        cwd=work,
        log_path=logs / "solvate.log",
        timeout=PREP_TIMEOUT_SECONDS,
        stdin_text=None,
    )
    if proc.returncode != 0 or not (work / "solv.gro").is_file():
        raise ComplexPrepError(f"gmx solvate failed; see {logs / 'solvate.log'}")

    ions_mdp = _ions_mdp(work)
    ions_tpr = work / "ions.tpr"
    grompp = build_grompp_command(
        executable=gmx_path,
        mdp=ions_mdp,
        gro=work / "solv.gro",
        top=topol,
        tpr=ions_tpr,
        maxwarn=2,
    )
    proc = run_fn(
        grompp,
        cwd=work,
        log_path=logs / "ions_grompp.log",
        timeout=GROMPP_TIMEOUT_SECONDS,
        stdin_text=None,
    )
    if proc.returncode != 0 or not ions_tpr.is_file():
        raise ComplexPrepError(
            f"gmx grompp (ions) failed; see {logs / 'ions_grompp.log'}"
        )

    genion = build_genion_command(
        executable=gmx_path,
        tpr=ions_tpr,
        gro=work / "solv_ions.gro",
        top=topol,
    )
    proc = run_fn(
        genion,
        cwd=work,
        log_path=logs / "genion.log",
        timeout=PREP_TIMEOUT_SECONDS,
        stdin_text=GENION_STDIN,
    )
    if proc.returncode != 0 or not (work / "solv_ions.gro").is_file():
        raise ComplexPrepError(f"gmx genion failed; see {logs / 'genion.log'}")

    dest_gro = md_input / "system.gro"
    dest_top = md_input / "system.top"
    shutil.copy2(work / "solv_ions.gro", dest_gro)
    shutil.copy2(topol, dest_top)
    for name in ("ligand.itp", "ligand_atomtypes.itp", "posre.itp"):
        src = work / name
        if src.is_file():
            shutil.copy2(src, md_input / name)

    return {
        "ok": True,
        "work_dir": str(work),
        "structure": str(dest_gro),
        "topology": str(dest_top),
        "ligand_charge": charge,
        "protein": str(protein_src),
        "ligand": str(ligand_src),
    }


def _ions_mdp(work: Path) -> Path:
    from tools.md.mdp import TEMPLATES_DIR

    src = (TEMPLATES_DIR / "ions.mdp").resolve()
    dest = work / "ions.mdp"
    if not src.is_file():
        raise ComplexPrepError(f"Human-written ions.mdp missing: {src}")
    shutil.copy2(src, dest)
    return dest


def _run_obabel(command: list[str], *, log_path: Path) -> None:
    if not command or Path(command[0]).name not in ALLOWED_OBABEL_BINARIES:
        raise ComplexPrepError("refusing to run a non-obabel converter")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.run(
            command,
            check=False,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=PREP_TIMEOUT_SECONDS,
            shell=False,
        )
    if proc.returncode != 0:
        raise ComplexPrepError(f"obabel exited {proc.returncode}; see {log_path}")


# Re-export allowlists so tests can assert prep never shells out.
ALLOWED_PREP_GMX = ALLOWED_GMX_BINARIES
ALLOWED_PREP_SUBCOMMANDS = ALLOWED_GMX_SUBCOMMANDS
