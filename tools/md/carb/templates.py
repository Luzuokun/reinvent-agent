"""Load and validate git-tracked oligomer templates. No LLM topology."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from tools.md.carb.constants import (
    ALT_TEMPLATE_ID,
    AMYLOSE_CHARGE,
    AMYLOSE_N_RESIDUES,
    AMYLOSE_RESNAME,
    COM_DISTANCE_MAX_NM,
    COM_DISTANCE_MIN_NM,
    DEFAULT_TEMPLATE_ID,
    FUCOIDAN_ALT_CHARGE,
    FUCOIDAN_CHARGE,
    FUCOIDAN_N_RESIDUES,
    FUCOIDAN_RESNAME,
    TEMPLATES_DIR,
)

_ITP_ATOM = re.compile(
    r"^\s*(\d+)\s+\S+\s+(\d+)\s+(\S+)\s+\S+\s+\d+\s+(-?\d+(?:\.\d+)?)"
)


class CarbTemplateError(ValueError):
    """Missing template file or charge/residue mismatch."""


def load_manifest(templates_dir: Path | None = None) -> dict[str, Any]:
    root = Path(templates_dir) if templates_dir is not None else TEMPLATES_DIR
    path = root / "manifest.json"
    if not path.is_file():
        raise CarbTemplateError(f"template manifest missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CarbTemplateError("template manifest is not an object")
    return data


def template_spec(template_id: str = DEFAULT_TEMPLATE_ID) -> dict[str, Any]:
    name = (template_id or DEFAULT_TEMPLATE_ID).strip().lower().replace("-", "_")
    if name in {"amylose26_fucoidan12", "default", "main"}:
        return {
            "template_id": DEFAULT_TEMPLATE_ID,
            "amylose": "amylose_26",
            "fucoidan": "fucoidan_12",
            "fucoidan_molname": "Fucoidan12",
            "expected_fucoidan_charge": FUCOIDAN_CHARGE,
            "alt_sulfation": False,
        }
    if name in {
        "amylose26_fucoidan12_alt",
        "alt",
        "alt_sulfation",
        "fucoidan_12_alt_sulfation",
    }:
        return {
            "template_id": ALT_TEMPLATE_ID,
            "amylose": "amylose_26",
            "fucoidan": "fucoidan_12_alt_sulfation",
            "fucoidan_molname": "Fucoidan12Alt",
            "expected_fucoidan_charge": FUCOIDAN_ALT_CHARGE,
            "alt_sulfation": True,
        }
    raise CarbTemplateError(
        f"Unknown carb template {template_id!r}; "
        f"allowed: {DEFAULT_TEMPLATE_ID}, {ALT_TEMPLATE_ID}"
    )


def parse_gro_residues(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 3:
        raise CarbTemplateError(f"GRO too short: {path}")
    atoms = lines[2:-1]
    residues: dict[int, str] = {}
    for line in atoms:
        if len(line) < 15:
            continue
        try:
            resid = int(line[0:5])
        except ValueError:
            continue
        resname = line[5:10].strip()
        residues[resid] = resname
    names = list(residues.values())
    return {
        "n_atoms": len(atoms),
        "n_residues": len(residues),
        "residue_names": sorted(set(names)),
        "residues": residues,
    }


def parse_pdb_residues(path: Path) -> dict[str, Any]:
    residues: dict[int, str] = {}
    n_atoms = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        n_atoms += 1
        resname = line[17:21].strip()
        try:
            resid = int(line[22:26])
        except ValueError:
            continue
        residues[resid] = resname
    return {
        "n_atoms": n_atoms,
        "n_residues": len(residues),
        "residue_names": sorted(set(residues.values())),
        "residues": residues,
    }


def parse_itp_charges(path: Path) -> dict[str, Any]:
    in_atoms = False
    charges: list[float] = []
    resids: set[int] = set()
    resnames: set[str] = set()
    molname = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            in_atoms = line.lower().replace(" ", "") == "[atoms]"
            continue
        if molname is None and not line.startswith("[") and not charges:
            parts = line.split()
            if len(parts) >= 1 and parts[0][0].isalpha() and "nrexcl" not in line:
                # moleculetype name line: Name nrexcl
                if len(parts) == 2 and parts[1].isdigit():
                    molname = parts[0]
        if not in_atoms:
            continue
        match = _ITP_ATOM.match(line)
        if not match:
            continue
        resids.add(int(match.group(2)))
        resnames.add(match.group(3))
        charges.append(float(match.group(4)))
    total = round(sum(charges), 6)
    return {
        "molname": molname,
        "n_atoms": len(charges),
        "n_residues": len(resids),
        "residue_names": sorted(resnames),
        "net_charge": total,
    }


def gro_com(path: Path, resname: str | None = None) -> tuple[float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines[2:-1]:
        if len(line) < 44:
            continue
        name = line[5:10].strip()
        if resname is not None and name != resname:
            continue
        xs.append(float(line[20:28]))
        ys.append(float(line[28:36]))
        zs.append(float(line[36:44]))
    n = len(xs) or 1
    return (sum(xs) / n, sum(ys) / n, sum(zs) / n)


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def validate_bundled_templates(templates_dir: Path | None = None) -> dict[str, Any]:
    """Assert residue counts and charges on the git-tracked fixtures."""
    root = Path(templates_dir) if templates_dir is not None else TEMPLATES_DIR
    manifest = load_manifest(root)
    errors: list[str] = []
    details: dict[str, Any] = {}

    amy = _check_oligomer(
        root,
        manifest["amylose_26"],
        expected_n=AMYLOSE_N_RESIDUES,
        expected_resname=AMYLOSE_RESNAME,
        expected_charge=AMYLOSE_CHARGE,
    )
    details["amylose_26"] = amy
    errors.extend(amy.get("errors") or [])

    fuc = _check_oligomer(
        root,
        manifest["fucoidan_12"],
        expected_n=FUCOIDAN_N_RESIDUES,
        expected_resname=FUCOIDAN_RESNAME,
        expected_charge=FUCOIDAN_CHARGE,
    )
    details["fucoidan_12"] = fuc
    errors.extend(fuc.get("errors") or [])

    alt = _check_oligomer(
        root,
        manifest["fucoidan_12_alt_sulfation"],
        expected_n=FUCOIDAN_N_RESIDUES,
        expected_resname=FUCOIDAN_RESNAME,
        expected_charge=FUCOIDAN_ALT_CHARGE,
    )
    details["fucoidan_12_alt_sulfation"] = alt
    errors.extend(alt.get("errors") or [])

    amy_gro = root / manifest["amylose_26"]["files"]["gro"]
    fuc_gro = root / manifest["fucoidan_12"]["files"]["gro"]
    separation = _dist(gro_com(amy_gro), gro_com(fuc_gro))
    details["com_separation_nm"] = round(separation, 4)
    if not (COM_DISTANCE_MIN_NM <= separation <= COM_DISTANCE_MAX_NM):
        errors.append(
            f"template COM separation {separation:.3f} nm is outside "
            f"{COM_DISTANCE_MIN_NM}–{COM_DISTANCE_MAX_NM} nm"
        )

    if errors:
        raise CarbTemplateError("; ".join(errors))
    return {
        "ok": True,
        "manifest": str(root / "manifest.json"),
        "details": details,
        "disclaimer": manifest.get("disclaimer"),
    }


def _check_oligomer(
    root: Path,
    spec: dict[str, Any],
    *,
    expected_n: int,
    expected_resname: str,
    expected_charge: float,
) -> dict[str, Any]:
    files = spec.get("files") or {}
    errors: list[str] = []
    parsed: dict[str, Any] = {}
    for kind in ("pdb", "gro", "itp"):
        path = root / files[kind]
        if not path.is_file():
            errors.append(f"missing {kind}: {path}")
            continue
        if kind == "pdb":
            parsed["pdb"] = parse_pdb_residues(path)
        elif kind == "gro":
            parsed["gro"] = parse_gro_residues(path)
        else:
            parsed["itp"] = parse_itp_charges(path)
    for src in ("pdb", "gro", "itp"):
        block = parsed.get(src) or {}
        n = block.get("n_residues")
        if n != expected_n:
            errors.append(f"{spec.get('files', {}).get(src)} residues {n} != {expected_n}")
        names = block.get("residue_names") or []
        if names and expected_resname not in names:
            errors.append(f"{src} residue names {names} missing {expected_resname}")
    charge = (parsed.get("itp") or {}).get("net_charge")
    if charge is not None and abs(float(charge) - float(expected_charge)) > 1e-4:
        errors.append(f"itp net charge {charge} != {expected_charge}")
    return {"ok": not errors, "errors": errors, "parsed": parsed}


def read_gro_atoms(path: Path) -> tuple[str, list[str], tuple[float, float, float]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    title = lines[0] if lines else "carb"
    atom_lines = lines[2:-1]
    box_vals = [float(x) for x in lines[-1].split()[:3]]
    box = (box_vals[0], box_vals[1], box_vals[2])
    return title, atom_lines, box


def shift_gro_lines(atom_lines: list[str], delta: tuple[float, float, float]) -> list[str]:
    out: list[str] = []
    dx, dy, dz = delta
    for line in atom_lines:
        x = float(line[20:28]) + dx
        y = float(line[28:36]) + dy
        z = float(line[36:44]) + dz
        out.append(f"{line[:20]}{x:8.3f}{y:8.3f}{z:8.3f}{line[44:]}")
    return out


def write_gro(
    path: Path,
    title: str,
    atom_lines: list[str],
    box: tuple[float, float, float],
) -> None:
    body = [title, f"{len(atom_lines):5d}", *atom_lines, f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}"]
    Path(path).write_text("\n".join(body) + "\n", encoding="utf-8")


def concatenate_complex(
    amylose_gro: Path,
    fucoidan_gro: Path,
    dest: Path,
    *,
    target_com_nm: float = 1.2,
) -> dict[str, Any]:
    """Place both oligomers at 0.8–1.5 nm COM distance and write one GRO."""
    _title_a, amy_lines, box_a = read_gro_atoms(amylose_gro)
    _title_b, fuc_lines, box_b = read_gro_atoms(fucoidan_gro)
    amy_com = gro_com(amylose_gro)
    fuc_com = gro_com(fucoidan_gro)
    current = _dist(amy_com, fuc_com)
    if current < COM_DISTANCE_MIN_NM or current > COM_DISTANCE_MAX_NM:
        # Shift fucoidan along +x to the target separation.
        dx = target_com_nm - (fuc_com[0] - amy_com[0])
        fuc_lines = shift_gro_lines(fuc_lines, (dx, 0.0, 0.0))
    # Renumber atom serials (columns 15-20).
    combined: list[str] = []
    serial = 1
    for line in amy_lines + fuc_lines:
        combined.append(f"{line[:15]}{serial:5d}{line[20:]}")
        serial += 1
        if serial > 99999:
            serial = 1
    box = (
        max(box_a[0], box_b[0], 8.0),
        max(box_a[1], box_b[1], 8.0),
        max(box_a[2], box_b[2], 8.0),
    )
    write_gro(dest, "Amylose-26 Fucoidan-12 protocol complex", combined, box)
    new_sep = _dist(gro_com(dest, AMYLOSE_RESNAME), gro_com(dest, FUCOIDAN_RESNAME))
    if not (COM_DISTANCE_MIN_NM <= new_sep <= COM_DISTANCE_MAX_NM):
        raise CarbTemplateError(
            f"assembled COM separation {new_sep:.3f} nm outside "
            f"{COM_DISTANCE_MIN_NM}–{COM_DISTANCE_MAX_NM} nm"
        )
    return {
        "structure": str(dest),
        "n_atoms": len(combined),
        "com_separation_nm": round(new_sep, 4),
    }


def write_index_ndx(gro: Path, dest: Path) -> Path:
    amy: list[int] = []
    fuc: list[int] = []
    system: list[int] = []
    _, atom_lines, _box = read_gro_atoms(gro)
    for i, line in enumerate(atom_lines, start=1):
        resname = line[5:10].strip()
        system.append(i)
        if resname == AMYLOSE_RESNAME:
            amy.append(i)
        elif resname == FUCOIDAN_RESNAME:
            fuc.append(i)

    def _block(name: str, ids: list[int]) -> str:
        rows = []
        for start in range(0, len(ids), 15):
            chunk = ids[start : start + 15]
            rows.append(" ".join(f"{n:5d}" for n in chunk))
        return f"[ {name} ]\n" + "\n".join(rows) + "\n"

    text = _block("System", system) + _block("Amylose", amy) + _block("Fucoidan", fuc)
    Path(dest).write_text(text, encoding="utf-8")
    return dest
