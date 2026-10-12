#!/usr/bin/env python3
"""Build git-tracked Amylose-26 / Fucoidan-12 protocol templates.

These files are **protocol templates** for automation smoke tests and
scientific-workflow wiring, not crystallographic depositions. Residue
counts, glycosidic connectivity, and net charges are internally consistent
and asserted by tests. Geometry is a chemically reasonable 4C1/1C4 chair
placed on a left-handed helix (amylose) or an extended 1→3/1→4 chain
(fucoidan).

Run from the repo root::

    python protocols/amylose_fucoidan/scripts/build_templates.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"

# CHARMM36 atom types (MacKerell jul2022 GROMACS port).
T_C1 = "CC3162"
T_C2 = "CC3161"
T_C5 = "CC3163"
T_C6_GLC = "CC321"
T_C6_FUC = "CC331"
T_O5 = "OC3C61"
T_OH = "OC311"
T_HO = "HCP1"
T_HC = "HCA1"
T_H6G = "HCA2"
T_H6F = "HCA3"
T_OS = "OG303"  # sulfate ester oxygen
T_S = "SG3O1"  # sulfate -1 sulfur
T_OSU = "OG2P1"  # sulfate terminal =O

MASS = {
    T_C1: 12.011,
    T_C2: 12.011,
    T_C5: 12.011,
    T_C6_GLC: 12.011,
    T_C6_FUC: 12.011,
    T_O5: 15.999,
    T_OH: 15.999,
    T_HO: 1.008,
    T_HC: 1.008,
    T_H6G: 1.008,
    T_H6F: 1.008,
    T_OS: 15.999,
    T_S: 32.06,
    T_OSU: 15.999,
}


def _vadd(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vscale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _rotz(
    p: tuple[float, float, float], angle: float
) -> tuple[float, float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])


def _translate(
    p: tuple[float, float, float], t: tuple[float, float, float]
) -> tuple[float, float, float]:
    return _vadd(p, t)


# Approximate 4C1 α-D-Glcp heavy-atom frame (nm), origin near ring centroid.
# Axial O1 (alpha). Equatorial O2/O3/O4/C6.
_RING = {
    "C1": (0.070, -0.020, -0.060),
    "C2": (0.140, 0.110, 0.010),
    "C3": (0.030, 0.200, 0.070),
    "C4": (-0.100, 0.120, 0.060),
    "C5": (-0.120, -0.010, -0.010),
    "O5": (-0.020, -0.090, -0.070),
    "O1": (0.090, -0.120, -0.180),  # alpha axial
    "O2": (0.270, 0.070, 0.040),
    "O3": (0.080, 0.330, 0.020),
    "O4": (-0.210, 0.200, 0.120),
    "C6": (-0.250, -0.090, 0.050),
    "O6": (-0.360, -0.010, 0.100),
}


def _h_on(heavy: tuple[float, float, float], direction: tuple[float, float, float], dist: float = 0.109) -> tuple[float, float, float]:
    n = math.sqrt(sum(x * x for x in direction)) or 1.0
    return _vadd(heavy, _vscale(direction, dist / n))


def _sulfate_on(oxygen: tuple[float, float, float], outward: tuple[float, float, float]) -> dict[str, tuple[float, float, float]]:
    n = math.sqrt(sum(x * x for x in outward)) or 1.0
    u = _vscale(outward, 1.0 / n)
    s_pos = _vadd(oxygen, _vscale(u, 0.160))
    # tetrahedral terminals around S
    def term(ax: tuple[float, float, float]) -> tuple[float, float, float]:
        nn = math.sqrt(sum(x * x for x in ax)) or 1.0
        return _vadd(s_pos, _vscale(ax, 0.145 / nn))

    perp = (-u[1], u[0], 0.2)
    return {
        "S": s_pos,
        "O1S": term(_vadd(u, (0.4, 0.7, 0.2))),
        "O2S": term(_vadd(u, (-0.8, 0.1, 0.3))),
        "O3S": term(_vadd(u, (0.1, -0.8, 0.4))),
        "_perp": perp,
    }


class Atom:
    __slots__ = ("name", "atype", "charge", "xyz", "resname", "resid", "element")

    def __init__(
        self,
        name: str,
        atype: str,
        charge: float,
        xyz: tuple[float, float, float],
        resname: str,
        resid: int,
        element: str,
    ) -> None:
        self.name = name
        self.atype = atype
        self.charge = charge
        self.xyz = xyz
        self.resname = resname
        self.resid = resid
        self.element = element


def _place_residue(
    local: dict[str, tuple[float, float, float]],
    *,
    angle: float,
    origin: tuple[float, float, float],
) -> dict[str, tuple[float, float, float]]:
    out = {}
    for name, xyz in local.items():
        out[name] = _translate(_rotz(xyz, angle), origin)
    return out


def build_amylose(n: int = 26) -> tuple[list[Atom], list[tuple[int, int]], list[str]]:
    """α-D-Glc 26-mer, α-(1→4), left-handed V-type helix, net charge 0."""
    atoms: list[Atom] = []
    bonds: list[tuple[int, int]] = []
    linkages: list[str] = []
    index: dict[tuple[int, str], int] = {}

    rise = 0.136
    radius = 0.38
    dtheta = -math.tau / 6.0  # left-handed 6-fold

    for i in range(1, n + 1):
        theta = (i - 1) * dtheta
        origin = (radius * math.cos(theta), radius * math.sin(theta), (i - 1) * rise)
        pos = _place_residue(_RING, angle=theta, origin=origin)
        reducing = i == n
        nonred = i == 1
        is_acceptor = i > 1  # O4 is glycosidic

        res_atoms: list[tuple[str, str, float, tuple[float, float, float], str]] = [
            # Donor residues drop O1/HO1 (sum -0.23); C1 is lowered to keep net 0.
            ("C1", T_C1, 0.11 if not reducing else 0.34, pos["C1"], "C"),
            ("H1", T_HC, 0.09, _h_on(pos["C1"], (0.0, 0.5, 1.0)), "H"),
            ("C5", T_C5, 0.11, pos["C5"], "C"),
            ("H5", T_HC, 0.09, _h_on(pos["C5"], (0.0, -1.0, 0.4)), "H"),
            ("O5", T_O5, -0.40, pos["O5"], "O"),
            ("C2", T_C2, 0.14, pos["C2"], "C"),
            ("H2", T_HC, 0.09, _h_on(pos["C2"], (0.3, 0.2, 1.0)), "H"),
            ("O2", T_OH, -0.65, pos["O2"], "O"),
            ("HO2", T_HO, 0.42, _h_on(pos["O2"], (1.0, 0.0, 0.2)), "H"),
            ("C3", T_C2, 0.14, pos["C3"], "C"),
            ("H3", T_HC, 0.09, _h_on(pos["C3"], (0.2, 0.4, 1.0)), "H"),
            ("O3", T_OH, -0.65, pos["O3"], "O"),
            ("HO3", T_HO, 0.42, _h_on(pos["O3"], (0.2, 1.0, 0.2)), "H"),
            ("C4", T_C2, 0.14, pos["C4"], "C"),
            ("H4", T_HC, 0.09, _h_on(pos["C4"], (-0.3, 0.2, 1.0)), "H"),
            ("O4", T_OH, -0.23 if is_acceptor else -0.65, pos["O4"], "O"),
            ("C6", T_C6_GLC, 0.05, pos["C6"], "C"),
            ("H61", T_H6G, 0.09, _h_on(pos["C6"], (-1.0, 0.2, 0.4)), "H"),
            ("H62", T_H6G, 0.09, _h_on(pos["C6"], (-0.4, -1.0, 0.2)), "H"),
            ("O6", T_OH, -0.65, pos["O6"], "O"),
            ("HO6", T_HO, 0.42, _h_on(pos["O6"], (-1.0, 0.3, 0.5)), "H"),
        ]
        if reducing:
            res_atoms[0] = ("C1", T_C1, 0.34, pos["C1"], "C")
            res_atoms.insert(2, ("O1", T_OH, -0.65, pos["O1"], "O"))
            res_atoms.insert(3, ("HO1", T_HO, 0.42, _h_on(pos["O1"], (0.2, -1.0, -0.5)), "H"))
        if nonred:
            res_atoms.append(("HO4", T_HO, 0.42, _h_on(pos["O4"], (-1.0, 0.4, 0.3)), "H"))

        start = len(atoms)
        for name, atype, charge, xyz, elem in res_atoms:
            atoms.append(Atom(name, atype, charge, xyz, "AGLC", i, elem))
            index[(i, name)] = len(atoms)  # 1-based

        # Intra-residue bonds (by name).
        pairs = [
            ("C1", "H1"),
            ("C1", "O5"),
            ("C1", "C2"),
            ("C2", "H2"),
            ("C2", "O2"),
            ("O2", "HO2"),
            ("C2", "C3"),
            ("C3", "H3"),
            ("C3", "O3"),
            ("O3", "HO3"),
            ("C3", "C4"),
            ("C4", "H4"),
            ("C4", "O4"),
            ("C4", "C5"),
            ("C5", "H5"),
            ("C5", "C6"),
            ("C6", "H61"),
            ("C6", "H62"),
            ("C6", "O6"),
            ("O6", "HO6"),
            ("C5", "O5"),
        ]
        if reducing:
            pairs.extend([("C1", "O1"), ("O1", "HO1")])
        if nonred:
            pairs.append(("O4", "HO4"))
        for a, b in pairs:
            bonds.append((index[(i, a)], index[(i, b)]))

        if i > 1:
            # α-(1→4): C1(i-1) — O4(i) — C4(i)
            bonds.append((index[(i - 1, "C1")], index[(i, "O4")]))
            linkages.append(f"AGLC{i-1}.C1 - AGLC{i}.O4  alpha-(1->4)")

        del start
    return atoms, bonds, linkages


def build_fucoidan(
    n: int = 12, *, alt_sulfation: bool = False
) -> tuple[list[Atom], list[tuple[int, int]], list[str], dict]:
    """α-L-Fuc 12-mer, alternating α-(1→3)/α-(1→4), C2 (and optional C4) sulfate."""
    atoms: list[Atom] = []
    bonds: list[tuple[int, int]] = []
    linkages: list[str] = []
    index: dict[tuple[int, str], int] = {}

    rise = 0.42
    radius = 0.28
    dtheta = -math.tau / 3.0

    # Shift so COM sits ~1.2 nm from an amylose helix along +x.
    xoff = 1.55

    sulfation = []
    for i in range(1, n + 1):
        sites = ["C2"]
        even = i % 2 == 0
        # Even residues are 1→3 acceptors (O3 glycosidic); C4 is free.
        if alt_sulfation and even:
            sites.append("C4")
        sulfation.append(sites)

    for i in range(1, n + 1):
        theta = (i - 1) * dtheta
        origin = (
            xoff + radius * math.cos(theta),
            radius * math.sin(theta),
            (i - 1) * rise,
        )
        # L-fucose 1C4 is approximated by reflecting the D-Glc 4C1 y axis.
        local = {k: (x, -y, z) for k, (x, y, z) in _RING.items()}
        # No O6; C6 is a methyl.
        pos = _place_residue(local, angle=theta, origin=origin)
        reducing = i == n
        linkage_from_prev = None if i == 1 else ("1-3" if (i - 1) % 2 == 1 else "1-4")
        o3_glycosidic = linkage_from_prev == "1-3"
        o4_glycosidic = linkage_from_prev == "1-4"
        sites = sulfation[i - 1]

        def c2_charge_block() -> list[tuple[str, str, float, tuple[float, float, float], str]]:
            block = [
                ("C2", T_C2, 0.14, pos["C2"], "C"),
                ("H2", T_HC, 0.09, _h_on(pos["C2"], (0.3, -0.2, 1.0)), "H"),
            ]
            if "C2" in sites:
                sul = _sulfate_on(pos["O2"], (1.0, 0.2, 0.1))
                block.extend(
                    [
                        ("O2", T_OS, -0.28, pos["O2"], "O"),
                        ("S2", T_S, 1.33, sul["S"], "S"),
                        ("O1S", T_OSU, -0.76, sul["O1S"], "O"),
                        ("O2S", T_OSU, -0.76, sul["O2S"], "O"),
                        ("O3S", T_OSU, -0.76, sul["O3S"], "O"),
                    ]
                )
            else:
                block.extend(
                    [
                        ("O2", T_OH, -0.65, pos["O2"], "O"),
                        ("HO2", T_HO, 0.42, _h_on(pos["O2"], (1.0, 0.0, 0.2)), "H"),
                    ]
                )
            return block

        def c4_charge_block() -> list[tuple[str, str, float, tuple[float, float, float], str]]:
            o4_q = -0.23 if o4_glycosidic else -0.65
            block = [
                ("C4", T_C2, 0.14, pos["C4"], "C"),
                ("H4", T_HC, 0.09, _h_on(pos["C4"], (-0.3, -0.2, 1.0)), "H"),
                ("O4", T_OH if "C4" not in sites else T_OS, o4_q if "C4" not in sites else -0.28, pos["O4"], "O"),
            ]
            if "C4" in sites:
                sul = _sulfate_on(pos["O4"], (-1.0, 0.3, 0.2))
                block.extend(
                    [
                        ("S4", T_S, 1.33, sul["S"], "S"),
                        ("O4S", T_OSU, -0.76, sul["O1S"], "O"),
                        ("O5S", T_OSU, -0.76, sul["O2S"], "O"),
                        ("O6S", T_OSU, -0.76, sul["O3S"], "O"),
                    ]
                )
            elif not o4_glycosidic:
                block.append(("HO4", T_HO, 0.42, _h_on(pos["O4"], (-1.0, 0.4, 0.3)), "H"))
            return block

        res_atoms: list[tuple[str, str, float, tuple[float, float, float], str]] = [
            # Donor residues drop O1/HO1 (sum -0.23); C1 is lowered to keep net 0.
            ("C1", T_C1, 0.11 if not reducing else 0.34, pos["C1"], "C"),
            ("H1", T_HC, 0.09, _h_on(pos["C1"], (0.0, 0.5, 1.0)), "H"),
        ]
        if reducing:
            res_atoms[0] = ("C1", T_C1, 0.34, pos["C1"], "C")
            res_atoms.extend(
                [
                    ("O1", T_OH, -0.65, pos["O1"], "O"),
                    ("HO1", T_HO, 0.42, _h_on(pos["O1"], (0.2, 1.0, -0.5)), "H"),
                ]
            )
        res_atoms.extend(
            [
                ("C5", T_C5, 0.11, pos["C5"], "C"),
                ("H5", T_HC, 0.09, _h_on(pos["C5"], (0.0, 1.0, 0.4)), "H"),
                ("O5", T_O5, -0.40, pos["O5"], "O"),
            ]
        )
        res_atoms.extend(c2_charge_block())
        o3_q = -0.23 if o3_glycosidic else -0.65
        res_atoms.extend(
            [
                ("C3", T_C2, 0.14, pos["C3"], "C"),
                ("H3", T_HC, 0.09, _h_on(pos["C3"], (0.2, -0.4, 1.0)), "H"),
                ("O3", T_OH, o3_q, pos["O3"], "O"),
            ]
        )
        if not o3_glycosidic:
            res_atoms.append(("HO3", T_HO, 0.42, _h_on(pos["O3"], (0.2, -1.0, 0.2)), "H"))
        res_atoms.extend(c4_charge_block())
        res_atoms.extend(
            [
                ("C6", T_C6_FUC, -0.27, pos["C6"], "C"),
                ("H61", T_H6F, 0.09, _h_on(pos["C6"], (-1.0, 0.2, 0.4)), "H"),
                ("H62", T_H6F, 0.09, _h_on(pos["C6"], (-0.4, 1.0, 0.2)), "H"),
                ("H63", T_H6F, 0.09, _h_on(pos["C6"], (-0.2, 0.2, -1.0)), "H"),
            ]
        )

        for name, atype, charge, xyz, elem in res_atoms:
            atoms.append(Atom(name, atype, charge, xyz, "AFUS", i, elem))
            index[(i, name)] = len(atoms)

        pairs = [
            ("C1", "H1"),
            ("C1", "O5"),
            ("C1", "C2"),
            ("C2", "H2"),
            ("C2", "O2"),
            ("C2", "C3"),
            ("C3", "H3"),
            ("C3", "O3"),
            ("C3", "C4"),
            ("C4", "H4"),
            ("C4", "O4"),
            ("C4", "C5"),
            ("C5", "H5"),
            ("C5", "C6"),
            ("C6", "H61"),
            ("C6", "H62"),
            ("C6", "H63"),
            ("C5", "O5"),
        ]
        if reducing:
            pairs.extend([("C1", "O1"), ("O1", "HO1")])
        if "C2" in sites:
            pairs.extend([("O2", "S2"), ("S2", "O1S"), ("S2", "O2S"), ("S2", "O3S")])
        else:
            pairs.append(("O2", "HO2"))
        if "C4" in sites:
            pairs.extend([("O4", "S4"), ("S4", "O4S"), ("S4", "O5S"), ("S4", "O6S")])
        elif not o4_glycosidic:
            pairs.append(("O4", "HO4"))
        if not o3_glycosidic:
            pairs.append(("O3", "HO3"))
        for a, b in pairs:
            bonds.append((index[(i, a)], index[(i, b)]))

        if i > 1:
            acceptor_o = "O3" if linkage_from_prev == "1-3" else "O4"
            bonds.append((index[(i - 1, "C1")], index[(i, acceptor_o)]))
            kind = "alpha-(1->3)" if linkage_from_prev == "1-3" else "alpha-(1->4)"
            linkages.append(f"AFUS{i-1}.C1 - AFUS{i}.{acceptor_o}  {kind}")

    n_sulf = sum(len(s) for s in sulfation)
    meta = {
        "n_residues": n,
        "residue_name": "AFUS",
        "n_sulfates": n_sulf,
        "net_charge": -float(n_sulf),
        "sulfation_sites": sulfation,
        "monomer": "alpha-L-Fucp",
        "linkage": "alternating alpha-(1->3)/alpha-(1->4)",
    }
    return atoms, bonds, linkages, meta


def _fmt_charge(q: float) -> str:
    return f"{q:.4f}"


def write_pdb(path: Path, atoms: list[Atom], title: str) -> None:
    lines = [f"TITLE     {title}", "REMARK    Protocol template — not a crystal structure"]
    for i, atom in enumerate(atoms, start=1):
        x, y, z = (c * 10.0 for c in atom.xyz)  # nm → Å
        name = f"{atom.name:>4s}" if len(atom.name) >= 4 else f" {atom.name:<3s}"
        lines.append(
            f"ATOM  {i:5d} {name} {atom.resname:4s}{atom.resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.element:>2s}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gro(path: Path, atoms: list[Atom], title: str, box: tuple[float, float, float]) -> None:
    lines = [title, f"{len(atoms):5d}"]
    for i, atom in enumerate(atoms, start=1):
        lines.append(
            f"{atom.resid:5d}{atom.resname:<5s}{atom.name:>5s}{i:5d}"
            f"{atom.xyz[0]:8.3f}{atom.xyz[1]:8.3f}{atom.xyz[2]:8.3f}"
        )
    lines.append(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_itp(
    path: Path,
    *,
    molname: str,
    atoms: list[Atom],
    bonds: list[tuple[int, int]],
    comment: str,
) -> float:
    lines = [
        f"; {comment}",
        "; Protocol template topology. Atom types match CHARMM36 jul2022.",
        "; Bonded parameters are looked up from the pinned forcefield by type.",
        "",
        "[ moleculetype ]",
        "; name       nrexcl",
        f"{molname:<12s} 3",
        "",
        "[ atoms ]",
        ";  nr   type  resnr residue  atom   cgnr     charge      mass",
    ]
    total = 0.0
    for i, atom in enumerate(atoms, start=1):
        total += atom.charge
        mass = MASS[atom.atype]
        lines.append(
            f"{i:6d} {atom.atype:>7s} {atom.resid:5d} {atom.resname:7s} "
            f"{atom.name:>5s} {i:5d} {_fmt_charge(atom.charge):>10s} {mass:10.5f}"
        )
    lines.extend(["", "[ bonds ]", ";  ai    aj  funct"])
    for a, b in bonds:
        lines.append(f"{a:6d} {b:6d}     1")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return total


def net_charge(atoms: list[Atom]) -> float:
    return sum(a.charge for a in atoms)


def n_residues(atoms: list[Atom]) -> int:
    return max(a.resid for a in atoms)


def com(atoms: list[Atom]) -> tuple[float, float, float]:
    n = len(atoms) or 1
    return (
        sum(a.xyz[0] for a in atoms) / n,
        sum(a.xyz[1] for a in atoms) / n,
        sum(a.xyz[2] for a in atoms) / n,
    )


def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main() -> None:
    TEMPLATES.mkdir(parents=True, exist_ok=True)
    amy_atoms, amy_bonds, amy_links = build_amylose(26)
    fuc_atoms, fuc_bonds, fuc_links, fuc_meta = build_fucoidan(12, alt_sulfation=False)
    alt_atoms, alt_bonds, alt_links, alt_meta = build_fucoidan(12, alt_sulfation=True)

    amy_q = net_charge(amy_atoms)
    fuc_q = net_charge(fuc_atoms)
    alt_q = net_charge(alt_atoms)
    if abs(amy_q) > 1e-6:
        raise SystemExit(f"amylose net charge {amy_q} != 0")
    if abs(fuc_q - (-12.0)) > 1e-6:
        raise SystemExit(f"fucoidan net charge {fuc_q} != -12")
    if abs(alt_q - (-18.0)) > 1e-6:
        raise SystemExit(f"alt fucoidan net charge {alt_q} != -18")
    if n_residues(amy_atoms) != 26 or n_residues(fuc_atoms) != 12:
        raise SystemExit("residue count mismatch")

    separation = dist(com(amy_atoms), com(fuc_atoms))
    if not (0.8 <= separation <= 1.5):
        # Shift fucoidan along x so the bundled pair sits in range.
        delta = 1.20 - separation
        fuc_atoms = [
            Atom(a.name, a.atype, a.charge, (a.xyz[0] + delta, a.xyz[1], a.xyz[2]), a.resname, a.resid, a.element)
            for a in fuc_atoms
        ]
        alt_atoms = [
            Atom(a.name, a.atype, a.charge, (a.xyz[0] + delta, a.xyz[1], a.xyz[2]), a.resname, a.resid, a.element)
            for a in alt_atoms
        ]
        separation = dist(com(amy_atoms), com(fuc_atoms))

    box = (8.0, 8.0, 8.0)
    write_pdb(TEMPLATES / "amylose_26.pdb", amy_atoms, "Amylose-26 alpha-D-Glc alpha-(1-4) left-handed helix")
    write_gro(TEMPLATES / "amylose_26.gro", amy_atoms, "Amylose-26 protocol template", box)
    write_itp(
        TEMPLATES / "amylose_26.itp",
        molname="Amylose26",
        atoms=amy_atoms,
        bonds=amy_bonds,
        comment="26 x alpha-D-Glcp, alpha-(1->4). Net charge 0.",
    )
    write_pdb(TEMPLATES / "fucoidan_12.pdb", fuc_atoms, "Fucoidan-12 C2-sulfated alpha-L-Fuc")
    write_gro(TEMPLATES / "fucoidan_12.gro", fuc_atoms, "Fucoidan-12 protocol template", box)
    write_itp(
        TEMPLATES / "fucoidan_12.itp",
        molname="Fucoidan12",
        atoms=fuc_atoms,
        bonds=fuc_bonds,
        comment="12 x alpha-L-Fucp, alt 1-3/1-4, C2 sulfate x12. Net charge -12.",
    )
    write_pdb(
        TEMPLATES / "fucoidan_12_alt_sulfation.pdb",
        alt_atoms,
        "Fucoidan-12 C2 + even-C4 sulfated alpha-L-Fuc",
    )
    write_gro(
        TEMPLATES / "fucoidan_12_alt_sulfation.gro",
        alt_atoms,
        "Fucoidan-12 alt sulfation protocol template",
        box,
    )
    write_itp(
        TEMPLATES / "fucoidan_12_alt_sulfation.itp",
        molname="Fucoidan12Alt",
        atoms=alt_atoms,
        bonds=alt_bonds,
        comment="12 x alpha-L-Fucp, alt 1-3/1-4, C2 x12 + C4(even) x6. Net charge -18.",
    )

    (TEMPLATES / "system.top").write_text(
        """\
; Amylose-26 × Fucoidan-12 protocol topology skeleton.
; CHARMM36 is fetched by `python -m tools.md.carb doctor|prepare` (not git-tracked).
; Do not let a model rewrite this file.
#include "charmm36.ff/forcefield.itp"
#include "amylose_26.itp"
#include "fucoidan_12.itp"
#include "charmm36.ff/tip3p.itp"
#include "charmm36.ff/ions.itp"

[ system ]
Amylose-26 / Fucoidan-12 protocol template

[ molecules ]
; Compound        nmols
Amylose26         1
Fucoidan12        1
""",
        encoding="utf-8",
    )

    manifest = {
        "kind": "protocol-template",
        "disclaimer": (
            "Protocol templates for automation smoke tests and scientific "
            "workflow wiring. Not a claim of an experimentally determined "
            "bound pose or binding mechanism."
        ),
        "default_template_id": "amylose26_fucoidan12",
        "com_separation_nm": round(separation, 4),
        "amylose_26": {
            "n_residues": 26,
            "residue_name": "AGLC",
            "monomer": "alpha-D-Glcp",
            "linkage": "alpha-(1->4)",
            "conformation": "left-handed V-type helix (6-fold, rise 0.136 nm)",
            "net_charge": 0.0,
            "n_atoms": len(amy_atoms),
            "files": {
                "pdb": "amylose_26.pdb",
                "gro": "amylose_26.gro",
                "itp": "amylose_26.itp",
            },
            "linkages": amy_links,
        },
        "fucoidan_12": {
            **{k: v for k, v in fuc_meta.items() if k != "sulfation_sites"},
            "n_atoms": len(fuc_atoms),
            "sulfation": "C2 on all 12 residues",
            "files": {
                "pdb": "fucoidan_12.pdb",
                "gro": "fucoidan_12.gro",
                "itp": "fucoidan_12.itp",
            },
            "linkages": fuc_links,
            "sulfation_sites": fuc_meta["sulfation_sites"],
        },
        "fucoidan_12_alt_sulfation": {
            **{k: v for k, v in alt_meta.items() if k != "sulfation_sites"},
            "n_atoms": len(alt_atoms),
            "sulfation": "C2 on all 12 residues; C4 on even residues (1->3 acceptors)",
            "files": {
                "pdb": "fucoidan_12_alt_sulfation.pdb",
                "gro": "fucoidan_12_alt_sulfation.gro",
                "itp": "fucoidan_12_alt_sulfation.itp",
            },
            "linkages": alt_links,
            "sulfation_sites": alt_meta["sulfation_sites"],
        },
    }
    (TEMPLATES / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote templates: amylose n={len(amy_atoms)} q={amy_q:.4f}; "
        f"fucoidan n={len(fuc_atoms)} q={fuc_q:.4f}; "
        f"alt n={len(alt_atoms)} q={alt_q:.4f}; COM {separation:.3f} nm"
    )


if __name__ == "__main__":
    main()
