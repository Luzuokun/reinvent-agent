"""Amylose–fucoidan carb protocol tests — mock gmx, no 200 ns default, no network."""

from __future__ import annotations

import inspect
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.md.carb.cli import build_parser, main as carb_main
from tools.md.carb.constants import (
    AMYLOSE_CHARGE,
    AMYLOSE_N_RESIDUES,
    AMYLOSE_RESNAME,
    DEFAULT_PROTOCOL,
    DEFAULT_TEMPLATE_ID,
    FUCOIDAN_ALT_CHARGE,
    FUCOIDAN_CHARGE,
    FUCOIDAN_N_RESIDUES,
    FUCOIDAN_RESNAME,
    PRODUCTION_NSTEPS,
    PRODUCTION_PROTOCOL,
    PROTOCOL_ID,
    TEMPLATES_DIR,
)
from tools.md.carb.doctor import doctor
from tools.md.carb.forcefield import (
    ForcefieldError,
    ensure_forcefield,
    forcefield_ready,
    load_pin,
    sha256_file,
)
from tools.md.carb.mdp import (
    MdpError,
    assert_carb_launch_nsteps,
    materialize_carb_protocol,
    parse_mdp_nsteps,
)  # materialize used in short-protocol test
from tools.md.carb.prepare import prepare_carb
from tools.md.carb.report import DISCLAIMER, report_carb
from tools.md.carb.run import run_carb
from tools.md.carb.scalars import normalize_carb_scalars
from tools.md.carb.templates import (
    BANNED_CGENFF_SULFATE_ESTER,
    SULFATE_ESTER_O,
    SULFATE_S,
    SULFATE_TERMINAL_O,
    charmm36_bondtype_exists,
    check_fucoidan_sulfate_charmm36_types,
    parse_itp_charges,
    parse_itp_typed_atoms,
    parse_pdb_residues,
    validate_bundled_templates,
)

REPO = Path(__file__).resolve().parent.parent

XVG_XY = """# fake
@ title "x"
    0.000000    0.100000
    0.050000    0.120000
    0.100000    0.110000
"""


def _opt(cmd: list[str], flag: str) -> Path | None:
    if flag not in cmd:
        return None
    return Path(cmd[cmd.index(flag) + 1])


def _copy_or_touch(src: Path | None, dest: Path | None, text: str = "ok\n") -> None:
    if dest is None:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src is not None and src.is_file():
        dest.write_bytes(src.read_bytes())
    else:
        dest.write_text(text, encoding="utf-8")


def _fake_gmx(command, *, cwd, log_path, timeout, stdin_text=None, allowed_binaries=None):
    assert Path(command[0]).name == "gmx"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("ok\n", encoding="utf-8")
    sub = command[1]
    if sub == "editconf":
        _copy_or_touch(_opt(command, "-f"), _opt(command, "-o"))
    elif sub == "solvate":
        _copy_or_touch(_opt(command, "-cp"), _opt(command, "-o"))
    elif sub == "grompp":
        tpr = _opt(command, "-o")
        if tpr:
            tpr.parent.mkdir(parents=True, exist_ok=True)
            tpr.write_bytes(b"tpr")
    elif sub == "genion":
        gro = _opt(command, "-o")
        src = _opt(command, "-c")
        _copy_or_touch(src, gro, "fake ionized gro\n")
        if gro and not gro.is_file():
            gro.write_text("fake ionized gro\n", encoding="utf-8")
    elif sub == "mdrun":
        deffnm = _opt(command, "-deffnm")
        if deffnm:
            deffnm.parent.mkdir(parents=True, exist_ok=True)
            Path(str(deffnm) + ".gro").write_text("fake gro\n", encoding="utf-8")
            Path(str(deffnm) + ".xtc").write_bytes(b"xtc")
            Path(str(deffnm) + ".edr").write_bytes(b"edr")
            Path(str(deffnm) + ".cpt").write_bytes(b"cpt")
            Path(str(deffnm) + ".tpr").write_bytes(b"tpr")
    elif sub in {"rms", "gyrate", "hbond", "sasa", "distance", "mindist", "energy"}:
        xvg = _opt(command, "-o") or _opt(command, "-num") or _opt(command, "-od") or _opt(command, "-oall")
        if xvg:
            xvg.write_text(XVG_XY, encoding="utf-8")
    return MagicMock(returncode=0)


def _fake_env() -> dict[str, object]:
    return {
        "gmx": True,
        "gmx_path": "/fake/bin/gmx",
        "gmx_binary": "gmx",
        "gmx_version": "GROMACS fake",
        "installs_packages": False,
    }


def _fake_ff() -> dict[str, object]:
    return {
        "ok": True,
        "fetched": False,
        "offline": True,
        "path": "/fake/charmm36-jul2022.ff",
        "alias": "/fake/charmm36.ff",
        "message": "cached",
    }


def test_bundled_templates_residue_counts_and_charges():
    result = validate_bundled_templates()
    assert result["ok"] is True
    amy = result["details"]["amylose_26"]["parsed"]
    fuc = result["details"]["fucoidan_12"]["parsed"]
    alt = result["details"]["fucoidan_12_alt_sulfation"]["parsed"]
    for src in ("pdb", "gro", "itp"):
        assert amy[src]["n_residues"] == AMYLOSE_N_RESIDUES
        assert fuc[src]["n_residues"] == FUCOIDAN_N_RESIDUES
        assert alt[src]["n_residues"] == FUCOIDAN_N_RESIDUES
    assert amy["itp"]["net_charge"] == AMYLOSE_CHARGE
    assert fuc["itp"]["net_charge"] == FUCOIDAN_CHARGE
    assert alt["itp"]["net_charge"] == FUCOIDAN_ALT_CHARGE
    assert AMYLOSE_RESNAME in amy["pdb"]["residue_names"]
    assert FUCOIDAN_RESNAME in fuc["pdb"]["residue_names"]
    sep = result["details"]["com_separation_nm"]
    assert 0.8 <= sep <= 1.5


def test_fucoidan_sulfate_uses_charmm36_carb_types():
    for name in ("fucoidan_12.itp", "fucoidan_12_alt_sulfation.itp"):
        path = TEMPLATES_DIR / name
        text = path.read_text(encoding="utf-8")
        assert BANNED_CGENFF_SULFATE_ESTER not in text
        assert "SG3O1" not in text
        atoms = parse_itp_typed_atoms(path)
        esters = [a for a in atoms if a["name"] in {"O2", "O4"} and a["type"] == SULFATE_ESTER_O]
        sulfurs = [a for a in atoms if a["name"] in {"S2", "S4"}]
        terminals = [a for a in atoms if a["name"] in {"O1S", "O2S", "O3S", "O4S", "O5S", "O6S"}]
        assert esters, f"{name} missing {SULFATE_ESTER_O} ester oxygen"
        assert all(a["type"] == SULFATE_S for a in sulfurs)
        assert all(a["type"] == SULFATE_TERMINAL_O for a in terminals)
        assert len(esters) == len(sulfurs)
        assert check_fucoidan_sulfate_charmm36_types(path) == []
    main = parse_itp_typed_atoms(TEMPLATES_DIR / "fucoidan_12.itp")
    assert sum(1 for a in main if a["type"] == SULFATE_ESTER_O) == 12
    alt = parse_itp_typed_atoms(TEMPLATES_DIR / "fucoidan_12_alt_sulfation.itp")
    assert sum(1 for a in alt if a["type"] == SULFATE_ESTER_O) == 18


def test_charmm36_ffbonded_has_cc3161_oc30p_not_og303():
    ready = forcefield_ready()
    if not ready["ok"]:
        pytest.skip("CHARMM36 cache not present")
    ffbonded = Path(ready["path"]) / "ffbonded.itp"
    assert ffbonded.is_file()
    assert charmm36_bondtype_exists(ffbonded, "CC3161", SULFATE_ESTER_O)
    assert not charmm36_bondtype_exists(ffbonded, "CC3161", BANNED_CGENFF_SULFATE_ESTER)


def test_pdb_and_itp_are_git_tracked():
    for name in (
        "amylose_26.pdb",
        "amylose_26.gro",
        "amylose_26.itp",
        "fucoidan_12.pdb",
        "fucoidan_12.gro",
        "fucoidan_12.itp",
        "fucoidan_12_alt_sulfation.pdb",
        "fucoidan_12_alt_sulfation.itp",
        "system.top",
        "manifest.json",
    ):
        assert (TEMPLATES_DIR / name).is_file()
    pdb = parse_pdb_residues(TEMPLATES_DIR / "amylose_26.pdb")
    assert pdb["n_residues"] == 26
    itp = parse_itp_charges(TEMPLATES_DIR / "fucoidan_12.itp")
    assert itp["net_charge"] == -12.0


def test_default_protocol_is_short_not_200ns():
    assert DEFAULT_PROTOCOL == "em-nvt-npt-short"
    assert DEFAULT_PROTOCOL != PRODUCTION_PROTOCOL
    from tools.md.carb.mdp import PROTOCOL_TEMPLATES

    assert "md_200ns" not in PROTOCOL_TEMPLATES[DEFAULT_PROTOCOL]
    assert "md_200ns" in PROTOCOL_TEMPLATES[PRODUCTION_PROTOCOL]


def test_materialize_short_protocol(tmp_path: Path):
    meta = materialize_carb_protocol(DEFAULT_PROTOCOL, tmp_path / "mdp")
    ids = [item["template_id"] for item in meta["files"]]
    assert ids == ["em", "nvt", "npt"]
    assert "md_200ns" not in ids
    for item in meta["files"]:
        nsteps = parse_mdp_nsteps(Path(item["dest"]).read_text(encoding="utf-8"))
        assert nsteps == 500
        assert nsteps < PRODUCTION_NSTEPS


def test_production_template_is_200ns():
    from tools.md.carb.mdp import carb_template_path

    text = carb_template_path("md_200ns").read_text(encoding="utf-8")
    assert parse_mdp_nsteps(text) == PRODUCTION_NSTEPS
    assert parse_mdp_nsteps(text) * 0.002 == 200_000.0  # ps


def test_launch_cap_blocks_200ns_on_short_protocol():
    with pytest.raises(MdpError, match="launch cap"):
        assert_carb_launch_nsteps(PRODUCTION_NSTEPS, DEFAULT_PROTOCOL)
    assert assert_carb_launch_nsteps(PRODUCTION_NSTEPS, PRODUCTION_PROTOCOL) == PRODUCTION_NSTEPS


def test_whitelist_rejects_integrator_and_free_text():
    with pytest.raises(MdpError, match="forbidden"):
        normalize_carb_scalars({"integrator": "sd"})
    with pytest.raises(MdpError, match="forbidden"):
        normalize_carb_scalars({"constraints": "none"})
    with pytest.raises(MdpError, match="scalar"):
        normalize_carb_scalars({"nsteps": "50; integrator = sd"})
    ok = normalize_carb_scalars(
        {"nsteps": 50, "dt": 0.001, "ref_t": 368, "salt_mol": 0.15, "box_d": 1.2, "n_replicas": 3}
    )
    assert ok["n_replicas"] == 3
    assert ok["salt_mol"] == 0.15


def test_prepare_dry_run_writes_nothing(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    with patch("tools.md.carb.prepare.ensure_forcefield", return_value=_fake_ff()):
        result = prepare_carb(proj, dry_run=True)
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert not (proj / "input" / "md" / "system.gro").exists()
    assert "gmx not run" in result["message"] or "dry-run" in result["message"]


def test_prepare_mocked_gmx_writes_system(tmp_path: Path):
    proj = tmp_path / "proj"
    proj.mkdir()
    with patch("tools.md.carb.prepare.ensure_forcefield", return_value=_fake_ff()):
        with patch("tools.md.carb.prepare.check_md_environment", return_value=_fake_env()):
            result = prepare_carb(proj, run_fn=_fake_gmx)
    assert result["ok"] is True
    assert Path(result["structure"]).is_file()
    assert Path(result["topology"]).is_file()
    top = Path(result["topology"]).read_text(encoding="utf-8")
    assert "Amylose26" in top
    assert "Fucoidan12" in top
    assert "prepare-complex" not in top
    assert (proj / "input" / "md" / "index.ndx").is_file()
    ndx = (proj / "input" / "md" / "index.ndx").read_text(encoding="utf-8")
    assert "[ Amylose ]" in ndx
    assert "[ Fucoidan ]" in ndx


def test_prepare_isolated_from_egfr_complex():
    import tools.md.carb.prepare as prep
    import tools.md.carb.cli as cli
    import tools.md.carb.run as run

    for mod in (prep, cli, run):
        src = inspect.getsource(mod)
        assert "from tools.md.complex" not in src
        assert "import tools.md.complex" not in src
        assert "prepare_complex(" not in src
    assert "build_acpype" not in inspect.getsource(prep)
    assert "build_pdb2gmx" not in inspect.getsource(prep)
    assert "amber99sb-ildn" not in inspect.getsource(prep)


def test_cli_has_no_prepare_complex_or_mdp_blob():
    help_text = build_parser().format_help()
    assert "doctor" in help_text
    assert "--approve-md" in help_text
    assert "--prepare-complex" not in help_text
    assert "--mdp-text" not in help_text
    assert "--integrator" not in help_text
    assert PROTOCOL_ID in help_text or "Amylose" in help_text
    assert PRODUCTION_PROTOCOL in help_text
    assert DEFAULT_PROTOCOL in help_text


def test_default_run_does_not_launch_200ns(tmp_path: Path):
    proj = tmp_path / "proj"
    (proj / "input" / "md").mkdir(parents=True)
    gro = proj / "input" / "md" / "system.gro"
    top = proj / "input" / "md" / "system.top"
    gro.write_text((TEMPLATES_DIR / "amylose_26.gro").read_text(encoding="utf-8"), encoding="utf-8")
    top.write_text((TEMPLATES_DIR / "system.top").read_text(encoding="utf-8"), encoding="utf-8")
    with patch("tools.md.carb.run.check_md_environment", return_value=_fake_env()):
        result = run_carb(proj, approve=True, assume_yes=True, run_fn=_fake_gmx)
    assert result["protocol"] == DEFAULT_PROTOCOL
    assert result["launched_200ns"] is False
    assert result["mdp"]["max_nsteps"] == 500
    assert result["mdp"]["max_nsteps"] < PRODUCTION_NSTEPS
    stages = []
    # smoke dir should have npt, not a 1e8-step md.mdp as the launched production file
    smoke_mdp = Path(result["output_dir"]) / "smoke" / "mdp"
    ids = [p.name for p in smoke_mdp.glob("*.mdp")] if smoke_mdp.is_dir() else []
    assert "md_200ns.mdp" not in ids
    assert result["success"] is True
    status = json.loads((Path(result["output_dir"]) / "job_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["invented_trajectory"] is False
    del stages


def test_run_without_approve_does_not_call_gmx(tmp_path: Path):
    proj = tmp_path / "proj"
    (proj / "input" / "md").mkdir(parents=True)
    gro = proj / "input" / "md" / "system.gro"
    top = proj / "input" / "md" / "system.top"
    gro.write_text((TEMPLATES_DIR / "amylose_26.gro").read_text(encoding="utf-8"), encoding="utf-8")
    top.write_text((TEMPLATES_DIR / "system.top").read_text(encoding="utf-8"), encoding="utf-8")
    mocked = MagicMock()
    with patch("tools.md.carb.run.check_md_environment", return_value=_fake_env()):
        result = run_carb(proj, approve=False, run_fn=mocked)
    mocked.assert_not_called()
    assert result["skipped"] is True
    assert result["launched_200ns"] is False
    assert "--approve-md" in result["message"]
    assert result["mdp"]["max_nsteps"] == 500


def test_production_without_approve_does_not_launch(tmp_path: Path):
    proj = tmp_path / "proj"
    (proj / "input" / "md").mkdir(parents=True)
    gro = proj / "input" / "md" / "system.gro"
    top = proj / "input" / "md" / "system.top"
    gro.write_text((TEMPLATES_DIR / "amylose_26.gro").read_text(encoding="utf-8"), encoding="utf-8")
    top.write_text((TEMPLATES_DIR / "system.top").read_text(encoding="utf-8"), encoding="utf-8")
    mocked = MagicMock()
    with patch("tools.md.carb.run.check_md_environment", return_value=_fake_env()):
        result = run_carb(
            proj, protocol=PRODUCTION_PROTOCOL, approve=False, run_fn=mocked
        )
    mocked.assert_not_called()
    assert result["would_launch_200ns"] is True
    assert result["launched_200ns"] is False
    assert result["skipped"] is True


def test_production_mocked_three_replicas_gpu(tmp_path: Path):
    proj = tmp_path / "proj"
    (proj / "input" / "md").mkdir(parents=True)
    gro = proj / "input" / "md" / "system.gro"
    top = proj / "input" / "md" / "system.top"
    gro.write_text((TEMPLATES_DIR / "amylose_26.gro").read_text(encoding="utf-8"), encoding="utf-8")
    top.write_text((TEMPLATES_DIR / "system.top").read_text(encoding="utf-8"), encoding="utf-8")
    with patch("tools.md.carb.run.check_md_environment", return_value=_fake_env()):
        result = run_carb(
            proj,
            protocol=PRODUCTION_PROTOCOL,
            approve=True,
            assume_yes=True,
            gpu=True,
            run_fn=_fake_gmx,
        )
    assert result["success"] is True
    assert result["launched_200ns"] is True
    assert result["n_replicas"] == 3
    assert result["gpu"] is True
    for i in (1, 2, 3):
        assert (Path(result["output_dir"]) / f"rep{i}" / "md.xtc").is_file()
        nvt = (Path(result["output_dir"]) / f"rep{i}" / "mdp" / "nvt.mdp").read_text(
            encoding="utf-8"
        )
        assert "gen_seed" in nvt
    md_text = (Path(result["output_dir"]) / "rep1" / "mdp" / "md_200ns.mdp").read_text(
        encoding="utf-8"
    )
    assert parse_mdp_nsteps(md_text) == PRODUCTION_NSTEPS
    status = json.loads((Path(result["output_dir"]) / "job_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert len(status["replicas"]) == 3


def test_failed_replica_does_not_invent_trajectory(tmp_path: Path):
    proj = tmp_path / "proj"
    (proj / "input" / "md").mkdir(parents=True)
    gro = proj / "input" / "md" / "system.gro"
    top = proj / "input" / "md" / "system.top"
    gro.write_text((TEMPLATES_DIR / "amylose_26.gro").read_text(encoding="utf-8"), encoding="utf-8")
    top.write_text((TEMPLATES_DIR / "system.top").read_text(encoding="utf-8"), encoding="utf-8")

    def boom(*_a, **_k):
        raise OSError("gmx exploded")

    with patch("tools.md.carb.run.check_md_environment", return_value=_fake_env()):
        result = run_carb(proj, approve=True, assume_yes=True, run_fn=boom)
    assert result["success"] is False
    assert result["invented_trajectory"] is False
    status = json.loads((Path(result["output_dir"]) / "job_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["invented_trajectory"] is False
    assert not (Path(result["output_dir"]) / "smoke" / "npt.xtc").is_file()


def test_charmm36_pin_is_url_and_sha256():
    pin = load_pin()
    assert pin["url"].startswith("https://")
    assert "charmm36-jul2022" in pin["url"]
    assert len(pin["sha256"]) == 64
    assert pin["extracts_to"] == "charmm36-jul2022.ff"


def test_forcefield_offline_when_cache_present(tmp_path: Path):
    pin = load_pin()
    ff = tmp_path / pin["extracts_to"]
    ff.mkdir()
    for name in pin["required_files"]:
        (ff / name).write_text("ok\n", encoding="utf-8")
    with patch("tools.md.carb.forcefield.urllib.request.urlopen") as mocked:
        result = ensure_forcefield(root=tmp_path, fetch=True)
    mocked.assert_not_called()
    assert result["ok"] is True
    assert result["fetched"] is False
    assert result["offline"] is True


def test_forcefield_rejects_sha_mismatch(tmp_path: Path):
    pin = load_pin()
    archive = tmp_path / pin["filename"]
    archive.write_bytes(b"not-the-charmm-tarball")
    with pytest.raises(ForcefieldError, match="sha256 mismatch"):
        ensure_forcefield(root=tmp_path, fetch=False)


def test_forcefield_extracts_valid_tarball(tmp_path: Path):
    pin = load_pin()
    inner = tmp_path / "build" / pin["extracts_to"]
    inner.mkdir(parents=True)
    for name in pin["required_files"]:
        (inner / name).write_text("; fake\n", encoding="utf-8")
    tar_src = tmp_path / "src.tgz"
    with tarfile.open(tar_src, "w:gz") as tf:
        tf.add(inner, arcname=pin["extracts_to"])
    digest = sha256_file(tar_src)
    dest_root = tmp_path / "ffroot"
    dest_root.mkdir()
    archive = dest_root / pin["filename"]
    archive.write_bytes(tar_src.read_bytes())
    fake_pin = dict(pin)
    fake_pin["sha256"] = digest
    with patch("tools.md.carb.forcefield.load_pin", return_value=fake_pin):
        result = ensure_forcefield(root=dest_root, fetch=False)
    assert result["ok"] is True
    assert (dest_root / pin["extracts_to"] / "forcefield.itp").is_file()


def test_doctor_ok_when_ff_mocked():
    with patch("tools.md.carb.doctor.ensure_forcefield", return_value=_fake_ff()):
        result = doctor(fetch_forcefield=True)
    assert result["templates"]["ok"] is True
    assert result["forcefield"]["ok"] is True
    assert result["protocol_id"] == PROTOCOL_ID
    assert result["isolated_from"] == "tools.md.complex.prepare_complex"


def test_report_skips_mmpbsa_and_does_not_invent(tmp_path: Path):
    proj = tmp_path / "proj"
    out = proj / "output" / "carb"
    out.mkdir(parents=True)
    with patch("tools.md.carb.report.check_md_environment", return_value=_fake_env()):
        with patch("tools.md.carb.analyze.shutil.which", return_value=None):
            with patch("tools.md.carb.report.gmx_mmpbsa_status", return_value={
                "status": "skipped",
                "binary": None,
                "message": "SKIPPED: gmx_MMPBSA not found on PATH. MM/PBSA was not run and no binding free energy was invented.",
            }):
                result = report_carb(proj, run_fn=_fake_gmx)
    assert result["mmpbsa"]["status"] == "skipped"
    assert "gmx_MMPBSA" in result["mmpbsa"]["message"]
    assert "SKIPPED" in result["mmpbsa"]["message"]
    html = Path(result["html"]).read_text(encoding="utf-8")
    assert "not prove a binding mechanism" in DISCLAIMER
    assert "binding mechanism" in html.lower() or "Disclaimer" in html
    assert result["invented_trajectory"] is False


def test_report_writes_fel_from_existing_xvg(tmp_path: Path):
    from tools.md.carb.analyze import write_fel_from_series
    from tools.md.analyze import parse_xvg

    rmsd = tmp_path / "rmsd.xvg"
    rg = tmp_path / "rg.xvg"
    rmsd.write_text(XVG_XY, encoding="utf-8")
    rg.write_text(XVG_XY, encoding="utf-8")
    fel = write_fel_from_series(
        tmp_path / "fel.csv", parse_xvg(rmsd), parse_xvg(rg), temperature=368.0
    )
    assert fel is not None
    text = Path(fel["path"]).read_text(encoding="utf-8")
    assert "deltaG_kJ_mol" in text


def test_cli_run_without_approve_exit_zero(tmp_path: Path):
    proj = tmp_path / "proj"
    (proj / "input" / "md").mkdir(parents=True)
    gro = proj / "input" / "md" / "system.gro"
    top = proj / "input" / "md" / "system.top"
    gro.write_text((TEMPLATES_DIR / "amylose_26.gro").read_text(encoding="utf-8"), encoding="utf-8")
    top.write_text((TEMPLATES_DIR / "system.top").read_text(encoding="utf-8"), encoding="utf-8")
    with patch("tools.md.carb.run.check_md_environment", return_value=_fake_env()):
        with patch("tools.md.carb.run._run_gmx") as mocked:
            code = carb_main(["run", "--project", str(proj)])
    mocked.assert_not_called()
    assert code == 0


def test_cli_default_protocol_is_not_200ns():
    parser = build_parser()
    args = parser.parse_args(["run", "--project", "x"])
    assert args.protocol == DEFAULT_PROTOCOL
    assert args.protocol != PRODUCTION_PROTOCOL
    assert args.approve_md is False


def test_materialize_carb_mdp_has_no_full_mdp_argument():
    from tools.md.carb.mdp import materialize_carb_mdp

    params = inspect.signature(materialize_carb_mdp).parameters
    for banned in ("text", "body", "mdp_text", "contents", "mdp"):
        assert banned not in params
