"""Independent GROMACS MD module tests — no GPU, no 100 ns, no network.

gmx launches are mocked. If the binary is missing, the real probe test is
skipped; mocked paths still run.
"""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.critic import CriticAgent
from agents.critic_schema import (
    CriticValidationError,
    build_evidence_summary,
    validate_critic_verdict,
)
from agents.executor import ExecutionAgent
from agents.plan_schema import PlanValidationError, validate_plan
from tools.environment import check_environment
from tools.mcp_allowlist import McpAllowlistError, invoke_mcp_tool
from tools.md.analyze import parse_xvg
from tools.md.commands import MdCommandError, build_mdrun_command
from tools.md.environment import check_md_environment
from tools.md.mdp import (
    DEFAULT_PROTOCOL,
    MAX_LAUNCH_NSTEPS,
    MdpError,
    assert_launch_nsteps,
    materialize_mdp,
    materialize_protocol,
    parse_mdp_nsteps,
    read_template_text,
    replace_mdp_scalar,
)
from tools.md.prepare import MdPrepareError, resolve_structure_path, resolve_topology_path
from tools.md.run import main as md_main, run_md

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "projects" / "demo_project"

RMSD_XVG = """# fake rmsd
@    title "RMSD"
@    xaxis  label "Time (ps)"
@    yaxis  label "RMSD (nm)"
    0.000000    0.000000
    0.050000    0.012000
    0.100000    0.020000
"""
RMSF_XVG = """# fake rmsf
@    title "RMSF"
    1    0.010000
    2    0.015000
    3    0.012000
"""


def _project(tmp_path: Path) -> tuple[Path, Path, Path]:
    proj = tmp_path / "proj"
    (proj / "input" / "md").mkdir(parents=True)
    (proj / "output" / "md").mkdir(parents=True)
    gro = proj / "input" / "md" / "system.gro"
    top = proj / "input" / "md" / "system.top"
    gro.write_text(
        (DEMO / "input" / "md" / "system.gro").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    top.write_text(
        (DEMO / "input" / "md" / "system.top").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return proj, gro, top


def _fake_env() -> dict[str, object]:
    return {
        "gmx": True,
        "gmx_path": "/fake/bin/gmx",
        "gmx_binary": "gmx",
        "gmx_version": "GROMACS fake",
        "installs_packages": False,
    }


def _opt(cmd: list[str], flag: str) -> Path | None:
    if flag not in cmd:
        return None
    return Path(cmd[cmd.index(flag) + 1])


def _fake_gmx(command, *, cwd, log_path, timeout, stdin_text=None):
    assert Path(command[0]).name == "gmx"
    assert command[1] in {"grompp", "mdrun", "rms", "rmsf"}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("ok\n", encoding="utf-8")
    sub = command[1]
    if sub == "grompp":
        tpr = _opt(command, "-o")
        if tpr:
            tpr.parent.mkdir(parents=True, exist_ok=True)
            tpr.write_bytes(b"tpr")
    elif sub == "mdrun":
        deffnm = _opt(command, "-deffnm")
        if deffnm:
            deffnm.parent.mkdir(parents=True, exist_ok=True)
            Path(str(deffnm) + ".gro").write_text("fake gro\n", encoding="utf-8")
            Path(str(deffnm) + ".xtc").write_bytes(b"xtc")
    elif sub == "rms":
        xvg = _opt(command, "-o")
        if xvg:
            xvg.write_text(RMSD_XVG, encoding="utf-8")
    elif sub == "rmsf":
        xvg = _opt(command, "-o")
        if xvg:
            xvg.write_text(RMSF_XVG, encoding="utf-8")
    return MagicMock(returncode=0)


def test_reinvent_env_check_does_not_probe_gmx():
    result = check_environment()
    assert "gmx" not in result
    assert "gromacs" not in result
    assert "python" in result


def test_md_env_check_never_installs():
    env = check_md_environment()
    assert env["installs_packages"] is False
    assert "gmx" in env
    assert isinstance(env["gmx"], bool)


@pytest.mark.skipif(
    shutil.which("gmx") is None and shutil.which("gmx_mpi") is None,
    reason="gmx not on PATH",
)
def test_gmx_probe_if_present():
    env = check_md_environment()
    assert env["gmx"] is True
    assert env["gmx_path"]


def test_production_template_is_100ns_not_default_protocol():
    text = read_template_text("production")
    assert parse_mdp_nsteps(text) == 50_000_000
    assert "continuation" in text.lower()
    assert DEFAULT_PROTOCOL == "em-nvt"
    smoke = read_template_text("nvt")
    assert parse_mdp_nsteps(smoke) == 500
    assert parse_mdp_nsteps(smoke) <= MAX_LAUNCH_NSTEPS


def test_materialize_allowlisted_nsteps_override(tmp_path: Path):
    dest = tmp_path / "nvt.mdp"
    meta = materialize_mdp("nvt", dest, overrides={"nsteps": 50, "dt": 0.001, "ref_t": 300})
    text = dest.read_text(encoding="utf-8")
    assert parse_mdp_nsteps(text) == 50
    assert "dt" in text and "0.001" in text
    assert "300" in text
    assert meta["overrides"]["nsteps"] == 50
    assert "integrator              = md" in text


def test_materialize_rejects_integrator_and_cutoff_overrides(tmp_path: Path):
    dest = tmp_path / "nvt.mdp"
    with pytest.raises(MdpError, match="forbidden"):
        materialize_mdp("nvt", dest, overrides={"integrator": "sd"})
    with pytest.raises(MdpError, match="forbidden"):
        materialize_mdp("nvt", dest, overrides={"rcoulomb": 1.0})
    with pytest.raises(MdpError, match="forbidden"):
        materialize_mdp("nvt", dest, overrides={"constraints": "all-bonds"})


def test_materialize_rejects_free_text_override(tmp_path: Path):
    dest = tmp_path / "nvt.mdp"
    with pytest.raises(MdpError, match="scalar"):
        materialize_mdp(
            "nvt",
            dest,
            overrides={"nsteps": "50; integrator = sd"},
        )


def test_materialize_mdp_has_no_full_mdp_argument():
    params = inspect.signature(materialize_mdp).parameters
    for banned in ("text", "body", "mdp_text", "contents", "mdp"):
        assert banned not in params


def test_replace_does_not_insert_missing_keys():
    text = read_template_text("minimization")
    with pytest.raises(MdpError, match="not present"):
        replace_mdp_scalar(text, "dt", 0.002)


def test_launch_cap_rejects_100ns():
    with pytest.raises(MdpError, match="launch cap"):
        assert_launch_nsteps(50_000_000)
    assert assert_launch_nsteps(500) == 500


def test_build_mdrun_rejects_foreign_executable():
    with pytest.raises(MdCommandError, match="not an allowlisted"):
        build_mdrun_command(
            executable="/bin/bash",
            tpr="em.tpr",
            deffnm="em",
        )


def test_run_gmx_passes_shell_false(tmp_path: Path):
    from tools.md.run import _run_gmx

    log_path = tmp_path / "grompp.log"
    with patch("tools.md.run.subprocess.run") as mocked:
        mocked.return_value = MagicMock(returncode=0)
        _run_gmx(
            [
                "/usr/bin/gmx",
                "grompp",
                "-f",
                "a.mdp",
                "-c",
                "a.gro",
                "-p",
                "a.top",
                "-o",
                "a.tpr",
            ],
            cwd=tmp_path,
            log_path=log_path,
            timeout=5,
        )
    assert mocked.call_args.kwargs["shell"] is False
    assert mocked.call_args.args[0][0] == "/usr/bin/gmx"
    with pytest.raises(Exception, match="non-GROMACS|refusing"):
        _run_gmx(
            ["/bin/bash", "-c", "echo pwned"],
            cwd=tmp_path,
            log_path=log_path,
            timeout=5,
        )


def test_build_mdrun_predefined_cpu_argv():
    cmd = build_mdrun_command(
        executable="/usr/bin/gmx",
        tpr="nvt.tpr",
        deffnm="nvt",
    )
    assert cmd[0] == "/usr/bin/gmx"
    assert cmd[1] == "mdrun"
    assert "-nb" in cmd and "cpu" in cmd
    assert "-nt" in cmd
    assert "bash" not in cmd


def test_structure_must_stay_under_input(tmp_path: Path):
    proj, _gro, _top = _project(tmp_path)
    outside = tmp_path / "secret.gro"
    outside.write_text("x\n", encoding="utf-8")
    with pytest.raises(MdPrepareError, match="escapes"):
        resolve_structure_path(proj, outside)


def test_topology_must_stay_under_input(tmp_path: Path):
    proj, _gro, _top = _project(tmp_path)
    outside = tmp_path / "secret.top"
    outside.write_text("x\n", encoding="utf-8")
    with pytest.raises(MdPrepareError, match="escapes"):
        resolve_topology_path(proj, outside)


def test_run_md_requires_approval(tmp_path: Path):
    proj, gro, top = _project(tmp_path)
    with patch("tools.md.run._run_gmx") as mocked:
        result = run_md(
            proj,
            structure=gro,
            topology=top,
            protocol="em-nvt",
            approve=False,
        )
    mocked.assert_not_called()
    assert result["skipped"] is True
    assert result["table_present"] is False
    assert result["rmsd_csv"] is None
    assert "--approve-md" in result["message"]
    assert not (proj / "output" / "md" / "rmsd.csv").is_file()
    mdp = Path(result["mdp"]["files"][0]["dest"])
    assert mdp.is_file()
    assert parse_mdp_nsteps(mdp.read_text(encoding="utf-8")) == 500


def test_run_md_mocked_gmx_writes_rmsd_rmsf_tables(tmp_path: Path):
    proj, gro, top = _project(tmp_path)
    with patch("tools.md.run.check_md_environment", return_value=_fake_env()):
        with patch("tools.md.run._run_gmx", side_effect=_fake_gmx) as mocked:
            result = run_md(
                proj,
                structure=gro,
                topology=top,
                protocol="em-nvt",
                nsteps=50,
                approve=True,
                assume_yes=True,
            )
    assert mocked.called
    for call in mocked.call_args_list:
        assert call.kwargs.get("shell") is False or "shell" not in call.kwargs
        cmd = call.args[0]
        assert cmd[0] == "/fake/bin/gmx"
        assert "bash" not in cmd
    assert result["success"] is True
    assert result["table_present"] is True
    assert result["n_frames"] == 3
    assert result["rmsd"]["last"] == 0.02
    rmsd = Path(result["rmsd_csv"])
    rmsf = Path(result["rmsf_csv"])
    assert rmsd.is_file()
    assert rmsf.is_file()
    assert rmsd.parent == (proj / "output" / "md").resolve()
    text = rmsd.read_text(encoding="utf-8")
    assert "rmsd_nm" in text
    assert "0.02" in text
    materialized = Path(result["mdp"]["files"][1]["dest"]).read_text(encoding="utf-8")
    assert parse_mdp_nsteps(materialized) == 50
    assert "50000000" not in materialized


def test_run_md_missing_gmx_does_not_invent_tables(tmp_path: Path):
    proj, gro, top = _project(tmp_path)
    env = {
        "gmx": False,
        "gmx_path": None,
        "gmx_binary": None,
        "gmx_version": None,
        "installs_packages": False,
    }
    with patch("tools.md.run.check_md_environment", return_value=env):
        with patch("tools.md.run._run_gmx") as mocked:
            result = run_md(
                proj,
                structure=gro,
                topology=top,
                protocol="em-nvt",
                approve=True,
                assume_yes=True,
            )
    mocked.assert_not_called()
    assert result["success"] is False
    assert result["table_present"] is False
    assert "not found on PATH" in result["message"]


def test_production_protocol_does_not_launch_mdrun(tmp_path: Path):
    proj, gro, top = _project(tmp_path)
    with patch("tools.md.run.check_md_environment", return_value=_fake_env()):
        with patch("tools.md.run._run_gmx") as mocked:
            result = run_md(
                proj,
                structure=gro,
                topology=top,
                protocol="production",
                approve=True,
                assume_yes=True,
            )
    mocked.assert_not_called()
    assert result["success"] is False
    assert result["table_present"] is False
    assert "Production" in result["message"]
    dest = Path(result["mdp"]["files"][0]["dest"])
    assert dest.is_file()
    assert parse_mdp_nsteps(dest.read_text(encoding="utf-8")) == 50_000_000


def test_nsteps_above_launch_cap_does_not_run(tmp_path: Path):
    proj, gro, top = _project(tmp_path)
    with patch("tools.md.run.check_md_environment", return_value=_fake_env()):
        with patch("tools.md.run._run_gmx") as mocked:
            result = run_md(
                proj,
                structure=gro,
                topology=top,
                protocol="nvt",
                nsteps=50_000_000,
                approve=True,
                assume_yes=True,
            )
    mocked.assert_not_called()
    assert result["success"] is False
    assert "launch cap" in result["message"]


def test_cli_without_approve_does_not_launch(tmp_path: Path):
    proj, gro, top = _project(tmp_path)
    with patch("tools.md.run._run_gmx") as mocked:
        code = md_main(
            [
                "--project",
                str(proj),
                "--structure",
                str(gro),
                "--topology",
                str(top),
                "--protocol",
                "em-nvt",
            ]
        )
    mocked.assert_not_called()
    assert code == 0


def test_cli_has_no_mdp_blob_or_integrator_flags():
    from tools.md.run import build_parser

    help_text = build_parser().format_help()
    assert "--approve-md" in help_text
    assert "--nsteps" in help_text
    assert "--mdp-text" not in help_text
    assert "--integrator" not in help_text
    assert "--cutoff" not in help_text
    assert "--mdp " not in help_text + " "


def test_parse_xvg_file(tmp_path: Path):
    path = tmp_path / "rmsd.xvg"
    path.write_text(RMSD_XVG, encoding="utf-8")
    rows = parse_xvg(path)
    assert rows[0] == (0.0, 0.0)
    assert rows[-1][1] == 0.02


def test_executor_does_not_run_gmx():
    plan = {
        "goal": "sneak md",
        "project_dir": str(DEMO),
        "approve_run": False,
        "skip_reinvent": True,
        "steps": ["md", "run_md", "gmx", "gromacs"],
        "notes": [],
    }
    with patch("subprocess.run") as mocked:
        results = ExecutionAgent().execute(
            plan, project_dir=DEMO, approve_run=False
        )
    mocked.assert_not_called()
    warnings = " ".join(results.get("warnings") or [])
    assert "independent module" in warnings
    assert "python -m tools.md" in warnings
    assert "md" not in results.get("steps", {})
    assert "gmx" not in results.get("steps", {})


def test_plan_schema_rejects_md_step():
    with pytest.raises(PlanValidationError, match="Unknown plan step"):
        validate_plan(
            {"steps": ["check_environment", "run_md"]},
            project_dir=DEMO,
            output_dir=DEMO / "output",
            source="llm",
        )
    with pytest.raises(PlanValidationError, match="Unknown plan step"):
        validate_plan(
            {"steps": ["gmx"]},
            project_dir=DEMO,
            output_dir=DEMO / "output",
            source="llm",
        )


def test_mcp_rejects_md_and_gmx_tools():
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("run_md", {"project_dir": str(DEMO)})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("md", {"structure": "/tmp/x"})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("gmx", {"command": "gmx mdrun"})


def test_critic_rejects_md_claim_without_table():
    with pytest.raises(CriticValidationError, match="invented out-of-scope"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["GROMACS RMSF looks stable."],
                "recommendation": "Proceed to wet lab.",
            },
            evidence={"analysis": {"total_molecules": 10}},
            source="llm",
        )


def test_critic_rejects_md_claim_if_only_plan_names_the_step():
    evidence = build_evidence_summary(
        {
            "plan": {"steps": ["md", "gmx"]},
            "steps": {},
        }
    )
    assert "md" not in (evidence or {}) or evidence.get("md") in (None, {})
    with pytest.raises(CriticValidationError, match="invented out-of-scope"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["Molecular dynamics RMSF is 0.1 nm."],
                "recommendation": "MD is stable.",
            },
            evidence=evidence,
            source="llm",
        )


def test_critic_allows_md_metrics_when_table_present():
    evidence = build_evidence_summary(
        {
            "plan": {"skip_reinvent": True, "steps": ["md"]},
            "steps": {
                "md": {
                    "ok": True,
                    "approved": True,
                    "skipped": False,
                    "engine": "gmx",
                    "protocol": "em-nvt",
                    "table_present": True,
                    "rmsd_csv": "output/md/rmsd.csv",
                    "rmsf_csv": "output/md/rmsf.csv",
                    "n_frames": 3,
                    "rmsd": {"mean": 0.012, "max": 0.02, "last": 0.02, "n": 3},
                    "rmsf": {"mean": 0.012, "max": 0.015, "n": 3},
                }
            },
        }
    )
    assert evidence["md"]["table_present"] is True
    result = validate_critic_verdict(
        {
            "status": "PASS",
            "issues": ["GROMACS RMSF mean is 0.012 nm (n=3)."],
            "recommendation": "Human review of the MD table is recommended.",
        },
        evidence=evidence,
        source="llm",
    )
    assert result["status"] == "PASS"


def test_critic_still_rejects_docking_when_only_md_table_present():
    evidence = build_evidence_summary(
        {
            "steps": {
                "md": {
                    "ok": True,
                    "engine": "gmx",
                    "table_present": True,
                    "rmsd_csv": "output/md/rmsd.csv",
                    "n_frames": 2,
                    "rmsd": {"mean": 0.1, "n": 2},
                }
            }
        }
    )
    with pytest.raises(CriticValidationError, match="vina"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["Vina docking score is -12.0."],
                "recommendation": "Great binders.",
            },
            evidence=evidence,
            source="llm",
        )


def test_deterministic_critic_warns_when_md_skipped():
    verdict = CriticAgent().review(
        {
            "plan": {"approve_run": False, "skip_reinvent": True},
            "steps": {
                "md": {
                    "ok": True,
                    "approved": False,
                    "skipped": True,
                    "engine": "gmx",
                    "table_present": False,
                    "n_frames": 0,
                }
            },
            "warnings": [],
            "errors": [],
        }
    )
    assert verdict["status"] == "WARNING"
    assert any("approve-md" in item for item in verdict["issues"])


def test_deterministic_critic_reads_md_table():
    verdict = CriticAgent().review(
        {
            "plan": {"approve_run": False, "skip_reinvent": True},
            "steps": {
                "md": {
                    "ok": True,
                    "approved": True,
                    "skipped": False,
                    "engine": "gmx",
                    "protocol": "em-nvt",
                    "table_present": True,
                    "rmsd_csv": "output/md/rmsd.csv",
                    "n_frames": 4,
                    "rmsd": {"mean": 0.11, "max": 0.2, "last": 0.12, "n": 4},
                }
            },
            "warnings": [],
            "errors": [],
        }
    )
    assert verdict["status"] == "PASS"
    assert any("rmsd_mean=0.11" in item for item in verdict["issues"])


def test_build_evidence_omits_md_key_without_table():
    evidence = build_evidence_summary(
        {
            "plan": {"steps": ["analyze_molecules"]},
            "steps": {
                "analyze_molecules": {"ok": True, "total_molecules": 2},
            },
        }
    )
    assert "md" not in evidence
    blob = json.dumps(evidence)
    assert '"md"' not in blob
    assert "gromacs" not in blob


def test_demo_md_fixture_exists():
    gro = DEMO / "input" / "md" / "system.gro"
    top = DEMO / "input" / "md" / "system.top"
    assert gro.is_file()
    assert top.is_file()
    resolved = resolve_structure_path(DEMO, "input/md/system.gro")
    assert resolved == gro.resolve()
    assert resolve_topology_path(DEMO, "input/md/system.top") == top.resolve()


def test_materialize_protocol_default_is_not_production(tmp_path: Path):
    meta = materialize_protocol("em-nvt", tmp_path / "mdp")
    assert meta["protocol"] == "em-nvt"
    assert meta["runnable"] is True
    ids = [item["template_id"] for item in meta["files"]]
    assert ids == ["minimization", "nvt"]
    assert "production" not in ids
    for item in meta["files"]:
        assert parse_mdp_nsteps(Path(item["dest"]).read_text(encoding="utf-8")) == 500
