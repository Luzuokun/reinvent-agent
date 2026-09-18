"""Independent docking module tests — no GPU, no network.

Vina/GNINA launches are mocked. If those binaries are missing, the real
probe test is skipped; mocked paths still run.
"""

from __future__ import annotations

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
from tools.docking.engines import (
    DockingEngineError,
    build_docking_command,
    parse_engine_log,
)
from tools.docking.environment import check_docking_environment
from tools.docking.prepare import DockingPrepareError, resolve_ligands_path, resolve_receptor_path
from tools.docking.run import main as docking_main, run_docking
from tools.environment import check_environment
from tools.mcp_allowlist import McpAllowlistError, invoke_mcp_tool

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "projects" / "demo_project"
MINI_PDBQT = (
    "REMARK tiny\n"
    "ATOM      1  C   LIG     1       0.000   0.000   0.000  1.00  0.00     0.000 C \n"
    "END\n"
)
VINA_LOG = (
    "mode |   affinity | dist from best mode\n"
    "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
    "-----+------------+----------+----------\n"
    "   1         -6.5      0.000      0.000\n"
    "   2         -6.1      1.200      2.000\n"
)


def _project(tmp_path: Path) -> tuple[Path, Path, Path]:
    proj = tmp_path / "proj"
    (proj / "input" / "docking").mkdir(parents=True)
    (proj / "output" / "docking").mkdir(parents=True)
    receptor = proj / "input" / "docking" / "receptor.pdbqt"
    ligand = proj / "input" / "docking" / "ligand.pdbqt"
    receptor.write_text(MINI_PDBQT, encoding="utf-8")
    ligand.write_text(MINI_PDBQT, encoding="utf-8")
    return proj, receptor, ligand


def _fake_vina_run(cmd, stdout=None, **_kwargs):
    if hasattr(stdout, "write"):
        stdout.write(VINA_LOG)
    return MagicMock(returncode=0)


def test_reinvent_env_check_does_not_probe_vina():
    result = check_environment()
    assert "vina" not in result
    assert "gnina" not in result
    assert "python" in result


def test_docking_env_check_never_installs():
    env = check_docking_environment()
    assert env["installs_packages"] is False
    assert "vina" in env
    assert "gnina" in env
    assert "obabel" in env
    assert isinstance(env["vina"], bool)
    assert isinstance(env["gnina"], bool)


@pytest.mark.skipif(shutil.which("vina") is None, reason="vina not on PATH")
def test_vina_probe_if_present():
    env = check_docking_environment()
    assert env["vina"] is True
    assert env["vina_path"]


@pytest.mark.skipif(shutil.which("gnina") is None, reason="gnina not on PATH")
def test_gnina_probe_if_present():
    env = check_docking_environment()
    assert env["gnina"] is True


def test_parse_vina_log_reads_best_score():
    modes = parse_engine_log(VINA_LOG, engine="vina")
    assert modes[0]["score"] == -6.5
    assert modes[0]["mode"] == 1
    assert len(modes) == 2


def test_build_command_rejects_foreign_executable():
    with pytest.raises(DockingEngineError, match="does not match engine"):
        build_docking_command(
            engine="vina",
            executable="/bin/bash",
            receptor="r.pdbqt",
            ligand="l.pdbqt",
            out="o.pdbqt",
            center=(0, 0, 0),
            size=(20, 20, 20),
            exhaustiveness=8,
            num_modes=1,
        )


def test_build_command_predefined_argv():
    cmd = build_docking_command(
        engine="vina",
        executable="/usr/bin/vina",
        receptor="r.pdbqt",
        ligand="l.pdbqt",
        out="o.pdbqt",
        center=(1, 2, 3),
        size=(20, 20, 20),
        exhaustiveness=8,
        num_modes=1,
    )
    assert cmd[0] == "/usr/bin/vina"
    assert "--receptor" in cmd
    assert "bash" not in cmd


def test_receptor_must_stay_under_input(tmp_path: Path):
    proj, _receptor, _ligand = _project(tmp_path)
    outside = tmp_path / "secret.pdbqt"
    outside.write_text(MINI_PDBQT, encoding="utf-8")
    with pytest.raises(DockingPrepareError, match="escapes"):
        resolve_receptor_path(proj, outside)


def test_ligands_must_stay_under_input_or_output(tmp_path: Path):
    proj, _receptor, _ligand = _project(tmp_path)
    outside = tmp_path / "ligands.smi"
    outside.write_text("CCO\n", encoding="utf-8")
    with pytest.raises(DockingPrepareError, match="escapes"):
        resolve_ligands_path(proj, outside)


def test_run_docking_requires_approval(tmp_path: Path):
    proj, receptor, ligand = _project(tmp_path)
    # Patch the per-ligand launcher, not subprocess.run: the latter is the
    # stdlib function, so a patch would also swallow vina --version in the
    # docking env check whenever vina is on PATH.
    with patch("tools.docking.run._dock_one") as mocked:
        result = run_docking(
            proj,
            receptor=receptor,
            ligands=ligand,
            engine="vina",
            center=(0, 0, 0),
            size=(20, 20, 20),
            approve=False,
        )
    mocked.assert_not_called()
    assert result["skipped"] is True
    assert result["table_present"] is False
    assert result["scores_csv"] is None
    assert "--approve-dock" in result["message"]
    assert not (proj / "output" / "docking" / "scores.csv").is_file()


def test_run_docking_mocked_vina_writes_score_table(tmp_path: Path):
    proj, receptor, ligand = _project(tmp_path)
    with patch("tools.docking.run.check_docking_environment") as env:
        env.return_value = {
            "vina": True,
            "vina_path": "/fake/bin/vina",
            "gnina": False,
            "gnina_path": None,
            "obabel": False,
            "obabel_path": None,
            "meeko": False,
            "rdkit": False,
            "installs_packages": False,
        }
        with patch("tools.docking.run.subprocess.run", side_effect=_fake_vina_run) as mocked:
            result = run_docking(
                proj,
                receptor=receptor,
                ligands=ligand,
                engine="vina",
                center=(0, 0, 0),
                size=(20, 20, 20),
                approve=True,
                assume_yes=True,
            )
    mocked.assert_called()
    assert mocked.call_args.kwargs.get("shell") is False
    cmd = mocked.call_args.args[0]
    assert cmd[0] == "/fake/bin/vina"
    assert result["success"] is True
    assert result["table_present"] is True
    assert result["n_scored"] == 1
    assert result["score"]["best"] == -6.5
    scores = Path(result["scores_csv"])
    assert scores.is_file()
    assert scores.parent == (proj / "output" / "docking").resolve()
    text = scores.read_text(encoding="utf-8")
    assert "-6.5" in text
    assert "vina" in text


def test_run_docking_missing_vina_does_not_invent_scores(tmp_path: Path):
    proj, receptor, ligand = _project(tmp_path)
    with patch("tools.docking.run.check_docking_environment") as env:
        env.return_value = {
            "vina": False,
            "vina_path": None,
            "gnina": False,
            "gnina_path": None,
            "obabel": False,
            "obabel_path": None,
            "meeko": False,
            "rdkit": False,
            "installs_packages": False,
        }
        with patch("tools.docking.run.subprocess.run") as mocked:
            result = run_docking(
                proj,
                receptor=receptor,
                ligands=ligand,
                engine="vina",
                center=(0, 0, 0),
                size=(20, 20, 20),
                approve=True,
                assume_yes=True,
            )
    mocked.assert_not_called()
    assert result["success"] is False
    assert result["table_present"] is False
    assert result["n_scored"] == 0
    assert "not found on PATH" in result["message"]


def test_unknown_engine_rejected(tmp_path: Path):
    proj, receptor, ligand = _project(tmp_path)
    with pytest.raises(DockingEngineError, match="Unknown docking engine"):
        run_docking(
            proj,
            receptor=receptor,
            ligands=ligand,
            engine="bash",
            center=(0, 0, 0),
            size=(20, 20, 20),
            approve=False,
        )


def test_cli_without_approve_does_not_launch(tmp_path: Path):
    proj, receptor, ligand = _project(tmp_path)
    with patch("tools.docking.run._dock_one") as mocked:
        code = docking_main(
            [
                "--project",
                str(proj),
                "--receptor",
                str(receptor),
                "--ligands",
                str(ligand),
                "--engine",
                "vina",
                "--center",
                "0",
                "0",
                "0",
                "--size",
                "20",
                "20",
                "20",
            ]
        )
    mocked.assert_not_called()
    assert code == 0


def test_executor_does_not_run_vina_or_gmx():
    plan = {
        "goal": "sneak docking",
        "project_dir": str(DEMO),
        "approve_run": False,
        "skip_reinvent": True,
        "steps": ["docking", "run_vina", "gmx"],
        "notes": [],
    }
    with patch("subprocess.run") as mocked:
        results = ExecutionAgent().execute(
            plan, project_dir=DEMO, approve_run=False
        )
    mocked.assert_not_called()
    warnings = " ".join(results.get("warnings") or [])
    assert "independent module" in warnings
    assert "run_vina" in warnings
    assert "gmx" in warnings
    assert "docking" not in results.get("steps", {})


def test_plan_schema_rejects_docking_step():
    with pytest.raises(PlanValidationError, match="Unknown plan step"):
        validate_plan(
            {"steps": ["check_environment", "run_vina"]},
            project_dir=DEMO,
            output_dir=DEMO / "output",
            source="llm",
        )
    with pytest.raises(PlanValidationError, match="Unknown plan step"):
        validate_plan(
            {"steps": ["docking"]},
            project_dir=DEMO,
            output_dir=DEMO / "output",
            source="llm",
        )


def test_mcp_rejects_docking_and_gmx_tools():
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("run_vina", {"project_dir": str(DEMO)})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("docking", {"receptor": "/tmp/x"})
    with pytest.raises(McpAllowlistError, match="Unknown tool"):
        invoke_mcp_tool("gmx", {"command": "gmx mdrun"})


def test_critic_rejects_docking_claim_without_table():
    with pytest.raises(CriticValidationError, match="invented out-of-scope"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["Docking scores look excellent against EGFR."],
                "recommendation": "Proceed to wet lab.",
            },
            evidence={"analysis": {"total_molecules": 10}},
            source="llm",
        )


def test_critic_rejects_docking_claim_if_only_plan_names_the_step():
    evidence = build_evidence_summary(
        {
            "plan": {"steps": ["docking", "run_vina"]},
            "steps": {},
        }
    )
    assert "docking" not in (evidence or {}) or evidence.get("docking") in (None, {})
    with pytest.raises(CriticValidationError, match="invented out-of-scope"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["Vina docking score is -12.0."],
                "recommendation": "Great binders.",
            },
            evidence=evidence,
            source="llm",
        )


def test_critic_allows_docking_scores_when_table_present():
    evidence = build_evidence_summary(
        {
            "plan": {"skip_reinvent": True, "steps": ["docking"]},
            "steps": {
                "docking": {
                    "ok": True,
                    "approved": True,
                    "skipped": False,
                    "engine": "vina",
                    "table_present": True,
                    "scores_csv": "output/docking/scores.csv",
                    "n_ligands": 2,
                    "n_scored": 2,
                    "score": {"mean": -5.1, "best": -7.2, "n": 2},
                }
            },
        }
    )
    assert evidence["docking"]["table_present"] is True
    result = validate_critic_verdict(
        {
            "status": "PASS",
            "issues": ["Best vina docking score is -7.2 (n=2)."],
            "recommendation": "Human review of the docking table is recommended.",
        },
        evidence=evidence,
        source="llm",
    )
    assert result["status"] == "PASS"


def test_critic_still_rejects_md_when_docking_table_present():
    evidence = build_evidence_summary(
        {
            "steps": {
                "docking": {
                    "ok": True,
                    "engine": "vina",
                    "table_present": True,
                    "scores_csv": "output/docking/scores.csv",
                    "n_scored": 1,
                    "score": {"best": -6.5, "mean": -6.5, "n": 1},
                }
            }
        }
    )
    with pytest.raises(CriticValidationError, match="gromacs"):
        validate_critic_verdict(
            {
                "status": "PASS",
                "issues": ["GROMACS RMSD confirms the vina pose."],
                "recommendation": "MD is stable.",
            },
            evidence=evidence,
            source="llm",
        )


def test_deterministic_critic_warns_when_docking_skipped():
    verdict = CriticAgent().review(
        {
            "plan": {"approve_run": False, "skip_reinvent": True},
            "steps": {
                "docking": {
                    "ok": True,
                    "approved": False,
                    "skipped": True,
                    "engine": "vina",
                    "table_present": False,
                    "n_scored": 0,
                }
            },
            "warnings": [],
            "errors": [],
        }
    )
    assert verdict["status"] == "WARNING"
    assert any("approve-dock" in item for item in verdict["issues"])


def test_deterministic_critic_reads_docking_table():
    verdict = CriticAgent().review(
        {
            "plan": {"approve_run": False, "skip_reinvent": True},
            "steps": {
                "docking": {
                    "ok": True,
                    "approved": True,
                    "skipped": False,
                    "engine": "vina",
                    "table_present": True,
                    "scores_csv": "output/docking/scores.csv",
                    "n_scored": 3,
                    "score": {"best": -8.1, "mean": -6.0, "n": 3},
                }
            },
            "warnings": [],
            "errors": [],
        }
    )
    assert verdict["status"] == "PASS"
    assert any("best=-8.1" in item for item in verdict["issues"])


def test_build_evidence_omits_docking_key_without_table():
    evidence = build_evidence_summary(
        {
            "plan": {"steps": ["analyze_molecules"]},
            "steps": {
                "analyze_molecules": {"ok": True, "total_molecules": 2},
            },
        }
    )
    assert "docking" not in evidence
    blob = json.dumps(evidence)
    # The analysis path must not unlock docking talk via a null key.
    assert '"docking"' not in blob


def test_demo_receptor_fixture_exists():
    receptor = DEMO / "input" / "docking" / "receptor.pdbqt"
    ligands = DEMO / "input" / "docking" / "ligands.smi"
    ligand = DEMO / "input" / "docking" / "ligand.pdbqt"
    assert receptor.is_file()
    assert ligands.is_file()
    resolved = resolve_receptor_path(DEMO, "input/docking/receptor.pdbqt")
    assert resolved == receptor.resolve()
    text = ligand.read_text(encoding="utf-8")
    assert "ROOT" in text
    assert "TORSDOF" in text
