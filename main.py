"""REINVENT4 Agent MVP entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from agents.critic import CriticAgent
from agents.executor import ExecutionAgent
from agents.planner import PlannerAgent
from tools import REPO_ROOT, dumps_pretty, load_agent_config


def _configure_logging(logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "agent.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_path


def _banner(text: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n{text}\n{line}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="REINVENT4 Agent MVP — deterministic workflow"
    )
    parser.add_argument(
        "--project",
        default="projects/demo_project",
        help="Path to REINVENT project directory",
    )
    parser.add_argument(
        "--goal",
        default="Run the REINVENT4 workflow and analyze generated molecules.",
        help="User goal text (recorded in plan/report)",
    )
    parser.add_argument(
        "--approve-run",
        action="store_true",
        help="Explicit human approval to launch the predefined REINVENT command",
    )
    parser.add_argument(
        "--skip-reinvent",
        action="store_true",
        help="Skip REINVENT and analyze an existing CSV (offline mode)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV path for analysis (optional override / offline mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reinvent -s (default from agent.yaml)",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "agent.yaml"),
        help="Path to agent.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent_config = load_agent_config(Path(args.config))
    logs_dirname = agent_config.get("logging", {}).get("logs_dirname", "logs")
    log_path = _configure_logging(REPO_ROOT / logs_dirname)
    logger = logging.getLogger("main")

    project_dir = Path(args.project)
    if not project_dir.is_absolute():
        project_dir = (Path.cwd() / project_dir).resolve()
    else:
        project_dir = project_dir.resolve()

    _banner("REINVENT4 AGENT")
    print(f"Project: {project_dir}")
    print(f"Goal:    {args.goal}")
    print(f"Log:     {log_path}")

    planner = PlannerAgent()
    plan = planner.create_plan(
        args.goal,
        project_dir=str(project_dir),
        approve_run=args.approve_run,
        skip_reinvent=args.skip_reinvent,
    )

    print("\n[Planning Agent]")
    print(f"Steps: {', '.join(plan['steps'])}")
    for note in plan.get("notes") or []:
        print(f"Note:  {note}")

    executor = ExecutionAgent(agent_config=agent_config)
    print("\n[Execution Agent]")
    results = executor.execute(
        plan,
        project_dir=project_dir,
        approve_run=args.approve_run,
        csv_override=args.csv,
        seed=args.seed,
    )

    env = results.get("steps", {}).get("check_environment") or {}
    print(
        f"Environment: reinvent={env.get('reinvent')} "
        f"rdkit={env.get('rdkit')} gpu={env.get('gpu')}"
    )
    validation = results.get("steps", {}).get("validate_project") or {}
    print(f"Validation:  ok={validation.get('ok')}")
    if validation.get("errors"):
        for err in validation["errors"]:
            print(f"  ! {err}")

    run = results.get("steps", {}).get("run_reinvent")
    if run:
        if run.get("skipped"):
            print("REINVENT:    skipped (no --approve-run)")
        else:
            print(
                f"REINVENT:    success={run.get('success')} "
                f"exit={run.get('exit_code')} "
                f"runtime={run.get('runtime_seconds')}s"
            )

    analysis = results.get("steps", {}).get("analyze_molecules") or {}
    if analysis:
        print("\n[Molecule Analysis]")
        if analysis.get("ok"):
            print(f"Total:    {analysis.get('total_molecules')}")
            print(f"Unique:   {analysis.get('unique_molecules')}")
            print(f"Dupes:    {analysis.get('duplicate_molecules')}")
            rdkit = analysis.get("rdkit") or {}
            if rdkit.get("available"):
                print(f"Valid:    {rdkit.get('valid_molecules')}")
                qed = (rdkit.get("qed") or {}).get("mean")
                if qed is not None:
                    print(f"Mean QED: {qed}")
        else:
            print("Analysis failed:")
            for err in analysis.get("errors") or []:
                print(f"  ! {err}")

    critic = CriticAgent(agent_config=agent_config)
    verdict = critic.review(results)
    results["steps"]["critic_review"] = verdict

    print("\n[Critic Agent]")
    print(f"Status: {verdict['status']}")
    for issue in verdict.get("issues") or []:
        print(f"  - {issue}")
    print(f"Recommendation: {verdict['recommendation']}")

    report_meta = executor.write_report(results, goal=args.goal, critic=verdict)
    print("\n[Report]")
    print(f"Wrote: {report_meta.get('report_path')}")

    summary_path = REPO_ROOT / logs_dirname / "last_run_summary.json"
    summary_path.write_text(dumps_pretty({"results": results, "critic": verdict}), encoding="utf-8")
    logger.info("Wrote summary %s", summary_path)

    _banner("DONE")
    if verdict["status"] == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
