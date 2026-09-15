"""REINVENT4 Agent MVP entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from agents.critic import CriticAgent
from agents.executor import ExecutionAgent
from agents.llm_client import SUPPORTED_PROVIDERS, overlay_provider
from agents.llm_critic import LLMCriticAgent, LLMCriticError
from agents.llm_planner import LLMPlannerAgent, LLMPlannerError
from agents.planner import PlannerAgent
from tools import REPO_ROOT, dumps_pretty, load_agent_config
from tools.artifacts import create_run_dir, write_run_result
from tools.reinvent import confirm_reinvent_launch, prepare_reinvent_command


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


def _exit_code_for_critic(status: str) -> int:
    """Map critic status to process exit code.

    PASS / WARNING → 0 (workflow completed; WARNING still needs human review)
    FAIL → 1
    """
    if str(status).upper() == "FAIL":
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="REINVENT4 Agent MVP — Planning → Execution → Tools → Critic"
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
        "--yes",
        action="store_true",
        help="Skip interactive Proceed? prompt when used with --approve-run",
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
    parser.add_argument(
        "--planner",
        choices=("deterministic", "llm"),
        default=None,
        help=(
            "Planner backend: deterministic (default) or llm. "
            "Overrides config planner.mode."
        ),
    )
    parser.add_argument(
        "--critic",
        choices=("deterministic", "llm"),
        default=None,
        help=(
            "Critic backend: deterministic (default) or llm. "
            "Overrides config critic.mode."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=None,
        help=(
            "LLM provider for --planner llm / --critic llm "
            "(openai, xai, gemini, openai_compatible). "
            "Overrides planner.provider and critic.provider; "
            "preset base_url and API key env apply unless yaml overrides them."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent_config = overlay_provider(
        load_agent_config(Path(args.config)),
        getattr(args, "provider", None),
    )
    logs_dirname = agent_config.get("logging", {}).get("logs_dirname", "logs")
    logs_dir = REPO_ROOT / logs_dirname
    log_path = _configure_logging(logs_dir)
    logger = logging.getLogger("main")
    run_dir = create_run_dir(logs_dir)

    project_dir = Path(args.project)
    if not project_dir.is_absolute():
        project_dir = (Path.cwd() / project_dir).resolve()
    else:
        project_dir = project_dir.resolve()

    config_name = agent_config.get("project", {}).get("config_name", "reinvent.toml")
    config_path = project_dir / config_name

    _banner("REINVENT4 AGENT")
    print(f"Project: {project_dir}")
    print(f"Goal:    {args.goal}")
    print(f"Log:     {log_path}")
    print(f"Run dir: {run_dir}")

    approve_run = bool(args.approve_run)
    if approve_run and not args.skip_reinvent:
        prepared = prepare_reinvent_command(
            config_path,
            seed=args.seed,
            project_dir=project_dir,
            logs_dir=logs_dir,
            agent_config=agent_config,
        )
        if not prepared.get("ok"):
            print(f"Cannot prepare REINVENT command: {prepared.get('message')}")
            approve_run = False
        elif not confirm_reinvent_launch(prepared, assume_yes=args.yes):
            print("Human declined or confirmation unavailable — REINVENT will not run.")
            approve_run = False

    try:
        plan = _create_plan(
            args,
            agent_config,
            project_dir=str(project_dir),
            approve_run=approve_run,
        )
    except LLMPlannerError as exc:
        print(f"\nERROR: {exc}")
        logger.error("LLM planner failed without fallback: %s", exc)
        return 2

    print("\n[Planning Agent]")
    planner_name = plan.get("planner") or "deterministic"
    print(f"Planner: {planner_name}")
    if plan.get("planner_fallback"):
        print("WARNING: LLM planner failed — using deterministic plan (fallback).")
        for warning in plan.get("planner_warnings") or []:
            print(f"WARNING: {warning}")
    print(f"Steps: {', '.join(plan['steps'])}")
    for note in plan.get("notes") or []:
        print(f"Note:  {note}")

    executor = ExecutionAgent(agent_config=agent_config)
    print("\n[Execution Agent]")
    results = executor.execute(
        plan,
        project_dir=project_dir,
        approve_run=approve_run,
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
        source = analysis.get("analysis_source")
        csv_path = analysis.get("csv_path")
        if source == "existing_csv" or (run and run.get("skipped")):
            print(
                "Source:   existing CSV "
                "(REINVENT was not executed — not a fresh generation)"
            )
            if csv_path:
                print(f"CSV:      {csv_path}")
        elif source == "fresh_run":
            print("Source:   fresh REINVENT run")
            if csv_path:
                print(f"CSV:      {csv_path}")
        elif csv_path:
            print(f"CSV:      {csv_path}")
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

    try:
        verdict = _review_with_critic(args, agent_config, results)
    except LLMCriticError as exc:
        print(f"\nERROR: {exc}")
        logger.error("LLM critic failed without fallback: %s", exc)
        return 2

    results["steps"]["critic_review"] = verdict

    print("\n[Critic Agent]")
    critic_name = verdict.get("critic") or "deterministic"
    print(f"Critic: {critic_name}")
    if verdict.get("critic_fallback"):
        print("WARNING: LLM critic failed — using deterministic critic (fallback).")
        for warning in verdict.get("critic_warnings") or []:
            print(f"WARNING: {warning}")
    print(f"Status: {verdict['status']}")
    for issue in verdict.get("issues") or []:
        print(f"  - {issue}")
    print(f"Recommendation: {verdict['recommendation']}")

    report_meta = executor.write_report(results, goal=args.goal, critic=verdict)
    print("\n[Report]")
    print(f"Wrote: {report_meta.get('report_path')}")

    exit_code = _exit_code_for_critic(str(verdict.get("status", "FAIL")))
    result_path = write_run_result(
        run_dir,
        goal=args.goal,
        plan=plan,
        results=results,
        critic=verdict,
        report=report_meta,
        exit_code=exit_code,
    )
    summary_path = logs_dir / "last_run_summary.json"
    summary_path.write_text(
        dumps_pretty(
            {
                "run_dir": str(run_dir),
                "result_json": str(result_path),
                "results": results,
                "critic": verdict,
            }
        ),
        encoding="utf-8",
    )
    logger.info("Wrote run artefact %s", result_path)
    logger.info("Wrote summary %s", summary_path)
    print(f"Result: {result_path}")

    _banner("DONE")
    return exit_code


def _create_plan(
    args: argparse.Namespace,
    agent_config: dict,
    *,
    project_dir: str,
    approve_run: bool,
) -> dict:
    planner_cfg = agent_config.get("planner") or {}
    mode = args.planner or planner_cfg.get("mode") or "deterministic"
    mode = str(mode).strip().lower()
    if mode not in ("deterministic", "llm"):
        raise LLMPlannerError(
            f"Unknown planner mode {mode!r}; use deterministic or llm"
        )

    if mode == "llm":
        planner: PlannerAgent | LLMPlannerAgent = LLMPlannerAgent(
            agent_config=overlay_provider(
                agent_config, getattr(args, "provider", None)
            )
        )
    else:
        planner = PlannerAgent()

    return planner.create_plan(
        args.goal,
        project_dir=project_dir,
        approve_run=approve_run,
        skip_reinvent=args.skip_reinvent,
    )


def _review_with_critic(
    args: argparse.Namespace,
    agent_config: dict,
    results: dict,
) -> dict:
    critic_cfg = agent_config.get("critic") or {}
    mode = args.critic or critic_cfg.get("mode") or "deterministic"
    mode = str(mode).strip().lower()
    if mode not in ("deterministic", "llm"):
        raise LLMCriticError(
            f"Unknown critic mode {mode!r}; use deterministic or llm"
        )

    if mode == "llm":
        critic: CriticAgent | LLMCriticAgent = LLMCriticAgent(
            agent_config=overlay_provider(
                agent_config, getattr(args, "provider", None)
            )
        )
    else:
        critic = CriticAgent(agent_config=agent_config)

    return critic.review(results)


if __name__ == "__main__":
    raise SystemExit(main())
