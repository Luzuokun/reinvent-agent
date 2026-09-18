"""Docking pipeline + CLI — separate human approval from REINVENT.

``python -m tools.docking --approve-dock`` is the only way this module launches
Vina/GNINA. The REINVENT executor never imports this package.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence, TextIO

from tools import dumps_pretty, load_agent_config
from tools.docking.engines import (
    ALLOWED_ENGINES,
    MAX_BOX_SIZE,
    MAX_EXHAUSTIVENESS,
    MAX_LIGANDS,
    MAX_NUM_MODES,
    build_docking_command,
    clamp_exhaustiveness,
    clamp_max_ligands,
    clamp_num_modes,
    executable_name,
    normalize_engine,
    parse_engine_log,
    validate_box,
)
from tools.docking.environment import check_docking_environment
from tools.docking.prepare import (
    DockingPrepareError,
    load_ligand_records,
    prepare_ligand,
    prepare_receptor,
    resolve_ligands_path,
    resolve_receptor_path,
)

OUTPUT_SUBDIR = "docking"


class DockingError(ValueError):
    """Sandbox, approval, or engine failure."""


def run_docking(
    project_dir: str | Path,
    *,
    receptor: str | Path,
    ligands: str | Path,
    engine: str = "vina",
    center: Sequence[float],
    size: Sequence[float],
    approve: bool = False,
    assume_yes: bool = False,
    exhaustiveness: int | None = None,
    num_modes: int | None = None,
    max_ligands: int | None = None,
    agent_config: dict[str, Any] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> dict[str, Any]:
    """Validate, optionally prepare + dock, and write ``output/docking/scores.csv``."""
    stdout = sys.stdout if stdout is None else stdout
    cfg = (agent_config or {}).get("docking") or {}
    engine = normalize_engine(engine)
    exhaustiveness = clamp_exhaustiveness(
        exhaustiveness, default=int(cfg.get("default_exhaustiveness", 8))
    )
    num_modes = clamp_num_modes(
        num_modes, default=int(cfg.get("default_num_modes", 1))
    )
    max_ligands = clamp_max_ligands(
        max_ligands, default=int(cfg.get("max_ligands", 20))
    )
    center_t, size_t = validate_box(center, size)

    project = Path(project_dir).expanduser().resolve()
    if not project.is_dir():
        raise DockingError(f"Project directory does not exist: {project}")

    try:
        receptor_path = resolve_receptor_path(project, receptor)
        ligands_path = resolve_ligands_path(project, ligands)
    except DockingPrepareError as exc:
        raise DockingError(str(exc)) from exc

    output_dir = _docking_output_dir(project)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = check_docking_environment()
    bin_name = executable_name(engine)
    engine_path = env.get(f"{engine}_path")
    planned = {
        "engine": engine,
        "executable": engine_path or bin_name,
        "receptor": str(receptor_path),
        "ligands": str(ligands_path),
        "center": list(center_t),
        "size": list(size_t),
        "exhaustiveness": exhaustiveness,
        "num_modes": num_modes,
        "max_ligands": max_ligands,
        "output_dir": str(output_dir),
    }

    result: dict[str, Any] = {
        "ok": False,
        "approved": bool(approve),
        "skipped": False,
        "success": False,
        "engine": engine,
        "project_dir": str(project),
        "output_dir": str(output_dir),
        "scores_csv": None,
        "table_present": False,
        "n_ligands": 0,
        "n_scored": 0,
        "score": None,
        "rows": [],
        "environment": env,
        "planned": planned,
        "command_template": [],
        "warnings": [],
        "errors": [],
        "message": "",
    }

    try:
        records = load_ligand_records(ligands_path, max_ligands=max_ligands)
    except DockingPrepareError as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        _write_result_json(output_dir, result)
        return result
    result["n_ligands"] = len(records)
    if len(records) >= max_ligands:
        result["warnings"].append(
            f"Ligand list truncated to max_ligands={max_ligands}"
        )

    dummy_ligand = output_dir / "prepared" / "ligand.pdbqt"
    dummy_out = output_dir / "poses" / "ligand_out.pdbqt"
    try:
        result["command_template"] = build_docking_command(
            engine=engine,
            executable=engine_path or bin_name,
            receptor=output_dir / "prepared" / "receptor.pdbqt",
            ligand=dummy_ligand,
            out=dummy_out,
            center=center_t,
            size=size_t,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
        )
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        _write_result_json(output_dir, result)
        return result

    if not approve:
        result["skipped"] = True
        result["ok"] = True
        result["message"] = (
            "Docking not launched (missing --approve-dock). "
            "Environment and paths were checked; Vina/GNINA was not run."
        )
        _write_result_json(output_dir, result)
        return result

    if not confirm_docking_launch(
        planned, assume_yes=assume_yes, stdin=stdin, stdout=stdout
    ):
        result["skipped"] = True
        result["approved"] = False
        result["ok"] = True
        result["message"] = (
            "Human declined or confirmation unavailable — docking will not run."
        )
        _write_result_json(output_dir, result)
        return result

    if not engine_path:
        result["errors"].append(
            f"{bin_name} not found on PATH (docking environment check is separate "
            "from REINVENT; this module does not auto-install)."
        )
        result["message"] = result["errors"][-1]
        _write_result_json(output_dir, result)
        return result

    prepared_dir = output_dir / "prepared"
    poses_dir = output_dir / "poses"
    logs_dir = output_dir / "logs"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        receptor_meta = prepare_receptor(
            receptor_path,
            prepared_dir,
            obabel_path=env.get("obabel_path"),
        )
    except DockingPrepareError as exc:
        result["errors"].append(str(exc))
        result["message"] = str(exc)
        _write_result_json(output_dir, result)
        return result
    result["receptor"] = receptor_meta

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for record in records:
        row = _dock_one(
            record,
            engine=engine,
            engine_path=str(engine_path),
            receptor_pdbqt=Path(receptor_meta["pdbqt"]),
            prepared_dir=prepared_dir,
            poses_dir=poses_dir,
            logs_dir=logs_dir,
            center=center_t,
            size=size_t,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
            obabel_path=env.get("obabel_path"),
            meeko_available=bool(env.get("meeko")),
        )
        rows.append(row)
    runtime = time.perf_counter() - started

    scored = [r for r in rows if r.get("status") == "ok" and r.get("score") is not None]
    scores = [float(r["score"]) for r in scored]
    scores_csv = output_dir / "scores.csv"
    _write_scores_csv(scores_csv, rows)
    result["rows"] = rows
    result["n_scored"] = len(scored)
    result["scores_csv"] = str(scores_csv)
    result["table_present"] = True
    result["runtime_seconds"] = round(runtime, 3)
    result["score"] = _score_stats(scores)
    result["ok"] = True
    result["success"] = len(scored) > 0
    if not scored:
        result["message"] = "Docking finished but no poses were scored."
        result["errors"].append(result["message"])
    else:
        result["message"] = (
            f"Wrote {len(scored)} docking score(s) to {scores_csv}"
        )
    _write_result_json(output_dir, result)
    return result


def confirm_docking_launch(
    planned: dict[str, Any],
    *,
    assume_yes: bool = False,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> bool:
    """Print the predefined docking job and require interactive confirmation."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    print("\n[Human approval required — docking]", file=stdout)
    print("The agent is ready to run this predefined docking job:", file=stdout)
    print(f"  engine:  {planned.get('engine')} ({planned.get('executable')})", file=stdout)
    print(f"  receptor:{planned.get('receptor')}", file=stdout)
    print(f"  ligands: {planned.get('ligands')}", file=stdout)
    print(f"  box:     center={planned.get('center')} size={planned.get('size')}", file=stdout)
    print(f"  exhaustiveness: {planned.get('exhaustiveness')}", file=stdout)
    print(f"  output:  {planned.get('output_dir')}", file=stdout)
    print(
        "This is independent of REINVENT (--approve-run does not apply here).",
        file=stdout,
    )
    if assume_yes:
        print("Proceeding due to --yes.", file=stdout)
        return True
    if not hasattr(stdin, "isatty") or not stdin.isatty():
        print(
            "Non-interactive stdin: re-run with --approve-dock --yes to launch.",
            file=stdout,
        )
        return False
    print("Proceed? [y/N]: ", end="", file=stdout, flush=True)
    reply = stdin.readline().strip().lower()
    return reply in ("y", "yes")


def _dock_one(
    record: dict[str, Any],
    *,
    engine: str,
    engine_path: str,
    receptor_pdbqt: Path,
    prepared_dir: Path,
    poses_dir: Path,
    logs_dir: Path,
    center: Sequence[float],
    size: Sequence[float],
    exhaustiveness: int,
    num_modes: int,
    obabel_path: str | None,
    meeko_available: bool,
) -> dict[str, Any]:
    ligand_id = str(record.get("ligand_id") or "lig")
    pose_path = poses_dir / f"{ligand_id}_out.pdbqt"
    log_path = logs_dir / f"{ligand_id}.log"
    row: dict[str, Any] = {
        "ligand_id": ligand_id,
        "smiles": record.get("smiles"),
        "engine": engine,
        "score": None,
        "cnn_score": None,
        "mode": None,
        "pose_path": str(pose_path),
        "log_path": str(log_path),
        "status": "error",
        "message": "",
        "command": [],
    }
    try:
        prepared = prepare_ligand(
            record,
            prepared_dir,
            obabel_path=obabel_path,
            meeko_available=meeko_available,
        )
        ligand_pdbqt = Path(prepared["pdbqt"])
        command = build_docking_command(
            engine=engine,
            executable=engine_path,
            receptor=receptor_pdbqt,
            ligand=ligand_pdbqt,
            out=pose_path,
            center=center,
            size=size,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
        )
        row["command"] = command
        with log_path.open("w", encoding="utf-8") as log_fh:
            proc = subprocess.run(
                command,
                check=False,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
                shell=False,
            )
        text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        modes = parse_engine_log(text, engine=engine)
        if proc.returncode != 0 and not modes:
            row["message"] = f"{engine} exited {proc.returncode}"
            return row
        if not modes:
            row["message"] = f"{engine} produced no scored poses"
            return row
        best = modes[0]
        row["status"] = "ok"
        row["score"] = best.get("score")
        row["cnn_score"] = best.get("cnn_score")
        row["mode"] = best.get("mode")
        row["message"] = "ok"
        return row
    except (DockingPrepareError, OSError, subprocess.TimeoutExpired) as exc:
        row["message"] = str(exc)
        return row


def _score_stats(scores: list[float]) -> dict[str, Any] | None:
    if not scores:
        return None
    return {
        "n": len(scores),
        "mean": round(sum(scores) / len(scores), 4),
        "best": round(min(scores), 4),  # Vina: more negative is better
        "worst": round(max(scores), 4),
    }


def _write_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fields = [
        "ligand_id",
        "smiles",
        "engine",
        "score",
        "cnn_score",
        "mode",
        "pose_path",
        "log_path",
        "status",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _write_result_json(output_dir: Path, result: dict[str, Any]) -> None:
    path = output_dir / "docking_result.json"
    # Keep the JSON free of huge pose logs; rows already omit stdout dumps.
    path.write_text(dumps_pretty(result), encoding="utf-8")
    result["result_json"] = str(path)


def _docking_output_dir(project: Path) -> Path:
    dest = (project / "output" / OUTPUT_SUBDIR).resolve()
    try:
        dest.relative_to(project.resolve())
    except ValueError as exc:
        raise DockingError(f"Docking output escapes project: {dest}") from exc
    return dest


def _critic_payload(docking: dict[str, Any]) -> dict[str, Any]:
    """Build a results dict the shared Critic can review (evidence only)."""
    return {
        "plan": {
            "approve_run": False,
            "skip_reinvent": True,
            "approve_dock": docking.get("approved"),
            "steps": ["docking"],
        },
        "steps": {"docking": docking},
        "warnings": docking.get("warnings") or [],
        "errors": docking.get("errors") or [],
        "aborted": bool(docking.get("errors")) and not docking.get("skipped"),
        "abort_reason": docking.get("message") if docking.get("errors") else None,
        "project_dir": docking.get("project_dir"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independent docking module (Vina or GNINA). "
            "Not part of the REINVENT executor. Requires --approve-dock to launch."
        )
    )
    parser.add_argument("--project", required=True, help="Project directory")
    parser.add_argument(
        "--receptor",
        required=True,
        help="Existing receptor .pdbqt or .pdb under <project>/input/",
    )
    parser.add_argument(
        "--ligands",
        required=True,
        help=(
            "Existing ligands file under <project>/input/ or <project>/output/ "
            "(.pdbqt, .sdf, .smi, or CSV with a SMILES column)"
        ),
    )
    parser.add_argument(
        "--engine",
        choices=ALLOWED_ENGINES,
        default="vina",
        help="Docking engine (PATH executable vina or gnina; not a free-form path)",
    )
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Search-box center (Å)",
    )
    parser.add_argument(
        "--size",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help=f"Search-box size (Å), each axis 1–{MAX_BOX_SIZE:g}",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=None,
        help=f"Engine exhaustiveness (default 8, max {MAX_EXHAUSTIVENESS})",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=None,
        dest="num_modes",
        help=f"Poses per ligand (default 1, max {MAX_NUM_MODES})",
    )
    parser.add_argument(
        "--max-ligands",
        type=int,
        default=None,
        dest="max_ligands",
        help=f"Cap on ligands from a SMILES/CSV file (default 20, max {MAX_LIGANDS})",
    )
    parser.add_argument(
        "--approve-dock",
        action="store_true",
        help="Human approval to launch the predefined Vina/GNINA command",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive Proceed? prompt when used with --approve-dock",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional agent.yaml (docking defaults only; never TOML)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent_config: dict[str, Any] = {}
    if args.config:
        agent_config = load_agent_config(Path(args.config))
    else:
        try:
            agent_config = load_agent_config()
        except (OSError, ValueError):
            agent_config = {}

    try:
        result = run_docking(
            args.project,
            receptor=args.receptor,
            ligands=args.ligands,
            engine=args.engine,
            center=args.center,
            size=args.size,
            approve=bool(args.approve_dock),
            assume_yes=bool(args.yes),
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            max_ligands=args.max_ligands,
            agent_config=agent_config,
        )
    except DockingError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(json.dumps(_public_result(result), indent=2, ensure_ascii=False))

    from agents.critic import CriticAgent

    verdict = CriticAgent(agent_config=agent_config).review(_critic_payload(result))
    print("\n[Critic Agent]")
    print(f"Status: {verdict.get('status')}")
    for issue in verdict.get("issues") or []:
        print(f"  - {issue}")
    print(f"Recommendation: {verdict.get('recommendation')}")

    if str(verdict.get("status", "")).upper() == "FAIL":
        return 1
    return 0


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    """CLI JSON: omit per-row command argv copies beyond the template."""
    payload = dict(result)
    rows = []
    for row in result.get("rows") or []:
        slim = dict(row)
        slim.pop("command", None)
        rows.append(slim)
    payload["rows"] = rows
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
