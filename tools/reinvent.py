"""Predefined REINVENT4 runner — never accepts arbitrary shell."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools import REPO_ROOT, load_agent_config, resolve_under_root


def prepare_reinvent_command(
    config_path: str | Path,
    *,
    seed: int | None = None,
    project_dir: str | Path | None = None,
    logs_dir: str | Path | None = None,
    agent_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the predefined REINVENT argv without executing it."""
    cfg = agent_config or load_agent_config()
    executable = cfg.get("reinvent", {}).get("executable", "reinvent")
    default_seed = int(cfg.get("reinvent", {}).get("default_seed", 42))
    seed = default_seed if seed is None else int(seed)

    config_path = Path(config_path).expanduser().resolve()
    if project_dir is None:
        project_dir = config_path.parent
    project_dir = Path(project_dir).expanduser().resolve()

    try:
        resolve_under_root(project_dir, config_path.relative_to(project_dir))
    except ValueError:
        return {
            "ok": False,
            "command": [],
            "cwd": str(project_dir),
            "config_path": str(config_path),
            "seed": seed,
            "message": f"Config path is outside project directory: {config_path}",
        }

    if logs_dir is None:
        logs_dir = REPO_ROOT / cfg.get("logging", {}).get("logs_dirname", "logs")
    logs_dir = Path(logs_dir).expanduser().resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"reinvent_{stamp}.log"
    stdout_path = logs_dir / f"reinvent_{stamp}.stdout.log"
    stderr_path = logs_dir / f"reinvent_{stamp}.stderr.log"

    reinvent_bin = shutil.which(executable)
    command = [
        reinvent_bin or executable,
        "-l",
        str(log_path),
        "-s",
        str(seed),
        str(config_path),
    ]
    return {
        "ok": True,
        "command": command,
        "cwd": str(project_dir),
        "config_path": str(config_path),
        "seed": seed,
        "log_path": str(log_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "reinvent_on_path": reinvent_bin is not None,
        "executable": reinvent_bin or executable,
    }


def confirm_reinvent_launch(
    prepared: dict[str, Any],
    *,
    assume_yes: bool = False,
    stdin=None,
    stdout=None,
) -> bool:
    """Print the predefined command and require interactive confirmation.

    Non-interactive sessions must pass ``assume_yes=True`` (CLI ``--yes``).
    """
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    command = prepared.get("command") or []
    cwd = prepared.get("cwd", "")
    print("\n[Human approval required]", file=stdout)
    print("The agent is ready to execute this predefined command:", file=stdout)
    print(f"  cwd: {cwd}", file=stdout)
    print(f"  cmd: {' '.join(str(c) for c in command)}", file=stdout)
    print("Estimated task: molecular sampling (CPU/GPU depending on TOML).", file=stdout)
    if assume_yes:
        print("Proceeding due to --yes.", file=stdout)
        return True
    if not hasattr(stdin, "isatty") or not stdin.isatty():
        print(
            "Non-interactive stdin: re-run with --approve-run --yes to launch.",
            file=stdout,
        )
        return False
    print("Proceed? [y/N]: ", end="", file=stdout, flush=True)
    reply = stdin.readline().strip().lower()
    return reply in ("y", "yes")


def run_reinvent(
    config_path: str | Path,
    *,
    approve: bool,
    seed: int | None = None,
    project_dir: str | Path | None = None,
    logs_dir: str | Path | None = None,
    agent_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a validated ``reinvent`` command, or prepare without executing.

    Requires ``approve=True`` (CLI ``--approve-run``) to launch the job.
    """
    prepared_cmd = prepare_reinvent_command(
        config_path,
        seed=seed,
        project_dir=project_dir,
        logs_dir=logs_dir,
        agent_config=agent_config,
    )
    if not prepared_cmd.get("ok"):
        return {
            "approved": bool(approve),
            "skipped": False,
            "success": False,
            "exit_code": 2,
            "runtime_seconds": 0.0,
            "config_path": prepared_cmd.get("config_path"),
            "cwd": prepared_cmd.get("cwd"),
            "command": prepared_cmd.get("command") or [],
            "message": prepared_cmd.get("message", "Failed to prepare REINVENT command"),
        }

    prepared = {
        "approved": bool(approve),
        "command": prepared_cmd["command"],
        "cwd": prepared_cmd["cwd"],
        "config_path": prepared_cmd["config_path"],
        "seed": prepared_cmd["seed"],
        "log_path": prepared_cmd["log_path"],
        "stdout_path": prepared_cmd["stdout_path"],
        "stderr_path": prepared_cmd["stderr_path"],
        "reinvent_on_path": prepared_cmd["reinvent_on_path"],
    }

    if not approve:
        return {
            **prepared,
            "skipped": True,
            "success": False,
            "exit_code": None,
            "runtime_seconds": 0.0,
            "message": "REINVENT not launched (missing --approve-run).",
        }

    config_path = Path(prepared_cmd["config_path"])
    project_dir = Path(prepared_cmd["cwd"])
    command = prepared_cmd["command"]
    stdout_path = Path(prepared_cmd["stdout_path"])
    stderr_path = Path(prepared_cmd["stderr_path"])

    if not prepared_cmd["reinvent_on_path"]:
        return {
            **prepared,
            "skipped": False,
            "success": False,
            "exit_code": 127,
            "runtime_seconds": 0.0,
            "message": f"Executable not found on PATH: {prepared_cmd.get('executable')}",
        }

    if not config_path.is_file():
        return {
            **prepared,
            "skipped": False,
            "success": False,
            "exit_code": 2,
            "runtime_seconds": 0.0,
            "message": f"Config not found: {config_path}",
        }

    started = time.perf_counter()
    try:
        with stdout_path.open("w", encoding="utf-8") as out_fh, stderr_path.open(
            "w", encoding="utf-8"
        ) as err_fh:
            proc = subprocess.run(
                command,
                cwd=str(project_dir),
                check=False,
                stdout=out_fh,
                stderr=err_fh,
                text=True,
                shell=False,
            )
        runtime = time.perf_counter() - started
        success = proc.returncode == 0
        return {
            **prepared,
            "skipped": False,
            "success": success,
            "exit_code": proc.returncode,
            "runtime_seconds": round(runtime, 3),
            "message": "REINVENT finished successfully."
            if success
            else f"REINVENT exited with code {proc.returncode}.",
        }
    except OSError as exc:
        runtime = time.perf_counter() - started
        return {
            **prepared,
            "skipped": False,
            "success": False,
            "exit_code": 1,
            "runtime_seconds": round(runtime, 3),
            "message": f"Failed to launch REINVENT: {exc}",
        }


def find_output_files(
    output_dir: str | Path,
    *,
    project_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Inventory files under the configured output directory only."""
    output_dir = Path(output_dir).expanduser().resolve()
    if project_dir is not None:
        project_dir = Path(project_dir).expanduser().resolve()
        resolve_under_root(project_dir, output_dir.relative_to(project_dir))

    if not output_dir.is_dir():
        return {
            "ok": False,
            "output_dir": str(output_dir),
            "files": [],
            "csv_files": [],
            "error": f"Output directory does not exist: {output_dir}",
        }

    files: list[dict[str, Any]] = []
    csv_files: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(output_dir))
        info = {
            "path": str(path),
            "relative": rel,
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
        }
        files.append(info)
        if path.suffix.lower() == ".csv":
            csv_files.append(str(path))

    return {
        "ok": True,
        "output_dir": str(output_dir),
        "files": files,
        "csv_files": csv_files,
        "count": len(files),
        "csv_count": len(csv_files),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="REINVENT runner / output finder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Prepare or run reinvent")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--approve-run", action="store_true")
    run_p.add_argument("--seed", type=int, default=None)

    find_p = sub.add_parser("find", help="List output files")
    find_p.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.cmd == "run":
        print(
            json.dumps(
                run_reinvent(args.config, approve=args.approve_run, seed=args.seed),
                indent=2,
            )
        )
    else:
        print(json.dumps(find_output_files(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
