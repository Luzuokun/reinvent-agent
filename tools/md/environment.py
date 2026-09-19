"""MD-only environment probe — never installs, never used by REINVENT."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from typing import Any

ALLOWED_GMX_BINARIES: tuple[str, ...] = ("gmx", "gmx_mpi")


def check_md_environment() -> dict[str, Any]:
    """Return structured GROMACS facts (JSON-compatible).

    Separate from ``tools.environment.check_environment`` so ``gmx`` is not
    mixed into the REINVENT executor env check. Never auto-installs.
    """
    gmx_path = None
    gmx_binary = None
    for name in ALLOWED_GMX_BINARIES:
        found = shutil.which(name)
        if found:
            gmx_path = found
            gmx_binary = name
            break

    gmx_version = _safe_version([gmx_path, "--version"]) if gmx_path else None

    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "gmx": gmx_path is not None,
        "gmx_path": gmx_path,
        "gmx_binary": gmx_binary,
        "gmx_version": gmx_version,
        "installs_packages": False,
    }


def _safe_version(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            shell=False,
        )
        text = (proc.stdout or proc.stderr or "").strip()
        return text.splitlines()[0] if text else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    print(json.dumps(check_md_environment(), indent=2))


if __name__ == "__main__":
    main()
