"""Environment inspection tool — never installs anything."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def check_environment() -> dict[str, Any]:
    """Return structured environment facts (JSON-compatible)."""
    reinvent_path = shutil.which("reinvent")
    conda_path = shutil.which("conda")

    rdkit_ok = False
    rdkit_version: str | None = None
    rdkit_error: str | None = None
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit import rdBase

        rdkit_ok = True
        rdkit_version = getattr(rdBase, "rdkitVersion", None) or "unknown"
    except Exception as exc:  # noqa: BLE001 — report, do not crash caller
        rdkit_error = str(exc)

    gpu = _probe_gpu()

    reinvent_version: str | None = None
    if reinvent_path:
        reinvent_version = _safe_version([reinvent_path, "--version"])

    result = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "conda": conda_path is not None,
        "conda_path": conda_path,
        "reinvent": reinvent_path is not None,
        "reinvent_path": reinvent_path,
        "reinvent_version": reinvent_version,
        "rdkit": rdkit_ok,
        "rdkit_version": rdkit_version,
        "rdkit_error": rdkit_error,
        "gpu": bool(gpu.get("available")),
        "gpu_details": gpu,
    }
    return result


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


def _probe_gpu() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            shell=False,
        )
        if proc.returncode != 0:
            return {
                "available": False,
                "reason": (proc.stderr or proc.stdout or "nvidia-smi failed").strip(),
            }
        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
        return {"available": bool(lines), "devices": lines}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc)}


def main() -> None:
    print(json.dumps(check_environment(), indent=2))


if __name__ == "__main__":
    main()
