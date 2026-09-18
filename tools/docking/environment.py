"""Docking-only environment probe — never installs, never used by REINVENT."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from typing import Any


def check_docking_environment() -> dict[str, Any]:
    """Return structured docking-tool facts (JSON-compatible).

    Separate from ``tools.environment.check_environment`` so vina/gnina are
    not mixed into the REINVENT executor env check. Never auto-installs.
    """
    vina_path = shutil.which("vina")
    gnina_path = shutil.which("gnina")
    obabel_path = shutil.which("obabel") or shutil.which("obabel3")

    rdkit_ok = False
    rdkit_version: str | None = None
    rdkit_error: str | None = None
    try:
        from rdkit import rdBase  # noqa: F401

        rdkit_ok = True
        rdkit_version = getattr(rdBase, "rdkitVersion", None) or "unknown"
    except Exception as exc:  # noqa: BLE001 — report, do not crash caller
        rdkit_error = str(exc)

    meeko_ok = False
    meeko_error: str | None = None
    try:
        import meeko  # noqa: F401

        meeko_ok = True
    except Exception as exc:  # noqa: BLE001
        meeko_error = str(exc)

    vina_version = _safe_version([vina_path, "--version"]) if vina_path else None
    gnina_path_version = _safe_version([gnina_path, "--version"]) if gnina_path else None
    obabel_version = _safe_version([obabel_path, "-V"]) if obabel_path else None

    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "vina": vina_path is not None,
        "vina_path": vina_path,
        "vina_version": vina_version,
        "gnina": gnina_path is not None,
        "gnina_path": gnina_path,
        "gnina_version": gnina_path_version,
        "obabel": obabel_path is not None,
        "obabel_path": obabel_path,
        "obabel_version": obabel_version,
        "meeko": meeko_ok,
        "meeko_error": None if meeko_ok else meeko_error,
        "rdkit": rdkit_ok,
        "rdkit_version": rdkit_version,
        "rdkit_error": rdkit_error,
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
    print(json.dumps(check_docking_environment(), indent=2))


if __name__ == "__main__":
    main()
