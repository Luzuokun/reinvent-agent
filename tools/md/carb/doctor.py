"""Environment + template + pinned CHARMM36 checks. Never auto-installs gmx."""

from __future__ import annotations

import shutil
from typing import Any

from tools import REPO_ROOT
from tools.md.carb.constants import DISK_FREE_MIN_BYTES, FF_ROOT, PROTOCOL_ID
from tools.md.carb.forcefield import ForcefieldError, ensure_forcefield, forcefield_ready, load_pin
from tools.md.carb.templates import CarbTemplateError, validate_bundled_templates
from tools.md.environment import check_md_environment


def _gpu_probe() -> dict[str, Any]:
    nvidia = shutil.which("nvidia-smi")
    return {
        "nvidia_smi": nvidia is not None,
        "nvidia_smi_path": nvidia,
        "required": False,
        "note": "GPU is optional; default mdrun is CPU. Pass --gpu to opt in.",
    }


def _disk_probe() -> dict[str, Any]:
    usage = shutil.disk_usage(REPO_ROOT)
    return {
        "ok": usage.free >= DISK_FREE_MIN_BYTES,
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
        "minimum_bytes": DISK_FREE_MIN_BYTES,
    }


def doctor(*, fetch_forcefield: bool = True) -> dict[str, Any]:
    env = check_md_environment()
    gpu = _gpu_probe()
    disk = _disk_probe()
    errors: list[str] = []
    warnings: list[str] = []

    templates: dict[str, Any]
    try:
        templates = validate_bundled_templates()
    except CarbTemplateError as exc:
        templates = {"ok": False, "message": str(exc)}
        errors.append(str(exc))

    pin = None
    try:
        pin = load_pin()
    except ForcefieldError as exc:
        errors.append(str(exc))

    ff: dict[str, Any]
    try:
        if fetch_forcefield:
            ff = ensure_forcefield(fetch=True)
        else:
            ready = forcefield_ready()
            if ready["ok"]:
                ff = {
                    "ok": True,
                    "fetched": False,
                    "offline": True,
                    "path": ready["path"],
                    "message": "CHARMM36 cache present.",
                }
            else:
                ff = {
                    "ok": False,
                    "fetched": False,
                    "path": ready["path"],
                    "missing": ready.get("missing"),
                    "message": (
                        "CHARMM36 not cached. doctor will fetch the pinned URL "
                        "when fetch is enabled."
                    ),
                }
                errors.append(ff["message"])
    except ForcefieldError as exc:
        ff = {"ok": False, "message": str(exc), "fetched": False}
        errors.append(str(exc))

    if not env.get("gmx"):
        warnings.append(
            "gmx not found on PATH (MD environment check never auto-installs)."
        )
    if not disk["ok"]:
        warnings.append(
            f"free disk {disk['free_bytes']} B is below {DISK_FREE_MIN_BYTES} B"
        )

    ok = bool(templates.get("ok")) and bool(ff.get("ok")) and not errors
    result = {
        "ok": ok,
        "protocol_id": PROTOCOL_ID,
        "engine": "gmx",
        "package": "gromacs",
        "kind": "carb-md-doctor",
        "environment": env,
        "gpu": gpu,
        "disk": disk,
        "templates": templates,
        "forcefield": ff,
        "pin": (
            {k: pin[k] for k in ("name", "url", "sha256", "extracts_to")}
            if pin
            else None
        ),
        "ff_root": str(FF_ROOT),
        "warnings": warnings,
        "errors": errors,
        "message": (
            "carb doctor ok"
            if ok
            else ("carb doctor failed: " + "; ".join(errors) if errors else "carb doctor failed")
        ),
        "isolated_from": "tools.md.complex.prepare_complex",
    }
    return result
