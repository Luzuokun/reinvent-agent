"""Job status file for carb MD. Failures never invent trajectories."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools import dumps_pretty

STATUSES = ("pending", "running", "failed", "completed", "skipped")


def status_path(output_dir: Path) -> Path:
    return Path(output_dir) / "job_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def replica_record(
    replica_id: int,
    *,
    seed: int,
    status: str = "pending",
    message: str = "",
    output_dir: str | None = None,
) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"unknown replica status {status}")
    return {
        "id": int(replica_id),
        "seed": int(seed),
        "status": status,
        "message": message,
        "output_dir": output_dir,
        "updated_at": utc_now(),
    }


def write_status(output_dir: Path, payload: dict[str, Any]) -> Path:
    dest = status_path(output_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["updated_at"] = utc_now()
    dest.write_text(dumps_pretty(body), encoding="utf-8")
    return dest


def read_status(output_dir: Path) -> dict[str, Any] | None:
    dest = status_path(output_dir)
    if not dest.is_file():
        return None
    try:
        data = json.loads(dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def mark(
    output_dir: Path,
    *,
    status: str,
    protocol: str,
    replicas: list[dict[str, Any]],
    message: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"unknown job status {status}")
    payload: dict[str, Any] = {
        "status": status,
        "protocol": protocol,
        "replicas": replicas,
        "message": message,
        "invented_trajectory": False,
    }
    if extra:
        payload.update(extra)
    write_status(output_dir, payload)
    return payload
