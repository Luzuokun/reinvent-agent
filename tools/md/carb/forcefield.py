"""Pinned CHARMM36 fetch (URL + sha256). Offline-ok when the cache is present."""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from tools.md.carb.constants import FF_ROOT, PIN_PATH

CHUNK = 1024 * 1024
FETCH_TIMEOUT_SECONDS = 120
USER_AGENT = "reinvent-agent-carb-charmm36/1.0"


class ForcefieldError(ValueError):
    """Pin missing, checksum mismatch, or fetch failure."""


Opener = Callable[..., Any]


def load_pin(path: Path | None = None) -> dict[str, Any]:
    pin_path = Path(path) if path is not None else PIN_PATH
    if not pin_path.is_file():
        raise ForcefieldError(f"CHARMM36 pin file missing: {pin_path}")
    data = json.loads(pin_path.read_text(encoding="utf-8"))
    for key in ("url", "sha256", "filename", "extracts_to"):
        if not data.get(key):
            raise ForcefieldError(f"CHARMM36 pin missing {key}: {pin_path}")
    sha = str(data["sha256"]).strip().lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ForcefieldError(f"CHARMM36 pin sha256 is not 64 hex chars: {sha!r}")
    data["sha256"] = sha
    data["pin_path"] = str(pin_path)
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def extracted_ff_dir(pin: dict[str, Any] | None = None, *, root: Path | None = None) -> Path:
    pin = pin or load_pin()
    dest = Path(root) if root is not None else FF_ROOT
    return dest / str(pin["extracts_to"])


def alias_ff_dir(pin: dict[str, Any] | None = None, *, root: Path | None = None) -> Path:
    pin = pin or load_pin()
    dest = Path(root) if root is not None else FF_ROOT
    return dest / str(pin.get("alias") or "charmm36.ff")


def tarball_path(pin: dict[str, Any] | None = None, *, root: Path | None = None) -> Path:
    pin = pin or load_pin()
    dest = Path(root) if root is not None else FF_ROOT
    return dest / str(pin["filename"])


def forcefield_ready(pin: dict[str, Any] | None = None, *, root: Path | None = None) -> dict[str, Any]:
    pin = pin or load_pin()
    ff_dir = extracted_ff_dir(pin, root=root)
    missing = [
        name
        for name in pin.get("required_files") or ["forcefield.itp"]
        if not (ff_dir / name).is_file()
    ]
    ok = ff_dir.is_dir() and not missing
    return {
        "ok": ok,
        "path": str(ff_dir),
        "missing": missing,
        "offline": ok,
    }


def ensure_forcefield(
    *,
    fetch: bool = True,
    root: Path | None = None,
    opener: Opener | None = None,
) -> dict[str, Any]:
    """Return a ready CHARMM36 tree, fetching only when the cache is missing.

    Checksum mismatch is a loud failure — no silent URL fallback.
    """
    pin = load_pin()
    dest = Path(root) if root is not None else FF_ROOT
    dest.mkdir(parents=True, exist_ok=True)
    status = forcefield_ready(pin, root=dest)
    if status["ok"]:
        _ensure_alias(pin, root=dest)
        return {
            "ok": True,
            "fetched": False,
            "offline": True,
            "path": status["path"],
            "alias": str(alias_ff_dir(pin, root=dest)),
            "pin": {k: pin[k] for k in ("name", "url", "sha256", "extracts_to")},
            "message": "CHARMM36 cache present; skipped network fetch.",
        }
    archive = tarball_path(pin, root=dest)
    if archive.is_file():
        _verify_and_extract(archive, pin, dest)
        _ensure_alias(pin, root=dest)
        ready = forcefield_ready(pin, root=dest)
        if not ready["ok"]:
            raise ForcefieldError(
                f"CHARMM36 extract missing files {ready['missing']} under {ready['path']}"
            )
        return {
            "ok": True,
            "fetched": False,
            "offline": True,
            "path": ready["path"],
            "alias": str(alias_ff_dir(pin, root=dest)),
            "pin": {k: pin[k] for k in ("name", "url", "sha256", "extracts_to")},
            "message": "CHARMM36 tarball cache verified and extracted; skipped network fetch.",
        }
    if not fetch:
        raise ForcefieldError(
            "CHARMM36 is not cached. Re-run doctor/prepare with network "
            f"or place the pinned tarball at {archive}"
        )
    _download(pin["url"], archive, opener=opener)
    _verify_and_extract(archive, pin, dest)
    _ensure_alias(pin, root=dest)
    ready = forcefield_ready(pin, root=dest)
    if not ready["ok"]:
        raise ForcefieldError(
            f"CHARMM36 extract missing files {ready['missing']} under {ready['path']}"
        )
    return {
        "ok": True,
        "fetched": True,
        "offline": False,
        "path": ready["path"],
        "alias": str(alias_ff_dir(pin, root=dest)),
        "pin": {k: pin[k] for k in ("name", "url", "sha256", "extracts_to")},
        "message": f"Fetched pinned CHARMM36 from {pin['url']} (sha256 ok).",
    }


def _download(url: str, dest: Path, *, opener: Opener | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        if opener is not None:
            resp_cm = opener(request, timeout=FETCH_TIMEOUT_SECONDS)
        else:
            resp_cm = urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS)
        with resp_cm as resp, tmp.open("wb") as fh:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if tmp.exists():
            tmp.unlink()
        raise ForcefieldError(
            f"CHARMM36 fetch failed from pinned URL {url}: {exc}. "
            "No fallback source. Place the tarball in forcefields/charmm36/ to work offline."
        ) from exc
    tmp.replace(dest)


def _verify_and_extract(archive: Path, pin: dict[str, Any], dest: Path) -> None:
    digest = sha256_file(archive)
    expected = pin["sha256"]
    if digest != expected:
        raise ForcefieldError(
            f"CHARMM36 sha256 mismatch for {archive}: got {digest}, "
            f"expected {expected}. Refusing to extract (no silent fallback)."
        )
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:gz") as tf:
            kwargs = {"path": dest}
            # Python 3.12+ data filter; older versions only accept path=.
            try:
                tf.extractall(filter="data", **kwargs)
            except TypeError:
                tf.extractall(**kwargs)
    except (tarfile.TarError, OSError) as exc:
        raise ForcefieldError(f"CHARMM36 tarball extract failed: {exc}") from exc


def _ensure_alias(pin: dict[str, Any], *, root: Path) -> None:
    versioned = extracted_ff_dir(pin, root=root)
    alias = alias_ff_dir(pin, root=root)
    if alias.exists() or alias.is_symlink():
        return
    try:
        alias.symlink_to(versioned.name)
    except OSError:
        # Last resort: callers can copy the versioned directory themselves.
        pass
