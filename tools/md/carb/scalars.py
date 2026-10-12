"""Allowlisted carb scalars: mdp keys plus prepare/run knobs. No free-text mdp."""

from __future__ import annotations

import re
from typing import Any

from tools.md.carb.constants import (
    ALLOWED_OVERRIDE_KEYS,
    DEFAULT_N_REPLICAS,
    MAX_BOX_D,
    MAX_REPLICAS,
    MAX_SALT_MOL,
    MIN_BOX_D,
    MIN_REPLICAS,
    MIN_SALT_MOL,
)
from tools.md.mdp import (
    FORBIDDEN_OVERRIDE_KEYS,
    MdpError,
    parse_override_value,
)

_SCALAR = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def canonicalize_carb_key(key: str) -> str:
    if not isinstance(key, str) or not key.strip():
        raise MdpError("override key is empty")
    raw = key.strip().lower().replace(" ", "")
    folded = raw.replace("-", "_")
    if raw in FORBIDDEN_OVERRIDE_KEYS or folded in FORBIDDEN_OVERRIDE_KEYS:
        raise MdpError(
            f"carb override {key!r} is forbidden (integrator / cutoff / "
            "constraint and similar keys stay in the human template)"
        )
    aliases = {
        "nsteps": "nsteps",
        "dt": "dt",
        "ref_t": "ref_t",
        "ref-t": "ref_t",
        "salt_mol": "salt_mol",
        "salt-mol": "salt_mol",
        "conc": "salt_mol",
        "box_d": "box_d",
        "box-d": "box_d",
        "n_replicas": "n_replicas",
        "n-replicas": "n_replicas",
        "replicas": "n_replicas",
    }
    canon = aliases.get(raw) or aliases.get(folded)
    if canon is None or canon not in ALLOWED_OVERRIDE_KEYS:
        raise MdpError(
            f"carb override {key!r} is not allowlisted; "
            f"allowed scalars: {list(ALLOWED_OVERRIDE_KEYS)}"
        )
    return canon


def parse_carb_value(key: str, value: Any) -> int | float:
    canon = canonicalize_carb_key(key)
    if canon in {"nsteps", "dt", "ref_t"}:
        return parse_override_value(canon, value)
    number = _as_number(canon, value)
    if canon == "salt_mol":
        salt = float(number)
        if salt < MIN_SALT_MOL or salt > MAX_SALT_MOL:
            raise MdpError(
                f"salt_mol {salt} is outside {MIN_SALT_MOL}–{MAX_SALT_MOL} M"
            )
        return salt
    if canon == "box_d":
        box = float(number)
        if box < MIN_BOX_D or box > MAX_BOX_D:
            raise MdpError(f"box_d {box} is outside {MIN_BOX_D}–{MAX_BOX_D} nm")
        return box
    replicas = int(number)
    if replicas != number:
        raise MdpError("n_replicas override must be an integer")
    if replicas < MIN_REPLICAS or replicas > MAX_REPLICAS:
        raise MdpError(
            f"n_replicas {replicas} is outside {MIN_REPLICAS}–{MAX_REPLICAS}"
        )
    return replicas


def normalize_carb_scalars(overrides: dict[str, Any] | None) -> dict[str, int | float]:
    if not overrides:
        return {}
    if not isinstance(overrides, dict):
        raise MdpError("carb overrides must be a mapping of allowlisted scalars")
    out: dict[str, int | float] = {}
    for raw_key, raw_value in overrides.items():
        canon = canonicalize_carb_key(str(raw_key))
        out[canon] = parse_carb_value(canon, raw_value)
    return out


def mdp_only(overrides: dict[str, int | float]) -> dict[str, int | float]:
    return {k: v for k, v in overrides.items() if k in {"nsteps", "dt", "ref_t"}}


def default_n_replicas(protocol: str, requested: int | None) -> int:
    if requested is not None:
        return int(parse_carb_value("n_replicas", requested))
    if protocol == "production_200ns":
        return DEFAULT_N_REPLICAS
    return 1


def _as_number(canon: str, value: Any) -> int | float:
    if isinstance(value, bool) or value is None:
        raise MdpError(f"{canon} override must be a number, not {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if not text or not _SCALAR.fullmatch(text):
            raise MdpError(f"{canon} override must be a scalar number; got {value!r}")
        if any(ch in text for ch in ";=\n\r#"):
            raise MdpError(f"{canon} override looks like free-text mdp")
        if canon == "n_replicas":
            if "." in text or "e" in text.lower():
                raise MdpError("n_replicas override must be an integer")
            return int(text)
        return float(text)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value) if canon == "n_replicas" else float(value)
    if isinstance(value, float):
        return float(value)
    raise MdpError(f"{canon} override must be a scalar number; got {value!r}")
