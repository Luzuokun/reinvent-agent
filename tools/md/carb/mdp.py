"""Human-written carb mdp templates + allowlisted scalar overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.md.carb.constants import (
    ALLOWED_PROTOCOLS,
    DEFAULT_PROTOCOL,
    MAX_LAUNCH_NSTEPS_200NS,
    MAX_LAUNCH_NSTEPS_SHORT,
    MDP_DIR,
    PRODUCTION_NSTEPS,
    PRODUCTION_PROTOCOL,
    REPLICA_SEEDS,
)
from tools.md.mdp import (
    FORBIDDEN_OVERRIDE_KEYS,
    MdpError,
    canonicalize_override_key,
    materialize_mdp as _materialize_mdp_generic,
    normalize_overrides,
    parse_mdp_nsteps,
    replace_mdp_scalar,
    template_has_key,
)

CARB_TEMPLATE_IDS: tuple[str, ...] = ("em", "nvt", "npt", "md_200ns", "ions")

PROTOCOL_TEMPLATES: dict[str, tuple[str, ...]] = {
    DEFAULT_PROTOCOL: ("em", "nvt", "npt"),
    PRODUCTION_PROTOCOL: ("em", "nvt", "npt", "md_200ns"),
}

_FILENAME = {
    "em": "em.mdp",
    "nvt": "nvt.mdp",
    "npt": "npt.mdp",
    "md_200ns": "md_200ns.mdp",
    "ions": "ions.mdp",
}


def normalize_carb_protocol(protocol: str) -> str:
    if not isinstance(protocol, str) or not protocol.strip():
        raise MdpError(
            f"protocol is required ({DEFAULT_PROTOCOL} or {PRODUCTION_PROTOCOL})"
        )
    name = protocol.strip().lower().replace("_", "-")
    aliases = {
        "short": DEFAULT_PROTOCOL,
        "em-nvt-npt": DEFAULT_PROTOCOL,
        "smoke": DEFAULT_PROTOCOL,
        "production": PRODUCTION_PROTOCOL,
        "production-200ns": PRODUCTION_PROTOCOL,
        "200ns": PRODUCTION_PROTOCOL,
        "md-200ns": PRODUCTION_PROTOCOL,
    }
    name = aliases.get(name, name)
    if name not in ALLOWED_PROTOCOLS:
        raise MdpError(
            f"Unknown carb protocol {protocol!r}; allowed: {list(ALLOWED_PROTOCOLS)}"
        )
    return name


def carb_template_path(template_id: str) -> Path:
    name = template_id.strip().lower()
    if name not in _FILENAME:
        raise MdpError(
            f"Unknown carb mdp template {template_id!r}; allowed: {list(CARB_TEMPLATE_IDS)}"
        )
    path = (MDP_DIR / _FILENAME[name]).resolve()
    if not path.is_file():
        raise MdpError(f"Human-written carb mdp missing: {path}")
    try:
        path.relative_to(MDP_DIR.resolve())
    except ValueError as exc:
        raise MdpError(f"carb mdp escapes protocol dir: {path}") from exc
    return path


def launch_nsteps_cap(protocol: str) -> int:
    name = normalize_carb_protocol(protocol)
    if name == PRODUCTION_PROTOCOL:
        return MAX_LAUNCH_NSTEPS_200NS
    return MAX_LAUNCH_NSTEPS_SHORT


def assert_carb_launch_nsteps(nsteps: int | None, protocol: str) -> int:
    if nsteps is None:
        raise MdpError("materialized carb mdp has no nsteps")
    cap = launch_nsteps_cap(protocol)
    name = normalize_carb_protocol(protocol)
    if nsteps > cap:
        raise MdpError(
            f"nsteps={nsteps} exceeds carb launch cap ({cap}) for {name}. "
            f"Default protocol is {DEFAULT_PROTOCOL}; {PRODUCTION_PROTOCOL} "
            f"({PRODUCTION_NSTEPS} steps = 200 ns) must be selected explicitly."
        )
    return int(nsteps)


def materialize_carb_mdp(
    template_id: str,
    dest: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
    gen_seed: int | None = None,
) -> dict[str, Any]:
    """Copy a human carb mdp and apply nsteps/dt/ref_t only.

    ``gen_seed`` is a frozen replica table value, not a user/LLM override.
    """
    source = carb_template_path(template_id)
    dest_path = Path(dest).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    mdp_overrides = {
        key: value
        for key, value in normalize_overrides(overrides).items()
        if key in {"nsteps", "dt", "ref_t"}
    }
    applied: dict[str, int | float] = {}
    skipped: dict[str, str] = {}
    for key, value in mdp_overrides.items():
        if not template_has_key(text, key):
            skipped[key] = "key absent from this template"
            continue
        text = replace_mdp_scalar(text, key, value)
        applied[key] = value
    if gen_seed is not None:
        text = _replace_unlisted_int(text, "gen_seed", int(gen_seed))
        applied["gen_seed"] = int(gen_seed)
    dest_path.write_text(text, encoding="utf-8")
    return {
        "ok": True,
        "template_id": template_id,
        "source": str(source),
        "dest": str(dest_path),
        "overrides": applied,
        "skipped_overrides": skipped,
        "nsteps": parse_mdp_nsteps(text),
    }


def materialize_carb_protocol(
    protocol: str,
    dest_dir: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
    replica_index: int = 0,
) -> dict[str, Any]:
    protocol = normalize_carb_protocol(protocol)
    dest_root = Path(dest_dir).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    seed = REPLICA_SEEDS[replica_index % len(REPLICA_SEEDS)]
    files: list[dict[str, Any]] = []
    for template_id in PROTOCOL_TEMPLATES[protocol]:
        dest = dest_root / f"{template_id}.mdp"
        gen_seed = seed if template_id == "nvt" else None
        files.append(
            materialize_carb_mdp(
                template_id, dest, overrides=overrides, gen_seed=gen_seed
            )
        )
    nsteps_values = [int(item["nsteps"]) for item in files if item.get("nsteps")]
    return {
        "ok": True,
        "protocol": protocol,
        "runnable": True,
        "files": files,
        "overrides": normalize_overrides(overrides) if overrides else {},
        "max_nsteps": max(nsteps_values) if nsteps_values else None,
        "replica_index": replica_index,
        "gen_seed": seed,
    }


def _replace_unlisted_int(text: str, key: str, value: int) -> str:
    """Replace a frozen protocol integer (replica seed), never from LLM keys."""
    if key in FORBIDDEN_OVERRIDE_KEYS and key != "gen_vel":
        # gen_seed is not in the forbidden integrator/cutoff set.
        pass
    folded = key.replace("-", "_")
    if folded in {"integrator", "constraints", "coulombtype"}:
        raise MdpError(f"refusing to rewrite {key}")
    found = False
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.split(";", 1)[0]
        if "=" in stripped and stripped.split("=", 1)[0].strip().lower().replace("-", "_") == folded:
            comment = ""
            if ";" in line:
                comment = line[line.index(";") :]
            pre, _eq, _rest = line.partition("=")
            lines.append(f"{pre}= {int(value)}{(' ' + comment) if comment else ''}".rstrip())
            found = True
            continue
        lines.append(line)
    if not found:
        return text
    trailing = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + trailing


# Re-export names tests may want.
__all__ = [
    "MdpError",
    "PROTOCOL_TEMPLATES",
    "assert_carb_launch_nsteps",
    "canonicalize_override_key",
    "launch_nsteps_cap",
    "materialize_carb_mdp",
    "materialize_carb_protocol",
    "normalize_carb_protocol",
    "parse_mdp_nsteps",
    "_materialize_mdp_generic",
]
