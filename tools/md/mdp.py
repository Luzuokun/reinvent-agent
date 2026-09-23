"""Human-written mdp templates + allowlisted scalar overrides.

Mirrors REINVENT TOML policy: the model never emits a full mdp. Callers may
only name a checked-in template ID and optionally replace ``nsteps``, ``dt``,
and ``ref_t`` with scalars. Integrator / cutoff / constraint keys are
rejected. Production 100 ns lives in ``experiments/md.mdp`` and is not the
default smoke-test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools import REPO_ROOT

TEMPLATES_DIR = REPO_ROOT / "experiments"

ALLOWED_TEMPLATE_IDS: tuple[str, ...] = (
    "minimization",
    "nvt",
    "nvt_eq",
    "md2ns",
    "production",
    "ions",
)
ALLOWED_PROTOCOLS: tuple[str, ...] = (
    "em",
    "nvt",
    "em-nvt",
    "em-nvt-md2ns",
    "production",
)
RUNNABLE_PROTOCOLS: tuple[str, ...] = ("em", "nvt", "em-nvt", "em-nvt-md2ns")

# Default smoke-test — not 100 ns production and not the 2 ns complex protocol.
DEFAULT_PROTOCOL = "em-nvt"

ALLOWED_OVERRIDE_KEYS: tuple[str, ...] = ("nsteps", "dt", "ref_t")
FORBIDDEN_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "integrator",
        "cutoff-scheme",
        "cutoff_scheme",
        "constraints",
        "constraint_algorithm",
        "constraint-algorithm",
        "rcoulomb",
        "rvdw",
        "coulombtype",
        "coulomb-type",
        "pme_order",
        "pme-order",
        "fourierspacing",
        "tcoupl",
        "pcoupl",
        "continuation",
        "gen_vel",
        "gen-vel",
        "ns_type",
        "nstlist",
        "vdw-modifier",
        "vdw_modifier",
        "compressed-x-grps",
        "tc-grps",
        "tc_grps",
        "title",
        "include",
        "define",
    }
)

# Hard caps so a typo cannot turn smoke-test into 100 ns, and so override
# values stay scalars in a physical range.
MAX_LAUNCH_NSTEPS = 10_000  # smoke-test protocols (em / nvt / em-nvt)
MAX_LAUNCH_NSTEPS_MD2NS = 1_000_000  # 2 ns at dt=0.002 ps; still rejects 100 ns
MAX_TEMPLATE_NSTEPS = 50_000_000  # production template value; launch still capped
MIN_NSTEPS = 1
MIN_DT = 0.0001
MAX_DT = 0.004
MIN_REF_T = 200.0
MAX_REF_T = 400.0

_SCALAR = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_KEY_LINE = re.compile(
    r"^(?P<pre>\s*)(?P<key>[A-Za-z][A-Za-z0-9_-]*)(?P<eq>\s*=\s*)(?P<val>\S+)(?P<rest>.*)$"
)


class MdpError(ValueError):
    """Unknown template, illegal override, or non-scalar mdp value."""


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    filename: str
    description: str


TEMPLATE_SPECS: dict[str, TemplateSpec] = {
    "minimization": TemplateSpec(
        template_id="minimization",
        filename="minimization.mdp",
        description="Steepest-descent energy minimization (smoke-test).",
    ),
    "nvt": TemplateSpec(
        template_id="nvt",
        filename="nvt.mdp",
        description="Short NVT equilibration (smoke-test).",
    ),
    "nvt_eq": TemplateSpec(
        template_id="nvt_eq",
        filename="nvt_eq.mdp",
        description="Short NVT equilibration for a solvated complex (~50 ps).",
    ),
    "md2ns": TemplateSpec(
        template_id="md2ns",
        filename="md2ns.mdp",
        description="2 ns production after NVT (not 100 ns).",
    ),
    "production": TemplateSpec(
        template_id="production",
        filename="md.mdp",
        description="100 ns production after NPT (not the default protocol).",
    ),
    "ions": TemplateSpec(
        template_id="ions",
        filename="ions.mdp",
        description="nsteps=0 steep mdp for gmx genion only.",
    ),
}

PROTOCOL_TEMPLATES: dict[str, tuple[str, ...]] = {
    "em": ("minimization",),
    "nvt": ("nvt",),
    "em-nvt": ("minimization", "nvt"),
    "em-nvt-md2ns": ("minimization", "nvt_eq", "md2ns"),
    "production": ("production",),
}

_KEY_ALIASES = {
    "nsteps": "nsteps",
    "dt": "dt",
    "ref_t": "ref_t",
    "ref-t": "ref_t",
}


def normalize_protocol(protocol: str) -> str:
    if not isinstance(protocol, str) or not protocol.strip():
        raise MdpError(
            "protocol is required (em, nvt, em-nvt, em-nvt-md2ns, production)"
        )
    name = protocol.strip().lower().replace("_", "-")
    if name in {"em+nvt", "emin-nvt", "min-nvt"}:
        name = "em-nvt"
    if name in {"min", "minimization", "em.mdp"}:
        name = "em"
    if name in {"md2ns", "2ns", "em-nvt-2ns", "em+nvt+md2ns"}:
        name = "em-nvt-md2ns"
    if name in {"md", "prod"}:
        name = "production"
    if name not in ALLOWED_PROTOCOLS:
        raise MdpError(
            f"Unknown MD protocol {protocol!r}; allowed: {list(ALLOWED_PROTOCOLS)}"
        )
    return name


def normalize_template_id(template_id: str) -> str:
    if not isinstance(template_id, str) or not template_id.strip():
        raise MdpError("template_id is required")
    name = template_id.strip().lower()
    if name in {"min", "em", "minimisation"}:
        name = "minimization"
    if name in {"nvt-eq", "nvteq"}:
        name = "nvt_eq"
    if name in {"md2ns", "2ns"}:
        name = "md2ns"
    if name in {"md", "prod"}:
        name = "production"
    if name not in ALLOWED_TEMPLATE_IDS:
        raise MdpError(
            f"Unknown mdp template {template_id!r}; allowed: {list(ALLOWED_TEMPLATE_IDS)}"
        )
    return name


def template_path(template_id: str) -> Path:
    spec = TEMPLATE_SPECS[normalize_template_id(template_id)]
    path = (TEMPLATES_DIR / spec.filename).resolve()
    if not path.is_file():
        raise MdpError(f"Human-written mdp template missing: {path}")
    try:
        path.relative_to(TEMPLATES_DIR.resolve())
    except ValueError as exc:
        raise MdpError(f"mdp template escapes experiments/: {path}") from exc
    return path


def protocol_template_ids(protocol: str) -> tuple[str, ...]:
    return PROTOCOL_TEMPLATES[normalize_protocol(protocol)]


def canonicalize_override_key(key: str) -> str:
    if not isinstance(key, str) or not key.strip():
        raise MdpError("override key is empty")
    raw = key.strip().lower().replace(" ", "")
    folded = raw.replace("-", "_")
    if raw in FORBIDDEN_OVERRIDE_KEYS or folded in FORBIDDEN_OVERRIDE_KEYS:
        raise MdpError(
            f"mdp override {key!r} is forbidden (integrator / cutoff / "
            "constraint and similar keys stay in the human template)"
        )
    canon = _KEY_ALIASES.get(raw) or _KEY_ALIASES.get(folded)
    if canon is None or canon not in ALLOWED_OVERRIDE_KEYS:
        raise MdpError(
            f"mdp override {key!r} is not allowlisted; "
            f"allowed scalars: {list(ALLOWED_OVERRIDE_KEYS)}"
        )
    return canon


def parse_override_value(key: str, value: Any) -> int | float:
    canon = canonicalize_override_key(key)
    if isinstance(value, bool) or value is None:
        raise MdpError(f"{canon} override must be a number, not {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if not text or not _SCALAR.fullmatch(text):
            raise MdpError(
                f"{canon} override must be a scalar number; got {value!r}"
            )
        if any(ch in text for ch in ";=\n\r#"):
            raise MdpError(f"{canon} override looks like free-text mdp")
        if canon == "nsteps":
            if "." in text or "e" in text.lower():
                raise MdpError("nsteps override must be an integer")
            number: int | float = int(text)
        else:
            number = float(text)
    elif isinstance(value, int) and not isinstance(value, bool):
        number = int(value) if canon == "nsteps" else float(value)
    elif isinstance(value, float):
        if canon == "nsteps":
            if not value.is_integer():
                raise MdpError("nsteps override must be an integer")
            number = int(value)
        else:
            number = float(value)
    else:
        raise MdpError(f"{canon} override must be a scalar number; got {value!r}")

    if canon == "nsteps":
        nsteps = int(number)
        if nsteps < MIN_NSTEPS:
            raise MdpError(f"nsteps {nsteps} is below minimum {MIN_NSTEPS}")
        if nsteps > MAX_TEMPLATE_NSTEPS:
            raise MdpError(
                f"nsteps {nsteps} exceeds template cap ({MAX_TEMPLATE_NSTEPS})"
            )
        return nsteps
    if canon == "dt":
        dt = float(number)
        if dt < MIN_DT or dt > MAX_DT:
            raise MdpError(f"dt {dt} is outside {MIN_DT}–{MAX_DT} ps")
        return dt
    ref_t = float(number)
    if ref_t < MIN_REF_T or ref_t > MAX_REF_T:
        raise MdpError(f"ref_t {ref_t} is outside {MIN_REF_T}–{MAX_REF_T} K")
    return ref_t


def normalize_overrides(overrides: dict[str, Any] | None) -> dict[str, int | float]:
    if not overrides:
        return {}
    if not isinstance(overrides, dict):
        raise MdpError("mdp overrides must be a mapping of allowlisted scalars")
    out: dict[str, int | float] = {}
    for raw_key, raw_value in overrides.items():
        canon = canonicalize_override_key(str(raw_key))
        out[canon] = parse_override_value(canon, raw_value)
    return out


def read_template_text(template_id: str) -> str:
    path = template_path(template_id)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise MdpError(f"Empty mdp template: {path}")
    return text


def template_has_key(text: str, key: str) -> bool:
    canon = canonicalize_override_key(key)
    for line in text.splitlines():
        match = _KEY_LINE.match(line)
        if not match:
            continue
        if _fold_key(match.group("key")) == canon:
            return True
    return False


def replace_mdp_scalar(text: str, key: str, value: int | float) -> str:
    canon = canonicalize_override_key(key)
    parsed = parse_override_value(canon, value)
    rendered = _render_scalar(canon, parsed)
    found = False
    lines: list[str] = []
    for line in text.splitlines():
        match = _KEY_LINE.match(line)
        if match and _fold_key(match.group("key")) == canon:
            found = True
            rest = match.group("rest")
            # Keep the original comment; drop any extra tokens on the value.
            comment = ""
            if ";" in rest:
                comment = rest[rest.index(";") :]
            lines.append(
                f"{match.group('pre')}{match.group('key')}{match.group('eq')}"
                f"{rendered}{comment}".rstrip()
            )
            continue
        lines.append(line)
    if not found:
        raise MdpError(
            f"allowlisted key {canon} is not present in the human template "
            "(refusing to insert new mdp parameters)"
        )
    trailing = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + trailing


def materialize_mdp(
    template_id: str,
    dest: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy a human template to ``dest`` and apply allowlisted scalar overrides.

    There is no ``text=`` / ``body=`` argument: a full mdp cannot be supplied.
    """
    template_id = normalize_template_id(template_id)
    dest_path = Path(dest).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    text = read_template_text(template_id)
    source = template_path(template_id)
    applied: dict[str, int | float] = {}
    skipped: dict[str, str] = {}
    for key, value in normalize_overrides(overrides).items():
        if not template_has_key(text, key):
            skipped[key] = "key absent from this template"
            continue
        text = replace_mdp_scalar(text, key, value)
        applied[key] = value
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


def materialize_protocol(
    protocol: str,
    dest_dir: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = normalize_protocol(protocol)
    overrides = normalize_overrides(overrides)
    dest_root = Path(dest_dir).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    applied_keys: set[str] = set()
    for template_id in protocol_template_ids(protocol):
        dest = dest_root / f"{template_id}.mdp"
        meta = materialize_mdp(template_id, dest, overrides=overrides)
        files.append(meta)
        applied_keys.update(meta["overrides"])
    missing = [key for key in overrides if key not in applied_keys]
    if missing:
        raise MdpError(
            "override key(s) not present in any template for protocol "
            f"{protocol}: {missing}"
        )
    nsteps_values = [int(item["nsteps"]) for item in files if item.get("nsteps")]
    return {
        "ok": True,
        "protocol": protocol,
        "runnable": protocol in RUNNABLE_PROTOCOLS,
        "files": files,
        "overrides": overrides,
        "max_nsteps": max(nsteps_values) if nsteps_values else None,
    }


def parse_mdp_nsteps(text: str) -> int | None:
    for line in text.splitlines():
        match = _KEY_LINE.match(line)
        if match and _fold_key(match.group("key")) == "nsteps":
            raw = match.group("val")
            try:
                return int(float(raw))
            except ValueError:
                return None
    return None


def launch_nsteps_cap(protocol: str | None = None) -> int:
    """Smoke-test cap is 10k steps; the 2 ns protocol may use 1e6. 100 ns is never launched."""
    if protocol is None:
        return MAX_LAUNCH_NSTEPS
    name = normalize_protocol(protocol)
    if name == "em-nvt-md2ns":
        return MAX_LAUNCH_NSTEPS_MD2NS
    return MAX_LAUNCH_NSTEPS


def assert_launch_nsteps(nsteps: int | None, protocol: str | None = None) -> int:
    if nsteps is None:
        raise MdpError("materialized mdp has no nsteps")
    cap = launch_nsteps_cap(protocol)
    if nsteps > cap:
        raise MdpError(
            f"nsteps={nsteps} exceeds this module's launch cap "
            f"({cap}) for protocol {protocol or DEFAULT_PROTOCOL}. "
            "Production 100 ns (nsteps=50000000) is not launched here. "
            "Use em-nvt for the smoke-test or em-nvt-md2ns for 2 ns. "
            "The 100 ns human template remains experiments/md.mdp."
        )
    if nsteps < MIN_NSTEPS:
        raise MdpError(f"nsteps {nsteps} is below minimum {MIN_NSTEPS}")
    return nsteps


def _fold_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _render_scalar(key: str, value: int | float) -> str:
    if key == "nsteps":
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"
