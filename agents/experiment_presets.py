"""Human-written experiment presets.

The planner / LLM may name a ``preset_id`` from ``ALLOWED_PRESET_IDS``.
It cannot invent TOML, argv, or scoring components. Fillable fields are
an allowlist per preset; currently only ``scaffold_path`` on
``sampling-cpu-scaffold``, which must already exist under the project
input directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from tools import REPO_ROOT

PRESETS_DIR = REPO_ROOT / "experiments"
AGENT_SUBDIR = ".agent"
SCAFFOLD_PLACEHOLDER = "{{scaffold_path}}"

# Hard ceiling — config may only *narrow* this set.
ALLOWED_PRESET_IDS: tuple[str, ...] = (
    "sampling-cpu-100",
    "sampling-cpu-1000",
    "sampling-cpu-scaffold",
)

_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._/-]*$")


class PresetError(ValueError):
    """Unknown preset, illegal fill field, or sandbox path violation."""


@dataclass(frozen=True)
class PresetSpec:
    """Metadata for one human-written preset (never model-authored)."""

    preset_id: str
    toml_name: str
    description: str
    fields: frozenset[str]


PRESET_SPECS: dict[str, PresetSpec] = {
    "sampling-cpu-100": PresetSpec(
        preset_id="sampling-cpu-100",
        toml_name="sampling-cpu-100.toml",
        description="CPU sampling, 100 SMILES (demo-sized).",
        fields=frozenset(),
    ),
    "sampling-cpu-1000": PresetSpec(
        preset_id="sampling-cpu-1000",
        toml_name="sampling-cpu-1000.toml",
        description="CPU sampling, 1000 SMILES.",
        fields=frozenset(),
    ),
    "sampling-cpu-scaffold": PresetSpec(
        preset_id="sampling-cpu-scaffold",
        toml_name="sampling-cpu-scaffold.toml",
        description=(
            "CPU sampling from an existing scaffold SMILES file. "
            "Fills only smiles_file; no QED/affinity scoring TOML."
        ),
        fields=frozenset({"scaffold_path"}),
    ),
}


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_preset_allowlist(configured: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return the effective preset allowlist.

    Config may only narrow the hardcoded set; unknown names are rejected
    so a typo cannot introduce a new experiment.
    """
    if not configured:
        return ALLOWED_PRESET_IDS
    names = [str(item) for item in configured]
    unknown = [name for name in names if name not in ALLOWED_PRESET_IDS]
    if unknown:
        raise PresetError(f"Config planner.presets contains unknown ids: {unknown}")
    seen: set[str] = set()
    ordered: list[str] = []
    for name in ALLOWED_PRESET_IDS:
        if name in names and name not in seen:
            seen.add(name)
            ordered.append(name)
    if not ordered:
        raise PresetError("planner.presets is empty after filtering")
    return tuple(ordered)


def get_preset(preset_id: str) -> PresetSpec:
    if not isinstance(preset_id, str) or not preset_id.strip():
        raise PresetError("preset_id must be a non-empty string")
    pid = preset_id.strip()
    spec = PRESET_SPECS.get(pid)
    if spec is None:
        raise PresetError(
            f"Unknown preset_id: {pid!r}. Allowlist: {list(ALLOWED_PRESET_IDS)}"
        )
    source = preset_source_toml(spec)
    if not source.is_file():
        raise PresetError(f"Preset TOML missing on disk: {source}")
    return spec


def preset_source_toml(spec: PresetSpec) -> Path:
    return (PRESETS_DIR / spec.toml_name).resolve()


def resolve_scaffold_path(
    scaffold_path: str | Path,
    *,
    project_dir: str | Path,
    input_dir: str | Path | None = None,
) -> Path:
    """Resolve ``scaffold_path`` and require it stay under project input/.

    Relative paths are tried against ``project_dir`` first (e.g.
    ``input/scaffold.smi``), then against ``input_dir`` (e.g. ``scaffold.smi``).
    Absolute paths are accepted only if they already resolve inside input_dir.
    The file must exist. Symlink escapes are rejected via ``Path.resolve``.
    """
    if scaffold_path is None or str(scaffold_path).strip() == "":
        raise PresetError("scaffold_path is empty")

    raw = Path(str(scaffold_path).strip())
    project_dir = Path(project_dir).expanduser().resolve()
    if input_dir is None:
        input_dir = project_dir / "input"
    input_dir = Path(input_dir).expanduser().resolve()

    if raw.is_absolute():
        candidates = [raw.expanduser().resolve()]
    else:
        candidates = [
            (Path.cwd() / raw).expanduser().resolve(),
            (project_dir / raw).resolve(),
            (input_dir / raw).resolve(),
        ]

    existing_under = [c for c in candidates if c.is_file() and _is_under(c, input_dir)]
    if existing_under:
        return existing_under[0]

    existing_outside = [c for c in candidates if c.exists() and not _is_under(c, input_dir)]
    if existing_outside:
        raise PresetError(
            f"scaffold_path escapes project input dir: {scaffold_path} "
            f"(input_dir={input_dir})"
        )

    under_input = [c for c in candidates if _is_under(c, input_dir)]
    if not under_input:
        raise PresetError(
            f"scaffold_path escapes project input dir: {scaffold_path} "
            f"(input_dir={input_dir})"
        )
    raise PresetError(
        f"scaffold_path does not exist or is not a file: {under_input[0]}"
    )


def relative_scaffold_for_toml(resolved: Path, *, project_dir: str | Path) -> str:
    """Return a TOML-safe project-relative POSIX path for smiles_file."""
    project_dir = Path(project_dir).expanduser().resolve()
    try:
        rel = resolved.resolve().relative_to(project_dir).as_posix()
    except ValueError as exc:
        raise PresetError(
            f"scaffold_path is not under project_dir: {resolved}"
        ) from exc
    if not _SAFE_RELATIVE.fullmatch(rel) or ".." in Path(rel).parts:
        raise PresetError(
            f"scaffold_path is not a safe relative path for TOML: {rel}"
        )
    return rel


def validate_preset_selection(
    preset_id: str | None,
    *,
    project_dir: str | Path,
    scaffold_path: str | Path | None = None,
    input_dir: str | Path | None = None,
    allowlist: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate preset id + allowlisted fields. Does not write files."""
    if preset_id is None or (isinstance(preset_id, str) and not preset_id.strip()):
        if scaffold_path is not None and str(scaffold_path).strip():
            raise PresetError(
                "scaffold_path was provided but preset_id is missing"
            )
        return {}

    effective = set(resolve_preset_allowlist(allowlist))
    spec = get_preset(preset_id)
    if spec.preset_id not in effective:
        raise PresetError(
            f"preset_id {spec.preset_id!r} is not in the configured allowlist"
        )

    normalized: dict[str, Any] = {
        "preset_id": spec.preset_id,
        "description": spec.description,
        "source_toml": str(preset_source_toml(spec)),
        "fields": sorted(spec.fields),
    }

    has_scaffold = scaffold_path is not None and str(scaffold_path).strip() != ""
    if has_scaffold and "scaffold_path" not in spec.fields:
        raise PresetError(
            f"scaffold_path is not an allowlisted field for preset {spec.preset_id}"
        )
    if "scaffold_path" in spec.fields and not has_scaffold:
        raise PresetError(
            f"preset {spec.preset_id} requires scaffold_path "
            "(existing file under the project input directory)"
        )
    if has_scaffold:
        resolved = resolve_scaffold_path(
            scaffold_path, project_dir=project_dir, input_dir=input_dir
        )
        rel = relative_scaffold_for_toml(resolved, project_dir=project_dir)
        normalized["scaffold_path"] = str(resolved)
        normalized["scaffold_relpath"] = rel

    return normalized


def materialize_preset(
    preset_id: str,
    *,
    project_dir: str | Path,
    scaffold_path: str | Path | None = None,
    input_dir: str | Path | None = None,
    allowlist: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Copy a human-written preset TOML into ``<project>/.agent/``.

    The only substitution is the ``{{scaffold_path}}`` placeholder on
    presets that allow it, replaced with a validated relative path.
    Model text is never written into the file.
    """
    project_dir = Path(project_dir).expanduser().resolve()
    if input_dir is None:
        input_dir = project_dir / "input"
    validated = validate_preset_selection(
        preset_id,
        project_dir=project_dir,
        scaffold_path=scaffold_path,
        input_dir=input_dir,
        allowlist=allowlist,
    )
    spec = get_preset(validated["preset_id"])
    source = preset_source_toml(spec)
    text = source.read_text(encoding="utf-8")

    if "scaffold_path" in spec.fields:
        rel = validated["scaffold_relpath"]
        if SCAFFOLD_PLACEHOLDER not in text:
            raise PresetError(
                f"Preset template missing {SCAFFOLD_PLACEHOLDER} placeholder: {source}"
            )
        text = text.replace(SCAFFOLD_PLACEHOLDER, rel)

    if "{{" in text or "}}" in text:
        raise PresetError(f"Unresolved placeholder in preset TOML: {source}")

    dest_dir = project_dir / AGENT_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{spec.preset_id}.toml"
    dest.write_text(text, encoding="utf-8")
    return {
        **validated,
        "config_path": str(dest.resolve()),
        "materialized": True,
    }
