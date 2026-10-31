"""ChEMBL REST client — predefined host only, never invents activity rows.

Network errors raise ``ChemblNetworkError`` (loud failure). Callers must not
catch that and fill in SMILES / IC50 values.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from tools.chembl.queries import activity_filter_params

CHEMBL_HOST = "www.ebi.ac.uk"
CHEMBL_API_PREFIX = "/chembl/api/data"
ACTIVITY_PATH = f"{CHEMBL_API_PREFIX}/activity"

DEFAULT_LIMIT = 50
MAX_LIMIT = 200
DEFAULT_TIMEOUT_SECONDS = 30

USER_AGENT = (
    "reinvent-agent/0.13 (+https://github.com/Luzuokun/reinvent-agent; "
    "chembl tool; never invents assay rows)"
)

_CHEMBL_ID_OK = re.compile(r"^CHEMBL[0-9]+$")

GetFn = Callable[[str], bytes]


class ChemblNetworkError(RuntimeError):
    """Raised when ChEMBL cannot be reached or the response is unusable."""


def clamp_limit(value: int | None, *, default: int = DEFAULT_LIMIT) -> int:
    if value is None:
        n = int(default)
    else:
        n = int(value)
    if n < 1:
        n = 1
    if n > MAX_LIMIT:
        n = MAX_LIMIT
    return n


def assert_chembl_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ChemblNetworkError(
            f"NETWORK FAILURE: refusing non-HTTPS ChEMBL URL {url!r}"
        )
    if parsed.hostname != CHEMBL_HOST:
        raise ChemblNetworkError(
            f"NETWORK FAILURE: refusing non-ChEMBL host {parsed.hostname!r}"
        )
    if not parsed.path.startswith(CHEMBL_API_PREFIX):
        raise ChemblNetworkError(
            f"NETWORK FAILURE: refusing non-ChEMBL-API path {parsed.path!r}"
        )


def fetch_activities(
    query_spec: Mapping[str, Any],
    *,
    limit: int = DEFAULT_LIMIT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    get_fn: GetFn | None = None,
) -> list[dict[str, Any]]:
    """Return activity dicts for a predefined query.

    Empty ChEMBL hit lists are a valid empty result (not invented rows).
    Transport / parse failures raise ``ChemblNetworkError``.
    """
    limit_n = clamp_limit(limit, default=int(query_spec.get("default_limit", DEFAULT_LIMIT)))
    getter = get_fn or (lambda url: http_get(url, timeout=timeout))
    params = activity_filter_params(query_spec)
    params["limit"] = str(limit_n)
    params["offset"] = "0"
    url = _chembl_url(ACTIVITY_PATH, params)
    payload = _load_json(getter(url), what="activity")
    return parse_activity_payload(payload, query_id=str(query_spec.get("id") or ""))


def parse_activity_payload(
    payload: dict[str, Any],
    *,
    query_id: str = "",
) -> list[dict[str, Any]]:
    """Map a ChEMBL activity JSON object to fixed-schema assay rows.

    Rows missing ``molecule_chembl_id`` are dropped. Nothing is invented for
    missing SMILES or values.
    """
    if not isinstance(payload, dict):
        raise ChemblNetworkError(
            "NETWORK FAILURE: ChEMBL activity JSON is not an object"
        )
    raw_list = payload.get("activities")
    if raw_list is None:
        # Some deployments nest under page_meta siblings only.
        raise ChemblNetworkError(
            "NETWORK FAILURE: ChEMBL activity JSON missing activities list"
        )
    if not isinstance(raw_list, list):
        raise ChemblNetworkError(
            "NETWORK FAILURE: ChEMBL activities field is not a list"
        )

    rows: list[dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        row = normalize_activity_row(item, query_id=query_id)
        if row is not None:
            rows.append(row)
    return rows


def normalize_activity_row(
    item: Mapping[str, Any],
    *,
    query_id: str = "",
) -> dict[str, Any] | None:
    """Keep a row only when molecule_chembl_id looks like a real ChEMBL id."""
    mol_id = str(item.get("molecule_chembl_id") or "").strip()
    if not _CHEMBL_ID_OK.match(mol_id):
        return None
    target_id = str(item.get("target_chembl_id") or "").strip()
    assay_id = str(item.get("assay_chembl_id") or "").strip()
    smiles = str(item.get("canonical_smiles") or "").strip()
    standard_type = str(item.get("standard_type") or "").strip()
    standard_units = str(item.get("standard_units") or "").strip()
    activity_id = str(item.get("activity_id") or "").strip()
    return {
        "molecule_chembl_id": mol_id,
        "canonical_smiles": smiles,
        "target_chembl_id": target_id if _CHEMBL_ID_OK.match(target_id) else "",
        "standard_type": standard_type,
        "standard_value": _optional_number(item.get("standard_value")),
        "standard_units": standard_units,
        "pchembl_value": _optional_number(item.get("pchembl_value")),
        "assay_chembl_id": assay_id if _CHEMBL_ID_OK.match(assay_id) else "",
        "activity_id": activity_id,
        "query_id": query_id,
        "origin": "chembl",
    }


def rows_with_smiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Subset that already carries a non-empty canonical SMILES (no invention)."""
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("canonical_smiles") or "").strip():
            out.append(row)
    return out


def http_get(url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bytes:
    assert_chembl_url(url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise ChemblNetworkError(
            f"NETWORK FAILURE: ChEMBL HTTP {exc.code} for {url}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ChemblNetworkError(
            f"NETWORK FAILURE: cannot reach ChEMBL ({exc.reason})"
        ) from exc
    except TimeoutError as exc:
        raise ChemblNetworkError(
            f"NETWORK FAILURE: ChEMBL timed out after {timeout}s"
        ) from exc


def _chembl_url(path: str, params: Mapping[str, str]) -> str:
    query = urllib.parse.urlencode(dict(params))
    return f"https://{CHEMBL_HOST}{path}?{query}"


def _load_json(raw: bytes, *, what: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ChemblNetworkError(
            f"NETWORK FAILURE: ChEMBL {what} response is not UTF-8"
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ChemblNetworkError(
            f"NETWORK FAILURE: ChEMBL {what} response is not JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ChemblNetworkError(
            f"NETWORK FAILURE: ChEMBL {what} JSON root is not an object"
        )
    return payload


def _optional_number(value: Any) -> str:
    """Store numeric assay fields as plain strings; never invent a value."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip()
    return text
