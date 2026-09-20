"""PubMed E-utilities client — predefined host only, never invents papers.

Network errors are raised as ``LiteratureNetworkError`` (loud failure).
Callers must not catch that and fill in citations.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable
from urllib.parse import urlparse

EUTILS_HOST = "eutils.ncbi.nlm.nih.gov"
ESEARCH_PATH = "/entrez/eutils/esearch.fcgi"
ESUMMARY_PATH = "/entrez/eutils/esummary.fcgi"
PUBMED_ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
DOI_URL = "https://doi.org/{doi}"

DEFAULT_RETMAX = 10
MAX_RETMAX = 50
MAX_QUERY_CHARS = 200
DEFAULT_TIMEOUT_SECONDS = 20

USER_AGENT = (
    "reinvent-agent/0.11 (+https://github.com/Luzuokun/reinvent-agent; "
    "literature tool; never invents citations)"
)

# PubMed query syntax (field tags, quotes, boolean ops). Not a shell, not a URL.
_QUERY_OK = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9 \t\"'()[\]{}+*?:,./_-]*$"
)

GetFn = Callable[[str], bytes]


class LiteratureNetworkError(RuntimeError):
    """Raised when NCBI cannot be reached or the response is unusable."""


def normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    text = " ".join(query.strip().split())
    if not text:
        raise ValueError("query is empty")
    if len(text) > MAX_QUERY_CHARS:
        raise ValueError(
            f"query exceeds {MAX_QUERY_CHARS} characters ({len(text)})"
        )
    lowered = text.lower()
    if "http://" in lowered or "https://" in lowered:
        raise ValueError("query must be a PubMed search string, not a URL")
    if any(ch in text for ch in ("\n", "\r", "\x00", ";", "|", "`", "$", "\\")):
        raise ValueError("query contains disallowed characters")
    if not _QUERY_OK.match(text):
        raise ValueError("query contains disallowed characters")
    return text


def clamp_retmax(value: int | None, *, default: int = DEFAULT_RETMAX) -> int:
    if value is None:
        n = int(default)
    else:
        n = int(value)
    if n < 1:
        n = 1
    if n > MAX_RETMAX:
        n = MAX_RETMAX
    return n


def assert_eutils_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise LiteratureNetworkError(
            f"NETWORK FAILURE: refusing non-HTTPS literature URL {url!r}"
        )
    if parsed.hostname != EUTILS_HOST:
        raise LiteratureNetworkError(
            f"NETWORK FAILURE: refusing non-NCBI host {parsed.hostname!r}"
        )
    if not parsed.path.startswith("/entrez/eutils/"):
        raise LiteratureNetworkError(
            f"NETWORK FAILURE: refusing non-E-utilities path {parsed.path!r}"
        )


def search_pubmed(
    query: str,
    *,
    retmax: int = DEFAULT_RETMAX,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    get_fn: GetFn | None = None,
) -> list[dict[str, Any]]:
    """Return sourced PubMed records for ``query``.

    Empty NCBI hit lists are a valid empty result (not invented papers).
    Transport / parse failures raise ``LiteratureNetworkError``.
    """
    query = normalize_query(query)
    retmax = clamp_retmax(retmax)
    getter = get_fn or (lambda url: http_get(url, timeout=timeout))

    search_url = _eutils_url(
        ESEARCH_PATH,
        {
            "db": "pubmed",
            "term": query,
            "retmax": str(retmax),
            "retmode": "json",
        },
    )
    search_payload = _load_json(getter(search_url), what="esearch")
    idlist = _id_list(search_payload)
    if not idlist:
        return []

    summary_url = _eutils_url(
        ESUMMARY_PATH,
        {
            "db": "pubmed",
            "id": ",".join(idlist),
            "retmode": "json",
        },
    )
    summary_payload = _load_json(getter(summary_url), what="esummary")
    return parse_esummary(summary_payload, expected_ids=idlist)


def parse_esummary(
    payload: dict[str, Any],
    *,
    expected_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Map an ESummary JSON object to sourced literature entries.

    Records without PMID, DOI, *and* URL are dropped. Nothing is invented
    for missing ids.
    """
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        raise LiteratureNetworkError(
            "NETWORK FAILURE: PubMed esummary JSON missing result object"
        )
    order = expected_ids or [
        str(uid) for uid in (result.get("uids") or []) if str(uid).strip()
    ]
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for uid in order:
        pmid = str(uid).strip()
        if not pmid or pmid in seen or pmid == "uids":
            continue
        seen.add(pmid)
        rec = result.get(pmid)
        if not isinstance(rec, dict):
            continue
        entry = sourced_entry_from_esummary(pmid, rec)
        if entry is not None:
            entries.append(entry)
    return entries


def sourced_entry_from_esummary(pmid: str, rec: dict[str, Any]) -> dict[str, Any] | None:
    doi = _doi_from_articleids(rec.get("articleids"))
    url = PUBMED_ARTICLE_URL.format(pmid=pmid) if pmid.isdigit() else None
    if doi and not url:
        url = DOI_URL.format(doi=doi)
    title = rec.get("title")
    title_text = title.strip() if isinstance(title, str) else ""
    journal = rec.get("fulljournalname") or rec.get("source")
    journal_text = journal.strip() if isinstance(journal, str) else ""
    pubdate = rec.get("pubdate") or rec.get("epubdate")
    pubdate_text = pubdate.strip() if isinstance(pubdate, str) else ""
    entry = {
        "pmid": pmid if pmid.isdigit() else "",
        "doi": doi or "",
        "url": url or "",
        "title": title_text,
        "journal": journal_text,
        "pubdate": pubdate_text,
        "origin": "pubmed",
    }
    if not is_sourced(entry):
        return None
    return entry


def is_sourced(entry: dict[str, Any]) -> bool:
    """True iff the record has a PMID, a DOI, and/or an http(s) URL."""
    pmid = str(entry.get("pmid") or "").strip()
    doi = str(entry.get("doi") or "").strip()
    url = str(entry.get("url") or "").strip()
    if pmid.isdigit():
        return True
    if doi and doi.lower() != "none":
        return True
    return _is_http_url(url)


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _doi_from_articleids(articleids: Any) -> str:
    if not isinstance(articleids, list):
        return ""
    for item in articleids:
        if not isinstance(item, dict):
            continue
        if str(item.get("idtype") or "").lower() != "doi":
            continue
        value = item.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _eutils_url(path: str, params: dict[str, str]) -> str:
    query = urllib.parse.urlencode(params)
    url = f"https://{EUTILS_HOST}{path}?{query}"
    assert_eutils_url(url)
    return url


def _id_list(payload: dict[str, Any]) -> list[str]:
    block = payload.get("esearchresult") if isinstance(payload, dict) else None
    if not isinstance(block, dict):
        raise LiteratureNetworkError(
            "NETWORK FAILURE: PubMed esearch JSON missing esearchresult"
        )
    raw = block.get("idlist") or []
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text.isdigit():
            ids.append(text)
    return ids


def _load_json(raw: bytes, *, what: str) -> dict[str, Any]:
    if not raw:
        raise LiteratureNetworkError(
            f"NETWORK FAILURE: empty PubMed {what} response"
        )
    try:
        text = raw.decode("utf-8")
        data = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiteratureNetworkError(
            f"NETWORK FAILURE: PubMed {what} response is not JSON ({exc})"
        ) from exc
    if not isinstance(data, dict):
        raise LiteratureNetworkError(
            f"NETWORK FAILURE: PubMed {what} JSON must be an object"
        )
    return data


class _HostLimitedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        assert_eutils_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def http_get(url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bytes:
    """HTTPS GET of an E-utilities URL. Never hits other hosts."""
    assert_eutils_url(url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        method="GET",
    )
    opener = urllib.request.build_opener(_HostLimitedRedirectHandler)
    try:
        with opener.open(request, timeout=timeout) as response:
            final_url = response.geturl()
            assert_eutils_url(final_url)
            return response.read()
    except LiteratureNetworkError:
        raise
    except urllib.error.HTTPError as exc:
        raise LiteratureNetworkError(
            f"NETWORK FAILURE: PubMed HTTP {exc.code} for {urlparse(url).path}"
        ) from exc
    except urllib.error.URLError as exc:
        reason = exc.reason if exc.reason else exc
        raise LiteratureNetworkError(
            f"NETWORK FAILURE: cannot reach PubMed ({reason})"
        ) from exc
    except TimeoutError as exc:
        raise LiteratureNetworkError(
            "NETWORK FAILURE: PubMed request timed out"
        ) from exc
    except OSError as exc:
        raise LiteratureNetworkError(
            f"NETWORK FAILURE: cannot reach PubMed ({exc})"
        ) from exc
