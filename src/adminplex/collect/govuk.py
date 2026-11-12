"""UK collector: GOV.UK Search API for discovery, Content API for full text.

Both are public, documented, key-free interfaces maintained by the Government
Digital Service, which makes the UK sample fully reproducible.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterator

from ..config import CONFIG, RAW
from ..io import get_with_retry, seen_keys, session, write_jsonl
from ..logging_config import get_logger

LOG = get_logger("adminplex.collect.govuk")
SEARCH = "https://www.gov.uk/api/search.json"
CONTENT = "https://www.gov.uk/api/content"
OUT = RAW / "govuk.jsonl.gz"

_FIELDS = "link,title,format,public_timestamp,organisations,content_store_document_type"


def discover(fmt: str, limit: int) -> list[dict[str, Any]]:
    """Page through the search API for one content format."""
    sess = session()
    found: list[dict[str, Any]] = []
    start, page = 0, 100
    while len(found) < limit:
        resp = get_with_retry(
            sess,
            SEARCH,
            params={"filter_format": fmt, "count": page, "start": start, "fields": _FIELDS},
        )
        if resp is None:
            break
        results = resp.json().get("results", [])
        if not results:
            break
        found.extend(results)
        start += page
        if start > 9900:  # search API pagination ceiling
            break
    LOG.info("govuk discover format=%s -> %d links", fmt, len(found[:limit]))
    return found[:limit]


# Keys that carry the procedural prose the citizen actually has to read.
_TEXT_KEYS = {
    "body",
    "introductory_paragraph",
    "more_information",
    "what_you_need_to_know",
    "other_ways_to_apply",
    "eligibility",
    "need_to_know",
    "downtime_message",
    "government_response",
    "summary",
    "description",
}
# Navigational or bookkeeping branches that would otherwise inflate length.
_SKIP_KEYS = {
    "change_history",
    "headers",
    "attachments",
    "external_related_links",
    "related_mainstream_content",
    "emphasised_organisations",
    "tags",
    "variants",
    "documents",
    "brand",
    "government",
    "political",
    "first_public_at",
    "start_button_text",
    "transaction_start_link",
    "will_continue_on",
}


def _walk_text(node: Any, out: list[str]) -> None:
    """Collect the procedural prose from the nested content-store payload."""
    if isinstance(node, dict):
        for key, val in node.items():
            if key in _SKIP_KEYS:
                continue
            if isinstance(val, str):
                looks_like_prose = "<" in val and len(val) > 80
                if key in _TEXT_KEYS or looks_like_prose:
                    out.append(val)
            else:
                _walk_text(val, out)
    elif isinstance(node, list):
        for item in node:
            _walk_text(item, out)


def fetch_one(item: dict[str, Any]) -> dict[str, Any] | None:
    sess = session()
    link = item.get("link", "")
    if not link.startswith("/"):
        return None
    resp = get_with_retry(sess, f"{CONTENT}{link}")
    if resp is None:
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None
    parts: list[str] = []
    _walk_text(payload.get("details", {}), parts)
    orgs = item.get("organisations") or []
    org = orgs[0].get("title") if orgs and isinstance(orgs[0], dict) else None
    return {
        "doc_id": f"UK:{link}",
        "country": "GB",
        "jurisdiction": "United Kingdom",
        "source": "gov.uk",
        "url": f"https://www.gov.uk{link}",
        "title": payload.get("title") or item.get("title"),
        "organisation": org,
        "admin_level": "central",
        "language": payload.get("locale", "en"),
        "doc_type": item.get("format"),
        "date": payload.get("first_published_at") or item.get("public_timestamp"),
        "html": "\n\n".join(parts),
    }


def run(limit_per_format: dict[str, int] | None = None) -> int:
    cfg = CONFIG["collection"]
    formats = limit_per_format or cfg["govuk"]["formats"]
    already = seen_keys(OUT)
    LOG.info("govuk: %d documents already stored", len(already))

    targets: list[dict[str, Any]] = []
    for fmt, cap in formats.items():
        for item in discover(fmt, cap):
            if f"UK:{item.get('link','')}" not in already:
                targets.append(item)
    LOG.info("govuk: %d new documents to fetch", len(targets))

    written = 0
    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as pool:
        futures = {pool.submit(fetch_one, t): t for t in targets}
        buffer: list[dict[str, Any]] = []
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            if rec and rec["html"]:
                buffer.append(rec)
            if len(buffer) >= 500:
                written += write_jsonl(OUT, buffer, append=True)
                buffer = []
                LOG.info("govuk: %d/%d fetched (%d written)", i, len(targets), written)
        if buffer:
            written += write_jsonl(OUT, buffer, append=True)
    LOG.info("govuk: done, %d new records", written)
    return written


if __name__ == "__main__":
    run()
