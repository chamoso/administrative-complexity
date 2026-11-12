"""US collector: the regulatory text behind each federal information collection.

Every PRA collection cites the Code of Federal Regulations parts that impose
it. The eCFR public API returns the authoritative text of those parts, so each
burden estimate can be paired with the rule text that produces it.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from ..config import CONFIG, RAW
from ..io import get_with_retry, seen_keys, session, write_jsonl
from ..logging_config import get_logger

LOG = get_logger("adminplex.collect.ecfr")
API = "https://www.ecfr.gov/api/versioner/v1/full"
SNAPSHOT = "2026-01-01"
OUT = RAW / "ecfr.jsonl.gz"

_TITLE_NAMES: dict[str, str] = {}


def title_names() -> dict[str, str]:
    global _TITLE_NAMES
    if _TITLE_NAMES:
        return _TITLE_NAMES
    resp = get_with_retry(session(), "https://www.ecfr.gov/api/versioner/v1/titles.json")
    if resp is not None:
        _TITLE_NAMES = {str(t["number"]): t["name"] for t in resp.json().get("titles", [])}
    return _TITLE_NAMES


def cited_parts(min_count: int = 1) -> list[tuple[str, str, int]]:
    """CFR parts cited by the PRA inventory, most frequently cited first."""
    path = RAW / "pra_collections.parquet"
    if not path.exists():
        raise FileNotFoundError("run adminplex.collect.pra first")
    df = pd.read_parquet(path)
    counter: Counter[tuple[str, str]] = Counter()
    for cell in df.loc[df.icr_status == "Active", "cfr_parts"].dropna():
        for token in str(cell).split(";"):
            if ":" not in token:
                continue
            title, part = token.split(":", 1)
            part = part.strip().split(".")[0].strip()  # subpart granularity is folded up
            if title.strip().isdigit() and re.fullmatch(r"\d+[A-Za-z]?", part or ""):
                counter[(title.strip(), part)] += 1
    parts = [(t, p, n) for (t, p), n in counter.most_common() if n >= min_count]
    LOG.info("%d distinct CFR parts cited by active collections", len(parts))
    return parts


def _xml_to_text(raw: bytes) -> tuple[str, str]:
    """Return (heading, plain text) for a CFR part."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return "", ""
    head = (root.findtext("HEAD") or "").strip()
    chunks: list[str] = []
    for node in root.iter():
        if node.tag in {"HEAD", "P", "PSPACE", "FP", "HED", "SUBJECT"}:
            text = "".join(node.itertext()).strip()
            if text:
                chunks.append(text)
    return head, "\n".join(chunks)


def fetch_part(title: str, part: str, citations: int) -> dict | None:
    sess = session()
    resp = get_with_retry(sess, f"{API}/{SNAPSHOT}/title-{title}.xml", params={"part": part}, timeout=60, tries=3, max_sleep=20.0)
    if resp is None:
        return None
    head, text = _xml_to_text(resp.content)
    if len(text) < 400:
        return None
    return {
        "doc_id": f"US:CFR-{title}-{part}",
        "country": "US",
        "jurisdiction": "United States",
        "source": "eCFR",
        "url": f"https://www.ecfr.gov/current/title-{title}/part-{part}",
        "title": head or f"{title} CFR Part {part}",
        "organisation": title_names().get(title, f"Title {title}"),
        "admin_level": "central",
        "language": "en",
        "doc_type": "regulation_part",
        "date": SNAPSHOT,
        "cfr_title": title,
        "cfr_part": part,
        "n_citing_collections": citations,
        "text": text,
    }


def run(max_parts: int | None = None) -> int:
    cfg = CONFIG["collection"]
    cap = max_parts or cfg["ecfr"]["max_parts"]
    already = seen_keys(OUT)
    todo = [(t, p, n) for t, p, n in cited_parts() if f"US:CFR-{t}-{p}" not in already][:cap]
    LOG.info("ecfr: %d parts to fetch (%d already stored)", len(todo), len(already))

    written = 0
    # The eCFR service answers intermittently; short retries with moderate
    # concurrency retrieve more parts per hour than patient serial requests.
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch_part, t, p, n): (t, p) for t, p, n in todo}
        buffer = []
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            if rec:
                buffer.append(rec)
            if len(buffer) >= 50:
                written += write_jsonl(OUT, buffer, append=True)
                buffer = []
                LOG.info("ecfr: %d/%d processed (%d written)", i, len(todo), written)
        if buffer:
            written += write_jsonl(OUT, buffer, append=True)
    LOG.info("ecfr: done, %d new records", written)
    return written


if __name__ == "__main__":
    run()
