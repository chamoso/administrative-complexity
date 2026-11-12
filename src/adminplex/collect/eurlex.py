"""EU collector: EUR-Lex implementing and delegated acts.

Works are discovered through the Publications Office SPARQL endpoint and the
consolidated text is retrieved from EUR-Lex by CELEX number. Implementing
regulations and decisions are the EU instruments that impose procedural
obligations directly on firms and national administrations.
"""
from __future__ import annotations

import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from ..config import CONFIG, RAW
from ..io import get_with_retry, seen_keys, session, write_jsonl
from ..logging_config import get_logger

LOG = get_logger("adminplex.collect.eurlex")
SPARQL = "http://publications.europa.eu/webapi/rdf/sparql"
# The EUR-Lex website answers automated requests with 202 and an empty body.
# The Publications Office serves the same documents through content
# negotiation on the Cellar repository, which is the route intended for
# machine access and returns the full text.
CELLAR = "http://publications.europa.eu/resource/celex/"
CELLAR_HEADERS = {"Accept": "application/xhtml+xml", "Accept-Language": "eng"}
OUT = RAW / "eurlex.jsonl.gz"

# Resource types that carry procedural obligations rather than political texts.
_TYPES = {
    "REG_IMPL": "implementing_regulation",
    "REG_DEL": "delegated_regulation",
    "DEC_IMPL": "implementing_decision",
    "DIR": "directive",
    "REG": "regulation",
}

_QUERY = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
SELECT DISTINCT ?celex ?date WHERE {{
  ?work cdm:work_has_resource-type <http://publications.europa.eu/resource/authority/resource-type/{rtype}> .
  ?work cdm:resource_legal_id_celex ?celex .
  ?work cdm:work_date_document ?date .
  FILTER (?date >= "2019-01-01"^^<http://www.w3.org/2001/XMLSchema#date>)
}}
ORDER BY DESC(?date)
LIMIT {limit}
"""


def discover(rtype: str, limit: int) -> list[dict[str, str]]:
    sess = session()
    resp = get_with_retry(
        sess,
        SPARQL,
        params={"query": _QUERY.format(rtype=rtype, limit=limit), "format": "application/sparql-results+json"},
        timeout=180,
    )
    if resp is None:
        LOG.warning("eurlex: SPARQL query failed for %s", rtype)
        return []
    try:
        bindings = resp.json()["results"]["bindings"]
    except (ValueError, KeyError):
        return []
    out = [
        {"celex": b["celex"]["value"], "date": b.get("date", {}).get("value", ""), "rtype": rtype}
        for b in bindings
        if "celex" in b
    ]
    LOG.info("eurlex: %s -> %d works", rtype, len(out))
    return out


def fetch_one(item: dict[str, str]) -> dict[str, Any] | None:
    sess = session()
    sess.headers.update(CELLAR_HEADERS)
    celex = item["celex"]
    resp = get_with_retry(sess, CELLAR + celex, timeout=90)
    if resp is None:
        return None
    # Cellar serves XHTML; the HTML parser reads it correctly for text
    # extraction, and the warning about the mismatch is not informative here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(("script", "style", "nav", "header", "footer")):
        tag.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))
    if len(text) < 600:
        return None
    title = soup.title.get_text(strip=True) if soup.title else celex
    heading = soup.find(["h1", "p"], class_=re.compile("doc-ti|title", re.I))
    if heading:
        title = heading.get_text(" ", strip=True)[:400] or title
    return {
        "doc_id": f"EU:{celex}",
        "country": "EU",
        "jurisdiction": "European Union",
        "source": "EUR-Lex",
        "url": f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}",
        "title": title,
        "organisation": "European Commission",
        "admin_level": "supranational",
        "language": "en",
        "doc_type": _TYPES.get(item["rtype"], "eu_act"),
        "date": item.get("date"),
        "text": text,
    }


def run(max_documents: int | None = None) -> int:
    cfg = CONFIG["collection"]
    cap = max_documents or cfg["eurlex"]["max_documents"]
    per_type = max(1, cap // len(_TYPES))
    already = seen_keys(OUT)

    targets: list[dict[str, str]] = []
    for rtype in _TYPES:
        targets.extend(t for t in discover(rtype, per_type) if f"EU:{t['celex']}" not in already)
    LOG.info("eurlex: %d works to fetch", len(targets))

    written = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_one, t): t for t in targets}
        buffer: list[dict[str, Any]] = []
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            if rec:
                buffer.append(rec)
            if len(buffer) >= 250:
                written += write_jsonl(OUT, buffer, append=True)
                buffer = []
                LOG.info("eurlex: %d/%d processed (%d written)", i, len(targets), written)
        if buffer:
            written += write_jsonl(OUT, buffer, append=True)
    if targets and written == 0:
        # Fetching thousands of documents and storing none is a failure, not an
        # empty result. Said plainly here it costs a minute to notice; left
        # implicit it costs however long the run takes.
        LOG.error("eurlex: %d works fetched and nothing stored — the source is "
                  "refusing or has changed; the corpus is unchanged", len(targets))
    else:
        LOG.info("eurlex: done, %d new records", written)
    return written


if __name__ == "__main__":
    run()
