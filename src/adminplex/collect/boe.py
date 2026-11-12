"""Spanish collector: Boletín Oficial del Estado open-data API.

Sections III (other provisions: grants, calls for applications, authorisations)
and V (official notices, including public procurement) contain the instruments
through which Spanish citizens and firms actually interact with the state.
"""
from __future__ import annotations

import random
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, Iterable

from ..config import CONFIG, RAW, SEED
from ..io import get_with_retry, seen_keys, session, write_jsonl
from ..logging_config import get_logger

LOG = get_logger("adminplex.collect.boe")
SUMMARY = "https://www.boe.es/datosabiertos/api/boe/sumario"
DOC_XML = "https://www.boe.es/diario_boe/xml.php"
OUT = RAW / "boe.jsonl.gz"

# Section codes carry the administrative meaning of the instrument.
_SECTION_LABEL = {
    "1": "general_provision",
    "2A": "appointment",
    "2B": "competitive_examination",
    "3": "other_provision",
    "4": "judicial_notice",
    "5A": "public_procurement",
    "5B": "official_notice",
    "5C": "private_notice",
}


def _as_list(node: Any) -> list[Any]:
    if node is None:
        return []
    return node if isinstance(node, list) else [node]


def business_days(start: str, end: str) -> list[str]:
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    out = []
    while d0 <= d1:
        if d0.weekday() < 6:  # the BOE publishes Monday to Saturday
            out.append(d0.strftime("%Y%m%d"))
        d0 += timedelta(days=1)
    return out


def index_day(day: str, keep_sections: Iterable[str]) -> list[dict[str, Any]]:
    """List the items published on one day within the sections of interest."""
    sess = session()
    # The open-data endpoint answers 400 unless a JSON response is requested.
    sess.headers.update({"Accept": "application/json"})
    resp = get_with_retry(sess, f"{SUMMARY}/{day}", timeout=40)
    if resp is None:
        return []
    try:
        payload = resp.json()
    except ValueError:
        return []
    if payload.get("status", {}).get("code") not in ("200", 200):
        return []
    items: list[dict[str, Any]] = []
    prefixes = tuple(keep_sections)
    for diario in _as_list(payload.get("data", {}).get("sumario", {}).get("diario")):
        for sec in _as_list(diario.get("seccion")):
            code = str(sec.get("codigo", ""))
            if not code.startswith(prefixes):
                continue
            for dep in _as_list(sec.get("departamento")):
                dep_name = dep.get("nombre")
                groups = _as_list(dep.get("epigrafe")) or [{"nombre": None, "item": dep.get("item")}]
                for grp in groups:
                    for item in _as_list(grp.get("item")):
                        ident = item.get("identificador")
                        if not ident:
                            continue
                        items.append(
                            {
                                "identificador": ident,
                                "titulo": item.get("titulo"),
                                "section": code,
                                "epigrafe": grp.get("nombre"),
                                "departamento": dep_name,
                                "day": day,
                            }
                        )
    return items


def fetch_document(item: dict[str, Any]) -> dict[str, Any] | None:
    sess = session()
    resp = get_with_retry(sess, DOC_XML, params={"id": item["identificador"]}, timeout=45)
    if resp is None:
        return None
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return None
    meta = root.find("metadatos")
    body = root.find("texto")
    if body is None:
        return None
    text = "\n".join(t.strip() for t in body.itertext() if t and t.strip())
    if not text:
        return None
    getm = (lambda tag: (meta.findtext(tag) or "").strip()) if meta is not None else (lambda tag: "")
    origin = getm("origen_legislativo")
    return {
        "doc_id": f"ES:{item['identificador']}",
        "country": "ES",
        "jurisdiction": "Spain",
        "source": "BOE",
        "url": f"https://www.boe.es/diario_boe/txt.php?id={item['identificador']}",
        "title": getm("titulo") or item.get("titulo"),
        "organisation": getm("departamento") or item.get("departamento"),
        "admin_level": "regional" if origin and origin.lower() != "estatal" else "central",
        "language": "es",
        "doc_type": _SECTION_LABEL.get(item["section"], "other"),
        "instrument_rank": getm("rango"),
        "topic": item.get("epigrafe"),
        "date": getm("fecha_disposicion") or item["day"],
        "text": text,
    }


def run(max_documents: int | None = None) -> int:
    cfg = CONFIG["collection"]
    bcfg = cfg["boe"]
    cap = max_documents or bcfg["max_documents"]
    days = business_days(bcfg["date_from"], bcfg["date_to"])
    random.Random(SEED).shuffle(days)  # spread the sample over the whole window

    already = seen_keys(OUT)
    LOG.info("boe: %d documents already stored; scanning up to %d publication days", len(already), len(days))

    written = 0
    day_chunk = 10  # small enough that the document cap is respected promptly
    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as pool:
        for offset in range(0, len(days), day_chunk):
            if written >= cap:
                break
            batch = days[offset : offset + day_chunk]
            pending: list[dict[str, Any]] = []
            for fut in as_completed({pool.submit(index_day, d, bcfg["sections"]): d for d in batch}):
                pending.extend(
                    it for it in fut.result() if f"ES:{it['identificador']}" not in already
                )
            docs: list[dict[str, Any]] = []
            for fut in as_completed({pool.submit(fetch_document, it): it for it in pending}):
                rec = fut.result()
                if rec:
                    docs.append(rec)
                if len(docs) >= 400:
                    written += write_jsonl(OUT, docs, append=True)
                    docs = []
            if docs:
                written += write_jsonl(OUT, docs, append=True)
            LOG.info(
                "boe: %d documents written after %d/%d days scanned",
                written,
                offset + len(batch),
                len(days),
            )
    LOG.info("boe: done, %d new records", written)
    return written


if __name__ == "__main__":
    run()
