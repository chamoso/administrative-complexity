"""Collector for jurisdictions that publish guidance as ordinary web pages.

Canada, Australia and New Zealand expose no procedural API, so the sample is
drawn from the published XML sitemaps and restricted by URL path to the
sections that contain services, benefits, permits and obligations.
"""
from __future__ import annotations

import gzip
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Iterable

from bs4 import BeautifulSoup

from ..config import CONFIG, RAW, SEED
from ..io import get_with_retry, seen_keys, session, write_jsonl
from ..logging_config import get_logger

LOG = get_logger("adminplex.collect.sitemap")
OUT = RAW / "sitemap_sources.jsonl.gz"

_STRIP = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg", "button")
_MAIN_SELECTORS = ("main", "[role=main]", "article", "#content", ".content", "#wb-cont", ".main-content")


@dataclass
class Source:
    key: str
    country: str
    jurisdiction: str
    label: str
    sitemap: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    language: str = "en"
    admin_level: str = "central"
    organisation_from_path: int = 0
    extra_headers: dict[str, str] = field(default_factory=dict)


SOURCES: dict[str, Source] = {
    "canada": Source(
        key="canada",
        country="CA",
        jurisdiction="Canada",
        label="canada.ca",
        sitemap="https://www.canada.ca/sitemap.xml",
        include=("/services/", "/en/revenue-agency/services/", "/immigration-refugees-citizenship/services/", "/employment-social-development/services/"),
        exclude=("/news/", "/media/", "/campaigns/", ".pdf", "/statistics/", "/transparency/"),
        organisation_from_path=2,
    ),
    "australia": Source(
        key="australia",
        country="AU",
        jurisdiction="Australia",
        label="business.gov.au",
        sitemap="https://business.gov.au/sitemap.xml",
        include=("/grants-and-programs/", "/planning/", "/registrations/", "/finance/", "/people/", "/risk-management/", "/products-and-services/", "/running/"),
        exclude=("/news/", "/events/", ".pdf"),
    ),
    "newzealand": Source(
        key="newzealand",
        country="NZ",
        jurisdiction="New Zealand",
        label="govt.nz",
        sitemap="https://www.govt.nz/sitemap.xml",
        include=("/browse/",),
        exclude=("/news/", ".pdf"),
    ),
    # The United States otherwise enters the corpus only through the Code of
    # Federal Regulations, which is a different genre of document from the
    # guidance the other jurisdictions publish. Without a comparable American
    # source, any comparison between countries is a comparison between genres.
    "unitedstates": Source(
        key="unitedstates",
        country="US",
        jurisdiction="United States",
        label="usa.gov",
        sitemap="https://www.usa.gov/sitemap.xml",
        include=("/",),
        exclude=("/es/", "/news/", "/about", "/contact", ".pdf", "/agency-index", "/state-", "/elected-officials"),
    ),
    # usa.gov alone is small next to gov.uk. These two match the other
    # jurisdictions' business and tax guidance, so that the comparison between
    # countries is between comparable material rather than whatever each
    # country happens to publish in bulk.
    "us_business": Source(
        key="us_business",
        country="US",
        jurisdiction="United States",
        label="sba.gov",
        sitemap="https://www.sba.gov/sitemap.xml",
        include=("/business-guide/", "/funding-programs/", "/federal-contracting/", "/partners/"),
        exclude=("/blog", "/events", "/about-sba", ".pdf", "/document/", "/article/"),
    ),
    "us_tax": Source(
        key="us_tax",
        country="US",
        jurisdiction="United States",
        label="irs.gov",
        sitemap="https://www.irs.gov/sitemap.xml",
        include=("/businesses/", "/individuals/", "/charities-non-profits/", "/credits-deductions/", "/filing/", "/payments/"),
        exclude=("/es/", "/zh-", "/ko/", "/ru/", "/vi/", "/ht/", "/newsroom/", "/pub/", ".pdf", "/irb/", "/statistics/"),
    ),
}


def _get_xml(url: str, sess) -> bytes | None:
    resp = get_with_retry(sess, url, timeout=60)
    if resp is None:
        return None
    body = resp.content
    if url.endswith(".gz"):
        try:
            body = gzip.decompress(body)
        except OSError:
            pass
    return body


def sitemap_urls(src: Source, hard_cap: int = 400_000) -> list[str]:
    """Expand a sitemap (or sitemap index) into the candidate page URLs.

    Large government sitemap indexes contain hundreds of child sitemaps, so the
    expansion is done concurrently, one wave per level of nesting.
    """
    sess = session()
    sess.headers.update(src.extra_headers)
    queue = [src.sitemap]
    seen_maps: set[str] = set()
    pages: list[str] = []
    while queue and len(pages) < hard_cap:
        wave = [u for u in queue if u not in seen_maps]
        queue = []
        seen_maps.update(wave)
        with ThreadPoolExecutor(max_workers=10) as pool:
            bodies = list(pool.map(lambda u: _get_xml(u, sess), wave))
        for body in bodies:
            if body is None:
                continue
            text = body.decode("utf-8", errors="replace")
            locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", text)
            if "<sitemapindex" in text[:2000]:
                queue.extend(locs)
            else:
                pages.extend(locs)
        LOG.info("%s: expanded %d sitemaps, %d page urls so far", src.key, len(seen_maps), len(pages))
    keep = [
        u
        for u in dict.fromkeys(pages)
        if any(inc in u for inc in src.include) and not any(exc in u for exc in src.exclude)
    ]
    LOG.info("%s: %d sitemap urls -> %d in scope (%d sitemaps read)", src.key, len(pages), len(keep), len(seen_maps))
    return keep


def extract_main_text(html: str) -> tuple[str, str]:
    """Return (title, main body text) with navigation and boilerplate removed."""
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    if soup.h1:
        title = soup.h1.get_text(" ", strip=True) or title
    for tag in soup(_STRIP):
        tag.decompose()
    node = None
    for sel in _MAIN_SELECTORS:
        found = soup.select_one(sel)
        if found and len(found.get_text(" ", strip=True)) > 400:
            node = found
            break
    node = node or soup.body or soup
    text = re.sub(r"\n{3,}", "\n\n", node.get_text("\n", strip=True))
    return title, text


def fetch_page(src: Source, url: str) -> dict[str, Any] | None:
    sess = session()
    sess.headers.update(src.extra_headers)
    resp = get_with_retry(sess, url, timeout=40)
    if resp is None or "html" not in resp.headers.get("Content-Type", ""):
        return None
    title, text = extract_main_text(resp.text)
    if len(text) < 300:
        return None
    org = src.label
    if src.organisation_from_path:
        parts = [p for p in url.split("/") if p][2:]
        if len(parts) > src.organisation_from_path:
            org = parts[src.organisation_from_path - 1].replace("-", " ").title()
    return {
        "doc_id": f"{src.country}:{url.split('://', 1)[-1]}",
        "country": src.country,
        "jurisdiction": src.jurisdiction,
        "source": src.label,
        "url": url,
        "title": title,
        "organisation": org,
        "admin_level": src.admin_level,
        "language": src.language,
        "doc_type": "service_guidance",
        "date": None,
        "text": text,
    }


def run(keys: Iterable[str] | None = None, caps: dict[str, int] | None = None) -> int:
    cfg = CONFIG["collection"]
    scfg = cfg["sitemap_sources"]
    keys = list(keys or SOURCES)
    caps = caps or {k: scfg.get(k, 2000) for k in keys}
    already = seen_keys(OUT)
    total = 0
    for key in keys:
        src = SOURCES[key]
        urls = [u for u in sitemap_urls(src) if f"{src.country}:{u.split('://',1)[-1]}" not in already]
        import random

        random.Random(SEED).shuffle(urls)
        urls = urls[: caps.get(key, 2000)]
        LOG.info("%s: fetching %d pages", key, len(urls))
        written = 0
        with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as pool:
            futures = {pool.submit(fetch_page, src, u): u for u in urls}
            buffer: list[dict[str, Any]] = []
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                if rec:
                    buffer.append(rec)
                if len(buffer) >= 300:
                    written += write_jsonl(OUT, buffer, append=True)
                    buffer = []
                    LOG.info("%s: %d/%d fetched (%d written)", key, i, len(urls), written)
            if buffer:
                written += write_jsonl(OUT, buffer, append=True)
        LOG.info("%s: done, %d records", key, written)
        total += written
    return total


if __name__ == "__main__":
    run()
