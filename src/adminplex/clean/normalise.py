"""Turn the raw collections into one analysable corpus.

Three things happen here, and each of them can change published numbers, so
each is logged: mark-up is removed, boilerplate that repeats across a source is
deleted, and duplicates and near-duplicates are dropped.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Iterable, Iterator

import pandas as pd
from bs4 import BeautifulSoup

from ..config import CONFIG, INTERIM, PROCESSED, RAW
from ..io import read_jsonl, save_json, save_table
from ..logging_config import get_logger

LOG = get_logger("adminplex.clean")

SOURCES = {
    "govuk": RAW / "govuk.jsonl.gz",
    "boe": RAW / "boe.jsonl.gz",
    "ecfr": RAW / "ecfr.jsonl.gz",
    "eurlex": RAW / "eurlex.jsonl.gz",
    "sitemap_sources": RAW / "sitemap_sources.jsonl.gz",
}

_WS = re.compile(r"[ \t ]+")
_NL = re.compile(r"\n{3,}")
_TAG = re.compile(r"<[^>]+>")

# Lines that are navigation or legal furniture rather than procedural content.
_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^skip to (?:main )?content$",
        r"^cookies? (?:on|policy|settings)",
        r"^we use (?:some essential )?cookies",
        r"^accept (?:all|additional) cookies$",
        r"^(?:print|share|email) this page$",
        r"^back to top$",
        r"^is this page useful\??$",
        r"^report a problem with this page$",
        r"^all content is available under",
        r"^open government licence",
        r"^©\s*crown copyright",
        r"^javascript is (?:disabled|required)",
        r"^loading\.{0,3}$",
        r"^menu$",
        r"^search$",
        r"^agencia estatal bolet[íi]n oficial del estado$",
        r"^este documento es de car[áa]cter informativo",
        r"^\s*$",
    )
]


def strip_markup(raw: str) -> str:
    if not raw:
        return ""
    if "<" in raw and re.search(r"<[a-zA-Z/]", raw):
        text = BeautifulSoup(raw, "lxml").get_text("\n", strip=True)
    else:
        text = raw
    text = _TAG.sub(" ", text)
    text = text.replace("\r", "\n").replace("​", "")
    text = _WS.sub(" ", text)
    return _NL.sub("\n\n", text).strip()


def _line_frequencies(records: list[dict[str, Any]], sample: int = 4000) -> Counter[str]:
    counter: Counter[str] = Counter()
    for rec in records[:sample]:
        for line in set(rec["text"].split("\n")):
            line = line.strip()
            if 3 < len(line) < 160:
                counter[line] += 1
    return counter


def remove_boilerplate(records: list[dict[str, Any]], threshold: float = 0.25) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop lines that repeat across a quarter or more of a source's documents."""
    if not records:
        return records, []
    freq = _line_frequencies(records)
    denom = min(len(records), 4000)
    repeated = {line for line, n in freq.items() if n / denom >= threshold}
    for rec in records:
        kept = []
        for line in rec["text"].split("\n"):
            stripped = line.strip()
            if stripped in repeated:
                continue
            if any(p.match(stripped) for p in _BOILERPLATE_PATTERNS):
                continue
            kept.append(line)
        rec["text"] = _NL.sub("\n\n", "\n".join(kept)).strip()
    return records, sorted(repeated)


def _shingle_hash(text: str, k: int = 12) -> str:
    """A cheap near-duplicate key: the hash of the rarest normalised shingles."""
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < k * 2:
        return hashlib.sha1(" ".join(toks).encode()).hexdigest()
    grams = {" ".join(toks[i : i + k]) for i in range(0, len(toks) - k + 1, 4)}
    picked = sorted(grams)[:64]
    return hashlib.sha1("|".join(picked).encode()).hexdigest()


def load_source(name: str, path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rec in read_jsonl(path):
        text = rec.get("text") or strip_markup(rec.get("html", ""))
        rec = {k: v for k, v in rec.items() if k != "html"}
        rec["text"] = strip_markup(text)
        if rec["text"]:
            records.append(rec)
    LOG.info("%s: %d raw documents with text", name, len(records))
    return records


def build_corpus() -> pd.DataFrame:
    cfg = CONFIG["collection"]
    min_words, max_chars = cfg["min_words"], cfg["max_chars"]
    audit: dict[str, Any] = {"sources": {}, "boilerplate": {}}

    frames: list[pd.DataFrame] = []
    for name, path in SOURCES.items():
        if not path.exists():
            LOG.warning("%s: no raw file, skipped", name)
            continue
        records = load_source(name, path)
        records, repeated = remove_boilerplate(records)
        audit["boilerplate"][name] = repeated[:40]
        df = pd.DataFrame(records)
        if df.empty:
            continue
        df["source_file"] = name
        frames.append(df)
        audit["sources"][name] = {"raw": len(df)}

    corpus = pd.concat(frames, ignore_index=True, sort=False)
    before = len(corpus)

    corpus["text"] = corpus["text"].str.slice(0, max_chars)
    corpus["n_words_raw"] = corpus["text"].str.count(r"\w+")
    corpus = corpus[corpus.n_words_raw >= min_words].copy()
    after_short = len(corpus)

    corpus["exact_hash"] = corpus["text"].map(lambda t: hashlib.sha1(t.encode()).hexdigest())
    corpus = corpus.drop_duplicates(subset="exact_hash", keep="first")
    after_exact = len(corpus)

    corpus["near_hash"] = corpus["text"].map(_shingle_hash)
    corpus = corpus.drop_duplicates(subset="near_hash", keep="first")
    after_near = len(corpus)

    corpus = corpus.drop_duplicates(subset="doc_id", keep="first").reset_index(drop=True)

    audit["filtering"] = {
        "raw_documents": before,
        "after_min_words": after_short,
        "removed_short": before - after_short,
        "after_exact_duplicates": after_exact,
        "removed_exact_duplicates": after_short - after_exact,
        "after_near_duplicates": after_near,
        "removed_near_duplicates": after_exact - after_near,
        "final": len(corpus),
        "min_words": min_words,
    }
    LOG.info("corpus: %s", audit["filtering"])
    save_json(audit, INTERIM / "cleaning_audit.json")

    keep = [
        "doc_id", "country", "jurisdiction", "source", "source_file", "url", "title",
        "organisation", "admin_level", "language", "doc_type", "date", "text",
        "cfr_title", "cfr_part", "n_citing_collections", "instrument_rank", "topic",
    ]
    for col in keep:
        if col not in corpus.columns:
            corpus[col] = None
    corpus = corpus[keep]
    # Parquet only: the corpus carries the full document text, so a CSV copy
    # runs to several gigabytes and adds nothing a reader of the parquet lacks.
    save_table(corpus, PROCESSED / "corpus", formats=("parquet",))
    return corpus


if __name__ == "__main__":
    build_corpus()
