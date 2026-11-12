"""Checkpointed IO helpers: every stage is restartable after a failure."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd
import requests

from .config import USER_AGENT
from .logging_config import get_logger

LOG = get_logger("adminplex.io")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]], append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "at" if append else "wt"
    n = 0
    with opener(path, mode, encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def seen_keys(path: str | Path, key: str = "doc_id") -> set[str]:
    """Keys already persisted, so a restarted crawl does not refetch them."""
    return {rec[key] for rec in read_jsonl(path) if key in rec}


CSV_CELL_BUDGET = 200_000_000  # characters; above this a CSV copy is not worth the disk


def save_table(df: pd.DataFrame, stem: str | Path, formats: tuple[str, ...] = ("parquet", "csv")) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if "parquet" in formats:
        df.to_parquet(stem.with_suffix(".parquet"), index=False)
    if "csv" in formats:
        # A CSV of a table holding full document text is an order of magnitude
        # larger than the parquet and has filled the disk before now. Parquet is
        # the canonical format; the CSV is a convenience copy, so drop it when
        # the convenience is not worth the gigabytes.
        text_cols = df.select_dtypes(include="object")
        cells = int(text_cols.map(lambda v: len(v) if isinstance(v, str) else 0).to_numpy().sum())
        if cells > CSV_CELL_BUDGET:
            LOG.info("skipping %s.csv (%.1f GB of text); parquet only", stem.name, cells / 1e9)
        else:
            df.to_csv(stem.with_suffix(".csv"), index=False)
    LOG.info("saved %s (%d rows, %d cols)", stem.name, len(df), df.shape[1])


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def session(retries: int = 3) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=retries)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def get_with_retry(
    sess: requests.Session,
    url: str,
    *,
    tries: int = 4,
    timeout: int = 45,
    backoff: float = 1.6,
    params: dict[str, Any] | None = None,
    max_sleep: float = 90.0,
) -> requests.Response | None:
    """GET with exponential backoff. Returns None when the resource is unusable.

    Throttling responses (429, 503) are backed off far more patiently than
    transient errors, and any Retry-After header the server sends is honoured:
    public APIs are shared infrastructure and a research crawl must not degrade
    them for other users.
    """
    for attempt in range(tries):
        try:
            resp = sess.get(url, timeout=timeout, params=params)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (404, 410, 403):
                return None
            if resp.status_code == 202:
                # Some services answer an automated request with 202 and an
                # empty body instead of refusing it outright. Retrying does not
                # help and the caller would otherwise see nothing at all, so
                # this is reported rather than swallowed.
                LOG.warning("202 with %d bytes from %s: the service is deferring "
                            "automated requests", len(resp.content), url)
                return None
            if resp.status_code in (429, 503):
                hinted = resp.headers.get("Retry-After")
                wait = float(hinted) if hinted and hinted.isdigit() else min(max_sleep, 5.0 * backoff ** (attempt + 1))
                time.sleep(min(max_sleep, wait))
                continue
        except requests.RequestException:
            pass
        time.sleep(min(max_sleep, backoff**attempt))
    return None
