"""Compute every measurable feature for every document.

The work is embarrassingly parallel across documents, so it is distributed over
the available cores and written in checkpointed chunks.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..config import INTERIM, PROCESSED
from ..io import save_table
from ..logging_config import get_logger
from . import readability, rulebased

LOG = get_logger("adminplex.features")


def _one(payload: tuple[str, str, str]) -> dict[str, Any]:
    doc_id, text, language = payload
    dims = asdict(rulebased.compute(text, language))
    read = asdict(readability.compute(text, language))
    read.pop("n_words", None)
    read.pop("n_sentences", None)
    out: dict[str, Any] = {"doc_id": doc_id}
    out.update(dims)
    out.update(read)
    return out


def compute_features(
    corpus: pd.DataFrame,
    text_column: str = "text",
    workers: int | None = None,
    chunk: int = 2000,
) -> pd.DataFrame:
    payloads = [
        (row.doc_id, getattr(row, text_column) or "", getattr(row, "language", "en") or "en")
        for row in corpus.itertuples()
    ]
    LOG.info("computing features for %d documents", len(payloads))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rec in enumerate(pool.map(_one, payloads, chunksize=64), 1):
            rows.append(rec)
            if i % 10000 == 0:
                LOG.info("  %d/%d documents processed", i, len(payloads))
    frame = pd.DataFrame(rows)
    LOG.info("features complete: %d rows x %d columns", *frame.shape)
    return frame


def run(corpus_path: str | Path | None = None, out_stem: str | Path | None = None) -> pd.DataFrame:
    corpus = pd.read_parquet(corpus_path or PROCESSED / "corpus.parquet")
    features = compute_features(corpus)
    save_table(features, out_stem or INTERIM / "features")
    return features


if __name__ == "__main__":
    run()
