"""Build the two frames on which the index is validated against real burden.

Part level  - CFR parts in the corpus, joined to the burden the OMB approved
              for the information collections that cite them.
Collection  - the collections themselves, described in the agency's own words,
              with the hours per response the agency reported.

The second frame is not part of the corpus, so its texts are written to a
separate scoring input: the index has to be computed on them the same way it is
computed on everything else, or the comparison is not a comparison.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.analysis.burden_link import (  # noqa: E402
    build_collection_frame,
    build_validation_frame,
)
from adminplex.config import PROCESSED  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.prepare_validation")

SCORING_COLUMNS = ["doc_id", "title", "text", "language", "country"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-chars", type=int, default=400,
                    help="collections whose description is shorter carry too little to assess")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    parts = build_validation_frame()
    LOG.info("part-level frame: %d CFR parts with a burden estimate", len(parts))
    if len(parts):
        LOG.info(
            "  median hours per response %.2f, interquartile range %.2f-%.2f",
            parts.median_hours_per_response.median(),
            parts.median_hours_per_response.quantile(0.25),
            parts.median_hours_per_response.quantile(0.75),
        )

    collections = build_collection_frame()
    usable = collections[collections.text.str.len() >= args.min_chars].copy()
    LOG.info(
        "collection-level frame: %d collections, %d with a description long enough to assess",
        len(collections), len(usable),
    )

    # Written with the text, unlike the analysis frame: this file is an input to
    # the assessment step, not a table anyone reads.
    out = PROCESSED / "validation_collections_scoring.parquet"
    usable[SCORING_COLUMNS].to_parquet(out, index=False)
    LOG.info("scoring input written: %s (%d documents)", out, len(usable))


if __name__ == "__main__":
    main()
