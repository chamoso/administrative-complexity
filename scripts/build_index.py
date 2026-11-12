"""Assemble the Administrative Complexity Index from the measured components.

Reads the rule-based features and the assessed scores, forms the fifteen
dimensions on a common scale, estimates every candidate aggregation, and writes
the comparison on which the choice of aggregation rests.

    python build_index.py --features data/interim/features.parquet \
                          --scores outputs/llm_scores --out outputs
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import CONFIG  # noqa: E402
from adminplex.index.build import build_all  # noqa: E402
from adminplex.index.dimensions import (  # noqa: E402
    ASSESSED_KEYS,
    DIMENSION_KEYS,
    build_dimension_scores,
)
from adminplex.io import save_json, save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.build_index")

ASSESSED_COLUMNS = [
    "llm_cognitive_load",
    "llm_legal_knowledge",
    "llm_clarity",
    "llm_n_requirements",
    "llm_n_documents",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/interim/features.parquet")
    ap.add_argument("--scores", default="outputs/llm_scores", help="directory of scores_*.jsonl")
    ap.add_argument("--corpus", default="data/processed/corpus.parquet")
    ap.add_argument("--tag", default="main", help="which scoring run to read")
    ap.add_argument("--out", default="outputs")
    return ap.parse_args()


def load_scores(directory: str, tag: str) -> pd.DataFrame:
    """Collapse the per-shard score files into one row per document."""
    files = sorted(glob.glob(str(Path(directory) / f"scores_{tag}_shard*.jsonl")))
    if not files:
        LOG.warning("no score files in %s for tag %r; the assessed dimensions will be empty", directory, tag)
        return pd.DataFrame(columns=["doc_id", *ASSESSED_COLUMNS])
    rows = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if "parsed_ok" not in frame.columns:
        frame["parsed_ok"] = False
    parsed = frame[frame["parsed_ok"].fillna(False).astype(bool)]
    LOG.info(
        "assessed scores: %d records from %d shards, %d parsed (%.1f%%)",
        len(frame), len(files), len(parsed), 100 * len(parsed) / max(1, len(frame)),
    )
    keep = ["doc_id", *[c for c in ASSESSED_COLUMNS if c in parsed.columns]]
    # A document scored more than once (a resubmitted shard) keeps its first record.
    return parsed[keep].drop_duplicates(subset="doc_id", keep="first")


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(args.features)
    scores = load_scores(args.scores, args.tag)
    merged = features.merge(scores, on="doc_id", how="left")
    LOG.info("merged: %d documents, %d columns", *merged.shape)

    coverage = {k: float(merged[k].notna().mean()) for k in ASSESSED_COLUMNS if k in merged.columns}
    for key, share in coverage.items():
        LOG.info("  %-22s assessed for %.1f%% of documents", key, 100 * share)

    dimensions = build_dimension_scores(merged, scaling=CONFIG["index"]["scaling"])
    usable = [k for k in DIMENSION_KEYS if dimensions[k].notna().any()]
    missing = [k for k in DIMENSION_KEYS if k not in usable]
    if missing:
        LOG.warning("dimensions with no data, excluded from every candidate: %s", ", ".join(missing))

    save_table(dimensions, out / "tables" / "dimension_scores")

    results = build_all(dimensions.set_index("doc_id"), CONFIG["index"]["candidate_models"])
    if not results:
        LOG.error("no candidate index could be estimated")
        return

    index_frame = pd.DataFrame({name: res.scores.values for name, res in results.items()})
    index_frame.insert(0, "doc_id", dimensions["doc_id"].values)
    save_table(index_frame, out / "tables" / "candidate_indices")

    # The candidates are compared on reliability, on how much of the common
    # variance they capture, and on how far they agree with one another: an
    # index whose ranking depends on the aggregation chosen is not an
    # instrument, and that has to be visible rather than asserted.
    agreement = index_frame[list(results)].corr(method="spearman")
    comparison = pd.DataFrame(
        [
            {
                "candidate": name,
                "alpha": res.diagnostics.get("alpha"),
                "omega": res.diagnostics.get("omega"),
                "first_component_share": res.diagnostics.get("first_component_share"),
                "kmo": res.diagnostics.get("kmo"),
                "mean_spearman_with_others": float(
                    agreement[name].drop(labels=[name]).mean()
                ) if len(results) > 1 else float("nan"),
            }
            for name, res in results.items()
        ]
    )
    save_table(comparison, out / "tables" / "index_comparison")
    save_table(agreement.reset_index().rename(columns={"index": "candidate"}),
               out / "tables" / "index_agreement")

    weights = pd.DataFrame({name: res.weights for name, res in results.items()}).reindex(usable)
    save_table(weights.reset_index().rename(columns={"index": "dimension"}),
               out / "tables" / "index_weights")

    save_json(
        {
            "n_documents": int(len(dimensions)),
            "dimensions_used": usable,
            "dimensions_missing": missing,
            "assessed_coverage": coverage,
            "scaling": CONFIG["index"]["scaling"],
            "diagnostics": {name: res.diagnostics for name, res in results.items()},
        },
        out / "tables" / "index_diagnostics.json",
    )

    print(comparison.round(3).to_string(index=False))
    print()
    print("Spearman agreement between candidates:")
    print(agreement.round(3).to_string())


if __name__ == "__main__":
    main()
