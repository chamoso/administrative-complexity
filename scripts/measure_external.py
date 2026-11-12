"""Measure documents that are not part of the corpus, on the corpus scale.

The burden-validation collections and the edited procedures of experiments 5
and 6 are not in the corpus, so nothing in the pipeline has measured them. They
still have to be measured exactly as the corpus was, and placed on the same
scale, or nothing can be compared with anything.

Two things are held fixed rather than re-estimated: the percentile scale, which
comes from the corpus distribution, and the aggregation weights, which are
estimated once on the corpus. Re-estimating either would let the instrument
redefine itself around whatever sample it is applied to.

    python measure_external.py --input data/processed/validation_collections_scoring.parquet \\
                               --scores outputs/llm_scores --tag collections \\
                               --out outputs/external/collections
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import CONFIG, INTERIM  # noqa: E402
from adminplex.features import pipeline as feature_pipeline  # noqa: E402
from adminplex.index.build import BUILDERS  # noqa: E402
from adminplex.index.dimensions import DIMENSION_KEYS, build_dimension_scores  # noqa: E402
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.external")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="parquet with doc_id, title, text, language")
    ap.add_argument("--scores", default="outputs/llm_scores")
    ap.add_argument("--tag", required=True, help="tag of the assessment run for these texts")
    ap.add_argument("--corpus-features", default=str(INTERIM / "features.parquet"))
    ap.add_argument("--corpus-tag", default="main",
                    help="assessment run for the corpus; its judged scores define the reference scale")
    ap.add_argument("--corpus-dimensions", default="outputs/tables/dimension_scores.parquet",
                    help="the corpus dimension scores the published index was built from")
    ap.add_argument("--index", default="pca")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def load_assessed(directory: str, tag: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(directory) / f"scores_{tag}_shard*.jsonl")))
    if not files:
        LOG.warning("no assessed scores for tag %r; the three judged dimensions will be missing", tag)
        return pd.DataFrame(columns=["doc_id"])
    rows = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if "parsed_ok" in frame.columns:
        frame = frame[frame.parsed_ok.fillna(False).astype(bool)]
    keep = ["doc_id", *[c for c in frame.columns if c.startswith("llm_")]]
    out = frame[keep].drop_duplicates(subset="doc_id", keep="first")
    LOG.info("assessed scores available for %d documents", len(out))
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    texts = pd.read_parquet(args.input)
    LOG.info("measuring %d external documents", len(texts))
    features = feature_pipeline.compute_features(texts)

    assessed = load_assessed(args.scores, args.tag)
    if not assessed.empty:
        features = features.merge(assessed, on="doc_id", how="left")
    save_table(features, out / "features", formats=("parquet",))

    # The reference scale has to be the whole instrument. The corpus feature
    # file holds only the counted measures; the judged ones live with the
    # assessment output, and leaving them out silently drops the three
    # dimensions that contribute most to the index.
    corpus_features = pd.read_parquet(args.corpus_features)
    corpus_assessed = load_assessed(args.scores, args.corpus_tag)
    if corpus_assessed.empty:
        LOG.error("no assessment for the corpus (tag %r): the judged dimensions would be dropped "
                  "and the index applied here would not be the published one", args.corpus_tag)
    else:
        corpus_features = corpus_features.merge(corpus_assessed, on="doc_id", how="left")

    scaling = CONFIG["index"]["scaling"]
    external_dims = build_dimension_scores(features, scaling=scaling, reference=corpus_features)
    # Reuse the corpus dimension scores the published index was built from
    # instead of recomputing them, so the weights are demonstrably the same.
    corpus_dims = (
        pd.read_parquet(args.corpus_dimensions)
        if Path(args.corpus_dimensions).exists()
        else build_dimension_scores(corpus_features, scaling=scaling)
    )

    usable = [
        k for k in DIMENSION_KEYS
        if k in external_dims.columns and external_dims[k].notna().any()
        and k in corpus_dims.columns and corpus_dims[k].notna().any()
    ]
    missing = [k for k in DIMENSION_KEYS if k not in usable]
    if missing:
        LOG.warning("dimensions unavailable for these documents: %s", ", ".join(missing))
    save_table(external_dims, out / "dimension_scores", formats=("parquet",))

    # Weights estimated on the corpus, applied unchanged.
    reference = BUILDERS[args.index](corpus_dims.set_index("doc_id")[usable])
    weights = pd.Series(reference.weights).reindex(usable).fillna(0.0)
    values = external_dims.set_index("doc_id")[usable].fillna(50.0)
    scores = (values * weights).sum(axis=1) / weights.abs().sum()

    index_frame = pd.DataFrame({"doc_id": scores.index, args.index: scores.values})
    save_table(index_frame, out / "candidate_indices", formats=("parquet",))

    LOG.info(
        "index on external documents: mean %.1f, sd %.1f (corpus weights, corpus scale)",
        float(scores.mean()), float(scores.std()),
    )
    print(f"{len(index_frame)} documents measured -> {out}")


if __name__ == "__main__":
    main()
