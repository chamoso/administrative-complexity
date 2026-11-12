"""Experiments 5 and 6: measure the edited procedures and compare them to the originals.

The edited texts are placed in the distribution of the original corpus rather
than ranked among themselves, so a fall in the index means the document became
less complex than the corpus, not merely less complex than the other edits.

Run after hpc_simplify.py has produced the edited texts and hpc_score.py has
assessed them.

    python analyse_simplification.py --simplified outputs/simplify \\
                                     --scores outputs/llm_scores_simplified
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import CONFIG, INTERIM, PROCESSED, SEED  # noqa: E402
from adminplex.features import pipeline as feature_pipeline  # noqa: E402
from adminplex.index.build import BUILDERS  # noqa: E402
from adminplex.index.dimensions import DIMENSION_KEYS, build_dimension_scores  # noqa: E402
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.simplification")
SEPARATOR = "::"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simplified", default="outputs/simplify")
    ap.add_argument("--scores", default="outputs/llm_scores_simplified")
    ap.add_argument("--tag", default="simplified")
    ap.add_argument("--index", default="pca")
    ap.add_argument("--features", default=str(INTERIM / "features.parquet"))
    ap.add_argument("--corpus-scores", default="outputs/llm_scores")
    ap.add_argument("--corpus-tag", default="main")
    ap.add_argument("--min-edit", type=float, default=0.02,
                    help="an edit that changes the text by less than this fraction did not happen")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--write-scoring-input", action="store_true",
                    help="only build the parquet the assessment step needs, then stop")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--max-original-words", type=int, default=None,
                    help="keep only documents short enough to have been edited in full. The "
                         "editing model saw an excerpt of --max-input-chars characters (8,000 by "
                         "default in hpc_simplify.py), so for longer documents the edited text "
                         "covers only part of the original and its length and index are not "
                         "comparable with the full document. 1,000 words stays below 8,000 "
                         "characters. Results are written with a suffix, not over the full tables.")
    return ap.parse_args()


def load_simplified(directory: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(directory) / "simplified_shard*.jsonl")))
    if not files:
        raise SystemExit(f"no edited texts in {directory}")
    rows = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    frame["edit_id"] = frame.doc_id + SEPARATOR + frame.intervention
    LOG.info("%d edited texts over %d documents and %d interventions",
             len(frame), frame.doc_id.nunique(), frame.intervention.nunique())
    return frame


def load_assessed(directory: str, tag: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(directory) / f"scores_{tag}_shard*.jsonl")))
    if not files:
        LOG.warning("no assessed scores for the edited texts; the three judged dimensions "
                    "will be missing from the comparison")
        return pd.DataFrame(columns=["doc_id"])
    rows = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    frame = frame[frame.get("parsed_ok", False).fillna(False).astype(bool)]
    keep = [c for c in frame.columns if c.startswith("llm_")]
    return frame[["doc_id", *keep]].drop_duplicates(subset="doc_id", keep="first")


def bootstrap_ci(values: np.ndarray, draws: int) -> tuple[float, float]:
    if len(values) < 2 or draws <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    args = parse_args()
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)

    edited = load_simplified(args.simplified)

    if args.write_scoring_input:
        scoring = edited.rename(columns={"edit_id": "_id"})[["_id", "title", "text", "language"]]
        scoring = scoring.rename(columns={"_id": "doc_id"})
        path = PROCESSED / "simplified_for_scoring.parquet"
        scoring.to_parquet(path, index=False)
        LOG.info("scoring input written: %s (%d texts)", path, len(scoring))
        return

    # Measure the edited texts exactly as the originals were measured.
    to_measure = edited.rename(columns={"edit_id": "measure_id"})[["measure_id", "text", "language"]]
    to_measure = to_measure.rename(columns={"measure_id": "doc_id"})
    edited_features = feature_pipeline.compute_features(to_measure)
    assessed = load_assessed(args.scores, args.tag)
    if not assessed.empty:
        edited_features = edited_features.merge(assessed, on="doc_id", how="left")

    # The reference must carry the judged measures too, or the three dimensions
    # that dominate the index are silently dropped from the comparison.
    original_features = pd.read_parquet(args.features)
    corpus_assessed = load_assessed(args.corpus_scores, args.corpus_tag)
    if corpus_assessed.empty:
        LOG.error("no assessment for the corpus (tag %r): the comparison would cover only the "
                  "counted dimensions", args.corpus_tag)
    else:
        original_features = original_features.merge(corpus_assessed, on="doc_id", how="left")
    edited_scores = build_dimension_scores(
        edited_features, scaling=CONFIG["index"]["scaling"], reference=original_features
    )
    original_scores = build_dimension_scores(
        original_features, scaling=CONFIG["index"]["scaling"], reference=original_features
    )

    usable = [k for k in DIMENSION_KEYS
              if k in edited_scores.columns and edited_scores[k].notna().any()
              and original_scores[k].notna().any()]
    builder = BUILDERS[args.index]
    # The aggregation is estimated once, on the original corpus, and applied
    # unchanged to the edited texts: refitting it would let the index redefine
    # itself around the intervention.
    reference_result = builder(original_scores.set_index("doc_id")[usable])
    weights = pd.Series(reference_result.weights).reindex(usable).fillna(0.0)

    def apply_index(scores: pd.DataFrame) -> pd.Series:
        values = scores.set_index("doc_id")[usable].fillna(50.0)
        return (values * weights).sum(axis=1) / weights.abs().sum()

    edited_index = apply_index(edited_scores)
    original_index = apply_index(original_scores)

    edited = edited.set_index("edit_id")
    comparison = pd.DataFrame({
        "edit_id": edited_index.index,
        "doc_id": [i.split(SEPARATOR)[0] for i in edited_index.index],
        "intervention": [i.split(SEPARATOR)[1] for i in edited_index.index],
        "index_after": edited_index.values,
    })
    comparison["index_before"] = comparison.doc_id.map(original_index)
    comparison["change"] = comparison.index_after - comparison.index_before
    comparison["words_before"] = comparison.edit_id.map(edited.original_words)
    comparison["words_after"] = comparison.edit_id.map(edited.produced_words)
    comparison["country"] = comparison.edit_id.map(edited.country)
    comparison = comparison.dropna(subset=["index_before", "index_after"])
    save_table(comparison, out / "experiment56_document_level")

    suffix = ""
    if args.max_original_words:
        comparison = comparison[comparison.words_before <= args.max_original_words].copy()
        suffix = f"_max{args.max_original_words}words"
        LOG.info("restricted to %d edits of %d documents with at most %d words",
                 len(comparison), comparison.doc_id.nunique(), args.max_original_words)

    # An instruction that the model declined to carry out is not an
    # intervention, and averaging over the documents it left alone would
    # report the model's compliance as if it were the effect of the edit.
    comparison["edit_size"] = (
        (comparison.words_before - comparison.words_after).abs()
        / comparison.words_before.clip(lower=1)
    )
    comparison["edited"] = comparison.edit_size >= args.min_edit
    for intervention, block in comparison.groupby("intervention"):
        LOG.info("%s: the text actually changed in %d of %d documents (%.0f%%)",
                 intervention, int(block.edited.sum()), len(block), 100 * block.edited.mean())

    rows = []
    for intervention, block in comparison.groupby("intervention"):
        change = block.change.to_numpy()
        applied = block[block.edited]
        lo, hi = bootstrap_ci(change, args.bootstrap)
        alo, ahi = bootstrap_ci(applied.change.to_numpy(), args.bootstrap)
        rows.append({
            "intervention": intervention,
            "n": int(len(block)),
            "n_actually_edited": int(len(applied)),
            "share_actually_edited": float(block.edited.mean()),
            "mean_change_where_edited": float(applied.change.mean()) if len(applied) else float("nan"),
            "ci_low_where_edited": alo,
            "ci_high_where_edited": ahi,
            "mean_index_before": float(block.index_before.mean()),
            "mean_index_after": float(block.index_after.mean()),
            "mean_change": float(change.mean()),
            "median_change": float(np.median(change)),
            "ci_low": lo,
            "ci_high": hi,
            "share_reduced": float((change < 0).mean()),
            "mean_word_change": float((block.words_after - block.words_before).mean()),
            "median_word_ratio": float((block.words_after / block.words_before.clip(lower=1)).median()),
        })
    summary = pd.DataFrame(rows)
    save_table(summary, out / f"experiment56_summary{suffix}")

    # Which dimensions each intervention actually moved.
    dim_rows = []
    edited_dims = edited_scores.set_index("doc_id")
    original_dims = original_scores.set_index("doc_id")
    for intervention, block in comparison.groupby("intervention"):
        for key in usable:
            after = edited_dims.loc[block.edit_id, key].to_numpy()
            before = original_dims.loc[block.doc_id, key].to_numpy()
            delta = after - before
            dim_rows.append({
                "intervention": intervention,
                "dimension": key,
                "mean_change": float(np.nanmean(delta)),
                "share_reduced": float(np.nanmean(delta < 0)),
            })
    by_dimension = pd.DataFrame(dim_rows).pivot(
        index="dimension", columns="intervention", values="mean_change"
    ).reset_index()
    save_table(by_dimension, out / f"experiment56_by_dimension{suffix}")

    print(summary.round(3).to_string(index=False))
    print()
    print(by_dimension.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
