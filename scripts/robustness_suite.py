"""How far the headline result depends on the analytical choices behind it.

The analysis makes a comparative claim from a chain of decisions: which documents to
keep, how to scale the dimensions, which aggregation to use, how to split the
folds. Each is defensible on its own and the reader has no way to tell how much
any one of them is carrying. This runs the same comparison under alternatives to
each, and reports every result whether it helps or not.

  Long documents      excluded above the ninetieth and seventy-fifth percentiles,
                      in case the association is driven by a tail of long texts.
  Unseen agencies     folds grouped by department, so every test fold contains
                      only bodies absent from training.
  Aggregation         the comparison repeated with each of the five candidate
                      indices rather than the principal component alone.
  Relative importance permutation importance per dimension, which does not assume
                      the linear coefficients are interpretable.
  Interval            bootstrap confidence intervals on every fit, not only the
                      spread across folds.

    python robustness_suite.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import CONFIG, PROCESSED, SEED  # noqa: E402
from adminplex.index.build import BUILDERS  # noqa: E402
from adminplex.index.dimensions import DIMENSION_KEYS  # noqa: E402
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.robustness")
READ = CONFIG["analysis"]["readability_indices"]
FOLDS = CONFIG["analysis"]["cv_folds"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", default="outputs/external/collections")
    ap.add_argument("--index", default="pca")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--out", default="outputs")
    return ap.parse_args()


def fit(X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None) -> np.ndarray:
    X = X.astype(float)
    X = X.fillna(X.mean())
    if groups is not None:
        cv = GroupKFold(n_splits=min(FOLDS, groups.nunique()))
        return cross_val_score(LinearRegression(), X.values, y.values, cv=cv,
                               groups=groups, scoring="r2")
    cv = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    return cross_val_score(LinearRegression(), X.values, y.values, cv=cv, scoring="r2")


def interval(X: pd.DataFrame, y: pd.Series, draws: int) -> tuple[float, float]:
    """Bootstrap interval on out-of-sample fit, resampling documents."""
    rng = np.random.default_rng(SEED)
    X = X.astype(float).fillna(X.astype(float).mean())
    scores = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        Xb, yb = X.iloc[idx], y.iloc[idx]
        if np.linalg.matrix_rank(Xb.values) < Xb.shape[1]:
            continue
        scores.append(fit(Xb, yb).mean())
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main() -> None:
    args = parse_args()
    out = Path(args.out) / "tables"
    ext = Path(args.external)

    frame = (pd.read_parquet(PROCESSED / "validation_collections.parquet")
             .merge(pd.read_parquet(ext / "features.parquet"), on="doc_id")
             .merge(pd.read_parquet(ext / "candidate_indices.parquet"), on="doc_id")
             .merge(pd.read_parquet(ext / "dimension_scores.parquet"), on="doc_id",
                    suffixes=("", "_dim")))
    frame = frame[frame.log_hours_per_response.notna() & np.isfinite(frame.log_hours_per_response)]
    dims = [c for c in DIMENSION_KEYS if c in frame.columns and frame[c].notna().any()]
    reads = [c for c in READ if c in frame.columns]
    LOG.info("%d collections, %d dimensions, %d departments", len(frame), len(dims),
             frame.department.nunique())

    rows = []

    def record(label: str, subset: pd.DataFrame, groups: pd.Series | None = None) -> None:
        y = subset.log_hours_per_response
        for name, cols in (("fifteen dimensions", dims), ("index alone", [args.index]),
                           ("readability", reads)):
            folds = fit(subset[cols], y, groups)
            lo, hi = interval(subset[cols], y, args.bootstrap) if groups is None else (np.nan, np.nan)
            rows.append({"variant": label, "specification": name, "n": len(subset),
                         "cv_r2": float(folds.mean()), "cv_r2_sd": float(folds.std()),
                         "ci_low": lo, "ci_high": hi})
        LOG.info("  %-34s dimensions %.4f  index %.4f  readability %.4f", label,
                 rows[-3]["cv_r2"], rows[-2]["cv_r2"], rows[-1]["cv_r2"])

    record("all collections", frame)
    for pct in (0.90, 0.75):
        cut = frame.n_words.quantile(pct)
        record(f"excluding the longest {100 - int(pct * 100)} per cent",
               frame[frame.n_words <= cut])
    record("folds grouped by department", frame, groups=frame.department.fillna("unknown"))

    # Does the choice of aggregation matter?
    corpus_dims = pd.read_parquet("outputs/tables/dimension_scores.parquet")
    usable = [k for k in DIMENSION_KEYS if k in corpus_dims.columns and corpus_dims[k].notna().any()]
    external = pd.read_parquet(ext / "dimension_scores.parquet").set_index("doc_id")
    y = frame.log_hours_per_response
    for name in CONFIG["index"]["candidate_models"]:
        try:
            weights = pd.Series(BUILDERS[name](corpus_dims.set_index("doc_id")[usable]).weights)
        except Exception as exc:
            LOG.warning("  aggregation %s unavailable: %s", name, exc)
            continue
        weights = weights.reindex(usable).fillna(0.0)
        values = external.reindex(frame.doc_id)[usable].fillna(50.0)
        score = (values * weights).sum(axis=1) / weights.abs().sum()
        folds = fit(pd.DataFrame({"aci": score.to_numpy()}), y)
        rows.append({"variant": f"aggregation: {name}", "specification": "index alone",
                     "n": len(frame), "cv_r2": float(folds.mean()),
                     "cv_r2_sd": float(folds.std()), "ci_low": np.nan, "ci_high": np.nan})
        LOG.info("  aggregation %-16s index alone %.4f", name, rows[-1]["cv_r2"])

    save_table(pd.DataFrame(rows), out / "robustness_variants")

    # Which dimensions carry the prediction, without assuming the coefficients
    # can be read directly.
    X = frame[dims].astype(float).fillna(frame[dims].astype(float).mean())
    model = LinearRegression().fit(X.values, y.values)
    result = permutation_importance(model, X.values, y.values, n_repeats=30,
                                   random_state=SEED, scoring="r2")
    importance = (pd.DataFrame({"dimension": dims,
                                "importance": result.importances_mean,
                                "sd": result.importances_std})
                  .sort_values("importance", ascending=False))
    save_table(importance, out / "robustness_importance")

    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print()
    print("Permutation importance, drop in out-of-sample fit when one dimension is shuffled:")
    print(importance.head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
