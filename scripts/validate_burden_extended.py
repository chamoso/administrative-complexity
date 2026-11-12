"""Three further tests of the index against approved burden hours.

The headline comparison predicts one collection's burden from one collection's
text with a linear model. That is the strictest reasonable specification and the
right one for comparing measures. It is not the only question worth asking, and
two of its choices are arbitrary.

  Level          Managers act on portfolios, not single forms. Averaging over
                 the collections a body administers cancels the noise in any
                 single estimate and asks whether bodies with more demanding
                 procedures impose more burden.

  Functional     Nothing says the relationship is linear. A model that can
  form           represent curvature and interaction sets an upper bound on
                 what the same measurements support.

  Dimensions     Clarity fails four stability checks, so the index is also
                 built without it.

Expectations were fixed before running: aggregation should raise the
correlation substantially, the non-linear model should raise fit, and dropping
clarity should change almost nothing. All three are reported whatever they show,
alongside the document-level linear result they extend.

    python validate_burden_extended.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import CONFIG, PROCESSED, SEED  # noqa: E402
from adminplex.index.dimensions import DIMENSION_KEYS  # noqa: E402
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.validate.extended")
READ = CONFIG["analysis"]["readability_indices"]
FOLDS = CONFIG["analysis"]["cv_folds"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", default="outputs/external/collections")
    ap.add_argument("--index", default="pca")
    ap.add_argument("--min-collections", type=int, default=5,
                    help="bodies with fewer collections than this are not aggregated")
    ap.add_argument("--out", default="outputs")
    return ap.parse_args()


def cv_mean(X: pd.DataFrame, y: pd.Series, model, groups=None) -> tuple[np.ndarray, float]:
    X = X.astype(float)
    X = X.fillna(X.mean())
    if groups is not None:
        splitter = GroupKFold(n_splits=min(FOLDS, groups.nunique()))
        folds = cross_val_score(model, X.values, y.values, cv=splitter, groups=groups, scoring="r2")
    else:
        splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        folds = cross_val_score(model, X.values, y.values, cv=splitter, scoring="r2")
    return folds, float(folds.mean())


def paired(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    sd = diff.std(ddof=1)
    t = diff.mean() / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1))) if np.isfinite(t) else float("nan")
    return {"difference": float(diff.mean()), "t": float(t), "p_value": p,
            "folds_favouring": int((diff > 0).sum()), "n_folds": n}


def main() -> None:
    args = parse_args()
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    ext = Path(args.external)

    criterion = pd.read_parquet(PROCESSED / "validation_collections.parquet")
    features = pd.read_parquet(ext / "features.parquet")
    indices = pd.read_parquet(ext / "candidate_indices.parquet")
    dims = pd.read_parquet(ext / "dimension_scores.parquet")

    frame = (criterion.merge(features, on="doc_id")
                      .merge(indices, on="doc_id")
                      .merge(dims, on="doc_id", suffixes=("", "_dim")))
    frame = frame[frame.log_hours_per_response.notna() & np.isfinite(frame.log_hours_per_response)]
    frame["log_words"] = np.log1p(frame.n_words)
    y = frame.log_hours_per_response
    dim_cols = [c for c in DIMENSION_KEYS if c in frame.columns and frame[c].notna().any()]
    read_cols = [c for c in READ if c in frame.columns]
    LOG.info("%d collections, %d dimensions, %d bodies",
             len(frame), len(dim_cols), frame.department.nunique())

    rows: list[dict] = []
    linear, boosted = LinearRegression(), HistGradientBoostingRegressor(random_state=SEED)

    # 1. Functional form, at document level.
    specs = {
        "index alone": frame[[args.index]],
        "fifteen dimensions": frame[dim_cols],
        "fourteen dimensions (no clarity)": frame[[c for c in dim_cols if c != "clarity"]],
        "eight readability indices": frame[read_cols],
    }
    fold_scores: dict[str, np.ndarray] = {}
    for name, X in specs.items():
        for label, model in (("linear", linear), ("gradient boosting", boosted)):
            folds, mean = cv_mean(X, y, model)
            fold_scores[f"{name} | {label}"] = folds
            rows.append({"level": "collection", "specification": name, "model": label,
                         "n": len(X), "cv_r2": mean, "cv_r2_sd": float(folds.std())})
            LOG.info("  %-34s %-18s CV-R2 = %.4f", name, label, mean)

    # 2. Level of analysis: one row per body.
    bodies = frame.dropna(subset=["department"]).groupby("department")
    keep = bodies.size()[bodies.size() >= args.min_collections].index
    agg = (frame[frame.department.isin(keep)]
           .groupby("department")
           .agg(**{c: (c, "mean") for c in dim_cols + read_cols + [args.index]},
                log_hours=("log_hours_per_response", "mean"),
                n_collections=("doc_id", "size"))
           .reset_index())
    LOG.info("%d bodies with at least %d collections", len(agg), args.min_collections)

    if len(agg) >= 30:
        ya = agg.log_hours
        for name, X in {
            "index alone": agg[[args.index]],
            "fifteen dimensions": agg[dim_cols],
            "eight readability indices": agg[read_cols],
        }.items():
            folds, mean = cv_mean(X, ya, linear)
            fold_scores[f"body-level {name}"] = folds
            rho = float(X.iloc[:, 0].corr(ya, method="spearman")) if X.shape[1] == 1 else float("nan")
            rows.append({"level": "body", "specification": name, "model": "linear",
                         "n": len(agg), "cv_r2": mean, "cv_r2_sd": float(folds.std()),
                         "spearman": rho})
            LOG.info("  body-level %-28s CV-R2 = %.4f  rho = %.3f", name, mean, rho)

    table = pd.DataFrame(rows)
    save_table(table, out / "burden_extended")

    contrasts = [
        ("index alone | gradient boosting", "index alone | linear"),
        ("fifteen dimensions | gradient boosting", "fifteen dimensions | linear"),
        ("fourteen dimensions (no clarity) | linear", "fifteen dimensions | linear"),
        ("fifteen dimensions | gradient boosting", "eight readability indices | gradient boosting"),
    ]
    tests = [{"comparison": f"{a} vs {b}", **paired(fold_scores[a], fold_scores[b])}
             for a, b in contrasts if a in fold_scores and b in fold_scores]
    if tests:
        save_table(pd.DataFrame(tests), out / "burden_extended_tests")

    print(table.round(4).to_string(index=False))
    if tests:
        print()
        print(pd.DataFrame(tests).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
