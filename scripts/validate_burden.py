"""Test whether the index explains administrative burden better than the
traditional textual measures.

The criterion is the burden the Office of Management and Budget approved for
each information collection: hours the public is estimated to spend complying.
It is produced by a different institution, for a different purpose, years apart
from the text being measured, which is what makes it a test rather than a
restatement.

Competing specifications are estimated on the same documents and compared on
out-of-sample fit. A measure that only fits in sample has not been validated.

    python validate_burden.py --level parts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import CONFIG, PROCESSED, SEED  # noqa: E402
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.validate")

READABILITY = CONFIG["analysis"]["readability_indices"]
LENGTH_ONLY = ["log_words"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=("parts", "collections"), default="parts")
    ap.add_argument("--features", default="data/interim/features.parquet")
    ap.add_argument("--indices", default="outputs/tables/candidate_indices.parquet")
    ap.add_argument("--dimensions", default="outputs/tables/dimension_scores.parquet")
    ap.add_argument("--index-name", default="pca", help="which candidate index is the ACI")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--bootstrap", type=int, default=CONFIG["index"]["bootstrap_draws"])
    return ap.parse_args()


def load_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    """Documents with both a measurement and an externally produced criterion."""
    if args.level == "parts":
        frame = pd.read_parquet(PROCESSED / "validation_parts.parquet")
        criterion = "log_median_hours_per_response"
    else:
        frame = pd.read_parquet(PROCESSED / "validation_collections.parquet")
        criterion = "log_hours_per_response"

    features = pd.read_parquet(args.features)
    indices = pd.read_parquet(args.indices)
    dimensions = pd.read_parquet(args.dimensions)

    merged = (
        frame.merge(features, on="doc_id", how="inner")
        .merge(indices, on="doc_id", how="inner")
        .merge(dimensions, on="doc_id", how="inner", suffixes=("", "_dim"))
    )
    merged["log_words"] = np.log1p(merged["n_words"].astype(float))
    LOG.info("%s level: %d documents with both a measurement and a burden estimate",
             args.level, len(merged))
    return merged, criterion


def cv_fold_scores(X: pd.DataFrame, y: pd.Series, folds: int) -> np.ndarray:
    """Out-of-sample R^2 in each fold, on a split shared by every specification."""
    if X.empty or len(X) < folds * 2:
        return np.array([])
    cv = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    return cross_val_score(LinearRegression(), X.values, y.values, cv=cv, scoring="r2")


def cv_r2(X: pd.DataFrame, y: pd.Series, folds: int) -> tuple[float, float]:
    scores = cv_fold_scores(X, y, folds)
    if scores.size == 0:
        return float("nan"), float("nan")
    return float(scores.mean()), float(scores.std())


def paired_comparison(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> dict[str, float | str]:
    """Compare two specifications fold by fold on the same splits.

    Two mean R² values a few thousandths apart, each with a spread across folds
    larger than the gap, are not distinguishable, and reporting the larger one
    as better would be reading noise. Because both specifications are evaluated
    on identical folds the difference can be tested pairwise, which is far more
    sensitive than comparing the two means and their spreads separately.
    """
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return {"comparison": f"{name_a} vs {name_b}", "n_folds": 0}
    diff = a - b
    n = len(diff)
    sd = float(diff.std(ddof=1))
    mean = float(diff.mean())
    t = mean / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    from scipy import stats

    p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1))) if np.isfinite(t) else float("nan")
    return {
        "comparison": f"{name_a} vs {name_b}",
        "n_folds": n,
        "mean_difference": mean,
        "sd_of_difference": sd,
        "t": float(t),
        "p_value": p,
        "folds_favouring_first": int((diff > 0).sum()),
        "distinguishable": "yes" if np.isfinite(p) and p < 0.05 else "no",
    }


def in_sample_fit(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    model = LinearRegression().fit(X.values, y.values)
    pred = model.predict(X.values)
    resid = y.values - pred
    n, k = len(y), X.shape[1]
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y.values - y.values.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")
    aic = n * np.log(rss / n) + 2 * (k + 1) if rss > 0 else float("nan")
    bic = n * np.log(rss / n) + np.log(n) * (k + 1) if rss > 0 else float("nan")
    return {"r2": r2, "adj_r2": adj, "aic": float(aic), "bic": float(bic), "n_predictors": k}


def bootstrap_r2(X: pd.DataFrame, y: pd.Series, draws: int) -> tuple[float, float]:
    if draws <= 0 or X.empty:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    n = len(y)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        Xb, yb = X.values[idx], y.values[idx]
        if np.linalg.matrix_rank(Xb) < Xb.shape[1]:
            continue
        model = LinearRegression().fit(Xb, yb)
        values.append(model.score(Xb, yb))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> None:
    args = parse_args()
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)

    frame, criterion = load_frame(args)
    frame = frame[frame[criterion].notna() & np.isfinite(frame[criterion])]
    if len(frame) < 50:
        LOG.error("only %d usable documents; the comparison would not mean anything", len(frame))
        return
    y = frame[criterion]

    aci = args.index_name
    readability_present = [c for c in READABILITY if c in frame.columns]
    dimension_cols = [c for c in frame.columns if c in CONFIG["index"]["dimensions"]]

    # Each specification answers a different question, and the order matters:
    # length alone is the null a sceptic would propose, readability is the
    # measure the field already uses, and the last two ask whether the index
    # adds anything once readability is already in the model.
    specifications: dict[str, list[str]] = {
        "length_only": [c for c in LENGTH_ONLY if c in frame.columns],
        "readability": readability_present,
        "readability_and_length": readability_present + [c for c in LENGTH_ONLY if c in frame.columns],
        "aci_only": [aci] if aci in frame.columns else [],
        "aci_and_readability": ([aci] if aci in frame.columns else []) + readability_present,
        "dimensions": dimension_cols,
    }

    folds = CONFIG["analysis"]["cv_folds"]
    rows = []
    fold_scores: dict[str, np.ndarray] = {}
    for name, cols in specifications.items():
        cols = [c for c in cols if c in frame.columns]
        if not cols:
            LOG.warning("specification %r has no available predictors, skipped", name)
            continue
        X = frame[cols].astype(float)
        X = X.fillna(X.mean())
        fit = in_sample_fit(X, y)
        folds_r2 = cv_fold_scores(X, y, folds)
        fold_scores[name] = folds_r2
        cv_mean = float(folds_r2.mean()) if folds_r2.size else float("nan")
        cv_sd = float(folds_r2.std()) if folds_r2.size else float("nan")
        lo, hi = bootstrap_r2(X, y, args.bootstrap)
        rows.append({
            "specification": name,
            **fit,
            "cv_r2": cv_mean,
            "cv_r2_sd": cv_sd,
            "r2_ci_low": lo,
            "r2_ci_high": hi,
        })
        LOG.info("  %-24s R2=%.3f  adjR2=%.3f  CV-R2=%.3f", name, fit["r2"], fit["adj_r2"], cv_mean)

    comparison = pd.DataFrame(rows).sort_values("cv_r2", ascending=False)
    save_table(comparison, out / f"burden_model_comparison_{args.level}")

    # The claims made are differences between specifications, so the
    # differences are tested, not just displayed side by side.
    contrasts = [
        ("aci_only", "readability"),
        ("aci_and_readability", "readability"),
        ("dimensions", "aci_only"),
        ("dimensions", "readability"),
        ("aci_only", "length_only"),
    ]
    tests = [
        paired_comparison(fold_scores[a], fold_scores[b], a, b)
        for a, b in contrasts
        if a in fold_scores and b in fold_scores
    ]
    if tests:
        table = pd.DataFrame(tests)
        save_table(table, out / f"burden_model_tests_{args.level}")
        print()
        print("Paired comparison on identical folds:")
        print(table.round(4).to_string(index=False))

    # Correlation of every candidate measure with the criterion, so the reader
    # can see the raw association before any model is fitted.
    candidates = [c for c in [aci, *readability_present, *LENGTH_ONLY] if c in frame.columns]
    correlations = pd.DataFrame({
        "measure": candidates,
        "pearson": [float(frame[c].astype(float).corr(y)) for c in candidates],
        "spearman": [float(frame[c].astype(float).corr(y, method="spearman")) for c in candidates],
    }).sort_values("spearman", key=abs, ascending=False)
    save_table(correlations, out / f"burden_correlations_{args.level}")

    print(comparison.round(4).to_string(index=False))
    print()
    print(correlations.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
