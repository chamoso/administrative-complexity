"""Recompute the reported results that no saved table records.

These numbers came from ad hoc analyses whose output was not kept. This script
recomputes each from the data, read-only (nothing is written), so the reported
values can be checked against them. Where the original procedure is not documented,
the plausible variants are all computed and printed side by side.

Run from the repository root on the cluster:

    python scripts/recompute_untabulated_results.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from adminplex.config import CONFIG, INTERIM, PROCESSED, SEED  # noqa: E402
from adminplex.index.build import BUILDERS  # noqa: E402
from adminplex.index.dimensions import DIMENSION_KEYS, build_dimension_scores  # noqa: E402

READ = CONFIG["analysis"]["readability_indices"]
FOLDS = CONFIG["analysis"]["cv_folds"]
SCALING = CONFIG["index"]["scaling"]
EXT = ROOT / "outputs" / "external" / "collections"
SCORES = ROOT / "outputs" / "llm_scores"


def load_tag(tag: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(SCORES / f"scores_{tag}_shard*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["doc_id"])
    if "parsed_ok" in frame.columns:
        frame = frame[frame.parsed_ok.fillna(False).astype(bool)]
    keep = ["doc_id", *[c for c in frame.columns if c.startswith("llm_")]]
    return frame[keep].drop_duplicates(subset="doc_id", keep="first")


def cv_folds(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    X = X.astype(float)
    X = X.fillna(X.mean())
    cv = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    return cross_val_score(LinearRegression(), X.values, y.values, cv=cv, scoring="r2")


def paired_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy import stats
    d = a - b
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(t), float(2 * (1 - stats.t.cdf(abs(t), df=len(d) - 1)))


def head(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Collection-level frame, exactly as validate_burden_extended.py builds it.
criterion = pd.read_parquet(PROCESSED / "validation_collections.parquet")
ext_features = pd.read_parquet(EXT / "features.parquet")
ext_index = pd.read_parquet(EXT / "candidate_indices.parquet")
ext_dims = pd.read_parquet(EXT / "dimension_scores.parquet")
frame = (criterion.merge(ext_features, on="doc_id")
                  .merge(ext_index, on="doc_id")
                  .merge(ext_dims, on="doc_id", suffixes=("", "_dim")))
frame = frame[frame.log_hours_per_response.notna() & np.isfinite(frame.log_hours_per_response)].copy()
frame["log_words"] = np.log1p(frame.n_words)
y = frame.log_hours_per_response
read_cols = [c for c in READ if c in frame.columns]
head(f"collections: n = {len(frame)} (reported 7,020); readability columns: {read_cols}")

# 1. Readability scores in possible ranges ---------------------------------
head("1. Collections with readability scores in possible ranges (reported: 6,493 with Flesch in [0, 100]; R2 0.004)")
print(frame[read_cols].describe(percentiles=[.001, .01, .99, .999]).T.round(2).to_string())
grade = [c for c in read_cols if c not in ("flesch", "lix", "rix")]
candidates = {
    "flesch in [0,100]": frame.flesch.between(0, 100),
    "flesch <= 121.22 and grades >= 0": (frame.flesch <= 121.22) & (frame[grade] >= 0).all(axis=1),
    "flesch in [0,121.22] and grades >= 0": frame.flesch.between(0, 121.22) & (frame[grade] >= 0).all(axis=1),
    "flesch in [0,100] and grades in [0,30]": frame.flesch.between(0, 100) & frame[grade].apply(lambda s: s.between(0, 30)).all(axis=1),
    "all finite and >= 0 (grades, lix, rix)": (frame[[c for c in read_cols if c != "flesch"]] >= 0).all(axis=1) & np.isfinite(frame[read_cols]).all(axis=1),
    "flesch >= 0 and grades >= 0": (frame.flesch >= 0) & (frame[grade] >= 0).all(axis=1),
}
for label, mask in candidates.items():
    sub = frame[mask]
    r2 = cv_folds(sub[read_cols], sub.log_hours_per_response).mean() if len(sub) > 100 else float("nan")
    print(f"  {label:45s} n = {int(mask.sum()):5d}   readability CV-R2 = {r2:.4f}")

# 2. Body level -------------------------------------------------------------
head("2. Bodies with >= 5 collections (reported: 36; rho 0.47, 95% CI 0.20-0.69; best readability 0.32; b = 0.099, p = 0.027; length p = 0.43)")
bodies = frame.dropna(subset=["department"])
size = bodies.groupby("department").size()
keep = size[size >= 5].index
agg = (bodies[bodies.department.isin(keep)].groupby("department")
       .agg(pca=("pca", "mean"), log_hours=("log_hours_per_response", "mean"),
            log_words=("log_words", "mean"), n_words=("n_words", "mean"),
            **{c: (c, "mean") for c in read_cols})
       .reset_index())
agg["log_mean_words"] = np.log1p(agg.n_words)
rho = agg.pca.corr(agg.log_hours, method="spearman")
print(f"  bodies = {len(agg)}, rho(index, hours) = {rho:.3f}")
for draws in (1000, 2000):
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(draws):
        s = agg.sample(len(agg), replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
        boots.append(s.pca.corr(s.log_hours, method="spearman"))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    print(f"  bootstrap ({draws} draws, resampling bodies): 95% CI {lo:.2f} to {hi:.2f}")
rr = {c: agg[c].corr(agg.log_hours, method="spearman") for c in read_cols}
print("  readability rho by index:", {k: round(v, 3) for k, v in rr.items()})
print(f"  best readability rho: max = {max(rr.values()):.3f}, max |rho| = {max(abs(v) for v in rr.values()):.3f}")
for lw in ("log_words", "log_mean_words"):
    fit = smf.ols(f"log_hours ~ pca + {lw}", data=agg).fit()
    print(f"  OLS log_hours ~ index + {lw:14s}: b_index = {fit.params['pca']:.3f} (p = {fit.pvalues['pca']:.3f}), "
          f"p_length = {fit.pvalues[lw]:.3f}")

# 3. Index rebuilt from other models -----------------------------------------
head("3. Index rebuilt from the 8B and 70B models (reported: rho 0.976 both; mean |diff| 2.8 and 3.3; >5 points 15.4% and 22.3%)")
corpus_features = pd.read_parquet(INTERIM / "features.parquet")
main = load_tag("main")
reference = corpus_features.merge(main, on="doc_id", how="left")
corpus_dims = pd.read_parquet(ROOT / "outputs" / "tables" / "dimension_scores.parquet")
usable = [k for k in DIMENSION_KEYS if k in corpus_dims.columns and corpus_dims[k].notna().any()]
weights = pd.Series(BUILDERS["pca"](corpus_dims.set_index("doc_id")[usable]).weights).reindex(usable).fillna(0.0)


def apply(dims: pd.DataFrame) -> pd.Series:
    values = dims.set_index("doc_id")[usable].fillna(50.0)
    return (values * weights).sum(axis=1) / weights.abs().sum()


published = pd.read_parquet(ROOT / "outputs" / "tables" / "candidate_indices.parquet").set_index("doc_id").pca
main_applied = apply(corpus_dims)


def compare(a: pd.Series, b: pd.Series) -> str:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = (j.a - j.b).abs()
    return (f"n = {len(j):5d}  rho = {j.a.corr(j.b, method='spearman'):.3f}  "
            f"mean|diff| = {d.mean():.2f}  share > 5 = {100 * (d > 5).mean():.1f}%")


for tag, label in (("alt_model", "8B"), ("large_model", "70B")):
    alt = load_tag(tag)
    if alt.empty:
        print(f"  {label}: no scores")
        continue
    ids = alt.doc_id
    llm_cols = [c for c in alt.columns if c.startswith("llm_")]
    base = corpus_features[corpus_features.doc_id.isin(ids)]
    sub_main = base.merge(main, on="doc_id", how="left")
    sub_alt = base.merge(alt, on="doc_id", how="left")
    # B1: corpus reference scale and corpus weights (as measure_external.py does)
    b1 = apply(build_dimension_scores(sub_alt, scaling=SCALING, reference=reference))
    print(f"  {label} B1 corpus scale + corpus weights vs published-weights main: {compare(b1, main_applied)}")
    # A1: ranks within the subsample for both, corpus weights
    a1_alt = apply(build_dimension_scores(sub_alt, scaling=SCALING))
    a1_main = apply(build_dimension_scores(sub_main, scaling=SCALING))
    print(f"  {label} A1 ranks within subsample + corpus weights:              {compare(a1_alt, a1_main)}")
    # A2: ranks within the subsample, PCA refitted on each (0-100)
    d_alt = build_dimension_scores(sub_alt, scaling=SCALING).set_index("doc_id")[usable]
    d_main = build_dimension_scores(sub_main, scaling=SCALING).set_index("doc_id")[usable]
    a2_alt = pd.Series(BUILDERS["pca"](d_alt).scores.values, index=d_alt.index)
    a2_main = pd.Series(BUILDERS["pca"](d_main).scores.values, index=d_main.index)
    print(f"  {label} A2 ranks within subsample + PCA refitted:                {compare(a2_alt, a2_main)}")
    # C: alternative scores substituted in the corpus, whole index rebuilt
    swapped = main.set_index("doc_id").copy()
    swapped.loc[swapped.index.intersection(alt.set_index("doc_id").index), llm_cols] = \
        alt.set_index("doc_id").loc[swapped.index.intersection(alt.set_index("doc_id").index), llm_cols]
    full = corpus_features.merge(swapped.reset_index(), on="doc_id", how="left")
    dims_full = build_dimension_scores(full, scaling=SCALING).set_index("doc_id")[usable]
    c_idx = pd.Series(BUILDERS["pca"](dims_full).scores.values, index=dims_full.index)
    print(f"  {label} C  substituted in corpus, PCA rebuilt, vs published:     {compare(c_idx.reindex(ids), published.reindex(ids))}")

# 4. Collections index from the 70B ratings ----------------------------------
head("4. Collection index rebuilt from the 70B ratings (reported: CV-R2 0.049 vs 0.062, t = -7.9; averaging adds nothing)")
large = load_tag("collections_large")
if large.empty:
    print("  no collections_large scores")
else:
    feats = ext_features.drop(columns=[c for c in ext_features.columns if c.startswith("llm_")])
    feats_large = feats.merge(large, on="doc_id", how="inner")
    idx_large = apply(build_dimension_scores(feats_large, scaling=SCALING, reference=reference))
    j = frame.set_index("doc_id").join(idx_large.rename("pca_large"), how="inner")
    print(f"  collections with 70B scores: {len(j)}")
    f_main = cv_folds(j[["pca"]], j.log_hours_per_response)
    f_large = cv_folds(j[["pca_large"]], j.log_hours_per_response)
    j["pca_avg"] = (j.pca + j.pca_large) / 2
    f_avg = cv_folds(j[["pca_avg"]], j.log_hours_per_response)
    t, p = paired_t(f_large, f_main)
    print(f"  CV-R2 main {f_main.mean():.4f} | 70B {f_large.mean():.4f} | average {f_avg.mean():.4f} | "
          f"70B vs main t = {t:.2f} (p = {p:.2g})")
    print(f"  rank correlation main vs 70B index on collections: {j.pca.corr(j.pca_large, method='spearman'):.3f}")
