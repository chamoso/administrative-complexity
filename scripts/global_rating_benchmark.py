"""Global-rating benchmark and human-validation intervals (read-only).

    python scripts/global_rating_benchmark.py

5. Benchmark: one global model rating of administrative complexity
   (scores_global, from scripts/hpc_score_global.py) against the fifteen dimensions.
6. Human validation: bootstrap intervals for the leave-one-out correlations of the
   coders and of the index, their difference, and a restricted-range check.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, cross_val_predict

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from adminplex.config import CONFIG, PROCESSED, SEED  # noqa: E402
from adminplex.index.dimensions import DIMENSION_KEYS  # noqa: E402

READ = CONFIG["analysis"]["readability_indices"]
EXT = ROOT / "outputs" / "external" / "collections"
DIMS = list(DIMENSION_KEYS)
JUDGED = ["cognitive_load", "legal_knowledge", "clarity"]

# --- 5. Global rating benchmark ------------------------------------------------
rows = []
for p in sorted(glob.glob(str(ROOT / "outputs" / "llm_scores" / "scores_global_shard*.jsonl"))):
    with open(p, encoding="utf-8") as fh:
        rows += [json.loads(line) for line in fh if line.strip()]
glob_scores = pd.DataFrame(rows)
print(f"\n=== 5. Global model rating (n = {len(glob_scores)}, parsed {glob_scores.parsed_ok.mean():.4f}) ===")
glob_scores = glob_scores[glob_scores.parsed_ok.fillna(False).astype(bool)][["doc_id", "llm_global_complexity"]]

crit = pd.read_parquet(PROCESSED / "validation_collections.parquet")
frame = (crit.merge(pd.read_parquet(EXT / "features.parquet"), on="doc_id")
              .merge(pd.read_parquet(EXT / "candidate_indices.parquet"), on="doc_id")
              .merge(pd.read_parquet(EXT / "dimension_scores.parquet"), on="doc_id", suffixes=("", "_dim"))
              .merge(glob_scores, on="doc_id", how="left"))
frame = frame[frame.log_hours_per_response.notna() & np.isfinite(frame.log_hours_per_response)].copy()
y = frame.log_hours_per_response.values
n = len(frame)
print(f"collections: {n}; with a global rating: {frame.llm_global_complexity.notna().sum()}")
print("distribution of the global rating:", frame.llm_global_complexity.value_counts().sort_index().to_dict())
print("Spearman of the global rating with the index and the judged dimensions:",
      {c: round(frame.llm_global_complexity.corr(frame[c], method="spearman"), 3) for c in ["pca", *JUDGED]})


def X(cols):
    m = frame[cols].astype(float)
    return m.fillna(m.mean()).values


SPECS = {
    "global rating": ["llm_global_complexity"],
    "global rating (categories)": None,
    "three model-assessed dimensions": JUDGED,
    "fifteen dimensions": DIMS,
    "fifteen dimensions + global rating": DIMS + ["llm_global_complexity"],
    "readability (8)": READ,
}
dummies = pd.get_dummies(frame.llm_global_complexity.fillna(frame.llm_global_complexity.median()).astype(int),
                         prefix="g", drop_first=True).astype(float).values
cv = KFold(10, shuffle=True, random_state=SEED)
splits = list(RepeatedKFold(n_splits=10, n_repeats=10, random_state=SEED).split(frame))
oof, reps = {}, {}
for name, cols in SPECS.items():
    M = dummies if cols is None else X(cols)
    pred = cross_val_predict(LinearRegression(), M, y, cv=cv)
    oof[name] = pred
    folds = [r2_score(y[te], pred[te]) for _, te in cv.split(M)]
    reps[name] = np.array([r2_score(y[te], LinearRegression().fit(M[tr], y[tr]).predict(M[te])) for tr, te in splits])
    print(f"  {name:36s} CV-R2 = {np.mean(folds):.3f}")
rng = np.random.default_rng(SEED)
ratio = (n / 10) / (n - n / 10)
for a, b in [("fifteen dimensions", "global rating"), ("three model-assessed dimensions", "global rating"),
             ("fifteen dimensions + global rating", "fifteen dimensions"), ("global rating", "readability (8)")]:
    d = reps[a] - reps[b]
    t = d.mean() / np.sqrt((1 / len(d) + ratio) * d.var(ddof=1))
    p = 2 * (1 - stats.t.cdf(abs(t), df=len(d) - 1))
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        diffs.append(r2_score(y[i], oof[a][i]) - r2_score(y[i], oof[b][i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  {a} - {b}: {d.mean():+.3f}  corrected t = {t:.2f} (p = {p:.2g})  bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")

fe = frame.dropna(subset=["department"])
fe = fe[fe.groupby("department").department.transform("size") >= 10]
D = pd.get_dummies(fe.department.astype(str)).astype(float).values
yv = fe.log_hours_per_response.values
skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
def oos(cols):
    M = np.hstack([D] + ([fe[cols].astype(float).fillna(fe[cols].astype(float).mean()).values] if cols else []))
    return np.array([r2_score(yv[te], LinearRegression().fit(M[tr], yv[tr]).predict(M[te]))
                     for tr, te in skf.split(M, fe.department.astype(str))])
base = oos([])
print(f"  within agencies (agency effects estimated in training): agency effects alone {base.mean():.3f}")
for name in ("global rating", "three model-assessed dimensions", "fifteen dimensions", "fifteen dimensions + global rating"):
    r = oos(SPECS[name])
    print(f"    + {name:36s} {r.mean():.3f} (gain {np.mean(r - base):+.3f}, folds {int(((r - base) > 0).sum())}/10)")

# --- 6. Human validation intervals ---------------------------------------------
print("\n=== 6. Human validation: bootstrap intervals ===")
files = sorted(glob.glob(str(ROOT / "coding" / "ratings_*.csv"))) or sorted(glob.glob(str(ROOT / "ratings" / "ratings_*.csv")))
fr = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
fr = fr[fr.item_id.str.extract(r"#(\d)$")[0].astype(int) == 1].copy()
fr["composite"] = pd.concat([fr.cognitive_load, fr.legal_knowledge, 11 - fr.clarity], axis=1).mean(axis=1)
key = pd.read_csv(ROOT / "coding" / "sample_key.csv")[["doc_id", "model_index"]]
wide = fr.pivot_table(index="doc_id", columns="coder", values="composite").join(key.set_index("doc_id"), how="inner")
coders = [c for c in wide.columns if c != "model_index"]
def loo(w):
    ceil, idx = [], []
    for h in coders:
        others = w[[c for c in coders if c != h]].mean(axis=1).rank()
        ceil.append(w[h].rank().corr(others)); idx.append(w.model_index.rank().corr(others))
    return np.mean(ceil), np.mean(idx)
c0, i0 = loo(wide)
rng6 = np.random.default_rng(SEED)
boot = np.array([loo(wide.iloc[rng6.integers(0, len(wide), len(wide))]) for _ in range(2000)])
print(f"  coder ceiling {c0:.3f} [{np.percentile(boot[:, 0], 2.5):.2f}, {np.percentile(boot[:, 0], 97.5):.2f}]")
print(f"  index         {i0:.3f} [{np.percentile(boot[:, 1], 2.5):.2f}, {np.percentile(boot[:, 1], 97.5):.2f}]")
d = boot[:, 1] - boot[:, 0]
print(f"  difference    {i0 - c0:.3f} [{np.percentile(d, 2.5):.2f}, {np.percentile(d, 97.5):.2f}]")
q = wide.model_index.quantile([0.2, 0.8])
c1, i1 = loo(wide[wide.model_index.between(q.iloc[0], q.iloc[1])])
print(f"  middle 60% of the index range: coder ceiling {c1:.3f}, index {i1:.3f}")
