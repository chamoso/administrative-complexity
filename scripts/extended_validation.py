"""Extended validation analyses.

Read-only. Run from the repository root on the cluster:

    python scripts/extended_validation.py

1. Within-agency validity: agency fixed effects, in sample (cluster-robust) and
   out of sample (folds stratified by agency, fixed effects estimated in training).
2. Rule-based versus model-assessed dimensions.
3. Inference: corrected resampled t-test (Nadeau and Bengio, 2003) over 10x10
   repeated cross-validation, and a bootstrap over pooled out-of-sample predictions.
4. Trends in the United Kingdom by subperiod and by document type.
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
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, cross_val_predict

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from adminplex.config import CONFIG, INTERIM, PROCESSED, SEED  # noqa: E402
from adminplex.index.dimensions import DIMENSION_KEYS, build_dimension_scores  # noqa: E402

READ = CONFIG["analysis"]["readability_indices"]
SCALING = CONFIG["index"]["scaling"]
EXT = ROOT / "outputs" / "external" / "collections"
JUDGED = ["cognitive_load", "legal_knowledge", "clarity"]
RULE = [k for k in DIMENSION_KEYS if k not in JUDGED]


def head(t):
    print(f"\n=== {t} ===")


def load_tag(tag):
    rows = []
    for p in sorted(glob.glob(str(ROOT / "outputs" / "llm_scores" / f"scores_{tag}_shard*.jsonl"))):
        with open(p, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    f = pd.DataFrame(rows)
    f = f[f.parsed_ok.fillna(False).astype(bool)]
    return f[["doc_id", *[c for c in f.columns if c.startswith("llm_")]]].drop_duplicates("doc_id")


# --- collection frame -------------------------------------------------------
crit = pd.read_parquet(PROCESSED / "validation_collections.parquet")
feats = pd.read_parquet(EXT / "features.parquet")
idx = pd.read_parquet(EXT / "candidate_indices.parquet")
dims = pd.read_parquet(EXT / "dimension_scores.parquet")
frame = crit.merge(feats, on="doc_id").merge(idx, on="doc_id").merge(dims, on="doc_id", suffixes=("", "_dim"))
frame = frame[frame.log_hours_per_response.notna() & np.isfinite(frame.log_hours_per_response)].copy()
frame["log_words"] = np.log1p(frame.n_words)

# Purely rule-based versions of requirements and documentary effort (no model counts)
corpus_feats = pd.read_parquet(INTERIM / "features.parquet")
llm_cols = [c for c in feats.columns if c.startswith("llm_")]
pure = build_dimension_scores(feats.drop(columns=llm_cols), scaling=SCALING,
                              reference=corpus_feats.drop(columns=[c for c in corpus_feats.columns if c.startswith("llm_")]))
pure = pure.set_index("doc_id")[["requirements", "documentary_effort"]].add_suffix("_rule")
frame = frame.join(pure, on="doc_id")
RULE_PURE = [k for k in RULE if k not in ("requirements", "documentary_effort")] + ["requirements_rule", "documentary_effort_rule"]
y = frame.log_hours_per_response
print(f"collections: {len(frame)}; departments: {frame.department.nunique()}")

SPECS = {
    "readability (8)": READ,
    "length": ["log_words"],
    "index": ["pca"],
    "rule-based 12 (pure)": RULE_PURE,
    "rule-based 12 (incl. model counts)": RULE,
    "model-assessed 3": JUDGED,
    "fifteen dimensions": list(DIMENSION_KEYS),
    "fourteen (no clarity)": [k for k in DIMENSION_KEYS if k != "clarity"],
}


def X_of(cols, f=frame):
    X = f[cols].astype(float)
    return X.fillna(X.mean())


# 2. Rule-based vs model-assessed, standard 10-fold --------------------------
head("2. Rule-based versus model-assessed dimensions (10-fold CV R2, same folds as validate_burden.py)")
cv = KFold(10, shuffle=True, random_state=SEED)
oof = {}
for name, cols in SPECS.items():
    pred = cross_val_predict(LinearRegression(), X_of(cols).values, y.values, cv=cv)
    oof[name] = pred
    folds = [r2_score(y.values[te], pred[te]) for _, te in cv.split(X_of(cols))]
    print(f"  {name:38s} CV-R2 = {np.mean(folds):.3f}")
print("  Spearman among model-assessed dimensions (corpus):")
cd = pd.read_parquet(ROOT / "outputs" / "tables" / "dimension_scores.parquet")
print(cd[JUDGED + ["length", "legal_density", "requirements"]].corr(method="spearman").round(2).to_string())

# 3. Inference ---------------------------------------------------------------
head("3. Inference: corrected resampled t (10x10 CV) and bootstrap of pooled out-of-sample R2 differences")
rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=SEED)
splits = list(rkf.split(frame))
def fold_r2(cols):
    X = X_of(cols).values
    out = []
    for tr, te in splits:
        m = LinearRegression().fit(X[tr], y.values[tr])
        out.append(r2_score(y.values[te], m.predict(X[te])))
    return np.array(out)
from scipy import stats
rng = np.random.default_rng(SEED)
pairs = [("index", "readability (8)"), ("fifteen dimensions", "index"), ("fifteen dimensions", "readability (8)"),
         ("fifteen dimensions", "rule-based 12 (pure)"), ("rule-based 12 (pure)", "readability (8)"),
         ("fifteen dimensions", "fourteen (no clarity)")]
cache = {}
n = len(frame); ratio = (n / 10) / (n - n / 10)
for a, b in pairs:
    for s in (a, b):
        if s not in cache:
            cache[s] = fold_r2(SPECS[s])
    d = cache[a] - cache[b]
    t = d.mean() / np.sqrt((1 / len(d) + ratio) * d.var(ddof=1))
    p = 2 * (1 - stats.t.cdf(abs(t), df=len(d) - 1))
    # bootstrap over observations of pooled out-of-sample predictions (single 10-fold)
    ya, pa, pb = y.values, oof[a], oof[b]
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        diffs.append(r2_score(ya[i], pa[i]) - r2_score(ya[i], pb[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  {a:22s} - {b:22s} mean diff {d.mean():+.3f}  corrected t = {t:5.2f} (p = {p:.2g})  "
          f"bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")

# 1. Within agency -----------------------------------------------------------
head("1. Within-agency validity (agency fixed effects)")
fe = frame.dropna(subset=["department"]).copy()
fe["dept"] = fe.department.astype(str)
print(f"  collections with agency: {len(fe)}, agencies: {fe.dept.nunique()}")
base = smf.ols("log_hours_per_response ~ C(dept)", data=fe).fit()
print(f"  in sample: agency effects alone R2 = {base.rsquared:.3f}")
def within_r2(cols):
    Xd = fe[cols].astype(float).fillna(fe[cols].astype(float).mean())
    Xd = Xd - Xd.groupby(fe.dept).transform("mean")
    yd = fe.log_hours_per_response - fe.groupby("dept").log_hours_per_response.transform("mean")
    m = LinearRegression().fit(Xd.values, yd.values)
    return r2_score(yd.values, m.predict(Xd.values))
for name in ("readability (8)", "length", "index", "rule-based 12 (pure)", "model-assessed 3", "fifteen dimensions"):
    print(f"  in sample: within-agency R2 of {name:24s} = {within_r2(SPECS[name]):.3f}")
f_idx = smf.ols("log_hours_per_response ~ C(dept) + pca", data=fe).fit(cov_type="cluster", cov_kwds={"groups": fe.dept})
print(f"  index coefficient with agency FE: {f_idx.params['pca']:.4f} (cluster-robust p = {f_idx.pvalues['pca']:.2g})")
f_ri = smf.ols("log_hours_per_response ~ C(dept) + pca + " + " + ".join(READ), data=fe).fit(cov_type="cluster", cov_kwds={"groups": fe.dept})
print(f"  index coefficient with agency FE and readability: {f_ri.params['pca']:.4f} (p = {f_ri.pvalues['pca']:.2g})")
# out of sample within agencies
big = fe.groupby("dept").dept.transform("size") >= 10
fs = fe[big].copy()
print(f"  out of sample: {len(fs)} collections in {fs.dept.nunique()} agencies with >= 10 collections")
skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
D = pd.get_dummies(fs.dept, drop_first=False).astype(float).values
yv = fs.log_hours_per_response.values
def oos(cols):
    X = np.hstack([D] + ([fs[cols].astype(float).fillna(fs[cols].astype(float).mean()).values] if cols else []))
    r = []
    for tr, te in skf.split(X, fs.dept):
        m = LinearRegression().fit(X[tr], yv[tr])
        r.append(r2_score(yv[te], m.predict(X[te])))
    return np.array(r)
r_base = oos([])
print(f"  agency effects alone: CV-R2 {r_base.mean():.3f}")
for name in ("readability (8)", "length", "index", "rule-based 12 (pure)", "model-assessed 3", "fifteen dimensions"):
    r = oos(SPECS[name]); d = r - r_base
    print(f"  agency effects + {name:24s}: CV-R2 {r.mean():.3f}  gain {d.mean():+.3f}  folds with gain {int((d > 0).sum())}/10")

# 4. UK trends ---------------------------------------------------------------
head("4. United Kingdom trends")
corpus = pd.read_parquet(PROCESSED / "corpus.parquet", columns=["doc_id", "country", "date", "doc_type"])
ci = pd.read_parquet(ROOT / "outputs" / "tables" / "candidate_indices.parquet")[["doc_id", "pca"]]
t = corpus.merge(ci, on="doc_id").merge(corpus_feats[["doc_id", "n_words"]], on="doc_id")
t["year"] = pd.to_datetime(t.date.astype(str), errors="coerce", utc=True, format="mixed").dt.year
gb = t[(t.country == "GB") & t.year.between(2015, 2026)].copy()
gb["log_words"] = np.log1p(gb.n_words)
def slope(d, extra=""):
    m = smf.ols("pca ~ year + log_words" + extra, data=d).fit()
    return m.params["year"], m.pvalues["year"], len(d)
for label, d, extra in [("2015-2026", gb, ""), ("2015-2019", gb[gb.year <= 2019], ""),
                        ("2019-2026", gb[gb.year >= 2019], ""), ("2020-2026", gb[gb.year >= 2020], ""),
                        ("excluding 2017-2019", gb[~gb.year.between(2017, 2019)], ""),
                        ("2015-2026, document type fixed", gb, " + C(doc_type)")]:
    b, p, n_ = slope(d, extra)
    print(f"  {label:32s} n = {n_:6d}  slope = {b:+.2f} points/year (p = {p:.2g})")
print("  share of document types by period:")
gb["period"] = np.where(gb.year <= 2018, "2015-2018", "2019-2026")
print((gb.groupby("period").doc_type.value_counts(normalize=True).unstack(0).round(3).head(8)).to_string())
