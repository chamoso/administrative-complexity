"""Sensitivity checks.

    python scripts/sensitivity_checks.py

Reads everything; writes only two new tables (review_family_*.csv) used for a
figure.

1. Language: French pages of canada.ca labelled as English.
2. Sensitivity: validation and collection comparisons without those pages.
3. Context: agency, obligation, affected public and reporting frequency held fixed.
4. Rule-based and model counts of requirements and documents against human coders.
5. Excerpting: share of documents longer than the 12,000-character limit.
6. Trends excluding the partial year 2026.
7. Profiles of the three families of demand by collection and over time (UK).
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from adminplex.config import CONFIG, INTERIM, PROCESSED, SEED  # noqa: E402
from adminplex.index.build import BUILDERS  # noqa: E402
from adminplex.index.dimensions import DIMENSION_KEYS, BURDEN_FAMILY as FAMILIES, build_dimension_scores  # noqa: E402
from run_experiments import adjusted_comparison  # noqa: E402

READ = CONFIG["analysis"]["readability_indices"]
SCALING = CONFIG["index"]["scaling"]
DIMS = list(DIMENSION_KEYS)
EXT = ROOT / "outputs" / "external" / "collections"
TABLES = ROOT / "outputs" / "tables"


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


def cv_r2(X, y, seed=SEED):
    cv = KFold(10, shuffle=True, random_state=seed)
    X = X.astype(float).fillna(X.astype(float).mean()).values
    pred = cross_val_predict(LinearRegression(), X, y, cv=cv)
    return float(np.mean([r2_score(y[te], pred[te]) for _, te in cv.split(X)]))


# ---------------------------------------------------------------------------
corpus = pd.read_parquet(PROCESSED / "corpus.parquet",
                         columns=["doc_id", "country", "source", "language", "doc_type", "date", "text"])
FR = re.compile(r"\b(les|des|est|une|pour|dans|vous|avec|sont|qui)\b")
EN = re.compile(r"\b(the|and|for|with|you|are|that|this|which)\b")
ES = re.compile(r"\b(los|las|del|para|por|con|una|que|sus)\b")
def detect(t):
    t = (t or "")[:4000].lower()
    return max((len(FR.findall(t)), "fr"), (len(EN.findall(t)), "en"), (len(ES.findall(t)), "es"))[1]
corpus["detected"] = corpus.text.map(detect)
corpus["n_chars"] = corpus.text.fillna("").str.len()
french_ca = corpus[(corpus.source == "canada.ca") & (corpus.detected == "fr")].doc_id
feats = pd.read_parquet(INTERIM / "features.parquet")
main = load_tag("main")
dims = pd.read_parquet(TABLES / "dimension_scores.parquet")
idx = pd.read_parquet(TABLES / "candidate_indices.parquet")[["doc_id", "pca"]]

head("1. Language")
print(pd.crosstab([corpus.source, corpus.language], corpus.detected).to_string())
ca = corpus[corpus.source == "canada.ca"].merge(idx, on="doc_id").merge(dims, on="doc_id")
print("canada.ca, mean by detected language:")
print(ca.groupby("detected")[["pca", "requirements", "conditional_decisions", "exceptions", "ambiguity",
                              "cognitive_load", "legal_knowledge"]].mean().round(1).to_string())

# ---------------------------------------------------------------------------
head("2. Sensitivity: scale and weights rebuilt without the French canada.ca pages")
keep = ~feats.doc_id.isin(french_ca)
ref = feats[keep].merge(main, on="doc_id", how="left")
cdims = build_dimension_scores(ref, scaling=SCALING)
usable = [k for k in DIMS if cdims[k].notna().any()]
w = pd.Series(BUILDERS["pca"](cdims.set_index("doc_id")[usable]).weights).reindex(usable).fillna(0.0)
ext_f = pd.read_parquet(EXT / "features.parquet")
edims = build_dimension_scores(ext_f, scaling=SCALING, reference=ref).set_index("doc_id")[usable]
eidx = (edims.fillna(50.0) * w).sum(axis=1) / w.abs().sum()
crit = pd.read_parquet(PROCESSED / "validation_collections.parquet")
fr2 = crit.merge(ext_f, on="doc_id")
fr2 = fr2[fr2.log_hours_per_response.notna() & np.isfinite(fr2.log_hours_per_response)].copy()
fr2 = fr2.join(edims.add_suffix("_new"), on="doc_id").join(eidx.rename("index_new"), on="doc_id")
fr2 = fr2.merge(pd.read_parquet(EXT / "candidate_indices.parquet"), on="doc_id")
fr2 = fr2.merge(pd.read_parquet(EXT / "dimension_scores.parquet"), on="doc_id", suffixes=("", "_old"))
y = fr2.log_hours_per_response.values
print(f"  collections {len(fr2)}; rank correlation of collection index, old vs new: "
      f"{fr2.pca.corr(fr2.index_new, method='spearman'):.4f}")
for label, cols in [("fifteen dimensions (published)", DIMS), ("fifteen dimensions (rebuilt)", [d + "_new" for d in usable]),
                    ("index (published)", ["pca"]), ("index (rebuilt)", ["index_new"]), ("readability", READ)]:
    print(f"  {label:34s} CV-R2 = {cv_r2(fr2[cols], y):.4f}")
work = (corpus[["doc_id", "country", "source"]].merge(idx.rename(columns={"pca": "aci"}), on="doc_id")
        .merge(feats[["doc_id", "n_words"]], on="doc_id"))
work["log_words"] = np.log1p(work.n_words)
print("  adjusted means by collection, all documents vs without French canada.ca pages:")
a_all = adjusted_comparison(work, "country", 200).set_index("country")
a_en = adjusted_comparison(work[~work.doc_id.isin(french_ca)], "country", 200).set_index("country")
print(pd.DataFrame({"raw_all": a_all.raw_mean, "adj_all": a_all.adjusted_mean,
                    "raw_en": a_en.raw_mean, "adj_en": a_en.adjusted_mean, "n_en": a_en.n}).round(1).to_string())

# ---------------------------------------------------------------------------
head("3. Context: agency, obligation, affected public and reporting frequency held fixed")
coll = (crit.merge(ext_f, on="doc_id").merge(pd.read_parquet(EXT / "candidate_indices.parquet"), on="doc_id")
            .merge(pd.read_parquet(EXT / "dimension_scores.parquet"), on="doc_id", suffixes=("", "_dim"))
            .merge(load_tag("global"), on="doc_id", how="left"))
coll = coll[coll.log_hours_per_response.notna() & np.isfinite(coll.log_hours_per_response)].copy()
ctx = ["department", "obligation", "affected_public", "reporting_frequency", "icr_type"]
for c in ctx:
    top = coll[c].astype(str).value_counts()
    keep_levels = top[top >= 20].index
    coll[c + "_c"] = coll[c].astype(str).where(coll[c].astype(str).isin(keep_levels), "other")
    print(f"  {c}: {coll[c + '_c'].nunique()} levels")
coll = coll[coll.groupby("department_c").department_c.transform("size") >= 10]
CTX = pd.get_dummies(coll[[c + "_c" for c in ctx]].astype(str)).astype(float).values
yv = coll.log_hours_per_response.values
skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
def oos(cols):
    M = np.hstack([CTX] + ([coll[cols].astype(float).fillna(coll[cols].astype(float).mean()).values] if cols else []))
    return np.array([r2_score(yv[te], LinearRegression().fit(M[tr], yv[tr]).predict(M[te]))
                     for tr, te in skf.split(M, coll.department_c)])
base = oos([])
print(f"  context alone (CV): {base.mean():.3f}  (n = {len(coll)})")
for name, cols in [("readability", READ), ("length", ["n_words"]), ("index", ["pca"]),
                   ("fifteen dimensions", DIMS), ("global rating", ["llm_global_complexity"])]:
    r = oos(cols)
    print(f"    + {name:20s} {r.mean():.3f}  gain {np.mean(r - base):+.3f}  folds {int(((r - base) > 0).sum())}/10")

# ---------------------------------------------------------------------------
head("4. Counts against human coders (first pass, 100 documents)")
files = sorted(glob.glob(str(ROOT / "coding" / "ratings_*.csv"))) or sorted(glob.glob(str(ROOT / "ratings" / "ratings_*.csv")))
hr = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
hr = hr[hr.item_id.str.extract(r"#(\d)$")[0].astype(int) == 1]
hm = hr.groupby("doc_id")[["n_requirements", "n_documents"]].mean().add_prefix("coders_")
j = (hm.join(feats.set_index("doc_id")[["n_requirement_sentences", "requirement_rate", "n_document_types",
                                         "document_mention_rate", "n_steps"]])
       .join(main.set_index("doc_id")[["llm_n_requirements", "llm_n_documents"]])
       .join(dims.set_index("doc_id")[["requirements", "documentary_effort", "steps"]]))
for target, cands in [("coders_n_requirements", ["n_requirement_sentences", "requirement_rate", "llm_n_requirements", "requirements", "n_steps", "steps"]),
                      ("coders_n_documents", ["n_document_types", "document_mention_rate", "llm_n_documents", "documentary_effort"])]:
    print(f"  {target}: " + ", ".join(f"{c} {j[target].corr(j[c], method='spearman'):.2f}" for c in cands))

# ---------------------------------------------------------------------------
head("5. Excerpting")
print(f"  corpus documents over 12,000 characters: {(corpus.n_chars > 12000).mean():.3f}")
sc = pd.read_parquet(PROCESSED / "validation_collections_scoring.parquet", columns=["doc_id", "text"])
sc["long"] = sc.text.fillna("").str.len() > 12000
print(f"  collection descriptions over 12,000 characters: {sc.long.mean():.4f}")
short = coll.doc_id.isin(sc[~sc.long].doc_id)
print(f"  fifteen dimensions, collections not excerpted: CV-R2 = {cv_r2(coll.loc[short, DIMS], coll.loc[short, 'log_hours_per_response'].values):.3f} (n = {int(short.sum())})")

# ---------------------------------------------------------------------------
head("6. Trends excluding the partial year 2026")
t = corpus[["doc_id", "country", "doc_type", "date"]].merge(idx, on="doc_id").merge(feats[["doc_id", "n_words"]], on="doc_id")
t["year"] = pd.to_datetime(t.date.astype(str), errors="coerce", utc=True, format="mixed").dt.year
t["log_words"] = np.log1p(t.n_words)
for country in ("GB", "EU", "ES"):
    d = t[(t.country == country) & t.year.between(2015, 2026)]
    for label, dd in (("all years", d), ("excluding 2026", d[d.year < 2026])):
        m = smf.ols("pca ~ year + log_words", data=dd).fit()
        print(f"  {country} {label:16s} n = {len(dd):6d}  slope {m.params['year']:+.2f} (p = {m.pvalues['year']:.2g})")

# ---------------------------------------------------------------------------
head("7. Profiles of the three families of demand")
fam = dims[["doc_id"]].copy()
for f, members in FAMILIES.items():
    fam[f] = dims[list(members)].mean(axis=1)
prof = corpus[["doc_id", "country", "source"]].merge(fam, on="doc_id")
prof = prof[~prof.doc_id.isin(french_ca)]
tab = prof.groupby("country")[list(FAMILIES)].mean().round(1)
print(tab.to_string())
tab.reset_index().to_csv(TABLES / "review_family_profiles.csv", index=False)
gb = t[(t.country == "GB") & t.year.between(2015, 2026)].merge(fam, on="doc_id")
rows = []
for f in FAMILIES:
    m1 = smf.ols(f"{f} ~ year + log_words", data=gb).fit()
    m2 = smf.ols(f"{f} ~ year + log_words + C(doc_type)", data=gb).fit()
    rows.append({"family": f, "slope": m1.params["year"], "slope_doc_type_fixed": m2.params["year"]})
    print(f"  GB {f:12s} slope {m1.params['year']:+.2f}; document type fixed {m2.params['year']:+.2f}")
pd.DataFrame(rows).to_csv(TABLES / "review_family_trends_gb.csv", index=False)
gb.groupby("year")[list(FAMILIES)].mean().round(2).reset_index().to_csv(TABLES / "review_family_by_year_gb.csv", index=False)
