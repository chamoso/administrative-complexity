"""Stability of the burden criterion between successive approvals (read-only).

The OMB inventory records, for each information collection, the previously approved
totals of responses and hours. This script compares hours per response in the current
and in the previous approval.

    python scripts/criterion_stability.py
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from adminplex.config import PROCESSED, RAW  # noqa: E402

rows = []
for _, el in ET.iterparse(RAW / "pra_inventory.xml", events=("end",)):
    if el.tag.split("}")[-1] != "InformationCollectionRequest":
        continue
    def num(path):
        t = el.findtext(path)
        try:
            return float(t)
        except (TypeError, ValueError):
            return np.nan
    rows.append({
        "icr_ref": el.findtext("ICRReferenceNumber"),
        "resp": num("./Burden/BurdenResponse/TotalQuantity"),
        "resp_prev": num("./Burden/BurdenResponse/PreviousTotalQuantity"),
        "hours": num("./Burden/BurdenHour/TotalQuantity"),
        "hours_prev": num("./Burden/BurdenHour/PreviousTotalQuantity"),
    })
    el.clear()
inv = pd.DataFrame(rows).drop_duplicates("icr_ref")
val = pd.read_parquet(PROCESSED / "validation_collections.parquet")
val = val[val.log_hours_per_response.notna() & np.isfinite(val.log_hours_per_response)]
d = val.merge(inv, on="icr_ref", how="left")
d = d[(d.resp > 0) & (d.hours > 0) & (d.resp_prev > 0) & (d.hours_prev > 0)].copy()
d["lh"] = np.log(d.hours / d.resp)
d["lh_prev"] = np.log(d.hours_prev / d.resp_prev)
d["abs_change"] = (d.lh - d.lh_prev).abs()
print(f"collections in the validation frame: {len(val)}; with a previous approval: {len(d)}")


def report(label, g):
    if len(g) < 30:
        return
    r = g.lh.corr(g.lh_prev)
    rho = g.lh.corr(g.lh_prev, method="spearman")
    same = (g.abs_change < 1e-9).mean()
    big = (g.abs_change > np.log(1.25)).mean()
    dbl = (g.abs_change > np.log(2)).mean()
    print(f"  {label:68s} n={len(g):5d} r={r:.3f} r2={r*r:.3f} rho={rho:.3f} unchanged={same:.2f} "
          f">25%={big:.2f} >2x={dbl:.2f}")


report("all", d)
for t, g in d.groupby("icr_type"):
    report(t[:68], g)
changed = d[d.abs_change > 1e-9]
report("only collections whose hours per response changed", changed)
print("  median absolute change among changed (ratio):", round(float(np.exp(changed.abs_change.median())), 2))
print("  variance of log hours per response:", round(float(d.lh.var()), 2),
      "| variance of the change:", round(float((d.lh - d.lh_prev).var()), 2))

# --- What the inventory's own metadata explains, and what the text adds ---------
import glob, json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from adminplex.config import CONFIG, SEED
from adminplex.index.dimensions import DIMENSION_KEYS

EXT = ROOT / "outputs" / "external" / "collections"
f = (val.merge(pd.read_parquet(EXT / "features.parquet"), on="doc_id")
        .merge(pd.read_parquet(EXT / "dimension_scores.parquet"), on="doc_id", suffixes=("", "_dim"))
        .merge(inv, on="icr_ref", how="left"))
print(f"\ncollections analysed: {len(f)}; words in the description: median {f.n_words.median():.0f}, "
      f"p10 {f.n_words.quantile(.1):.0f}, p90 {f.n_words.quantile(.9):.0f}")
ctx = ["department", "obligation", "affected_public", "reporting_frequency", "icr_type"]
for c in ctx:
    top = f[c].astype(str).value_counts()
    f[c + "_c"] = f[c].astype(str).where(f[c].astype(str).isin(top[top >= 20].index), "other")
f = f[f.groupby("department_c").department_c.transform("size") >= 10].copy()
CTX = pd.get_dummies(f[[c + "_c" for c in ctx]].astype(str)).astype(float).values
meta = np.column_stack([np.log1p(f[c].fillna(0).astype(float)) for c in ("n_instruments", "n_cfr_parts", "n_statutes")])
y = f.log_hours_per_response.values
skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
READ = CONFIG["analysis"]["readability_indices"]
def oos(blocks):
    M = np.hstack(blocks)
    return np.array([r2_score(y[te], LinearRegression().fit(M[tr], y[tr]).predict(M[te]))
                     for tr, te in skf.split(M, f.department_c)])
def X(cols):
    m = f[cols].astype(float); return m.fillna(m.mean()).values
base = oos([CTX, meta])
print(f"  context + counts of instruments, CFR parts and statutes: CV-R2 {base.mean():.3f}")
for name, cols in [("readability", READ), ("fifteen dimensions", list(DIMENSION_KEYS))]:
    r = oos([CTX, meta, X(cols)])
    print(f"    + {name:20s} {r.mean():.3f} gain {np.mean(r - base):+.3f} folds {int(((r - base) > 0).sum())}/10")
has_prev = (f.resp_prev > 0) & (f.hours_prev > 0) & (f.resp > 0) & (f.hours > 0)
g = f[has_prev].copy(); g["lh_prev"] = np.log(g.hours_prev / g.resp_prev)
print(f"  previous approval alone (n = {len(g)}): R2 = {np.corrcoef(g.lh_prev, g.log_hours_per_response)[0,1]**2:.3f}")
