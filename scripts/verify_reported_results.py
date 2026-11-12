"""Check every reported number against its source table.

Run from the repository root on the cluster, where the corpus and all tables live:

    python scripts/verify_reported_results.py

Each table is read as parquet if present, otherwise as CSV. Checks whose table is
missing are reported as SKIP rather than failing, and the numbers that no table
records (ad hoc analyses) are listed at the end for a manual check.
"""
import json
import sys
from pathlib import Path

import pandas as pd

T = Path("outputs/tables")
ok = bad = skipped = 0


def load(name: str) -> pd.DataFrame | None:
    parquet, csv = T / f"{name}.parquet", T / f"{name}.csv"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except ImportError:  # no parquet engine installed: fall back to the CSV copy
            pass
    return pd.read_csv(csv) if csv.exists() else None


def check(label, claimed, actual, tol=0.005):
    global ok, bad, skipped
    if actual is None or (isinstance(actual, float) and pd.isna(actual)):
        print(f"  SKIP {label:58s} reported={claimed}  (no table)")
        skipped += 1
        return
    good = abs(float(claimed) - float(actual)) <= tol
    print(f"  {'OK  ' if good else 'FAIL'} {label:58s} reported={claimed}  table={float(actual):.4f}")
    ok, bad = ok + good, bad + (not good)


def section(title):
    print(f"\n{title}")


# --------------------------------------------------------------------------
section("CORPUS")
corpus_path = Path("data/processed/corpus.parquet")
if corpus_path.exists():
    corpus = pd.read_parquet(corpus_path, columns=["doc_id", "country", "source"])
    check("total documents", 51189, len(corpus), tol=0)
    for c, n in [("GB", 20393), ("ES", 11729), ("CA", 8847), ("EU", 3385), ("AU", 955),
                 ("NZ", 397), ("US", 5483)]:
        check(f"documents {c}", n, int((corpus.country == c).sum()), tol=0)
    by_source = corpus.source.value_counts()
    print("  (sources, compare with Table 3:)", by_source.to_dict())
else:
    print("  SKIP corpus not available here")

# --------------------------------------------------------------------------
section("INTERNAL STRUCTURE ")
diag_path = T / "index_diagnostics.json"
if diag_path.exists():
    diag = json.load(open(diag_path))["diagnostics"]
    pca, efa = diag.get("pca", {}), diag.get("efa", {})
    check("alpha", 0.866, pca.get("alpha", diag.get("equal_weight", {}).get("alpha")))
    check("omega", 0.893, pca.get("omega", diag.get("equal_weight", {}).get("omega")))
    check("first component share", 0.368, (pca.get("explained_variance_ratio") or [None])[0])
    check("KMO", 0.818, efa.get("kmo"))
    check("EFA cumulative common variance", 0.552, efa.get("cumulative_variance"), tol=0.01)
agree = load("index_agreement")
if agree is not None:
    agree = agree.set_index(agree.columns[0])
    off = [agree.iloc[i, j] for i in range(len(agree)) for j in range(len(agree)) if i != j]
    check("min agreement between aggregations", 0.967, min(off), tol=0.002)
    check("max agreement between aggregations", 0.996, max(off), tol=0.002)
contrib = load("experiment7_dimension_contributions")
if contrib is not None:
    contrib = contrib.set_index("dimension")
    for dim, claimed in [("cognitive_load", 0.123), ("legal_knowledge", 0.111),
                         ("legal_density", 0.084), ("length", 0.082)]:
        check(f"share of explained variance {dim}", claimed,
              contrib.loc[dim, "share_of_explained"], tol=0.001)
scores = load("dimension_scores")
if scores is not None and {"steps", "dependencies"} <= set(scores.columns):
    check("steps-dependencies correlation", 0.94,
          scores.steps.rank().corr(scores.dependencies.rank()), tol=0.01)  # Spearman without scipy
else:
    print("  MANUAL steps-dependencies correlation 0.94 (dimension score matrix)")

# --------------------------------------------------------------------------
section("H1/H2: COLLECTIONS ")
comp = load("burden_model_comparison_collections")
if comp is not None:
    comp = comp.set_index("specification")
    for spec, claimed in [("dimensions", 0.122), ("aci_and_readability", 0.070),
                          ("aci_only", 0.062), ("readability_and_length", 0.010),
                          ("length_only", 0.006), ("readability", 0.004)]:
        check(f"CV-R2 {spec}", claimed, comp.loc[spec, "cv_r2"], tol=0.001)
    ratio = comp.loc["aci_and_readability", "cv_r2"] / comp.loc["readability", "cv_r2"]
    check("index added to readability: 'about eighteen-fold'", 18, ratio, tol=0.5)
tests = load("burden_model_tests_collections")
if tests is not None:
    tests = tests.set_index("comparison")
    check("diff index vs readability", 0.058, tests.loc["aci_only vs readability", "mean_difference"], tol=0.001)
    check("t index vs readability", 9.5, tests.loc["aci_only vs readability", "t"], tol=0.1)
    check("folds favouring index", 10, tests.loc["aci_only vs readability", "folds_favouring_first"], tol=0)
    check("diff dimensions vs index", 0.060, tests.loc["dimensions vs aci_only", "mean_difference"], tol=0.001)
    check("t dimensions vs index", 10.1, tests.loc["dimensions vs aci_only", "t"], tol=0.1)
    check("folds favouring dimensions", 10, tests.loc["dimensions vs aci_only", "folds_favouring_first"], tol=0)
rv = load("robustness_variants")
if rv is not None:
    def cell(variant, spec, col="cv_r2"):
        r = rv[(rv.variant == variant) & (rv.specification == spec)]
        return float(r[col].iloc[0]) if len(r) else None
    for spec, lo, hi in [("fifteen dimensions", 0.109, 0.138), ("index alone", 0.051, 0.074),
                         ("readability", 0.001, 0.010)]:
        check(f"bootstrap CI low  {spec}", lo, cell("all collections", spec, "ci_low"), tol=0.001)
        check(f"bootstrap CI high {spec}", hi, cell("all collections", spec, "ci_high"), tol=0.001)
    for variant, vals in [("excluding the longest 10 per cent", (0.129, 0.074, 0.004)),
                          ("excluding the longest 25 per cent", (0.120, 0.064, 0.004)),
                          ("folds grouped by department", (-0.008, -0.057, -0.114))]:
        for spec, claimed in zip(("fifteen dimensions", "index alone", "readability"), vals):
            check(f"{variant[:28]} | {spec}", claimed, cell(variant, spec), tol=0.001)
    for agg, claimed in [("irt_graded", 0.082), ("pca", 0.062), ("efa", 0.053),
                         ("equal_weight", 0.052), ("expert_weight", 0.045)]:
        check(f"aggregation {agg}", claimed, cell(f"aggregation: {agg}", "index alone"), tol=0.001)

section("H1/H2: REGULATORY TEXTS ")
cp = load("burden_model_comparison_parts")
if cp is not None:
    cp = cp.set_index("specification")
    for spec, claimed in [("dimensions", 0.040), ("aci_and_readability", 0.029),
                          ("aci_only", -0.008), ("readability_and_length", 0.028),
                          ("length_only", -0.010), ("readability", 0.029)]:
        check(f"parts CV-R2 {spec}", claimed, cp.loc[spec, "cv_r2"], tol=0.001)
    check("parts fold SD min (index)", 0.014, cp.loc["aci_only", "cv_r2_sd"], tol=0.001)
    check("parts fold SD max (dimensions)", 0.043, cp.loc["dimensions", "cv_r2_sd"], tol=0.001)
cc = load("burden_correlations_parts")
if cc is not None:
    check("parts rank correlation of index", 0.030,
          cc.set_index("measure").loc["pca", "spearman"], tol=0.001)

section("ROBUSTNESS ")
ext = load("burden_extended")
if ext is not None:
    def ecell(level, spec, model="linear", col="cv_r2"):
        r = ext[(ext.level == level) & (ext.specification == spec) & (ext.model == model)]
        return float(r[col].iloc[0]) if len(r) else None
    check("gradient boosting, dimensions", 0.125, ecell("collection", "fifteen dimensions", "gradient boosting"), tol=0.001)
    check("gradient boosting, index", 0.041, ecell("collection", "index alone", "gradient boosting"), tol=0.001)
    check("gradient boosting, readability", -0.036, ecell("collection", "eight readability indices", "gradient boosting"), tol=0.001)
    check("without clarity", 0.118, ecell("collection", "fourteen dimensions (no clarity)"), tol=0.001)
    check("bodies (n)", 36, ecell("body", "index alone", col="n"), tol=0)
    check("body-level rank correlation", 0.47, ecell("body", "index alone", col="spearman"), tol=0.005)
ext_t = load("burden_extended_tests")
if ext_t is not None:
    ext_t = ext_t.set_index("comparison")
    check("p boosting vs linear (dimensions)", 0.69,
          ext_t.loc["fifteen dimensions | gradient boosting vs fifteen dimensions | linear", "p_value"], tol=0.01)
    check("p without clarity", 0.04,
          ext_t.loc["fourteen dimensions (no clarity) | linear vs fifteen dimensions | linear", "p_value"], tol=0.005)
sc = load("review_scale_confound")
if sc is not None:
    sc = sc.set_index("specification")
    check("index coefficient", 0.062, sc.loc["hours per response ~ index", "index_coefficient"], tol=0.001)
    check("index coefficient + programme size", 0.052, sc.loc["+ programme size", "index_coefficient"], tol=0.001)
ri = load("robustness_importance")
if ri is not None:
    ri = ri.set_index("dimension").importance
    for dim, claimed in [("cognitive_load", 0.059), ("legal_knowledge", 0.031), ("clarity", 0.017),
                         ("conditional_decisions", 0.011), ("dependencies", 0.010),
                         ("documentary_effort", 0.007)]:
        check(f"permutation importance {dim}", claimed, ri.get(dim), tol=0.001)

# --------------------------------------------------------------------------
section("STABILITY")
cons = load("experiment10_consistency")
if cons is not None:
    cons = cons.set_index("measure")
    check("ICC cognitive load", 0.946, cons.loc["llm_cognitive_load", "reliability"], tol=0.001)
    check("ICC legal knowledge", 0.933, cons.loc["llm_legal_knowledge", "reliability"], tol=0.001)
    check("between-document SD clarity", 0.95, cons.loc["llm_clarity", "between_document_sd"], tol=0.01)
    check("between-document SD cognitive load", 1.91, cons.loc["llm_cognitive_load", "between_document_sd"], tol=0.01)
    check("between-document SD legal knowledge", 1.91, cons.loc["llm_legal_knowledge", "between_document_sd"], tol=0.01)
rob = load("experiment9_robustness")
if rob is not None:
    kap = rob.dropna(subset=["quadratic_weighted_kappa"])
    reps = kap[kap.comparison.str.startswith("repeat")].groupby("measure").quadratic_weighted_kappa
    lo_, hi_ = reps.min(), reps.max()
    check("kappa between repeats, cognitive load (min)", 0.93, lo_.get("llm_cognitive_load"), tol=0.005)
    check("kappa between repeats, cognitive load (max)", 0.96, hi_.get("llm_cognitive_load"), tol=0.005)
    check("kappa between repeats, legal knowledge (min)", 0.92, lo_.get("llm_legal_knowledge"), tol=0.005)
    check("kappa between repeats, legal knowledge (max)", 0.96, hi_.get("llm_legal_knowledge"), tol=0.005)
    var = kap[kap.comparison == "variant"].set_index("measure").quadratic_weighted_kappa
    check("kappa paraphrased rubric, cognitive load", 0.956, var.get("llm_cognitive_load"), tol=0.002)
    check("kappa paraphrased rubric, legal knowledge", 0.955, var.get("llm_legal_knowledge"), tol=0.002)
    models = kap[~kap.comparison.str.startswith("repeat") & (kap.comparison != "variant")
                 & kap.measure.isin(["llm_cognitive_load", "llm_legal_knowledge"])]
    print("  model comparisons found:", sorted(models.comparison.unique()))
    if len(models):
        check("min kappa across models", 0.66, models.quadratic_weighted_kappa.min(), tol=0.006)
        check("max kappa across models", 0.78, models.quadratic_weighted_kappa.max(), tol=0.006)

section("HUMAN VALIDATION ")
h = load("human_validation_summary")
if h is not None:
    def hv(kind, measure, col):
        r = h[(h.kind == kind) & (h.measure == measure)]
        return float(r[col].iloc[0]) if len(r) else None
    for m, kc, km, rc, rm in [("cognitive_load", 0.41, 0.44, 0.59, 0.55),
                              ("legal_knowledge", 0.50, 0.28, 0.50, 0.58),
                              ("clarity", 0.46, 0.12, 0.51, 0.24)]:
        check(f"kappa coders {m}", kc, hv("between coders", m, "mean_kappa"), tol=0.006)
        check(f"kappa model {m}", km, hv("coder to model", m, "mean_kappa"), tol=0.006)
        check(f"rho coders {m}", rc, hv("between coders", m, "mean_spearman"), tol=0.006)
        check(f"rho model {m}", rm, hv("coder to model", m, "mean_spearman"), tol=0.006)
    for m, rc, rm in [("n_requirements", 0.53, 0.48), ("n_documents", 0.64, 0.40)]:
        check(f"rho coders {m}", rc, hv("between coders", m, "mean_spearman"), tol=0.006)
        check(f"rho model {m}", rm, hv("coder to model", m, "mean_spearman"), tol=0.006)
    check("held-out coder vs rest", 0.75, hv("coder vs rest (ceiling)", "overall_judgement", "mean_spearman"), tol=0.006)
    check("index vs rest", 0.86, hv("index vs rest (same target)", "overall_judgement", "mean_spearman"), tol=0.006)
ln = load("generalisation_by_language")
if ln is not None:
    ln = ln.set_index("language")
    check("panel vs index, English", 0.89, ln.loc["en", "panel_vs_index"], tol=0.005)
    check("panel vs index, Spanish", 0.91, ln.loc["es", "panel_vs_index"], tol=0.005)
    check("between coders, English", 0.78, ln.loc["en", "between_coders"], tol=0.005)
    check("between coders, Spanish", 0.72, ln.loc["es", "between_coders"], tol=0.005)
lo = load("generalisation_leave_country_out")
if lo is not None:
    lo = lo.set_index("country")
    check("leave-one-out min rank correlation", 0.995, lo.rank_correlation.min(), tol=0.001)
    check("leave-one-out max rank correlation", 0.999, lo.rank_correlation.max(), tol=0.001)
    check("leave-one-out max absolute shift ('up to 10.2')", 10.2, lo.mean_shift.abs().max(), tol=0.05)
    print("  shifts by jurisdiction:", lo.mean_shift.round(1).to_dict())

# --------------------------------------------------------------------------
section("H3: TRENDS ")
slopes = load("trend_slopes")
if slopes is not None:
    slopes = slopes.set_index("scope")
    check("dated documents", 33431, slopes.loc["all", "n"], tol=0)
    check("pooled raw slope", 1.03, slopes.loc["all", "points_per_year_raw"], tol=0.01)
    for scope, claimed in [("GB", -0.89), ("EU", -0.91), ("ES", 0.25)]:
        check(f"adjusted slope {scope}", claimed, slopes.loc[scope, "points_per_year_adjusted"], tol=0.01)
tr = load("trend_by_country")
if tr is not None:
    gb = tr[tr.country == "GB"].set_index("year").mean_aci
    check("UK mean 2017", 51, gb.get(2017.0), tol=0.5)
    check("UK mean 2026", 40, gb.get(2026.0), tol=0.5)
    first = tr.groupby("country").year.min()
    check("first year with >= 30 documents, GB", 2015, first.get("GB"), tol=0)
    check("first year with >= 30 documents, ES", 2019, first.get("ES"), tol=0)
    check("first year with >= 30 documents, EU", 2022, first.get("EU"), tol=0)
    us = tr[tr.country == "US"]
    check("US (eCFR) dated documents, all 2026", 2130, us[us.year == 2026.0].n.sum(), tol=0)
    if slopes is not None:
        # Yearly cells under 30 documents are omitted from trend_by_country, so the
        # totals come from trend_slopes; the eCFR adds its 2,130 documents dated 2026.
        n_scope = slopes["n"]
        check("dated documents = GB + ES + EU + US", 33431,
              n_scope.get("GB") + n_scope.get("ES") + n_scope.get("EU") + us[us.year == 2026.0].n.sum(), tol=0)
        eu = tr[tr.country == "EU"]
        check("EU acts dated 2022 onwards ('98%')", 0.98, eu[eu.year >= 2022].n.sum() / n_scope.get("EU"), tol=0.005)

section("EXPLORATORY: COUNTRIES AND BODIES ")
ctry = load("experiment2_by_country_adjusted")
if ctry is not None:
    ctry = ctry.set_index("country")
    check("raw mean AU", 28.9, ctry.loc["AU", "raw_mean"], tol=0.06)
    check("raw mean EU", 71.9, ctry.loc["EU", "raw_mean"], tol=0.06)
    for c, claimed in [("EU", 52.3), ("AU", 46.1), ("GB", 45.4), ("US", 43.7), ("ES", 43.0),
                       ("NZ", 42.2), ("CA", 36.4)]:
        check(f"adjusted mean {c}", claimed, ctry.loc[c, "adjusted_mean"], tol=0.06)
    check("adjustment AU", 17.1, ctry.loc["AU", "adjustment"], tol=0.06)
    check("adjustment US", -13.6, ctry.loc["US", "adjustment"], tol=0.06)
org = load("organisation_ranking")
if org is not None:
    ranked = org.sort_values("adjusted_mean", ascending=False).reset_index(drop=True)
    org = org.set_index("organisation")
    check("bodies with >= 30 documents", 176, len(org), tol=0)
    check("DVLA adjustment", 14.6, org.loc["Driver and Vehicle Licensing Agency", "adjustment"], tol=0.1)
    check("DVLA median words", 237, org.loc["Driver and Vehicle Licensing Agency", "median_words"], tol=1)
    check("European Commission adjustment", -9.7, org.loc["European Commission", "adjustment"], tol=0.1)
    check("European Commission rank after adjustment", 2,
          int(ranked.index[ranked.organisation == "European Commission"][0]) + 1, tol=0)

section("SIMPLIFICATION")
s = load("experiment56_summary")
if s is not None:
    s = s.set_index("intervention")
    check("removal share edited", 0.51, s.loc["removal", "share_actually_edited"], tol=0.005)
    check("rewrite share edited", 0.98, s.loc["rewrite", "share_actually_edited"], tol=0.006)
    check("removal change where edited", -3.98, s.loc["removal", "mean_change_where_edited"], tol=0.01)
    check("rewrite change where edited", -5.51, s.loc["rewrite", "mean_change_where_edited"], tol=0.01)
doc = load("experiment56_document_level")
if doc is not None:
    # The editing model saw an 8,000-character excerpt, so only documents of at most
    # 1,000 words were edited in full.
    doc = doc.assign(ratio=doc.words_after / doc.words_before.clip(lower=1))
    doc["edited"] = (doc.words_before - doc.words_after).abs() / doc.words_before.clip(lower=1) >= 0.02
    per_doc = doc.drop_duplicates("doc_id").words_before
    check("share of sampled documents over 1,000 words", 0.40, (per_doc > 1000).mean(), tol=0.005)
    for label, mask, n_docs, vals in [
        ("<=1000 words", doc.words_before <= 1000, 903,
         {"removal": (0.267, -1.90, 1.00), "rewrite": (0.977, -1.74, 0.72)}),
        (">1300 words", doc.words_before > 1300, 490,
         {"removal": (0.982, -5.26, 0.33), "rewrite": (1.000, -12.65, 0.15)}),
    ]:
        sub = doc[mask]
        check(f"{label}: documents", n_docs, sub.doc_id.nunique(), tol=0)
        for iv, (share, change, ratio) in vals.items():
            b = sub[sub.intervention == iv]
            check(f"{label} {iv}: share changed", share, b.edited.mean(), tol=0.0005)
            check(f"{label} {iv}: mean change where changed", change, b[b.edited].change.mean(), tol=0.005)
            check(f"{label} {iv}: median length ratio", ratio, b.ratio.median(), tol=0.005)
sm = load("experiment56_summary_max1000words")
if sm is not None:
    sm = sm.set_index("intervention")
    for iv, lo, hi in [("removal", -2.32, -1.48), ("rewrite", -2.18, -1.27)]:
        check(f"<=1000 words {iv}: CI low (analysis script)", lo, sm.loc[iv, "ci_low_where_edited"], tol=0.005)
        check(f"<=1000 words {iv}: CI high (analysis script)", hi, sm.loc[iv, "ci_high_where_edited"], tol=0.005)
bd = load("experiment56_by_dimension_max1000words")
if bd is not None:
    bd = bd.set_index("dimension")
    for dim, rem, rew in [("syntactic_complexity", -0.9, -18.0), ("cross_references", -0.3, -11.6),
                          ("ambiguity", -0.1, -9.3), ("redundancy", -0.3, -9.0), ("clarity", 0.3, -7.6),
                          ("legal_density", -0.3, -5.5), ("legal_knowledge", -0.1, -4.4),
                          ("cognitive_load", -0.8, -4.2), ("exceptions", -0.1, -3.5),
                          ("requirements", -1.9, -2.3), ("documentary_effort", -1.7, -1.4),
                          ("conditional_decisions", -0.6, 2.6), ("length", 0.5, 11.9),
                          ("dependencies", -0.4, 12.3), ("steps", -0.3, 21.0)]:
        check(f"Table S3 {dim} removal", rem, bd.loc[dim, "removal"], tol=0.05)
        check(f"Table S3 {dim} rewriting", rew, bd.loc[dim, "rewrite"], tol=0.05)

# --------------------------------------------------------------------------
print("""
NUMBERS WITHOUT A TABLE (checked on the cluster, 10 Nov 2026):
  - 73,802 / 22,416 / 176 + 21 duplicates / 51,189: outputs/logs/clean.log (final run)
  - -0.107 with log annual responses: outputs/logs/checks.log
  - 6,493 collections with Flesch in [0, 100], readability CV-R2 0.004;
    bodies: 95% CI 0.20-0.69, best readability |rho| 0.32, b = 0.099 (p = 0.027),
    length p = 0.43; model change: rho 0.976, mean |diff| 2.8 / 3.3, 15.4% / 22.3%
    over five points; 70B collection index CV-R2 0.049 (t = -7.9):
    scripts/recompute_untabulated_results.py
  - Within-agency fixed effects , rule-based vs model-assessed dimensions,
    corrected resampled t-tests, bootstrap CIs of differences and UK subperiod trends: scripts/extended_validation.py
  - Global-rating benchmark (Section 4.3) and human-validation intervals (Section 4.4):
    scripts/global_rating_benchmark.py
  - Context controls, counts against coders, excerpting share, French canada.ca pages,
    trends without 2026 and family profiles: scripts/sensitivity_checks.py
  - Runtime: features 91 s for 51,189 documents on 48 cores (outputs/logs/features.log);
    LLM scoring about 6.3 documents/s per GPU on two GPUs (logs/score_main_shard*.log)
""")

print(f"=== {ok} OK, {bad} FAIL, {skipped} SKIP ===")
sys.exit(1 if bad else 0)
