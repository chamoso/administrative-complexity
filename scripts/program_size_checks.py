"""Checks requested by review: scale confounding and cross-language comparability.

Two objections deserve numbers rather than prose.

  Scale        Annual burden hours conflate the time one response takes with how
               many responses there are. The criterion used throughout is hours
               per response, not the annual total, but a measure of complexity
               could still be picking up the size of a programme. If it is, the
               association should weaken once the number of responses is held
               fixed.

  Language     Several conclusions compare countries whose documents are in
               different languages. Equivalent measurement across languages
               cannot be assumed from equivalent lexicons. Full invariance
               testing needs a criterion in each language, which does not exist
               here, but the structure of the dimensions can be compared, and a
               structure that differs sharply by language would be a warning.

    python program_size_checks.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import CONFIG, PROCESSED  # noqa: E402
from adminplex.index.dimensions import DIMENSION_KEYS  # noqa: E402
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.checks")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="pca")
    ap.add_argument("--external", default="outputs/external/collections")
    ap.add_argument("--out", default="outputs")
    return ap.parse_args()


def scale_confound(args) -> pd.DataFrame:
    ext = Path(args.external)
    d = (pd.read_parquet(PROCESSED / "validation_collections.parquet")
         .merge(pd.read_parquet(ext / "candidate_indices.parquet"), on="doc_id")
         .merge(pd.read_parquet(ext / "features.parquet")[["doc_id", "n_words"]], on="doc_id"))
    d = d[d.log_hours_per_response.notna() & np.isfinite(d.log_hours_per_response)]
    d = d[d.annual_responses > 0].copy()
    d["log_responses"] = np.log1p(d.annual_responses)
    d["log_words"] = np.log1p(d.n_words)
    d = d.rename(columns={args.index: "aci"})

    LOG.info("%d collections with a response count", len(d))
    LOG.info("correlation of the index with programme size (log responses): %.3f",
             d.aci.corr(d.log_responses, method="spearman"))

    rows = []
    for label, formula in [
        ("hours per response ~ index", "log_hours_per_response ~ aci"),
        ("+ programme size", "log_hours_per_response ~ aci + log_responses"),
        ("+ programme size and length", "log_hours_per_response ~ aci + log_responses + log_words"),
        ("annual total ~ index", "np.log1p(annual_burden_hours) ~ aci"),
        ("annual total ~ index + size", "np.log1p(annual_burden_hours) ~ aci + log_responses"),
    ]:
        fit = smf.ols(formula, data=d).fit()
        rows.append({
            "specification": label,
            "n": int(fit.nobs),
            "index_coefficient": float(fit.params["aci"]),
            "index_p": float(fit.pvalues["aci"]),
            "r_squared": float(fit.rsquared),
            "responses_p": float(fit.pvalues.get("log_responses", float("nan"))),
        })
        LOG.info("  %-30s coef %.4f (p=%.4g)  R2=%.3f", label,
                 rows[-1]["index_coefficient"], rows[-1]["index_p"], rows[-1]["r_squared"])
    return pd.DataFrame(rows)


def language_structure(args) -> pd.DataFrame:
    dims = pd.read_parquet("outputs/tables/dimension_scores.parquet")
    corpus = pd.read_parquet(PROCESSED / "corpus.parquet", columns=["doc_id", "language", "country"])
    d = dims.merge(corpus, on="doc_id")
    keys = [k for k in DIMENSION_KEYS if k in d.columns and d[k].notna().any()]

    big = d.language.value_counts()
    languages = big[big >= 500].index.tolist()
    LOG.info("languages with at least 500 documents: %s", ", ".join(languages))

    english = d[d.language == "en"][keys].corr(method="spearman")
    rows = []
    for lang in languages:
        block = d[d.language == lang]
        corr = block[keys].corr(method="spearman")
        # Similarity of the correlation structure: how far the pattern of
        # relations between dimensions is the same one in each language.
        triu = np.triu_indices(len(keys), k=1)
        rows.append({
            "language": lang,
            "n": int(len(block)),
            "structure_similarity_to_english": float(
                np.corrcoef(english.values[triu], corr.values[triu])[0, 1]
            ),
            "mean_index_dimension_sd": float(block[keys].std().mean()),
        })
        LOG.info("  %-4s n=%6d  structure similarity %.3f", lang, len(block), rows[-1]
                 ["structure_similarity_to_english"])
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)

    print("=== Scale: does the index track programme size rather than complexity? ===")
    scale = scale_confound(args)
    save_table(scale, out / "review_scale_confound")
    print(scale.round(4).to_string(index=False))

    print()
    print("=== Language: is the structure of the dimensions the same one? ===")
    lang = language_structure(args)
    save_table(lang, out / "review_language_structure")
    print(lang.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
