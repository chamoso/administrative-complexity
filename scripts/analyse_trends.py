"""Has administrative complexity risen, and which bodies produce the most of it?

Two questions the corpus can answer that a single cross-section cannot.

  Over time    Three jurisdictions publish dated material continuously, so the
               index can be traced year by year. Regulatory accumulation is
               usually argued from counts of rules; here it is measured on what
               the rules demand of the reader.

  By body      Which organisations publish the most demanding procedures, once
               document length and type are held fixed. This is the comparison
               a public manager can act on, and the one most easily made
               unfairly, so it is reported adjusted as well as raw.

    python analyse_trends.py --index pca
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import INTERIM, PROCESSED, SEED  # noqa: E402
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.trends")

MIN_PER_YEAR = 50
MIN_PER_BODY = 30


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="pca")
    ap.add_argument("--indices", default="outputs/tables/candidate_indices.parquet")
    ap.add_argument("--features", default=str(INTERIM / "features.parquet"))
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--from-year", type=int, default=2015)
    ap.add_argument("--to-year", type=int, default=2026)
    return ap.parse_args()


def parse_dates(series: pd.Series) -> pd.Series:
    """Dates arrive in whatever format each source publishes.

    Inferring one format for the whole column silently discards every row that
    does not match it, which is how a corpus with dates on two thirds of its
    documents came to look as though it had almost none.
    """
    text = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    compact = text.str.fullmatch(r"\d{8}").fillna(False)
    parsed.loc[compact] = pd.to_datetime(text[compact], format="%Y%m%d", errors="coerce")

    rest = ~compact & text.notna()
    parsed.loc[rest] = pd.to_datetime(text[rest], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)

    LOG.info("dates parsed for %d of %d documents (%.0f%%)",
             parsed.notna().sum(), len(parsed), 100 * parsed.notna().mean())
    return parsed


def yearly_trend(frame: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    keys = ["year"] + ([group] if group else [])
    rows = []
    for key, block in frame.groupby(keys, observed=True):
        if len(block) < MIN_PER_YEAR:
            continue
        record = {"year": key[0] if isinstance(key, tuple) else key}
        if group:
            record[group] = key[1]
        record.update({
            "n": int(len(block)),
            "mean_aci": float(block.aci.mean()),
            "median_aci": float(block.aci.median()),
            "sd": float(block.aci.std(ddof=1)),
            "se": float(block.aci.std(ddof=1) / np.sqrt(len(block))),
            "median_words": float(block.n_words.median()),
        })
        rows.append(record)
    return pd.DataFrame(rows).sort_values(keys)


def trend_slope(frame: pd.DataFrame) -> dict[str, float]:
    """Change per year, with and without holding document length fixed.

    A rise in the index that disappears once length is held fixed is a rise in
    how much is published, not in how much is demanded, and the two should not
    be reported as the same thing.
    """
    import statsmodels.formula.api as smf

    work = frame.dropna(subset=["aci", "year", "n_words"]).copy()
    work["log_words"] = np.log1p(work.n_words)
    raw = smf.ols("aci ~ year", data=work).fit()
    adjusted = smf.ols("aci ~ year + log_words", data=work).fit()
    return {
        "n": int(len(work)),
        "points_per_year_raw": float(raw.params["year"]),
        "p_raw": float(raw.pvalues["year"]),
        "points_per_year_adjusted": float(adjusted.params["year"]),
        "p_adjusted": float(adjusted.pvalues["year"]),
        "words_per_year": float(smf.ols("n_words ~ year", data=work).fit().params["year"]),
    }


def by_organisation(frame: pd.DataFrame) -> pd.DataFrame:
    """Bodies ranked raw and with length and document type held fixed."""
    import statsmodels.formula.api as smf

    counts = frame.organisation.value_counts()
    keep = counts[counts >= MIN_PER_BODY].index
    work = frame[frame.organisation.isin(keep)].dropna(subset=["aci", "n_words"]).copy()
    work["log_words"] = np.log1p(work.n_words)
    LOG.info("%d bodies with at least %d documents, covering %d documents",
             len(keep), MIN_PER_BODY, len(work))

    formula = "aci ~ log_words + C(organisation)"
    if work.doc_type.nunique() > 1:
        formula += " + C(doc_type)"
    model = smf.ols(formula, data=work).fit()

    rows = []
    for body, block in work.groupby("organisation", observed=True):
        counterfactual = work.copy()
        counterfactual["organisation"] = body
        rows.append({
            "organisation": body,
            "country": block.country.mode().iat[0],
            "n": int(len(block)),
            "raw_mean": float(block.aci.mean()),
            "adjusted_mean": float(model.predict(counterfactual).mean()),
            "median_words": float(block.n_words.median()),
        })
    out = pd.DataFrame(rows)
    out["adjustment"] = out.adjusted_mean - out.raw_mean
    return out.sort_values("adjusted_mean", ascending=False)


def main() -> None:
    args = parse_args()
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(
        PROCESSED / "corpus.parquet",
        columns=["doc_id", "country", "source", "organisation", "doc_type", "date"],
    )
    indices = pd.read_parquet(args.indices)
    features = pd.read_parquet(args.features, columns=["doc_id", "n_words"])
    frame = (
        corpus.merge(indices[["doc_id", args.index]], on="doc_id")
        .merge(features, on="doc_id", how="left")
        .rename(columns={args.index: "aci"})
    )
    frame["parsed_date"] = parse_dates(frame["date"])
    frame["year"] = frame.parsed_date.dt.year

    dated = frame[frame.year.between(args.from_year, args.to_year)].copy()
    LOG.info("%d documents dated between %d and %d, from %d sources",
             len(dated), args.from_year, args.to_year, dated.source.nunique())

    save_table(yearly_trend(dated), out / "trend_overall")
    save_table(yearly_trend(dated, "country"), out / "trend_by_country")

    slopes = [{"scope": "all", **trend_slope(dated)}]
    for country, block in dated.groupby("country", observed=True):
        if len(block) >= 500 and block.year.nunique() >= 5:
            slopes.append({"scope": country, **trend_slope(block)})
    slope_table = pd.DataFrame(slopes)
    save_table(slope_table, out / "trend_slopes")

    bodies = by_organisation(frame)
    save_table(bodies, out / "organisation_ranking")

    print("--- Complexity over time, change per year ---")
    print(slope_table.round(4).to_string(index=False))
    print()
    print("--- Ten most demanding bodies, length and document type held fixed ---")
    print(bodies.head(10).round(1).to_string(index=False))
    print()
    print("--- Ten least demanding ---")
    print(bodies.tail(10).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
