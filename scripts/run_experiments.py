"""Comparisons, sensitivity and ablation on the assembled index.

Covers the experiments that need only the index and the dimensions:

  2  complexity by country
  3  complexity by level of administration
  4  complexity by type of procedure
  7  which dimension accounts for most of the variation
  8  ablation: what the index loses when a dimension is removed

Group comparisons are reported with an effect size and a bootstrap interval
rather than a p-value alone: with tens of thousands of documents almost any
difference is significant, and significance would say nothing about whether the
difference matters.

    python run_experiments.py --index pca
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import CONFIG, PROCESSED, SEED  # noqa: E402
from adminplex.index.build import BUILDERS, cronbach_alpha  # noqa: E402
from adminplex.index.dimensions import DIMENSION_KEYS  # noqa: E402
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.experiments")

GROUPINGS = {
    "country": ("country", 2),
    "admin_level": ("admin_level", 3),
    "doc_type": ("doc_type", 4),
}
MIN_GROUP = 30  # a group smaller than this cannot support a claim about it


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="pca", help="which candidate index is the ACI")
    ap.add_argument("--indices", default="outputs/tables/candidate_indices.parquet")
    ap.add_argument("--dimensions", default="outputs/tables/dimension_scores.parquet")
    ap.add_argument("--features", default="data/interim/features.parquet")
    ap.add_argument("--corpus", default=None, help="defaults to the processed corpus")
    ap.add_argument("--bootstrap", type=int, default=CONFIG["index"]["bootstrap_draws"])
    ap.add_argument("--out", default="outputs")
    return ap.parse_args()


def bootstrap_mean_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2 or draws <= 0:
        return float("nan"), float("nan")
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cliffs_delta(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, cap: int = 4000) -> float:
    """Probability a random document from a scores above one from b, rescaled.

    Nonparametric, so it survives the skew in most of the dimensions, and it is
    interpretable without reference to a distributional assumption.
    """
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if len(a) > cap:
        a = a[rng.integers(0, len(a), cap)]
    if len(b) > cap:
        b = b[rng.integers(0, len(b), cap)]
    greater = (a[:, None] > b[None, :]).sum()
    less = (a[:, None] < b[None, :]).sum()
    return float((greater - less) / (len(a) * len(b)))


def compare_groups(frame: pd.DataFrame, column: str, value: str, draws: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    counts = frame[column].value_counts()
    keep = counts[counts >= MIN_GROUP].index
    dropped = [str(g) for g in counts[counts < MIN_GROUP].index]
    if dropped:
        LOG.info("  %s: %d groups below %d documents, not reported (%s)",
                 column, len(dropped), MIN_GROUP, ", ".join(dropped[:6]))
    overall = frame.loc[frame[column].isin(keep), value].to_numpy()
    for group in keep:
        vals = frame.loc[frame[column] == group, value].to_numpy()
        rest = frame.loc[frame[column].isin(keep) & (frame[column] != group), value].to_numpy()
        lo, hi = bootstrap_mean_ci(vals, draws, rng)
        rows.append({
            column: group,
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "sd": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "ci_low": lo,
            "ci_high": hi,
            "difference_from_rest": float(vals.mean() - rest.mean()) if len(rest) else float("nan"),
            "cliffs_delta_vs_rest": cliffs_delta(vals, rest, rng),
        })
    out = pd.DataFrame(rows).sort_values("mean", ascending=False)
    out.attrs["overall_mean"] = float(overall.mean()) if len(overall) else float("nan")
    return out


def source_spread(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """How far each group is spread across sources.

    A group found in only one source cannot be separated from that source: any
    difference attributed to it is equally a difference between collections of
    documents that were gathered in different ways and are of different sizes.
    """
    tab = frame.groupby([column, "source"], observed=True).size().unstack(fill_value=0)
    return pd.DataFrame({
        column: tab.index,
        "n": tab.sum(axis=1).values,
        "n_sources": (tab > 0).sum(axis=1).values,
        "largest_source": tab.idxmax(axis=1).values,
        "largest_source_share": (tab.max(axis=1) / tab.sum(axis=1)).values,
    }).reset_index(drop=True)


def adjusted_comparison(frame: pd.DataFrame, column: str, draws: int) -> pd.DataFrame:
    """Group means after holding document length, and where possible source, fixed.

    Length is itself a dimension of the index, so this is not a corrected
    version of the raw comparison but the answer to a different question: is
    this group more complex than documents of its size usually are. Both are
    reported, because they say different things.

    Adjusted means are population-average predictions: every document is
    assigned in turn to each group and the predictions are averaged, so the
    comparison does not depend on which level happens to be the reference.
    """
    import statsmodels.formula.api as smf

    spread = source_spread(frame, column).set_index(column)
    nested = bool((spread["n_sources"] <= 1).all())
    formula = f"aci ~ log_words + C({column})"
    if not nested:
        formula += " + C(source)"
    else:
        LOG.warning(
            "  every %s appears in a single source; source cannot be held fixed and the "
            "adjustment controls length only", column
        )

    work = frame[[column, "aci", "log_words", "source"]].dropna()
    model = smf.ols(formula, data=work).fit()

    rows = []
    rng = np.random.default_rng(SEED)
    for group in work[column].unique():
        counterfactual = work.copy()
        counterfactual[column] = group
        adjusted = float(model.predict(counterfactual).mean())
        observed = work.loc[work[column] == group, "aci"]
        boot = []
        for _ in range(min(draws, 200)):
            idx = rng.integers(0, len(observed), len(observed))
            boot.append(float(observed.to_numpy()[idx].mean()))
        rows.append({
            column: group,
            "n": int(len(observed)),
            "raw_mean": float(observed.mean()),
            "adjusted_mean": adjusted,
            "adjustment": adjusted - float(observed.mean()),
            "raw_ci_low": float(np.percentile(boot, 2.5)) if boot else float("nan"),
            "raw_ci_high": float(np.percentile(boot, 97.5)) if boot else float("nan"),
            "n_sources": int(spread.loc[group, "n_sources"]) if group in spread.index else 0,
            "largest_source_share": float(spread.loc[group, "largest_source_share"]) if group in spread.index else float("nan"),
            "source_held_fixed": not nested,
        })
    out = pd.DataFrame(rows).sort_values("adjusted_mean", ascending=False)
    out.attrs["formula"] = formula
    out.attrs["r_squared"] = float(model.rsquared)
    LOG.info("  adjusted on %s (R2=%.3f)", formula, model.rsquared)
    return out


def within_source_comparison(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """The same comparison repeated inside each source separately.

    Where a difference survives within a single source it is not an artefact of
    how that source was gathered.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for source, block in frame.groupby("source", observed=True):
        counts = block[column].value_counts()
        levels = counts[counts >= MIN_GROUP].index
        if len(levels) < 2:
            continue
        for group in levels:
            vals = block.loc[block[column] == group, "aci"].to_numpy()
            rest = block.loc[block[column].isin(levels) & (block[column] != group), "aci"].to_numpy()
            rows.append({
                "source": source,
                column: group,
                "n": int(len(vals)),
                "mean": float(vals.mean()),
                "difference_from_rest_of_source": float(vals.mean() - rest.mean()) if len(rest) else float("nan"),
                "cliffs_delta_within_source": cliffs_delta(vals, rest, rng),
            })
    return pd.DataFrame(rows)


def dimension_contributions(dimensions: pd.DataFrame, index_values: pd.Series) -> pd.DataFrame:
    """Experiment 7: how much of the variation in the index each dimension tracks."""
    rows = []
    for key in dimensions.columns:
        col = dimensions[key].astype(float)
        if col.notna().sum() < 10:
            continue
        rows.append({
            "dimension": key,
            "pearson_with_index": float(col.corr(index_values)),
            "spearman_with_index": float(col.corr(index_values, method="spearman")),
            "sd": float(col.std(ddof=0)),
            "mean": float(col.mean()),
        })
    out = pd.DataFrame(rows)
    out["share_of_explained"] = out.pearson_with_index.abs() ** 2 / (out.pearson_with_index.abs() ** 2).sum()
    return out.sort_values("share_of_explained", ascending=False)


def ablation(dimensions: pd.DataFrame, builder_name: str, reference: pd.Series) -> pd.DataFrame:
    """Experiment 8: rebuild the index without each dimension in turn."""
    builder = BUILDERS[builder_name]
    rows = []
    for key in dimensions.columns:
        reduced = dimensions.drop(columns=[key])
        try:
            result = builder(reduced)
        except Exception as exc:
            LOG.warning("  ablation without %s failed: %s", key, exc)
            continue
        scores = pd.Series(result.scores.values, index=dimensions.index)
        rows.append({
            "dimension_removed": key,
            "spearman_with_full": float(scores.corr(reference, method="spearman")),
            "mean_absolute_shift": float((scores - reference).abs().mean()),
            "max_absolute_shift": float((scores - reference).abs().max()),
            "alpha_without": cronbach_alpha(reduced),
        })
    return pd.DataFrame(rows).sort_values("spearman_with_full")


def main() -> None:
    args = parse_args()
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(
        args.corpus or PROCESSED / "corpus.parquet",
        columns=["doc_id", "country", "jurisdiction", "source", "organisation", "admin_level", "doc_type", "language"],
    )
    indices = pd.read_parquet(args.indices)
    dimensions = pd.read_parquet(args.dimensions)

    if args.index not in indices.columns:
        LOG.error("index %r not among the candidates: %s", args.index, list(indices.columns))
        return

    features = pd.read_parquet(args.features, columns=["doc_id", "n_words"])
    frame = (
        corpus.merge(indices[["doc_id", args.index]], on="doc_id", how="inner")
        .merge(features, on="doc_id", how="left")
        .rename(columns={args.index: "aci"})
    )
    frame["log_words"] = np.log1p(frame["n_words"].astype(float))
    LOG.info("experiments on %d documents, index = %s", len(frame), args.index)

    # Experiments 2, 3 and 4 differ only in what defines a group.
    for name, (column, number) in GROUPINGS.items():
        if column not in frame.columns:
            LOG.warning("no %s column in the corpus, experiment %d skipped", column, number)
            continue
        subset = frame[frame[column].notna() & (frame[column].astype(str) != "")]
        if subset.empty:
            LOG.warning("%s is empty for every document, experiment %d skipped", column, number)
            continue

        table = compare_groups(subset, column, "aci", args.bootstrap)
        save_table(table, out / f"experiment{number}_by_{name}")
        print(f"\n--- Experiment {number}: complexity by {name} (raw) ---")
        print(table.round(2).to_string(index=False))

        counts = subset[column].value_counts()
        reportable = subset[subset[column].isin(counts[counts >= MIN_GROUP].index)]
        adjusted = adjusted_comparison(reportable, column, args.bootstrap)
        save_table(adjusted, out / f"experiment{number}_by_{name}_adjusted")
        print(f"\n--- Experiment {number}: {name}, holding length"
              f"{' and source' if adjusted.source_held_fixed.iloc[0] else ''} fixed ---")
        print(adjusted.round(2).to_string(index=False))

        within = within_source_comparison(reportable, column)
        if not within.empty:
            save_table(within, out / f"experiment{number}_by_{name}_within_source")
            print(f"\n--- Experiment {number}: {name}, within each source ---")
            print(within.round(2).to_string(index=False))
        else:
            LOG.info("  no source contains two or more %s groups; no within-source comparison", name)

    dims = dimensions.set_index("doc_id")
    dims = dims.loc[:, [c for c in DIMENSION_KEYS if c in dims.columns and dims[c].notna().any()]]
    aligned = frame.set_index("doc_id")["aci"].reindex(dims.index)

    contributions = dimension_contributions(dims, aligned)
    save_table(contributions, out / "experiment7_dimension_contributions")
    print("\n--- Experiment 7: what drives the index ---")
    print(contributions.round(3).to_string(index=False))

    ablation_table = ablation(dims, args.index, aligned)
    save_table(ablation_table, out / "experiment8_ablation")
    print("\n--- Experiment 8: ablation ---")
    print(ablation_table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
