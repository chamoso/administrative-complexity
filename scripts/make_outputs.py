"""Produce the descriptive table and every figure.

Reads what the earlier stages wrote and produces nothing new analytically: if a
number appears in a figure it also appears in a table, and both come from the
same file.

    python make_outputs.py --index pca
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import CONFIG, PROCESSED  # noqa: E402
from adminplex.index.dimensions import DIMENSIONS, DIMENSION_KEYS  # noqa: E402
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402
from adminplex.viz import figures  # noqa: E402

LOG = get_logger("adminplex.outputs")
READABILITY = CONFIG["analysis"]["readability_indices"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="pca")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--features", default="data/interim/features.parquet")
    return ap.parse_args()


def corpus_description(corpus: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Table 1: what the corpus is made of."""
    frame = corpus.merge(features[["doc_id", "n_words", "n_sentences"]], on="doc_id", how="left")
    rows = []
    for (country, source), block in frame.groupby(["country", "source"], observed=True):
        rows.append({
            "country": country,
            "source": source,
            "documents": len(block),
            "share_of_corpus": len(block) / len(frame),
            "median_words": float(block.n_words.median()),
            "p10_words": float(block.n_words.quantile(0.10)),
            "p90_words": float(block.n_words.quantile(0.90)),
            "languages": ", ".join(sorted(block.language.dropna().unique())[:4]),
            "document_types": block.doc_type.nunique(),
            "organisations": block.organisation.nunique(),
        })
    table = pd.DataFrame(rows).sort_values("documents", ascending=False)
    total = pd.DataFrame([{
        "country": "All", "source": "All", "documents": len(frame), "share_of_corpus": 1.0,
        "median_words": float(frame.n_words.median()),
        "p10_words": float(frame.n_words.quantile(0.10)),
        "p90_words": float(frame.n_words.quantile(0.90)),
        # A count where the other rows list names, kept as text so the column
        # has one type.
        "languages": f"{frame.language.nunique()} in total",
        "document_types": frame.doc_type.nunique(),
        "organisations": frame.organisation.nunique(),
    }])
    return pd.concat([table, total], ignore_index=True)


def dimension_description(dimensions: pd.DataFrame) -> pd.DataFrame:
    """Table 2: the dimensions, what each is built from, and how it behaves."""
    definitions = {d.key: d for d in DIMENSIONS}
    rows = []
    for key in DIMENSION_KEYS:
        if key not in dimensions.columns:
            continue
        col = dimensions[key].astype(float)
        d = definitions[key]
        rows.append({
            "dimension": d.label,
            "key": key,
            "source": d.source,
            "measured_from": ", ".join(d.features),
            "description": d.note,
            "coverage": float(col.notna().mean()),
            "mean": float(col.mean()),
            "sd": float(col.std(ddof=0)),
            "p10": float(col.quantile(0.10)),
            "p90": float(col.quantile(0.90)),
        })
    return pd.DataFrame(rows)


def maybe(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    LOG.info("not present, skipped: %s", path.name)
    return None


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    tables, figs = out / "tables", out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    figures.apply_style()

    corpus = pd.read_parquet(
        PROCESSED / "corpus.parquet",
        columns=["doc_id", "country", "source", "organisation", "admin_level", "doc_type", "language"],
    )
    features = pd.read_parquet(args.features)
    dimensions = pd.read_parquet(tables / "dimension_scores.parquet")
    indices = pd.read_parquet(tables / "candidate_indices.parquet")

    save_table(corpus_description(corpus, features), tables / "table1_corpus")
    save_table(dimension_description(dimensions.drop(columns=["doc_id"])), tables / "table2_dimensions")
    LOG.info("descriptive tables written")

    frame = (
        corpus.merge(indices[["doc_id", args.index]], on="doc_id")
        .merge(features, on="doc_id", how="left")
        .rename(columns={args.index: "aci"})
    )
    dims = dimensions.set_index("doc_id")
    dims = dims.loc[:, [c for c in DIMENSION_KEYS if c in dims.columns and dims[c].notna().any()]]

    figures.index_distribution(frame["aci"], figs)
    figures.dimension_correlations(dims, figs)
    figures.index_vs_readability(frame, "aci", READABILITY, figs)

    diagnostics = tables / "index_diagnostics.json"
    if diagnostics.exists():
        import json
        diag = json.loads(diagnostics.read_text())
        explained = diag.get("diagnostics", {}).get("pca", {}).get("explained_variance_ratio")
        if explained:
            figures.pca_scree(explained, figs)

    for group, number in (("country", 2), ("admin_level", 3), ("doc_type", 4)):
        if group not in frame.columns or frame[group].notna().sum() == 0:
            continue
        figures.group_violins(frame.dropna(subset=[group]), group, "aci", figs,
                              name=f"fig{number}_violin_{group}")
        adjusted = maybe(tables / f"experiment{number}_by_{group}_adjusted.parquet")
        if adjusted is not None:
            figures.group_ranking(adjusted, group, figs, name=f"fig{number}_ranking_{group}")

    trend = maybe(tables / "trend_by_country.parquet")
    if trend is not None:
        figures.trend_over_time(trend, figs, name="fig5_trend")

    bodies = maybe(tables / "organisation_ranking.parquet")
    if bodies is not None:
        figures.organisation_ranking(bodies, figs, name="fig6_organisations")

    profiles = maybe(tables / "cluster_profiles.parquet")
    cluster_summary = maybe(tables / "cluster_summary.parquet")
    if profiles is not None and cluster_summary is not None:
        figures.cluster_profiles(profiles, cluster_summary, figs)

    ablation = maybe(tables / "experiment8_ablation.parquet")
    if ablation is not None:
        figures.ablation_plot(ablation, figs)

    for level in ("parts", "collections"):
        comparison = maybe(tables / f"burden_model_comparison_{level}.parquet")
        if comparison is not None:
            figures.burden_fit(comparison, figs, name=f"fig_burden_fit_{level}")

    written = sorted(p.name for p in figs.glob("*.pdf"))
    LOG.info("%d figures written: %s", len(written), ", ".join(written))


if __name__ == "__main__":
    main()
