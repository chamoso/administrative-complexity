"""Does the measurement travel between jurisdictions?

The obvious test, training on six countries and testing on the seventh, cannot be
run: the burden criterion exists only for United States federal collections, so
six of the seven jurisdictions have nothing to test against. Two questions that
can be answered stand in for it.

  Does the instrument change if a country is removed?
      The percentile scale and the aggregation weights are both estimated on the
      corpus. Re-estimating them with one jurisdiction held out and rescoring that
      jurisdiction's documents shows whether its position depends on having been
      part of the estimation. A measure whose scores shift when a country leaves
      the reference is not a measure of that country's documents.

  Does agreement with human judgement hold in both languages?
      The coding sample covers English and Spanish documents. Agreement computed
      separately in each is the only cross-language validity evidence available,
      and it is against human judgement rather than against a criterion.

    python check_generalisation.py
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import CONFIG, INTERIM, PROCESSED  # noqa: E402
from adminplex.index.build import BUILDERS  # noqa: E402
from adminplex.index.dimensions import DIMENSION_KEYS, build_dimension_scores  # noqa: E402
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.generalisation")
SCALES = ["cognitive_load", "legal_knowledge", "clarity"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="pca")
    ap.add_argument("--scores", default="outputs/llm_scores")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--coding", default="ratings")
    ap.add_argument("--out", default="outputs")
    return ap.parse_args()


def load_llm(directory: str, tag: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(str(Path(directory) / f"scores_{tag}_shard*.jsonl")):
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("parsed_ok"):
                rows.append(rec)
    frame = pd.DataFrame(rows)
    keep = ["doc_id", *[c for c in frame.columns if c.startswith("llm_")]]
    return frame[keep].drop_duplicates("doc_id")


def leave_one_country_out(args) -> pd.DataFrame:
    features = pd.read_parquet(INTERIM / "features.parquet").merge(
        load_llm(args.scores, args.tag), on="doc_id", how="left")
    corpus = pd.read_parquet(PROCESSED / "corpus.parquet", columns=["doc_id", "country"])
    features = features.merge(corpus, on="doc_id")
    published = pd.read_parquet("outputs/tables/candidate_indices.parquet")[["doc_id", args.index]]
    scaling = CONFIG["index"]["scaling"]

    rows = []
    for country in sorted(features.country.unique()):
        held = features[features.country == country]
        rest = features[features.country != country]
        if len(held) < 200:
            continue
        # Scale and weights estimated without the held-out jurisdiction.
        rest_dims = build_dimension_scores(rest, scaling=scaling)
        usable = [k for k in DIMENSION_KEYS if k in rest_dims.columns and rest_dims[k].notna().any()]
        weights = pd.Series(BUILDERS[args.index](rest_dims.set_index("doc_id")[usable]).weights)
        weights = weights.reindex(usable).fillna(0.0)
        held_dims = build_dimension_scores(held, scaling=scaling, reference=rest)
        values = held_dims.set_index("doc_id")[usable].fillna(50.0)
        rescored = ((values * weights).sum(axis=1) / weights.abs().sum()).rename("held_out")

        joined = rescored.reset_index().merge(published, on="doc_id")
        rows.append({
            "country": country,
            "n": int(len(joined)),
            "rank_correlation": float(joined.held_out.corr(joined[args.index], method="spearman")),
            "mean_shift": float((joined.held_out - joined[args.index]).mean()),
            "mean_absolute_shift": float((joined.held_out - joined[args.index]).abs().mean()),
            "share_moving_over_5": float(((joined.held_out - joined[args.index]).abs() > 5).mean()),
        })
        LOG.info("  %-3s n=%6d  rank %.3f  mean shift %+.1f  moving >5 points %.0f%%",
                 country, rows[-1]["n"], rows[-1]["rank_correlation"],
                 rows[-1]["mean_shift"], 100 * rows[-1]["share_moving_over_5"])
    return pd.DataFrame(rows)


def agreement_by_language(args) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(args.coding) / "ratings_*.csv")))
    if not files:
        LOG.warning("no human ratings in %s", args.coding)
        return pd.DataFrame()
    human = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    human = human[~human.item_id.str.endswith("#2")]
    human["composite"] = pd.concat(
        [human.cognitive_load, human.legal_knowledge, 11 - human.clarity], axis=1).mean(axis=1)

    key = pd.read_csv("coding/sample_key.csv")[["doc_id", "language", "model_index"]]
    model = load_llm(args.scores, args.tag)
    joined = human.merge(key, on="doc_id").merge(model, on="doc_id", how="left")

    rows = []
    for language, block in joined.groupby("language"):
        panel = block.groupby("doc_id").agg(
            panel=("composite", "mean"), model_index=("model_index", "first")).dropna()
        row = {"language": language, "n_documents": int(len(panel)),
               "panel_vs_index": float(panel.panel.corr(panel.model_index, method="spearman"))}
        # Between-coder agreement inside the language, as the local ceiling.
        wide = block.pivot_table(index="doc_id", columns="coder", values="composite")
        pairs = [wide[a].corr(wide[b], method="spearman")
                 for i, a in enumerate(wide.columns) for b in wide.columns[i + 1:]]
        row["between_coders"] = float(np.nanmean(pairs)) if pairs else float("nan")
        for scale in SCALES:
            got = block.dropna(subset=[scale, f"llm_{scale}"])
            if len(got) >= 30:
                row[f"{scale}_model_vs_coder"] = float(
                    got[scale].corr(got[f"llm_{scale}"], method="spearman"))
        rows.append(row)
        LOG.info("  %s: n=%d  panel vs index %.3f  between coders %.3f",
                 language, row["n_documents"], row["panel_vs_index"], row["between_coders"])
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out = Path(args.out) / "tables"

    print("=== Rescoring each jurisdiction with it held out of the estimation ===")
    loco = leave_one_country_out(args)
    save_table(loco, out / "generalisation_leave_country_out")
    print(loco.round(3).to_string(index=False))

    print()
    print("=== Agreement with human judgement, by language ===")
    lang = agreement_by_language(args)
    if not lang.empty:
        save_table(lang, out / "generalisation_by_language")
        print(lang.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
