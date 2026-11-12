"""Build the sample and the offline tool for human coding.

Three design decisions carry the validity of the exercise.

  Same input      Coders see exactly the passage the model saw, produced by the
                  same excerpting function. Showing a person the whole document
                  and the model an extract would compare two different tasks.

  Blind           Nothing in the tool reveals the model's ratings, and documents
                  appear in an order that differs by coder, so a coder cannot
                  anchor on the previous document's difficulty.

  Repeats         Just under a fifth of the sample appears twice for each coder,
                  unlabelled.
                  That gives a within-coder reliability, which is the benchmark
                  the model's own repeat reliability should be judged against:
                  a model no less consistent than a person is doing well.

The sample spans the range of the index rather than being drawn at random, so
that agreement is estimated across the whole scale instead of the middle where
most documents sit.

    python make_coding_sample.py --n 100 --coders c1 c2 c3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import CONFIG, PROCESSED, SEED  # noqa: E402
from adminplex.features.llm import _RUBRIC, excerpt  # noqa: E402
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.coding")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="documents each coder rates")
    ap.add_argument("--coders", nargs="+", default=["coder1", "coder2", "coder3"])
    ap.add_argument("--sources", nargs="+", default=["gov.uk", "BOE"],
                    help="channels to sample from; keep to languages the coders read")
    ap.add_argument("--repeat-share", type=float, default=0.18,
                    help="share shown twice; below about 0.15 the consistency estimate is too noisy")
    ap.add_argument("--index", default="pca")
    ap.add_argument("--out", default="coding")
    return ap.parse_args()


def draw_sample(args: argparse.Namespace) -> pd.DataFrame:
    corpus = pd.read_parquet(
        PROCESSED / "corpus.parquet",
        columns=["doc_id", "source", "language", "title", "organisation", "url", "text"],
    )
    indices = pd.read_parquet("outputs/tables/candidate_indices.parquet")
    frame = corpus[corpus.source.isin(args.sources)].merge(
        indices[["doc_id", args.index]], on="doc_id"
    ).rename(columns={args.index: "model_index"})

    # Spread across the scale and across sources: five bands of the index within
    # each source, equal numbers from each.
    frame["band"] = frame.groupby("source", observed=True).model_index.transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False)
    )
    per_cell = max(1, args.n // (len(args.sources) * 5))
    picked = (frame.groupby(["source", "band"], observed=True, group_keys=False)
                   .apply(lambda g: g.sample(n=min(per_cell, len(g)), random_state=SEED)))
    picked = picked.sample(frac=1.0, random_state=SEED).head(args.n).reset_index(drop=True)
    LOG.info("sampled %d documents from %s", len(picked), ", ".join(args.sources))
    LOG.info("index range covered: %.1f to %.1f", picked.model_index.min(), picked.model_index.max())
    return picked


def build_payload(sample: pd.DataFrame, coder: str, repeat_share: float,
                  max_chars: int) -> list[dict]:
    """One coder's queue: every document once, a share of them twice."""
    rng = random.Random(f"{SEED}-{coder}")
    items = [
        {
            "item_id": f"{row.doc_id}#1",
            "doc_id": row.doc_id,
            "title": row.title or "(untitled)",
            "organisation": row.organisation or "",
            "url": row.url or "",
            "language": row.language,
            # The same passage the model was given.
            "text": excerpt(row.text or "", max_chars),
        }
        for row in sample.itertuples()
    ]
    repeats = rng.sample(items, k=max(1, int(len(items) * repeat_share)))
    items += [dict(r, item_id=r["item_id"].replace("#1", "#2")) for r in repeats]
    rng.shuffle(items)
    return items


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sample = draw_sample(args)
    # The key is kept apart from the tool so a coder cannot see the model's view.
    save_table(sample.drop(columns=["text"]), out / "sample_key")

    template = (Path(__file__).parent / "coding_template.html").read_text(encoding="utf-8")
    max_chars = CONFIG["llm"]["max_input_chars"]
    for coder in args.coders:
        items = build_payload(sample, coder, args.repeat_share, max_chars)
        page = (template
                .replace("__CODER__", coder)
                .replace("__RUBRIC__", json.dumps(_RUBRIC.strip()))
                .replace("__ITEMS__", json.dumps(items, ensure_ascii=False)))
        path = out / f"rate_{coder}.html"
        path.write_text(page, encoding="utf-8")
        LOG.info("%s: %d items (%d documents plus repeats) -> %s",
                 coder, len(items), len(sample), path.name)

    (out / "README.md").write_text(f"""# Human coding

## For each coder

1. Open `rate_<yourname>.html` by double-clicking it. No installation, no
   internet connection, nothing to log into.
2. Read the rubric at the top once. It is the same rubric the automated
   measurement was given.
3. Rate each passage. Progress is saved in the browser as you go, so you can
   close the tab and come back.
4. When the counter reaches the end, press **Download my ratings** and send the
   CSV file back.

Roughly {len(sample)} passages plus a few repeats. Most take two to four minutes.
Do them in several sittings rather than one; fatigue shows up in the data.

Some passages appear twice. That is deliberate and measures consistency. Rate the
second appearance as you find it, without trying to recall the first.

## For the analysis

Put the returned CSV files in this directory and run:

    python scripts/analyse_human_coding.py --coding coding

It reports agreement between coders, agreement between each coder and the
automated measurement, and each coder's consistency on the repeated passages.
""", encoding="utf-8")

    print(f"{len(sample)} documents, {len(args.coders)} coders -> {out}/")
    for coder in args.coders:
        print(f"  send {out}/rate_{coder}.html to {coder}")


if __name__ == "__main__":
    main()
