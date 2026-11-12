"""Three real documents at low, middle and high complexity, with their profiles.

A reader cannot judge whether a measure is measuring anything without seeing
what it calls simple and what it calls hard. This picks documents at the tenth,
fiftieth and ninetieth percentiles of the index from a single source, so that the
three differ in complexity rather than in where they were published, and prints
the profile of each alongside an extract.

    python make_exemplars.py --source gov.uk
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import PROCESSED  # noqa: E402
from adminplex.index.dimensions import DIMENSIONS, DIMENSION_KEYS  # noqa: E402
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.exemplars")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gov.uk", help="hold the publication channel fixed")
    ap.add_argument("--index", default="pca")
    ap.add_argument("--extract-words", type=int, default=90)
    ap.add_argument("--out", default="outputs")
    return ap.parse_args()


def extract(text: str, words: int) -> str:
    clean = re.sub(r"\s+", " ", str(text)).strip()
    parts = clean.split()
    return " ".join(parts[:words]) + ("..." if len(parts) > words else "")


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(PROCESSED / "corpus.parquet",
                             columns=["doc_id", "source", "title", "organisation", "url", "text"])
    indices = pd.read_parquet("outputs/tables/candidate_indices.parquet")
    dims = pd.read_parquet("outputs/tables/dimension_scores.parquet")
    frame = corpus[corpus.source == args.source].merge(indices[["doc_id", args.index]], on="doc_id")
    frame = frame.merge(dims, on="doc_id").rename(columns={args.index: "aci"})
    keys = [k for k in DIMENSION_KEYS if k in frame.columns and frame[k].notna().any()]
    LOG.info("%d documents from %s", len(frame), args.source)

    picks = {}
    for label, q in (("low", 0.10), ("middle", 0.50), ("high", 0.90)):
        target = frame.aci.quantile(q)
        # The document closest to the percentile, among those long enough to
        # show a reader what the text actually looks like.
        candidates = frame[frame.text.str.split().str.len().between(250, 1200)]
        row = candidates.iloc[(candidates.aci - target).abs().argsort().iloc[0]]
        picks[label] = row

    labels = {d.key: d.label for d in DIMENSIONS}
    table = pd.DataFrame({
        "dimension": [labels[k] for k in keys],
        **{f"{label} (ACI {row.aci:.0f})": [round(float(row[k]), 1) for k in keys]
           for label, row in picks.items()},
    })
    save_table(table, out / "appendix_exemplar_profiles")

    lines = ["# Appendix F. Three documents and their profiles", "",
             f"All three come from {args.source}, so they differ in what they demand rather than",
             "in where they were published. Each dimension is a percentile within the corpus:",
             "50 is the corpus median, 90 means only one document in ten is more demanding on",
             "that dimension.", ""]
    for label, row in picks.items():
        lines += [
            f"## {label.capitalize()} complexity (index {row.aci:.0f})", "",
            f"**{row.title}**  ", f"{row.organisation or 'unattributed'} — <{row.url}>", "",
            "> " + extract(row.text, args.extract_words), "",
        ]
    lines += ["## Profiles", "",
              "| Dimension | " + " | ".join(c for c in table.columns[1:]) + " |",
              "|---" * len(table.columns) + "|"]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")

    path = out / "appendix_f_exemplars.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("written: %s", path)
    print(table.to_string(index=False))
    print()
    for label, row in picks.items():
        print(f"{label:7s} ACI {row.aci:5.1f}  {str(row.title)[:66]}")


if __name__ == "__main__":
    main()
