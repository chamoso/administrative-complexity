"""Experiments 9 and 10: how stable the judged dimensions are.

Three things are varied, one at a time, and the assessment is repeated:

  the rubric    paraphrased, keeping every scale and anchor intact
  the model     a second open-weights model of a different family and size
  the sampling  the same rubric and model run repeatedly at a temperature
                above zero, which is the only source of variation left

A measure whose scores move when the wording of the rubric changes is
measuring the wording. The agreement statistics reported here are what allow
the reader to judge that for themselves, so they are reported whatever they
say.

    python analyse_robustness.py --scores outputs/llm_scores --main main \\
        --variants variant alt_model --repeats repeat1 repeat2 repeat3
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.robustness")

SCALES = ["llm_cognitive_load", "llm_legal_knowledge", "llm_clarity"]
COUNTS = ["llm_n_requirements", "llm_n_documents"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/llm_scores")
    ap.add_argument("--main", default="main")
    ap.add_argument("--variants", nargs="*", default=["variant", "alt_model"])
    ap.add_argument("--repeats", nargs="*", default=["repeat1", "repeat2", "repeat3"])
    ap.add_argument("--out", default="outputs")
    return ap.parse_args()


def load_tag(directory: str, tag: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(directory) / f"scores_{tag}_shard*.jsonl")))
    if not files:
        return pd.DataFrame()
    rows = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if frame.empty or "parsed_ok" not in frame.columns:
        return pd.DataFrame()
    frame = frame[frame.parsed_ok.fillna(False).astype(bool)]
    keep = ["doc_id", *[c for c in SCALES + COUNTS if c in frame.columns]]
    return frame[keep].drop_duplicates(subset="doc_id", keep="first")


def quadratic_weighted_kappa(a: np.ndarray, b: np.ndarray, lo: int = 1, hi: int = 10) -> float:
    """Agreement on an ordered scale, penalising distant disagreements more.

    The standard measure for rubric-scored ordinal judgements; unlike a
    correlation it is sensitive to systematic shifts, which is what a changed
    rubric would produce.
    """
    a = np.clip(np.round(a).astype(int), lo, hi)
    b = np.clip(np.round(b).astype(int), lo, hi)
    n = hi - lo + 1
    observed = np.zeros((n, n))
    for x, y in zip(a, b):
        observed[x - lo, y - lo] += 1
    weights = np.array([[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)])
    hist_a = np.bincount(a - lo, minlength=n).astype(float)
    hist_b = np.bincount(b - lo, minlength=n).astype(float)
    expected = np.outer(hist_a, hist_b) / len(a)
    denom = float((weights * expected).sum())
    return 1.0 - float((weights * observed).sum()) / denom if denom > 0 else float("nan")


def agreement(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> list[dict]:
    merged = reference.merge(other, on="doc_id", suffixes=("_a", "_b"))
    rows = []
    for column in SCALES + COUNTS:
        ca, cb = f"{column}_a", f"{column}_b"
        if ca not in merged.columns or cb not in merged.columns:
            continue
        a, b = merged[ca].astype(float), merged[cb].astype(float)
        usable = a.notna() & b.notna()
        a, b = a[usable].to_numpy(), b[usable].to_numpy()
        if len(a) < 20:
            continue
        row = {
            "comparison": label,
            "measure": column,
            "n": int(len(a)),
            "pearson": float(np.corrcoef(a, b)[0, 1]),
            "spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
            "mean_absolute_difference": float(np.abs(a - b).mean()),
            "mean_shift": float((b - a).mean()),
            "exact_agreement": float((np.round(a) == np.round(b)).mean()),
            "within_one_point": float((np.abs(a - b) <= 1).mean()),
        }
        if column in SCALES:
            row["quadratic_weighted_kappa"] = quadratic_weighted_kappa(a, b)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)

    main_scores = load_tag(args.scores, args.main)
    if main_scores.empty:
        LOG.error("no scores for the main run (tag %r)", args.main)
        return
    LOG.info("main run: %d documents", len(main_scores))

    rows = []
    for tag in args.variants:
        other = load_tag(args.scores, tag)
        if other.empty:
            LOG.warning("no scores for variant %r, skipped", tag)
            continue
        LOG.info("variant %s: %d documents", tag, len(other))
        rows.extend(agreement(main_scores, other, tag))

    # Experiment 10: repeated runs of the identical configuration.
    repeats = [load_tag(args.scores, tag) for tag in args.repeats]
    repeats = [r for r in repeats if not r.empty]
    if len(repeats) >= 2:
        for i in range(len(repeats)):
            for j in range(i + 1, len(repeats)):
                rows.extend(agreement(repeats[i], repeats[j], f"repeat {i + 1} vs {j + 1}"))

        merged = repeats[0][["doc_id"]]
        for k, frame in enumerate(repeats):
            merged = merged.merge(frame, on="doc_id", suffixes=("", f"_r{k}"))
        stability = []
        for column in SCALES + COUNTS:
            cols = [c for c in merged.columns if c.startswith(column)]
            if len(cols) < 2:
                continue
            values = merged[cols].astype(float)
            within_sd = values.std(axis=1, ddof=1)
            between_sd = values.mean(axis=1).std(ddof=1)
            stability.append({
                "measure": column,
                "n_documents": int(len(values)),
                "n_runs": len(cols),
                "mean_within_document_sd": float(within_sd.mean()),
                "between_document_sd": float(between_sd),
                # Share of the variation that is between documents rather than
                # between runs of the same document: an intraclass reliability.
                "reliability": float(
                    between_sd**2 / (between_sd**2 + within_sd.pow(2).mean())
                ) if between_sd > 0 else float("nan"),
                "identical_in_every_run": float((values.nunique(axis=1) == 1).mean()),
            })
        if stability:
            save_table(pd.DataFrame(stability), out / "experiment10_consistency")
            print("\n--- Experiment 10: consistency across identical runs ---")
            print(pd.DataFrame(stability).round(3).to_string(index=False))
    else:
        LOG.warning("fewer than two repeat runs available; experiment 10 not produced")

    if rows:
        table = pd.DataFrame(rows)
        save_table(table, out / "experiment9_robustness")
        print("\n--- Experiment 9: agreement when the rubric or the model changes ---")
        print(table.round(3).to_string(index=False))
    else:
        LOG.warning("no variant runs available; experiment 9 not produced")


if __name__ == "__main__":
    main()
