"""Agreement between coders, and between coders and the automated measurement.

The comparison that matters is not whether the model agrees with people. It is
whether the model agrees with people about as well as people agree with each
other. A task on which two trained coders reach kappa 0.5 does not permit a
machine to reach 0.9, and a model that matches the human ceiling has done as well
as the task allows.

Three quantities are therefore reported side by side:

  between coders   the ceiling the task itself imposes
  coder to model   what the automated measurement achieves against that ceiling
  within coder     each coder's own consistency on the repeated passages

    python analyse_human_coding.py --coding coding
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.human")

SCALES = ["cognitive_load", "legal_knowledge", "clarity"]
COUNTS = ["n_requirements", "n_documents"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coding", default="coding", help="directory holding the returned CSV files")
    ap.add_argument("--scores", default="outputs/llm_scores")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--sample-key", default="coding/sample_key.csv",
                    help="fallback source for the model's index when the per-scale scores "
                         "are not to hand")
    ap.add_argument("--out", default="outputs")
    return ap.parse_args()


def quadratic_kappa(a: np.ndarray, b: np.ndarray, lo: int = 1, hi: int = 10) -> float:
    a = np.clip(np.round(a).astype(int), lo, hi)
    b = np.clip(np.round(b).astype(int), lo, hi)
    n = hi - lo + 1
    observed = np.zeros((n, n))
    for x, y in zip(a, b):
        observed[x - lo, y - lo] += 1
    weights = np.array([[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)])
    expected = np.outer(np.bincount(a - lo, minlength=n), np.bincount(b - lo, minlength=n)) / len(a)
    denominator = float((weights * expected).sum())
    return 1.0 - float((weights * observed).sum()) / denominator if denominator > 0 else float("nan")


def agreement(a: pd.Series, b: pd.Series, measure: str, label: str,
              min_n: int = 20) -> dict | None:
    """One pairwise comparison, or nothing if there is too little overlap.

    The repeated-passage comparison has far fewer pairs than the others by
    design, so it uses a lower floor. Its estimates are correspondingly
    imprecise and the count is reported alongside every figure.
    """
    both = a.notna() & b.notna()
    x, y = a[both].to_numpy(float), b[both].to_numpy(float)
    if len(x) < min_n:
        return None
    row = {
        "comparison": label,
        "measure": measure,
        "n": int(len(x)),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(pd.Series(x).corr(pd.Series(y), method="spearman")),
        "mean_absolute_difference": float(np.abs(x - y).mean()),
        "mean_shift": float((y - x).mean()),
        "within_one_point": float((np.abs(x - y) <= 1).mean()),
    }
    if measure in SCALES:
        row["quadratic_weighted_kappa"] = quadratic_kappa(x, y)
    return row


def check_plausible(frame: pd.DataFrame) -> list[str]:
    """Does this look like it was produced by people reading documents?

    Ratings that arrive already tabulated cannot be told from ratings that were
    typed, except by their traces. Three traces are cheap to check and were each
    absent from a set of files that reached us once: a timestamp per rating rather
    than one for the whole file, some disagreement with oneself on the repeated
    passages, and a session long enough to have read the documents. None of these
    proves provenance. All three failing at once means the numbers should not be
    reported as human coding.
    """
    problems = []
    for coder, block in frame.groupby("coder"):
        stamps = block.rated_at.nunique()
        if stamps < 0.5 * len(block):
            problems.append(f"{coder}: {stamps} distinct timestamps for {len(block)} ratings")
        try:
            times = pd.to_datetime(block.rated_at, format="mixed", utc=True)
            hours = (times.max() - times.min()).total_seconds() / 3600
            if hours < 0.35 * len(block) / 60:
                problems.append(f"{coder}: {len(block)} ratings in {hours:.2f} hours")
        except Exception:
            problems.append(f"{coder}: timestamps could not be read")
        repeated = block[block.doc_id.duplicated(keep=False)]
        if len(repeated) >= 4:
            pairs = repeated.groupby("doc_id")[SCALES + COUNTS].nunique()
            identical = (pairs.max(axis=1) == 1).mean()
            if identical > 0.9:
                problems.append(f"{coder}: {identical:.0%} of repeated passages rated identically")
    return problems


def load_human(directory: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(directory) / "ratings_*.csv")))
    if not files:
        raise SystemExit(f"no ratings_*.csv in {directory}/ — nothing returned yet")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame["pass"] = frame.item_id.str.extract(r"#(\d)$").astype(int)
    LOG.info("%d ratings from %d coders over %d documents",
             len(frame), frame.coder.nunique(), frame.doc_id.nunique())
    problems = check_plausible(frame)
    if problems:
        LOG.error("these ratings do not look like human coding, and agreement computed "
                  "from them would not mean anything:")
        for problem in problems:
            LOG.error("    %s", problem)
        LOG.error("check the source of the files before using any figure below")
    else:
        LOG.info("plausibility checks passed: timestamps, session length and "
                 "self-disagreement on repeats all look like typed ratings")
    return frame


def human_composite(frame: pd.DataFrame) -> pd.Series:
    """One overall human judgement per rating, on the scale the index runs on.

    Clarity is reverse-coded so that every component points the same way, higher
    meaning more demanding, and the three are averaged. This is the closest a
    coder gets to the quantity the index reports.
    """
    return pd.concat([frame.cognitive_load, frame.legal_knowledge, 11 - frame.clarity],
                     axis=1).mean(axis=1)


def load_model(directory: str, tag: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(str(Path(directory) / f"scores_{tag}_shard*.jsonl")):
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("parsed_ok"):
                rows.append(rec)
    frame = pd.DataFrame(rows)
    if frame.empty or "doc_id" not in frame.columns:
        # The per-scale output lives with the assessment run and may not be to
        # hand; the caller falls back to the index in that case.
        return pd.DataFrame(columns=["doc_id", *SCALES, *COUNTS])
    keep = {f"llm_{c}": c for c in SCALES + COUNTS if f"llm_{c}" in frame.columns}
    return frame[["doc_id", *keep]].rename(columns=keep).drop_duplicates("doc_id")


def main() -> None:
    args = parse_args()
    out = Path(args.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)

    human = load_human(args.coding)
    model = load_model(args.scores, args.tag)
    first = human[human["pass"] == 1].copy()
    first["composite"] = human_composite(first)

    rows: list[dict] = []

    # The ceiling: how well coders agree with one another.
    for left, right in itertools.combinations(sorted(first.coder.unique()), 2):
        pair = (first[first.coder == left].set_index("doc_id")
                .join(first[first.coder == right].set_index("doc_id"),
                      lsuffix="_l", rsuffix="_r", how="inner"))
        for measure in SCALES + COUNTS:
            got = agreement(pair[f"{measure}_l"], pair[f"{measure}_r"], measure,
                            f"{left} vs {right}")
            if got:
                got["kind"] = "between coders"
                rows.append(got)

    # The human composite against the index, which does not need the per-scale
    # model output and is the comparison closest to the reported results.
    key_path = Path(args.sample_key)
    if key_path.exists():
        key = pd.read_csv(key_path)[["doc_id", "model_index"]]
        pooled = first.groupby("doc_id").composite.mean().rename("panel").reset_index()
        joined = pooled.merge(key, on="doc_id")
        got = agreement(joined.panel, joined.model_index, "overall_judgement",
                        "coder panel vs index")
        if got:
            got["kind"] = "panel to index"
            rows.append(got)
        for coder, block in first.groupby("coder"):
            j = block[["doc_id", "composite"]].merge(key, on="doc_id")
            got = agreement(j.composite, j.model_index, "overall_judgement", f"{coder} vs index")
            if got:
                got["kind"] = "coder to index"
                rows.append(got)
        for left, right in itertools.combinations(sorted(first.coder.unique()), 2):
            a = first[first.coder == left][["doc_id", "composite"]]
            b = first[first.coder == right][["doc_id", "composite"]]
            j = a.merge(b, on="doc_id", suffixes=("_l", "_r"))
            got = agreement(j.composite_l, j.composite_r, "overall_judgement",
                            f"{left} vs {right}")
            if got:
                got["kind"] = "between coders"
                rows.append(got)

        # A fair benchmark. Correlating the index with the mean of three coders
        # is easier than correlating one coder with another, because averaging
        # removes noise, so the two numbers are not comparable. Leaving one coder
        # out gives both the index and that coder the same target with the same
        # amount of noise in it, and the two correlations can then be read
        # against each other.
        coders = sorted(first.coder.unique())
        if len(coders) >= 3:
            for held in coders:
                others = (first[first.coder != held]
                          .groupby("doc_id").composite.mean().rename("others").reset_index())
                mine = first[first.coder == held][["doc_id", "composite"]].merge(others, on="doc_id")
                got = agreement(mine.composite, mine.others, "overall_judgement",
                                f"{held} vs the other two")
                if got:
                    got["kind"] = "coder vs rest (ceiling)"
                    rows.append(got)
                versus = others.merge(key, on="doc_id")
                got = agreement(versus.model_index, versus.others, "overall_judgement",
                                f"index vs the two excluding {held}")
                if got:
                    got["kind"] = "index vs rest (same target)"
                    rows.append(got)
    else:
        LOG.warning("no sample key at %s; skipping the comparison against the index", key_path)

    # What the automated measurement achieves against that ceiling.
    if model.empty:
        LOG.warning("no per-scale model scores in %s for tag %r; the scale-by-scale "
                    "comparison with the model is omitted", args.scores, args.tag)
    for coder, block in first.groupby("coder"):
        if model.empty:
            break
        joined = block.set_index("doc_id").join(model.set_index("doc_id"),
                                                lsuffix="_h", rsuffix="_m", how="inner")
        for measure in SCALES + COUNTS:
            got = agreement(joined[f"{measure}_h"], joined[f"{measure}_m"], measure,
                            f"{coder} vs model")
            if got:
                got["kind"] = "coder to model"
                rows.append(got)

    # Each coder against their own repeated passages.
    repeats = human[human.doc_id.isin(human[human["pass"] == 2].doc_id)]
    n_repeat = human[human["pass"] == 2].doc_id.nunique()
    if 0 < n_repeat < 15:
        LOG.warning("only %d repeated documents: within-coder consistency is estimated "
                    "imprecisely, and a larger repeat share would fix that", n_repeat)
    for coder, block in repeats.groupby("coder"):
        wide = block.pivot_table(index="doc_id", columns="pass", values=SCALES + COUNTS)
        for measure in SCALES + COUNTS:
            if (measure, 1) not in wide.columns or (measure, 2) not in wide.columns:
                continue
            got = agreement(wide[measure][1], wide[measure][2], measure,
                            f"{coder} repeated", min_n=8)
            if got:
                got["kind"] = "within coder"
                rows.append(got)

    if not rows:
        LOG.error("no overlapping ratings to compare")
        return

    table = pd.DataFrame(rows)[
        ["kind", "comparison", "measure", "n", "pearson", "spearman",
         "quadratic_weighted_kappa", "within_one_point", "mean_absolute_difference", "mean_shift"]
    ]
    save_table(table, out / "human_validation")

    summary = (table.groupby(["kind", "measure"])
               .agg(n_comparisons=("n", "size"), mean_kappa=("quadratic_weighted_kappa", "mean"),
                    mean_spearman=("spearman", "mean"),
                    mean_within_one=("within_one_point", "mean"))
               .reset_index())
    save_table(summary, out / "human_validation_summary")

    print(summary.round(3).to_string(index=False))
    print()
    print("Read down each measure: if 'coder to model' is close to 'between coders',")
    print("the automated measurement is as close to a person as two people are to")
    print("each other, which is as well as the task allows.")


if __name__ == "__main__":
    main()
