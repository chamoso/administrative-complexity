"""Is the simplification result an artefact of the editor and scorer sharing a family?

The documents in experiments 5 and 6 were edited by the same open-weights model
that scores them. An edit could therefore be rewarded for producing text this
particular scorer finds simple rather than text a reader finds simple.

Rescoring the edited documents with a model from a different family tests that.
The comparison is made between the two interventions rather than before and
after, because a different model rates everything about a point higher and that
shift would otherwise be mistaken for a smaller effect. The contrast between
rewriting and removal is immune to a constant shift, so if the rewrite advantage
comes from shared provenance it should shrink under the outside scorer.

    python check_circularity.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.config import SEED  # noqa: E402
from adminplex.io import save_table  # noqa: E402
from adminplex.logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.circularity")
SCALES = ["llm_cognitive_load", "llm_legal_knowledge", "llm_clarity"]
SEP = "::"


def load(tag: str, directory: str = "outputs/llm_scores") -> pd.DataFrame:
    rows = []
    for path in glob.glob(str(Path(directory) / f"scores_{tag}_shard*.jsonl")):
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("parsed_ok"):
                rows.append(rec)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    keep = ["doc_id"] + [c for c in SCALES if c in frame.columns]
    return frame[keep].drop_duplicates(subset="doc_id")


def split_ids(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["intervention"] = out.doc_id.str.split(SEP).str[-1]
    out["original"] = out.doc_id.str.split(SEP).str[0]
    return out[out.intervention.isin(["removal", "rewrite"])]


def main() -> None:
    inside = split_ids(load("simplified", "outputs/llm_scores_simplified"))
    outside = split_ids(load("simplified_large"))
    if inside.empty or outside.empty:
        LOG.error("scores missing: same-family %d rows, outside-family %d rows",
                  len(inside), len(outside))
        return
    LOG.info("same family: %d edited texts; different family: %d", len(inside), len(outside))

    rng = np.random.default_rng(SEED)
    rows = []
    for label, frame in (("same family (14B)", inside), ("different family (70B)", outside)):
        wide = frame.pivot_table(index="original", columns="intervention", values=SCALES)
        for scale in SCALES:
            if (scale, "removal") not in wide.columns or (scale, "rewrite") not in wide.columns:
                continue
            pair = wide[scale][["removal", "rewrite"]].dropna()
            if len(pair) < 50:
                continue
            diff = (pair.rewrite - pair.removal).to_numpy()
            boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            rows.append({
                "scorer": label,
                "measure": scale.replace("llm_", ""),
                "n_documents": int(len(pair)),
                "mean_removal": float(pair.removal.mean()),
                "mean_rewrite": float(pair.rewrite.mean()),
                "rewrite_minus_removal": float(diff.mean()),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
                "wilcoxon_p": float(stats.wilcoxon(pair.rewrite, pair.removal).pvalue),
            })

    table = pd.DataFrame(rows)
    save_table(table, Path("outputs/tables/review_circularity"))
    print(table.round(3).to_string(index=False))

    print()
    print("A rewrite advantage that survives an outside scorer is not an artefact of the")
    print("editor and the scorer belonging to the same family. Negative values mean the")
    print("rewritten text is rated less demanding than the requirement-stripped text.")


if __name__ == "__main__":
    main()
