"""Mapping from measured features to the fifteen dimensions of the index.

Each dimension is defined once, here, as a named combination of measured
features and a direction. Keeping the definition in one place is what makes the
index auditable: nothing about a dimension is decided inside a model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    features: tuple[str, ...]
    source: str  # "rule" or "assessed"
    note: str


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("length", "Length", ("log_words", "log_sentences", "log_paragraphs"), "rule",
              "Volume of text the reader must process."),
    Dimension("syntactic_complexity", "Syntactic complexity",
              ("mean_sentence_length", "p90_sentence_length", "subordination_rate", "clauses_per_sentence"), "rule",
              "Sentence length and degree of subordination."),
    Dimension("legal_density", "Legal density", ("legal_citation_rate", "distinct_legal_sources"), "rule",
              "Density and variety of citations to binding instruments."),
    Dimension("cross_references", "Cross-references", ("cross_reference_rate",), "rule",
              "Pointers that send the reader to another text."),
    Dimension("exceptions", "Exceptions", ("exception_rate",), "rule",
              "Carve-outs, provisos and disapplications."),
    Dimension("requirements", "Requirements", ("requirement_rate", "n_requirement_sentences", "llm_n_requirements"), "rule",
              "Obligations the reader must satisfy."),
    Dimension("steps", "Steps", ("n_steps",), "rule",
              "Distinct actions in the procedure."),
    Dimension("dependencies", "Dependencies", ("dag_depth", "dag_width", "dag_density", "precedence_rate"), "rule",
              "Ordering constraints between steps."),
    Dimension("ambiguity", "Ambiguity", ("ambiguity_rate",), "rule",
              "Discretionary and vague formulations."),
    Dimension("cognitive_load", "Cognitive load", ("llm_cognitive_load",), "assessed",
              "Mental effort required to establish what must be done."),
    Dimension("legal_knowledge", "Required legal knowledge", ("llm_legal_knowledge",), "assessed",
              "Prior legal or technical knowledge presupposed."),
    Dimension("documentary_effort", "Documentary effort",
              ("n_document_types", "document_mention_rate", "llm_n_documents"), "rule",
              "Number and variety of documents to be produced."),
    Dimension("conditional_decisions", "Conditional decisions", ("conditional_rate",), "rule",
              "Branching conditions the reader must evaluate."),
    Dimension("redundancy", "Redundancy", ("redundancy_ratio",), "rule",
              "Repetition of the same content within the document."),
    Dimension("clarity", "Lack of clarity", ("llm_clarity_reversed",), "assessed",
              "Clarity of the text, reverse-coded so that high means complex."),
)

DIMENSION_KEYS = tuple(d.key for d in DIMENSIONS)
ASSESSED_KEYS = tuple(d.key for d in DIMENSIONS if d.source == "assessed")

# Weights for the theory-led benchmark index. They follow the standard
# tripartite account of administrative burden: costs of learning what is
# required, costs of complying, and costs arising from uncertainty.
BURDEN_FAMILY = {
    "learning": ("length", "syntactic_complexity", "legal_density", "cross_references",
                 "legal_knowledge", "clarity"),
    "compliance": ("requirements", "steps", "dependencies", "documentary_effort"),
    "uncertainty": ("exceptions", "ambiguity", "conditional_decisions", "redundancy", "cognitive_load"),
}
EXPERT_FAMILY_WEIGHTS = {"learning": 1 / 3, "compliance": 1 / 3, "uncertainty": 1 / 3}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the transformed features the dimensions are built from."""
    out = df.copy()
    for src, dst in (("n_words", "log_words"), ("n_sentences", "log_sentences"), ("n_paragraphs", "log_paragraphs")):
        out[dst] = np.log1p(out[src].astype(float))
    if "llm_clarity" in out.columns:
        out["llm_clarity_reversed"] = 11.0 - out["llm_clarity"].astype(float)
    return out


def percentile_rank(series: pd.Series) -> pd.Series:
    """Rank-transform to 0-100 within the corpus.

    A rank transform is used rather than a z-score because most of the counts
    are strongly right-skewed and because ranks are invariant to the scale of a
    measure, which keeps the dimensions comparable across languages.
    """
    valid = series.notna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if valid.sum() == 0:
        return out
    out.loc[valid] = series[valid].rank(pct=True, method="average") * 100.0
    return out


def percentile_rank_against(series: pd.Series, reference: pd.Series) -> pd.Series:
    """Position each value in the distribution of a reference corpus.

    Needed whenever documents are scored that were not part of the corpus the
    scale was built on — an edited version of a procedure, or a text from a
    validation sample. Ranking such documents among themselves would put the
    least complex of them at zero and the most complex at one hundred whatever
    their absolute level, and any before-and-after comparison would be empty.
    """
    ref = np.sort(reference.dropna().astype(float).to_numpy())
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if ref.size == 0:
        return out
    valid = series.notna()
    if valid.sum() == 0:
        return out
    values = series[valid].astype(float).to_numpy()
    # Midpoint of the tied range, matching the average ranking used in-corpus.
    left = np.searchsorted(ref, values, side="left")
    right = np.searchsorted(ref, values, side="right")
    out.loc[valid] = (left + right) / 2.0 / ref.size * 100.0
    return out


def build_dimension_scores(
    df: pd.DataFrame,
    scaling: str = "percentile_rank",
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One column per dimension, on a common 0-100 scale.

    With a reference frame, values are placed in that corpus's distribution
    instead of being ranked within df.
    """
    feats = prepare_features(df)
    ref_feats = prepare_features(reference) if reference is not None else None
    scores = pd.DataFrame(index=feats.index)
    for dim in DIMENSIONS:
        present = [f for f in dim.features if f in feats.columns]
        if not present:
            scores[dim.key] = np.nan
            continue
        if scaling == "percentile_rank":
            if ref_feats is not None:
                parts = [
                    percentile_rank_against(feats[f].astype(float), ref_feats[f].astype(float))
                    for f in present
                    if f in ref_feats.columns
                ] or [percentile_rank(feats[f].astype(float)) for f in present]
            else:
                parts = [percentile_rank(feats[f].astype(float)) for f in present]
        else:
            parts = []
            for f in present:
                col = feats[f].astype(float)
                sd = col.std(ddof=0)
                parts.append((col - col.mean()) / sd if sd > 0 else col * 0.0)
        scores[dim.key] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    scores.insert(0, "doc_id", feats["doc_id"].values)
    return scores
