"""Construction of the Administrative Complexity Index.

Five candidate aggregations are estimated on the same dimension scores and
compared on reliability, structure and external validity. The chosen index is
the one that is defensible on all three, not the one with the best single
statistic; the comparison table is written to outputs/tables.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..config import CONFIG, SEED
from ..logging_config import get_logger
from .dimensions import BURDEN_FAMILY, DIMENSION_KEYS, EXPERT_FAMILY_WEIGHTS

LOG = get_logger("adminplex.index.build")
warnings.filterwarnings("ignore", category=UserWarning)


def _standardise(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame - frame.mean()) / frame.std(ddof=0).replace(0, np.nan)


def cronbach_alpha(frame: pd.DataFrame) -> float:
    items = frame.dropna()
    k = items.shape[1]
    if k < 2 or len(items) < 3:
        return float("nan")
    item_var = items.var(axis=0, ddof=1).sum()
    total_var = items.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var / total_var)) if total_var > 0 else float("nan")


def mcdonald_omega(frame: pd.DataFrame) -> float:
    """Omega from a single-factor solution: reliability without tau-equivalence."""
    items = _standardise(frame.dropna())
    if items.shape[1] < 3 or len(items) < 10:
        return float("nan")
    pca = PCA(n_components=1, random_state=SEED).fit(items.values)
    loadings = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
    common = loadings.sum() ** 2
    unique = float(np.sum(1 - loadings**2))
    return float(common / (common + unique)) if common + unique > 0 else float("nan")


def _rescale_0_100(values: np.ndarray | pd.Series) -> pd.Series:
    series = pd.Series(np.asarray(values, dtype=float))
    lo, hi = series.min(), series.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.full(len(series), 50.0))
    return (series - lo) / (hi - lo) * 100.0


@dataclass
class IndexResult:
    name: str
    scores: pd.Series
    weights: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def equal_weight(scores: pd.DataFrame) -> IndexResult:
    keys = list(scores.columns)
    values = scores.mean(axis=1, skipna=True)
    return IndexResult("equal_weight", _rescale_0_100(values), {k: 1 / len(keys) for k in keys},
                       {"rationale": "unweighted mean of dimension percentiles"})


def expert_weight(scores: pd.DataFrame) -> IndexResult:
    """Theory-led weights: three equally weighted families of burden costs."""
    weights: dict[str, float] = {}
    for family, members in BURDEN_FAMILY.items():
        present = [m for m in members if m in scores.columns]
        if not present:
            continue
        share = EXPERT_FAMILY_WEIGHTS[family] / len(present)
        for m in present:
            weights[m] = weights.get(m, 0.0) + share
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    values = sum(scores[k].fillna(scores[k].mean()) * w for k, w in weights.items())
    return IndexResult("expert_weight", _rescale_0_100(values), weights,
                       {"rationale": "equal weight to learning, compliance and uncertainty costs"})


def pca_index(scores: pd.DataFrame) -> IndexResult:
    filled = scores.fillna(scores.mean())
    std = _standardise(filled).fillna(0.0)
    pca = PCA(n_components=min(5, std.shape[1]), random_state=SEED).fit(std.values)
    comp = pca.components_[0]
    if comp.sum() < 0:  # orient so that higher means more complex
        comp = -comp
    values = std.values @ comp
    weights = dict(zip(scores.columns, comp / np.abs(comp).sum()))
    return IndexResult("pca", _rescale_0_100(values), weights, {
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "first_component_share": float(pca.explained_variance_ratio_[0]),
        "eigenvalues": [float(v) for v in pca.explained_variance_],
    })


def efa_index(scores: pd.DataFrame, n_factors: int | None = None) -> IndexResult:
    """Oblique factor solution, aggregated by explained variance of each factor."""
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

    n_factors = n_factors or CONFIG["index"]["n_factors"]
    filled = scores.fillna(scores.mean())
    std = _standardise(filled).fillna(0.0)

    chi2, p_value = calculate_bartlett_sphericity(std.values)
    _, kmo_model = calculate_kmo(std.values)

    fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="ml")
    fa.fit(std.values)
    loadings = pd.DataFrame(fa.loadings_, index=scores.columns,
                            columns=[f"F{i+1}" for i in range(n_factors)])
    factor_scores = fa.transform(std.values)
    variance = fa.get_factor_variance()[1]  # proportion of variance per factor
    share = variance / variance.sum()
    values = factor_scores @ share
    # A dimension's contribution is its loading on each factor, weighted by the
    # factor's share of common variance.
    contribution = (loadings.abs() * share).sum(axis=1)
    weights = (contribution / contribution.sum()).to_dict()
    return IndexResult("efa", _rescale_0_100(values), weights, {
        "n_factors": n_factors,
        "kmo": float(kmo_model),
        "bartlett_chi2": float(chi2),
        "bartlett_p": float(p_value),
        "variance_explained": [float(v) for v in variance],
        "cumulative_variance": float(variance.sum()),
        "loadings": loadings.round(3).to_dict(),
        "communalities": dict(zip(scores.columns, [float(c) for c in fa.get_communalities()])),
    })


def irt_index(scores: pd.DataFrame, n_categories: int = 5, max_n: int = 6000) -> IndexResult:
    """Graded response model on quintile-discretised dimensions.

    Treating each dimension as an ordered item makes the measurement model
    explicit: a document's position on the latent trait is estimated from how
    'difficult' each dimension is to reach.
    """
    from girth import grm_mml

    rng = np.random.default_rng(SEED)
    filled = scores.fillna(scores.mean())
    idx = filled.index
    if len(filled) > max_n:
        sample_idx = rng.choice(len(filled), size=max_n, replace=False)
    else:
        sample_idx = np.arange(len(filled))

    ordinal = filled.apply(
        lambda col: pd.qcut(col.rank(method="first"), n_categories, labels=False, duplicates="drop") + 1
    )
    matrix = ordinal.values[sample_idx].T.astype(int)  # items x respondents
    estimates = grm_mml(matrix)
    discrimination = np.asarray(estimates["Discrimination"], dtype=float)
    difficulty = np.asarray(estimates["Difficulty"], dtype=float)

    # Score the whole corpus with the estimated discriminations.
    disc = np.where(np.isfinite(discrimination) & (discrimination > 0), discrimination, np.nanmean(discrimination))
    std = _standardise(filled).fillna(0.0)
    values = std.values @ disc
    weights = dict(zip(scores.columns, disc / disc.sum()))
    return IndexResult("irt_graded", _rescale_0_100(values), weights, {
        "n_categories": n_categories,
        "n_estimation_sample": int(len(sample_idx)),
        "discrimination": dict(zip(scores.columns, [float(d) for d in discrimination])),
        "mean_difficulty": dict(zip(scores.columns, [float(np.nanmean(d)) for d in difficulty])),
    })


def sem_index(scores: pd.DataFrame, max_n: int = 8000) -> IndexResult:
    """Second-order confirmatory model with the three theorised cost families."""
    import semopy

    rng = np.random.default_rng(SEED)
    filled = scores.fillna(scores.mean())
    sample = filled.sample(n=min(max_n, len(filled)), random_state=SEED)
    std = _standardise(sample).fillna(0.0)

    blocks = []
    for family, members in BURDEN_FAMILY.items():
        present = [m for m in members if m in std.columns]
        if len(present) >= 2:
            blocks.append(f"{family} =~ " + " + ".join(present))
    blocks.append("complexity =~ " + " + ".join(f for f in BURDEN_FAMILY if any(m in std.columns for m in BURDEN_FAMILY[f])))
    desc = "\n".join(blocks)

    model = semopy.Model(desc)
    model.fit(std)
    stats = semopy.calc_stats(model).to_dict("records")[0]
    est = model.inspect()
    loadings = {
        row["rval"]: float(row["Estimate"])
        for _, row in est.iterrows()
        if row["op"] == "~" and row["lval"] in BURDEN_FAMILY
    }
    weights_raw = {k: abs(v) for k, v in loadings.items() if k in std.columns}
    if not weights_raw:
        weights_raw = {k: 1.0 for k in std.columns}
    total = sum(weights_raw.values())
    weights = {k: v / total for k, v in weights_raw.items()}
    values = sum(_standardise(filled).fillna(0.0)[k] * w for k, w in weights.items())
    return IndexResult("sem", _rescale_0_100(values), weights, {
        "fit": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in stats.items()},
        "n_estimation_sample": int(len(sample)),
        "specification": desc,
    })


BUILDERS = {
    "equal_weight": equal_weight,
    "expert_weight": expert_weight,
    "pca": pca_index,
    "efa": efa_index,
    "irt_graded": irt_index,
    "sem": sem_index,
}


def build_all(scores: pd.DataFrame, candidates: list[str] | None = None) -> dict[str, IndexResult]:
    dims = [c for c in DIMENSION_KEYS if c in scores.columns and scores[c].notna().any()]
    frame = scores[dims]
    LOG.info("building candidate indices on %d dimensions, %d documents", len(dims), len(frame))
    out: dict[str, IndexResult] = {}
    for name in candidates or list(BUILDERS):
        try:
            result = BUILDERS[name](frame)
            result.diagnostics["alpha"] = cronbach_alpha(frame)
            result.diagnostics["omega"] = mcdonald_omega(frame)
            out[name] = result
            LOG.info("  %-12s built (alpha=%.3f)", name, result.diagnostics["alpha"])
        except Exception as exc:  # a failed candidate must not stop the comparison
            LOG.warning("  %-12s failed: %s: %s", name, type(exc).__name__, exc)
    return out
