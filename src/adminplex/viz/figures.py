"""Figures.

Everything is drawn in a single style: no colour is load-bearing, so the
figures survive greyscale printing; no chart carries more ink than the finding
needs. Each function returns the path it wrote.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..logging_config import get_logger  # noqa: E402

LOG = get_logger("adminplex.viz")

GREY = "#4d4d4d"
LIGHT = "#bdbdbd"
DARK = "#1a1a1a"

STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GREY,
    "axes.labelcolor": DARK,
    "text.color": DARK,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def _save(fig: plt.Figure, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pdf"
    fig.savefig(path)
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    LOG.info("figure written: %s", path.name)
    return path


def index_distribution(index: pd.Series, out: Path, name: str = "fig1_index_distribution") -> Path:
    """The distribution of the index, with the quartiles marked."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.hist(index.dropna(), bins=60, color=LIGHT, edgecolor="white", linewidth=0.3)
    for q in (0.25, 0.5, 0.75):
        value = index.quantile(q)
        ax.axvline(value, color=DARK, linewidth=0.9, linestyle=(0, (4, 2)))
        ax.text(value, ax.get_ylim()[1] * 0.96, f"Q{int(q * 4)}", ha="center", va="top",
                fontsize=7, color=DARK, backgroundcolor="white")
    ax.set_xlabel("Administrative Complexity Index")
    ax.set_ylabel("Documents")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    return _save(fig, out, name)


def dimension_correlations(dimensions: pd.DataFrame, out: Path,
                           name: str = "fig2_dimension_correlations",
                           annotate_above: float = 0.45) -> Path:
    """Correlation between dimensions: how far the index is measuring one thing.

    Only the larger coefficients are printed. A fifteen by fifteen matrix with
    every cell labelled is a wall of numbers in which the pattern is harder to
    see than in the shading alone; the values that matter for the argument are
    the high ones, and those are the ones written in.
    """
    corr = dimensions.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    im = ax.imshow(corr.values, cmap="Greys", vmin=0, vmax=1)
    labels = [c.replace("_", " ") for c in corr.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.values[i, j]
            if abs(value) < annotate_above and i != j:
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.55 else DARK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Spearman correlation", fontsize=9)
    cbar.outline.set_visible(False)
    ax.set_title(f"Between dimensions of the index (values shown above {annotate_above:.2f})")
    return _save(fig, out, name)


def pca_scree(explained: list[float], out: Path, name: str = "fig3_pca_scree") -> Path:
    """Variance explained by successive components."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = np.arange(1, len(explained) + 1)
    ax.bar(x, np.asarray(explained) * 100, color=LIGHT, edgecolor=GREY, linewidth=0.5)
    ax.plot(x, np.cumsum(explained) * 100, color=DARK, linewidth=1.1, marker="o", markersize=3,
            label="cumulative")
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_xticks(x)
    ax.legend(loc="center right")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    return _save(fig, out, name)


PRETTY = {
    "flesch": "Flesch Reading Ease",
    "flesch_kincaid": "Flesch–Kincaid grade",
    "smog": "SMOG",
    "gunning_fog": "Gunning Fog",
    "ari": "Automated Readability Index",
    "coleman_liau": "Coleman–Liau",
    "lix": "LIX",
    "rix": "RIX",
}


def index_vs_readability(frame: pd.DataFrame, index_col: str, readability_cols: list[str],
                         out: Path, name: str = "fig4_index_vs_readability") -> Path:
    """How far the index tracks the measures the field already uses.

    Axes are clipped to the first and ninety-ninth percentiles. A handful of
    documents without sentence punctuation produce grade levels in the
    hundreds, and on unclipped axes those few points compress every real
    document into a vertical line, hiding the relationship the figure exists to
    show. The correlations printed are computed on all the data, not the
    clipped view.
    """
    cols = [c for c in readability_cols if c in frame.columns][:6]
    if not cols:
        raise ValueError("no readability columns present")
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows), squeeze=False)
    sample = frame.sample(n=min(6000, len(frame)), random_state=20260726)
    for i, (ax, col) in enumerate(zip(axes.ravel(), cols)):
        lo, hi = frame[col].quantile([0.01, 0.99])
        ax.scatter(sample[col], sample[index_col], s=2.5, alpha=0.15, color=GREY, linewidths=0)
        ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        rho = frame[col].corr(frame[index_col], method="spearman")
        ax.set_xlabel(PRETTY.get(col, col.replace("_", " ")), fontsize=8)
        ax.set_ylabel("Administrative Complexity Index" if i % ncols == 0 else "", fontsize=8)
        ax.text(0.04, 0.95, f"ρ = {rho:.2f}", transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))
        ax.grid(True, linewidth=0.4)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    return _save(fig, out, name)


def group_violins(frame: pd.DataFrame, group: str, value: str, out: Path,
                  name: str | None = None, min_n: int = 30, top: int = 12) -> Path:
    """Distribution of the index within each group, ordered by median."""
    counts = frame[group].value_counts()
    keep = counts[counts >= min_n].index[:top]
    subset = frame[frame[group].isin(keep)]
    order = subset.groupby(group, observed=True)[value].median().sort_values()
    data = [subset.loc[subset[group] == g, value].dropna().to_numpy() for g in order.index]

    fig, ax = plt.subplots(figsize=(5.6, 0.34 * len(order) + 1.4))
    parts = ax.violinplot(data, vert=False, showextrema=False, widths=0.85)
    for body in parts["bodies"]:
        body.set_facecolor(LIGHT)
        body.set_edgecolor(GREY)
        body.set_linewidth(0.5)
        body.set_alpha(1.0)
    medians = [np.median(d) for d in data]
    ax.scatter(medians, np.arange(1, len(data) + 1), color=DARK, s=10, zorder=3)
    ax.set_yticks(range(1, len(order) + 1),
                  [f"{g}  (n={counts[g]:,})".replace(",", " ") for g in order.index])
    ax.set_xlabel("Administrative Complexity Index")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    return _save(fig, out, name or f"fig_violin_{group}")


def group_ranking(table: pd.DataFrame, label_col: str, out: Path,
                  raw_col: str = "raw_mean", adjusted_col: str = "adjusted_mean",
                  name: str | None = None) -> Path:
    """Group means before and after holding document length fixed.

    Showing both in one figure is the point: the gap between the two is the
    part of an apparent difference that is a difference in document size.
    """
    tab = table.sort_values(adjusted_col)
    y = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(5.4, 0.34 * len(tab) + 1.4))
    ax.hlines(y, tab[raw_col], tab[adjusted_col], color=LIGHT, linewidth=1.4, zorder=1)
    ax.scatter(tab[raw_col], y, s=22, facecolor="white", edgecolor=GREY, linewidth=0.9,
               zorder=2, label="raw mean")
    ax.scatter(tab[adjusted_col], y, s=22, color=DARK, zorder=3, label="holding length fixed")
    ax.set_yticks(y, [f"{v}  (n={int(n):,})".replace(",", " ")
                      for v, n in zip(tab[label_col], tab["n"])])
    ax.set_xlabel("Administrative Complexity Index")
    ax.legend(loc="lower right")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    return _save(fig, out, name or f"fig_ranking_{label_col}")


def ablation_plot(table: pd.DataFrame, out: Path, name: str = "fig_ablation") -> Path:
    """How far the ranking moves when each dimension is removed."""
    tab = table.sort_values("spearman_with_full")
    y = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(5.0, 0.32 * len(tab) + 1.2))
    ax.barh(y, tab["spearman_with_full"], color=LIGHT, edgecolor=GREY, linewidth=0.5)
    ax.set_yticks(y, [d.replace("_", " ") for d in tab["dimension_removed"]])
    ax.set_xlim(min(0.85, tab["spearman_with_full"].min() - 0.02), 1.0)
    ax.set_xlabel("Spearman correlation with the full index")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    return _save(fig, out, name)


def trend_over_time(by_country: pd.DataFrame, out: Path, name: str = "fig_trend") -> Path:
    """The index year by year, with the volume published behind it.

    Both series are drawn because the headline depends on which one is read:
    the index per document is flat or falling while the words published per
    document rise, and a figure showing only the first would invite the reader
    to conclude that nothing is happening.
    """
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(5.6, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (country, block) in enumerate(by_country.groupby("country", observed=True)):
        block = block.sort_values("year")
        if len(block) < 3:
            continue
        shade = 0.15 + 0.6 * (i / max(1, by_country.country.nunique() - 1))
        top.errorbar(block.year, block.mean_aci, yerr=1.96 * block.se, label=str(country),
                     color=str(shade), marker=markers[i % len(markers)], markersize=3.5,
                     linewidth=1.1, capsize=2, elinewidth=0.6)
        bottom.plot(block.year, block.median_words, color=str(shade),
                    marker=markers[i % len(markers)], markersize=3.5, linewidth=1.1)

    top.set_ylabel("Administrative Complexity Index")
    top.legend(ncol=3, loc="upper left")
    top.grid(axis="y")
    top.set_axisbelow(True)
    bottom.set_ylabel("Median words")
    bottom.set_xlabel("Year of publication")
    bottom.grid(axis="y")
    bottom.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, out, name)


def organisation_ranking(table: pd.DataFrame, out: Path, top: int = 15,
                         name: str = "fig_organisations") -> Path:
    """The most and least demanding bodies, raw and holding length fixed.

    The distance between the two points is the finding for any given body: a
    short document is not the same as an undemanding one.
    """
    head = table.nlargest(top, "adjusted_mean")
    tail = table.nsmallest(top // 2, "adjusted_mean")
    tab = pd.concat([head, tail]).sort_values("adjusted_mean")
    y = np.arange(len(tab))

    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(tab) + 1.8))
    ax.hlines(y, tab.raw_mean, tab.adjusted_mean, color=LIGHT, linewidth=1.3, zorder=1)
    ax.scatter(tab.raw_mean, y, s=20, facecolor="white", edgecolor=GREY, linewidth=0.9,
               zorder=2, label="raw mean")
    ax.scatter(tab.adjusted_mean, y, s=20, color=DARK, zorder=3, label="holding length and type fixed")

    def shorten(text: str, limit: int = 38) -> str:
        """Trim on a word boundary; a label cut mid-word reads as a defect."""
        if len(text) <= limit:
            return text
        cut = text[:limit].rsplit(" ", 1)[0]
        return (cut or text[:limit]) + "…"

    ax.set_yticks(y, [f"{shorten(o)}  ({c}, n={int(n)})"
                      for o, c, n in zip(tab.organisation, tab.country, tab.n)],
                  fontsize=9)
    ax.axhline(len(tail) - 0.5, color=LIGHT, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xlabel("Administrative Complexity Index")
    # Above the plot, so it cannot sit on top of the lowest-ranked bodies.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, out, name)


def cluster_profiles(profiles: pd.DataFrame, summary: pd.DataFrame, out: Path,
                     name: str = "fig_cluster_profiles") -> Path:
    """Each type of complexity as a profile across the dimensions.

    Drawn as departures from the corpus mean rather than as levels: what
    distinguishes the types is where they are unusual, and a plot of levels
    would mostly show that every type is near the middle on most dimensions.
    """
    dims = [c for c in profiles.columns if c != "cluster"]
    matrix = profiles.set_index("cluster")[dims]
    deviation = matrix - 50.0
    order = summary.sort_values("mean_index", ascending=False)["cluster"].tolist() \
        if "mean_index" in summary.columns else list(matrix.index)
    deviation = deviation.loc[order]

    labels = []
    for c in order:
        row = summary[summary.cluster == c]
        share = float(row["share"].iloc[0]) if len(row) else float("nan")
        index = float(row["mean_index"].iloc[0]) if "mean_index" in summary.columns and len(row) else float("nan")
        labels.append(f"type {c}  ({share:.0%}, ACI {index:.0f})")

    # A diverging palette would put equal-and-opposite departures at the same
    # lightness, so the pattern would vanish the moment the page is printed in
    # black and white. A single ramp running from below-average to above-average
    # is monotone in lightness and therefore says the same thing in either.
    limit = float(np.abs(deviation.values).max())
    fig, ax = plt.subplots(figsize=(0.52 * len(dims) + 2.6, 0.5 * len(order) + 1.8))
    im = ax.imshow(deviation.values, cmap="Greys", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(dims)), [d.replace("_", " ") for d in dims], rotation=45, ha="right")
    ax.set_yticks(range(len(order)), labels)
    for i in range(len(order)):
        for j in range(len(dims)):
            value = deviation.values[i, j]
            ax.text(j, i, f"{value:+.0f}", ha="center", va="center", fontsize=6,
                    color="white" if value > limit * 0.5 else DARK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Difference from corpus mean", fontsize=8)
    cbar.outline.set_visible(False)
    return _save(fig, out, name)


SPEC_LABELS = {
    "dimensions": "The fifteen dimensions,\nkept separate",
    "aci_and_readability": "Index + readability indices",
    "aci_only": "Administrative Complexity\nIndex alone",
    "readability_and_length": "Readability indices + length",
    "length_only": "Document length alone",
    "readability": "Eight readability indices\n(the measures in current use)",
}


def burden_fit(comparison: pd.DataFrame, out: Path, name: str = "fig_burden_fit",
               reference: str = "readability") -> Path:
    """Out-of-sample fit of each specification against the burden criterion.

    The comparison is the central result, so the figure states it rather
    than leaving the reader to divide the bars: each is annotated with how many
    times better it does than the measures administrations currently use, and
    the specifications built on the new measure are separated from those built
    on the old ones.
    """
    tab = comparison.sort_values("cv_r2").copy()
    base = float(tab.loc[tab.specification == reference, "cv_r2"].iloc[0]) \
        if reference in set(tab.specification) else float("nan")
    y = np.arange(len(tab))
    new_measure = tab.specification.isin(["dimensions", "aci_only", "aci_and_readability"])
    colours = [DARK if flag else LIGHT for flag in new_measure]

    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(tab) + 1.2))
    ax.barh(y, tab["cv_r2"], color=colours, edgecolor=GREY, linewidth=0.5, height=0.55)
    spread = tab["cv_r2_sd"] if "cv_r2_sd" in tab.columns else pd.Series(0.0, index=tab.index)
    if "cv_r2_sd" in tab.columns:
        ax.errorbar(tab["cv_r2"], y, xerr=spread, fmt="none", ecolor=GREY,
                    elinewidth=0.8, capsize=2)

    span = float(tab["cv_r2"].max())
    for yi, (value, sd, spec) in enumerate(zip(tab["cv_r2"], spread, tab["specification"])):
        note = f"{value:.3f}"
        if np.isfinite(base) and base > 0 and spec != reference:
            note += f"   ({value / base:.0f}× the measures in use)"
        # Clear of the error bar, not just of the bar, or the two collide.
        ax.text(value + float(sd) + 0.02 * span, yi, note, va="center", fontsize=7.5, color=DARK)

    ax.set_yticks(y, [SPEC_LABELS.get(s, s.replace("_", " ")) for s in tab["specification"]])
    ax.set_xlabel("Cross-validated $R^2$ against burden hours approved by the OMB")
    ax.set_xlim(0, span * 1.42)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, out, name)
