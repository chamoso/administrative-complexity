"""Link the index to an externally produced measure of administrative burden.

Under the Paperwork Reduction Act each federal information collection carries
an OMB-approved estimate of the hours the public spends complying with it. The
collection also names the parts of the Code of Federal Regulations that impose
it. Joining the two gives, for each regulatory text in the corpus, a burden
estimate that was produced by a different institution for a different purpose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CONFIG, PROCESSED, RAW
from ..io import save_table
from ..logging_config import get_logger

LOG = get_logger("adminplex.analysis.burden")


def _explode_citations(pra: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in pra.itertuples():
        if not row.cfr_parts:
            continue
        parts = {p for p in str(row.cfr_parts).split(";") if ":" in p}
        for token in parts:
            title, part = token.split(":", 1)
            rows.append(
                {
                    "cfr_title": title.strip(),
                    "cfr_part": part.strip().split(".")[0].strip(),
                    "icr_ref": row.icr_ref,
                    "omb_control_number": row.omb_control_number,
                    "agency_code": row.agency_code,
                    "department": row.department,
                    "annual_responses": row.annual_responses,
                    "annual_burden_hours": row.annual_burden_hours,
                    "annual_burden_cost": row.annual_burden_cost,
                    "hours_per_response": row.hours_per_response,
                    "affected_public": row.affected_public,
                    "obligation": row.obligation,
                    "n_instruments": row.n_instruments,
                    "n_cfr_parts": row.n_cfr_parts,
                }
            )
    return pd.DataFrame(rows)


def build_validation_frame(corpus: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per CFR part: index inputs on the left, burden on the right."""
    cfg = CONFIG["burden"]
    pra = pd.read_parquet(RAW / "pra_collections.parquet")
    pra = pra[pra.icr_status == "Active"].copy()
    pra = pra[pra.annual_responses.fillna(0) >= cfg["min_annual_responses"]]
    pra = pra[pra.hours_per_response.notna() & np.isfinite(pra.hours_per_response)]
    pra = pra[pra.hours_per_response > 0]

    links = _explode_citations(pra)
    LOG.info("%d collection-to-part links from %d collections", len(links), pra.icr_ref.nunique())

    # Each collection spreads its burden over the parts it cites, so a part is
    # not credited with burden that another rule is equally responsible for.
    links["weight"] = 1.0 / links["n_cfr_parts"].clip(lower=1)

    grouped = links.groupby(["cfr_title", "cfr_part"])
    part_level = grouped.apply(
        lambda g: pd.Series(
            {
                "n_collections": g.icr_ref.nunique(),
                "total_annual_responses": g.annual_responses.sum(),
                "total_annual_burden_hours": g.annual_burden_hours.sum(),
                "weighted_burden_hours": (g.annual_burden_hours * g.weight).sum(),
                "median_hours_per_response": g.hours_per_response.median(),
                "mean_hours_per_response": g.hours_per_response.mean(),
                "burden_weighted_hours_per_response": (
                    (g.annual_burden_hours.sum() / g.annual_responses.sum())
                    if g.annual_responses.sum() > 0
                    else np.nan
                ),
                "n_agencies": g.agency_code.nunique(),
                "department": g.department.mode().iat[0] if len(g.department.mode()) else None,
                "share_mandatory": (g.obligation.astype(str).str.contains("Mandatory")).mean(),
                "share_business": (g.affected_public.astype(str).str.contains("Private Sector|Business")).mean(),
                "mean_instruments": g.n_instruments.mean(),
            }
        ),
        include_groups=False,
    ).reset_index()

    if corpus is None:
        corpus = pd.read_parquet(PROCESSED / "corpus.parquet")
    texts = corpus[corpus.source == "eCFR"].copy()
    texts["cfr_title"] = texts["cfr_title"].astype(str)
    texts["cfr_part"] = texts["cfr_part"].astype(str)

    merged = texts.merge(part_level, on=["cfr_title", "cfr_part"], how="inner")
    LOG.info("validation frame: %d CFR parts matched to burden estimates", len(merged))

    # Winsorise the criterion: a handful of collections report implausibly
    # large per-response estimates that would otherwise drive the regressions.
    pct = cfg["winsorise_pct"]
    for col in ("median_hours_per_response", "mean_hours_per_response", "burden_weighted_hours_per_response"):
        lo, hi = merged[col].quantile([pct, 1 - pct])
        merged[f"{col}_w"] = merged[col].clip(lo, hi)
        merged[f"log_{col}"] = np.log(merged[f"{col}_w"].clip(lower=1e-3))
    merged["log_total_burden_hours"] = np.log1p(merged["total_annual_burden_hours"].clip(lower=0))

    save_table(merged.drop(columns=["text"], errors="ignore"), PROCESSED / "validation_parts")
    return merged


def build_collection_frame() -> pd.DataFrame:
    """Collection-level frame, using each agency's own description as the text."""
    pra = pd.read_parquet(RAW / "pra_collections.parquet")
    pra = pra[(pra.icr_status == "Active") & pra.abstract.str.len().gt(200)].copy()
    pra = pra[pra.hours_per_response.notna() & (pra.hours_per_response > 0)]
    pra["doc_id"] = "USICR:" + pra["icr_ref"].astype(str)
    pra["text"] = pra["title"].fillna("") + ". " + pra["abstract"].fillna("")
    pra["language"] = "en"
    pra["country"] = "US"
    lo, hi = pra.hours_per_response.quantile([0.01, 0.99])
    pra["hours_per_response_w"] = pra.hours_per_response.clip(lo, hi)
    pra["log_hours_per_response"] = np.log(pra.hours_per_response_w.clip(lower=1e-3))
    LOG.info("collection-level frame: %d active collections with usable abstracts", len(pra))
    save_table(pra.drop(columns=["text"], errors="ignore"), PROCESSED / "validation_collections")
    return pra
