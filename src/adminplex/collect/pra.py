"""US collector: OMB/OIRA inventory of federal information collections.

Under the Paperwork Reduction Act every federal information collection carries
an OMB-approved estimate of the hours the public spends complying with it.
Those estimates are produced by the agencies themselves, independently of this
study, and are therefore used as the external criterion against which the index
is validated.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import CONFIG, RAW
from ..io import get_with_retry, save_table, session
from ..logging_config import get_logger

LOG = get_logger("adminplex.collect.pra")
XML_PATH = RAW / "pra_inventory.xml"
OUT = RAW / "pra_collections"

# The four-digit AgencyCode identifies the (sub-)agency and is used as the unit
# for agency fixed effects. For the coarser departmental grouping reported in
# the tables we map the registrable domain of the agency contact address
# recorded in the inventory, which is unambiguous and machine-verifiable.
# Addresses outside this table (11% of records) fall to the residual category.
_DOMAIN_TO_DEPARTMENT = {
    **{d: "Agriculture" for d in ("usda.gov", "fs.fed.us")},
    **{
        d: "Health and Human Services"
        for d in ("hhs.gov", "cdc.gov", "hrsa.gov", "nih.gov", "samhsa.gov", "ahrq.gov", "ihs.gov", "fda.gov")
    },
    **{d: "Labor" for d in ("dol.gov", "bls.gov", "osha.gov", "msha.gov")},
    **{d: "Homeland Security" for d in ("dhs.gov", "uscg.mil", "tsa.gov", "fema.gov", "cbp.gov")},
    "va.gov": "Veterans Affairs",
    **{
        d: "Defense"
        for d in ("mail.mil", "health.mil", "army.mil", "navy.mil", "af.mil", "osd.mil", "dodea.edu")
    },
    "ssa.gov": "Social Security Administration",
    **{
        d: "Treasury"
        for d in ("treas.gov", "treasury.gov", "irs.gov", "ttb.gov", "fincen.gov", "occ.gov")
    },
    **{
        d: "Transportation"
        for d in ("dot.gov", "faa.gov", "nhtsa.gov", "fra.gov", "fmcsa.gov", "phmsa.gov", "marad.gov")
    },
    **{d: "Environmental Protection Agency" for d in ("epa.gov", "epamail.epa.gov")},
    "fcc.gov": "Federal Communications Commission",
    "hud.gov": "Housing and Urban Development",
    **{d: "Federal Reserve System" for d in ("frb.gov", "federalreserve.gov")},
    "sec.gov": "Securities and Exchange Commission",
    **{
        d: "Justice"
        for d in ("usdoj.gov", "atf.gov", "dea.gov", "fbi.gov", "bop.gov", "justice.gov")
    },
    **{
        d: "Interior"
        for d in ("doi.gov", "nps.gov", "usgs.gov", "bia.gov", "osmre.gov", "blm.gov", "fws.gov", "boem.gov", "bsee.gov", "usbr.gov")
    },
    "fdic.gov": "Federal Deposit Insurance Corporation",
    "ftc.gov": "Federal Trade Commission",
    "gsa.gov": "General Services Administration",
    **{
        d: "Commerce"
        for d in ("doc.gov", "noaa.gov", "uspto.gov", "census.gov", "nist.gov", "ntia.gov", "trade.gov")
    },
    "rrb.gov": "Railroad Retirement Board",
    **{d: "Energy" for d in ("doe.gov", "ferc.gov")},
    "ed.gov": "Education",
    "sba.gov": "Small Business Administration",
    "cpsc.gov": "Consumer Product Safety Commission",
    "nrc.gov": "Nuclear Regulatory Commission",
    "opm.gov": "Office of Personnel Management",
    "nasa.gov": "NASA",
    "exim.gov": "Export-Import Bank",
    "pbgc.gov": "Pension Benefit Guaranty Corporation",
    "state.gov": "State",
    "nsf.gov": "National Science Foundation",
    "cftc.gov": "Commodity Futures Trading Commission",
    "ncua.gov": "National Credit Union Administration",
    "usaid.gov": "USAID",
    "consumerfinance.gov": "Consumer Financial Protection Bureau",
    "fhfa.gov": "Federal Housing Finance Agency",
    "nlrb.gov": "National Labor Relations Board",
    "eeoc.gov": "Equal Employment Opportunity Commission",
}
_RESIDUAL_DEPARTMENT = "Other/independent agency"


def _department(email: str | None) -> str:
    if not email or "@" not in email:
        return _RESIDUAL_DEPARTMENT
    labels = email.split("@")[-1].lower().strip().strip(".").split(".")
    for width in (3, 2):  # match sub-domains such as cms.hhs.gov before hhs.gov
        if len(labels) >= width:
            key = ".".join(labels[-width:])
            if key in _DOMAIN_TO_DEPARTMENT:
                return _DOMAIN_TO_DEPARTMENT[key]
    return _RESIDUAL_DEPARTMENT


def download(force: bool = False) -> Path:
    if XML_PATH.exists() and not force and XML_PATH.stat().st_size > 10_000_000:
        LOG.info("PRA inventory already downloaded (%.1f MB)", XML_PATH.stat().st_size / 1e6)
        return XML_PATH
    sess = session()
    resp = get_with_retry(sess, CONFIG["burden"]["pra_inventory_url"], timeout=300)
    if resp is None:
        raise RuntimeError("could not download the PRA inventory")
    XML_PATH.write_bytes(resp.content)
    LOG.info("downloaded PRA inventory (%.1f MB)", len(resp.content) / 1e6)
    return XML_PATH


def _num(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return None


def parse() -> pd.DataFrame:
    """Flatten the inventory to one row per information collection request."""
    root = ET.parse(download()).getroot()
    rows: list[dict[str, Any]] = []
    for icr in root.findall("InformationCollectionRequest"):
        ref = icr.findtext("ICRReferenceNumber")
        control = icr.findtext("OMBControlNumber") or ""
        status = icr.findtext("ICRStatus")
        abstract = (icr.findtext("Abstract") or "").strip()

        responses = _num(icr.findtext("./Burden/BurdenResponse/TotalQuantity"))
        hours = _num(icr.findtext("./Burden/BurdenHour/TotalQuantity"))
        cost = _num(icr.findtext("./Burden/BurdenCost/TotalAmount"))

        cfr: list[str] = []
        publics: set[str] = set()
        obligations: set[str] = set()
        frequencies: set[str] = set()
        forms: list[str] = []
        lob: set[str] = set()
        for coll in icr.findall(".//InformationCollection"):
            for cit in coll.findall(".//CFRCitation/CFRCitation"):
                title, part = cit.findtext("CFRTitle"), cit.findtext("CFRPart")
                if title and part:
                    cfr.append(f"{title.strip()}:{part.strip()}")
            for pub in coll.findall(".//AffectedPublicCode/PublicCode"):
                if pub.text:
                    publics.add(pub.text.strip())
            if coll.findtext("ObligationCode"):
                obligations.add(coll.findtext("ObligationCode").strip())
            for freq in coll.findall(".//ReportingFrequency"):
                if freq.text:
                    frequencies.add(freq.text.strip())
            for inst in coll.findall(".//Instrument"):
                name = inst.findtext("FormName") or inst.findtext("FormNumber")
                if name:
                    forms.append(name.strip())
            for line in coll.findall(".//LineOfBusiness"):
                if line.text:
                    lob.add(line.text.strip())

        statutes = [
            (st.findtext("NameOfLaw") or "").strip()
            for st in icr.findall(".//AuthorizingStatute//USCode")
            if st.findtext("NameOfLaw")
        ]
        agency = icr.findtext("AgencyCode") or control[:4]
        contact = icr.findtext(".//AgencyContact/Person/ElectronicAddress")
        rows.append(
            {
                "icr_ref": ref,
                "omb_control_number": control,
                "agency_code": agency,
                "department": _department(contact),
                "title": (icr.findtext("Title") or "").strip(),
                "abstract": abstract,
                "icr_status": status,
                "icr_type": icr.findtext("ICRTypeCode"),
                "expiration": icr.findtext("./Expiration/ExpirationDate"),
                "annual_responses": responses,
                "annual_burden_hours": hours,
                "annual_burden_cost": cost,
                "federal_cost": _num(icr.findtext("AnnualFederalCostAmount")),
                "n_cfr_parts": len(set(cfr)),
                "cfr_parts": ";".join(sorted(set(cfr))),
                "affected_public": ";".join(sorted(publics)),
                "obligation": ";".join(sorted(obligations)),
                "reporting_frequency": ";".join(sorted(frequencies)),
                "n_instruments": len(forms),
                "instruments": ";".join(forms[:12]),
                "line_of_business": ";".join(sorted(lob)),
                "n_statutes": len(statutes),
                "statutes": ";".join(sorted(set(statutes))[:8]),
                "pii": icr.findtext("PIIFlag"),
                "privacy_act_statement": icr.findtext("PrivacyActStatementFlag"),
                "healthcare": icr.findtext("HealthcareIndicator"),
            }
        )
    df = pd.DataFrame(rows)
    # Collections whose contact address is missing inherit the departmental
    # label that dominates the other collections of the same agency code.
    resolved = df.loc[df.department != _RESIDUAL_DEPARTMENT]
    if len(resolved):
        by_agency = resolved.groupby("agency_code")["department"].agg(
            lambda s: s.value_counts().idxmax()
        )
        fill = df["agency_code"].map(by_agency)
        mask = (df.department == _RESIDUAL_DEPARTMENT) & fill.notna()
        df.loc[mask, "department"] = fill[mask]
        LOG.info("departmental label imputed from agency code for %d collections", int(mask.sum()))
    df["hours_per_response"] = df["annual_burden_hours"] / df["annual_responses"].replace(0, pd.NA)
    LOG.info("parsed %d information collections (%d active)", len(df), (df.icr_status == "Active").sum())
    return df


def run() -> pd.DataFrame:
    df = parse()
    save_table(df, OUT)
    return df


if __name__ == "__main__":
    run()
