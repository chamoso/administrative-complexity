"""Tests for the functions that produce the reported results.

These are not tests of the research claim. They check that each measure counts
what its name says it counts, on text where the right answer can be worked out
by hand, and that the properties the analysis relies on actually hold: rates
independent of document length, scales bounded where they are supposed to be,
and a reference scale that does not shift when different documents are scored
against it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adminplex.features import rulebased  # noqa: E402
from adminplex.features.llm import excerpt, parse_response  # noqa: E402
from adminplex.features.readability import sentences, words  # noqa: E402
from adminplex.index.dimensions import (  # noqa: E402
    percentile_rank,
    percentile_rank_against,
)


# --------------------------------------------------------------------------
# Tokenisation, on which every rate depends


def test_words_and_sentences_agree_with_a_hand_count():
    text = "The applicant must submit the form. A fee of £30 applies. Payment is due within 14 days."
    assert len(sentences(text)) == 3
    assert len(words(text)) == pytest.approx(len(text.split()), abs=2)


def test_empty_text_produces_zeroes_not_errors():
    dim = rulebased.compute("", "en")
    assert dim.n_words == 0
    assert dim.exception_rate == 0.0
    assert dim.redundancy_ratio == 0.0


# --------------------------------------------------------------------------
# Each measure counts what it says it counts


def test_exceptions_are_counted():
    plain = "You must send the form to the council. The council will reply."
    with_carve_outs = (
        "You must send the form to the council, except where you are exempt, "
        "unless the council decides otherwise, and without prejudice to any appeal."
    )
    assert rulebased.compute(plain, "en").exception_rate == 0.0
    assert rulebased.compute(with_carve_outs, "en").exception_rate > 0.0


def test_ambiguity_picks_up_discretionary_language():
    firm = "The council will decide within 14 days. You must attend the appointment."
    vague = "The council may decide, where appropriate, and will normally reply, if applicable."
    assert rulebased.compute(vague, "en").ambiguity_rate > rulebased.compute(firm, "en").ambiguity_rate


def test_spanish_lexicons_are_used_for_spanish_text():
    spanish = (
        "El solicitante deberá presentar el certificado, salvo que esté exento, "
        "sin perjuicio de lo dispuesto en el artículo 12."
    )
    as_spanish = rulebased.compute(spanish, "es")
    as_english = rulebased.compute(spanish, "en")
    assert as_spanish.language_used == "es"
    assert as_spanish.exception_rate > 0.0
    # The English lexicon should not find Spanish carve-outs; if it does, the
    # two language paths are not actually separate.
    assert as_spanish.exception_rate > as_english.exception_rate


def test_unknown_language_falls_back_to_english_rather_than_failing():
    dim = rulebased.compute("You must submit the form unless exempt.", "xx")
    assert dim.language_used == "en"
    assert dim.exception_rate > 0.0


def test_redundancy_detects_repetition():
    varied = (
        "Applicants register on the portal before the closing date. "
        "Evidence of residence must accompany every submission received. "
        "The committee meets quarterly to consider outstanding cases. "
        "Fees are refunded when an application is withdrawn early. "
        "Appeals go to the regional tribunal within twenty-eight days. "
        "Successful claimants receive written notice by second-class post. "
        "Records are destroyed six years after a file is closed. "
        "Interpreters can be arranged at no cost on request."
    )
    repeated = " ".join(["the applicant must submit the completed form to the local authority"] * 12)
    assert rulebased.redundancy(repeated) > 0.5
    assert rulebased.redundancy(varied) < 0.2


def test_short_text_has_no_redundancy_rather_than_a_division_error():
    assert rulebased.redundancy("too short to shingle") == 0.0


# --------------------------------------------------------------------------
# Properties the analysis relies on


def test_rates_do_not_grow_with_document_length():
    """A rate per thousand words must be stable when a document is duplicated.

    Length is a separate dimension; if the other rates tracked length too, the
    index would be measuring size several times over.
    """
    passage = (
        "The applicant must submit the certificate, except where exempt. "
        "The authority may request further evidence if appropriate. "
    )
    once = rulebased.compute(passage, "en")
    thrice = rulebased.compute(passage * 3, "en")
    for field in ("exception_rate", "ambiguity_rate", "requirement_rate", "conditional_rate"):
        assert getattr(thrice, field) == pytest.approx(getattr(once, field), rel=0.1), field


def test_step_extraction_finds_an_enumerated_procedure():
    text = (
        "How to apply.\n\n"
        "1. Complete the application form and sign it.\n"
        "2. Attach proof of address issued within the last three months.\n"
        "3. Pay the fee of thirty pounds using the online service.\n"
        "4. Wait for the authority to confirm receipt of your application.\n"
    )
    dim = rulebased.compute(text, "en")
    assert dim.n_steps >= 4
    assert dim.dag_depth >= 2


def test_dependencies_deepen_when_steps_refer_back():
    sequential = (
        "1. Register on the portal and create an account there.\n"
        "2. Upload the completed form to the portal you registered on.\n"
        "3. Pay the application fee using the online payment service.\n"
    )
    with_backrefs = (
        "1. Register on the portal and create an account there.\n"
        "2. Upload the completed form once step 1 has been approved.\n"
        "3. Pay the fee only after step 2 is complete and confirmed.\n"
    )
    plain = rulebased.compute(sequential, "en")
    linked = rulebased.compute(with_backrefs, "en")
    assert linked.dag_edges >= plain.dag_edges


def test_dag_metrics_survive_a_cycle():
    """Back-references can point forwards and create a cycle; the metrics must
    still return rather than raise, because a single malformed procedure would
    otherwise stop the whole corpus."""
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    metrics = rulebased.dag_metrics(graph)
    assert metrics["dag_nodes"] == 3
    assert metrics["dag_depth"] >= 1


# --------------------------------------------------------------------------
# The assessment: parsing what the model returns


def test_parses_a_well_formed_response():
    raw = (
        '{"cognitive_load": 7, "legal_knowledge": 5, "clarity": 4, '
        '"n_requirements": 3, "n_documents": 2, "justification": "several conditions"}'
    )
    parsed = parse_response(raw)
    assert parsed["cognitive_load"] == 7.0
    assert parsed["justification"] == "several conditions"


def test_recovers_json_wrapped_in_commentary():
    raw = 'Here is my assessment:\n{"cognitive_load": 3, "legal_knowledge": 2, "clarity": 8}\nHope that helps.'
    assert parse_response(raw)["clarity"] == 8.0


def test_repairs_a_trailing_comma():
    raw = '{"cognitive_load": 3, "legal_knowledge": 2, "clarity": 8,}'
    assert parse_response(raw) is not None


def test_scores_outside_the_scale_are_clipped_not_accepted():
    raw = '{"cognitive_load": 15, "legal_knowledge": -3, "clarity": 8}'
    parsed = parse_response(raw)
    assert parsed["cognitive_load"] == 10.0
    assert parsed["legal_knowledge"] == 1.0


def test_unusable_responses_are_rejected_rather_than_guessed():
    assert parse_response("") is None
    assert parse_response("I cannot assess this document.") is None
    # Missing one of the three scales makes the record unusable.
    assert parse_response('{"cognitive_load": 5, "legal_knowledge": 4}') is None


def test_long_documents_keep_their_middle_and_end():
    text = "START " + ("filler " * 5000) + "MIDDLE" + ("filler " * 5000) + " END"
    trimmed = excerpt(text, max_chars=600)
    assert len(trimmed) <= 700
    assert trimmed.startswith("START")
    assert trimmed.rstrip().endswith("END")


# --------------------------------------------------------------------------
# Scaling


def test_percentile_rank_spans_the_scale():
    ranked = percentile_rank(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert ranked.min() == pytest.approx(25.0)
    assert ranked.max() == pytest.approx(100.0)


def test_percentile_rank_keeps_missing_values_missing():
    ranked = percentile_rank(pd.Series([1.0, np.nan, 3.0]))
    assert ranked.isna().sum() == 1


def test_reference_scale_does_not_depend_on_what_is_scored_against_it():
    """The property the before-and-after comparison rests on.

    A document must get the same score whether it is measured alone or
    alongside others, or a fall in the index after an edit could just be the
    other edited documents moving around it.
    """
    reference = pd.Series(np.arange(0.0, 100.0))
    alone = percentile_rank_against(pd.Series([50.0]), reference)
    in_company = percentile_rank_against(pd.Series([50.0, 1.0, 99.0]), reference)
    assert alone.iloc[0] == pytest.approx(in_company.iloc[0])
    assert alone.iloc[0] == pytest.approx(50.5, abs=1.0)


def test_reference_scale_handles_values_beyond_the_reference_range():
    reference = pd.Series([10.0, 20.0, 30.0])
    scored = percentile_rank_against(pd.Series([0.0, 100.0]), reference)
    assert scored.iloc[0] == pytest.approx(0.0)
    assert scored.iloc[1] == pytest.approx(100.0)


def test_reference_scale_without_a_reference_returns_missing_not_zero():
    scored = percentile_rank_against(pd.Series([1.0, 2.0]), pd.Series(dtype=float))
    assert scored.isna().all()
