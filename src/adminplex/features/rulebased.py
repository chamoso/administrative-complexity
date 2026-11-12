"""Observable dimensions of administrative complexity.

Twelve of the fifteen dimensions are computed by explicit rules over the text.
Every count is normalised per 1,000 words unless the construct is inherently a
volume (dimension 1) or a structural property of the procedure (dimensions 7
and 8), so that the resulting profile is comparable across documents of very
different length.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from . import lexicons as lx
from .readability import sentences, words

_ENUM_RE = re.compile(r"^\s*(?:\(?\d{1,2}[.)]|[a-z]\)|[ivx]{1,4}[.)]|[-•*])\s+", re.IGNORECASE | re.MULTILINE)
_PARA_RE = re.compile(r"\n\s*\n")
_STEP_HEAD_RE = {
    "en": re.compile(r"\b(?:step|stage)\s+(\d{1,2})\b", re.IGNORECASE),
    "es": re.compile(r"\b(?:paso|etapa|fase)\s+(\d{1,2})\b", re.IGNORECASE),
}
_BACKREF_RE = {
    "en": re.compile(r"\b(?:step|stage|paragraph|point)\s+(\d{1,2})\b", re.IGNORECASE),
    "es": re.compile(r"\b(?:paso|etapa|fase|apartado|punto)\s+(\d{1,2})\b", re.IGNORECASE),
}


def _rate(count: int, n_words: int) -> float:
    return 1000.0 * count / n_words if n_words else 0.0


def _lang(language: str | None) -> str:
    code = (language or "en")[:2].lower()
    return code if code in lx.LANGUAGES else "en"


@dataclass
class Dimensions:
    # 1 length
    n_words: int = 0
    n_sentences: int = 0
    n_paragraphs: int = 0
    # 2 syntactic complexity
    mean_sentence_length: float = 0.0
    p90_sentence_length: float = 0.0
    subordination_rate: float = 0.0
    clauses_per_sentence: float = 0.0
    # 3 legal density
    legal_citation_rate: float = 0.0
    distinct_legal_sources: int = 0
    # 4 cross-references
    cross_reference_rate: float = 0.0
    # 5 exceptions
    exception_rate: float = 0.0
    # 6 requirements
    requirement_rate: float = 0.0
    n_requirement_sentences: int = 0
    # 7 steps
    n_steps: int = 0
    # 8 dependencies
    dag_nodes: int = 0
    dag_edges: int = 0
    dag_depth: int = 0
    dag_width: int = 0
    dag_density: float = 0.0
    precedence_rate: float = 0.0
    # 9 ambiguity
    ambiguity_rate: float = 0.0
    # 12 documentary effort
    n_document_types: int = 0
    document_mention_rate: float = 0.0
    # 13 conditional decisions
    conditional_rate: float = 0.0
    # 14 redundancy
    redundancy_ratio: float = 0.0
    # bookkeeping
    language_used: str = "en"


def _syntactic(text: str, sents: list[str], n_words: int, lang: str) -> dict[str, float]:
    lengths = [len(words(s)) for s in sents] or [0]
    lengths_sorted = sorted(lengths)
    p90 = lengths_sorted[min(len(lengths_sorted) - 1, int(0.9 * len(lengths_sorted)))]
    sub = len(lx.for_language(lx.SUBORDINATOR_RE, lang).findall(text))
    # A clause boundary is approximated by a subordinator or a comma; this
    # proxy is validated against a parsed subsample in the test suite.
    commas = text.count(",")
    n_s = max(1, len(sents))
    return {
        "mean_sentence_length": sum(lengths) / n_s,
        "p90_sentence_length": float(p90),
        "subordination_rate": sub / n_s,
        "clauses_per_sentence": 1.0 + (sub + commas) / n_s,
    }


def _distinct_legal_sources(text: str, lang: str) -> int:
    pat = lx.for_language(lx.LEGAL_CITATION_RE, lang)
    return len({m.group(0).lower().strip() for m in pat.finditer(text)})


def _document_types(text: str, lang: str) -> int:
    found = 0
    for rx in lx.DOCUMENT_TYPES.get(lang, lx.DOCUMENT_TYPES["en"]):
        if rx.search(text):
            found += 1
    return found


def _requirement_sentences(sents: list[str], lang: str) -> int:
    deontic = lx.for_language(lx.DEONTIC_RE, lang)
    docs = lx.for_language(lx.DOCUMENT_NOUN_RE, lang)
    return sum(1 for s in sents if deontic.search(s) and docs.search(s))


def extract_steps(text: str, sents: list[str], lang: str) -> list[str]:
    """Recover the procedural steps as an ordered list of text units."""
    explicit = _STEP_HEAD_RE.get(lang, _STEP_HEAD_RE["en"]).findall(text)
    enumerated = [m.group(0) for m in _ENUM_RE.finditer(text)]
    marker = lx.for_language(lx.STEP_MARKER_RE, lang)
    if explicit:
        blocks = re.split(_STEP_HEAD_RE.get(lang, _STEP_HEAD_RE["en"]), text)
        return [b.strip() for b in blocks[1:] if b.strip()][:60]
    if len(enumerated) >= 3:
        pieces = _ENUM_RE.split(text)
        return [p.strip() for p in pieces[1:] if len(p.strip()) > 20][:60]
    return [s for s in sents if marker.search(s)][:60]


def build_dag(steps: list[str], lang: str) -> nx.DiGraph:
    """A procedure graph: sequential edges plus explicit precedence edges.

    Nodes are steps. A chain edge links consecutive steps. Additional edges are
    added when a step refers back to an earlier numbered step, or when it
    carries an explicit precedence marker ('once the licence has been granted').
    """
    graph = nx.DiGraph()
    for i, step in enumerate(steps):
        graph.add_node(i, text=step[:200])
    for i in range(len(steps) - 1):
        graph.add_edge(i, i + 1)
    backref = _BACKREF_RE.get(lang, _BACKREF_RE["en"])
    precedence = lx.for_language(lx.PRECEDENCE_RE, lang)
    for i, step in enumerate(steps):
        for match in backref.finditer(step):
            target = int(match.group(1)) - 1
            if 0 <= target < len(steps) and target != i:
                graph.add_edge(target, i)
        if precedence.search(step) and i >= 2:
            graph.add_edge(max(0, i - 2), i)
    return graph


def dag_metrics(graph: nx.DiGraph) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {"dag_nodes": 0, "dag_edges": 0, "dag_depth": 0, "dag_width": 0, "dag_density": 0.0}
    working = graph
    if not nx.is_directed_acyclic_graph(working):
        working = graph.copy()
        while not nx.is_directed_acyclic_graph(working):
            cycle = nx.find_cycle(working, orientation="original")
            working.remove_edge(cycle[0][0], cycle[0][1])
    depth = nx.dag_longest_path_length(working) + 1 if working.number_of_nodes() else 0
    levels: dict[int, int] = {}
    for node in nx.topological_sort(working):
        preds = list(working.predecessors(node))
        levels[node] = 1 + max((levels[p] for p in preds), default=-1)
    width = max(sum(1 for v in levels.values() if v == lvl) for lvl in set(levels.values()))
    n = working.number_of_nodes()
    density = working.number_of_edges() / n if n else 0.0
    return {
        "dag_nodes": n,
        "dag_edges": working.number_of_edges(),
        "dag_depth": int(depth),
        "dag_width": int(width),
        "dag_density": density,
    }


def redundancy(text: str, shingle: int = 8) -> float:
    """Share of repeated word 8-grams: how much of the text restates itself."""
    toks = [w.lower() for w in words(text)]
    if len(toks) < shingle * 2:
        return 0.0
    grams = [" ".join(toks[i : i + shingle]) for i in range(len(toks) - shingle + 1)]
    unique = len(set(grams))
    return 1.0 - unique / len(grams)


def compute(text: str, language: str | None = "en") -> Dimensions:
    lang = _lang(language)
    toks = words(text)
    sents = sentences(text)
    n_words = len(toks)
    if n_words == 0:
        return Dimensions(language_used=lang)

    steps = extract_steps(text, sents, lang)
    graph = build_dag(steps, lang)
    dag = dag_metrics(graph)

    counts = {
        "legal": len(lx.for_language(lx.LEGAL_CITATION_RE, lang).findall(text)),
        "xref": len(lx.for_language(lx.CROSS_REFERENCE_RE, lang).findall(text)),
        "exception": len(lx.for_language(lx.EXCEPTION_RE, lang).findall(text)),
        "deontic": len(lx.for_language(lx.DEONTIC_RE, lang).findall(text)),
        "ambiguity": len(lx.for_language(lx.AMBIGUITY_RE, lang).findall(text)),
        "conditional": len(lx.for_language(lx.CONDITIONAL_RE, lang).findall(text)),
        "document": len(lx.for_language(lx.DOCUMENT_NOUN_RE, lang).findall(text)),
        "precedence": len(lx.for_language(lx.PRECEDENCE_RE, lang).findall(text)),
    }

    dim = Dimensions(
        n_words=n_words,
        n_sentences=len(sents),
        n_paragraphs=len([p for p in _PARA_RE.split(text) if p.strip()]),
        legal_citation_rate=_rate(counts["legal"], n_words),
        distinct_legal_sources=_distinct_legal_sources(text, lang),
        cross_reference_rate=_rate(counts["xref"], n_words),
        exception_rate=_rate(counts["exception"], n_words),
        requirement_rate=_rate(counts["deontic"], n_words),
        n_requirement_sentences=_requirement_sentences(sents, lang),
        n_steps=len(steps),
        ambiguity_rate=_rate(counts["ambiguity"], n_words),
        n_document_types=_document_types(text, lang),
        document_mention_rate=_rate(counts["document"], n_words),
        conditional_rate=_rate(counts["conditional"], n_words),
        precedence_rate=_rate(counts["precedence"], n_words),
        redundancy_ratio=redundancy(text),
        language_used=lang,
        **{k: v for k, v in dag.items()},
    )
    for key, val in _syntactic(text, sents, n_words, lang).items():
        setattr(dim, key, val)
    return dim


def as_dict(text: str, language: str | None = "en") -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(compute(text, language))
