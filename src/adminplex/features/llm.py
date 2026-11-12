"""The three judgement-based dimensions of the index.

Cognitive load, required legal knowledge and clarity cannot be counted from the
surface of a text: they are assessments of what the document demands of the
person who has to comply with it. They are therefore elicited with a fixed
rubric, at temperature zero, and every score must be accompanied by a written
justification, so that a public manager can audit any individual rating.

The scoring itself runs on a local open-weights model; nothing in the design
depends on a particular vendor, and the robustness checks re-score a subsample
with a second model and with paraphrased rubrics.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

RUBRIC_VERSION = "aci-rubric-1.0"

SYSTEM_PROMPT = (
    "You are assessing official administrative documents on behalf of a public "
    "administration research team. You apply the rubric literally and "
    "conservatively. You never speculate about information that is not in the "
    "text. You always answer with a single JSON object and nothing else."
)

_RUBRIC = """You will assess one official administrative document.

Rate the document on three scales. Each scale runs from 1 to 10.

COGNITIVE_LOAD - how much mental effort a member of the intended audience needs
to work out what they have to do.
  1-2  A single obvious action, no conditions to track.
  3-4  A few actions in a fixed order, all stated plainly.
  5-6  Several actions, some conditions, the reader must hold a few facts in mind.
  7-8  Many interacting conditions, exceptions or deadlines that must be tracked together.
  9-10 The reader must reconstruct the procedure themselves from scattered conditions.

LEGAL_KNOWLEDGE - how much prior legal or technical knowledge the reader needs
before the document becomes usable.
  1-2  None: everyday language.
  3-4  Familiarity with common administrative terms.
  5-6  The reader must know what specific cited instruments require.
  7-8  The reader must interpret statutory language or reconcile several instruments.
  9-10 Professional legal or technical advice is effectively required.

CLARITY - how clearly the document communicates what must be done.
  1-2  Very unclear: the required action cannot be identified with confidence.
  3-4  Unclear: the reader must infer key steps.
  5-6  Adequate: the main action is identifiable, details are vague.
  7-8  Clear: actions, conditions and deadlines are stated explicitly.
  9-10 Exemplary: unambiguous, ordered and complete.

Also report, as integers:
  N_REQUIREMENTS - distinct things the reader must supply, prove or do.
  N_DOCUMENTS    - distinct documents, certificates or forms the reader must obtain or submit.

Answer with exactly this JSON object and no other text:
{"cognitive_load": <1-10>, "legal_knowledge": <1-10>, "clarity": <1-10>,
 "n_requirements": <integer>, "n_documents": <integer>,
 "justification": "<one or two sentences citing what in the document drove the scores>"}
"""

# Paraphrased rubric used only in the robustness check (experiment 9).
_RUBRIC_VARIANT = _RUBRIC.replace(
    "how much mental effort a member of the intended audience needs\nto work out what they have to do",
    "the mental workload the intended reader must sustain in order to establish\ntheir obligations",
).replace(
    "how much prior legal or technical knowledge the reader needs\nbefore the document becomes usable",
    "the level of specialised legal or technical background presupposed by the text",
).replace(
    "how clearly the document communicates what must be done",
    "how transparently the text conveys the action the reader must take",
)

_FIELDS = ("cognitive_load", "legal_knowledge", "clarity", "n_requirements", "n_documents")


def excerpt(text: str, max_chars: int = 12000) -> str:
    """Long documents are represented by head, middle and tail.

    Truncating only the head would bias the assessment towards introductory
    material, where procedural conditions are least likely to appear.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= max_chars:
        return text
    part = max_chars // 3
    mid = len(text) // 2
    return (
        text[:part]
        + "\n\n[...]\n\n"
        + text[mid - part // 2 : mid + part // 2]
        + "\n\n[...]\n\n"
        + text[-part:]
    )


def build_prompt(title: str, text: str, language: str = "en", variant: bool = False, max_chars: int = 12000) -> str:
    rubric = _RUBRIC_VARIANT if variant else _RUBRIC
    return (
        f"{rubric}\n"
        f"---\nDOCUMENT TITLE: {title or '(untitled)'}\n"
        f"DOCUMENT LANGUAGE: {language}\n"
        f"DOCUMENT TEXT:\n{excerpt(text, max_chars)}\n---\n"
        "JSON:"
    )


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_response(raw: str) -> dict[str, Any] | None:
    """Recover the rubric scores from a model response, or None if unusable."""
    if not raw:
        return None
    match = _JSON_RE.search(raw)
    if not match:
        return None
    blob = match.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        blob = re.sub(r",\s*}", "}", blob.replace("\n", " "))
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            return None
    out: dict[str, Any] = {}
    for field in _FIELDS:
        val = data.get(field)
        if isinstance(val, str):
            digits = re.search(r"-?\d+", val)
            val = int(digits.group(0)) if digits else None
        if isinstance(val, (int, float)):
            out[field] = float(val)
    if not {"cognitive_load", "legal_knowledge", "clarity"} <= set(out):
        return None
    for field in ("cognitive_load", "legal_knowledge", "clarity"):
        out[field] = min(10.0, max(1.0, out[field]))
    out["justification"] = str(data.get("justification", ""))[:600]
    return out


@dataclass
class ScoringJob:
    doc_id: str
    title: str
    text: str
    language: str = "en"

    def prompt(self, variant: bool = False, max_chars: int = 12000) -> str:
        return build_prompt(self.title, self.text, self.language, variant, max_chars)


def jobs_from_frame(df, text_column: str = "text", limit: int | None = None) -> list[ScoringJob]:
    rows = df.head(limit) if limit else df
    return [
        ScoringJob(
            doc_id=row.doc_id,
            title=getattr(row, "title", "") or "",
            text=getattr(row, text_column) or "",
            language=getattr(row, "language", "en") or "en",
        )
        for row in rows.itertuples()
    ]
