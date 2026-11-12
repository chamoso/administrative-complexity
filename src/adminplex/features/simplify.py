"""The two interventions applied to procedures in experiments 5 and 6.

Removal strips out what a procedure demands and leaves the rest of the text
alone. Rewriting changes how the same demands are expressed and removes
nothing. Keeping the two apart is the whole point of the pair: one asks what
the requirements cost, the other asks what the prose costs.

The prompts live here rather than in the script that runs them so that the
documentation can quote the instruction that was actually used.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You edit official administrative documents for a public administration "
    "research team. You follow the instruction exactly and return only the "
    "edited document, with no commentary, preamble or explanation."
)

REMOVAL_PROMPT = """Remove from the document below every requirement it places on the reader:
every document, certificate, form, declaration, proof or annex the reader must
obtain, produce or submit, and every condition the reader must satisfy.

Keep everything else exactly as it is: keep the purpose, the entitlement, the
contact details, the deadlines that are not conditions, and the wording of the
sentences you do not remove. Do not summarise. Do not rewrite for clarity. Do
not add anything.

DOCUMENT:
{text}

EDITED DOCUMENT:"""

REWRITE_PROMPT = """Rewrite the document below in plain language.

Every obligation must survive the rewrite: every document the reader must
supply, every condition they must meet, every deadline, every step. Remove
nothing that the reader must do. Change only how it is said: shorter sentences,
ordinary words, the steps in the order they must be taken, conditions stated
once and plainly.

DOCUMENT:
{text}

REWRITTEN DOCUMENT:"""

INTERVENTIONS: dict[str, str] = {
    "removal": REMOVAL_PROMPT,
    "rewrite": REWRITE_PROMPT,
}
