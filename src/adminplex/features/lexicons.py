"""Language-specific marker inventories used by the rule-based dimensions.

The inventories are deliberately explicit rather than learned: a public manager
must be able to read them, contest them and apply them to a new document by
hand. Each entry is a regular-expression fragment matched on lower-cased text
with word boundaries.
"""
from __future__ import annotations

import re

LANGUAGES = ("en", "es")

# --- 3. Legal density -------------------------------------------------------
LEGAL_CITATION = {
    "en": [
        r"\b\d+\s*u\.?s\.?c\.?\b",
        r"\b\d+\s*cfr\b",
        r"\bsection\s+\d+",
        r"\bsections\s+\d+",
        r"\bparagraph\s+\(?\w{1,3}\)?",
        r"\bschedule\s+\d+",
        r"\bregulation\s+\d+",
        r"\bs\.\s?\d+",
        r"\bact\s+(?:19|20)\d{2}\b",
        r"\bstatutory instrument\b",
        r"\bpart\s+\d+\b",
        r"\barticle\s+\d+",
        r"\bdirective\s+\d{2,4}/\d+",
        r"\bregulation\s*\(e[uc]\)\s*(?:no\.?\s*)?\d+",
    ],
    "es": [
        r"\bart[íi]culo\s+\d+",
        r"\barts?\.\s*\d+",
        r"\bley\s+\d+/\d{2,4}",
        r"\bley\s+org[áa]nica\b",
        r"\breal decreto(?:-ley)?\s+\d+/\d{2,4}",
        r"\br\.?\s?d\.?\s+\d+/\d{2,4}",
        r"\borden\s+[a-z]{3,4}/\d+/\d{4}",
        r"\bresoluci[óo]n de\b",
        r"\breglamento\s*\(u?e\)\s*(?:n\.?[ºo]?\s*)?\d+",
        r"\bdirectiva\s+\d{2,4}/\d+",
        r"\bdisposici[óo]n (?:adicional|transitoria|final|derogatoria)\b",
        r"\bapartado\s+\d+",
        r"\banexo\s+[ivx\d]+",
        r"\bbolet[íi]n oficial del estado\b",
    ],
}

# --- 4. Cross-references ----------------------------------------------------
CROSS_REFERENCE = {
    "en": [
        r"\bsee\s+(?:section|paragraph|part|article|schedule|annex|regulation|chapter)\b",
        r"\b(?:as )?(?:set out|described|defined|specified|provided|listed|prescribed) in\b",
        r"\brefer(?:red)? to in\b",
        r"\bin accordance with\b",
        r"\bpursuant to\b",
        r"\bunder (?:section|regulation|article|part|paragraph)\b",
        r"\bwithin the meaning of\b",
        r"\bsubject to (?:section|regulation|article|paragraph|part)\b",
        r"\bfor the purposes of (?:section|regulation|article|paragraph)\b",
    ],
    "es": [
        r"\bv[ée]ase\b",
        r"\bconforme a l?[oa]?s?\b",
        r"\bde conformidad con\b",
        r"\bcon arreglo a\b",
        r"\ben virtud de\b",
        r"\bde acuerdo con lo\b",
        r"\bseg[úu]n lo (?:dispuesto|establecido|previsto|se[ñn]alado)\b",
        r"\ba (?:que se refiere|los efectos de)\b",
        r"\b(?:previst|establecid|dispuest|se[ñn]alad|indicad)[oa]s? en\b",
        r"\bregulad[oa] en\b",
        r"\ben los t[ée]rminos (?:establecidos|previstos)\b",
    ],
}

# --- 5. Exceptions ----------------------------------------------------------
EXCEPTION = {
    "en": [
        r"\bexcept\b",
        r"\bexcept for\b",
        r"\bunless\b",
        r"\bsave (?:that|for|where|as)\b",
        r"\bnotwithstanding\b",
        r"\bwithout prejudice to\b",
        r"\bother than\b",
        r"\bprovided that\b",
        r"\bsubject to\b",
        r"\bdoes not apply\b",
        r"\bexemption\b",
        r"\bexempt from\b",
        r"\bderogation\b",
        r"\bshall not apply\b",
    ],
    "es": [
        r"\bexcepto\b",
        r"\bsalvo\b",
        r"\bno obstante\b",
        r"\bsin perjuicio de\b",
        r"\bsiempre que\b",
        r"\ba excepci[óo]n de\b",
        r"\bcon excepci[óo]n de\b",
        r"\bexenci[óo]n\b",
        r"\bexent[oa]s? de\b",
        r"\bno ser[áa] de aplicaci[óo]n\b",
        r"\bquedan? exclu[íi]d[oa]s?\b",
        r"\bsalvedad\b",
    ],
}

# --- 6/12. Requirements and documentary effort ------------------------------
DEONTIC = {
    "en": [r"\bmust\b", r"\bshall\b", r"\bis required to\b", r"\bare required to\b", r"\bhas to\b", r"\bhave to\b", r"\bneed to\b", r"\bare obliged to\b", r"\bmandatory\b", r"\brequired\b"],
    "es": [r"\bdeber[áa]n?\b", r"\bdeber[áa]\b", r"\bser[áa] obligatorio\b", r"\bes obligatori[oa]\b", r"\bhabr[áa] de\b", r"\best[áa]n? obligad[oa]s?\b", r"\bse exigir[áa]\b", r"\bqueda obligad[oa]\b", r"\bpreceptiv[oa]\b", r"\brequerid[oa]\b"],
}

DOCUMENT_NOUN = {
    "en": [
        r"\bform\b", r"\bapplication form\b", r"\bcertificate\b", r"\bdeclaration\b", r"\bstatement\b",
        r"\bevidence\b", r"\bproof of\b", r"\bcopy of\b", r"\bannex\b", r"\bappendix\b", r"\bschedule\b",
        r"\breceipt\b", r"\binvoice\b", r"\breport\b", r"\brecord\b", r"\blicen[cs]e\b", r"\bpermit\b",
        r"\bidentity document\b", r"\bpassport\b", r"\baccounts\b", r"\bdocumentation\b", r"\bsupporting document",
    ],
    "es": [
        r"\bformulario\b", r"\bsolicitud\b", r"\bcertificad[oa]\b", r"\bdeclaraci[óo]n responsable\b",
        r"\bdeclaraci[óo]n\b", r"\bacreditaci[óo]n\b", r"\bjustificante\b", r"\bcopia\b", r"\banexo\b",
        r"\bmemoria\b", r"\bfactura\b", r"\binforme\b", r"\blicencia\b", r"\bautorizaci[óo]n\b",
        r"\bdocumento nacional de identidad\b", r"\bdni\b", r"\bnif\b", r"\bpoder notarial\b",
        r"\bescritura\b", r"\bdocumentaci[óo]n\b", r"\bcuentas anuales\b", r"\bavales?\b",
    ],
}

# --- 9. Ambiguity -----------------------------------------------------------
AMBIGUITY = {
    "en": [
        r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bgenerally\b", r"\bnormally\b", r"\busually\b",
        r"\btypically\b", r"\bwhere appropriate\b", r"\bas appropriate\b", r"\bif applicable\b",
        r"\bas necessary\b", r"\breasonable\b", r"\breasonably\b", r"\bsubstantially\b", r"\bapproximately\b",
        r"\bin principle\b", r"\bwherever possible\b", r"\bas soon as (?:possible|practicable)\b",
        r"\bat the discretion of\b", r"\bsuitable\b", r"\badequate\b", r"\bfrom time to time\b",
    ],
    "es": [
        r"\bpodr[áa]n?\b", r"\bpodr[íi]an?\b", r"\bpreferentemente\b", r"\bnormalmente\b",
        r"\bgeneralmente\b", r"\bhabitualmente\b", r"\bcuando proceda\b", r"\ben su caso\b",
        r"\bsi procede\b", r"\ben la medida de lo posible\b", r"\brazonable\b", r"\badecuad[oa]\b",
        r"\bsuficiente\b", r"\baproximadamente\b", r"\ba criterio de\b", r"\ben principio\b",
        r"\bcon car[áa]cter general\b", r"\ba la mayor brevedad\b", r"\bsegún corresponda\b",
    ],
}

# --- 13. Conditional decisions ---------------------------------------------
CONDITIONAL = {
    "en": [
        r"\bif\b", r"\bwhere\s+(?:you|the|a|an|any)\b", r"\bin the event that\b", r"\bin case of\b",
        r"\bwhen\s+(?:you|the|a|an|any)\b", r"\bshould you\b", r"\bdepending on\b", r"\botherwise\b",
        r"\bonly if\b", r"\bunless you\b",
    ],
    "es": [
        r"\bsi\s+(?:el|la|los|las|usted|se|no|hay)\b", r"\ben caso de\b", r"\bcuando\s+(?:el|la|los|las|se)\b",
        r"\bsupuesto de que\b", r"\ben el supuesto\b", r"\bde lo contrario\b", r"\bs[óo]lo si\b",
        r"\ben funci[óo]n de\b", r"\bsiempre y cuando\b",
    ],
}

# --- 2. Subordination -------------------------------------------------------
SUBORDINATOR = {
    "en": [
        r"\bwhich\b", r"\bwho\b", r"\bwhom\b", r"\bwhose\b", r"\bthat\b", r"\bbecause\b", r"\balthough\b",
        r"\bwhereas\b", r"\bwhile\b", r"\bwhether\b", r"\bafter\b", r"\bbefore\b", r"\bsince\b",
        r"\bso that\b", r"\bin order (?:to|that)\b", r"\bsuch that\b", r"\bunless\b", r"\bif\b", r"\bwhen\b",
    ],
    "es": [
        r"\bque\b", r"\bquien(?:es)?\b", r"\bcuyo?s?\b", r"\bcuya?s?\b", r"\bporque\b", r"\baunque\b",
        r"\bmientras\b", r"\bsi\b", r"\bcuando\b", r"\bdonde\b", r"\bpara que\b", r"\ba fin de que\b",
        r"\bde modo que\b", r"\bantes de que\b", r"\bdespu[ée]s de que\b", r"\bsiempre que\b",
    ],
}

# --- 7/8. Steps and ordering dependencies ----------------------------------
STEP_MARKER = {
    "en": [r"\bstep\s+\d+", r"\bstage\s+\d+", r"^\s*\d+[.)]\s+", r"\bfirst(?:ly)?\b", r"\bthen\b", r"\bnext\b", r"\bfinally\b", r"\bafterwards\b", r"\bsubsequently\b"],
    "es": [r"\bpaso\s+\d+", r"\betapa\s+\d+", r"\bfase\s+\d+", r"^\s*\d+[.)]\s+", r"\ben primer lugar\b", r"\bposteriormente\b", r"\ba continuaci[óo]n\b", r"\bseguidamente\b", r"\bpor [úu]ltimo\b", r"\bfinalmente\b"],
}

PRECEDENCE = {
    "en": [
        r"\bbefore\s+(?:you|the|submitting|applying)\b", r"\bonce\s+(?:you|the|your)\b",
        r"\bafter\s+(?:you|the|receiving|submitting)\b", r"\bprior to\b", r"\bhaving\s+\w+ed\b",
        r"\byou must first\b", r"\bmust have (?:already )?(?:been|obtained|submitted)\b",
        r"\bupon receipt of\b", r"\bfollowing (?:the )?(?:approval|submission|receipt)\b",
    ],
    "es": [
        r"\bcon car[áa]cter previo\b", r"\bpreviamente\b", r"\buna vez\b", r"\btras (?:la|el|haber)\b",
        r"\bcon anterioridad a\b", r"\bantes de (?:presentar|solicitar|la|el)\b",
        r"\bdeber[áa] haber\b", r"\bhabiendo\s+\w+do\b", r"\bcumplimentad[oa] (?:el|la)\b",
        r"\brecibida la\b", r"\bdespu[ée]s de\b",
    ],
}


def compile_all(patterns: dict[str, list[str]]) -> dict[str, re.Pattern[str]]:
    """Combine each language's fragments into one alternation for speed."""
    return {
        lang: re.compile("|".join(f"(?:{p})" for p in frags), re.IGNORECASE | re.MULTILINE)
        for lang, frags in patterns.items()
    }


LEGAL_CITATION_RE = compile_all(LEGAL_CITATION)
CROSS_REFERENCE_RE = compile_all(CROSS_REFERENCE)
EXCEPTION_RE = compile_all(EXCEPTION)
DEONTIC_RE = compile_all(DEONTIC)
DOCUMENT_NOUN_RE = compile_all(DOCUMENT_NOUN)
AMBIGUITY_RE = compile_all(AMBIGUITY)
CONDITIONAL_RE = compile_all(CONDITIONAL)
SUBORDINATOR_RE = compile_all(SUBORDINATOR)
STEP_MARKER_RE = compile_all(STEP_MARKER)
PRECEDENCE_RE = compile_all(PRECEDENCE)

# Distinct document types, used for dimension 12 (documentary effort).
DOCUMENT_TYPES = {
    lang: [re.compile(p, re.IGNORECASE) for p in frags] for lang, frags in DOCUMENT_NOUN.items()
}


def for_language(mapping: dict[str, re.Pattern[str]], language: str | None) -> re.Pattern[str]:
    """Fall back to English patterns for languages without an inventory."""
    return mapping.get((language or "en")[:2].lower(), mapping["en"])
