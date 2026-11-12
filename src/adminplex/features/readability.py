"""Classical readability indices, implemented so that the comparison with the
Administrative Complexity Index is exactly reproducible.

The formulas are the published ones. Syllable counting is language-specific
because the English heuristic is not valid for Spanish; the Spanish counter
follows the standard vowel-group rule with diphthong merging.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:['’-][A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)*")
_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n{2,}")

_EN_VOWELS = "aeiouy"
_ES_VOWELS = "aeiouáéíóúü"
_ES_STRONG = "aeoáéó"
_ES_WEAK = "iuíúü"


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    return [p for p in parts if _WORD_RE.search(p)]


def syllables_en(word: str) -> int:
    w = word.lower()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    groups = re.findall(rf"[{_EN_VOWELS}]+", w)
    count = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1
    if w.endswith(("les", "le")) and len(w) > 2 and w[-3] not in _EN_VOWELS:
        count += 0
    return max(1, count)


def syllables_es(word: str) -> int:
    w = word.lower()
    w = re.sub(r"[^a-záéíóúüñ]", "", w)
    if not w:
        return 0
    count = 0
    i = 0
    while i < len(w):
        if w[i] in _ES_VOWELS:
            count += 1
            j = i + 1
            # merge diphthongs and triphthongs: two adjacent vowels form one
            # nucleus unless both are strong
            while j < len(w) and w[j] in _ES_VOWELS:
                if w[j - 1] in _ES_STRONG and w[j] in _ES_STRONG:
                    break
                j += 1
            i = j
        else:
            i += 1
    return max(1, count)


def count_syllables(word: str, language: str) -> int:
    return syllables_es(word) if language.startswith("es") else syllables_en(word)


@dataclass
class Readability:
    n_words: int
    n_sentences: int
    n_syllables: int
    n_complex_words: int
    n_long_words: int
    n_chars: int
    flesch: float
    flesch_kincaid: float
    smog: float
    gunning_fog: float
    ari: float
    coleman_liau: float
    lix: float
    rix: float


def compute(text: str, language: str = "en") -> Readability:
    toks = words(text)
    sents = sentences(text)
    n_w = len(toks)
    n_s = max(1, len(sents))
    if n_w == 0:
        return Readability(0, 0, 0, 0, 0, 0, *([math.nan] * 8))

    syl = [count_syllables(t, language) for t in toks]
    n_syl = sum(syl)
    n_complex = sum(1 for s in syl if s >= 3)
    n_long = sum(1 for t in toks if len(t) > 6)
    n_chars = sum(len(t) for t in toks)

    asl = n_w / n_s                      # average sentence length
    asw = n_syl / n_w                    # average syllables per word
    pct_complex = 100.0 * n_complex / n_w
    long_ratio = 100.0 * n_long / n_w

    flesch = 206.835 - 1.015 * asl - 84.6 * asw
    fk = 0.39 * asl + 11.8 * asw - 15.59
    smog = 1.0430 * math.sqrt(n_complex * (30.0 / n_s)) + 3.1291
    fog = 0.4 * (asl + pct_complex)
    ari = 4.71 * (n_chars / n_w) + 0.5 * asl - 21.43
    cli = 0.0588 * (100.0 * n_chars / n_w) - 0.296 * (100.0 * n_s / n_w) - 15.8
    lix = asl + long_ratio
    rix = n_long / n_s

    return Readability(
        n_words=n_w,
        n_sentences=len(sents),
        n_syllables=n_syl,
        n_complex_words=n_complex,
        n_long_words=n_long,
        n_chars=n_chars,
        flesch=flesch,
        flesch_kincaid=fk,
        smog=smog,
        gunning_fog=fog,
        ari=ari,
        coleman_liau=cli,
        lix=lix,
        rix=rix,
    )


def as_dict(text: str, language: str = "en") -> dict[str, float]:
    return asdict(compute(text, language))
