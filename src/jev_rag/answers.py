"""Scoring a generated answer against JQaRA's gold answer.

JQaRA carries one gold answer per question, inherited from JAQKET (AI王), and
those answers are short proper nouns: 絶対零度, 加藤シゲアキ, 天邪鬼.

The model is asked to put its answer in ``<answer>…</answer>`` and only that
span is scored, following llm-jp-eval's handling of the same dataset. Scoring
the whole response instead would count an answer correct whenever the gold
string appears anywhere in it, and that inflates in ways which differ by
route: three questions in the test split contain the gold answer in the
question itself (音読み/訓読み, 上方置換/下方置換, マッシュポテト/スイートポテト),
130 of 1,667 gold answers are one or two characters after normalisation, and a
hedged answer naming several candidates would score correct for naming the
right one among them.

Two figures are reported. Exact match after normalisation is the primary one,
matching AI王's own scoring. Containment within the extracted span is reported
beside it because exact match alone is unreliable for verbose answers
(Adlakha et al., TACL 2024): a disagreement between the two is the signal that
the models are answering correctly but in a different form.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_QUOTED = re.compile(r"[「『](.*?)[」』]")
_STRIPPED = str.maketrans("", "", "・=-")


def normalise_answer(text: str) -> str:
    """AI王's normalisation: width, case, quotes, separators, whitespace."""
    text = unicodedata.normalize("NFKC", text.replace("～", "〜")).lower()
    text = _QUOTED.sub(r"\1", text)
    return re.sub(r"\s+", "", text.translate(_STRIPPED))


def extract_answer(generated: str) -> str | None:
    """The tagged span, or None when the model did not produce one."""
    match = ANSWER_TAG.search(generated or "")
    return match.group(1) if match else None


@dataclass(frozen=True)
class AnswerScore:
    parsed: bool
    exact: bool
    contains: bool
    extracted: str | None = None
    matched: str | None = None

    @property
    def correct(self) -> bool:
        """The primary verdict."""
        return self.exact


def score_answer(generated: str, gold: list[str]) -> AnswerScore:
    span = extract_answer(generated)
    if span is None:
        return AnswerScore(parsed=False, exact=False, contains=False)

    predicted = normalise_answer(span)
    for answer in gold:
        needle = normalise_answer(answer)
        if not needle:
            continue
        if predicted == needle:
            return AnswerScore(True, True, True, extracted=span, matched=answer)
    for answer in gold:
        needle = normalise_answer(answer)
        if needle and needle in predicted:
            return AnswerScore(True, False, True, extracted=span, matched=answer)
    return AnswerScore(parsed=True, exact=False, contains=False, extracted=span)
