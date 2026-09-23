"""Scoring a generated answer against JQaRA's gold answer.

JQaRA carries one gold answer per question, inherited from JAQKET, and those
answers are short proper nouns: 絶対零度, 加藤シゲアキ, 天邪鬼. A generated
answer counts as correct when it contains the gold string after normalisation.

Two things about this rule are worth stating, because every end-to-end number
rests on it.

Containment rather than equality: the model is asked for the term alone but
writes a sentence often enough that equality would measure instruction
following rather than retrieval. The cost is that a wrong answer containing
the gold string as a substring is scored correct.

One gold answer per question: a correct answer worded differently (a reading
instead of the kanji, a fuller or shorter form of a name) is scored wrong.
This depresses every route equally, so it is a floor on the absolute numbers
rather than a bias between them — but it means the absolute figures understate
what the pipelines actually get right.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Removed before comparison: the gold answers carry none of it, and a model
# writing 「絶対零度」 or 加藤 シゲアキ is not answering differently.
_IGNORED = re.compile(r"[\s　「」『』（）()\[\]【】、。,.・:：;；!！?？\"'`]")


def normalise_answer(text: str) -> str:
    """Fold the differences that are not differences of answer."""
    return _IGNORED.sub("", unicodedata.normalize("NFKC", text)).lower()


def contains_answer(generated: str, gold: list[str]) -> bool:
    return score_answer(generated, gold).correct


@dataclass(frozen=True)
class AnswerScore:
    correct: bool
    matched: str | None = None


def score_answer(generated: str, gold: list[str]) -> AnswerScore:
    """Whether the generation contains any gold answer, and which one."""
    haystack = normalise_answer(generated or "")
    if not haystack:
        return AnswerScore(correct=False)
    for answer in gold:
        needle = normalise_answer(answer)
        if needle and needle in haystack:
            return AnswerScore(correct=True, matched=answer)
    return AnswerScore(correct=False)
