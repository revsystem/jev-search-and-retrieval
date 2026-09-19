"""Typed Jev (System One) questions and answers.

Wire format follows the TypeSafe System One API: a request carries one shared
``state`` plus a map of named questions, each of which is a Noul (probability),
a Choice (categorical) or a Score (ordinal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10


@dataclass(frozen=True)
class Noul:
    """Yes/no proposition. The answer is a probability, with no confidence value."""

    instructions: str
    criteria: dict[str, str] | None = None

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            body["criteria"] = self.criteria
        return body


@dataclass(frozen=True)
class Choice:
    """Categorical selection. ``criteria`` maps each option to a description or None."""

    instructions: str
    criteria: dict[str, str | None]

    def __post_init__(self) -> None:
        if not 2 <= len(self.criteria) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"a choice needs 2..{MAX_CHOICE_OPTIONS} options")

    def payload(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True)
class Score:
    """Ordinal placement on an ordered scale. ``criteria`` lists the levels, low to high."""

    instructions: str
    criteria: list[str]

    def __post_init__(self) -> None:
        if not 2 <= len(self.criteria) <= MAX_SCORE_LEVELS:
            raise ValueError(f"a score needs 2..{MAX_SCORE_LEVELS} levels")

    @property
    def levels(self) -> int:
        return len(self.criteria)

    def payload(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


Question = Noul | Choice | Score


@dataclass(frozen=True)
class NoulAnswer:
    value: float
    confidence: float | None = None


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float | None = None


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float | None = None

    def normalised(self, levels: int) -> float:
        """Map the raw ordinal score onto 0..1 so it can be fused with other signals."""
        return self.score / (levels - 1) if levels > 1 else 0.0


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


def parse_answer(body: dict[str, Any]) -> Answer:
    kind = body.get("type")
    if kind == "noul":
        return NoulAnswer(value=float(body["noul"]))
    if kind == "choice":
        return ChoiceAnswer(
            choice=body["choice"],
            probabilities={k: float(v) for k, v in (body.get("probabilities") or {}).items()},
            confidence=body.get("confidence"),
        )
    if kind == "score":
        return ScoreAnswer(
            score=float(body["score"]),
            legend=body.get("legend") or {},
            probabilities={k: float(v) for k, v in (body.get("probabilities") or {}).items()},
            confidence=body.get("confidence"),
        )
    raise ValueError(f"unknown Jev answer type: {kind!r}")
