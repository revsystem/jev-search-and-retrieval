"""OpenAI's Decisions API as a relevance judge, for comparison with Jev.

The Decisions API (`POST /v1/decisions`, public beta, `gpt-6-luna` only)
answers typed questions about one shared input, as Jev does, but two parts of
Jev's request have no counterpart:

- There is no `criteria` field. The written definition of relevance is
  appended to the instruction text instead.
- The input is a string or messages, not structured state, and nothing in the
  documentation says a backticked path such as `candidates[3].text` is
  resolved. The state is sent as JSON text with the same paths; whether the
  model follows them is checked with `jev-rag probe-order` before a full run.

Answers are matched to questions by name. A refusal or an answer that never
came scores 0.0, as Jev's missing answers do, and is counted on the meter so
the report shows how many there were.
"""

from __future__ import annotations

import time
from typing import Any

from jev_rag.jev.prompts import RELEVANCE_CRITERIA
from jev_rag.usage import UsageMeter

DECISIONS_MODEL = "gpt-6-luna"


def with_criteria(instruction: str) -> str:
    """The instruction followed by the definition Jev receives in `criteria`."""
    return (
        f"{instruction}\n判定基準: はい（true）= {RELEVANCE_CRITERIA['true']}"
        f"いいえ（false）= {RELEVANCE_CRITERIA['false']}"
    )


def predicate(name: str, instructions: str) -> dict[str, Any]:
    return {"type": "predicate", "name": name, "instructions": instructions}


class DecisionsClient:
    def __init__(self, sdk: Any, model: str = DECISIONS_MODEL) -> None:
        self.sdk = sdk
        self.model = model
        self.meter = UsageMeter()

    def evaluate(
        self, input_text: str, questions: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Answers by question name; refusals and absent names are left out and counted."""
        started = time.monotonic()
        decision = self.sdk.decisions.create(
            model=self.model, input=input_text, questions=questions
        )
        answers: dict[str, dict[str, Any]] = {}
        refused: set[str | None] = set()
        for answer in decision.answers:
            body = answer.model_dump()
            if body["type"] == "refusal":
                refused.add(body.get("name"))
            else:
                answers[body["name"]] = body
        self.meter.record(
            model=decision.model,
            input_tokens=decision.usage.input_tokens,
            seconds=time.monotonic() - started,
            refusals=len(refused),
            missing=sum(
                1 for q in questions if q["name"] not in answers and q["name"] not in refused
            ),
        )
        return answers
