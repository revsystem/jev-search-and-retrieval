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

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from jev_rag.jev.crossencode import GRADE, RELEVANCE
from jev_rag.jev.prompts import (
    NOUL_INSTRUCTION,
    PLAIN_NOUL_INSTRUCTION,
    RELEVANCE_CRITERIA,
    RELEVANCE_LEVELS,
)
from jev_rag.jev.state import candidate_path, candidate_state
from jev_rag.types import RetrievedDoc
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


def _probability(answer: dict[str, Any] | None) -> float:
    return float(answer["probability"]) if answer and answer["type"] == "predicate" else 0.0


class DecisionsPointwiseRanker:
    """jev_pointwise on the Decisions API: ten candidates per request, one predicate each.

    Questions are named by position (`c0`, `c1`, ...) rather than by document id,
    so a dataset's id format can never collide with the API's naming rules.
    """

    DEFAULT_BATCH_SIZE = 10

    def __init__(
        self,
        client: DecisionsClient,
        batch_size: int = DEFAULT_BATCH_SIZE,
        plain: bool = False,
        max_workers: int = 8,
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self.plain = plain
        self.max_workers = max_workers
        base = "decisions_pointwise_plain" if plain else "decisions_pointwise"
        self.name = base if batch_size == self.DEFAULT_BATCH_SIZE else f"{base}@{batch_size}"

    @property
    def meter(self) -> UsageMeter:
        return self.client.meter

    def _instruction(self, index: int) -> str:
        if self.plain:
            return PLAIN_NOUL_INSTRUCTION.format(path=candidate_path(index))
        return with_criteria(NOUL_INSTRUCTION.format(path=candidate_path(index)))

    def _judge(self, query: str, group: list[RetrievedDoc]) -> dict[str, float]:
        state, position = candidate_state(query, group)
        names = {doc.doc_id: f"c{position[doc.doc_id]}" for doc in group}
        questions = [
            predicate(names[doc.doc_id], self._instruction(position[doc.doc_id])) for doc in group
        ]
        answers = self.client.evaluate(json.dumps(state, ensure_ascii=False), questions)
        return {doc.doc_id: _probability(answers.get(names[doc.doc_id])) for doc in group}

    def rank(self, question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if not docs:
            return []
        groups = [docs[i : i + self.batch_size] for i in range(0, len(docs), self.batch_size)]
        scores: dict[str, float] = {}
        # the batches are independent, as in JevReranker.noul_rerank
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for result in pool.map(lambda group: self._judge(question, group), groups):
                scores.update(result)
        ranked = [replace(doc, score=scores[doc.doc_id]) for doc in docs]
        ranked.sort(key=lambda doc: doc.score, reverse=True)
        return ranked


class DecisionsCrossEncodeRanker:
    """jev_crossencode on the Decisions API: one request per candidate.

    The predicate orders the candidates and the four-level score breaks ties,
    as in jev_rag.jev.crossencode.CrossEncoder.
    """

    name = "decisions_crossencode"

    def __init__(self, client: DecisionsClient, max_workers: int = 8) -> None:
        self.client = client
        self.max_workers = max_workers

    @property
    def meter(self) -> UsageMeter:
        return self.client.meter

    def _judge(self, query: str, doc: RetrievedDoc) -> tuple[float, float | None]:
        questions = [
            predicate("relevance", with_criteria(RELEVANCE)),
            {
                "type": "score",
                "name": "grade",
                "instructions": GRADE,
                "levels": [{"label": level} for level in RELEVANCE_LEVELS],
            },
        ]
        answers = self.client.evaluate(
            json.dumps({"query": query, "candidate": doc.text}, ensure_ascii=False), questions
        )
        grade = answers.get("grade")
        return (
            _probability(answers.get("relevance")),
            float(grade["score"]) if grade and grade["type"] == "score" else None,
        )

    def rank(self, question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if not docs:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            judgements = list(pool.map(lambda doc: self._judge(question, doc), docs))
        ranked = [
            replace(doc, score=relevance, jev_score=grade)
            for doc, (relevance, grade) in zip(docs, judgements, strict=True)
        ]
        ranked.sort(key=lambda doc: (doc.score, doc.jev_score or 0.0), reverse=True)
        return ranked
