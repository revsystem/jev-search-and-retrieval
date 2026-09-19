"""Use case: rerank results with pairwise comparisons.

A pairwise Choice asks a strictly easier question than an absolute score —
"which of these two answers the query better" needs no shared scale across
requests. The cost is that comparisons multiply, so the schedule below runs
round robin only while the whole tournament fits in one request, and otherwise
gives every candidate a fixed number of opponents. Because Jev answers every
question in a request in parallel, a whole round costs about one question.
"""

from __future__ import annotations

from dataclasses import replace

from jev_rag.evaluation import RetrievedDoc
from jev_rag.jev.client import MAX_QUESTIONS_PER_REQUEST, JevClient
from jev_rag.jev.questions import Choice, ChoiceAnswer

INSTRUCTION = (
    "state.query により良く答えるのはどちらの候補ですか。"
    'state.candidates["{left}"] と state.candidates["{right}"] を比較してください。'
)


def pair_schedule(ids: list[str], opponents: int = 3, limit: int = MAX_QUESTIONS_PER_REQUEST):
    """Comparisons to run, as ordered (left, right) pairs."""
    count = len(ids)
    if count < 2:
        return []
    if count * (count - 1) // 2 <= limit:
        return [(ids[i], ids[j]) for i in range(count) for j in range(i + 1, count)]

    # Distance-major order: every candidate is compared once before any is
    # compared twice, so truncating at the limit still covers the shortlist.
    seen: set[frozenset[str]] = set()
    schedule: list[tuple[str, str]] = []
    for distance in range(1, min(opponents, count - 1) + 1):
        for index in range(count):
            left, right = ids[index], ids[(index + distance) % count]
            key = frozenset((left, right))
            if key in seen:
                continue
            seen.add(key)
            schedule.append((left, right))
            if len(schedule) == limit:
                return schedule
    return schedule


class PairwiseReranker:
    def __init__(self, client: JevClient, opponents: int = 3) -> None:
        self.client = client
        self.opponents = opponents

    def rerank(
        self, query: str, docs: list[RetrievedDoc], top_k: int | None = None
    ) -> list[RetrievedDoc]:
        if len(docs) < 2:
            return list(docs[:top_k]) if top_k else list(docs)

        texts = {doc.chunk_id: doc.text for doc in docs}
        schedule = pair_schedule(list(texts), self.opponents)
        questions = {
            f"p{index}": Choice(
                INSTRUCTION.format(left=left, right=right),
                criteria={
                    left: "こちらの方が良い根拠になる。",
                    right: "こちらの方が良い根拠になる。",
                },
            )
            for index, (left, right) in enumerate(schedule)
        }
        answers = self.client.evaluate({"query": query, "candidates": texts}, questions)

        # A win counts by its probability, so a narrow call moves the ranking
        # less than a decisive one.
        totals = dict.fromkeys(texts, 0.0)
        matches = dict.fromkeys(texts, 0)
        for index, (left, right) in enumerate(schedule):
            answer = answers.get(f"p{index}")
            if not isinstance(answer, ChoiceAnswer):
                continue
            for side in (left, right):
                totals[side] += answer.probabilities.get(side, 0.0)
                matches[side] += 1

        original = {doc.chunk_id: position for position, doc in enumerate(docs)}
        ranked = [
            replace(
                doc,
                score=(
                    totals[doc.chunk_id] / matches[doc.chunk_id] if matches[doc.chunk_id] else 0.0
                ),
                vector_score=doc.score,
            )
            for doc in docs
        ]
        ranked.sort(key=lambda doc: doc.score, reverse=True)
        for position, doc in enumerate(ranked):
            doc.rank_delta = original[doc.chunk_id] - position
        return ranked[:top_k] if top_k else ranked
