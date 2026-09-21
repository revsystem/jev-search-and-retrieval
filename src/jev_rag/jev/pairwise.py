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

from jev_rag.jev.client import JevClient
from jev_rag.jev.questions import Choice, ChoiceAnswer
from jev_rag.jev.state import candidate_path, candidate_state
from jev_rag.types import RetrievedDoc

# How many comparisons one reranking is willing to pay for. This is a cost
# decision, not an API limit: TypeSafe caps a request by tokens, not by
# question count. Comparisons grow quadratically, so the schedule stops here
# and gives every candidate a fixed number of opponents instead.
DEFAULT_COMPARISON_BUDGET = 64

INSTRUCTION = "`query` により良く答えるのはどちらの候補ですか。"

# Choice option keys are shown to the model, unlike question ids, so they are
# neutral labels rather than document ids; each one is described by the path
# of the candidate it stands for.
OPTIONS = ("a", "b")
OPTION_DESCRIPTION = "{path} のほうが良い根拠になる。"


def pair_schedule(ids: list[str], opponents: int = 3, limit: int = DEFAULT_COMPARISON_BUDGET):
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
    def __init__(
        self,
        client: JevClient,
        opponents: int = 3,
        comparison_budget: int = DEFAULT_COMPARISON_BUDGET,
    ) -> None:
        self.client = client
        self.opponents = opponents
        self.comparison_budget = comparison_budget

    def rerank(
        self, query: str, docs: list[RetrievedDoc], top_k: int | None = None
    ) -> list[RetrievedDoc]:
        if len(docs) < 2:
            return list(docs[:top_k]) if top_k else list(docs)

        state, position = candidate_state(query, docs)
        schedule = pair_schedule(
            [doc.doc_id for doc in docs], self.opponents, self.comparison_budget
        )
        questions = {
            f"p{index}": Choice(
                INSTRUCTION,
                criteria={
                    option: OPTION_DESCRIPTION.format(path=candidate_path(position[side]))
                    for option, side in zip(OPTIONS, (left, right), strict=True)
                },
            )
            for index, (left, right) in enumerate(schedule)
        }
        answers = self.client.evaluate(state, questions)

        # A win counts by its probability, so a narrow call moves the ranking
        # less than a decisive one.
        totals = {doc.doc_id: 0.0 for doc in docs}
        matches = {doc.doc_id: 0 for doc in docs}
        for index, (left, right) in enumerate(schedule):
            answer = answers.get(f"p{index}")
            if not isinstance(answer, ChoiceAnswer):
                continue
            # the neutral labels map back to documents by their position in the pair
            for option, side in zip(OPTIONS, (left, right), strict=True):
                totals[side] += answer.probabilities.get(option, 0.0)
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
