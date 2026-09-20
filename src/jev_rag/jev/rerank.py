"""Jev-based relevance filtering and reranking.

Two judgements sit between retrieval and answer generation:

``relevance_filter``
    A Noul per candidate. Drops documents that are merely topical, so the
    generator is not handed plausible-looking non-evidence.
``rerank``
    A Score per candidate on an ordinal evidence scale. The fractional score
    gives a fine-grained order, which is then fused with the vector score.

Fusing rather than replacing the vector score is deliberate: published
measurements report Jev helping on top of semantic candidates (+0.06..+0.09
nDCG@10) rather than beating a dedicated cross-encoder on its own.
"""

from __future__ import annotations

from dataclasses import replace

from jev_rag.jev.client import JevClient
from jev_rag.jev.prompts import (
    NOUL_INSTRUCTION,
    RELEVANCE_CRITERIA,
    RELEVANCE_LEVELS,
    SCORE_INSTRUCTION,
)
from jev_rag.jev.questions import Noul, NoulAnswer, Score, ScoreAnswer
from jev_rag.types import RetrievedDoc


def min_max_normalise(values: list[float]) -> list[float]:
    """Map scores onto 0..1. A flat vector carries no ranking signal, so it maps to 1."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [1.0] * len(values)
    return [round((value - low) / (high - low), 10) for value in values]


def fuse_scores(vector: list[float], jev: list[float], vector_weight: float) -> list[float]:
    return [vector_weight * v + (1 - vector_weight) * j for v, j in zip(vector, jev, strict=True)]


class JevReranker:
    def __init__(
        self,
        client: JevClient,
        vector_weight: float = 0.3,
        levels: list[str] | None = None,
    ) -> None:
        self.client = client
        self.vector_weight = vector_weight
        self.levels = levels or RELEVANCE_LEVELS

    def _state(self, query: str, docs: list[RetrievedDoc]) -> dict[str, object]:
        return {"query": query, "candidates": {doc.chunk_id: doc.text for doc in docs}}

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        top_k: int | None = None,
        vector_scores: dict[str, float] | None = None,
    ) -> list[RetrievedDoc]:
        """Rescore candidates with Jev and fuse with the vector signal.

        ``vector_scores`` supplies already-normalised vector scores. Pass it
        whenever the candidate list has been filtered: min-max over a shrunken
        list stretches a 0.01 similarity gap into the full 0..1 range and lets
        the vector signal dominate the fusion it is only meant to temper.
        """
        if not docs:
            return []

        questions = {
            doc.chunk_id: Score(SCORE_INSTRUCTION.format(key=doc.chunk_id), criteria=self.levels)
            for doc in docs
        }
        answers = self.client.evaluate(self._state(query, docs), questions)

        raw = [_as_score(answers.get(doc.chunk_id)) for doc in docs]
        jev_norm = [value / (len(self.levels) - 1) for value in raw]
        vector_norm = (
            [vector_scores[doc.chunk_id] for doc in docs]
            if vector_scores is not None
            else min_max_normalise([doc.score for doc in docs])
        )
        fused = fuse_scores(vector_norm, jev_norm, self.vector_weight)

        original_rank = {doc.chunk_id: index for index, doc in enumerate(docs)}
        scored = [
            replace(doc, score=score, vector_score=doc.score, jev_score=jev)
            for doc, score, jev in zip(docs, fused, raw, strict=True)
        ]
        scored.sort(key=lambda doc: doc.score, reverse=True)
        for new_rank, doc in enumerate(scored):
            doc.rank_delta = original_rank[doc.chunk_id] - new_rank
        return scored[:top_k] if top_k else scored

    def noul_rerank(
        self, query: str, docs: list[RetrievedDoc], top_k: int | None = None
    ) -> list[RetrievedDoc]:
        """Score query-to-candidate relevance and sort on the probability itself.

        The documented pattern: a shortlist from embeddings or BM25, one Noul
        per query-candidate pair, and a sort on the returned value. No
        generative model is asked to invent a scale, and the vector score is
        deliberately not fused in — use ``rerank`` when you want that.
        """
        if not docs:
            return []

        questions = {
            doc.chunk_id: Noul(
                NOUL_INSTRUCTION.format(key=doc.chunk_id), criteria=RELEVANCE_CRITERIA
            )
            for doc in docs
        }
        answers = self.client.evaluate(self._state(query, docs), questions)

        original = {doc.chunk_id: position for position, doc in enumerate(docs)}
        ranked = [
            replace(
                doc,
                score=_as_noul(answers.get(doc.chunk_id)),
                vector_score=doc.score,
                jev_relevance=_as_noul(answers.get(doc.chunk_id)),
            )
            for doc in docs
        ]
        ranked.sort(key=lambda doc: doc.score, reverse=True)
        for position, doc in enumerate(ranked):
            doc.rank_delta = original[doc.chunk_id] - position
        return ranked[:top_k] if top_k else ranked

    def relevance_filter(
        self, query: str, docs: list[RetrievedDoc], threshold: float = 0.2, keep_min: int = 1
    ) -> list[RetrievedDoc]:
        """Keep only usable evidence, ordered by relevance, never returning nothing."""
        if not docs:
            return []

        questions = {
            doc.chunk_id: Noul(
                NOUL_INSTRUCTION.format(key=doc.chunk_id), criteria=RELEVANCE_CRITERIA
            )
            for doc in docs
        }
        answers = self.client.evaluate(self._state(query, docs), questions)

        judged = [
            replace(doc, jev_relevance=_as_noul(answers.get(doc.chunk_id)), vector_score=doc.score)
            for doc in docs
        ]
        judged.sort(key=lambda doc: doc.jev_relevance or 0.0, reverse=True)
        kept = [doc for doc in judged if (doc.jev_relevance or 0.0) >= threshold]
        return kept or judged[:keep_min]


def _as_score(answer: object) -> float:
    return answer.score if isinstance(answer, ScoreAnswer) else 0.0


def _as_noul(answer: object) -> float:
    return answer.value if isinstance(answer, NoulAnswer) else 0.0
