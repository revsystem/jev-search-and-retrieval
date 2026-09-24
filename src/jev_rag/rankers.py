"""The pipelines under comparison.

Every ranker takes the same input — a question and its candidate passages —
and returns the same candidates in a new order. That uniformity is the point:
the only thing that differs between rows of the comparison table is how the
order was decided.

Two rows are the classic baseline (an embedding model, then a dedicated
cross-encoder reranker). The rest implement TypeSafe's published search and
retrieval use cases; see ``jev_rag.usecases`` for the mapping.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Protocol

from jev_rag.jev.client import JevClient
from jev_rag.jev.crossencode import CrossEncoder
from jev_rag.jev.pairwise import PairwiseReranker
from jev_rag.jev.rerank import JevReranker, fuse_scores, min_max_normalise
from jev_rag.types import RetrievedDoc


class Ranker(Protocol):
    name: str

    def rank(self, question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]: ...


def _sorted(docs: list[RetrievedDoc], original: list[RetrievedDoc]) -> list[RetrievedDoc]:
    positions = {doc.doc_id: index for index, doc in enumerate(original)}
    ranked = sorted(docs, key=lambda d: d.score, reverse=True)
    for index, doc in enumerate(ranked):
        doc.rank_delta = positions[doc.doc_id] - index
    return ranked


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class EmbeddingRanker:
    """The classic baseline: rank candidates by embedding similarity."""

    name = "embedding"

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        # A run evaluates several routes over the same queries, and the hybrid
        # route needs the same vectors the embedding route already paid for.
        self._cache: dict[str, list[float]] = {}

    def _vectors(self, docs: list[RetrievedDoc]) -> list[list[float]]:
        missing = [doc for doc in docs if doc.doc_id not in self._cache]
        if missing:
            fresh = self.embedder.embed_documents([doc.text for doc in missing])
            self._cache.update(
                {doc.doc_id: vector for doc, vector in zip(missing, fresh, strict=True)}
            )
        return [self._cache[doc.doc_id] for doc in docs]

    def rank(self, question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if not docs:
            return []
        query = self.embedder.embed_query(question)
        vectors = self._vectors(docs)
        scored = [
            replace(doc, score=_cosine(query, vector), vector_score=_cosine(query, vector))
            for doc, vector in zip(docs, vectors, strict=True)
        ]
        return _sorted(scored, docs)


class CohereRerankRanker:
    """The classic reranking baseline: a dedicated cross-encoder service."""

    name = "cohere_rerank"

    def __init__(self, reranker) -> None:
        self.reranker = reranker

    def rank(self, question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if not docs:
            return []
        return self.reranker.rerank(question, docs, top_k=len(docs))


class JevPointwiseRanker:
    """Score query-to-candidate relevance: one Noul per candidate.

    Also the "replace embeddings" shape: no embedding model is in this path.
    ``batch_size`` sets how many candidates share one request, and so how many
    distractors each question is read against; a batch of one is
    cross-encoding.
    """

    # Measured on JQaRA (30 queries, 100 candidates): accuracy is flat from 1
    # to 10 candidates per request and falls off a cliff after it (nDCG@10
    # 0.835 at 1, 0.836 at 10, 0.578 at 25, 0.204 at 100). As a judge on the
    # synthetic set, ten of one question's candidates together beat one at a
    # time (PR-AUC 0.912 vs 0.753), unrelated padding is harmless up to 25 and
    # ruinous from 50 (`jev-rag dilution`). Ten is where every curve is good.
    DEFAULT_BATCH_SIZE = 10

    def __init__(
        self,
        client: JevClient,
        batch_size: int | None = DEFAULT_BATCH_SIZE,
        plain: bool = False,
    ) -> None:
        self.reranker = JevReranker(client)
        self.batch_size = batch_size
        # plain drops the written relevance criteria, to separate what the
        # model judges on its own from what the definition we wrote adds
        self.plain = plain
        base = "jev_pointwise_plain" if plain else "jev_pointwise"
        self.name = (
            base
            if batch_size == self.DEFAULT_BATCH_SIZE
            else (f"{base}@all" if batch_size is None else f"{base}@{batch_size}")
        )

    def rank(self, question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if not docs:
            return []
        return self.reranker.noul_rerank(
            question, docs, batch_size=self.batch_size, plain=self.plain
        )


class JevCrossEncodeRanker:
    """Cross-encode: one request per pair, so no candidate colours another.

    This is the shape TypeSafe's own reranking cookbook uses.
    """

    name = "jev_crossencode"

    def __init__(self, client: JevClient, with_grade: bool = False) -> None:
        self.encoder = CrossEncoder(client, with_grade=with_grade)

    def rank(self, question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        return self.encoder.rerank(question, docs) if docs else []


class JevPairwiseRanker:
    """Rerank with pairwise comparisons, ordering by weighted win rate."""

    name = "jev_pairwise"

    def __init__(self, client: JevClient, opponents: int = 3) -> None:
        self.reranker = PairwiseReranker(client, opponents=opponents)

    def rank(self, question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        return self.reranker.rerank(question, docs) if docs else []


class HybridRanker:
    """Supplement embeddings: fuse the vector order with the Jev judgement."""

    name = "jev_hybrid"

    def __init__(self, embedding: Ranker, jev: Ranker, vector_weight: float = 0.3) -> None:
        self.embedding = embedding
        self.jev = jev
        self.vector_weight = vector_weight

    def rank(self, question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if not docs:
            return []
        by_vector = {d.doc_id: d.score for d in self.embedding.rank(question, docs)}
        by_jev = {d.doc_id: d.score for d in self.jev.rank(question, docs)}

        order = [doc.doc_id for doc in docs]
        fused = fuse_scores(
            min_max_normalise([by_vector[i] for i in order]),
            min_max_normalise([by_jev[i] for i in order]),
            self.vector_weight,
        )
        scored = [
            replace(
                doc,
                score=score,
                vector_score=by_vector[doc.doc_id],
                jev_score=by_jev[doc.doc_id],
            )
            for doc, score in zip(docs, fused, strict=True)
        ]
        return _sorted(scored, docs)


def build_rankers(names: list[str], client: JevClient | None = None, settings=None) -> dict:
    """Instantiate the named rankers, creating only the clients they need."""

    shared: dict[str, Ranker] = {}

    def embedding():
        from jev_rag.bedrock import CohereEmbedder

        if "embedding" not in shared:
            shared["embedding"] = EmbeddingRanker(CohereEmbedder(settings))
        return shared["embedding"]

    def cohere_rerank():
        from jev_rag.bedrock import BedrockReranker

        return CohereRerankRanker(BedrockReranker(settings))

    available = {
        "embedding": embedding,
        "cohere_rerank": cohere_rerank,
        "jev_pointwise": lambda: JevPointwiseRanker(client),
        "jev_pointwise_plain": lambda: JevPointwiseRanker(client, plain=True),
        "jev_crossencode": lambda: JevCrossEncodeRanker(client, with_grade=True),
        "jev_pairwise": lambda: JevPairwiseRanker(client),
        "jev_hybrid": lambda: HybridRanker(embedding(), JevPointwiseRanker(client)),
    }
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(f"unknown ranker(s): {unknown}; available: {sorted(available)}")
    return {name: available[name]() for name in names}


ALL_RANKERS = [
    "embedding",
    "cohere_rerank",
    "jev_pointwise",
    "jev_crossencode",
    "jev_pairwise",
    "jev_hybrid",
]
