"""Retrieval pipelines under comparison.

``BaselinePipeline``       vector search only — the classic RAG shape.
``ClassicRerankPipeline``  vector search, then a dedicated cross-encoder reranker.
``JevPipeline``            vector search, then Jev relevance filtering and scoring,
                           optionally with a Jev-planned metadata filter.
"""

from __future__ import annotations

from typing import Any, Protocol

from jev_rag.evaluation import RetrievedDoc
from jev_rag.jev.client import JevClient
from jev_rag.jev.enrich import QueryPlanner
from jev_rag.jev.rerank import JevReranker, min_max_normalise


class Retriever(Protocol):
    def search(
        self, query: str, top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[RetrievedDoc]: ...


class CrossEncoder(Protocol):
    def rerank(self, query: str, docs: list[RetrievedDoc], top_k: int) -> list[RetrievedDoc]: ...


class BaselinePipeline:
    name = "baseline"

    def __init__(self, retriever: Retriever, top_k: int = 5) -> None:
        self.retriever = retriever
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        return self.retriever.search(query, top_k=self.top_k)


class ClassicRerankPipeline:
    name = "classic_rerank"

    def __init__(
        self, retriever: Retriever, reranker: CrossEncoder, top_k: int = 5, candidate_k: int = 30
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.candidate_k = candidate_k

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        candidates = self.retriever.search(query, top_k=self.candidate_k)
        return self.reranker.rerank(query, candidates, top_k=self.top_k)


class JevPipeline:
    name = "jev"

    def __init__(
        self,
        retriever: Retriever,
        client: JevClient,
        top_k: int = 5,
        candidate_k: int = 30,
        vector_weight: float = 0.3,
        relevance_threshold: float = 0.2,
        filter_first: bool = True,
        plan_query: bool = False,
    ) -> None:
        self.retriever = retriever
        self.reranker = JevReranker(client, vector_weight=vector_weight)
        self.planner = QueryPlanner(client)
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.relevance_threshold = relevance_threshold
        self.filter_first = filter_first
        self.plan_query = plan_query
        self.last_plan = None

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        metadata_filter = None
        if self.plan_query:
            self.last_plan = self.planner.plan(query)
            metadata_filter = self.last_plan.metadata_filter()

        candidates = self.retriever.search(
            query, top_k=self.candidate_k, metadata_filter=metadata_filter
        )
        if not candidates:
            return []

        # Normalise once, over the unfiltered candidate set, so that filtering
        # cannot distort the weight the vector signal carries into the fusion.
        normalised = dict(
            zip(
                [doc.chunk_id for doc in candidates],
                min_max_normalise([doc.score for doc in candidates]),
                strict=True,
            )
        )
        if self.filter_first:
            candidates = self.reranker.relevance_filter(
                query, candidates, threshold=self.relevance_threshold
            )
        return self.reranker.rerank(query, candidates, top_k=self.top_k, vector_scores=normalised)
