"""Use case: replace or supplement embeddings in RAG pipelines.

``JevRetriever`` retrieves with no embedding model in the path at all: the
corpus is swept in batches of Noul questions and ranked on the returned
probabilities. It is bounded by cost, not by index size, so it suits a corpus
that has already been narrowed — a single document, a metadata slice, a
per-user workspace — rather than a whole vector index.

``HybridRetriever`` keeps the embedding index as the recall stage and adds the
Jev judgement as the precision stage, fusing the two signals.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from jev_rag.documents import Chunk
from jev_rag.evaluation import RetrievedDoc
from jev_rag.jev.client import MAX_QUESTIONS_PER_REQUEST, JevClient
from jev_rag.jev.prompts import NOUL_INSTRUCTION, RELEVANCE_CRITERIA
from jev_rag.jev.questions import Noul, NoulAnswer
from jev_rag.jev.rerank import fuse_scores, min_max_normalise
from jev_rag.vector_store import matches_filter


def _judge(client: JevClient, query: str, items: list[tuple[str, str]]) -> dict[str, float]:
    """One Noul per candidate, against a state holding just this batch."""
    state = {"query": query, "candidates": dict(items)}
    questions = {
        key: Noul(NOUL_INSTRUCTION.format(key=key), criteria=RELEVANCE_CRITERIA) for key, _ in items
    }
    answers = client.evaluate(state, questions)
    return {
        key: (answers[key].value if isinstance(answers.get(key), NoulAnswer) else 0.0)
        for key, _ in items
    }


class JevRetriever:
    """Retrieval without an embedding model."""

    def __init__(
        self,
        client: JevClient,
        corpus: list[Chunk],
        metadata: dict[str, dict[str, Any]] | None = None,
        batch_size: int = MAX_QUESTIONS_PER_REQUEST,
        max_workers: int = 8,
    ) -> None:
        self.client = client
        self.corpus = corpus
        self.metadata = metadata or {}
        self.batch_size = min(batch_size, MAX_QUESTIONS_PER_REQUEST)
        self.max_workers = max_workers

    def search(
        self, query: str, top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[RetrievedDoc]:
        pool = [
            chunk
            for chunk in self.corpus
            if matches_filter(self.metadata.get(chunk.chunk_id, chunk.metadata()), metadata_filter)
        ]
        if not pool:
            return []

        items = [(chunk.chunk_id, chunk.text) for chunk in pool]
        batches = [items[i : i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            judged: dict[str, float] = {}
            for result in executor.map(lambda b: _judge(self.client, query, b), batches):
                judged.update(result)

        hits = [
            RetrievedDoc(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=judged.get(chunk.chunk_id, 0.0),
                metadata=self.metadata.get(chunk.chunk_id, chunk.metadata()),
                jev_relevance=judged.get(chunk.chunk_id, 0.0),
            )
            for chunk in pool
        ]
        hits.sort(key=lambda doc: doc.score, reverse=True)
        return hits[:top_k]


class HybridRetriever:
    """Embeddings for recall, Jev for precision, fused into one ranking."""

    def __init__(
        self,
        retriever,
        client: JevClient,
        vector_weight: float = 0.3,
        candidate_k: int = 30,
    ) -> None:
        self.retriever = retriever
        self.client = client
        self.vector_weight = vector_weight
        self.candidate_k = candidate_k

    def search(
        self, query: str, top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[RetrievedDoc]:
        candidates = self.retriever.search(query, self.candidate_k, metadata_filter)
        if not candidates:
            return []

        judged = _judge(self.client, query, [(doc.chunk_id, doc.text) for doc in candidates])
        fused = fuse_scores(
            min_max_normalise([doc.score for doc in candidates]),
            [judged[doc.chunk_id] for doc in candidates],
            self.vector_weight,
        )
        hits = [
            replace(doc, score=score, vector_score=doc.score, jev_relevance=judged[doc.chunk_id])
            for doc, score in zip(candidates, fused, strict=True)
        ]
        hits.sort(key=lambda doc: doc.score, reverse=True)
        return hits[:top_k]
