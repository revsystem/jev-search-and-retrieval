"""Offline demonstration of the comparison harness.

No credentials and no network. The embedding model is replaced by a character
bigram vectoriser and Jev's answers come from a hand-written fixture, so the
numbers below say that the wiring works — they are not evidence about Jev.
Real measurements need ``jev-rag evaluate`` against Bedrock and a Jev route.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jev_rag.documents import Chunk
from jev_rag.evaluation import (
    EvalQuery,
    build_qrels,
    compare,
    format_table,
    mean_metrics,
    score_ranking,
)
from jev_rag.jev.client import JevClient
from jev_rag.jev.context import ContextSelector
from jev_rag.jev.crossencode import CrossEncoder
from jev_rag.jev.enrich import ChunkEnricher
from jev_rag.jev.pairwise import PairwiseReranker
from jev_rag.jev.rerank import JevReranker
from jev_rag.jev.retrieval import HybridRetriever, JevRetriever
from jev_rag.pipelines import BaselinePipeline, JevPipeline, RerankPipeline
from jev_rag.vector_store import EmbeddingRetriever, InMemoryVectorStore

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "demo" / "offline.yaml"
ENRICH_KEYS = {"topic", "content_type", "self_contained", "info_density"}


class LexicalEmbedder:
    """Character-bigram vectoriser.

    It stands in for Cohere Embed v4 and shares its most relevant weakness:
    a passage that repeats the query's wording outranks a passage that answers
    it. That is the failure the reranking comparison is about.
    """

    def __init__(self, corpus: list[str]) -> None:
        vocabulary = sorted({bigram for text in corpus for bigram in self._bigrams(text)})
        self.index = {bigram: position for position, bigram in enumerate(vocabulary)}

    @staticmethod
    def _bigrams(text: str) -> list[str]:
        cleaned = "".join(text.split())
        return [cleaned[i : i + 2] for i in range(max(len(cleaned) - 1, 0))]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * len(self.index)
        for bigram in self._bigrams(text):
            position = self.index.get(bigram)
            if position is not None:
                vector[position] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@dataclass
class FixtureTransport:
    """Replays hand-written System One answers for the demo corpus."""

    fixture: dict[str, Any]
    model: str = "jev-fixture"
    requests: list[dict[str, Any]] = field(default_factory=list)

    def build_body(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return {"model": self.model, "state": state, "questions": questions}

    def send(self, body: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(body)
        state, questions = body["state"], body["questions"]
        if set(questions) <= ENRICH_KEYS:
            answers = self._enrichment(state, questions)
        elif "single_topic" in questions:
            answers = self._plan(state, questions)
        elif "candidate" in state:
            answers = self._cross_encode(state, questions)
        elif any(key.endswith(("__usefulness", "__redundant")) for key in questions):
            answers = self._context(state, questions)
        elif all(q["type"] == "choice" for q in questions.values()):
            answers = self._pairwise(state, questions)
        else:
            answers = self._relevance(state, questions)
        return {"model": self.model, "answers": answers, "usage": {"input_tokens": 0}}

    def _query_entry(self, question: str) -> dict[str, Any]:
        for entry in self.fixture["queries"]:
            if entry["question"] == question:
                return entry
        raise KeyError(f"no fixture for query {question!r}")

    def _chunk_entry(self, text: str) -> dict[str, Any]:
        for entry in self.fixture["corpus"]:
            if entry["text"] in text:
                return entry
        raise KeyError("no fixture for the enriched chunk")

    def _enrichment(self, state: dict, questions: dict) -> dict[str, Any]:
        entry = self._chunk_entry(state["text"])
        digits = any(character.isdigit() for character in entry["text"])
        answers: dict[str, Any] = {}
        if "topic" in questions:
            answers["topic"] = _choice(entry["topic"], questions["topic"]["criteria"], 0.85)
        if "content_type" in questions:
            kind = "statistic" if digits else "narrative"
            answers["content_type"] = _choice(kind, questions["content_type"]["criteria"], 0.8)
        if "self_contained" in questions:
            dependent = entry["text"].startswith("本節") or entry["text"].startswith("本項")
            answers["self_contained"] = {"type": "noul", "noul": 0.1 if dependent else 0.9}
        if "info_density" in questions:
            answers["info_density"] = _score(
                3.0 if digits else 1.0, questions["info_density"]["criteria"], 0.8
            )
        return answers

    def _plan(self, state: dict, questions: dict) -> dict[str, Any]:
        entry = self._query_entry(state["query"])
        return {
            "topic": _choice(entry["topic"], questions["topic"]["criteria"], 0.88),
            "single_topic": {"type": "noul", "noul": 0.9},
        }

    def _cross_encode(self, state: dict, questions: dict) -> dict[str, Any]:
        entry = self._query_entry(state["query"])["jev"]
        chunk = self._chunk_entry(state["candidate"])["chunk_id"]
        answers: dict[str, Any] = {}
        if "relevance" in questions:
            answers["relevance"] = {
                "type": "noul",
                "noul": entry["relevance"].get(chunk, 0.05),
            }
        if "grade" in questions:
            answers["grade"] = _score(
                entry["score"].get(chunk, 0.2), questions["grade"]["criteria"], 0.8
            )
        return answers

    def _context(self, state: dict, questions: dict) -> dict[str, Any]:
        entry = self._query_entry(state["query"])["jev"]
        redundant = entry.get("redundant") or {}
        answers: dict[str, Any] = {}
        for key, question in questions.items():
            chunk, _, kind = key.rpartition("__")
            if kind == "usefulness":
                answers[key] = _score(entry["score"].get(chunk, 0.2), question["criteria"], 0.8)
            else:
                answers[key] = {"type": "noul", "noul": redundant.get(chunk, 0.05)}
        return answers

    def _pairwise(self, state: dict, questions: dict) -> dict[str, Any]:
        entry = self._query_entry(state["query"])["jev"]
        answers: dict[str, Any] = {}
        for key, question in questions.items():
            left, right = list(question["criteria"])
            left_p, right_p = _pair_probabilities(
                entry["relevance"].get(left, 0.05), entry["relevance"].get(right, 0.05)
            )
            winner, confidence = (left, left_p) if left_p >= right_p else (right, right_p)
            answers[key] = {
                "type": "choice",
                "choice": winner,
                "probabilities": {left: left_p, right: right_p},
                "confidence": confidence,
            }
        return answers

    def _relevance(self, state: dict, questions: dict) -> dict[str, Any]:
        entry = self._query_entry(state["query"])["jev"]
        answers: dict[str, Any] = {}
        for key, question in questions.items():
            if question["type"] == "noul":
                answers[key] = {"type": "noul", "noul": entry["relevance"].get(key, 0.05)}
            else:
                answers[key] = _score(entry["score"].get(key, 0.2), question["criteria"], 0.8)
        return answers


def _pair_probabilities(left_value: float, right_value: float) -> tuple[float, float]:
    """Turn two pointwise relevance values into a calibrated pairwise split."""
    total = left_value + right_value
    if total <= 0:
        return 0.5, 0.5
    return left_value / total, right_value / total


def _choice(winner: str, options: dict[str, Any], confidence: float) -> dict[str, Any]:
    spread = (1 - confidence) / max(len(options) - 1, 1)
    return {
        "type": "choice",
        "choice": winner,
        "probabilities": {o: (confidence if o == winner else spread) for o in options},
        "confidence": confidence,
    }


def _score(value: float, levels: list[str], confidence: float) -> dict[str, Any]:
    return {
        "type": "score",
        "score": value,
        "legend": {str(i): label for i, label in enumerate(levels)},
        "probabilities": {str(i): 1 / len(levels) for i in range(len(levels))},
        "confidence": confidence,
    }


def load_fixture(path: str | Path = FIXTURE) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _build_store(embedder: LexicalEmbedder, keys, texts, metadata) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.put(list(keys), embedder.embed_documents(list(texts)), list(metadata))
    return store


def _pipelines(chunks, client, plain, enriched_retriever, top_k):
    """One pipeline per use case on the official search and retrieval map."""
    candidate_k = len(chunks)
    reranker = JevReranker(client)
    pairwise = PairwiseReranker(client)
    cross = CrossEncoder(client, with_grade=True)

    def staged(name, rerank):
        return RerankPipeline(name, plain, rerank, top_k=top_k, candidate_k=candidate_k)

    return {
        # the classic pipeline the others are measured against
        "baseline": BaselinePipeline(plain, top_k=top_k),
        # replace embeddings
        "jev_only": BaselinePipeline(JevRetriever(client, chunks), top_k=top_k, name="jev_only"),
        # supplement embeddings
        "jev_hybrid": BaselinePipeline(
            HybridRetriever(plain, client, candidate_k=candidate_k),
            top_k=top_k,
            name="jev_hybrid",
        ),
        # score query-to-candidate relevance
        "jev_noul": staged("jev_noul", reranker.noul_rerank),
        # rerank with pairwise comparisons
        "jev_pairwise": staged("jev_pairwise", pairwise.rerank),
        # cross-encode query and candidate
        "jev_crossencode": staged("jev_crossencode", cross.rerank),
        # enriched embeddings, planned metadata filter, filter then score fusion
        "jev_full": JevPipeline(
            enriched_retriever,
            client,
            top_k=top_k,
            candidate_k=candidate_k,
            plan_query=True,
        ),
    }


def _context_report(client, pipeline, queries, budget_chars: int) -> list[str]:
    """Use case: select useful context for downstream AI workflows."""
    selector = ContextSelector(client, budget_chars=budget_chars)
    lines = [
        f"文脈選択（予算 {budget_chars} 文字、入力は jev_noul の上位6件）:",
        "| query | 候補 | 採用 | 文字数 | 除外理由 |",
        "|---|---|---|---|---|",
    ]
    for query in queries:
        candidates = pipeline.retrieve(query.question)[:6]
        selection = selector.select(query.question, candidates)
        dropped = ", ".join(f"{cid}:{why}" for cid, why in sorted(selection.reasons.items()))
        before = sum(len(doc.text) for doc in candidates)
        lines.append(
            f"| {query.query_id} | {len(candidates)} | "
            f"{len(selection.selected)} | {before} -> {selection.used_chars} | {dropped or '-'} |"
        )
    return lines


def run(path: str | Path = FIXTURE, top_k: int = 5, budget_chars: int = 200) -> str:
    fixture = load_fixture(path)
    chunks = [
        Chunk(chunk_id=e["chunk_id"], page=e["page"], section=e["section"], text=e["text"])
        for e in fixture["corpus"]
    ]
    queries = [
        EvalQuery(query_id=e["query_id"], question=e["question"], qrels=e["qrels"])
        for e in fixture["queries"]
    ]

    client = JevClient(FixtureTransport(fixture))
    enriched = ChunkEnricher(client).enrich(chunks)

    embedder = LexicalEmbedder(
        [c.embedding_text() for c in chunks] + [e.embedding_text() for e in enriched]
    )
    plain = EmbeddingRetriever(
        embedder,
        _build_store(
            embedder,
            [c.chunk_id for c in chunks],
            [c.embedding_text() for c in chunks],
            [{**c.metadata(), "text": c.text} for c in chunks],
        ),
    )
    enriched_retriever = EmbeddingRetriever(
        embedder,
        _build_store(
            embedder,
            [e.chunk.chunk_id for e in enriched],
            [e.embedding_text() for e in enriched],
            [{**e.metadata(), "text": e.chunk.text} for e in enriched],
        ),
    )

    pipelines = _pipelines(chunks, client, plain, enriched_retriever, top_k)
    results: dict[str, dict[str, float]] = {}
    traces: list[str] = []
    for name, pipeline in pipelines.items():
        per_query = []
        for query in queries:
            ranked = pipeline.retrieve(query.question)
            per_query.append(score_ranking(ranked, build_qrels(query, chunks), k=top_k))
            if name in ("baseline", "jev_noul"):
                order = " > ".join(doc.chunk_id for doc in ranked[:3])
                traces.append(f"  {query.query_id} {name:<16} {order}")
        results[name] = mean_metrics(per_query)

    metrics = [
        f"ndcg@{top_k}",
        f"strict_ndcg@{top_k}",
        f"recall@{top_k}",
        f"precision@{top_k}",
        "mrr",
    ]
    return "\n".join(
        [
            f"オフラインデモ（合成コーパス{len(chunks)}件 / クエリ{len(queries)}件）",
            "Jevの回答は台本化したフィクスチャ。ハーネスの動作確認であり、Jevの性能評価ではない。",
            "",
            format_table(compare(results, baseline="baseline"), metrics),
            "",
            "上位3件の並び:",
            *sorted(traces),
            "",
            *_context_report(client, pipelines["jev_noul"], queries, budget_chars),
        ]
    )
