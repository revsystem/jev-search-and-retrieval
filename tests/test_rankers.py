"""Every pipeline under comparison, exercised against a scripted Jev."""

from jev_rag.jev.client import JevClient
from jev_rag.jev.transport import FakeTransport
from jev_rag.rankers import (
    CohereRerankRanker,
    EmbeddingRanker,
    HybridRanker,
    JevCrossEncodeRanker,
    JevPairwiseRanker,
    JevPointwiseRanker,
    build_rankers,
)
from jev_rag.types import RetrievedDoc


def docs() -> list[RetrievedDoc]:
    return [
        RetrievedDoc(doc_id="decoy", text="話題は近いが答えでない", score=0.0, label=0),
        RetrievedDoc(doc_id="answer", text="答えそのもの", score=0.0, label=1),
    ]


class StubEmbedder:
    """Scores by lexical overlap so the stand-in has a defensible ordering."""

    def __init__(self, scores: dict[str, list[float]]) -> None:
        self.scores = scores

    def embed_query(self, text):
        return self.scores["__query__"]

    def embed_documents(self, texts):
        return [self.scores[t] for t in texts]


def test_the_embedding_ranker_sorts_by_cosine_similarity():
    embedder = StubEmbedder(
        {"__query__": [1.0, 0.0], "話題は近いが答えでない": [0.2, 1.0], "答えそのもの": [1.0, 0.1]}
    )
    ranked = EmbeddingRanker(embedder).rank("問い", docs())
    assert [d.doc_id for d in ranked] == ["answer", "decoy"]
    assert ranked[0].vector_score == ranked[0].score


def test_the_embedding_ranker_batches_the_candidates_once():
    seen = []

    class Counting(StubEmbedder):
        def embed_documents(self, texts):
            seen.append(len(texts))
            return super().embed_documents(texts)

    embedder = Counting(
        {"__query__": [1.0], "話題は近いが答えでない": [0.5], "答えそのもの": [1.0]}
    )
    EmbeddingRanker(embedder).rank("問い", docs())
    assert seen == [2]


def test_the_cohere_reranker_uses_the_scores_the_service_returns():
    class StubRerank:
        def rerank(self, query, documents, top_k):
            order = sorted(documents, key=lambda d: d.text == "答えそのもの", reverse=True)
            for rank, doc in enumerate(order):
                doc.score = 1.0 - rank * 0.5
            return order

    ranked = CohereRerankRanker(StubRerank()).rank("問い", docs())
    assert [d.doc_id for d in ranked] == ["answer", "decoy"]


def test_the_pointwise_jev_ranker_sorts_on_the_returned_probability():
    transport = FakeTransport(nouls={"decoy": 0.05, "answer": 0.93})
    ranked = JevPointwiseRanker(JevClient(transport)).rank("問い", docs())
    assert [d.doc_id for d in ranked] == ["answer", "decoy"]
    assert ranked[0].score == 0.93


def test_the_pointwise_ranker_puts_every_candidate_in_one_shared_state():
    transport = FakeTransport(nouls={"decoy": 0.1, "answer": 0.9})
    candidates = docs()
    JevPointwiseRanker(JevClient(transport)).rank("問い", candidates)
    state = transport.requests[0]["state"]["candidates"]
    assert [entry["text"] for entry in state] == [doc.text for doc in candidates]


def test_the_cross_encoder_sends_one_request_per_candidate():
    transport = FakeTransport(nouls={"relevance": 0.5})
    JevCrossEncodeRanker(JevClient(transport)).rank("問い", docs())
    assert len(transport.requests) == 2
    assert all(set(r["state"]) == {"query", "candidate"} for r in transport.requests)


def test_the_pairwise_ranker_compares_candidates_against_each_other():
    # the single pair is (decoy, answer), so option "b" is the answer
    transport = FakeTransport(choices={"winner": "b"}, confidence=0.9)
    ranked = JevPairwiseRanker(JevClient(transport)).rank("問い", docs())
    assert [d.doc_id for d in ranked] == ["answer", "decoy"]


def test_the_hybrid_ranker_fuses_the_embedding_order_with_the_jev_judgement():
    embedder = StubEmbedder(
        {"__query__": [1.0, 0.0], "話題は近いが答えでない": [1.0, 0.0], "答えそのもの": [0.0, 1.0]}
    )
    transport = FakeTransport(nouls={"decoy": 0.02, "answer": 0.97})
    hybrid = HybridRanker(EmbeddingRanker(embedder), JevPointwiseRanker(JevClient(transport)), 0.3)
    assert [d.doc_id for d in hybrid.rank("問い", docs())] == ["answer", "decoy"]


def test_a_full_vector_weight_leaves_the_embedding_order_alone():
    embedder = StubEmbedder(
        {"__query__": [1.0, 0.0], "話題は近いが答えでない": [1.0, 0.0], "答えそのもの": [0.0, 1.0]}
    )
    transport = FakeTransport(nouls={"decoy": 0.02, "answer": 0.97})
    hybrid = HybridRanker(EmbeddingRanker(embedder), JevPointwiseRanker(JevClient(transport)), 1.0)
    assert [d.doc_id for d in hybrid.rank("問い", docs())] == ["decoy", "answer"]


def test_every_ranker_reports_a_name():
    transport = FakeTransport()
    assert JevPointwiseRanker(JevClient(transport)).name == "jev_pointwise"
    assert JevCrossEncodeRanker(JevClient(transport)).name == "jev_crossencode"


def test_the_registry_offers_one_ranker_per_comparison_row():
    client = JevClient(FakeTransport())
    names = set(build_rankers(["jev_pointwise", "jev_crossencode"], client=client))
    assert names == {"jev_pointwise", "jev_crossencode"}


def test_the_registry_rejects_an_unknown_name():
    import pytest

    with pytest.raises(SystemExit, match="unknown"):
        build_rankers(["nope"], client=JevClient(FakeTransport()))
