"""Use case: replace or supplement embeddings in RAG pipelines."""

from jev_rag.documents import Chunk
from jev_rag.evaluation import RetrievedDoc
from jev_rag.jev.client import JevClient
from jev_rag.jev.retrieval import HybridRetriever, JevRetriever
from jev_rag.jev.transport import FakeTransport


def corpus() -> list[Chunk]:
    return [
        Chunk(chunk_id="c1", page=1, section="", text="生成AIの利用経験は26.7%であった。"),
        Chunk(chunk_id="c2", page=1, section="", text="本節では動向を整理する。"),
        Chunk(chunk_id="c3", page=2, section="", text="トラヒックは38.0Tbpsに達した。"),
    ]


class StubEmbeddingRetriever:
    def __init__(self, docs: list[RetrievedDoc]) -> None:
        self.docs = docs
        self.calls: list[dict] = []

    def search(self, query, top_k, metadata_filter=None):
        self.calls.append({"query": query, "top_k": top_k, "metadata_filter": metadata_filter})
        return self.docs[:top_k]


# --- replace: no embedding model in the path at all ---


def test_retrieval_without_any_embedding_model():
    transport = FakeTransport(nouls={"c1": 0.95, "c2": 0.05, "c3": 0.02})
    hits = JevRetriever(JevClient(transport), corpus()).search("生成AIの利用率は?", top_k=2)
    assert [h.chunk_id for h in hits] == ["c1", "c2"]
    assert hits[0].score == 0.95


def test_the_whole_corpus_is_judged_one_noul_per_chunk():
    transport = FakeTransport(nouls={"c1": 0.9, "c2": 0.1, "c3": 0.1})
    JevRetriever(JevClient(transport), corpus()).search("問い", top_k=3)
    asked = {key for request in transport.requests for key in request["questions"]}
    assert asked == {"c1", "c2", "c3"}


def test_a_small_corpus_is_swept_in_one_request():
    # sizing follows the token budget, so many short chunks still fit in one call
    small = [Chunk(chunk_id=f"c{i}", page=1, section="", text="本文") for i in range(70)]
    transport = FakeTransport(noul=0.5)
    JevRetriever(JevClient(transport), small).search("問い", top_k=5)
    assert len(transport.requests) == 1


def test_a_corpus_too_heavy_for_one_request_is_swept_in_batches():
    big = [Chunk(chunk_id=f"c{i}", page=1, section="", text="あ" * 3_000) for i in range(30)]
    transport = FakeTransport(noul=0.5)
    hits = JevRetriever(JevClient(transport), big).search("問い", top_k=5)
    assert len(transport.requests) > 1
    assert len(hits) == 5


def test_every_chunk_is_judged_exactly_once_across_the_batches():
    big = [Chunk(chunk_id=f"c{i}", page=1, section="", text="あ" * 3_000) for i in range(30)]
    transport = FakeTransport(noul=0.5)
    JevRetriever(JevClient(transport), big).search("問い", top_k=30)
    asked = [key for request in transport.requests for key in request["questions"]]
    assert sorted(asked) == sorted(c.chunk_id for c in big)


def test_a_batch_carries_only_its_own_candidates_in_the_state():
    big = [Chunk(chunk_id=f"c{i}", page=1, section="", text="あ" * 3_000) for i in range(30)]
    transport = FakeTransport(noul=0.5)
    JevRetriever(JevClient(transport), big).search("問い", top_k=5)
    for request in transport.requests:
        assert set(request["state"]["candidates"]) == set(request["questions"])


def test_a_metadata_filter_narrows_the_sweep_before_jev_is_asked():
    transport = FakeTransport(noul=0.5)
    store = corpus()
    retriever = JevRetriever(
        JevClient(transport),
        store,
        metadata={"c1": {"page": 1}, "c2": {"page": 1}, "c3": {"page": 2}},
    )
    retriever.search("問い", top_k=3, metadata_filter={"page": 2})
    assert set(transport.requests[0]["questions"]) == {"c3"}


def test_sweeping_an_empty_corpus_sends_nothing():
    transport = FakeTransport()
    assert JevRetriever(JevClient(transport), []).search("問い", top_k=3) == []
    assert transport.requests == []


# --- supplement: fused with the embedding retriever ---


def test_hybrid_fuses_the_embedding_rank_with_the_jev_judgement():
    embedding_hits = [
        RetrievedDoc(chunk_id="c2", text="本節では動向を整理する。", score=0.88),
        RetrievedDoc(chunk_id="c1", text="生成AIの利用経験は26.7%であった。", score=0.60),
    ]
    transport = FakeTransport(nouls={"c1": 0.97, "c2": 0.04})
    hybrid = HybridRetriever(
        StubEmbeddingRetriever(embedding_hits), JevClient(transport), vector_weight=0.3
    )
    assert [h.chunk_id for h in hybrid.search("問い", top_k=2)] == ["c1", "c2"]


def test_hybrid_overfetches_candidates_before_judging():
    stub = StubEmbeddingRetriever([])
    HybridRetriever(stub, JevClient(FakeTransport()), candidate_k=25).search("問い", top_k=5)
    assert stub.calls[0]["top_k"] == 25


def test_hybrid_passes_the_metadata_filter_through():
    stub = StubEmbeddingRetriever([])
    HybridRetriever(stub, JevClient(FakeTransport())).search(
        "問い", top_k=5, metadata_filter={"topic": "ai"}
    )
    assert stub.calls[0]["metadata_filter"] == {"topic": "ai"}


def test_a_full_vector_weight_leaves_the_embedding_order_intact():
    embedding_hits = [
        RetrievedDoc(chunk_id="c2", text="b", score=0.88),
        RetrievedDoc(chunk_id="c1", text="a", score=0.60),
    ]
    transport = FakeTransport(nouls={"c1": 0.97, "c2": 0.04})
    hybrid = HybridRetriever(
        StubEmbeddingRetriever(embedding_hits), JevClient(transport), vector_weight=1.0
    )
    assert [h.chunk_id for h in hybrid.search("問い", top_k=2)] == ["c2", "c1"]
