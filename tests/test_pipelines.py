from jev_rag.evaluation import RetrievedDoc
from jev_rag.jev.client import JevClient
from jev_rag.jev.transport import FakeTransport
from jev_rag.pipelines import BaselinePipeline, ClassicRerankPipeline, JevPipeline


class StubRetriever:
    """Returns a fixed candidate list and records how it was called."""

    def __init__(self, docs: list[RetrievedDoc]) -> None:
        self.docs = docs
        self.calls: list[dict] = []

    def search(self, query, top_k, metadata_filter=None):
        self.calls.append({"query": query, "top_k": top_k, "metadata_filter": metadata_filter})
        return self.docs[:top_k]


class StubCrossEncoder:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def rerank(self, query, docs, top_k):
        ranked = sorted(docs, key=lambda d: self.scores.get(d.chunk_id, 0.0), reverse=True)
        return ranked[:top_k]


def candidates() -> list[RetrievedDoc]:
    return [
        RetrievedDoc(chunk_id="topical", text="話題は近いが根拠にならない", score=0.81),
        RetrievedDoc(chunk_id="partial", text="部分的な根拠", score=0.77),
        RetrievedDoc(chunk_id="evidence", text="直接の根拠", score=0.61),
    ]


def test_baseline_returns_the_vector_order_untouched():
    pipeline = BaselinePipeline(StubRetriever(candidates()), top_k=3)
    assert [d.chunk_id for d in pipeline.retrieve("問い")] == ["topical", "partial", "evidence"]


def test_baseline_requests_exactly_top_k_candidates():
    retriever = StubRetriever(candidates())
    BaselinePipeline(retriever, top_k=2).retrieve("問い")
    assert retriever.calls[0]["top_k"] == 2


def test_classic_rerank_overfetches_then_trims():
    retriever = StubRetriever(candidates())
    pipeline = ClassicRerankPipeline(
        retriever,
        StubCrossEncoder({"evidence": 0.9, "partial": 0.5, "topical": 0.1}),
        top_k=2,
        candidate_k=3,
    )
    ranked = pipeline.retrieve("問い")
    assert retriever.calls[0]["top_k"] == 3
    assert [d.chunk_id for d in ranked] == ["evidence", "partial"]


def test_jev_pipeline_lifts_buried_evidence_above_a_topical_match():
    transport = FakeTransport(
        scores={"topical": 0.3, "partial": 1.8, "evidence": 3.0},
        nouls={"topical": 0.05, "partial": 0.6, "evidence": 0.97},
        score_levels=4,
    )
    pipeline = JevPipeline(
        StubRetriever(candidates()), JevClient(transport), top_k=3, candidate_k=3, vector_weight=0.3
    )
    ranked = pipeline.retrieve("問い")
    assert ranked[0].chunk_id == "evidence"


def test_jev_pipeline_drops_a_merely_topical_candidate():
    transport = FakeTransport(
        scores={"topical": 0.3, "partial": 1.8, "evidence": 3.0},
        nouls={"topical": 0.05, "partial": 0.6, "evidence": 0.97},
        score_levels=4,
    )
    pipeline = JevPipeline(
        StubRetriever(candidates()),
        JevClient(transport),
        top_k=3,
        candidate_k=3,
        relevance_threshold=0.2,
    )
    assert "topical" not in {d.chunk_id for d in pipeline.retrieve("問い")}


def test_jev_pipeline_without_filtering_keeps_every_candidate():
    transport = FakeTransport(
        scores={"topical": 0.3, "partial": 1.8, "evidence": 3.0},
        nouls={"topical": 0.05, "partial": 0.6, "evidence": 0.97},
        score_levels=4,
    )
    pipeline = JevPipeline(
        StubRetriever(candidates()),
        JevClient(transport),
        top_k=3,
        candidate_k=3,
        filter_first=False,
    )
    assert len(pipeline.retrieve("問い")) == 3


def test_query_planning_narrows_the_vector_search_by_metadata():
    transport = FakeTransport(
        choices={"topic": "ai"},
        nouls={"single_topic": 0.9, "topical": 0.9, "partial": 0.9, "evidence": 0.9},
        scores={"topical": 1.0, "partial": 1.0, "evidence": 3.0},
        score_levels=4,
        confidence=0.9,
    )
    retriever = StubRetriever(candidates())
    pipeline = JevPipeline(retriever, JevClient(transport), top_k=3, candidate_k=3, plan_query=True)
    pipeline.retrieve("生成AIの利用率は?")
    assert retriever.calls[0]["metadata_filter"] == {"topic": {"$in": ["ai", "unknown"]}}


def test_pipelines_expose_a_stable_name_for_reporting():
    assert BaselinePipeline(StubRetriever([]), top_k=1).name == "baseline"
    jev = JevPipeline(StubRetriever([]), JevClient(FakeTransport()), top_k=1)
    assert jev.name == "jev"
