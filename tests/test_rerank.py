from jev_rag.evaluation import RetrievedDoc
from jev_rag.jev.client import JevClient
from jev_rag.jev.rerank import JevReranker, fuse_scores, min_max_normalise
from jev_rag.jev.transport import FakeTransport


def candidates() -> list[RetrievedDoc]:
    return [
        RetrievedDoc(chunk_id="top", text="話題は近いが根拠にならない文章", score=0.81),
        RetrievedDoc(chunk_id="middle", text="部分的な根拠", score=0.77),
        RetrievedDoc(chunk_id="buried", text="直接の根拠となる決定的な記述", score=0.61),
    ]


def test_min_max_normalise_maps_to_unit_interval():
    assert min_max_normalise([0.61, 0.77, 0.81]) == [0.0, 0.8, 1.0]


def test_min_max_normalise_handles_a_flat_vector():
    assert min_max_normalise([0.5, 0.5]) == [1.0, 1.0]


def test_score_rerank_lifts_a_buried_document_to_the_top():
    transport = FakeTransport(scores={"top": 0.4, "middle": 1.5, "buried": 3.0}, score_levels=4)
    reranker = JevReranker(JevClient(transport), vector_weight=0.0)
    ranked = reranker.rerank("問い", candidates())
    assert [d.chunk_id for d in ranked] == ["buried", "middle", "top"]


def test_rerank_asks_one_question_per_candidate_against_shared_state():
    transport = FakeTransport(scores={"top": 1.0, "middle": 1.0, "buried": 1.0}, score_levels=4)
    JevReranker(JevClient(transport)).rerank("問い", candidates())
    request = transport.requests[0]
    assert len(request["questions"]) == 3
    assert request["state"]["query"] == "問い"
    assert set(request["state"]["candidates"]) == {"top", "middle", "buried"}


def test_relevance_filter_drops_candidates_below_the_threshold():
    transport = FakeTransport(nouls={"top": 0.05, "middle": 0.30, "buried": 0.95})
    reranker = JevReranker(JevClient(transport))
    kept = reranker.relevance_filter("問い", candidates(), threshold=0.2)
    assert [d.chunk_id for d in kept] == ["buried", "middle"]


def test_relevance_filter_never_returns_an_empty_context():
    transport = FakeTransport(nouls={"top": 0.01, "middle": 0.01, "buried": 0.02})
    reranker = JevReranker(JevClient(transport))
    kept = reranker.relevance_filter("問い", candidates(), threshold=0.5)
    assert [d.chunk_id for d in kept] == ["buried"]


def test_fusion_blends_vector_and_jev_signals():
    fused = fuse_scores(vector=[1.0, 0.0], jev=[0.0, 1.0], vector_weight=0.25)
    assert fused == [0.25, 0.75]


def test_fusion_with_full_vector_weight_preserves_the_original_order():
    transport = FakeTransport(scores={"top": 0.0, "middle": 0.0, "buried": 3.0}, score_levels=4)
    reranker = JevReranker(JevClient(transport), vector_weight=1.0)
    ranked = reranker.rerank("問い", candidates())
    assert [d.chunk_id for d in ranked] == ["top", "middle", "buried"]


def test_rerank_records_the_rank_delta_for_reporting():
    transport = FakeTransport(scores={"top": 0.0, "middle": 1.0, "buried": 3.0}, score_levels=4)
    reranker = JevReranker(JevClient(transport), vector_weight=0.0)
    ranked = reranker.rerank("問い", candidates())
    buried = next(d for d in ranked if d.chunk_id == "buried")
    assert buried.rank_delta == 2


def test_precomputed_vector_scores_are_used_verbatim():
    transport = FakeTransport(scores={"top": 0.0, "middle": 0.0, "buried": 0.0}, score_levels=4)
    reranker = JevReranker(JevClient(transport), vector_weight=1.0)
    ranked = reranker.rerank(
        "問い", candidates(), vector_scores={"top": 0.1, "middle": 0.2, "buried": 0.9}
    )
    assert [d.chunk_id for d in ranked] == ["buried", "middle", "top"]


def test_filtering_a_candidate_set_does_not_restretch_the_vector_signal():
    # top and middle sit 0.04 apart out of a 0.20 spread; after dropping one
    # candidate that gap must not become the whole 0..1 range.
    transport = FakeTransport(scores={"middle": 3.0, "buried": 1.0}, score_levels=4)
    reranker = JevReranker(JevClient(transport), vector_weight=0.3)
    kept = [d for d in candidates() if d.chunk_id != "top"]
    ranked = reranker.rerank("問い", kept, vector_scores={"middle": 0.8, "buried": 0.0})
    assert [d.chunk_id for d in ranked] == ["middle", "buried"]
