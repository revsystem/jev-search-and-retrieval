"""Use case: score query-to-candidate relevance.

A shortlist from embeddings, one Noul per query-candidate pair, and a sort on
the returned probability — no generative model asked to invent a scale.
"""

from jev_rag.evaluation import RetrievedDoc
from jev_rag.jev.client import JevClient
from jev_rag.jev.rerank import JevReranker
from jev_rag.jev.transport import FakeTransport


def shortlist() -> list[RetrievedDoc]:
    return [
        RetrievedDoc(chunk_id="topical", text="話題は近い", score=0.81),
        RetrievedDoc(chunk_id="evidence", text="直接の根拠", score=0.61),
    ]


def test_sorting_uses_the_returned_probability_directly():
    transport = FakeTransport(nouls={"topical": 0.11, "evidence": 0.94})
    ranked = JevReranker(JevClient(transport)).noul_rerank("問い", shortlist())
    assert [d.chunk_id for d in ranked] == ["evidence", "topical"]
    assert ranked[0].score == 0.94


def test_one_noul_is_asked_per_query_candidate_pair():
    transport = FakeTransport(nouls={"topical": 0.1, "evidence": 0.9})
    JevReranker(JevClient(transport)).noul_rerank("問い", shortlist())
    questions = transport.requests[0]["questions"]
    assert len(questions) == 2
    assert all(q["type"] == "noul" for q in questions.values())


def test_no_generative_scale_is_invented_so_the_vector_score_is_not_fused():
    transport = FakeTransport(nouls={"topical": 0.51, "evidence": 0.52})
    ranked = JevReranker(JevClient(transport), vector_weight=0.9).noul_rerank("問い", shortlist())
    # even with a heavy vector weight configured, this use case ignores it
    assert [d.chunk_id for d in ranked] == ["evidence", "topical"]


def test_the_original_vector_score_is_preserved_for_reporting():
    transport = FakeTransport(nouls={"topical": 0.1, "evidence": 0.9})
    ranked = JevReranker(JevClient(transport)).noul_rerank("問い", shortlist())
    assert next(d for d in ranked if d.chunk_id == "evidence").vector_score == 0.61


def test_rank_movement_is_recorded():
    transport = FakeTransport(nouls={"topical": 0.1, "evidence": 0.9})
    ranked = JevReranker(JevClient(transport)).noul_rerank("問い", shortlist())
    assert ranked[0].rank_delta == 1


def test_top_k_trims_the_shortlist():
    transport = FakeTransport(nouls={"topical": 0.1, "evidence": 0.9})
    ranked = JevReranker(JevClient(transport)).noul_rerank("問い", shortlist(), top_k=1)
    assert len(ranked) == 1


def test_an_empty_shortlist_needs_no_request():
    transport = FakeTransport()
    assert JevReranker(JevClient(transport)).noul_rerank("問い", []) == []
    assert transport.requests == []
