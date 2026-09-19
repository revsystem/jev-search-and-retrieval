"""Use case: cross-encode queries and candidates for higher precision."""

from jev_rag.evaluation import RetrievedDoc
from jev_rag.jev.client import JevClient
from jev_rag.jev.crossencode import CrossEncoder
from jev_rag.jev.transport import FakeTransport


def docs() -> list[RetrievedDoc]:
    return [
        RetrievedDoc(chunk_id="a", text="話題は近い", score=0.9),
        RetrievedDoc(chunk_id="b", text="直接の根拠", score=0.4),
    ]


def test_each_pair_gets_its_own_request_with_only_that_candidate_in_state():
    transport = FakeTransport(nouls={"relevance": 0.5})
    CrossEncoder(JevClient(transport)).rerank("問い", docs())
    assert len(transport.requests) == 2
    for request in transport.requests:
        assert set(request["state"]) == {"query", "candidate"}
        assert isinstance(request["state"]["candidate"], str)


def test_no_other_candidate_leaks_into_the_shared_state():
    transport = FakeTransport(nouls={"relevance": 0.5})
    CrossEncoder(JevClient(transport)).rerank("問い", docs())
    states = [request["state"]["candidate"] for request in transport.requests]
    assert sorted(states) == ["直接の根拠", "話題は近い"]
    assert all(states.count(s) == 1 for s in states)


def test_ranking_follows_the_per_pair_judgement():
    # Cross-encoding runs the pairs concurrently, so the stub must derive its
    # answer from the request alone rather than from mutable shared state.
    class Scripted(FakeTransport):
        def send(self, body):
            self.requests.append(body)
            value = 0.95 if body["state"]["candidate"] == "直接の根拠" else 0.10
            return {"answers": {"relevance": {"type": "noul", "noul": value}}}

    ranked = CrossEncoder(JevClient(Scripted())).rerank("問い", docs())
    assert [d.chunk_id for d in ranked] == ["b", "a"]
    assert ranked[0].score == 0.95


def test_both_a_probability_and_an_ordinal_grade_can_be_requested():
    transport = FakeTransport(nouls={"relevance": 0.8}, scores={"grade": 2.0}, score_levels=4)
    CrossEncoder(JevClient(transport), with_grade=True).rerank("問い", docs())
    assert set(transport.requests[0]["questions"]) == {"relevance", "grade"}


def test_the_ordinal_grade_breaks_ties_between_equal_probabilities():
    class Scripted(FakeTransport):
        def send(self, body):
            self.requests.append(body)
            grade = 3.0 if body["state"]["candidate"] == "直接の根拠" else 1.0
            return {
                "answers": {
                    "relevance": {"type": "noul", "noul": 0.9},
                    "grade": {"type": "score", "score": grade, "legend": {}, "confidence": 0.8},
                }
            }

    ranked = CrossEncoder(JevClient(Scripted()), with_grade=True).rerank("問い", docs())
    assert [d.chunk_id for d in ranked] == ["b", "a"]


def test_top_k_trims_the_result():
    transport = FakeTransport(nouls={"relevance": 0.5})
    assert len(CrossEncoder(JevClient(transport)).rerank("問い", docs(), top_k=1)) == 1


def test_an_empty_shortlist_sends_nothing():
    transport = FakeTransport()
    assert CrossEncoder(JevClient(transport)).rerank("問い", []) == []
    assert transport.requests == []
