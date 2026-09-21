"""Use case: rerank results with pairwise comparisons."""

import pytest

from jev_rag.jev.client import JevClient
from jev_rag.jev.pairwise import DEFAULT_COMPARISON_BUDGET, PairwiseReranker, pair_schedule
from jev_rag.jev.transport import FakeTransport
from jev_rag.types import RetrievedDoc


def docs(*ids: str) -> list[RetrievedDoc]:
    return [RetrievedDoc(doc_id=i, text=f"文書{i}", score=0.5) for i in ids]


def test_a_small_shortlist_is_compared_round_robin():
    assert pair_schedule(["a", "b", "c"], opponents=2, limit=64) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]


def test_a_large_shortlist_falls_back_to_sampled_opponents():
    ids = [f"c{i}" for i in range(60)]
    schedule = pair_schedule(ids, opponents=3, limit=DEFAULT_COMPARISON_BUDGET)
    assert len(schedule) <= DEFAULT_COMPARISON_BUDGET
    # every candidate still gets compared at least once
    assert {c for pair in schedule for c in pair} == set(ids)


def test_the_schedule_never_pairs_a_candidate_with_itself():
    schedule = pair_schedule([f"c{i}" for i in range(10)], opponents=3, limit=8)
    assert all(left != right for left, right in schedule)


def test_the_schedule_is_deterministic():
    ids = [f"c{i}" for i in range(15)]
    assert pair_schedule(ids, opponents=3, limit=20) == pair_schedule(ids, opponents=3, limit=20)


def test_each_comparison_is_a_two_option_choice():
    transport = FakeTransport(choices={}, confidence=0.9)
    PairwiseReranker(JevClient(transport)).rerank("問い", docs("a", "b"))
    question = transport.requests[0]["questions"]["winner"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"a", "b"}


def test_the_candidate_that_wins_its_comparisons_ranks_first():
    # the only pair is (a, b), so option "b" is the second document
    transport = FakeTransport(choices={"winner": "b"}, confidence=0.9)
    ranked = PairwiseReranker(JevClient(transport)).rerank("問い", docs("a", "b"))
    assert [d.chunk_id for d in ranked] == ["b", "a"]


def test_wins_are_weighted_by_the_returned_probability():
    """c beats both opponents narrowly, a beats b, so c leads on win rate."""

    class Scripted(FakeTransport):
        def send(self, body):
            self.requests.append(body)
            texts = [entry["text"] for entry in body["state"]["candidates"]]
            # "文書c" wins whenever it takes part, otherwise the first side wins
            winner = "b" if texts[1] == "文書c" else "a"
            loser = "a" if winner == "b" else "b"
            return {
                "answers": {
                    "winner": {
                        "type": "choice",
                        "choice": winner,
                        "probabilities": {winner: 0.55, loser: 0.45},
                        "confidence": 0.55,
                    }
                }
            }

    ranked = PairwiseReranker(JevClient(Scripted())).rerank("問い", docs("a", "b", "c"))
    assert ranked[0].chunk_id == "c"


def test_each_comparison_is_its_own_request():
    # a shared state would leave every comparison reading the whole shortlist
    transport = FakeTransport(confidence=0.9)
    PairwiseReranker(JevClient(transport)).rerank("問い", docs("a", "b", "c"))
    assert len(transport.requests) == 3
    assert all(len(r["state"]["candidates"]) == 2 for r in transport.requests)


def test_the_win_rate_becomes_the_new_score():
    transport = FakeTransport(choices={"p0": "b"}, confidence=0.8)
    ranked = PairwiseReranker(JevClient(transport)).rerank("問い", docs("a", "b"))
    assert ranked[0].score == 0.8
    assert ranked[1].score == pytest.approx(0.2)


def test_the_original_vector_score_is_preserved():
    transport = FakeTransport(choices={"p0": "b"}, confidence=0.8)
    ranked = PairwiseReranker(JevClient(transport)).rerank("問い", docs("a", "b"))
    assert all(d.vector_score == 0.5 for d in ranked)


def test_a_single_candidate_needs_no_comparison():
    transport = FakeTransport()
    ranked = PairwiseReranker(JevClient(transport)).rerank("問い", docs("a"))
    assert [d.chunk_id for d in ranked] == ["a"]
    assert transport.requests == []
