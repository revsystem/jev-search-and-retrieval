"""How much of the candidate set shares one request.

TypeSafe documents that accuracy falls as the state grows with content
unrelated to the decision (model-jaggedness/jev-1.13, "Large state full of
irrelevant detail"). Putting 100 candidates in one state gives every question
99 distractors, so the size of the batch is a variable worth controlling
rather than an implementation detail.
"""

import pytest

from jev_rag.jev.client import JevClient
from jev_rag.jev.pairwise import PairwiseReranker
from jev_rag.jev.rerank import JevReranker
from jev_rag.jev.transport import FakeTransport
from jev_rag.types import RetrievedDoc


def docs(count: int = 10) -> list[RetrievedDoc]:
    return [RetrievedDoc(doc_id=f"d{i}", text=f"候補{i}の本文", score=0.0) for i in range(count)]


def test_the_whole_shortlist_shares_one_request_by_default():
    transport = FakeTransport(noul=0.5)
    JevReranker(JevClient(transport)).noul_rerank("問い", docs())
    assert len(transport.requests) == 1
    assert len(transport.requests[0]["state"]["candidates"]) == 10


@pytest.mark.parametrize("batch_size,expected", [(1, 10), (5, 2), (10, 1), (25, 1)])
def test_a_batch_size_splits_the_shortlist_into_that_many_requests(batch_size, expected):
    transport = FakeTransport(noul=0.5)
    JevReranker(JevClient(transport)).noul_rerank("問い", docs(), batch_size=batch_size)
    assert len(transport.requests) == expected


def test_a_batch_carries_only_its_own_candidates_as_distractors():
    transport = FakeTransport(noul=0.5)
    JevReranker(JevClient(transport)).noul_rerank("問い", docs(), batch_size=5)
    for request in transport.requests:
        assert len(request["state"]["candidates"]) == 5
        assert len(request["questions"]) == 5


def test_a_batch_of_one_is_cross_encoding():
    transport = FakeTransport(noul=0.5)
    JevReranker(JevClient(transport)).noul_rerank("問い", docs(3), batch_size=1)
    for request in transport.requests:
        assert len(request["state"]["candidates"]) == 1
        assert "`candidates[0].text`" in next(iter(request["questions"].values()))["instructions"]


def test_batching_still_ranks_every_candidate():
    transport = FakeTransport(nouls={f"d{i}": i / 10 for i in range(10)})
    ranked = JevReranker(JevClient(transport)).noul_rerank("問い", docs(), batch_size=3)
    assert len(ranked) == 10
    assert [d.doc_id for d in ranked][:2] == ["d9", "d8"]


def test_each_batch_keeps_its_own_index_alignment():
    import re

    transport = FakeTransport(noul=0.5)
    candidates = docs(9)
    JevReranker(JevClient(transport)).noul_rerank("問い", candidates, batch_size=4)
    by_id = {doc.doc_id: doc.text for doc in candidates}
    for request in transport.requests:
        for key, question in request["questions"].items():
            index = int(re.search(r"candidates\[(\d+)\]", question["instructions"]).group(1))
            assert request["state"]["candidates"][index]["text"] == by_id[key]


# --- pairwise: the state must hold only the pair being compared ---


def test_a_comparison_sees_only_its_two_candidates():
    transport = FakeTransport(choices={}, confidence=0.9)
    PairwiseReranker(JevClient(transport)).rerank("問い", docs(5))
    assert len(transport.requests) > 1
    for request in transport.requests:
        assert len(request["state"]["candidates"]) == 2
        assert len(request["questions"]) == 1


def test_a_comparison_addresses_its_pair_by_the_first_two_positions():
    transport = FakeTransport(choices={}, confidence=0.9)
    PairwiseReranker(JevClient(transport)).rerank("問い", docs(4))
    for request in transport.requests:
        criteria = next(iter(request["questions"].values()))["criteria"]
        assert "`candidates[0].text`" in criteria["a"]
        assert "`candidates[1].text`" in criteria["b"]


def test_the_winning_label_still_maps_back_to_the_right_document():
    # option "b" always wins, so the second member of every pair gains
    transport = FakeTransport(choices={}, confidence=0.95)

    class AlwaysB(FakeTransport):
        def send(self, body):
            self.requests.append(body)
            options = list(next(iter(body["questions"].values()))["criteria"])
            return {
                "answers": {
                    next(iter(body["questions"])): {
                        "type": "choice",
                        "choice": options[1],
                        "probabilities": {options[0]: 0.05, options[1]: 0.95},
                        "confidence": 0.95,
                    }
                }
            }

    transport = AlwaysB()
    candidates = docs(3)
    ranked = PairwiseReranker(JevClient(transport), opponents=2).rerank("問い", candidates)
    # the last document is never the first member of a pair in a round robin,
    # so it wins every comparison it takes part in
    assert ranked[0].doc_id == "d2"


def test_a_shortlist_of_one_needs_no_comparison():
    transport = FakeTransport()
    assert PairwiseReranker(JevClient(transport)).rerank("問い", docs(1))[0].doc_id == "d0"
    assert transport.requests == []
