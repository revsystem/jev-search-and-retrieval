"""Splitting a shortlist that cannot share one request.

Jev rejects an oversized request with 400 max_tokens_exceeded. The binding
limit for a reranking request is the 32k budget for the state plus the single
longest question, and JQaRA's largest shortlists exceed it: 100 passages of up
to 400 Japanese characters is over 32k tokens of state on its own.

A caller asking to rank a shortlist should not have to know that. The batch
size is chosen to fit unless one is given explicitly.
"""

from jev_rag.jev.budget import estimate_tokens, fitting_batch_size
from jev_rag.jev.client import JevClient
from jev_rag.jev.rerank import JevReranker
from jev_rag.jev.transport import FakeTransport
from jev_rag.types import RetrievedDoc


def docs(count: int, chars: int = 400) -> list[RetrievedDoc]:
    return [RetrievedDoc(doc_id=f"d{i}", text="あ" * chars, score=0.0) for i in range(count)]


def test_the_estimate_allows_for_undercounting():
    # measured against reported usage, the raw character count came in ~10%
    # under, and the limits are hard
    assert estimate_tokens("あ" * 1000) >= 1000


def test_a_small_shortlist_needs_no_splitting():
    assert fitting_batch_size("問い", docs(5), question_tokens=120) == 5


def test_a_shortlist_too_large_for_one_state_is_split():
    size = fitting_batch_size("問い", docs(100), question_tokens=120)
    assert 0 < size < 100


def test_each_batch_fits_the_state_plus_question_budget():
    from jev_rag.jev.budget import STATE_PLUS_QUESTION_BUDGET

    candidates = docs(100)
    size = fitting_batch_size("問い", candidates, question_tokens=120)
    batch = candidates[:size]
    state = {"query": "問い", "candidates": [{"text": d.text} for d in batch]}
    assert estimate_tokens(state) + 120 <= STATE_PLUS_QUESTION_BUDGET


def test_at_least_one_candidate_is_always_attempted():
    # a single passage over the budget cannot be split further; the caller
    # should see the service's own error rather than an empty batch
    assert fitting_batch_size("問い", docs(1, chars=40_000), question_tokens=120) == 1


def test_ranking_a_large_shortlist_splits_instead_of_failing():
    transport = FakeTransport(noul=0.5)
    ranked = JevReranker(JevClient(transport)).noul_rerank("問い", docs(100))
    assert len(ranked) == 100
    assert len(transport.requests) > 1


def test_an_explicit_batch_size_is_still_honoured():
    transport = FakeTransport(noul=0.5)
    JevReranker(JevClient(transport)).noul_rerank("問い", docs(20, chars=10), batch_size=5)
    assert len(transport.requests) == 4


def test_an_explicit_batch_size_is_capped_to_what_fits():
    # asking for 100 per request on passages this large cannot be honoured
    transport = FakeTransport(noul=0.5)
    JevReranker(JevClient(transport)).noul_rerank("問い", docs(100), batch_size=100)
    assert len(transport.requests) > 1
    for request in transport.requests:
        assert estimate_tokens(request["state"]) < 32_000
