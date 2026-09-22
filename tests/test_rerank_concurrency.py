"""Batches of a split shortlist are independent, so they go out together.

The sweep's timing column compares batch sizes, and a sequential loop over
batches would make a small batch size look slower than it is — the comparison
would measure the implementation rather than the request shape.
"""

import threading

from jev_rag.jev.client import JevClient
from jev_rag.jev.rerank import JevReranker
from jev_rag.jev.transport import FakeTransport
from jev_rag.types import RetrievedDoc


def docs(count: int) -> list[RetrievedDoc]:
    return [RetrievedDoc(doc_id=f"d{i}", text=f"候補{i}", score=0.0) for i in range(count)]


class Concurrent(FakeTransport):
    """Records how many sends were in flight at once."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0
        self.barrier = threading.Barrier(3, timeout=5)

    def send(self, body):
        with self.lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            pass
        with self.lock:
            self.in_flight -= 1
        return super().send(body)


def test_batches_are_sent_concurrently():
    transport = Concurrent(noul=0.5)
    JevReranker(JevClient(transport)).noul_rerank("問い", docs(9), batch_size=3)
    assert transport.peak >= 3


def test_every_candidate_is_still_scored_once():
    transport = FakeTransport(nouls={f"d{i}": i / 20 for i in range(20)})
    ranked = JevReranker(JevClient(transport)).noul_rerank("問い", docs(20), batch_size=4)
    assert len(ranked) == 20
    assert len({d.doc_id for d in ranked}) == 20
    assert ranked[0].doc_id == "d19"


def test_a_single_batch_needs_no_pool():
    transport = FakeTransport(noul=0.5)
    JevReranker(JevClient(transport)).noul_rerank("問い", docs(3), batch_size=10)
    assert len(transport.requests) == 1
