"""Guards on the end-to-end comparison being a comparison.

Every one of these was a finding from review: a generation failure counted as
a wrong answer, a baseline name that stops matching, and a route that cannot
be built at all.
"""

import pytest

from jev_rag.dataset import EvalQuery
from jev_rag.endtoend import answer_queries, format_answers, paired_difference
from jev_rag.types import RetrievedDoc


def query(qid="q1"):
    return EvalQuery(
        query_id=qid,
        question="最も低い温度は?",
        candidates=[RetrievedDoc(doc_id="a", text="絶対零度は最低温度。", score=0.0, label=1)],
        answers=["絶対零度"],
    )


class Ranker:
    name = "r"

    def rank(self, question, docs):
        return docs


class Generator:
    def __init__(self, replies):
        self.replies = list(replies)

    def answer(self, question, docs, **kw):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_a_generation_failure_is_counted_separately_from_a_wrong_answer():
    result = answer_queries(
        Ranker(),
        [query("q1"), query("q2")],
        Generator([RuntimeError("throttled"), "<answer>絶対零度</answer>"]),
    )
    assert result["errors"] == 1
    # the failed call is out of the denominator; the one real answer was right
    assert result["accuracy"] == 1.0
    assert result["scored"] == 1


def test_the_error_count_reaches_the_table():
    result = answer_queries(Ranker(), [query()], Generator([RuntimeError("throttled")]))
    assert "1" in format_answers({"r": result}, baseline="r")


def test_an_empty_generation_is_a_wrong_answer_not_an_error():
    result = answer_queries(Ranker(), [query()], Generator([""]))
    assert result["errors"] == 0
    assert result["accuracy"] == 0.0


def test_a_missing_baseline_is_reported_rather_than_silently_dropping_deltas():
    result = answer_queries(Ranker(), [query()], Generator(["<answer>絶対零度</answer>"]))
    table = format_answers({"r+select": result}, baseline="embedding")
    assert "embedding" in table


def test_the_paired_difference_uses_the_same_queries():
    # routes are run over identical queries, so the difference is paired and
    # its spread is much tighter than the two independent errors suggest
    base = [{"query_id": "q1", "correct": True}, {"query_id": "q2", "correct": False}]
    other = [{"query_id": "q1", "correct": True}, {"query_id": "q2", "correct": True}]
    diff = paired_difference(base, other)
    assert diff["mean"] == pytest.approx(0.5)
    assert diff["n"] == 2


def test_the_paired_difference_ignores_queries_only_one_route_answered():
    base = [{"query_id": "q1", "correct": True}]
    other = [{"query_id": "q1", "correct": True}, {"query_id": "q2", "correct": True}]
    assert paired_difference(base, other)["n"] == 1


def test_no_shared_queries_gives_no_difference():
    assert paired_difference([{"query_id": "a", "correct": True}], []) == {}
