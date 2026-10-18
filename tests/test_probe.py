"""Do a batched route's probabilities follow the documents or their positions?"""

import pytest

from jev_rag.dataset import EvalQuery
from jev_rag.probe import (
    follows_documents,
    order_agreement,
    resolves_paths,
    reversed_queries,
    run_probe,
)
from jev_rag.types import RetrievedDoc


def queries():
    return [
        EvalQuery(
            query_id=f"q{i}",
            question="問",
            candidates=[
                RetrievedDoc(doc_id=f"q{i}-d{j}", text=f"文書{j}", score=0.0, label=int(j == 0))
                for j in range(4)
            ],
        )
        for i in range(3)
    ]


class ByDocument:
    """A judge that reads the document: the relevant one scores high wherever it sits."""

    name = "by_document"

    def rank(self, question, docs):
        for doc in docs:
            doc.score = 0.9 if doc.label else 0.1 + 0.01 * int(doc.doc_id[-1])
        return docs


class ByPosition:
    """A judge that only sees positions: the first slot always scores high."""

    name = "by_position"

    def rank(self, question, docs):
        for position, doc in enumerate(docs):
            doc.score = 0.9 - 0.2 * position
        return docs


class Broken:
    name = "broken"

    def rank(self, question, docs):
        raise RuntimeError("400 bad request")


def test_reversing_keeps_every_document_and_flips_the_order():
    flipped = reversed_queries(queries())
    assert [d.doc_id for d in flipped[0].candidates] == ["q0-d3", "q0-d2", "q0-d1", "q0-d0"]
    assert [d.doc_id for d in queries()[0].candidates] == ["q0-d0", "q0-d1", "q0-d2", "q0-d3"]


def test_a_judge_that_reads_documents_agrees_with_itself_in_both_orders(tmp_path):
    result = run_probe({"by_document": ByDocument()}, queries(), tmp_path / "p.json")
    entry = result["by_document"]
    assert entry["median_abs_diff"] == pytest.approx(0.0)
    assert entry["spearman"] == pytest.approx(1.0)
    assert entry["follows_documents"] is True


def test_a_judge_that_reads_positions_fails_the_check(tmp_path):
    result = run_probe({"by_position": ByPosition()}, queries(), tmp_path / "p.json")
    assert result["by_position"]["follows_documents"] is False
    assert (tmp_path / "p.json").exists()


def test_a_route_that_fails_every_question_is_recorded_not_crashed(tmp_path):
    result = run_probe({"broken": Broken()}, queries(), tmp_path / "p.json")["broken"]
    assert result["error"] == "全問失敗"
    assert len(result["failed"]) == 6
    assert result["follows_documents"] is False


def test_agreement_is_computed_per_document_not_per_position():
    forward = [
        {"query_id": "q", "doc_id": "a", "score": 0.9},
        {"query_id": "q", "doc_id": "b", "score": 0.1},
    ]
    backward = list(reversed(forward))
    assert order_agreement(forward, backward)["median_abs_diff"] == pytest.approx(0.0)


def test_the_thresholds_are_the_ones_in_the_spec():
    assert follows_documents({"median_abs_diff": 0.1, "spearman": 0.8}) is True
    assert follows_documents({"median_abs_diff": 0.11, "spearman": 0.9}) is False
    assert follows_documents({"median_abs_diff": 0.05, "spearman": 0.79}) is False
    # the spec fails a batched route that is 0.2 or more below its counterpart
    assert resolves_paths(batched=0.71, single=0.90) is True
    assert resolves_paths(batched=0.70, single=0.90) is False


class FailsWhenReversed:
    """Reads documents, but every request for q0 fails when the order is reversed."""

    name = "fails_when_reversed"

    def rank(self, question, docs):
        if docs[0].doc_id == "q0-d3":
            raise RuntimeError("429 rate limited")
        return ByDocument().rank(question, docs)


def test_a_question_that_fails_in_one_order_is_left_out_of_the_comparison(tmp_path):
    result = run_probe({"r": FailsWhenReversed()}, queries(), tmp_path / "p.json")["r"]
    assert result["pairs"] == 8
    assert result["unmatched"] == 4
    assert result["failed"] == ["q0"]
