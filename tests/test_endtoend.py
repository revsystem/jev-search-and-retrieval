"""The end-to-end comparison: does a better ranking produce a better answer."""

from jev_rag.dataset import EvalQuery
from jev_rag.endtoend import answer_queries
from jev_rag.types import RetrievedDoc


def query() -> EvalQuery:
    return EvalQuery(
        query_id="q1",
        question="最も低い温度を何という?",
        candidates=[
            RetrievedDoc(doc_id="wrong", text="ケルビンは温度の単位である。", score=0.0, label=0),
            RetrievedDoc(doc_id="right", text="絶対零度は摂氏-273.15度。", score=0.0, label=1),
        ],
        answers=["絶対零度"],
    )


class FixedRanker:
    def __init__(self, name, order):
        self.name = name
        self.order = order

    def rank(self, question, docs):
        by_id = {d.doc_id: d for d in docs}
        return [by_id[i] for i in self.order]


class EchoGenerator:
    """Answers with the first passage it is given, like a perfectly faithful model."""

    def __init__(self):
        self.calls = []

    def answer(self, question, docs, **kw):
        self.calls.append([d.doc_id for d in docs])
        return docs[0].text if docs else ""


def test_a_ranking_that_puts_the_evidence_first_answers_correctly():
    generator = EchoGenerator()
    result = answer_queries(FixedRanker("good", ["right", "wrong"]), [query()], generator, top_k=1)
    assert result["accuracy"] == 1.0


def test_a_ranking_that_buries_the_evidence_answers_wrongly():
    generator = EchoGenerator()
    result = answer_queries(FixedRanker("bad", ["wrong", "right"]), [query()], generator, top_k=1)
    assert result["accuracy"] == 0.0


def test_only_the_top_k_passages_reach_the_generator():
    generator = EchoGenerator()
    answer_queries(FixedRanker("good", ["right", "wrong"]), [query()], generator, top_k=1)
    assert generator.calls == [["right"]]


def test_each_answer_is_kept_for_inspection():
    generator = EchoGenerator()
    result = answer_queries(FixedRanker("good", ["right", "wrong"]), [query()], generator, top_k=1)
    row = result["per_query"][0]
    assert row["query_id"] == "q1"
    assert row["gold"] == ["絶対零度"]
    assert "絶対零度" in row["answer"]
    assert row["correct"] is True


def test_the_standard_error_is_reported():
    generator = EchoGenerator()
    queries = [query(), query()]
    result = answer_queries(FixedRanker("good", ["right", "wrong"]), queries, generator, top_k=1)
    assert "stderr" in result


def test_whether_the_evidence_was_in_the_prompt_is_recorded_separately():
    # separates a retrieval failure from a generation failure: if the passage
    # was there and the answer is still wrong, ranking is not the problem
    generator = EchoGenerator()
    result = answer_queries(FixedRanker("bad", ["wrong", "right"]), [query()], generator, top_k=1)
    assert result["per_query"][0]["evidence_in_prompt"] is False
    assert result["evidence_rate"] == 0.0


def test_a_generator_failure_is_recorded_rather_than_aborting_the_run():
    class Broken:
        def answer(self, question, docs, **kw):
            raise RuntimeError("throttled")

    result = answer_queries(FixedRanker("good", ["right", "wrong"]), [query()], Broken(), top_k=1)
    assert result["per_query"][0]["correct"] is False
    assert "throttled" in result["per_query"][0]["error"]


def test_a_query_with_no_gold_answer_is_skipped():
    bare = EvalQuery(query_id="q2", question="?", candidates=query().candidates, answers=[])
    result = answer_queries(FixedRanker("good", ["right", "wrong"]), [bare], EchoGenerator(), 1)
    assert result["per_query"] == []
    assert result["accuracy"] == 0.0
