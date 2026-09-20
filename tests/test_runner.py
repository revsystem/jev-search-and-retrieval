from jev_rag.dataset import EvalQuery
from jev_rag.report import format_comparison, format_movers
from jev_rag.runner import evaluate_rankers
from jev_rag.types import RetrievedDoc


class FixedRanker:
    def __init__(self, name, order):
        self.name = name
        self.order = order

    def rank(self, question, docs):
        by_id = {d.doc_id: d for d in docs}
        ranked = [by_id[i] for i in self.order]
        for rank, doc in enumerate(ranked):
            doc.score = 1.0 - rank * 0.1
        return ranked


def query() -> EvalQuery:
    return EvalQuery(
        query_id="q1",
        question="問い",
        candidates=[
            RetrievedDoc(doc_id="a", text="不正解", score=0.0, label=0),
            RetrievedDoc(doc_id="b", text="正解", score=0.0, label=1),
        ],
    )


def test_a_perfect_ranker_beats_a_reversed_one():
    result = evaluate_rankers(
        {"good": FixedRanker("good", ["b", "a"]), "bad": FixedRanker("bad", ["a", "b"])},
        [query()],
        k=10,
    )
    assert result["good"]["metrics"]["ndcg@10"] > result["bad"]["metrics"]["ndcg@10"]


def test_each_ranker_sees_an_unscored_copy_of_the_candidates():
    shared = [query()]
    evaluate_rankers({"first": FixedRanker("first", ["b", "a"])}, shared, k=10)
    assert all(c.score == 0.0 for c in shared[0].candidates)


def test_per_query_rows_are_kept_for_reporting():
    result = evaluate_rankers({"good": FixedRanker("good", ["b", "a"])}, [query()], k=10)
    rows = result["good"]["per_query"]
    assert rows[0]["query_id"] == "q1"
    assert rows[0]["ndcg@10"] == 1.0
    assert rows[0]["top"] == ["b", "a"]


def test_a_failing_ranker_is_recorded_rather_than_aborting_the_run():
    class Broken:
        name = "broken"

        def rank(self, question, docs):
            raise RuntimeError("no credentials")

    result = evaluate_rankers(
        {"broken": Broken(), "good": FixedRanker("good", ["b", "a"])}, [query()], k=10
    )
    assert "error" in result["broken"]
    assert result["good"]["metrics"]["ndcg@10"] == 1.0


def test_the_comparison_table_shows_a_delta_against_the_baseline():
    table = format_comparison(
        {
            "embedding": {"metrics": {"ndcg@10": 0.50}},
            "jev_crossencode": {"metrics": {"ndcg@10": 0.70}},
        },
        baseline="embedding",
        metrics=["ndcg@10"],
    )
    assert "0.700" in table and "+0.200" in table


def test_the_comparison_table_reports_a_failed_ranker_instead_of_a_number():
    table = format_comparison(
        {"embedding": {"metrics": {"ndcg@10": 0.5}}, "x": {"error": "no credentials"}},
        baseline="embedding",
        metrics=["ndcg@10"],
    )
    assert "no credentials" in table


def test_movers_show_where_a_ranker_lifted_a_buried_answer():
    results = {
        "embedding": {"per_query": [{"query_id": "q1", "question": "問い", "first_relevant": 7}]},
        "jev": {"per_query": [{"query_id": "q1", "question": "問い", "first_relevant": 1}]},
    }
    text = format_movers(results, baseline="embedding", ranker="jev", limit=5)
    assert "q1" in text and "7" in text and "1" in text
