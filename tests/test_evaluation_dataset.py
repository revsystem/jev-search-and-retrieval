import pytest

from jev_rag.documents import Chunk
from jev_rag.evaluation import EvalQuery, build_qrels, compare


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, page=1, section="", text=text)


CORPUS = [
    chunk("c1", "生成AIの利用率は前年から大きく上昇した。"),
    chunk("c2", "生成AIという言葉の説明にとどまる一般的な記述。"),
    chunk("c3", "ブロードバンド回線の敷設状況について。"),
]


def test_keyword_rules_produce_graded_relevance_labels():
    query = EvalQuery(
        query_id="q1",
        question="生成AIの利用率はどう変化したか",
        relevant={"all": ["生成AI", "利用率"], "grade": 3},
        partial={"all": ["生成AI"], "grade": 1},
    )
    qrels = build_qrels(query, CORPUS)
    assert qrels == {"c1": 3, "c2": 1}


def test_a_chunk_is_graded_by_its_strongest_matching_rule():
    query = EvalQuery(
        query_id="q1",
        question="q",
        relevant={"all": ["生成AI"], "grade": 3},
        partial={"all": ["生成AI"], "grade": 1},
    )
    assert build_qrels(query, CORPUS)["c2"] == 3


def test_explicit_chunk_ids_override_keyword_rules():
    query = EvalQuery(query_id="q1", question="q", qrels={"c3": 3})
    assert build_qrels(query, CORPUS) == {"c3": 3}


def test_a_query_without_any_labelling_rule_is_rejected():
    with pytest.raises(ValueError):
        build_qrels(EvalQuery(query_id="q1", question="q"), CORPUS)


def test_compare_reports_one_row_per_pipeline_and_a_delta():
    report = compare(
        {
            "baseline": {"ndcg@5": 0.40, "recall@5": 0.50},
            "jev": {"ndcg@5": 0.62, "recall@5": 0.50},
        },
        baseline="baseline",
    )
    jev_row = next(r for r in report if r["pipeline"] == "jev")
    assert jev_row["ndcg@5"] == 0.62
    assert jev_row["delta_ndcg@5"] == pytest.approx(0.22)


def test_binary_metrics_ignore_merely_topical_chunks():
    from jev_rag.evaluation import RetrievedDoc, score_ranking

    ranked = [RetrievedDoc(chunk_id="topical", text="", score=1.0)]
    scores = score_ranking(ranked, {"topical": 1, "evidence": 3}, k=1)
    # grade 1 is a topical decoy: it must not count as a hit ...
    assert scores["hit@1"] == 0.0
    assert scores["precision@1"] == 0.0
    # ... but nDCG still credits its small graded gain
    assert scores["ndcg@1"] > 0.0
