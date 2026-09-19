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


def test_a_regex_rule_can_demand_a_number_next_to_a_unit():
    # a bare "%" matches almost every page of a statistics whitepaper; requiring
    # a number in front of it is what actually identifies a reported figure
    corpus = [
        chunk("hit", "利用率は26.7%であった。"),
        chunk("miss", "割合(%)を示す図表である。"),
    ]
    query = EvalQuery(
        query_id="q",
        question="q",
        relevant={"regex": r"\d+(\.\d+)?\s*%", "grade": 3},
    )
    assert build_qrels(query, corpus) == {"hit": 3}


def test_a_regex_rule_combines_with_keyword_conditions():
    corpus = [
        chunk("both", "生成AIの利用率は26.7%であった。"),
        chunk("number_only", "契約数は38.0%増加した。"),
    ]
    query = EvalQuery(
        query_id="q",
        question="q",
        relevant={"all": ["生成AI"], "regex": r"\d+(\.\d+)?\s*%", "grade": 3},
    )
    assert build_qrels(query, corpus) == {"both": 3}


def test_an_invalid_regex_is_reported_against_its_query():
    query = EvalQuery(query_id="q07", question="q", relevant={"regex": "(", "grade": 3})
    with pytest.raises(ValueError, match="q07"):
        build_qrels(query, CORPUS)


def test_a_regex_only_rule_counts_as_a_labelling_rule():
    query = EvalQuery(query_id="q", question="q", relevant={"regex": "生成AI", "grade": 2})
    assert build_qrels(query, CORPUS)


def test_a_query_can_declare_more_than_two_grades():
    corpus = [
        chunk("direct", "生成AIの利用率は26.7%であった。"),
        chunk("partial_evidence", "生成AIの利用は拡大している。"),
        chunk("topical", "生成AIとは何かを説明する。"),
    ]
    query = EvalQuery(
        query_id="q",
        question="q",
        rules=[
            {"all": ["生成AI", "利用"], "regex": r"\d+(\.\d+)?\s*%", "grade": 3},
            {"all": ["生成AI", "利用"], "grade": 2},
            {"all": ["生成AI"], "grade": 1},
        ],
    )
    assert build_qrels(query, corpus) == {"direct": 3, "partial_evidence": 2, "topical": 1}


def test_the_rules_list_and_the_shorthand_fields_can_be_combined():
    query = EvalQuery(
        query_id="q",
        question="q",
        relevant={"all": ["生成AI", "利用率"], "grade": 3},
        rules=[{"all": ["生成AI"], "grade": 1}],
    )
    assert build_qrels(query, CORPUS) == {"c1": 3, "c2": 1}


def test_a_rules_list_alone_satisfies_the_labelling_requirement():
    query = EvalQuery(query_id="q", question="q", rules=[{"all": ["生成AI"], "grade": 2}])
    assert build_qrels(query, CORPUS) == {"c1": 2, "c2": 2}


def test_min_sentences_separates_prose_from_a_chart_dump():
    # a chart dump carries figures but no sentences, so it cannot be quoted as
    # evidence even though it contains the numbers
    corpus = [
        chunk("prose", "利用頻度について調査した。ほぼ毎日が39.2%であった。"),
        chunk("chart", "図表 利用頻度 39.2 31.9 36.6 37.4 18.5 17.2"),
    ]
    query = EvalQuery(
        query_id="q",
        question="q",
        rules=[{"all": ["利用頻度"], "min_sentences": 2, "grade": 3}],
    )
    assert build_qrels(query, corpus) == {"prose": 3}


def test_min_sentences_counts_japanese_full_stops():
    corpus = [chunk("one", "一文だけである。"), chunk("two", "一文目。二文目。")]
    query = EvalQuery(query_id="q", question="q", rules=[{"min_sentences": 2, "grade": 3}])
    assert build_qrels(query, corpus) == {"two": 3}


def test_min_sentences_combines_with_the_other_conditions():
    corpus = [chunk("hit", "生成AIの利用は26.7%だった。前年から上昇した。")]
    query = EvalQuery(
        query_id="q",
        question="q",
        rules=[{"all": ["生成AI"], "regex": r"[0-9]+\.[0-9]+", "min_sentences": 2, "grade": 3}],
    )
    assert build_qrels(query, corpus) == {"hit": 3}


def test_min_sentences_alone_counts_as_a_labelling_rule():
    query = EvalQuery(query_id="q", question="q", rules=[{"min_sentences": 1, "grade": 1}])
    assert build_qrels(query, CORPUS)
