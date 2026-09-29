"""Every table in the article, rebuilt from the saved scores without calling an API."""

import json

import pytest

from jev_rag import cli
from jev_rag.judge import paired_bootstrap
from jev_rag.judge_report import (
    format_report,
    mean_query_pr_auc,
    precision_at_recall,
    top_k_hit,
)


def rows(query_id, scored):
    return [
        {"query_id": query_id, "doc_id": f"{query_id}-{i}", "label": label, "score": score}
        for i, (score, label) in enumerate(scored)
    ]


# --- per-question PR-AUC -----------------------------------------------------


def test_the_per_question_value_averages_each_question_separately():
    # q1 is perfect; q2 puts its relevant document second (AP = 1/2)
    table = rows("q1", [(0.9, 1), (0.1, 0)]) + rows("q2", [(0.9, 0), (0.8, 1)])
    assert mean_query_pr_auc(table) == pytest.approx(0.75)


def test_a_question_whose_candidates_are_all_relevant_is_left_out():
    table = rows("q1", [(0.9, 1), (0.1, 0)]) + rows("q2", [(0.5, 1), (0.4, 1)])
    assert mean_query_pr_auc(table) == pytest.approx(1.0)


def test_the_per_question_value_ignores_how_scores_compare_across_questions():
    # the same ordering inside each question, but question 2 is scored far lower
    table = rows("q1", [(0.9, 1), (0.8, 0)]) + rows("q2", [(0.2, 1), (0.1, 0)])
    assert mean_query_pr_auc(table) == pytest.approx(1.0)


# --- one threshold for every question -------------------------------------------


def test_the_threshold_keeps_the_requested_share_of_relevant_documents():
    table = rows("q", [(0.9, 1), (0.8, 0), (0.7, 1), (0.1, 0)])
    precision, threshold = precision_at_recall(table, keep=0.9)
    # both relevant documents are needed to reach 90%, so the cut falls at 0.7
    assert threshold == pytest.approx(0.7)
    assert precision == pytest.approx(2 / 3)


def test_every_document_tied_at_the_threshold_is_kept():
    table = rows("q", [(0.9, 1), (0.5, 1), (0.5, 0), (0.5, 0), (0.1, 0)])
    precision, threshold = precision_at_recall(table, keep=0.9)
    assert threshold == pytest.approx(0.5)
    # the two irrelevant documents at 0.5 are kept with the relevant one
    assert precision == pytest.approx(2 / 4)


# --- the top of each question's ranking -----------------------------------------


def test_top_k_counts_questions_with_a_relevant_document_in_the_first_k():
    table = rows("q1", [(0.9, 1), (0.1, 0)]) + rows("q2", [(0.9, 0), (0.8, 0), (0.7, 1)])
    assert top_k_hit(table, 1) == pytest.approx(0.5)
    assert top_k_hit(table, 3) == pytest.approx(1.0)


# --- intervals for any of these measures ----------------------------------------


def test_the_bootstrap_accepts_the_per_question_measure():
    better = [r for q in range(20) for r in rows(f"q{q}", [(0.9, 1), (0.1, 0)])]
    worse = [r for q in range(20) for r in rows(f"q{q}", [(0.1, 1), (0.9, 0)])]
    result = paired_bootstrap(better, worse, samples=200, seed=0, metric=mean_query_pr_auc)
    assert result["difference"] == pytest.approx(0.5)
    assert result["low"] > 0


# --- the report ------------------------------------------------------------------


def judge_file(routes):
    return {name: {"scored": {"pairs": table}} for name, table in routes.items()}


def small_judge():
    good = [r for q in range(10) for r in rows(f"q{q}", [(0.9, 1), (0.4, 0), (0.2, 0)])]
    poor = [r for q in range(10) for r in rows(f"q{q}", [(0.3, 1), (0.6, 0), (0.2, 0)])]
    names = ["bm25", "embedding", "cohere_rerank", "jev_pointwise_plain"]
    return judge_file({"jev_pointwise": good, **{name: poor for name in names}})


def test_the_report_has_every_table_the_article_uses():
    text = format_report({"miracl": small_judge()}, dilution=None, samples=100, seed=0)
    for heading in [
        "並べ替える力",
        "質問をまたいで",
        "上位に関連文書",
        "1 つのしきい値",
        "判定基準のありなし",
    ]:
        assert heading in text
    assert "MIRACL" in text


def test_a_negative_difference_uses_the_minus_sign_of_the_article():
    from jev_rag.judge_report import _diff

    assert _diff(0.931, 0.932, 3) == "−0.001"


def test_the_report_skips_a_route_that_was_not_measured():
    data = small_judge()
    del data["jev_pointwise_plain"]
    text = format_report({"miracl": data}, dilution=None, samples=50, seed=0)
    assert "判定基準のありなし" not in text


def test_the_command_reads_the_saved_files(tmp_path, capsys):
    (tmp_path / "judge-miracl.json").write_text(json.dumps(small_judge()), encoding="utf-8")
    code = cli.main(["report-judge", "--results-dir", str(tmp_path), "--samples", "50"])
    assert code == 0
    assert "MIRACL" in capsys.readouterr().out


def test_the_command_fails_when_nothing_has_been_measured(tmp_path, capsys):
    assert cli.main(["report-judge", "--results-dir", str(tmp_path)]) == 1
    assert "jev-rag judge" in capsys.readouterr().out
