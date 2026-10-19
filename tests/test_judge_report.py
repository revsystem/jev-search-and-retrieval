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


def measured(rows_, models, tokens, refusals=0):
    return {
        "scored": {
            "pairs": rows_,
            "failed": [],
            "usage": {
                "requests": 1,
                "input_tokens": tokens,
                "refusals": refusals,
                "missing": 0,
                "models": models,
                "request_seconds": [0.1],
            },
        }
    }


def decisions_judges():
    jev = rows("q1", [(0.9, 1), (0.1, 0)]) + rows("q2", [(0.8, 1), (0.3, 0)])
    decisions = rows("q1", [(0.7, 1), (0.6, 0)]) + rows("q2", [(0.4, 1), (0.5, 0)])
    return {
        "synthetic": {
            "jev_pointwise": measured(jev, ["jev-1.14.0"], 1_000_000),
            "decisions_pointwise": measured(decisions, ["gpt-6-luna"], 2_000_000, refusals=1),
        }
    }


def test_the_report_compares_decisions_with_jev():
    report = format_report(decisions_judges(), None, samples=50)
    assert "### Decisions と Jev 1.14" in report
    assert "jev-1.14.0" in report and "gpt-6-luna" in report
    # Jev bills $0.042 and Decisions $0.10 per million input tokens
    assert "0.0420" in report and "0.2000" in report
    assert "| 1/0/0 |" in report
    assert "Decisions − Jev" in report


def test_the_decisions_section_is_left_out_without_decisions_routes():
    judges = decisions_judges()
    del judges["synthetic"]["decisions_pointwise"]
    assert "Decisions と Jev" not in format_report(judges, None, samples=50)


def test_the_difference_is_taken_over_questions_both_routes_answered():
    judges = decisions_judges()
    entry = judges["synthetic"]["decisions_pointwise"]["scored"]
    entry["pairs"] = [r for r in entry["pairs"] if r["query_id"] == "q1"]
    entry["failed"] = ["q2"]
    report = format_report(judges, None, samples=50)
    # on q1 alone Jev scores 1.0 and Decisions 1.0, so the difference is zero
    assert "1 問で比較" in report
    assert "PR-AUC +0.000" in report


def test_routes_with_no_question_in_common_are_reported_as_not_comparable():
    judges = decisions_judges()
    for route, kept in (("jev_pointwise", "q1"), ("decisions_pointwise", "q2")):
        entry = judges["synthetic"][route]["scored"]
        entry["pairs"] = [r for r in entry["pairs"] if r["query_id"] == kept]
    report = format_report(judges, None, samples=50)
    assert "比較できる問題がない" in report


def test_sections_with_no_rows_are_left_out():
    report = format_report(decisions_judges(), None, samples=50)
    assert "Cohere Rerank" not in report
    assert report.startswith("### Decisions と Jev 1.14")


def test_the_question_count_is_the_questions_the_difference_was_computed_on():
    judges = decisions_judges()
    for route in ("jev_pointwise", "decisions_pointwise"):
        # q3 has only relevant candidates, so it has no per-question PR-AUC
        judges["synthetic"][route]["scored"]["pairs"] += rows("q3", [(0.9, 1), (0.8, 1)])
    report = format_report(judges, None, samples=50)
    assert "2 問で比較" in report
