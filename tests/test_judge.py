"""Evaluating each route as a relevance judge, not only as a ranker.

Every dataset labels each (query, document) pair, so a route's scores can be scored as
a binary classifier. Two views are kept apart because they answer different
questions:

- per-query ROC-AUC: can the route order one query's candidates? That is all
  a reranker needs.
- global PR-AUC / ROC-AUC over every pair pooled: does the score mean the same
  thing across queries? A judge needs that, because a threshold is only useful
  if 0.8 is 0.8 whatever the query.

Calibration (ECE, Brier) is computed only for routes that emit a probability.
"""

import pytest

from jev_rag.dataset import EvalQuery
from jev_rag.judge import (
    calibration_bins,
    format_judge,
    judge_metrics,
    paired_bootstrap,
    run_dilution,
    run_judge,
    score_pairs,
)
from jev_rag.types import RetrievedDoc


def pairs(scores_labels, query_id="q"):
    return [
        {"query_id": query_id, "doc_id": f"{query_id}-{i}", "label": label, "score": score}
        for i, (score, label) in enumerate(scores_labels)
    ]


# --- discrimination -------------------------------------------------------


def test_perfect_separation_scores_one():
    result = judge_metrics(pairs([(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)]), probabilistic=False)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)


def test_reversed_scores_have_zero_roc_auc():
    result = judge_metrics(pairs([(0.1, 1), (0.9, 0)]), probabilistic=False)
    assert result["roc_auc"] == pytest.approx(0.0)


def test_a_constant_score_has_chance_roc_auc():
    result = judge_metrics(pairs([(0.5, 1), (0.5, 0), (0.5, 0)]), probabilistic=False)
    assert result["roc_auc"] == pytest.approx(0.5)


def test_pr_auc_reflects_class_imbalance_unlike_roc_auc():
    # a constant score has chance ROC-AUC 0.5, but PR-AUC falls to the base rate
    rows = pairs([(0.5, 1)] + [(0.5, 0)] * 9)
    result = judge_metrics(rows, probabilistic=False)
    assert result["pr_auc"] == pytest.approx(0.1)


def test_per_query_auc_measures_ordering_within_each_query():
    # each query is perfectly ordered, but the scales disagree across queries:
    # query a's positives sit below query b's negatives
    rows = pairs([(0.3, 1), (0.1, 0)], "a") + pairs([(0.9, 1), (0.5, 0)], "b")
    result = judge_metrics(rows, probabilistic=False)
    assert result["per_query_roc_auc"] == pytest.approx(1.0)
    assert result["roc_auc"] < 1.0


def test_a_query_with_only_one_class_is_left_out_of_the_per_query_mean():
    rows = pairs([(0.9, 1), (0.1, 0)], "a") + pairs([(0.5, 0), (0.4, 0)], "b")
    result = judge_metrics(rows, probabilistic=False)
    assert result["per_query_roc_auc"] == pytest.approx(1.0)
    assert result["per_query_count"] == 1


def test_the_best_f1_threshold_is_reported_with_its_operating_point():
    rows = pairs([(0.9, 1), (0.7, 1), (0.6, 0), (0.2, 0)])
    result = judge_metrics(rows, probabilistic=False)
    assert result["best_f1"] == pytest.approx(1.0)
    assert result["best_f1_threshold"] == pytest.approx(0.7)


def test_positive_rate_is_reported():
    result = judge_metrics(pairs([(0.9, 1), (0.1, 0), (0.2, 0), (0.3, 0)]), probabilistic=False)
    assert result["positive_rate"] == pytest.approx(0.25)
    assert result["pairs"] == 4


# --- calibration ----------------------------------------------------------


def test_calibration_is_omitted_for_a_score_that_is_not_a_probability():
    result = judge_metrics(pairs([(3.2, 1), (-1.0, 0)]), probabilistic=False)
    assert "ece" not in result
    assert "brier" not in result


def test_a_probabilistic_route_reports_its_fixed_half_threshold():
    # Noul's 0.5 means "as likely yes as no"; a judge should be usable there
    rows = pairs([(0.9, 1), (0.6, 0), (0.4, 1), (0.1, 0)])
    result = judge_metrics(rows, probabilistic=True, bins=2)
    assert result["precision_at_half"] == pytest.approx(0.5)
    assert result["recall_at_half"] == pytest.approx(0.5)


def test_perfect_calibration_has_zero_ece():
    # within each bin the mean probability equals the observed rate
    rows = pairs([(0.0, 0), (0.0, 0), (1.0, 1), (1.0, 1)])
    result = judge_metrics(rows, probabilistic=True, bins=2)
    assert result["ece"] == pytest.approx(0.0)


def test_overconfidence_shows_up_in_ece():
    # says 0.9 for everything, right half the time: ECE is |0.5 - 0.9| = 0.4
    rows = pairs([(0.9, 1), (0.9, 0), (0.9, 1), (0.9, 0)])
    result = judge_metrics(rows, probabilistic=True, bins=1)
    assert result["ece"] == pytest.approx(0.4)


def test_brier_score_is_the_mean_squared_error_of_the_probability():
    # two right with full confidence, two wrong with full confidence
    rows = pairs([(1.0, 1), (0.0, 0), (0.0, 1), (1.0, 0)])
    result = judge_metrics(rows, probabilistic=True, bins=1)
    assert result["brier"] == pytest.approx(0.5)


def test_bins_hold_equal_counts_so_crowded_low_scores_do_not_leave_bins_empty():
    rows = pairs([(0.01, 0)] * 8 + [(0.99, 1)] * 2)
    table = calibration_bins(rows, bins=5)
    assert [b["count"] for b in table] == [2, 2, 2, 2, 2]


def test_each_bin_reports_mean_probability_and_observed_rate():
    rows = pairs([(0.2, 0), (0.4, 1)])
    (only,) = calibration_bins(rows, bins=1)
    assert only["mean_score"] == pytest.approx(0.3)
    assert only["positive_rate"] == pytest.approx(0.5)


# --- scoring every pair ---------------------------------------------------


def query():
    return EvalQuery(
        query_id="q1",
        question="問い",
        candidates=[
            RetrievedDoc(doc_id="a", text="正解", score=0.0, label=1),
            RetrievedDoc(doc_id="b", text="不正解", score=0.0, label=0),
        ],
    )


class FixedScores:
    name = "fixed"

    def rank(self, question, docs):
        for doc in docs:
            doc.score = 0.9 if doc.doc_id == "a" else 0.2
        return sorted(docs, key=lambda d: d.score, reverse=True)


def test_every_candidate_is_recorded_with_its_label_and_score():
    rows = score_pairs(FixedScores(), [query()])["pairs"]
    assert {(r["doc_id"], r["label"], r["score"]) for r in rows} == {("a", 1, 0.9), ("b", 0, 0.2)}


def test_scoring_does_not_mutate_the_shared_candidates():
    shared = [query()]
    score_pairs(FixedScores(), shared)
    assert all(c.score == 0.0 for c in shared[0].candidates)


def test_a_failing_query_is_recorded_and_the_rest_continue():
    class Flaky(FixedScores):
        def __init__(self):
            self.calls = 0

        def rank(self, question, docs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("throttled")
            return super().rank(question, docs)

    second = query()
    second.query_id = "q2"
    result = score_pairs(Flaky(), [query(), second])
    assert result["failed"] == ["q1"]
    assert {r["query_id"] for r in result["pairs"]} == {"q2"}


def test_the_table_has_one_row_per_route_and_consistent_columns():
    good = judge_metrics(pairs([(0.9, 1), (0.1, 0)]), probabilistic=True, bins=1)
    plain = judge_metrics(pairs([(0.9, 1), (0.1, 0)]), probabilistic=False)
    table = format_judge({"jev": good, "cohere": plain, "broken": {"error": "boom"}})
    rows = [line for line in table.splitlines() if line.startswith("|")]
    assert len({line.count("|") for line in rows}) == 1
    assert "jev" in table and "cohere" in table and "boom" in table


# --- a whole run --------------------------------------------------------------


class Constant:
    """Scores relevant candidates 0.9 and the rest 0.1, counting its calls."""

    def __init__(self):
        self.calls = 0

    def rank(self, question, docs):
        self.calls += 1
        for doc in docs:
            doc.score = 0.9 if doc.label else 0.1
        return sorted(docs, key=lambda d: -d.score)


def two_queries():
    return [
        EvalQuery(
            query_id=q,
            question="Q",
            candidates=[
                RetrievedDoc(doc_id=f"{q}-a", text="a", score=0.0, label=1),
                RetrievedDoc(doc_id=f"{q}-b", text="b", score=0.0, label=0),
            ],
        )
        for q in ("q1", "q2")
    ]


def test_a_run_keeps_the_pairs_and_the_metrics(tmp_path):
    out = tmp_path / "judge.json"
    results = run_judge({"jev_pointwise": Constant()}, two_queries(), out)
    route = results["jev_pointwise"]
    assert route["pr_auc"] == pytest.approx(1.0)
    assert len(route["scored"]["pairs"]) == 4
    assert "ece" in route  # a Jev route emits probabilities


def test_a_run_adds_the_trivial_baselines_without_calibration(tmp_path):
    results = run_judge({}, two_queries(), tmp_path / "judge.json")
    assert {"bm25", "term_overlap", "length"} <= set(results)
    assert "ece" not in results["bm25"]


def test_a_rerun_skips_routes_already_scored(tmp_path):
    out = tmp_path / "judge.json"
    run_judge({"cohere_rerank": Constant()}, two_queries(), out)
    again = Constant()
    run_judge({"cohere_rerank": again}, two_queries(), out)
    assert again.calls == 0


def test_a_route_that_failed_every_query_is_retried(tmp_path):
    class Broken:
        def rank(self, question, docs):
            raise RuntimeError("throttled")

    out = tmp_path / "judge.json"
    first = run_judge({"cohere_rerank": Broken()}, two_queries(), out)
    assert "error" in first["cohere_rerank"]
    retry = Constant()
    run_judge({"cohere_rerank": retry}, two_queries(), out)
    assert retry.calls == 2


# --- is a difference more than noise? ----------------------------------------


def many_queries(better_margin):
    rows_a, rows_b = [], []
    for q in range(30):
        for i, label in enumerate([1, 0, 0, 0]):
            base = {"query_id": f"q{q}", "doc_id": f"q{q}-{i}", "label": label}
            rows_a.append(base | {"score": (0.9 if label else 0.1) + 0.01 * i})
            # b ranks one negative per query above every positive when the margin is on
            confused = better_margin and i == 1
            score = 0.95 if confused else (0.9 if label else 0.1) - 0.01 * i
            rows_b.append(base | {"score": score})
    return rows_a, rows_b


def test_a_clear_difference_has_an_interval_above_zero():
    a, b = many_queries(better_margin=True)
    result = paired_bootstrap(a, b, samples=500, seed=0)
    assert result["difference"] > 0
    assert result["low"] > 0


def test_identical_routes_have_an_interval_around_zero():
    a, _ = many_queries(better_margin=False)
    result = paired_bootstrap(a, a, samples=200, seed=0)
    assert result["difference"] == pytest.approx(0.0)
    assert result["low"] <= 0 <= result["high"]


def test_the_bootstrap_uses_only_queries_both_routes_scored():
    a, b = many_queries(better_margin=True)
    result = paired_bootstrap(a, [r for r in b if r["query_id"] != "q0"], samples=50, seed=0)
    assert result["queries"] == 29


# --- how the pool around a document changes its score ----------------------


def own_and_foreign():
    queries = two_queries()
    for q in queries:
        q.candidates.append(RetrievedDoc(doc_id="other-x", text="x", score=0.0, label=0))
    return queries


def test_dilution_scores_only_each_question_s_own_documents(tmp_path):
    results = run_dilution(
        lambda batch: Constant(), {(3, 2): own_and_foreign()}, tmp_path / "d.json"
    )
    (entry,) = results.values()
    assert entry["pairs"] == 4
    assert entry["distractors"]["count"] == 2


def test_dilution_reports_how_distractors_were_scored(tmp_path):
    results = run_dilution(
        lambda batch: Constant(), {(3, 2): own_and_foreign()}, tmp_path / "d.json"
    )
    (entry,) = results.values()
    assert entry["distractors"]["mean_score"] == pytest.approx(0.1)
    assert entry["distractors"]["share_at_half"] == 0.0
    assert entry["requests"] == 4  # two queries, three documents, two per request


def test_dilution_resumes_finished_settings(tmp_path):
    out = tmp_path / "d.json"
    run_dilution(lambda batch: Constant(), {(3, 2): own_and_foreign()}, out)
    again = Constant()
    run_dilution(lambda batch: again, {(3, 2): own_and_foreign()}, out)
    assert again.calls == 0


def test_a_route_that_times_its_requests_has_the_times_saved(tmp_path):
    class Timed(Constant):
        def __init__(self):
            super().__init__()
            self.latencies_ms = []

        def rank(self, question, docs):
            self.latencies_ms.append(42.0)
            return super().rank(question, docs)

    results = run_judge({"decider_single": Timed()}, two_queries(), tmp_path / "judge.json")
    assert results["decider_single"]["scored"]["latencies_ms"] == [42.0, 42.0]
