"""Ranking metrics for binary relevance, the convention JQaRA is scored with."""

import pytest

from jev_rag.metrics import mrr_at_k, ndcg_at_k, recall_at_k, score_ranking
from jev_rag.types import RetrievedDoc


def ranked(*labels: int) -> list[RetrievedDoc]:
    return [
        RetrievedDoc(doc_id=str(i), text="", score=0.0, label=label)
        for i, label in enumerate(labels)
    ]


def test_a_perfect_ranking_scores_one():
    assert ndcg_at_k(ranked(1, 1, 0, 0), k=2, total_relevant=2) == 1.0


def test_a_ranking_that_buries_the_answer_scores_less():
    assert ndcg_at_k(ranked(0, 1), k=2, total_relevant=1) < 1.0


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved():
    assert ndcg_at_k(ranked(0, 0), k=2, total_relevant=1) == 0.0


def test_ndcg_matches_the_textbook_value():
    # one relevant document in second place: DCG = 1/log2(3), IDCG = 1/log2(2)
    import math

    assert ndcg_at_k(ranked(0, 1), k=2, total_relevant=1) == pytest.approx(1 / math.log2(3))


def test_the_ideal_ranking_accounts_for_relevant_documents_beyond_k():
    # 3 relevant overall but only 2 can fit in the cut-off
    perfect = ndcg_at_k(ranked(1, 1, 1), k=2, total_relevant=3)
    assert perfect == 1.0


def test_mrr_uses_the_first_relevant_rank():
    assert mrr_at_k(ranked(0, 1), k=10) == 0.5
    assert mrr_at_k(ranked(1, 0), k=10) == 1.0


def test_mrr_ignores_relevant_documents_past_the_cut_off():
    assert mrr_at_k(ranked(0, 0, 1), k=2) == 0.0


def test_recall_is_against_every_relevant_document():
    assert recall_at_k(ranked(1, 0), k=2, total_relevant=4) == 0.25


def test_recall_of_a_query_with_no_relevant_documents_is_zero():
    assert recall_at_k(ranked(0), k=1, total_relevant=0) == 0.0


def test_score_ranking_reports_the_standard_set():
    scores = score_ranking(ranked(1, 0, 1), total_relevant=2, k=10)
    assert set(scores) == {"ndcg@10", "mrr@10", "recall@10"}
    assert scores["recall@10"] == 1.0
