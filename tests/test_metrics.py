import math

from jev_rag.evaluation import RetrievedDoc, hit_rate, mrr, ndcg_at_k, precision_at_k, recall_at_k


def docs(*ids: str) -> list[RetrievedDoc]:
    return [RetrievedDoc(chunk_id=i, text="", score=0.0) for i in ids]


QRELS = {"a": 3, "b": 2, "c": 1}


def test_ndcg_is_one_for_ideal_ranking():
    assert ndcg_at_k(docs("a", "b", "c"), QRELS, k=3) == 1.0


def test_ndcg_penalises_buried_relevant_documents():
    ideal = ndcg_at_k(docs("a", "b", "c"), QRELS, k=3)
    buried = ndcg_at_k(docs("c", "b", "a"), QRELS, k=3)
    assert 0.0 < buried < ideal


def test_ndcg_matches_manual_computation():
    # ranking: c(gain 1), x(0), a(gain 7)
    ranked = docs("c", "x", "a")
    dcg = 1 / math.log2(2) + 0 + 7 / math.log2(4)
    idcg = 7 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranked, QRELS, k=3) == dcg / idcg


def test_ndcg_is_zero_without_relevant_results():
    assert ndcg_at_k(docs("x", "y"), QRELS, k=2) == 0.0


def test_recall_counts_only_graded_relevant_documents():
    assert recall_at_k(docs("a", "x"), QRELS, k=2) == 1 / 3


def test_precision_at_k_uses_k_as_denominator():
    assert precision_at_k(docs("a", "x", "y"), QRELS, k=3) == 1 / 3


def test_mrr_uses_first_relevant_rank():
    assert mrr(docs("x", "b"), QRELS) == 0.5
    assert mrr(docs("x", "y"), QRELS) == 0.0


def test_hit_rate_is_binary():
    assert hit_rate(docs("x", "c"), QRELS, k=2) == 1.0
    assert hit_rate(docs("x", "c"), QRELS, k=1) == 0.0


def test_relevance_threshold_is_configurable():
    # with min_grade=2, "c" (grade 1) no longer counts as relevant
    assert hit_rate(docs("c"), QRELS, k=1, min_grade=2) == 0.0
    assert hit_rate(docs("b"), QRELS, k=1, min_grade=2) == 1.0
