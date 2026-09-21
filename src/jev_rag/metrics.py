"""Ranking metrics for binary relevance.

JQaRA labels a passage 1 when it lets a model answer the question and 0
otherwise, and the published leaderboard reports nDCG@10, so these follow the
same convention and stay comparable with those numbers.
"""

from __future__ import annotations

import math

from jev_rag.types import RetrievedDoc


def ndcg_at_k(ranked: list[RetrievedDoc], k: int, total_relevant: int) -> float:
    """Normalised discounted cumulative gain.

    The ideal ranking puts as many relevant passages as will fit at the top, so
    a query with fewer relevant passages than ``k`` can still reach 1.0.
    """
    dcg = sum(
        doc.label / math.log2(rank + 1) for rank, doc in enumerate(ranked[:k], 1) if doc.label
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, total_relevant) + 1))
    return dcg / ideal if ideal else 0.0


def mrr_at_k(ranked: list[RetrievedDoc], k: int) -> float:
    for rank, doc in enumerate(ranked[:k], 1):
        if doc.label:
            return 1 / rank
    return 0.0


def recall_at_k(ranked: list[RetrievedDoc], k: int, total_relevant: int) -> float:
    if not total_relevant:
        return 0.0
    return sum(1 for doc in ranked[:k] if doc.label) / total_relevant


def score_ranking(ranked: list[RetrievedDoc], total_relevant: int, k: int = 10) -> dict[str, float]:
    return {
        f"ndcg@{k}": ndcg_at_k(ranked, k, total_relevant),
        f"mrr@{k}": mrr_at_k(ranked, k),
        f"recall@{k}": recall_at_k(ranked, k, total_relevant),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def standard_errors(rows: list[dict[str, float]]) -> dict[str, float]:
    """Standard error of each mean.

    A ranking difference is only worth reporting against the spread it was
    measured over; a margin of 0.13 on ten queries is not the same claim as
    the same margin on three hundred.
    """
    if len(rows) < 2:
        return dict.fromkeys(rows[0], 0.0) if rows else {}
    count = len(rows)
    errors = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        errors[key] = math.sqrt(variance / count)
    return errors
