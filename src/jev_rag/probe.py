"""Do a route's probabilities follow the documents or their positions?

A batched route names each candidate by its position in the request
(`candidates[3].text`). If the model does not resolve those paths, reversing
the candidates moves the probabilities with the positions, not with the
documents. strands-decider failed exactly this way. Each route is run on the
same questions in both orders and the per-document probabilities are compared.

A one-at-a-time route sees the same input in both orders, so its disagreement
is run-to-run noise; read it first as the floor for the batched routes.
Thresholds are fixed in .claude/docs/specs/openai-decisions-comparison.md.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from jev_rag.dataset import EvalQuery
from jev_rag.judge import score_pairs
from jev_rag.judge_report import mean_query_pr_auc
from jev_rag.runner import save

MAX_MEDIAN_DIFF = 0.1
MIN_SPEARMAN = 0.8
MAX_BATCH_GAP = 0.2


def reversed_queries(queries: list[EvalQuery]) -> list[EvalQuery]:
    return [replace(q, candidates=list(reversed(q.candidates))) for q in queries]


def order_agreement(
    forward: list[dict[str, Any]], backward: list[dict[str, Any]]
) -> dict[str, Any]:
    other = {(r["query_id"], r["doc_id"]): r["score"] for r in backward}
    pairs = [(r["score"], other[(r["query_id"], r["doc_id"])]) for r in forward]
    a, b = (np.array(side, dtype=float) for side in zip(*pairs, strict=True))
    return {
        "pairs": len(pairs),
        "median_abs_diff": float(np.median(np.abs(a - b))),
        # a constant score has no rank order; nan fails the check below
        "spearman": float(spearmanr(a, b).statistic),
    }


def follows_documents(agreement: dict[str, Any]) -> bool:
    rho = agreement["spearman"]
    return (
        agreement["median_abs_diff"] <= MAX_MEDIAN_DIFF
        and not math.isnan(rho)
        and rho >= MIN_SPEARMAN
    )


def resolves_paths(batched: float, single: float) -> bool:
    """A batched route far below its one-at-a-time counterpart is not reading the paths."""
    return round(single - batched, 10) <= MAX_BATCH_GAP


def run_probe(rankers: dict[str, Any], queries: list[EvalQuery], out: str | Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    flipped = reversed_queries(queries)
    for name, ranker in rankers.items():
        forward, backward = score_pairs(ranker, queries), score_pairs(ranker, flipped)
        failed = forward["failed"] + backward["failed"]
        if not forward["pairs"] or not backward["pairs"]:
            results[name] = {"error": "全問失敗", "failed": failed, "follows_documents": False}
            save(results, out)
            continue
        agreement = order_agreement(forward["pairs"], backward["pairs"])
        results[name] = agreement | {
            "follows_documents": follows_documents(agreement),
            "mean_query_pr_auc": mean_query_pr_auc(forward["pairs"]),
            "failed": failed,
            "forward": forward["pairs"],
            "backward": backward["pairs"],
        }
        save(results, out)
    return results
