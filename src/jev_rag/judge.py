"""Each route as a relevance judge.

Every dataset here labels each (query, document) pair, so a route's scores can
be evaluated as a binary classifier rather than only as an ordering. The distinction matters
for the claim "Jev as a Judge":

- per-query ROC-AUC asks whether a route orders one query's candidates. That
  is all a reranker needs.
- global PR-AUC and ROC-AUC pool every pair, so they ask whether the score
  means the same thing across queries. A judge needs that: a threshold is only
  useful if 0.8 is 0.8 whatever the query.

PR-AUC is the headline because relevant documents are the minority of each
pool, and ROC-AUC flatters a skew like that. Both are unchanged by any monotone rescaling
of the score, so a route that does not emit probabilities is not penalised.

Calibration — ECE and Brier score — is computed only for routes that emit a
probability. TypeSafe documents Noul answers as calibrated; nothing else in
the comparison claims to be, and an ECE for a score that is not a probability
would measure nothing.

The negatives are hard in all three datasets — judged retrieval results in
MIRACL, near misses written to share the question's terms in J-RAGBench and
the synthetic set — so absolute AUCs sit below what a random-negative
benchmark would report.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from jev_rag.dataset import EvalQuery
from jev_rag.runner import save
from jev_rag.synthetic import baseline_scores


def score_pairs(ranker: Any, queries: list[EvalQuery]) -> dict[str, Any]:
    """Every candidate's score beside its gold label, one row per pair.

    The expensive part is the scoring; every metric below is computed from this
    table, so a new metric never needs the API again.
    """
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    dropped = 0
    for query in queries:
        try:
            ranked = ranker.rank(query.question, query.as_documents())
        except Exception:  # noqa: BLE001 - one throttled query must not end the run
            failed.append(query.query_id)
            continue
        # a route returning fewer candidates than it was given would bias the
        # pooled metrics toward whatever it kept
        dropped += len(query.candidates) - len(ranked)
        rows.extend(
            {
                "query_id": query.query_id,
                "doc_id": doc.doc_id,
                "label": int(doc.label),
                "score": float(doc.score),
            }
            for doc in ranked
        )
    return {
        "pairs": rows,
        "failed": failed,
        "dropped": dropped,
        "seconds": round(time.monotonic() - started, 1),
    }


def calibration_bins(rows: list[dict[str, Any]], bins: int = 10) -> list[dict[str, float]]:
    """Equal-count bins over the predicted probability.

    Equal-width bins would leave most of them empty: nearly every candidate is
    irrelevant, so the probabilities pile up near zero.
    """
    order = np.argsort([row["score"] for row in rows], kind="stable")
    table = []
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        scores = np.array([rows[i]["score"] for i in chunk])
        labels = np.array([rows[i]["label"] for i in chunk])
        table.append(
            {
                "count": int(len(chunk)),
                "mean_score": float(scores.mean()),
                "positive_rate": float(labels.mean()),
            }
        )
    return table


def judge_metrics(
    rows: list[dict[str, Any]], probabilistic: bool, bins: int = 10
) -> dict[str, Any]:
    labels = np.array([row["label"] for row in rows])
    scores = np.array([row["score"] for row in rows])
    result: dict[str, Any] = {"pairs": len(rows), "positive_rate": float(labels.mean())}

    if len(set(labels.tolist())) < 2:
        return result | {"error": "正例か負例の一方しかない"}

    result["pr_auc"] = float(average_precision_score(labels, scores))
    result["roc_auc"] = float(roc_auc_score(labels, scores))

    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    per_query = [
        roc_auc_score([r["label"] for r in group], [r["score"] for r in group])
        for group in by_query.values()
        if len({r["label"] for r in group}) == 2
    ]
    result["per_query_roc_auc"] = float(np.mean(per_query)) if per_query else None
    result["per_query_count"] = len(per_query)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = int(np.argmax(f1))
    result["best_f1"] = float(f1[best])
    result["best_f1_threshold"] = float(thresholds[best])
    result["best_f1_precision"] = float(precision[best])
    result["best_f1_recall"] = float(recall[best])

    if probabilistic:
        predicted = scores >= 0.5
        true_positive = int((predicted & (labels == 1)).sum())
        result["precision_at_half"] = true_positive / max(int(predicted.sum()), 1)
        result["recall_at_half"] = true_positive / max(int((labels == 1).sum()), 1)
        table = calibration_bins(rows, bins=bins)
        total = sum(b["count"] for b in table)
        result["ece"] = sum(
            b["count"] / total * abs(b["positive_rate"] - b["mean_score"]) for b in table
        )
        result["brier"] = float(np.mean((scores - labels) ** 2))
        result["calibration"] = table
    return result


def run_judge(rankers: dict[str, Any], queries: list[EvalQuery], out: str | Path) -> dict[str, Any]:
    """Score every route once and save after each, so a rerun resumes.

    A route already measured in ``out`` is not paid for again; one that failed
    every query is. The trivial baselines are added for free: if BM25 judges as
    well as a model, the dataset is not testing judgement.
    """
    target = Path(out)
    results = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    for name, rows in baseline_scores(queries).items():
        scored = {"pairs": rows, "failed": [], "dropped": 0, "seconds": 0.0}
        results[name] = judge_metrics(rows, probabilistic=False) | {"scored": scored}
    for name, ranker in rankers.items():
        if "pr_auc" in results.get(name, {}):
            continue
        scored = score_pairs(ranker, queries)
        if scored["pairs"]:
            metrics = judge_metrics(scored["pairs"], probabilistic=name.startswith("jev"))
        else:
            metrics = {"error": "全問失敗"}
        results[name] = metrics | {"scored": scored}
        save(results, target)
    save(results, target)
    return results


def format_judge(results: dict[str, Any]) -> str:
    header = [
        "経路",
        "PR-AUC",
        "ROC-AUC",
        "問題内ROC-AUC",
        "最良F1(閾値)",
        "P/R@0.5",
        "ECE",
        "Brier",
        "ペア数",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    def number(value, digits=3):
        return "-" if value is None else f"{value:.{digits}f}"

    for name, result in results.items():
        if name.startswith("_"):
            continue
        if "error" in result and "pr_auc" not in result:
            cells = [name, result["error"]] + [""] * (len(header) - 2)
            lines.append("| " + " | ".join(cells) + " |")
            continue
        half = (
            f"{result['precision_at_half']:.3f}/{result['recall_at_half']:.3f}"
            if "precision_at_half" in result
            else "-"
        )
        cells = [
            name,
            number(result.get("pr_auc")),
            number(result.get("roc_auc")),
            number(result.get("per_query_roc_auc")),
            f"{number(result.get('best_f1'))}({number(result.get('best_f1_threshold'), 2)})",
            half,
            number(result.get("ece")),
            number(result.get("brier")),
            str(result.get("pairs", "-")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
