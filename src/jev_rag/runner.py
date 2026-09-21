"""Run every ranker over the same queries and collect comparable numbers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from jev_rag.dataset import EvalQuery
from jev_rag.metrics import mean_metrics, score_ranking, standard_errors


def _first_relevant(ranked) -> int | None:
    for rank, doc in enumerate(ranked, 1):
        if doc.label:
            return rank
    return None


def evaluate_rankers(
    rankers: dict[str, Any], queries: list[EvalQuery], k: int = 10, progress=None
) -> dict[str, Any]:
    """Score each ranker on each query.

    A ranker that raises is recorded and the run continues: one missing
    credential should not throw away the rows that did succeed.
    """
    results: dict[str, Any] = {}
    for name, ranker in rankers.items():
        started = time.monotonic()
        rows: list[dict[str, Any]] = []
        try:
            for query in queries:
                ranked = ranker.rank(query.question, query.as_documents())
                metrics = score_ranking(ranked, query.total_relevant, k=k)
                rows.append(
                    {
                        "query_id": query.query_id,
                        "question": query.question,
                        "first_relevant": _first_relevant(ranked),
                        "top": [doc.doc_id for doc in ranked[:5]],
                        **metrics,
                    }
                )
                if progress:
                    progress(name, len(rows), len(queries))
        except Exception as error:  # noqa: BLE001 - reported per ranker, run continues
            results[name] = {"error": f"{type(error).__name__}: {error}"}
            continue
        scores = [{m: r[m] for m in r if "@" in m} for r in rows]
        results[name] = {
            "metrics": mean_metrics(scores),
            "stderr": standard_errors(scores),
            "queries": len(rows),
            "per_query": rows,
            "seconds": round(time.monotonic() - started, 1),
        }
    return results


def save(results: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise SystemExit(f"{target} がありません。先に `jev-rag evaluate` を実行してください。")
    return json.loads(target.read_text(encoding="utf-8"))
