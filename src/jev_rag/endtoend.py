"""The end-to-end comparison.

The ranking comparison says a route puts better passages at the top. This says
whether that changes the answer a RAG pipeline gives, which is the thing
anyone actually wants to know. Everything but the ranking route is held fixed:
the same questions, the same candidate pool, the same top-k, the same prompt,
the same generation model.

``evidence_in_prompt`` is recorded beside the score so a wrong answer can be
attributed. If the relevant passage was in the prompt and the answer is still
wrong, the ranking is not what failed.
"""

from __future__ import annotations

import time
from typing import Any

from jev_rag.answers import score_answer
from jev_rag.dataset import EvalQuery
from jev_rag.metrics import mean_metrics, standard_errors


def answer_queries(
    ranker: Any,
    queries: list[EvalQuery],
    generator: Any,
    top_k: int = 5,
    progress=None,
) -> dict[str, Any]:
    """Rank, hand the top passages to the generator, score against the gold answer."""
    started = time.monotonic()
    rows: list[dict[str, Any]] = []

    for query in queries:
        if not query.answers:
            continue
        ranked = ranker.rank(query.question, query.as_documents())
        context = ranked[:top_k]
        evidence = any(doc.label for doc in context)

        row: dict[str, Any] = {
            "query_id": query.query_id,
            "question": query.question,
            "gold": list(query.answers),
            "evidence_in_prompt": evidence,
        }
        try:
            generated = generator.answer(query.question, context)
        except Exception as error:  # noqa: BLE001 - one throttled call must not end the run
            row.update(answer="", correct=False, error=f"{type(error).__name__}: {error}")
        else:
            verdict = score_answer(generated, query.answers)
            row.update(answer=generated, correct=verdict.correct, matched=verdict.matched)
        rows.append(row)
        if progress:
            progress(getattr(ranker, "name", "?"), len(rows), len(queries))

    scores = [
        {"accuracy": float(row["correct"]), "evidence_rate": float(row["evidence_in_prompt"])}
        for row in rows
    ]
    means = mean_metrics(scores)
    return {
        "accuracy": means.get("accuracy", 0.0),
        "evidence_rate": means.get("evidence_rate", 0.0),
        "stderr": standard_errors(scores),
        "queries": len(rows),
        "seconds": round(time.monotonic() - started, 1),
        "per_query": rows,
    }


def format_answers(results: dict[str, Any], baseline: str) -> str:
    """One row per pipeline: how often the answer was right."""
    reference = (results.get(baseline) or {}).get("accuracy")
    lines = [
        "| 構成 | 回答正解率 | 正解文書がプロンプトに入った割合 | n | 秒 |",
        "|---|---|---|---|---|",
    ]
    for name, result in results.items():
        if "error" in result:
            lines.append(f"| {name} | {result['error']} | | | |")
            continue
        error = (result.get("stderr") or {}).get("accuracy", 0.0)
        cell = f"{result['accuracy']:.3f}±{error:.3f}"
        if name != baseline and reference is not None:
            cell += f" ({result['accuracy'] - reference:+.3f})"
        lines.append(
            f"| {name} | {cell} | {result['evidence_rate']:.3f} | "
            f"{result['queries']} | {result.get('seconds', '-')} |"
        )
    return "\n".join(lines)
