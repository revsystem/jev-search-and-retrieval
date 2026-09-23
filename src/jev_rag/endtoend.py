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
            # Not a wrong answer: the ranking never got its chance. Counting it
            # as one would move a route's score for a reason unrelated to the
            # thing being compared.
            row.update(
                answer="",
                correct=False,
                contains=False,
                parsed=False,
                error=f"{type(error).__name__}: {error}",
            )
        else:
            verdict = score_answer(generated, query.answers)
            row.update(
                answer=generated,
                extracted=verdict.extracted,
                correct=verdict.correct,
                contains=verdict.contains,
                parsed=verdict.parsed,
                matched=verdict.matched,
            )
        rows.append(row)
        if progress:
            progress(getattr(ranker, "name", "?"), len(rows), len(queries))

    scored = [row for row in rows if "error" not in row]
    scores = [
        {
            "accuracy": float(row["correct"]),
            "contains": float(row.get("contains", False)),
            "evidence_rate": float(row["evidence_in_prompt"]),
        }
        for row in scored
    ]
    means = mean_metrics(scores)
    return {
        "accuracy": means.get("accuracy", 0.0),
        "contains": means.get("contains", 0.0),
        "evidence_rate": means.get("evidence_rate", 0.0),
        "unparsed": sum(1 for row in scored if not row.get("parsed", False)),
        "errors": len(rows) - len(scored),
        "scored": len(scored),
        "stderr": standard_errors(scores),
        "queries": len(rows),
        "seconds": round(time.monotonic() - started, 1),
        "per_query": rows,
    }


def paired_difference(baseline_rows: list[dict], other_rows: list[dict]) -> dict[str, float]:
    """Per-query difference in correctness, with its own standard error.

    The routes answer the same questions, so the difference is paired. Reading
    it off two independent standard errors overstates the spread and can hide
    a real difference behind overlapping error bars.
    """
    base = {row["query_id"]: bool(row["correct"]) for row in baseline_rows if "error" not in row}
    diffs = [
        float(bool(row["correct"])) - float(base[row["query_id"]])
        for row in other_rows
        if "error" not in row and row["query_id"] in base
    ]
    if not diffs:
        return {}
    mean = sum(diffs) / len(diffs)
    if len(diffs) < 2:
        return {"mean": mean, "stderr": 0.0, "n": len(diffs)}
    variance = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    return {"mean": mean, "stderr": (variance / len(diffs)) ** 0.5, "n": len(diffs)}


def format_answers(results: dict[str, Any], baseline: str) -> str:
    """One row per pipeline: how often the answer was right.

    The saved results also carry a ``_run`` block describing the run, which is
    not a pipeline and has none of these fields.
    """
    results = {name: result for name, result in results.items() if name != "_run"}
    if baseline not in results:
        names = ", ".join(results)
        return f"注意: 基準の経路 {baseline!r} が結果に無い（あるのは {names}）。差分は出せない。"
    reference = (results.get(baseline) or {}).get("accuracy")
    base_rows = (results.get(baseline) or {}).get("per_query", [])
    header = (
        "| 構成 | 回答正解率(完全一致) | 部分一致 "
        "| 正解文書がプロンプトに入った割合 | 抽出失敗 | API失敗 | n | 秒 |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for name, result in results.items():
        if "error" in result:
            blanks = " | ".join([""] * (len(lines[0].split("|")) - 4))
            lines.append(f"| {name} | {result['error']} | {blanks} |")
            continue
        error = (result.get("stderr") or {}).get("accuracy", 0.0)
        cell = f"{result['accuracy']:.3f}±{error:.3f}"
        if name != baseline and reference is not None:
            paired = paired_difference(base_rows, result.get("per_query", []))
            if paired:
                cell += f" ({paired['mean']:+.3f}±{paired['stderr']:.3f})"
            else:
                cell += f" ({result['accuracy'] - reference:+.3f})"
        lines.append(
            f"| {name} | {cell} | {result.get('contains', 0.0):.3f} | "
            f"{result['evidence_rate']:.3f} | {result.get('unparsed', 0)} | "
            f"{result.get('errors', 0)} | {result.get('scored', result['queries'])} | "
            f"{result.get('seconds', '-')} |"
        )
    return "\n".join(lines)
