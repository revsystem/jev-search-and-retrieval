"""Turn results into the tables a write-up needs."""

from __future__ import annotations

from typing import Any

# Published nDCG@10 on the JQaRA test split, for context in the comparison.
PUBLISHED_JQARA = {
    "multilingual-e5-large (埋め込み)": 0.554,
    "GLuCoSE-base-ja-v2 (埋め込み)": 0.606,
    "ruri-large (埋め込み)": 0.629,
    "japanese-reranker-cross-encoder-large-v1": 0.710,
    "ruri-reranker-base": 0.743,
    "ruri-reranker-large": 0.771,
}


def format_comparison(
    results: dict[str, Any], baseline: str, metrics: list[str], published: bool = False
) -> str:
    reference = (results.get(baseline) or {}).get("metrics", {})
    header = ["ranker", *metrics, "n", "秒"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for name, result in results.items():
        if "error" in result:
            cells = " | ".join([result["error"]] + [""] * len(metrics))
            lines.append(f"| {name} | {cells} |")
            continue
        cells = []
        for metric in metrics:
            value = result["metrics"].get(metric)
            if value is None:
                cells.append("-")
                continue
            error = (result.get("stderr") or {}).get(metric)
            cell = f"{value:.3f}" if error is None else f"{value:.3f}±{error:.3f}"
            if name != baseline and metric in reference:
                cell += f" ({value - reference[metric]:+.3f})"
            cells.append(cell)
        run = f"{result.get('queries', '-')} | {result.get('seconds', '-')}"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {run} |")

    if published:
        lines.append("")
        lines.append("参考: JQaRA テストセットで公開されている nDCG@10")
        lines.append("| model | ndcg@10 |")
        lines.append("|---|---|")
        for model, score in PUBLISHED_JQARA.items():
            lines.append(f"| {model} | {score:.3f} |")
    return "\n".join(lines)


def format_movers(results: dict[str, Any], baseline: str, ranker: str, limit: int = 10) -> str:
    """Queries where the reranker pulled the answer up the most.

    A mean score says whether something improved; these say what improving
    looked like, which is what a reader remembers.
    """
    before = {r["query_id"]: r for r in (results.get(baseline) or {}).get("per_query", [])}
    after = {r["query_id"]: r for r in (results.get(ranker) or {}).get("per_query", [])}

    moved = []
    for query_id, row in after.items():
        was, now = before.get(query_id, {}).get("first_relevant"), row.get("first_relevant")
        if was and now and was > now:
            moved.append((was - now, was, now, query_id, row["question"]))
    moved.sort(reverse=True)

    lines = [
        f"{baseline} で沈んでいた正解を {ranker} が引き上げた例（上位{limit}件）",
        "| 改善 | 前の順位 | 後の順位 | query_id | 質問 |",
        "|---|---|---|---|---|",
    ]
    for gain, was, now, query_id, question in moved[:limit]:
        lines.append(f"| +{gain} | {was} | {now} | {query_id} | {question[:60]} |")
    return "\n".join(lines)
