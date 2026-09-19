"""Retrieval metrics, keyword-rule relevance judgements and pipeline comparison."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jev_rag.documents import Chunk

Qrels = dict[str, int]


@dataclass
class RetrievedDoc:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_score: float | None = None
    jev_score: float | None = None
    jev_relevance: float | None = None
    rank_delta: int = 0


def _graded(qrels: Qrels, min_grade: int) -> Qrels:
    return {cid: grade for cid, grade in qrels.items() if grade >= min_grade}


def ndcg_at_k(ranked: list[RetrievedDoc], qrels: Qrels, k: int, min_grade: int = 1) -> float:
    relevant = _graded(qrels, min_grade)
    gains = [(2 ** relevant.get(doc.chunk_id, 0)) - 1 for doc in ranked[:k]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: list[RetrievedDoc], qrels: Qrels, k: int, min_grade: int = 1) -> float:
    relevant = _graded(qrels, min_grade)
    if not relevant:
        return 0.0
    found = sum(1 for doc in ranked[:k] if doc.chunk_id in relevant)
    return found / len(relevant)


def precision_at_k(ranked: list[RetrievedDoc], qrels: Qrels, k: int, min_grade: int = 1) -> float:
    relevant = _graded(qrels, min_grade)
    return sum(1 for doc in ranked[:k] if doc.chunk_id in relevant) / k if k else 0.0


def mrr(ranked: list[RetrievedDoc], qrels: Qrels, min_grade: int = 1) -> float:
    relevant = _graded(qrels, min_grade)
    for rank, doc in enumerate(ranked, 1):
        if doc.chunk_id in relevant:
            return 1 / rank
    return 0.0


def hit_rate(ranked: list[RetrievedDoc], qrels: Qrels, k: int, min_grade: int = 1) -> float:
    relevant = _graded(qrels, min_grade)
    return 1.0 if any(doc.chunk_id in relevant for doc in ranked[:k]) else 0.0


# Grade 1 means "shares the topic but is not evidence". Counting it as a hit
# would reward exactly the failure this comparison is about, so the binary
# metrics start at grade 2 while nDCG keeps using the full graded scale.
EVIDENCE_GRADE = 2


def score_ranking(ranked: list[RetrievedDoc], qrels: Qrels, k: int = 5) -> dict[str, float]:
    return {
        f"ndcg@{k}": ndcg_at_k(ranked, qrels, k),
        f"strict_ndcg@{k}": ndcg_at_k(ranked, qrels, k, min_grade=3),
        f"recall@{k}": recall_at_k(ranked, qrels, k, min_grade=EVIDENCE_GRADE),
        f"precision@{k}": precision_at_k(ranked, qrels, k, min_grade=EVIDENCE_GRADE),
        f"hit@{k}": hit_rate(ranked, qrels, k, min_grade=EVIDENCE_GRADE),
        "mrr": mrr(ranked, qrels, min_grade=EVIDENCE_GRADE),
    }


@dataclass
class EvalQuery:
    """A question plus the rules that decide which chunks count as evidence for it."""

    query_id: str
    question: str
    relevant: dict[str, Any] | None = None
    partial: dict[str, Any] | None = None
    rules: list[dict[str, Any]] = field(default_factory=list)
    qrels: Qrels | None = None
    note: str = ""

    def grading_rules(self) -> list[dict[str, Any]]:
        """Every labelling rule, shorthand fields first then the explicit list."""
        return [rule for rule in (self.relevant, self.partial) if rule] + list(self.rules)


def _matches(rule: dict[str, Any], text: str, pattern: re.Pattern | None) -> bool:
    required = rule.get("all") or []
    optional = rule.get("any") or []
    forbidden = rule.get("none") or []
    if any(word in text for word in forbidden):
        return False
    if not all(word in text for word in required):
        return False
    if pattern is not None and not pattern.search(text):
        return False
    return bool(any(word in text for word in optional)) if optional else True


def _compile(rule: dict[str, Any], query_id: str) -> re.Pattern | None:
    """A rule may demand a pattern as well as keywords.

    A bare "%" matches nearly every page of a statistics whitepaper, so a rule
    that means "reports a figure" needs to ask for a number beside the unit.
    """
    expression = rule.get("regex")
    if not expression:
        return None
    try:
        return re.compile(expression)
    except re.error as error:
        raise ValueError(
            f"query {query_id!r} has an invalid regex {expression!r}: {error}"
        ) from error


def build_qrels(query: EvalQuery, corpus: list[Chunk]) -> Qrels:
    """Grade every chunk for one query, either from explicit ids or from keyword rules."""
    if query.qrels:
        return dict(query.qrels)

    rules = query.grading_rules()
    if not rules:
        raise ValueError(f"query {query.query_id!r} has neither explicit qrels nor keyword rules")

    compiled = [(rule, _compile(rule, query.query_id)) for rule in rules]
    graded: Qrels = {}
    for chunk in corpus:
        grades = [
            int(rule.get("grade", 1))
            for rule, pattern in compiled
            if _matches(rule, chunk.text, pattern)
        ]
        if grades:
            graded[chunk.chunk_id] = max(grades)
    return graded


def load_queries(path: str | Path) -> list[EvalQuery]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [EvalQuery(**entry) for entry in raw.get("queries", [])]


def mean_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0]
    return {key: sum(row[key] for row in per_query) / len(per_query) for key in keys}


def compare(results: dict[str, dict[str, float]], baseline: str) -> list[dict[str, Any]]:
    """Turn {pipeline: metrics} into report rows carrying the delta against the baseline."""
    reference = results.get(baseline, {})
    rows: list[dict[str, Any]] = []
    for pipeline, metrics in results.items():
        row: dict[str, Any] = {"pipeline": pipeline, **metrics}
        for metric, value in metrics.items():
            if metric in reference:
                row[f"delta_{metric}"] = value - reference[metric]
        rows.append(row)
    return rows


def format_table(rows: list[dict[str, Any]], metrics: list[str]) -> str:
    header = ["pipeline", *metrics]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [str(row["pipeline"])]
        for metric in metrics:
            value = row.get(metric)
            delta = row.get(f"delta_{metric}")
            cell = "-" if value is None else f"{value:.3f}"
            if delta:
                cell += f" ({delta:+.3f})"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
