"""Use case: cross-encode queries and candidates for higher precision.

Listwise reranking puts every candidate in one shared state, which is cheap but
lets neighbouring candidates colour a judgement. Cross-encoding gives each
query-candidate pair a request of its own, so nothing else is in the context.
It costs one request per candidate, sent concurrently.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from jev_rag.jev.client import JevClient
from jev_rag.jev.prompts import RELEVANCE_CRITERIA, RELEVANCE_LEVELS
from jev_rag.jev.questions import Noul, NoulAnswer, Score, ScoreAnswer
from jev_rag.types import RetrievedDoc

RELEVANCE = "`candidate` は `query` に答えるための根拠として役に立ちますか。"
GRADE = "`candidate` が `query` の根拠としてどの程度有用かを判定してください。"


class CrossEncoder:
    def __init__(self, client: JevClient, with_grade: bool = False, max_workers: int = 8) -> None:
        self.client = client
        self.with_grade = with_grade
        self.max_workers = max_workers

    def rerank(
        self, query: str, docs: list[RetrievedDoc], top_k: int | None = None
    ) -> list[RetrievedDoc]:
        if not docs:
            return []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            judgements = list(executor.map(lambda doc: self._judge(query, doc), docs))

        ranked = [
            replace(doc, score=relevance, vector_score=doc.score, jev_score=grade)
            for doc, (relevance, grade) in zip(docs, judgements, strict=True)
        ]
        original = {doc.chunk_id: position for position, doc in enumerate(docs)}
        ranked.sort(key=lambda doc: (doc.score, doc.jev_score or 0.0), reverse=True)
        for position, doc in enumerate(ranked):
            doc.rank_delta = original[doc.chunk_id] - position
        return ranked[:top_k] if top_k else ranked

    def _judge(self, query: str, doc: RetrievedDoc) -> tuple[float, float | None]:
        questions = {"relevance": Noul(RELEVANCE, criteria=RELEVANCE_CRITERIA)}
        if self.with_grade:
            questions["grade"] = Score(GRADE, criteria=RELEVANCE_LEVELS)
        answers = self.client.evaluate({"query": query, "candidate": doc.text}, questions)

        relevance = answers.get("relevance")
        grade = answers.get("grade")
        return (
            relevance.value if isinstance(relevance, NoulAnswer) else 0.0,
            grade.score if isinstance(grade, ScoreAnswer) else None,
        )
