"""Use case: select useful context for downstream AI workflows.

Reranking decides an order; this decides what actually goes into the prompt.
Two judgements per candidate — how much it is worth, and whether the other
candidates already say it — then a greedy pack under a budget. Dropping
redundant passages matters because a duplicated statistic wastes budget and
gives the answering model a false sense of corroboration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from jev_rag.evaluation import RetrievedDoc
from jev_rag.jev.client import JevClient
from jev_rag.jev.prompts import RELEVANCE_LEVELS
from jev_rag.jev.questions import Noul, NoulAnswer, Score, ScoreAnswer

USEFULNESS = (
    'state.query に答える文脈として state.candidates["{key}"] がどれだけ有用かを判定してください。'
)
REDUNDANT = 'state.candidates["{key}"] の情報は、他の候補がすでに伝えている内容と重複していますか。'
REDUNDANT_CRITERIA = {
    "true": "同じ事実・数値を他の候補が既に述べており、追加しても新しい情報がない。",
    "false": "他の候補にはない事実・数値・観点を含む。",
}


@dataclass
class ContextSelection:
    selected: list[RetrievedDoc] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    used_chars: int = 0

    def as_prompt(self) -> str:
        return "\n\n".join(
            f"[p.{doc.metadata.get('page', '?')}] {doc.text}" for doc in self.selected
        )


class ContextSelector:
    def __init__(
        self,
        client: JevClient,
        budget_chars: int = 8000,
        min_usefulness: float = 1.0,
        max_redundancy: float = 0.5,
    ) -> None:
        self.client = client
        self.budget_chars = budget_chars
        self.min_usefulness = min_usefulness
        self.max_redundancy = max_redundancy

    def select(self, query: str, docs: list[RetrievedDoc]) -> ContextSelection:
        if not docs:
            return ContextSelection()

        state = {"query": query, "candidates": {doc.chunk_id: doc.text for doc in docs}}
        questions = {}
        for doc in docs:
            questions[f"{doc.chunk_id}__usefulness"] = Score(
                USEFULNESS.format(key=doc.chunk_id), criteria=RELEVANCE_LEVELS
            )
            questions[f"{doc.chunk_id}__redundant"] = Noul(
                REDUNDANT.format(key=doc.chunk_id), criteria=REDUNDANT_CRITERIA
            )
        answers = self.client.evaluate(state, questions)

        judged = [
            (
                doc,
                _score(answers.get(f"{doc.chunk_id}__usefulness")),
                _noul(answers.get(f"{doc.chunk_id}__redundant")),
            )
            for doc in docs
        ]
        # Stable sort: equally useful candidates keep their retrieval order.
        judged.sort(key=lambda row: row[1], reverse=True)

        result = ContextSelection()
        for doc, usefulness, redundancy in judged:
            if usefulness < self.min_usefulness:
                result.reasons[doc.chunk_id] = "not_useful"
            elif redundancy >= self.max_redundancy:
                result.reasons[doc.chunk_id] = "redundant"
            elif result.used_chars + len(doc.text) > self.budget_chars:
                result.reasons[doc.chunk_id] = "over_budget"
            else:
                result.selected.append(replace(doc, jev_score=usefulness))
                result.used_chars += len(doc.text)

        if not result.selected:
            # Handing the answering model nothing is worse than handing it the
            # best of a weak set; the reasons still record why it was doubted.
            best, usefulness, _ = judged[0]
            result.selected.append(replace(best, jev_score=usefulness))
            result.used_chars = len(best.text)
            result.reasons.pop(best.chunk_id, None)
        return result


def _score(answer: object) -> float:
    return answer.score if isinstance(answer, ScoreAnswer) else 0.0


def _noul(answer: object) -> float:
    return answer.value if isinstance(answer, NoulAnswer) else 0.0
