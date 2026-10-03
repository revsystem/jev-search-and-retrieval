"""strands-decider as a relevance judge, for comparison with Jev.

strands-decider (github.com/strands-labs/strands-decider) serves the same
`POST /v1/systemone` shape as Jev from a local GPU, so the questions Jev
received can be sent unchanged. Two things about the 0.1.0 server shape this
module:

- Its window is 4,096 tokens, and a state that does not fit is cut without an
  error. A cut state would score candidates the model never read, so every
  group is sized against the window with the model's own tokenizer before it is
  sent, and a candidate that cannot fit on its own is refused.
- Concurrent requests are not verified upstream and the engine keeps
  per-request offsets on itself, so requests go one at a time. The server's own
  `latency_ms` is kept per request; it times the model, not the network.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import cache
from typing import Any

import httpx

from jev_rag.jev.prompts import NOUL_INSTRUCTION, PLAIN_NOUL_INSTRUCTION, RELEVANCE_CRITERIA
from jev_rag.jev.state import candidate_path
from jev_rag.types import RetrievedDoc

DECIDER_URL = "http://127.0.0.1:8099/v1"
DECIDER_MODEL = "strands-decider-2B-hobson-v19"
TOKENIZER = "Qwen/Qwen3.5-2B-Base"
# 4,096-token window less room for the longest question (about 150 tokens for
# the Japanese criteria) and a margin
STATE_BUDGET = 3_600

# The same question in English, for the model whose training data is mostly English
EN_INSTRUCTION = "Is {path} useful as evidence for answering `query`?"
EN_CRITERIA = {
    "true": (
        "It deals with exactly what the query refers to and contains facts, figures or "
        "definitions that support the answer."
    ),
    "false": (
        "It only overlaps in topic, refers to something else, or contains no information "
        "that could serve as evidence."
    ),
}


def render_state_text(state: dict[str, Any]) -> str:
    """The state exactly as strands_decider.prompting.render_state writes it."""
    return (
        f"<state>\n{json.dumps(state, indent=2, ensure_ascii=False, sort_keys=False)}\n</state>\n"
    )


@cache
def _tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(TOKENIZER)


def count_decider_tokens(text: str) -> int:
    return len(_tokenizer().encode(text).ids)


def _state(query: str, group: list[RetrievedDoc]) -> dict[str, Any]:
    return {"query": query, "candidates": [{"text": doc.text} for doc in group]}


def group_by_budget(
    query: str,
    docs: list[RetrievedDoc],
    batch_size: int,
    budget: int = STATE_BUDGET,
    count_tokens: Callable[[str], int] = count_decider_tokens,
) -> list[list[RetrievedDoc]]:
    """Consecutive groups of at most `batch_size` whose rendered state fits `budget`."""
    groups: list[list[RetrievedDoc]] = []
    current: list[RetrievedDoc] = []
    for doc in docs:
        trial = current + [doc]
        fits = count_tokens(render_state_text(_state(query, trial))) <= budget
        if current and (len(trial) > batch_size or not fits):
            groups.append(current)
            current, trial = [], [doc]
            fits = count_tokens(render_state_text(_state(query, trial))) <= budget
        if not fits:
            raise ValueError(f"{doc.doc_id}: 候補文書 1 件だけでも窓に収まらない")
        current = trial
    if current:
        groups.append(current)
    return groups


class DeciderRanker:
    """Score each candidate with a Noul, as jev_pointwise does, against the local server."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        batch_size: int = 10,
        plain: bool = False,
        english: bool = False,
        budget: int = STATE_BUDGET,
        count_tokens: Callable[[str], int] = count_decider_tokens,
        model: str = DECIDER_MODEL,
    ) -> None:
        self.client = client or httpx.Client(base_url=DECIDER_URL, timeout=600)
        self.batch_size = batch_size
        self.plain = plain
        self.english = english
        self.budget = budget
        self.count_tokens = count_tokens
        self.model = model
        self.latencies_ms: list[float] = []

    def _question(self, index: int) -> dict[str, Any]:
        path = candidate_path(index)
        if self.english:
            return {
                "type": "noul",
                "instructions": EN_INSTRUCTION.format(path=path),
                "criteria": EN_CRITERIA,
            }
        if self.plain:
            return {"type": "noul", "instructions": PLAIN_NOUL_INSTRUCTION.format(path=path)}
        return {
            "type": "noul",
            "instructions": NOUL_INSTRUCTION.format(path=path),
            "criteria": RELEVANCE_CRITERIA,
        }

    def rank(self, question: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        scored: list[RetrievedDoc] = []
        groups = group_by_budget(question, docs, self.batch_size, self.budget, self.count_tokens)
        for group in groups:
            body = {
                "model": self.model,
                "state": _state(question, group),
                "questions": {f"c_{i}": self._question(i) for i in range(len(group))},
            }
            response = self.client.post("/systemone", json=body)
            response.raise_for_status()
            payload = response.json()
            self.latencies_ms.append(float(payload.get("latency_ms", 0.0)))
            for i, doc in enumerate(group):
                doc.score = float(payload["answers"][f"c_{i}"]["noul"])
                scored.append(doc)
        return sorted(scored, key=lambda d: d.score, reverse=True)
