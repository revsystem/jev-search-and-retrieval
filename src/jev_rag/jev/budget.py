"""Sizing a System One request against the documented token budget.

TypeSafe documents two limits for Jev 1.13: 64k tokens per request covering the
state plus every question, and 32k for the state plus the single longest
question. It documents no cap on the number of questions — the line-by-line
search cookbook scores 218 candidates in one call — so requests are split by
what they weigh, not by how many questions they carry.
"""

from __future__ import annotations

import json
import re
from typing import Any

TOTAL_TOKEN_BUDGET = 64_000
STATE_PLUS_QUESTION_BUDGET = 32_000
# Leaves room for the wrapper JSON and for the estimate being approximate.
SAFETY_MARGIN = 0.9
# Measured against a reported usage of 35,356 tokens where this estimate said
# 32,069: the character count runs about 10% under. The limits are hard and a
# breach costs the whole request, so the correction is generous.
UNDERCOUNT_CORRECTION = 1.25

_CJK = re.compile(r"[　-〿぀-ゟ゠-ヿ㐀-鿿＀-￯]")


def estimate_tokens(value: Any) -> int:
    """Approximate tokens for text or a JSON-serialisable structure.

    Japanese runs near one token per character while ASCII runs nearer one per
    four, and this corpus is Japanese, so the two are counted separately rather
    than with a single characters-per-token ratio.
    """
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    cjk = len(_CJK.findall(text))
    raw = cjk + (len(text) - cjk + 3) // 4
    return int(raw * UNDERCOUNT_CORRECTION)


def fitting_batch_size(
    query: str,
    docs,
    question_tokens: int,
    pair_budget: int = STATE_PLUS_QUESTION_BUDGET,
) -> int:
    """How many candidates can share one state.

    The binding limit for a listwise reranking request is the state plus the
    single longest question, not the total: 100 Japanese passages of a few
    hundred characters exceed it on their own, and the service answers 400
    max_tokens_exceeded. Splitting is the caller's only option, so it is done
    here rather than left to them.
    """
    overhead = estimate_tokens({"query": query, "candidates": []}) + question_tokens
    room = int(pair_budget * SAFETY_MARGIN) - overhead
    used = 0
    for count, doc in enumerate(docs):
        used += estimate_tokens({"text": doc.text})
        if used > room:
            return max(count, 1)
    return max(len(docs), 1)


def batch_by_budget(
    items: list[tuple[str, Any]],
    state_tokens: int,
    max_questions: int | None = None,
    total_budget: int = TOTAL_TOKEN_BUDGET,
    pair_budget: int = STATE_PLUS_QUESTION_BUDGET,
) -> list[list[tuple[str, Any]]]:
    """Split (key, payload) pairs into requests that each fit beside the state."""
    room = int(total_budget * SAFETY_MARGIN) - state_tokens
    if room <= 0:
        raise ValueError(f"state alone needs {state_tokens} tokens of a {total_budget} budget")

    batches: list[list[tuple[str, Any]]] = []
    current: list[tuple[str, Any]] = []
    used = 0
    for key, payload in items:
        cost = estimate_tokens(payload)
        if state_tokens + cost > pair_budget:
            raise ValueError(
                f"question {key!r} needs {cost} tokens beside a {state_tokens}-token state, "
                f"over the {pair_budget} budget for the state plus one question"
            )
        full = max_questions is not None and len(current) >= max_questions
        if current and (full or used + cost > room):
            batches.append(current)
            current, used = [], 0
        current.append((key, payload))
        used += cost
    if current:
        batches.append(current)
    return batches
