"""Building the shared state, and the paths that point into it.

TypeSafe references a part of the state from an instruction with a backticked
dot-and-index path relative to the state root, such as
``ticket.messages[0].text``. Question ids are not sent to the model, so the
instruction is the only thing that tells it which candidate to judge.

Candidates therefore go in as an array and are addressed by position. Position
and question id are two different keys for the same passage, so they are
derived together here rather than at each call site.
"""

from __future__ import annotations

from typing import Any

from jev_rag.types import RetrievedDoc


def candidate_state(query: str, docs: list[RetrievedDoc]) -> tuple[dict[str, Any], dict[str, int]]:
    """The shared state, plus the position of each candidate by document id."""
    state = {"query": query, "candidates": [{"text": doc.text} for doc in docs]}
    return state, {doc.doc_id: index for index, doc in enumerate(docs)}


def candidate_path(index: int) -> str:
    return f"`candidates[{index}].text`"
