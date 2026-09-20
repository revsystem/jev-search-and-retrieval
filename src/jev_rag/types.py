"""Shared value types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedDoc:
    """One candidate passage, as it moves through a ranker.

    ``score`` is whatever the current ranker assigned. The other score fields
    keep the earlier signals so a report can show what a rerank actually moved.
    """

    doc_id: str
    text: str
    score: float
    label: int = 0
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    vector_score: float | None = None
    jev_score: float | None = None
    jev_relevance: float | None = None
    rank_delta: int = 0

    @property
    def chunk_id(self) -> str:
        """Alias kept so question ids read naturally in Jev requests."""
        return self.doc_id
