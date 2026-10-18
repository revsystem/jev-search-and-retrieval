"""Per-route usage: which model answered, the tokens billed, and answers that never came.

A route's requests run on worker threads, so every update takes a lock.
``run_judge`` resets the meter before a route and saves a snapshot after it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageMeter:
    requests: int = 0
    input_tokens: int = 0
    refusals: int = 0
    missing: int = 0
    models: set[str] = field(default_factory=set)
    seconds: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(
        self,
        *,
        model: str,
        input_tokens: int,
        seconds: float,
        refusals: int = 0,
        missing: int = 0,
    ) -> None:
        with self._lock:
            self.requests += 1
            self.input_tokens += input_tokens
            self.refusals += refusals
            self.missing += missing
            if model:
                self.models.add(model)
            self.seconds.append(seconds)

    def reset(self) -> None:
        with self._lock:
            self.requests = self.input_tokens = self.refusals = self.missing = 0
            self.models = set()
            self.seconds = []

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": self.requests,
                "input_tokens": self.input_tokens,
                "refusals": self.refusals,
                "missing": self.missing,
                "models": sorted(self.models),
                "request_seconds": list(self.seconds),
            }
