"""Transports for the Jev System One endpoint.

Two routes exist for the same request body:

``GatewayTransport``
    Vercel AI Gateway's TypeSafe-compatible passthrough. Usable today with an
    AI Gateway key, before a TypeSafe account clears the waiting list.
``DirectTransport``
    TypeSafe's own ``api.typesafe.ai`` endpoint, for once the account is open.

``FakeTransport`` scripts answers offline so the comparison harness and the
tests run with no credentials and no network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

DIRECT_BASE_URL = "https://api.typesafe.ai/v1"
GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/typesafe/v1"

RETRYABLE_STATUS = {408, 409, 425, 429}


def should_retry(status_code: int) -> bool:
    """Rate limits and server faults are worth another attempt; a bad key is not."""
    return status_code in RETRYABLE_STATUS or status_code >= 500


class JevTransport(Protocol):
    model: str

    def build_body(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]: ...

    def send(self, body: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class HttpTransport:
    """Shared HTTP behaviour. Subclasses only pin the base URL and the model id."""

    api_key: str
    base_url: str
    model: str
    timeout: float = 60.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError(f"{type(self).__name__} requires an API key")

    @property
    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/systemone"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def build_body(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return {"model": self.model, "state": state, "questions": questions}

    def send(self, body: dict[str, Any]) -> dict[str, Any]:
        import httpx

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = httpx.post(
                    self.url, json=body, headers=self.headers(), timeout=self.timeout
                )
            except Exception as error:  # noqa: BLE001 - transport faults are retried
                last = error
            else:
                if not should_retry(response.status_code):
                    # Raises immediately on 401/403/422 rather than sleeping through
                    # three attempts on an error no retry can fix. The body is
                    # included because the status alone does not say which
                    # question or field the service objected to.
                    if response.status_code >= 400:
                        raise RuntimeError(
                            f"Jev {response.status_code} from {self.url}: {response.text[:600]}"
                        )
                    return response.json()
                last = RuntimeError(f"retryable status {response.status_code}")
            if attempt < self.max_retries - 1:
                time.sleep(2**attempt)
        raise RuntimeError(f"Jev request to {self.url} failed: {last}") from last


@dataclass
class DirectTransport(HttpTransport):
    base_url: str = DIRECT_BASE_URL
    model: str = "jev-latest"


@dataclass
class GatewayTransport(HttpTransport):
    base_url: str = GATEWAY_BASE_URL
    model: str = "typesafe-ai/jev"


@dataclass
class FakeTransport:
    """Deterministic offline stand-in. Answers are keyed by question id."""

    noul: float | None = None
    nouls: dict[str, float] = field(default_factory=dict)
    choices: dict[str, str] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    score_levels: int = 4
    confidence: float = 0.9
    model: str = "jev-fake"
    requests: list[dict[str, Any]] = field(default_factory=list)

    def build_body(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return {"model": self.model, "state": state, "questions": questions}

    def send(self, body: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(body)
        answers = {key: self._answer(key, q) for key, q in body["questions"].items()}
        return {"model": self.model, "answers": answers, "usage": {"input_tokens": 0}}

    def _answer(self, key: str, question: dict[str, Any]) -> dict[str, Any]:
        kind = question["type"]
        if kind == "noul":
            fallback = self.noul if self.noul is not None else 0.5
            return {"type": "noul", "noul": self.nouls.get(key, fallback)}
        if kind == "choice":
            options = list(question["criteria"])
            winner = self.choices.get(key, options[0])
            spread = (1 - self.confidence) / max(len(options) - 1, 1)
            return {
                "type": "choice",
                "choice": winner,
                "probabilities": {o: (self.confidence if o == winner else spread) for o in options},
                "confidence": self.confidence,
            }
        if kind == "score":
            levels = question.get("criteria") or []
            return {
                "type": "score",
                "score": self.scores.get(key, 0.0),
                "legend": {str(i): label for i, label in enumerate(levels)},
                "probabilities": {str(i): 1 / max(len(levels), 1) for i in range(len(levels))},
                "confidence": self.confidence,
            }
        raise ValueError(f"unknown question type: {kind!r}")


def build_transport(name: str, api_key: str | None = None, **kwargs: Any) -> JevTransport:
    if name == "direct":
        return DirectTransport(api_key=api_key or "", **kwargs)
    if name == "gateway":
        return GatewayTransport(api_key=api_key or "", **kwargs)
    if name == "fake":
        return FakeTransport(**kwargs)
    raise ValueError(f"unknown Jev transport: {name!r} (expected direct, gateway or fake)")
