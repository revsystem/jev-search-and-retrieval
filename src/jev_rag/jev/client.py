"""Jev client: question batching, parallel dispatch and answer parsing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from jev_rag.jev.questions import Answer, Question, parse_answer
from jev_rag.jev.transport import JevTransport

# Gateway limit: at most 32 questions share one state in a single request.
MAX_QUESTIONS_PER_REQUEST = 32


def batched(questions: dict[str, Question], size: int) -> list[dict[str, Question]]:
    keys = list(questions)
    return [{k: questions[k] for k in keys[i : i + size]} for i in range(0, len(keys), size)]


class JevClient:
    """Evaluates typed questions against one shared state.

    Questions in a single request are answered in parallel and in isolation, so
    asking thirty of them costs barely more latency than asking one. Requests
    above the gateway limit are split, always against the identical state.
    """

    def __init__(
        self,
        transport: JevTransport,
        max_questions: int = MAX_QUESTIONS_PER_REQUEST,
        max_workers: int = 4,
    ) -> None:
        self.transport = transport
        self.max_questions = min(max_questions, MAX_QUESTIONS_PER_REQUEST)
        self.max_workers = max_workers

    def evaluate(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        if not questions:
            raise ValueError("at least one question is required")

        batches = batched(questions, self.max_questions)
        if len(batches) == 1:
            responses = [self._send(state, batches[0])]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                responses = list(pool.map(lambda b: self._send(state, b), batches))

        answers: dict[str, Answer] = {}
        for response in responses:
            for key, body in (response.get("answers") or {}).items():
                answers[key] = parse_answer(body)
        return answers

    def _send(self, state: Any, questions: dict[str, Question]) -> dict[str, Any]:
        wire = {key: question.payload() for key, question in questions.items()}
        return self.transport.send(self.transport.build_body(state, wire))
