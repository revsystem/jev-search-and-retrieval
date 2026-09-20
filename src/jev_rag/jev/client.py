"""Jev client: question batching, parallel dispatch and answer parsing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from jev_rag.jev.budget import batch_by_budget, estimate_tokens
from jev_rag.jev.questions import Answer, Question, parse_answer
from jev_rag.jev.transport import JevTransport


class JevClient:
    """Evaluates typed questions against one shared state.

    Questions in a single request are answered in parallel and in isolation, so
    asking thirty of them costs barely more latency than asking one. A set too
    heavy for one request is split by the documented token budget rather than
    by a question count, always against the identical state.
    """

    def __init__(
        self,
        transport: JevTransport,
        max_questions: int | None = None,
        max_workers: int = 4,
    ) -> None:
        self.transport = transport
        self.max_questions = max_questions
        self.max_workers = max_workers

    def evaluate(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        if not questions:
            raise ValueError("at least one question is required")

        sized = batch_by_budget(
            [(key, question.payload()) for key, question in questions.items()],
            state_tokens=estimate_tokens(state),
            max_questions=self.max_questions,
        )
        batches = [{key: questions[key] for key, _ in batch} for batch in sized]
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
