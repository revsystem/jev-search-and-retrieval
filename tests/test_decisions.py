"""OpenAI's Decisions API as a relevance judge, exercised against a scripted SDK."""

import json

import pytest
from openai.types import Decision

from jev_rag.decisions import (
    DecisionsClient,
    DecisionsCrossEncodeRanker,
    DecisionsPointwiseRanker,
    predicate,
    with_criteria,
)
from jev_rag.types import RetrievedDoc


class FakeDecisions:
    """Stands in for `OpenAI().decisions`. `reply` builds the answers for one request."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def create(self, **body):
        self.requests.append(body)
        return Decision.model_validate(
            {
                "answers": self.reply(body),
                "model": "gpt-6-luna",
                "usage": {
                    "input_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 120,
                },
            }
        )


class FakeSDK:
    def __init__(self, reply):
        self.decisions = FakeDecisions(reply)


def predicates(probabilities):
    """Answer each predicate with the probability keyed by its name, in reverse order."""

    def reply(body):
        return [
            {"type": "predicate", "name": q["name"], "probability": probabilities[q["name"]]}
            for q in reversed(body["questions"])
        ]

    return reply


def test_criteria_are_appended_to_the_instruction_text():
    text = with_criteria("`candidate` は役に立ちますか。")
    assert text == (
        "`candidate` は役に立ちますか。\n判定基準: はい（true）= "
        "クエリが指す対象そのものを扱い、回答を裏づける事実・数値・定義を含む。"
        "いいえ（false）= 話題が重なるだけ、参照先が異なる、または根拠となる情報を含まない。"
    )


def test_the_client_sends_the_model_input_and_questions():
    sdk = FakeSDK(predicates({"c0": 0.8}))
    client = DecisionsClient(sdk)
    client.evaluate('{"query": "x"}', [predicate("c0", "質問")])
    assert sdk.decisions.requests == [
        {
            "model": "gpt-6-luna",
            "input": '{"query": "x"}',
            "questions": [{"type": "predicate", "name": "c0", "instructions": "質問"}],
        }
    ]


def test_answers_are_matched_by_name_not_by_order():
    client = DecisionsClient(FakeSDK(predicates({"c0": 0.8, "c1": 0.1})))
    answers = client.evaluate("{}", [predicate("c0", "a"), predicate("c1", "b")])
    assert answers["c0"]["probability"] == 0.8
    assert answers["c1"]["probability"] == 0.1


def test_refusals_and_absent_answers_are_left_out_and_counted():
    def reply(body):
        return [
            {"type": "refusal", "name": "c0"},
            {"type": "predicate", "name": "c1", "probability": 0.4},
        ]

    client = DecisionsClient(FakeSDK(reply))
    answers = client.evaluate("{}", [predicate(f"c{i}", "q") for i in range(3)])
    assert set(answers) == {"c1"}
    usage = client.meter.snapshot()
    assert usage["refusals"] == 1
    assert usage["missing"] == 1


def test_the_meter_records_the_model_and_billed_tokens():
    client = DecisionsClient(FakeSDK(predicates({"c0": 0.5})))
    client.evaluate("{}", [predicate("c0", "q")])
    usage = client.meter.snapshot()
    assert usage["models"] == ["gpt-6-luna"]
    assert usage["input_tokens"] == 120
    assert usage["requests"] == 1


def test_the_input_keeps_japanese_readable():
    sdk = FakeSDK(predicates({"c0": 0.5}))
    state = json.dumps({"query": "火星"}, ensure_ascii=False)
    DecisionsClient(sdk).evaluate(state, [predicate("c0", "q")])
    assert "火星" in sdk.decisions.requests[0]["input"]


def candidates(n):
    return [RetrievedDoc(doc_id=f"d{i}", text=f"文書{i}", score=0.0, label=0) for i in range(n)]


def by_position(body):
    """Probability 0.1 * (position + 1), so the order of the result shows the mapping."""
    return [
        {"type": "predicate", "name": q["name"], "probability": 0.1 * (int(q["name"][1:]) + 1)}
        for q in body["questions"]
    ]


def test_pointwise_sends_ten_candidates_per_request_as_json_text():
    sdk = FakeSDK(by_position)
    DecisionsPointwiseRanker(DecisionsClient(sdk)).rank("問い", candidates(12))
    sizes = sorted(len(r["questions"]) for r in sdk.decisions.requests)
    assert sizes == [2, 10]
    first = next(r for r in sdk.decisions.requests if len(r["questions"]) == 10)
    state = json.loads(first["input"])
    assert state["query"] == "問い"
    assert [c["text"] for c in state["candidates"]] == [f"文書{i}" for i in range(10)]


def test_pointwise_questions_point_at_each_candidate_with_the_criteria():
    sdk = FakeSDK(by_position)
    DecisionsPointwiseRanker(DecisionsClient(sdk)).rank("問い", candidates(2))
    questions = sdk.decisions.requests[0]["questions"]
    assert questions[1] == predicate(
        "c1",
        with_criteria("`candidates[1].text` は `query` に答えるための根拠として役に立ちますか。"),
    )


def test_the_plain_route_sends_no_criteria():
    sdk = FakeSDK(by_position)
    ranker = DecisionsPointwiseRanker(DecisionsClient(sdk), plain=True)
    ranker.rank("問い", candidates(1))
    assert ranker.name == "decisions_pointwise_plain"
    assert sdk.decisions.requests[0]["questions"][0]["instructions"] == (
        "`candidates[0].text` は `query` に関連しますか。"
    )


def test_pointwise_sorts_on_the_probability_of_each_document():
    ranked = DecisionsPointwiseRanker(DecisionsClient(FakeSDK(by_position))).rank(
        "問い", candidates(3)
    )
    assert [d.doc_id for d in ranked] == ["d2", "d1", "d0"]
    assert ranked[0].score == pytest.approx(0.3)


def test_a_refused_candidate_scores_zero():
    def reply(body):
        return [{"type": "refusal", "name": "c0"}] + by_position(body)[1:]

    ranked = DecisionsPointwiseRanker(DecisionsClient(FakeSDK(reply))).rank("問い", candidates(2))
    assert {d.doc_id: d.score for d in ranked}["d0"] == 0.0


def test_crossencode_sends_one_request_per_candidate_with_a_predicate_and_a_score():
    def reply(body):
        return [
            {"type": "predicate", "name": "relevance", "probability": 0.6},
            {
                "type": "score",
                "name": "grade",
                "score": 2.0,
                "confidence": 0.5,
                "probabilities": [
                    {"value": i, "label": f"l{i}", "probability": 0.25} for i in range(4)
                ],
            },
        ]

    sdk = FakeSDK(reply)
    ranked = DecisionsCrossEncodeRanker(DecisionsClient(sdk)).rank("問い", candidates(3))
    assert len(sdk.decisions.requests) == 3
    request = sdk.decisions.requests[0]
    assert json.loads(request["input"]).keys() == {"query", "candidate"}
    relevance, grade = request["questions"]
    assert relevance == predicate(
        "relevance",
        with_criteria("`candidate` は `query` に答えるための根拠として役に立ちますか。"),
    )
    assert grade["type"] == "score"
    assert grade["instructions"] == (
        "`candidate` が `query` の根拠としてどの程度有用かを判定してください。"
    )
    assert [level["label"] for level in grade["levels"]] == [
        "無関係。クエリの主題とも参照先とも一致しない。",
        "話題は重なるが、クエリへの回答根拠にはならない。",
        "部分的な根拠を含む。回答の一部を裏づけるが単独では不十分。",
        "直接的な根拠を含む。この記述だけでクエリに答えられる。",
    ]
    assert ranked[0].score == pytest.approx(0.6)
    assert ranked[0].jev_score == pytest.approx(2.0)


def test_every_route_exposes_the_clients_meter():
    client = DecisionsClient(FakeSDK(by_position))
    assert DecisionsPointwiseRanker(client).meter is client.meter
    assert DecisionsCrossEncodeRanker(client).meter is client.meter


def test_refusals_without_a_name_are_each_counted():
    def reply(body):
        return [{"type": "refusal"}, {"type": "refusal"}]

    client = DecisionsClient(FakeSDK(reply))
    client.evaluate("{}", [predicate("c0", "q"), predicate("c1", "q")])
    assert client.meter.snapshot()["refusals"] == 2


def test_the_sdk_retries_and_times_out_like_the_jev_transport():
    from jev_rag.config import DecisionsSettings

    sdk = DecisionsSettings(api_key="sk-test").build_client().sdk
    # Jev makes three attempts with a 60-second timeout; the SDK counts retries after the first
    assert sdk.max_retries == 2
    assert sdk.timeout == 60
