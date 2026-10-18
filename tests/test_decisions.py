"""OpenAI's Decisions API as a relevance judge, exercised against a scripted SDK."""

import json

from openai.types import Decision

from jev_rag.decisions import DecisionsClient, predicate, with_criteria


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
