"""strands-decider as a relevance judge, through its local System One server.

The request shape is the one Jev takes, so the comparison holds the data, the
questions and the metrics fixed. What differs is forced by the model: a
4,096-token window that the 0.1.0 server fills by silently cutting the state,
and a server that must be called one request at a time.
"""

import json

import httpx
import pytest

from jev_rag.decider import (
    EN_CRITERIA,
    DeciderRanker,
    group_by_budget,
    render_state_text,
)
from jev_rag.types import RetrievedDoc


def docs(n, text="本文"):
    return [RetrievedDoc(doc_id=f"d{i}", text=f"{text}{i}", score=0.0) for i in range(n)]


class Recorder:
    """A fake server: answers every Noul with 0.1 * (position + 1) and records requests."""

    def __init__(self):
        self.requests = []
        self.in_flight = 0
        self.max_in_flight = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        body = json.loads(request.content)
        self.requests.append(body)
        answers = {
            name: {"type": "noul", "noul": round(0.1 * (int(name.split("_")[1]) + 1), 2)}
            for name in body["questions"]
        }
        self.in_flight -= 1
        return httpx.Response(
            200,
            json={
                "model": "m",
                "answers": answers,
                "usage": {"input_tokens": 1},
                "latency_ms": 12.5,
            },
        )


def ranker(recorder, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(recorder), base_url="http://decider/v1")
    return DeciderRanker(client=client, count_tokens=lambda text: len(text), **kwargs)


# --- the state the server will read -----------------------------------------------


def test_the_state_is_rendered_the_way_the_server_renders_it():
    text = render_state_text({"query": "問い", "candidates": [{"text": "本文"}]})
    assert text.startswith('<state>\n{\n  "query": "問い"')
    assert text.endswith("\n}\n</state>\n")


# --- grouping by the window ---------------------------------------------------------


def test_groups_hold_at_most_the_batch_size():
    groups = group_by_budget("問い", docs(25), batch_size=10, budget=10_000, count_tokens=len)
    assert [len(g) for g in groups] == [10, 10, 5]


def test_a_group_closes_before_its_state_would_exceed_the_budget():
    # each candidate adds roughly 30 characters of rendered JSON
    groups = group_by_budget("問い", docs(6), batch_size=10, budget=120, count_tokens=len)
    assert all(
        len(render_state_text({"query": "問い", "candidates": [{"text": d.text} for d in g]}))
        <= 120
        for g in groups
    )
    assert sum(len(g) for g in groups) == 6


def test_a_single_candidate_over_the_budget_is_refused_rather_than_cut():
    with pytest.raises(ValueError, match="d0"):
        group_by_budget(
            "問い", docs(1, text="長" * 500), batch_size=10, budget=100, count_tokens=len
        )


# --- requests ------------------------------------------------------------------------


def test_requests_are_sent_one_at_a_time_and_every_candidate_is_scored():
    recorder = Recorder()
    ranked = ranker(recorder, batch_size=10, budget=10_000).rank("問い", docs(12))
    assert recorder.max_in_flight == 1
    assert len(recorder.requests) == 2
    assert {d.doc_id for d in ranked} == {f"d{i}" for i in range(12)}


def test_each_question_points_at_its_own_candidate():
    recorder = Recorder()
    ranker(recorder, batch_size=10, budget=10_000).rank("問い", docs(2))
    questions = recorder.requests[0]["questions"]
    assert "`candidates[0].text`" in questions["c_0"]["instructions"]
    assert "`candidates[1].text`" in questions["c_1"]["instructions"]


def test_scores_come_from_the_noul_of_each_candidate():
    recorder = Recorder()
    ranked = ranker(recorder, batch_size=10, budget=10_000).rank("問い", docs(3))
    assert {d.doc_id: d.score for d in ranked} == {"d0": 0.1, "d1": 0.2, "d2": 0.3}


def test_server_latency_is_recorded_per_request():
    recorder = Recorder()
    route = ranker(recorder, batch_size=1, budget=10_000)
    route.rank("問い", docs(3))
    assert route.latencies_ms == [12.5, 12.5, 12.5]


# --- the question variants -------------------------------------------------------------


def test_the_default_question_is_the_japanese_one_jev_received():
    recorder = Recorder()
    ranker(recorder, batch_size=10, budget=10_000).rank("問い", docs(1))
    question = recorder.requests[0]["questions"]["c_0"]
    assert "根拠" in question["instructions"]
    assert "criteria" in question


def test_the_plain_question_carries_no_criteria():
    recorder = Recorder()
    ranker(recorder, batch_size=10, budget=10_000, plain=True).rank("問い", docs(1))
    question = recorder.requests[0]["questions"]["c_0"]
    assert "criteria" not in question
    assert "関連" in question["instructions"]


def test_the_english_question_translates_instruction_and_criteria():
    recorder = Recorder()
    ranker(recorder, batch_size=10, budget=10_000, english=True).rank("問い", docs(1))
    question = recorder.requests[0]["questions"]["c_0"]
    assert question["instructions"].startswith("Is `candidates[0].text` useful")
    assert question["criteria"] == EN_CRITERIA


def test_the_routes_are_registered():
    from jev_rag.rankers import DECIDER_ROUTES

    assert set(DECIDER_ROUTES) == {
        "decider_pointwise",
        "decider_single",
        "decider_pointwise_plain",
        "decider_pointwise_en",
    }
