"""Request sizing follows the documented token budget, not a question count.

TypeSafe documents 64k tokens per request covering the state plus all
questions, and 32k for the state plus the single longest question. There is no
documented cap on how many questions one request may carry — the line-by-line
search cookbook scores 218 candidates in a single call.
"""

import pytest

from jev_rag.jev.budget import (
    STATE_PLUS_QUESTION_BUDGET,
    TOTAL_TOKEN_BUDGET,
    batch_by_budget,
    estimate_tokens,
)


def test_japanese_costs_about_one_token_per_character():
    assert 90 <= estimate_tokens("あ" * 100) <= 150


def test_ascii_costs_well_under_one_token_per_character():
    assert estimate_tokens("a" * 100) < 50


def test_structured_state_is_measured_as_sent():
    assert estimate_tokens({"query": "あ" * 100}) >= estimate_tokens("あ" * 100)


def test_a_small_question_set_stays_in_one_batch():
    items = [(f"q{i}", "短い質問") for i in range(50)]
    assert len(batch_by_budget(items, state_tokens=100)) == 1


def test_a_set_over_the_total_budget_is_split():
    heavy = "あ" * 5_000
    items = [(f"q{i}", heavy) for i in range(30)]
    batches = batch_by_budget(items, state_tokens=1_000)
    assert len(batches) > 1
    assert sum(len(b) for b in batches) == 30


def test_no_batch_exceeds_the_total_budget():
    heavy = "あ" * 5_000
    batches = batch_by_budget([(f"q{i}", heavy) for i in range(30)], state_tokens=1_000)
    for batch in batches:
        assert 1_000 + sum(estimate_tokens(t) for _, t in batch) <= TOTAL_TOKEN_BUDGET


def test_every_question_survives_the_split_exactly_once():
    items = [(f"q{i}", "あ" * 4_000) for i in range(25)]
    keys = [key for batch in batch_by_budget(items, state_tokens=500) for key, _ in batch]
    assert sorted(keys) == sorted(key for key, _ in items)


def test_a_single_question_too_large_for_the_pair_budget_is_rejected():
    with pytest.raises(ValueError, match="32"):
        batch_by_budget([("q", "あ" * 40_000)], state_tokens=1_000)


def test_the_pair_budget_is_measured_against_the_state_not_the_batch():
    # one question fits beside a small state but not beside a large one
    question = "あ" * 20_000
    assert batch_by_budget([("q", question)], state_tokens=1_000)
    with pytest.raises(ValueError, match="32"):
        batch_by_budget([("q", question)], state_tokens=STATE_PLUS_QUESTION_BUDGET)


def test_an_optional_count_cap_still_applies_when_asked_for():
    items = [(f"q{i}", "短い") for i in range(10)]
    assert len(batch_by_budget(items, state_tokens=10, max_questions=4)) == 3


def test_no_count_cap_by_default():
    items = [(f"q{i}", "短い") for i in range(300)]
    assert len(batch_by_budget(items, state_tokens=10)) == 1
