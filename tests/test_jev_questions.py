import pytest

from jev_rag.jev.questions import Choice, Noul, Score, parse_answer


def test_noul_payload_is_minimal_without_criteria():
    assert Noul("Is this relevant?").payload() == {
        "type": "noul",
        "instructions": "Is this relevant?",
    }


def test_noul_payload_carries_true_false_criteria():
    q = Noul("Relevant?", criteria={"true": "yes", "false": "no"})
    assert q.payload()["criteria"] == {"true": "yes", "false": "no"}


def test_choice_requires_at_least_two_options():
    with pytest.raises(ValueError):
        Choice("Pick one", criteria={"only": None})


def test_choice_rejects_more_than_255_options():
    with pytest.raises(ValueError):
        Choice("Pick one", criteria={f"o{i}": None for i in range(256)})


def test_choice_payload_keeps_option_descriptions():
    q = Choice("Topic?", criteria={"ai": "AI topics", "network": None})
    assert q.payload() == {
        "type": "choice",
        "instructions": "Topic?",
        "criteria": {"ai": "AI topics", "network": None},
    }


def test_score_requires_at_least_two_levels():
    with pytest.raises(ValueError):
        Score("How relevant?", criteria=["only one"])


def test_score_rejects_more_than_ten_levels():
    with pytest.raises(ValueError):
        Score("How relevant?", criteria=[f"level {i}" for i in range(11)])


def test_score_payload_preserves_level_order():
    q = Score("How relevant?", criteria=["none", "partial", "direct"])
    assert q.payload()["criteria"] == ["none", "partial", "direct"]


def test_parse_noul_answer():
    answer = parse_answer({"type": "noul", "noul": 0.92})
    assert answer.value == 0.92
    assert answer.confidence is None


def test_parse_choice_answer():
    answer = parse_answer(
        {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
            "confidence": 0.82,
        }
    )
    assert answer.choice == "technical"
    assert answer.confidence == 0.82
    assert answer.probabilities["technical"] == 0.85


def test_parse_score_answer_keeps_fractional_value():
    answer = parse_answer(
        {
            "type": "score",
            "score": 1.6,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
            "confidence": 0.78,
        }
    )
    assert answer.score == 1.6
    assert answer.normalised(levels=3) == pytest.approx(0.8)


def test_parse_rejects_unknown_answer_type():
    with pytest.raises(ValueError):
        parse_answer({"type": "vibes", "value": 1})
