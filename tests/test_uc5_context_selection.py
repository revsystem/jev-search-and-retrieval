"""Use case: select useful context for downstream AI workflows."""

from jev_rag.evaluation import RetrievedDoc
from jev_rag.jev.client import JevClient
from jev_rag.jev.context import ContextSelector
from jev_rag.jev.transport import FakeTransport


def candidates() -> list[RetrievedDoc]:
    return [
        RetrievedDoc(chunk_id="useful", text="あ" * 100, score=0.9),
        RetrievedDoc(chunk_id="duplicate", text="い" * 100, score=0.85),
        RetrievedDoc(chunk_id="filler", text="う" * 100, score=0.80),
    ]


def scripted() -> FakeTransport:
    return FakeTransport(
        scores={"useful__usefulness": 3.0, "duplicate__usefulness": 2.5, "filler__usefulness": 0.3},
        nouls={"useful__redundant": 0.02, "duplicate__redundant": 0.93, "filler__redundant": 0.10},
        score_levels=4,
    )


def test_usefulness_and_redundancy_are_asked_for_every_candidate():
    transport = scripted()
    ContextSelector(JevClient(transport)).select("問い", candidates())
    asked = {key for request in transport.requests for key in request["questions"]}
    assert asked == {
        "useful__usefulness",
        "duplicate__usefulness",
        "filler__usefulness",
        "useful__redundant",
        "duplicate__redundant",
        "filler__redundant",
    }


def test_redundant_context_is_dropped_even_when_it_is_useful():
    result = ContextSelector(JevClient(scripted())).select("問い", candidates())
    assert "duplicate" not in {doc.chunk_id for doc in result.selected}
    assert result.reasons["duplicate"] == "redundant"


def test_low_value_context_is_dropped():
    result = ContextSelector(JevClient(scripted())).select("問い", candidates())
    assert result.reasons["filler"] == "not_useful"


def test_the_useful_passage_survives():
    result = ContextSelector(JevClient(scripted())).select("問い", candidates())
    assert [doc.chunk_id for doc in result.selected] == ["useful"]


def test_selection_respects_a_character_budget():
    transport = FakeTransport(
        scores={f"{c}__usefulness": 3.0 for c in ("useful", "duplicate", "filler")},
        nouls={f"{c}__redundant": 0.0 for c in ("useful", "duplicate", "filler")},
        score_levels=4,
    )
    result = ContextSelector(JevClient(transport), budget_chars=250).select("問い", candidates())
    assert len(result.selected) == 2
    assert result.reasons["filler"] == "over_budget"
    assert result.used_chars == 200


def test_the_most_useful_context_is_packed_first():
    transport = FakeTransport(
        scores={"useful__usefulness": 1.0, "duplicate__usefulness": 3.0, "filler__usefulness": 2.0},
        nouls={f"{c}__redundant": 0.0 for c in ("useful", "duplicate", "filler")},
        score_levels=4,
    )
    result = ContextSelector(JevClient(transport), budget_chars=150).select("問い", candidates())
    assert [doc.chunk_id for doc in result.selected] == ["duplicate"]


def test_selected_context_carries_its_usefulness_score():
    result = ContextSelector(JevClient(scripted())).select("問い", candidates())
    assert result.selected[0].jev_score == 3.0


def test_the_selection_never_returns_nothing_when_candidates_exist():
    transport = FakeTransport(
        scores={f"{c}__usefulness": 0.0 for c in ("useful", "duplicate", "filler")},
        nouls={f"{c}__redundant": 1.0 for c in ("useful", "duplicate", "filler")},
        score_levels=4,
    )
    result = ContextSelector(JevClient(transport)).select("問い", candidates())
    assert len(result.selected) == 1


def test_no_candidates_means_no_request():
    transport = FakeTransport()
    result = ContextSelector(JevClient(transport)).select("問い", [])
    assert result.selected == [] and transport.requests == []
