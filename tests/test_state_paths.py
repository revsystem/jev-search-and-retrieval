"""State references must follow the documented path syntax, and must point at
the candidate their question is about.

TypeSafe documents a backticked dot-and-index path relative to the state root,
such as ``ticket.messages[0].text`` (primitives.md, "Reference specific
fields"). Question ids are not sent to the model, so the instruction itself has
to carry the reference.

Using an array index in the path opens a failure this file is here to catch:
the question is keyed by doc id while the path is keyed by position, and
nothing structural keeps the two pointing at the same passage. A mismatch
mis-scores every candidate and looks like the model being bad at the language.
"""

import re

import pytest

from jev_rag.jev.client import JevClient
from jev_rag.jev.context import ContextSelector
from jev_rag.jev.crossencode import CrossEncoder
from jev_rag.jev.pairwise import PairwiseReranker
from jev_rag.jev.rerank import JevReranker
from jev_rag.jev.transport import FakeTransport
from jev_rag.types import RetrievedDoc

INDEXED_PATH = re.compile(r"`candidates\[(\d+)\]\.text`")


def docs() -> list[RetrievedDoc]:
    return [
        RetrievedDoc(doc_id="5191928", text="絶対零度は摂氏マイナス273.15度である。", score=0.3),
        RetrievedDoc(doc_id="4823111", text="ケルビンは熱力学温度の単位である。", score=0.2),
        RetrievedDoc(doc_id="9900021", text="無関係な本文。", score=0.1),
    ]


def request_of(transport: FakeTransport) -> dict:
    assert transport.requests, "no request was sent"
    return transport.requests[0]


def run(builder) -> tuple[FakeTransport, list[RetrievedDoc]]:
    transport = FakeTransport(noul=0.5, score_levels=4, confidence=0.9)
    candidates = docs()
    builder(JevClient(transport), candidates)
    return transport, candidates


LISTWISE = [
    pytest.param(lambda c, d: JevReranker(c).noul_rerank("問い", d), id="noul_rerank"),
    pytest.param(lambda c, d: JevReranker(c).rerank("問い", d), id="score_rerank"),
    pytest.param(lambda c, d: JevReranker(c).relevance_filter("問い", d), id="relevance_filter"),
    pytest.param(lambda c, d: ContextSelector(c).select("問い", d), id="context_selector"),
]


@pytest.mark.parametrize("builder", LISTWISE)
def test_the_state_holds_candidates_as_an_array(builder):
    transport, _ = run(builder)
    candidates = request_of(transport)["state"]["candidates"]
    assert isinstance(candidates, list)
    assert all(set(entry) == {"text"} for entry in candidates)


@pytest.mark.parametrize("builder", LISTWISE)
def test_every_instruction_uses_a_backticked_indexed_path(builder):
    transport, _ = run(builder)
    for question in request_of(transport)["questions"].values():
        assert INDEXED_PATH.search(question["instructions"]), question["instructions"]


@pytest.mark.parametrize("builder", LISTWISE)
def test_no_instruction_keeps_the_state_prefix(builder):
    transport, _ = run(builder)
    for question in request_of(transport)["questions"].values():
        assert "state." not in question["instructions"]


@pytest.mark.parametrize("builder", LISTWISE)
def test_each_question_points_at_the_candidate_it_is_keyed_by(builder):
    """The assertion that matters: index in the path resolves to the keyed doc."""
    transport, candidates = run(builder)
    request = request_of(transport)
    by_id = {doc.doc_id: doc.text for doc in candidates}

    for key, question in request["questions"].items():
        doc_id = key.split("__")[0]
        match = INDEXED_PATH.search(question["instructions"])
        index = int(match.group(1))
        assert request["state"]["candidates"][index]["text"] == by_id[doc_id]


@pytest.mark.parametrize("builder", LISTWISE)
def test_the_query_is_referenced_by_its_own_backticked_path(builder):
    # the redundancy question compares candidates with each other, so it is the
    # one judgement that legitimately does not mention the query
    transport, _ = run(builder)
    asked = [
        question
        for key, question in request_of(transport)["questions"].items()
        if not key.endswith("__redundant")
    ]
    assert asked
    for question in asked:
        assert "`query`" in question["instructions"]


def test_cross_encoding_references_the_single_candidate_directly():
    transport = FakeTransport(noul=0.5)
    CrossEncoder(JevClient(transport)).rerank("問い", docs())
    for request in transport.requests:
        for question in request["questions"].values():
            assert "`candidate`" in question["instructions"]
            assert "state." not in question["instructions"]


def test_pairwise_options_are_neutral_labels_not_document_ids():
    """Choice option keys are shown to the model, so a raw id would be noise."""
    transport = FakeTransport(choices={}, confidence=0.9)
    PairwiseReranker(JevClient(transport)).rerank("問い", docs())
    for question in request_of(transport)["questions"].values():
        assert set(question["criteria"]) == {"a", "b"}


def test_pairwise_describes_each_option_by_its_indexed_path():
    transport = FakeTransport(choices={}, confidence=0.9)
    PairwiseReranker(JevClient(transport)).rerank("問い", docs())
    request = request_of(transport)
    for question in request["questions"].values():
        for description in question["criteria"].values():
            assert INDEXED_PATH.search(description), description


def test_pairwise_maps_the_winning_label_back_to_the_right_document():
    # option "a" always wins; the candidate at the first index of each pair
    # must be the one that gains
    transport = FakeTransport(choices={}, confidence=0.95)
    ranked = PairwiseReranker(JevClient(transport), opponents=2).rerank("問い", docs())
    request = request_of(transport)
    texts = [entry["text"] for entry in request["state"]["candidates"]]

    winners: set[str] = set()
    for question in request["questions"].values():
        index = int(INDEXED_PATH.search(question["criteria"]["a"]).group(1))
        winners.add(texts[index])
    top = next(d for d in ranked if d.score == max(r.score for r in ranked))
    assert top.text in winners


def test_no_module_writes_the_old_state_prefixed_reference():
    """Guard the class of bug, not just the sites that had it.

    The first fix missed cli.py because the search was scoped to the jev
    package. This walks the syntax tree of every source file, so the next
    module that grows a question cannot reintroduce the form, and comments
    and prose about "the state" cannot raise a false alarm.
    """
    import ast as ast_module
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in sorted(src.rglob("*.py")):
        if path.name == "state.py":
            continue
        tree = ast_module.parse(path.read_text(encoding="utf-8"))
        for node in ast_module.walk(tree):
            if not isinstance(node, ast_module.Constant) or not isinstance(node.value, str):
                continue
            if re.search(r"\bstate\.[a-z_]+", node.value):
                offenders.append(f"{path.name}:{node.lineno}: {node.value[:60]}")
    assert offenders == [], "state-prefixed reference in a string literal:\n" + "\n".join(offenders)
