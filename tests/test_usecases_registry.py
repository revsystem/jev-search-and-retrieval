"""The registry must keep matching TypeSafe's published Search and retrieval list.

These tests are the guard against drift: if a module is renamed, a pipeline
disappears or a test file is deleted, the mapping the README advertises fails
here instead of silently becoming a lie.
"""

import importlib
from pathlib import Path

import pytest

from jev_rag.rankers import ALL_RANKERS
from jev_rag.usecases import DECISION_SHAPES, USE_CASES

ROOT = Path(__file__).resolve().parents[1]

DOCUMENTED = [
    "Replace or supplement embeddings in RAG pipelines with semantic search, scoring, and ranking.",
    "Score query-to-candidate relevance.",
    "Rerank results with pairwise comparisons.",
    "Cross-encode queries and candidates for higher precision.",
    "Select useful context for downstream AI workflows.",
]


def test_the_registry_holds_exactly_the_published_use_cases():
    assert [uc.bullet for uc in USE_CASES] == DOCUMENTED


@pytest.mark.parametrize("use_case", USE_CASES, ids=lambda uc: uc.key)
def test_every_use_case_names_an_importable_module(use_case):
    assert importlib.import_module(use_case.module)


@pytest.mark.parametrize("use_case", USE_CASES, ids=lambda uc: uc.key)
def test_every_named_pipeline_is_runnable(use_case):
    assert set(use_case.pipelines) <= set(ALL_RANKERS)


@pytest.mark.parametrize("use_case", USE_CASES, ids=lambda uc: uc.key)
def test_every_use_case_has_a_test_file(use_case):
    assert (ROOT / use_case.tests).exists()


@pytest.mark.parametrize("use_case", USE_CASES, ids=lambda uc: uc.key)
def test_every_use_case_has_an_entry_point(use_case):
    assert use_case.pipelines or use_case.entrypoint


@pytest.mark.parametrize("use_case", USE_CASES, ids=lambda uc: uc.key)
def test_decision_shapes_come_from_the_published_table(use_case):
    assert use_case.shapes
    assert set(use_case.shapes) <= DECISION_SHAPES


def test_the_keys_are_unique():
    assert len({uc.key for uc in USE_CASES}) == len(USE_CASES)


def test_the_registry_renders_a_table():
    from jev_rag.usecases import format_registry

    table = format_registry()
    assert "jev_crossencode" in table
    assert "Score query-to-candidate relevance." in table
