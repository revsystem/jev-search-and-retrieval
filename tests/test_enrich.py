import pytest

from jev_rag.documents import Chunk
from jev_rag.jev.client import JevClient
from jev_rag.jev.enrich import TOPICS, ChunkEnricher, QueryPlanner
from jev_rag.jev.transport import FakeTransport


def chunk(cid: str = "p1-c0") -> Chunk:
    return Chunk(
        chunk_id=cid, page=1, section="第1節　生成AIの動向", text="利用率は13.5%であった。"
    )


def test_enrichment_asks_every_question_in_one_request_per_chunk():
    transport = FakeTransport(
        choices={"topic": "ai", "content_type": "statistic"},
        nouls={"self_contained": 0.2},
        scores={"info_density": 2.0},
        score_levels=4,
    )
    ChunkEnricher(JevClient(transport)).enrich([chunk()])
    assert len(transport.requests) == 1
    assert set(transport.requests[0]["questions"]) == {
        "topic",
        "content_type",
        "self_contained",
        "info_density",
    }


def test_enrichment_prepends_the_jev_labels_to_the_embedded_text():
    transport = FakeTransport(
        choices={"topic": "ai", "content_type": "statistic"},
        nouls={"self_contained": 0.9},
        scores={"info_density": 3.0},
        score_levels=4,
    )
    enriched = ChunkEnricher(JevClient(transport)).enrich([chunk()])[0]
    text = enriched.embedding_text()
    assert TOPICS["ai"] in text
    assert "利用率は13.5%であった。" in text


def test_a_context_dependent_chunk_gets_its_section_context_restored():
    transport = FakeTransport(
        choices={"topic": "ai", "content_type": "statistic"},
        nouls={"self_contained": 0.1},
        scores={"info_density": 2.0},
        score_levels=4,
    )
    enriched = ChunkEnricher(JevClient(transport)).enrich([chunk()])[0]
    assert "第1節　生成AIの動向" in enriched.embedding_text()
    assert enriched.self_contained is False


def test_enrichment_exposes_filterable_metadata_for_s3_vectors():
    transport = FakeTransport(
        choices={"topic": "network", "content_type": "definition"},
        nouls={"self_contained": 0.9},
        scores={"info_density": 1.0},
        score_levels=4,
    )
    metadata = ChunkEnricher(JevClient(transport)).enrich([chunk()])[0].metadata()
    assert metadata["topic"] == "network"
    assert metadata["content_type"] == "definition"
    assert metadata["page"] == 1


def test_low_confidence_labels_fall_back_to_unlabelled():
    transport = FakeTransport(
        choices={"topic": "ai", "content_type": "statistic"},
        nouls={"self_contained": 0.9},
        scores={"info_density": 1.0},
        score_levels=4,
        confidence=0.2,
    )
    enriched = ChunkEnricher(JevClient(transport), min_confidence=0.6).enrich([chunk()])[0]
    assert enriched.topic is None
    assert enriched.metadata()["topic"] == "unknown"


def test_query_planner_returns_a_metadata_filter_when_confident():
    transport = FakeTransport(choices={"topic": "ai"}, nouls={"single_topic": 0.9}, confidence=0.9)
    plan = QueryPlanner(JevClient(transport)).plan("生成AIの利用率は?")
    assert plan.topic == "ai"
    assert plan.metadata_filter() == {"topic": {"$in": ["ai", "unknown"]}}


def test_query_planner_declines_to_filter_a_broad_query():
    transport = FakeTransport(choices={"topic": "ai"}, nouls={"single_topic": 0.2}, confidence=0.9)
    plan = QueryPlanner(JevClient(transport)).plan("白書の概要を教えて")
    assert plan.metadata_filter() is None


def test_query_planner_declines_to_filter_on_a_low_confidence_topic():
    transport = FakeTransport(choices={"topic": "ai"}, nouls={"single_topic": 0.9}, confidence=0.3)
    assert QueryPlanner(JevClient(transport)).plan("曖昧な問い").metadata_filter() is None


def test_topic_taxonomy_is_a_valid_jev_choice_space():
    assert 2 <= len(TOPICS) <= 255
    assert all(isinstance(v, str) and v for v in TOPICS.values())


def test_enricher_batches_chunks_without_exceeding_the_question_limit():
    transport = FakeTransport(
        choices={"topic": "ai", "content_type": "statistic"},
        nouls={"self_contained": 0.9},
        scores={"info_density": 1.0},
        score_levels=4,
    )
    enriched = ChunkEnricher(JevClient(transport)).enrich([chunk(f"p1-c{i}") for i in range(12)])
    assert len(enriched) == 12
    assert all(len(r["questions"]) <= 32 for r in transport.requests)


def test_enrichment_is_reported_per_chunk_id():
    transport = FakeTransport(
        choices={"topic": "ai", "content_type": "statistic"},
        nouls={"self_contained": 0.9},
        scores={"info_density": 1.0},
        score_levels=4,
    )
    enriched = ChunkEnricher(JevClient(transport)).enrich([chunk("a"), chunk("b")])
    assert [e.chunk.chunk_id for e in enriched] == ["a", "b"]


@pytest.mark.parametrize("topic", list(TOPICS))
def test_every_topic_label_round_trips_through_the_planner(topic):
    transport = FakeTransport(choices={"topic": topic}, nouls={"single_topic": 0.9}, confidence=0.9)
    assert QueryPlanner(JevClient(transport)).plan("q").topic == topic
