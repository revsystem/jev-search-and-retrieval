from jev_rag.vector_store import (
    EmbeddingRetriever,
    InMemoryVectorStore,
    cosine_similarity,
    matches_filter,
)


def test_cosine_similarity_of_identical_vectors_is_one():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_of_a_zero_vector_is_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_no_filter_matches_everything():
    assert matches_filter({"topic": "ai"}, None) is True


def test_in_operator_keeps_listed_values():
    assert matches_filter({"topic": "ai"}, {"topic": {"$in": ["ai", "unknown"]}}) is True
    assert matches_filter({"topic": "network"}, {"topic": {"$in": ["ai", "unknown"]}}) is False


def test_bare_value_is_treated_as_equality():
    assert matches_filter({"topic": "ai"}, {"topic": "ai"}) is True
    assert matches_filter({"topic": "ai"}, {"topic": "network"}) is False


def test_numeric_comparison_operators():
    assert matches_filter({"info_density": 2.5}, {"info_density": {"$gte": 2}}) is True
    assert matches_filter({"info_density": 1.0}, {"info_density": {"$gte": 2}}) is False


def test_missing_metadata_key_fails_a_comparison_rather_than_raising():
    assert matches_filter({}, {"info_density": {"$gte": 2}}) is False


def test_store_returns_nearest_neighbours_first():
    store = InMemoryVectorStore()
    store.put(
        ["a", "b"],
        [[1.0, 0.0], [0.0, 1.0]],
        [{"text": "A", "topic": "ai"}, {"text": "B", "topic": "network"}],
    )
    hits = store.search([1.0, 0.1], top_k=2)
    assert [h.chunk_id for h in hits] == ["a", "b"]
    assert hits[0].text == "A"


def test_store_applies_the_metadata_filter_before_ranking():
    store = InMemoryVectorStore()
    store.put(
        ["a", "b"],
        [[1.0, 0.0], [0.0, 1.0]],
        [{"text": "A", "topic": "ai"}, {"text": "B", "topic": "network"}],
    )
    hits = store.search([1.0, 0.0], top_k=2, metadata_filter={"topic": {"$in": ["network"]}})
    assert [h.chunk_id for h in hits] == ["b"]


def test_retriever_embeds_the_query_before_searching():
    class Embedder:
        def embed_query(self, text):
            return [1.0, 0.0] if "A" in text else [0.0, 1.0]

    store = InMemoryVectorStore()
    store.put(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], [{"text": "A"}, {"text": "B"}])
    assert EmbeddingRetriever(Embedder(), store).search("A", top_k=1)[0].chunk_id == "a"
