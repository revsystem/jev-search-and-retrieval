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


class _Conflict(Exception):
    """Stands in for the SDK's per-client ConflictException."""


class StubS3Vectors:
    """Records calls the way botocore's generated client would accept them."""

    def __init__(self, conflict_on: set[str] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.conflict_on = conflict_on or set()

        class exceptions:  # noqa: N801 - mirrors the SDK's client.exceptions
            ConflictException = _Conflict

        self.exceptions = exceptions

    def _record(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if name in self.conflict_on:
            raise _Conflict(f"{name} already exists")
        return {}

    def create_vector_bucket(self, **kw):
        return self._record("create_vector_bucket", **kw)

    def create_index(self, **kw):
        return self._record("create_index", **kw)

    def put_vectors(self, **kw):
        return self._record("put_vectors", **kw)

    def query_vectors(self, **kw):
        self._record("query_vectors", **kw)
        return {"vectors": [{"key": "a", "distance": 0.25, "metadata": {"text": "A"}}]}


def _store(client):
    from jev_rag.config import VectorStoreSettings
    from jev_rag.vector_store import S3VectorStore

    return S3VectorStore(VectorStoreSettings(bucket="b", index="i", region="us-east-1"), client)


def test_index_creation_uses_the_operation_the_sdk_exposes():
    client = StubS3Vectors()
    _store(client).ensure_index(1024)
    names = [name for name, _ in client.calls]
    assert names == ["create_vector_bucket", "create_index"]


def test_index_creation_sends_the_documented_parameters():
    client = StubS3Vectors()
    _store(client).ensure_index(1024)
    _, kwargs = client.calls[1]
    assert kwargs["dataType"] == "float32"
    assert kwargs["distanceMetric"] == "cosine"
    assert kwargs["dimension"] == 1024
    assert "nonFilterableMetadataKeys" in kwargs["metadataConfiguration"]


def test_an_existing_bucket_and_index_are_not_an_error():
    client = StubS3Vectors(conflict_on={"create_vector_bucket", "create_index"})
    _store(client).ensure_index(1024)
    assert len(client.calls) == 2


def test_any_other_failure_still_propagates():
    import pytest

    class Broken(StubS3Vectors):
        def create_vector_bucket(self, **kw):
            raise ValueError("access denied")

    with pytest.raises(ValueError):
        _store(Broken()).ensure_index(1024)


def test_vectors_are_sent_as_float32_records():
    client = StubS3Vectors()
    _store(client).put(["a"], [[0.5, 0.25]], [{"text": "A"}])
    _, kwargs = client.calls[0]
    assert kwargs["vectors"] == [
        {"key": "a", "data": {"float32": [0.5, 0.25]}, "metadata": {"text": "A"}}
    ]


def test_a_query_converts_distance_into_a_similarity():
    client = StubS3Vectors()
    hits = _store(client).search([0.1, 0.2], top_k=3)
    assert hits[0].score == 0.75
    assert hits[0].text == "A"


def test_a_query_without_a_filter_omits_the_field():
    client = StubS3Vectors()
    _store(client).search([0.1], top_k=1)
    assert "filter" not in client.calls[0][1]


def test_a_query_with_a_filter_passes_it_through():
    client = StubS3Vectors()
    _store(client).search([0.1], top_k=1, metadata_filter={"topic": {"$in": ["ai"]}})
    assert client.calls[0][1]["filter"] == {"topic": {"$in": ["ai"]}}
