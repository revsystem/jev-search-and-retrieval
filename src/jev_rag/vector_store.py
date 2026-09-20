"""Vector stores: Amazon S3 Vectors, plus an in-memory store for offline runs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from jev_rag.config import VectorStoreSettings
from jev_rag.evaluation import RetrievedDoc

# S3 Vectors accepts at most 500 vectors per PutVectors request.
PUT_BATCH_SIZE = 500
# Chunk text is stored alongside the vector but never filtered on, so it is
# declared non-filterable: filterable metadata counts against a tighter budget.
NON_FILTERABLE_KEYS = ["text", "section"]


def matches_filter(metadata: dict[str, Any], expression: dict[str, Any] | None) -> bool:
    """Evaluate the subset of the S3 Vectors filter grammar this project uses."""
    if not expression:
        return True
    for key, condition in expression.items():
        value = metadata.get(key)
        if isinstance(condition, dict):
            for operator, operand in condition.items():
                if operator == "$eq" and value != operand:
                    return False
                if operator == "$ne" and value == operand:
                    return False
                if operator == "$in" and value not in operand:
                    return False
                if operator == "$nin" and value in operand:
                    return False
                if operator == "$gte" and not (value is not None and value >= operand):
                    return False
                if operator == "$lte" and not (value is not None and value <= operand):
                    return False
        elif value != condition:
            return False
    return True


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@dataclass
class InMemoryVectorStore:
    """Offline stand-in with the same surface as the S3 Vectors store."""

    vectors: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def ensure_index(self, dimension: int) -> None:  # noqa: ARG002 - parity with S3VectorStore
        return None

    def put(self, keys: list[str], vectors: list[list[float]], metadata: list[dict]) -> None:
        for key, vector, meta in zip(keys, vectors, metadata, strict=True):
            self.vectors[key] = vector
            self.metadata[key] = meta

    def search(
        self, query_vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[RetrievedDoc]:
        scored = [
            RetrievedDoc(
                chunk_id=key,
                text=self.metadata[key].get("text", ""),
                score=cosine_similarity(query_vector, vector),
                metadata=self.metadata[key],
            )
            for key, vector in self.vectors.items()
            if matches_filter(self.metadata[key], metadata_filter)
        ]
        scored.sort(key=lambda doc: doc.score, reverse=True)
        return scored[:top_k]


class S3VectorStore:
    """Amazon S3 Vectors index used as the RAG vector store."""

    def __init__(self, settings: VectorStoreSettings | None = None, client: Any = None) -> None:
        self.settings = settings or VectorStoreSettings()
        if client is None:
            import boto3

            client = boto3.client("s3vectors", region_name=self.settings.region)
        self.client = client

    def ensure_index(self, dimension: int) -> None:
        """Create the vector bucket and index when they do not exist yet."""
        self._ignore_conflict(
            self.client.create_vector_bucket, vectorBucketName=self.settings.bucket
        )
        self._ignore_conflict(
            self.client.create_index,
            vectorBucketName=self.settings.bucket,
            indexName=self.settings.index,
            dataType="float32",
            dimension=dimension,
            distanceMetric="cosine",
            metadataConfiguration={"nonFilterableMetadataKeys": NON_FILTERABLE_KEYS},
        )

    def _ignore_conflict(self, operation, **kwargs) -> None:
        """Creating a bucket or index that already exists is the expected path."""
        try:
            operation(**kwargs)
        except self.client.exceptions.ConflictException:
            return

    def put(self, keys: list[str], vectors: list[list[float]], metadata: list[dict]) -> None:
        records = [
            {"key": key, "data": {"float32": [float(v) for v in vector]}, "metadata": meta}
            for key, vector, meta in zip(keys, vectors, metadata, strict=True)
        ]
        for start in range(0, len(records), PUT_BATCH_SIZE):
            self.client.put_vectors(
                vectorBucketName=self.settings.bucket,
                indexName=self.settings.index,
                vectors=records[start : start + PUT_BATCH_SIZE],
            )

    def search(
        self, query_vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[RetrievedDoc]:
        request: dict[str, Any] = {
            "vectorBucketName": self.settings.bucket,
            "indexName": self.settings.index,
            "queryVector": {"float32": [float(v) for v in query_vector]},
            "topK": top_k,
            "returnDistance": True,
            "returnMetadata": True,
        }
        if metadata_filter:
            request["filter"] = metadata_filter
        response = self.client.query_vectors(**request)
        return [
            RetrievedDoc(
                chunk_id=hit["key"],
                text=(hit.get("metadata") or {}).get("text", ""),
                # cosine distance -> similarity, so that higher is always better
                score=1.0 - float(hit.get("distance", 0.0)),
                metadata=hit.get("metadata") or {},
            )
            for hit in response.get("vectors", [])
        ]


class EmbeddingRetriever:
    """Binds an embedding model to a store so pipelines can just call ``search``."""

    def __init__(self, embedder, store) -> None:
        self.embedder = embedder
        self.store = store

    def search(
        self, query: str, top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[RetrievedDoc]:
        return self.store.search(self.embedder.embed_query(query), top_k, metadata_filter)
