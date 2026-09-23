"""Amazon Bedrock clients: Cohere Embed v4, Cohere Rerank and GPT-5.6 Luna."""

from __future__ import annotations

import json
from typing import Any

from jev_rag.config import BedrockSettings
from jev_rag.types import RetrievedDoc

# Cohere's embedding endpoint accepts at most 96 texts per call.
EMBED_BATCH_SIZE = 96


def _client(service: str, region: str):
    import boto3

    return boto3.client(service, region_name=region)


class CohereEmbedder:
    """Cohere Embed v4 through Bedrock's InvokeModel API.

    ``input_type`` matters: documents and queries are embedded into the same
    space but with different prefixes, and mixing them up quietly costs recall.
    """

    def __init__(self, settings: BedrockSettings | None = None, client: Any = None) -> None:
        self.settings = settings or BedrockSettings()
        self.client = client or _client("bedrock-runtime", self.settings.region)

    def _invoke(self, texts: list[str], input_type: str) -> list[list[float]]:
        body = {
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"],
            "output_dimension": self.settings.embedding_dimension,
            "truncate": "END",
        }
        response = self.client.invoke_model(
            modelId=self.settings.embedding_model_id, body=json.dumps(body)
        )
        payload = json.loads(response["body"].read())
        embeddings = payload["embeddings"]
        # Embed v4 returns {"embeddings": {"float": [...]}}; older revisions a bare list.
        return embeddings["float"] if isinstance(embeddings, dict) else embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            vectors.extend(self._invoke(texts[start : start + EMBED_BATCH_SIZE], "search_document"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._invoke([text], "search_query")[0]


class BedrockReranker:
    """Cohere Rerank v3.5 through the Bedrock Rerank API — the classic baseline."""

    def __init__(self, settings: BedrockSettings | None = None, client: Any = None) -> None:
        self.settings = settings or BedrockSettings()
        self.client = client or _client("bedrock-agent-runtime", self.settings.region)

    def rerank(self, query: str, docs: list[RetrievedDoc], top_k: int) -> list[RetrievedDoc]:
        if not docs:
            return []
        response = self.client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=[
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": doc.text}},
                }
                for doc in docs
            ],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": min(top_k, len(docs)),
                    "modelConfiguration": {"modelArn": self.settings.rerank_model_arn},
                },
            },
        )
        ranked: list[RetrievedDoc] = []
        for position, result in enumerate(response["results"]):
            doc = docs[result["index"]]
            doc.vector_score = doc.score
            doc.score = result["relevanceScore"]
            doc.rank_delta = result["index"] - position
            ranked.append(doc)
        return ranked


SYSTEM_PROMPT = (
    "与えられた文章だけを根拠に、クイズの質問へ答えてください。"
    "答えは名称のみを一つ、余計な説明を付けずに出力します。"
    "根拠が見当たらない場合は「不明」とだけ答えてください。"
)


class BedrockGenerator:
    """Answer generation with GPT-5.6 Luna through the Bedrock Converse API."""

    def __init__(self, settings: BedrockSettings | None = None, client: Any = None) -> None:
        self.settings = settings or BedrockSettings()
        self.client = client or _client("bedrock-runtime", self.settings.region)

    def answer(self, query: str, docs: list[RetrievedDoc], max_tokens: int = 64) -> str:
        context = "\n\n".join(f"- {doc.text}" for doc in docs)
        response = self.client.converse(
            modelId=self.settings.generation_model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[
                {"role": "user", "content": [{"text": f"# 文章\n{context}\n\n# 質問\n{query}"}]}
            ],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        return "".join(block.get("text", "") for block in response["output"]["message"]["content"])
