"""Environment-driven settings for the comparison harness."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

PDF_URL = "https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/r08/pdf/n1110000.pdf"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class BedrockSettings:
    region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    embedding_model_id: str = field(
        default_factory=lambda: _env("BEDROCK_EMBEDDING_MODEL_ID", "cohere.embed-v4:0")
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(_env("BEDROCK_EMBEDDING_DIMENSION", "1024"))
    )
    generation_model_id: str = field(
        default_factory=lambda: _env("BEDROCK_GENERATION_MODEL_ID", "us.openai.gpt-5.6-luna")
    )
    rerank_model_id: str = field(
        default_factory=lambda: _env("BEDROCK_RERANK_MODEL_ID", "cohere.rerank-v3-5:0")
    )

    @property
    def rerank_model_arn(self) -> str:
        return f"arn:aws:bedrock:{self.region}::foundation-model/{self.rerank_model_id}"


@dataclass
class VectorStoreSettings:
    bucket: str = field(default_factory=lambda: _env("S3_VECTOR_BUCKET", "jev-rag-comparison"))
    index: str = field(default_factory=lambda: _env("S3_VECTOR_INDEX", "soumu-whitepaper-r08"))
    region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))


@dataclass
class JevSettings:
    transport: str = field(default_factory=lambda: _env("JEV_TRANSPORT", "gateway"))
    gateway_api_key: str = field(default_factory=lambda: _env("AI_GATEWAY_API_KEY"))
    gateway_base_url: str = field(
        default_factory=lambda: _env(
            "JEV_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/typesafe/v1"
        )
    )
    gateway_model: str = field(default_factory=lambda: _env("JEV_GATEWAY_MODEL", "typesafe-ai/jev"))
    direct_api_key: str = field(default_factory=lambda: _env("TYPESAFE_API_KEY"))
    direct_base_url: str = field(
        default_factory=lambda: _env("JEV_DIRECT_BASE_URL", "https://api.typesafe.ai/v1")
    )
    direct_model: str = field(default_factory=lambda: _env("JEV_DIRECT_MODEL", "jev-latest"))

    def build_client(self):
        """Build a JevClient for the configured route.

        The request body is identical on both routes; only the host, the key
        and the routed model id differ.
        """
        from jev_rag.jev.client import JevClient
        from jev_rag.jev.transport import build_transport

        if self.transport == "gateway":
            transport = build_transport(
                "gateway",
                api_key=self.gateway_api_key,
                base_url=self.gateway_base_url,
                model=self.gateway_model,
            )
        elif self.transport == "direct":
            transport = build_transport(
                "direct",
                api_key=self.direct_api_key,
                base_url=self.direct_base_url,
                model=self.direct_model,
            )
        else:
            transport = build_transport(self.transport)
        return JevClient(transport)


@dataclass
class Settings:
    bedrock: BedrockSettings = field(default_factory=BedrockSettings)
    store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    jev: JevSettings = field(default_factory=JevSettings)
    pdf_url: str = PDF_URL
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 5
    candidate_k: int = 30
