"""Index-time chunk enrichment and query-time planning with Jev.

Jev is not an embedding model and cannot change what Cohere Embed v4 produces
for a given string. What it can change is *which string gets embedded* and
*which slice of the index gets searched*:

* ``ChunkEnricher`` labels each chunk (topic, content type, whether it stands
  on its own, how dense it is) in one request, then rewrites the text handed to
  the embedding model so that context-dependent fragments carry their section
  back. The same labels become S3 Vectors metadata.
* ``QueryPlanner`` classifies the query into the same taxonomy and decides
  whether narrowing the search by that metadata is safe.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from jev_rag.documents import Chunk
from jev_rag.jev.client import JevClient
from jev_rag.jev.questions import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer

TOPICS: dict[str, str] = {
    "ai": "生成AI・AIの利活用、普及状況、影響",
    "network": "通信インフラ、ブロードバンド、モバイル、トラヒック",
    "data": "データ流通・利活用、プラットフォーム、コンテンツ",
    "security": "サイバーセキュリティ、情報セキュリティ、偽・誤情報",
    "society": "国民生活・利用動向、デジタル活用、人材",
    "economy": "ICT産業・市場規模、企業活動、経済効果",
    "policy": "政策・制度・規制、国際比較、国際動向",
    "other": "上記のいずれにも明確に当てはまらない",
}

CONTENT_TYPES: dict[str, str] = {
    "statistic": "統計数値、調査結果、割合や金額などの実績値",
    "definition": "用語や制度の定義・説明",
    "narrative": "動向や背景の説明的な記述",
    "figure": "図表のタイトル、凡例、注記",
    "case": "企業や自治体の事例・取組の紹介",
}

DENSITY_LEVELS = [
    "具体的な情報がほとんどない（見出し、ページ番号、断片）",
    "一般的な記述のみで固有の事実を含まない",
    "固有の事実や説明を含む",
    "数値・固有名詞を伴う具体的な事実を複数含む",
]


@dataclass
class EnrichedChunk:
    chunk: Chunk
    topic: str | None
    content_type: str | None
    self_contained: bool
    info_density: float

    def embedding_text(self) -> str:
        """The string actually sent to Cohere Embed v4."""
        header: list[str] = []
        if self.topic:
            header.append(f"分野: {TOPICS[self.topic]}")
        if self.content_type:
            header.append(f"種別: {CONTENT_TYPES[self.content_type]}")
        if not self.self_contained and self.chunk.section:
            header.append(f"文脈: {self.chunk.section}")
        return "\n".join([*header, self.chunk.text])

    def metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "page": self.chunk.page,
            "section": self.chunk.section,
            "topic": self.topic or "unknown",
            "content_type": self.content_type or "unknown",
            "info_density": round(self.info_density, 2),
        }


class ChunkEnricher:
    """One Jev request per chunk, four questions answered in parallel inside it."""

    def __init__(
        self,
        client: JevClient,
        min_confidence: float = 0.5,
        self_contained_threshold: float = 0.5,
        max_workers: int = 8,
    ) -> None:
        self.client = client
        self.min_confidence = min_confidence
        self.self_contained_threshold = self_contained_threshold
        self.max_workers = max_workers

    def enrich(self, chunks: list[Chunk]) -> list[EnrichedChunk]:
        if not chunks:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._enrich_one, chunks))

    def _enrich_one(self, chunk: Chunk) -> EnrichedChunk:
        state = {"section": chunk.section, "page": chunk.page, "text": chunk.text}
        answers = self.client.evaluate(
            state,
            {
                "topic": Choice("この断片が扱う分野を選んでください。", criteria=dict(TOPICS)),
                "content_type": Choice(
                    "この断片の記述の種別を選んでください。", criteria=dict(CONTENT_TYPES)
                ),
                "self_contained": Noul(
                    "この断片は前後の文脈なしで単独で意味が通りますか。",
                    criteria={
                        "true": "主語と対象が明示され、単独で読んで理解できる。",
                        "false": "指示語や省略があり、前後の文脈がないと対象が分からない。",
                    },
                ),
                "info_density": Score(
                    "この断片が含む具体的な情報の量を評価してください。", criteria=DENSITY_LEVELS
                ),
            },
        )

        topic = self._confident_choice(answers.get("topic"), TOPICS)
        content_type = self._confident_choice(answers.get("content_type"), CONTENT_TYPES)
        self_contained = answers.get("self_contained")
        density = answers.get("info_density")
        return EnrichedChunk(
            chunk=chunk,
            topic=topic,
            content_type=content_type,
            self_contained=isinstance(self_contained, NoulAnswer)
            and self_contained.value >= self.self_contained_threshold,
            info_density=density.score if isinstance(density, ScoreAnswer) else 0.0,
        )

    def _confident_choice(self, answer: object, vocabulary: dict[str, str]) -> str | None:
        if not isinstance(answer, ChoiceAnswer) or answer.choice not in vocabulary:
            return None
        if (answer.confidence or 0.0) < self.min_confidence:
            return None
        return answer.choice


@dataclass
class QueryPlan:
    query: str
    topic: str | None
    topic_confidence: float
    single_topic: float
    min_confidence: float = 0.5
    single_topic_threshold: float = 0.5

    def metadata_filter(self) -> dict[str, Any] | None:
        """An S3 Vectors metadata filter, or None when narrowing would risk recall."""
        if not self.topic or self.topic == "other":
            return None
        if self.single_topic < self.single_topic_threshold:
            return None
        if self.topic_confidence < self.min_confidence:
            return None
        # "unknown" is kept so that chunks Jev could not label confidently stay reachable.
        return {"topic": {"$in": [self.topic, "unknown"]}}


class QueryPlanner:
    def __init__(self, client: JevClient, min_confidence: float = 0.5) -> None:
        self.client = client
        self.min_confidence = min_confidence

    def plan(self, query: str) -> QueryPlan:
        answers = self.client.evaluate(
            {"query": query},
            {
                "topic": Choice("この質問が属する分野を選んでください。", criteria=dict(TOPICS)),
                "single_topic": Noul(
                    "この質問は単一の分野に限定されていますか。",
                    criteria={
                        "true": "対象の分野が一つに絞り込める。",
                        "false": "複数分野にまたがる、または白書全体を問う広い質問。",
                    },
                ),
            },
        )
        topic = answers.get("topic")
        single = answers.get("single_topic")
        return QueryPlan(
            query=query,
            topic=topic.choice if isinstance(topic, ChoiceAnswer) else None,
            topic_confidence=(topic.confidence or 0.0) if isinstance(topic, ChoiceAnswer) else 0.0,
            single_topic=single.value if isinstance(single, NoulAnswer) else 0.0,
            min_confidence=self.min_confidence,
        )
