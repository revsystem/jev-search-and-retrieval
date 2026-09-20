"""JQaRA: the evaluation corpus.

JQaRA pairs 1,667 Japanese questions with 100 Wikipedia passages each and
labels every passage 1 when it lets a model answer the question. Those labels
come with the dataset, which is why this project uses it: relevance judgements
are the part of a retrieval comparison that is hardest to invent credibly, and
here they already exist and are public.

Licence: the questions and answers inherit CC-BY-SA-4.0 from JAQKET, and the
passages are Wikipedia text under CC BY-SA 4.0 or GFDL. Attribution is required
and derivatives share alike.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_rag.types import RetrievedDoc

HF_URL = (
    "https://huggingface.co/datasets/hotchpotch/JQaRA/resolve/main/data/test-00000-of-00001.parquet"
)
DEFAULT_PATH = Path("data/raw/jqara_test.parquet")


@dataclass
class EvalQuery:
    query_id: str
    question: str
    candidates: list[RetrievedDoc]

    @property
    def total_relevant(self) -> int:
        return sum(1 for c in self.candidates if c.label)

    def as_documents(self) -> list[RetrievedDoc]:
        """A fresh, unscored copy, so one ranker cannot see another's scores."""
        return [
            RetrievedDoc(doc_id=c.doc_id, text=c.text, score=0.0, label=c.label, title=c.title)
            for c in self.candidates
        ]


def to_queries(rows: list[dict[str, Any]], max_candidates: int | None = None) -> list[EvalQuery]:
    """Group flat dataset rows into one query per question.

    A capped candidate list keeps the relevant passages: dropping them at
    random would quietly make some queries unanswerable and depress every
    ranker equally, which looks like a result but is an artefact of sampling.

    The kept list is then shuffled, because keeping the relevant passages puts
    them at the front, and an input order that encodes the labels hands every
    ranker a free head start — a ranker that returns its input unchanged would
    score perfectly. The shuffle is seeded per query so runs stay comparable.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    questions: dict[str, str] = {}
    for row in rows:
        grouped.setdefault(row["q_id"], []).append(row)
        questions[row["q_id"]] = row["question"]

    queries: list[EvalQuery] = []
    for query_id, group in grouped.items():
        if max_candidates is not None:
            relevant = [r for r in group if r["label"]]
            rest = [r for r in group if not r["label"]]
            group = (relevant + rest)[:max_candidates]
        group = sorted(group, key=lambda r: str(r["passage_row_id"]))
        random.Random(f"{query_id}:{len(group)}").shuffle(group)
        candidates = [
            RetrievedDoc(
                doc_id=str(row["passage_row_id"]),
                text=row["text"],
                score=0.0,
                label=int(row["label"]),
                title=row.get("title", ""),
            )
            for row in group
        ]
        if not any(c.label for c in candidates):
            continue
        queries.append(
            EvalQuery(query_id=query_id, question=questions[query_id], candidates=candidates)
        )
    return queries


def sample_queries(queries: list[EvalQuery], count: int, seed: int = 0) -> list[EvalQuery]:
    if count >= len(queries):
        return list(queries)
    return random.Random(seed).sample(queries, count)


def load_jqara(path: str | Path = DEFAULT_PATH, max_candidates: int | None = None):
    """Read the parquet file into evaluation queries."""
    target = Path(path)
    if not target.exists():
        raise SystemExit(f"{target} がありません。先に `jev-rag prepare` を実行してください。")

    import pandas as pd

    frame = pd.read_parquet(target)
    return to_queries(frame.to_dict("records"), max_candidates=max_candidates)


def download(path: str | Path = DEFAULT_PATH) -> Path:
    import httpx

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", HF_URL, follow_redirects=True, timeout=600.0) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return target
