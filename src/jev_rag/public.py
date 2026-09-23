"""Public anchors for the judge evaluation: J-RAGBench and MIRACL Japanese dev.

J-RAGBench (neoai-inc/Japanese-RAG-Generator-Benchmark, CC BY-SA 4.0) pairs
business-flavoured questions about fictional companies with relevant chunks and
negatives written to share keywords or meaning without being evidence. Its 54
unanswerable variants repeat a question with the evidence removed; a pool with
no positive contributes nothing to PR-AUC, so they are left out.

MIRACL (miracl/miracl, Apache-2.0) is the one whose labels were observed: human
assessors judged passages that retrieval systems actually returned. The qrels
name passages by id; the text comes from miracl/miracl-corpus, which is large
enough (14 gzipped shards, about 1 GB for Japanese) that only the judged
passages are kept.
"""

from __future__ import annotations

import csv
import gzip
import json
import random
import urllib.request
from collections.abc import Iterable
from pathlib import Path

from jev_rag.dataset import EvalQuery, passage_text
from jev_rag.types import RetrievedDoc

JRAGBENCH_URL = (
    "https://huggingface.co/datasets/neoai-inc/Japanese-RAG-Generator-Benchmark"
    "/resolve/main/eval.jsonl"
)
MIRACL_BASE = "https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-ja"
MIRACL_QRELS_URL = f"{MIRACL_BASE}/qrels/qrels.miracl-v1.0-ja-dev.tsv"
MIRACL_TOPICS_URL = f"{MIRACL_BASE}/topics/topics.miracl-v1.0-ja-dev.tsv"
MIRACL_CORPUS_URL = (
    "https://huggingface.co/datasets/miracl/miracl-corpus/resolve/main"
    "/miracl-corpus-v1.0-ja/docs-{shard}.jsonl.gz"
)
MIRACL_SHARDS = 14
RAW = Path("data/raw")


def _shuffled(query_id: str, candidates: list[RetrievedDoc]) -> list[RetrievedDoc]:
    # the files list positives first; that order would hand every ranker a head start
    random.Random(query_id).shuffle(candidates)
    return candidates


def load_jragbench(path: str | Path) -> list[EvalQuery]:
    queries = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(filter(None, lines)):
        row = json.loads(line)
        if not row["positive"]:
            continue
        query_id = f"jr-{index:03d}"
        chunks = [(text, 1) for text in row["positive"]] + [(text, 0) for text in row["negative"]]
        candidates = [
            RetrievedDoc(doc_id=f"{query_id}-c{i:02d}", text=text, score=0.0, label=label)
            for i, (text, label) in enumerate(chunks)
        ]
        queries.append(
            EvalQuery(
                query_id=query_id,
                question=row["question"],
                candidates=_shuffled(query_id, candidates),
                answers=[row["answer"]] if row.get("answer") else [],
            )
        )
    return queries


def _qrels(path: str | Path) -> list[tuple[str, str, int]]:
    with open(path, encoding="utf-8", newline="") as f:
        return [(q, docid, int(rel)) for q, _, docid, rel in csv.reader(f, delimiter="\t")]


def miracl_docids(qrels_path: str | Path) -> set[str]:
    return {docid for _, docid, _ in _qrels(qrels_path)}


def extract_passages(shards: Iterable[str | Path], wanted: set[str]) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["docid"] in wanted:
                    found[row["docid"]] = {"title": row["title"], "text": row["text"]}
    return found


def load_miracl(
    qrels_path: str | Path, topics_path: str | Path, passages: dict[str, dict[str, str]]
) -> list[EvalQuery]:
    with open(topics_path, encoding="utf-8", newline="") as f:
        topics = dict(csv.reader(f, delimiter="\t"))
    judged = _qrels(qrels_path)
    missing = sorted({docid for _, docid, _ in judged} - set(passages))
    if missing:
        raise ValueError(
            f"コーパスに本文がない判定済みパッセージ: {missing[:10]} ほか計{len(missing)}件"
        )
    grouped: dict[str, list[RetrievedDoc]] = {}
    for q, docid, rel in judged:
        p = passages[docid]
        grouped.setdefault(q, []).append(
            RetrievedDoc(
                doc_id=docid,
                text=passage_text(p["title"], p["text"]),
                score=0.0,
                label=int(rel > 0),
                title=p["title"],
            )
        )
    return [
        EvalQuery(query_id=f"mi-{q}", question=topics[q], candidates=_shuffled(f"mi-{q}", docs))
        for q, docs in grouped.items()
    ]


def _fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)
    return path


def prepare_public(raw: Path = RAW) -> dict[str, Path]:
    """Download both datasets and keep only the MIRACL passages that were judged."""
    jragbench = _fetch(JRAGBENCH_URL, raw / "jragbench_eval.jsonl")
    qrels = _fetch(MIRACL_QRELS_URL, raw / "miracl_ja_dev_qrels.tsv")
    topics = _fetch(MIRACL_TOPICS_URL, raw / "miracl_ja_dev_topics.tsv")
    passages_path = raw / "miracl_ja_dev_passages.json"
    if not passages_path.exists():
        shards = [
            _fetch(
                MIRACL_CORPUS_URL.format(shard=i), raw / "miracl-corpus-ja" / f"docs-{i}.jsonl.gz"
            )
            for i in range(MIRACL_SHARDS)
        ]
        passages = extract_passages(shards, miracl_docids(qrels))
        passages_path.write_text(json.dumps(passages, ensure_ascii=False), encoding="utf-8")
    return {"jragbench": jragbench, "qrels": qrels, "topics": topics, "passages": passages_path}


def load_public(raw: Path = RAW) -> dict[str, list[EvalQuery]]:
    passages = json.loads((raw / "miracl_ja_dev_passages.json").read_text(encoding="utf-8"))
    return {
        "jragbench": load_jragbench(raw / "jragbench_eval.jsonl"),
        "miracl": load_miracl(
            raw / "miracl_ja_dev_qrels.tsv", raw / "miracl_ja_dev_topics.tsv", passages
        ),
    }
