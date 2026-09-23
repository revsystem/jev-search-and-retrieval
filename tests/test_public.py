"""Public anchors for the judge evaluation: J-RAGBench and MIRACL Japanese dev.

Both are used as published. The only thing dropped is what cannot be scored:
J-RAGBench questions with no relevant chunk (the unanswerable variants).
"""

import gzip
import json

import pytest

from jev_rag.public import extract_passages, load_jragbench, load_miracl, miracl_docids


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_jragbench_labels_positive_and_negative_chunks(tmp_path):
    path = tmp_path / "eval.jsonl"
    write_jsonl(
        path, [{"question": "Q1", "answer": "A", "positive": ["p1", "p2"], "negative": ["n1"]}]
    )
    (query,) = load_jragbench(path)
    assert sorted((c.text, c.label) for c in query.candidates) == [("n1", 0), ("p1", 1), ("p2", 1)]
    assert query.answers == ["A"]


def test_jragbench_drops_questions_without_a_relevant_chunk(tmp_path):
    # the unanswerable variants repeat an answerable question with its evidence removed;
    # a pool with no positive has no PR-AUC to contribute
    path = tmp_path / "eval.jsonl"
    write_jsonl(
        path,
        [
            {"question": "Q1", "answer": "A", "positive": ["p1"], "negative": ["n1"]},
            {"question": "Q1", "answer": "", "positive": [], "negative": ["n1", "n2"]},
        ],
    )
    assert [q.query_id for q in load_jragbench(path)] == ["jr-000"]


def test_jragbench_doc_ids_are_unique_within_a_query(tmp_path):
    path = tmp_path / "eval.jsonl"
    write_jsonl(path, [{"question": "Q", "answer": "A", "positive": ["x"], "negative": ["x", "y"]}])
    (query,) = load_jragbench(path)
    assert len({c.doc_id for c in query.candidates}) == 3


def miracl_files(tmp_path):
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("0\tQ0\t10#1\t1\n0\tQ0\t10#0\t0\n3\tQ0\t20#0\t1\n", encoding="utf-8")
    topics = tmp_path / "topics.tsv"
    topics.write_text("0\t出身はどこ\n3\t初代の司会は誰\n", encoding="utf-8")
    return qrels, topics


def test_miracl_docids_are_every_judged_passage(tmp_path):
    qrels, _ = miracl_files(tmp_path)
    assert miracl_docids(qrels) == {"10#1", "10#0", "20#0"}


def test_passages_are_extracted_from_gzipped_shards(tmp_path):
    shard = tmp_path / "docs-0.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as f:
        for docid in ("10#0", "10#1", "99#0"):
            f.write(json.dumps({"docid": docid, "title": "T", "text": f"本文{docid}"}) + "\n")
    passages = extract_passages([shard], {"10#0", "10#1"})
    assert set(passages) == {"10#0", "10#1"}
    assert passages["10#1"] == {"title": "T", "text": "本文10#1"}


def test_miracl_queries_carry_human_labels_and_titled_text(tmp_path):
    qrels, topics = miracl_files(tmp_path)
    passages = {
        "10#1": {"title": "人物", "text": "ボストン生まれ"},
        "10#0": {"title": "人物", "text": "経歴"},
        "20#0": {"title": "番組", "text": "初代司会"},
    }
    queries = {q.query_id: q for q in load_miracl(qrels, topics, passages)}
    assert queries["mi-0"].question == "出身はどこ"
    labels = {c.doc_id: c.label for c in queries["mi-0"].candidates}
    assert labels == {"10#1": 1, "10#0": 0}
    (only,) = queries["mi-3"].candidates
    assert only.text == "番組 初代司会"


def test_miracl_reports_passages_missing_from_the_corpus(tmp_path):
    qrels, topics = miracl_files(tmp_path)
    passages = {"10#1": {"title": "", "text": "a"}, "10#0": {"title": "", "text": "b"}}
    with pytest.raises(ValueError, match="20#0"):
        load_miracl(qrels, topics, passages)
