"""The synthetic judge dataset: loading, verification views, and the gate.

The spec is .claude/docs/specs/synthetic-judge-dataset.md. Labels come from
construction: a document is relevant when its generator wrote one of the
question's required facts into it. Nothing in this module asks a model whether
a document is relevant, because that is the very judgement being evaluated.

Three things happen here between generation and the paid measurement:

- ``write_views`` strips the construction record and shuffles the documents,
  so a verifier cannot see which were meant to be relevant.
- ``compare_verification`` lists every place a verifier's spec-compliance
  check disagrees with the construction record.
- ``baseline_scores`` scores every pair with length, term overlap and BM25.
  If a trivial scorer separates the classes, the construction has leaked a
  cue and nothing measured on top of it means anything.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from jev_rag.dataset import EvalQuery
from jev_rag.types import RetrievedDoc

CONSTRUCTION_FIELDS = {"role", "contains_facts", "near_miss"}
QUESTION_CHECKS = ("needs_multiple_docs", "non_factoid", "answerable_from_facts")
_TERM = re.compile(r"[一-鿿]{2,}|[゠-ヿ]{2,}|[A-Za-z0-9]{2,}")


def terms(text: str) -> set[str]:
    """Kanji, katakana and alphanumeric runs: the content words of a Japanese text."""
    return set(_TERM.findall(unicodedata.normalize("NFKC", text)))


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.name}: JSON として読めない: {error}") from error


def load_synthetic(directory: str | Path) -> list[EvalQuery]:
    """Read one generator's output into evaluation queries."""
    queries = []
    for path in sorted(Path(directory).glob("*.json")):
        item = _read(path)
        candidates = []
        for doc in item["documents"]:
            positive = doc["role"] == "positive"
            if positive != bool(doc.get("contains_facts")):
                raise ValueError(
                    f"{doc['doc_id']}: role={doc['role']} と contains_facts が矛盾している"
                )
            candidates.append(
                RetrievedDoc(doc_id=doc["doc_id"], text=doc["text"], score=0.0, label=int(positive))
            )
        # Same treatment as the public data: an input order that follows the
        # construction would hand every ranker a head start.
        candidates.sort(key=lambda d: d.doc_id)
        random.Random(f"{item['id']}:{len(candidates)}").shuffle(candidates)
        queries.append(
            EvalQuery(
                query_id=item["id"],
                question=item["question"],
                candidates=candidates,
                category=item.get("question_type", ""),
            )
        )
    return queries


def write_views(source: str | Path, target: str | Path, seed: int = 0) -> list[Path]:
    """Verification views: the construction record removed, the order shuffled."""
    out = Path(target)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for path in sorted(Path(source).glob("*.json")):
        destination = out / path.name
        if destination.exists():
            continue
        item = _read(path)
        documents = [
            {k: v for k, v in doc.items() if k not in CONSTRUCTION_FIELDS}
            for doc in item["documents"]
        ]
        random.Random(f"{seed}:{item['id']}").shuffle(documents)
        view = {
            "id": item["id"],
            "question": item["question"],
            "required_facts": item["required_facts"],
            "documents": documents,
        }
        destination.write_text(json.dumps(view, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(destination)
    return written


def compare_verification(generated: str | Path, verified: str | Path) -> dict[str, Any]:
    """Every disagreement between the construction record and the compliance check."""
    findings: list[dict[str, Any]] = []
    unverified: list[str] = []
    checked = 0
    for path in sorted(Path(generated).glob("*.json")):
        item = _read(path)
        verdict_path = Path(verified) / path.name
        if not verdict_path.exists():
            unverified.append(item["id"])
            continue
        checked += 1
        report = _read(verdict_path)
        by_doc = {v["doc_id"]: v for v in report.get("verdicts", [])}
        for doc in item["documents"]:
            verdict = by_doc.get(doc["doc_id"])
            if verdict is None:
                findings.append(
                    {"id": item["id"], "doc_id": doc["doc_id"], "kind": "missing_verdict"}
                )
                continue
            expected = set(doc.get("contains_facts") or [])
            found = set(verdict.get("facts_stated") or [])
            if expected != found:
                findings.append(
                    {
                        "id": item["id"],
                        "doc_id": doc["doc_id"],
                        "kind": "fact_mismatch",
                        "role": doc["role"],
                        "expected": sorted(expected),
                        "found": sorted(found),
                        "note": verdict.get("note", ""),
                    }
                )
            if verdict.get("leak_phrase"):
                findings.append(
                    {
                        "id": item["id"],
                        "doc_id": doc["doc_id"],
                        "kind": "leak_phrase",
                        "note": verdict.get("note", ""),
                    }
                )
        checks = report.get("question_checks", {})
        for key in QUESTION_CHECKS:
            if checks.get(key) is False:
                findings.append({"id": item["id"], "kind": key, "note": checks.get("note", "")})
    return {"checked_questions": checked, "unverified": unverified, "findings": findings}


def _overlap(question: str, text: str) -> float:
    wanted = terms(question)
    return len(wanted & terms(text)) / len(wanted) if wanted else 0.0


def sanity_report(queries: list[EvalQuery]) -> dict[str, Any]:
    """Length and term overlap by label. They should look alike."""
    by_label: dict[str, dict[str, list[float]]] = {
        "positive": {"length": [], "overlap": []},
        "negative": {"length": [], "overlap": []},
    }
    for query in queries:
        for doc in query.candidates:
            bucket = by_label["positive" if doc.label else "negative"]
            bucket["length"].append(len(doc.text))
            bucket["overlap"].append(_overlap(query.question, doc.text))

    def summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }

    return {
        "length": {label: summary(v["length"]) for label, v in by_label.items()},
        "term_overlap": {label: summary(v["overlap"]) for label, v in by_label.items()},
    }


def _bm25(query: str, docs: list[RetrievedDoc], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 within one question's candidate pool, over content-word runs."""
    bags = [Counter(_TERM.findall(unicodedata.normalize("NFKC", d.text))) for d in docs]
    lengths = [sum(bag.values()) for bag in bags]
    average = statistics.fmean(lengths) if lengths else 0.0
    count = len(docs)
    scores = []
    for bag, length in zip(bags, lengths, strict=True):
        total = 0.0
        for term in terms(query):
            frequency = bag.get(term, 0)
            if not frequency:
                continue
            containing = sum(1 for other in bags if term in other)
            idf = math.log((count - containing + 0.5) / (containing + 0.5) + 1)
            norm = frequency + k1 * (1 - b + b * length / average) if average else frequency
            total += idf * frequency * (k1 + 1) / norm
        scores.append(total)
    return scores


def baseline_scores(queries: list[EvalQuery]) -> dict[str, list[dict[str, Any]]]:
    """Trivial scorers for the gate. A high PR-AUC from any of them is a leak."""
    out: dict[str, list[dict[str, Any]]] = {"length": [], "term_overlap": [], "bm25": []}
    for query in queries:
        docs = query.candidates
        bm25 = _bm25(query.question, docs)
        for doc, bm in zip(docs, bm25, strict=True):
            base = {"query_id": query.query_id, "doc_id": doc.doc_id, "label": doc.label}
            out["length"].append(base | {"score": float(len(doc.text))})
            out["term_overlap"].append(base | {"score": _overlap(query.question, doc.text)})
            out["bm25"].append(base | {"score": bm})
    return out
