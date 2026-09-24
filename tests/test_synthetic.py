"""The synthetic judge dataset: loading, verification views, and the gate.

The spec is .claude/docs/specs/synthetic-judge-dataset.md. Labels come from
construction: a document is relevant when the generator wrote a required fact
into it. Nothing here asks a model whether a document is relevant.
"""

import json

import pytest

from jev_rag.synthetic import (
    baseline_scores,
    compare_verification,
    load_synthetic,
    sanity_report,
    terms,
    with_distractors,
    write_views,
)


def item(qid="a-001"):
    return {
        "id": qid,
        "domain": "人事・総務",
        "question_type": "comparison",
        "question": "北辰精機の2024年度と2023年度で、在宅勤務手当の支給条件はどう変わったか",
        "required_facts": [
            {"id": "f1", "fact": "2023年度は週3日以上の在宅で月5,000円"},
            {"id": "f2", "fact": "2024年度は週2日以上の在宅で月7,000円"},
        ],
        "documents": [
            {
                "doc_id": f"{qid}-d01",
                "role": "positive",
                "contains_facts": ["f1"],
                "text": (
                    "北辰精機の2023年度規程では、在宅勤務が週3日以上の社員に月5,000円を支給する。"
                ),
            },
            {
                "doc_id": f"{qid}-d02",
                "role": "negative",
                "contains_facts": [],
                "near_miss": "同じ制度の一般論",
                "text": (
                    "北辰精機では在宅勤務手当の見直しを検討しており、各部署の意見を集めている。"
                ),
            },
            {
                "doc_id": f"{qid}-d03",
                "role": "positive",
                "contains_facts": ["f2"],
                "text": (
                    "北辰精機は2024年度から、在宅勤務が週2日以上の社員に月7,000円を支給する。"
                ),
            },
        ],
        "reference_answer": (
            "2023年度は週3日以上で月5,000円、2024年度は週2日以上で月7,000円に変わった。"
        ),
    }


def write(directory, *items):
    directory.mkdir(parents=True, exist_ok=True)
    for entry in items:
        (directory / f"{entry['id']}.json").write_text(
            json.dumps(entry, ensure_ascii=False), encoding="utf-8"
        )


# --- loading ---------------------------------------------------------------


def test_labels_come_from_the_construction_record(tmp_path):
    write(tmp_path, item())
    (query,) = load_synthetic(tmp_path)
    labels = {c.doc_id: c.label for c in query.candidates}
    assert labels == {"a-001-d01": 1, "a-001-d02": 0, "a-001-d03": 1}


def test_a_document_carrying_one_required_fact_is_relevant(tmp_path):
    # evidence worth retrieving even though it cannot answer on its own
    write(tmp_path, item())
    (query,) = load_synthetic(tmp_path)
    one_fact = next(c for c in query.candidates if c.doc_id == "a-001-d01")
    assert one_fact.label == 1


def test_the_question_and_type_are_kept(tmp_path):
    write(tmp_path, item())
    (query,) = load_synthetic(tmp_path)
    assert query.question.startswith("北辰精機")
    assert query.query_id == "a-001"


def test_a_malformed_file_is_reported_by_name(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "a-009.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="a-009"):
        load_synthetic(tmp_path)


def test_a_positive_without_facts_is_rejected(tmp_path):
    bad = item()
    bad["documents"][0]["contains_facts"] = []
    write(tmp_path, bad)
    with pytest.raises(ValueError, match="a-001-d01"):
        load_synthetic(tmp_path)


# --- verification views ----------------------------------------------------


def test_views_strip_every_construction_field(tmp_path):
    write(tmp_path / "gen", item())
    write_views(tmp_path / "gen", tmp_path / "views", seed=0)
    view = json.loads((tmp_path / "views" / "a-001.json").read_text(encoding="utf-8"))
    for doc in view["documents"]:
        assert set(doc) == {"doc_id", "text"}
    assert "reference_answer" not in view


def test_views_keep_what_the_checks_need(tmp_path):
    write(tmp_path / "gen", item())
    write_views(tmp_path / "gen", tmp_path / "views", seed=0)
    view = json.loads((tmp_path / "views" / "a-001.json").read_text(encoding="utf-8"))
    assert view["question"].startswith("北辰精機")
    assert [f["id"] for f in view["required_facts"]] == ["f1", "f2"]


def test_views_do_not_overwrite_an_existing_view(tmp_path):
    write(tmp_path / "gen", item())
    (tmp_path / "views").mkdir()
    (tmp_path / "views" / "a-001.json").write_text("{}", encoding="utf-8")
    written = write_views(tmp_path / "gen", tmp_path / "views", seed=0)
    assert written == []


def test_view_order_does_not_follow_the_construction_order(tmp_path):
    many = item()
    many["documents"] = [
        {"doc_id": f"a-001-d{i:02d}", "role": "negative", "contains_facts": [], "text": f"本文{i}"}
        for i in range(10)
    ]
    many["documents"][0]["role"] = "positive"
    many["documents"][0]["contains_facts"] = ["f1"]
    write(tmp_path / "gen", many)
    write_views(tmp_path / "gen", tmp_path / "views", seed=0)
    view = json.loads((tmp_path / "views" / "a-001.json").read_text(encoding="utf-8"))
    assert [d["doc_id"] for d in view["documents"]] != [d["doc_id"] for d in many["documents"]]


# --- comparing construction with verification -----------------------------


def verdict(doc_id, facts, leak=False):
    return {"doc_id": doc_id, "facts_stated": facts, "leak_phrase": leak, "note": ""}


def test_agreement_produces_no_findings(tmp_path):
    write(tmp_path / "gen", item())
    write(
        tmp_path / "verify",
        {
            "id": "a-001",
            "verdicts": [
                verdict("a-001-d01", ["f1"]),
                verdict("a-001-d02", []),
                verdict("a-001-d03", ["f2"]),
            ],
            "question_checks": {
                "needs_multiple_docs": True,
                "non_factoid": True,
                "answerable_from_facts": True,
            },
        },
    )
    report = compare_verification(tmp_path / "gen", tmp_path / "verify")
    assert report["findings"] == []
    assert report["checked_questions"] == 1


def test_a_negative_found_to_state_a_fact_is_flagged(tmp_path):
    write(tmp_path / "gen", item())
    write(
        tmp_path / "verify",
        {
            "id": "a-001",
            "verdicts": [
                verdict("a-001-d01", ["f1"]),
                verdict("a-001-d02", ["f2"]),
                verdict("a-001-d03", ["f2"]),
            ],
            "question_checks": {
                "needs_multiple_docs": True,
                "non_factoid": True,
                "answerable_from_facts": True,
            },
        },
    )
    (finding,) = compare_verification(tmp_path / "gen", tmp_path / "verify")["findings"]
    assert finding["doc_id"] == "a-001-d02"
    assert finding["kind"] == "fact_mismatch"


def test_a_leak_phrase_and_a_failed_question_check_are_both_flagged(tmp_path):
    write(tmp_path / "gen", item())
    write(
        tmp_path / "verify",
        {
            "id": "a-001",
            "verdicts": [
                verdict("a-001-d01", ["f1"], leak=True),
                verdict("a-001-d02", []),
                verdict("a-001-d03", ["f2"]),
            ],
            "question_checks": {
                "needs_multiple_docs": True,
                "non_factoid": False,
                "answerable_from_facts": True,
            },
        },
    )
    kinds = {
        f["kind"] for f in compare_verification(tmp_path / "gen", tmp_path / "verify")["findings"]
    }
    assert kinds == {"leak_phrase", "non_factoid"}


def test_questions_not_yet_verified_are_counted_separately(tmp_path):
    write(tmp_path / "gen", item("a-001"), item("a-002"))
    write(tmp_path / "verify", {"id": "a-001", "verdicts": [], "question_checks": {}})
    report = compare_verification(tmp_path / "gen", tmp_path / "verify")
    assert report["unverified"] == ["a-002"]


# --- the gate before the paid run -------------------------------------------


def test_terms_pick_out_kanji_katakana_and_alphanumeric_runs():
    assert {"北辰精機", "在宅勤務手当", "2024"} <= terms("北辰精機の在宅勤務手当は2024年度に")


def test_sanity_reports_length_and_overlap_by_label(tmp_path):
    write(tmp_path, item())
    report = sanity_report(load_synthetic(tmp_path))
    assert set(report["length"]) == {"positive", "negative"}
    assert set(report["term_overlap"]) == {"positive", "negative"}


def test_baselines_score_every_pair(tmp_path):
    write(tmp_path, item())
    scores = baseline_scores(load_synthetic(tmp_path))
    assert set(scores) == {"length", "term_overlap", "bm25"}
    assert all(len(rows) == 3 for rows in scores.values())


def test_bm25_prefers_the_document_sharing_rare_terms(tmp_path):
    write(tmp_path, item())
    rows = baseline_scores(load_synthetic(tmp_path))["bm25"]
    by_doc = {r["doc_id"]: r["score"] for r in rows}
    # d03 shares 2024年度 and 週2日; the general-policy negative shares little
    assert by_doc["a-001-d03"] > by_doc["a-001-d02"]


# --- padding a pool with documents from other questions ---------------------


def test_distractors_pad_each_pool_to_the_requested_size(tmp_path):
    write(tmp_path, item("a-001"), item("a-002"), item("a-003"))
    queries = with_distractors(load_synthetic(tmp_path), total=7, seed=0)
    assert all(len(q.candidates) == 7 for q in queries)


def test_distractors_come_from_other_questions_and_are_negative(tmp_path):
    write(tmp_path, item("a-001"), item("a-002"), item("a-003"))
    (first, *_) = with_distractors(load_synthetic(tmp_path), total=7, seed=0)
    added = [c for c in first.candidates if not c.doc_id.startswith(first.query_id)]
    assert len(added) == 4
    assert all(c.label == 0 for c in added)


def test_the_original_documents_are_all_kept(tmp_path):
    write(tmp_path, item("a-001"), item("a-002"))
    (first, _) = with_distractors(load_synthetic(tmp_path), total=5, seed=0)
    own = {c.doc_id for c in first.candidates if c.doc_id.startswith(first.query_id)}
    assert own == {"a-001-d01", "a-001-d02", "a-001-d03"}


def test_padding_does_not_touch_the_source_queries(tmp_path):
    write(tmp_path, item("a-001"), item("a-002"))
    source = load_synthetic(tmp_path)
    with_distractors(source, total=5, seed=0)
    assert all(len(q.candidates) == 3 for q in source)
