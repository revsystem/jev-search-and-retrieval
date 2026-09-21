"""Loading and sampling JQaRA."""

import pytest

from jev_rag.dataset import sample_queries, to_queries


def row(q, pid, title, text, label):
    return {
        "q_id": q,
        "question": f"問{q[-1]}",
        "passage_row_id": pid,
        "title": title,
        "text": text,
        "label": label,
    }


def rows():
    return [
        row("q1", 1, "A", "あ", 1),
        row("q1", 2, "B", "い", 0),
        row("q2", 3, "C", "う", 1),
        row("q2", 4, "D", "え", 1),
    ]


def test_rows_are_grouped_into_one_query_per_question():
    queries = to_queries(rows())
    assert [q.query_id for q in queries] == ["q1", "q2"]
    assert queries[0].question == "問1"


def test_each_query_keeps_all_of_its_candidates():
    assert [len(q.candidates) for q in to_queries(rows())] == [2, 2]


def test_the_relevant_count_comes_from_the_labels():
    assert [q.total_relevant for q in to_queries(rows())] == [1, 2]


def test_candidates_carry_their_label_and_title():
    first = to_queries(rows())[0].candidates[0]
    assert first.label == 1
    assert first.title == "A"
    assert first.doc_id == "1"


def test_a_query_with_no_relevant_passage_is_dropped():
    only_negative = [row("q3", 9, "", "お", 0)]
    assert to_queries(only_negative) == []


def test_sampling_is_deterministic_for_a_seed():
    queries = to_queries(rows())
    assert sample_queries(queries, 1, seed=7) == sample_queries(queries, 1, seed=7)


def test_sampling_returns_the_requested_number():
    assert len(sample_queries(to_queries(rows()), 1, seed=0)) == 1


def test_asking_for_more_than_exists_returns_everything():
    assert len(sample_queries(to_queries(rows()), 99, seed=0)) == 2


def test_candidates_can_be_capped_per_query():
    queries = to_queries(rows(), max_candidates=1)
    assert all(len(q.candidates) == 1 for q in queries)


def test_capping_keeps_at_least_one_relevant_candidate():
    # the cap must not silently make a query unanswerable
    padded = rows() + [row("q1", i, "", "x", 0) for i in range(10, 30)]
    query = next(q for q in to_queries(padded, max_candidates=5) if q.query_id == "q1")
    assert query.total_relevant >= 1
    assert len(query.candidates) == 5


def test_an_eval_query_exposes_its_candidates_as_documents():
    query = to_queries(rows())[0]
    docs = query.as_documents()
    assert [d.doc_id for d in docs] == ["1", "2"]
    assert all(d.score == 0.0 for d in docs)


def test_loading_an_absent_file_says_how_to_get_it():
    from jev_rag.dataset import load_jqara

    with pytest.raises(SystemExit, match="prepare"):
        load_jqara("does/not/exist.parquet")


def test_the_candidate_order_does_not_leak_the_labels():
    # capping keeps the relevant passages, which would otherwise leave them at
    # the front of the list — a ranker that does nothing would then score
    # perfectly, and every real ranker would be flattered by the same artefact
    padded = [row("q1", 1, "", "hit", 1)] + [row("q1", i, "", "x", 0) for i in range(2, 40)]
    query = to_queries(padded, max_candidates=20)[0]
    assert query.candidates[0].label == 0


def test_the_shuffle_is_deterministic_for_a_query():
    padded = [row("q1", 1, "", "hit", 1)] + [row("q1", i, "", "x", 0) for i in range(2, 40)]
    first = [c.doc_id for c in to_queries(padded, max_candidates=20)[0].candidates]
    again = [c.doc_id for c in to_queries(padded, max_candidates=20)[0].candidates]
    assert first == again


def test_shuffling_keeps_every_candidate():
    padded = [row("q1", 1, "", "hit", 1)] + [row("q1", i, "", "x", 0) for i in range(2, 40)]
    query = to_queries(padded, max_candidates=20)[0]
    assert len(query.candidates) == 20
    assert query.total_relevant == 1


def test_the_passage_carries_its_title_like_the_published_evaluation():
    # JQaRA's evaluator concatenates "{title} {text}" by default, and the
    # published nDCG@10 figures come from that form
    from jev_rag.dataset import passage_text

    assert passage_text("絶対零度", "摂氏マイナス273.15度。") == "絶対零度 摂氏マイナス273.15度。"


def test_a_passage_without_a_title_is_left_alone():
    from jev_rag.dataset import passage_text

    assert passage_text("", "本文のみ。") == "本文のみ。"


def test_candidates_are_built_from_the_concatenated_passage():
    candidate = to_queries(rows())[0].candidates[0]
    assert candidate.text == "A あ"
    assert candidate.title == "A"
