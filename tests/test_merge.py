"""Merging result files.

A run can lose routes to an expired credential or a rejected request, and
re-running only those leaves the measurement split across files. The report
should still be able to show one table.
"""

import json

from jev_rag.runner import merge_results


def result(ndcg: float, queries: int = 100) -> dict:
    return {"metrics": {"ndcg@10": ndcg}, "stderr": {"ndcg@10": 0.02}, "queries": queries}


def test_routes_from_several_files_end_up_in_one_table():
    merged = merge_results([{"a": result(0.5)}, {"b": result(0.6)}])
    assert set(merged) == {"a", "b"}


def test_a_later_file_replaces_an_earlier_measurement_of_the_same_route():
    merged = merge_results([{"a": result(0.5)}, {"a": result(0.8)}])
    assert merged["a"]["metrics"]["ndcg@10"] == 0.8


def test_a_failed_route_does_not_replace_a_successful_one():
    # re-running is how a failure gets fixed, not how a success gets lost
    merged = merge_results([{"a": result(0.8)}, {"a": {"error": "token expired"}}])
    assert merged["a"]["metrics"]["ndcg@10"] == 0.8


def test_a_successful_route_replaces_an_earlier_failure():
    merged = merge_results([{"a": {"error": "token expired"}}, {"a": result(0.8)}])
    assert "error" not in merged["a"]


def test_run_metadata_is_kept_from_the_last_file_that_has_it():
    merged = merge_results([{"_run": {"queries": 10}}, {"_run": {"queries": 100}}])
    assert merged["_run"]["queries"] == 100


def test_routes_measured_over_different_query_counts_are_flagged():
    from jev_rag.runner import inconsistent_query_counts

    merged = merge_results([{"a": result(0.5, queries=100)}, {"b": result(0.6, queries=30)}])
    assert inconsistent_query_counts(merged) == {"a": 100, "b": 30}


def test_a_consistent_merge_is_not_flagged():
    from jev_rag.runner import inconsistent_query_counts

    merged = merge_results([{"a": result(0.5)}, {"b": result(0.6)}])
    assert inconsistent_query_counts(merged) == {}


def test_merging_reads_from_disk(tmp_path):
    from jev_rag.runner import merge_files

    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text(json.dumps({"a": result(0.5)}), encoding="utf-8")
    second.write_text(json.dumps({"b": result(0.6)}), encoding="utf-8")
    assert set(merge_files([first, second])) == {"a", "b"}
