import pytest

from jev_rag.cli import main


def test_demo_runs_without_credentials(capsys):
    assert main(["demo", "--top-k", "3"]) == 0
    out = capsys.readouterr().out
    assert "baseline" in out and "jev_noul" in out


def test_demo_output_states_that_jev_answers_are_scripted(capsys):
    main(["demo"])
    assert "Jevの性能評価ではない" in capsys.readouterr().out


def test_check_works_against_the_offline_transport(capsys):
    assert main(["check", "--transport", "fake"]) == 0
    assert "noul=" in capsys.readouterr().out


def test_every_use_case_has_a_pipeline_name():
    from jev_rag.cli import PIPELINE_NAMES

    assert {"jev_only", "jev_hybrid", "jev_noul", "jev_pairwise", "jev_crossencode"} <= set(
        PIPELINE_NAMES
    )


def test_the_demo_exercises_every_use_case(capsys):
    main(["demo"])
    out = capsys.readouterr().out
    for name in ("jev_only", "jev_hybrid", "jev_noul", "jev_pairwise", "jev_crossencode"):
        assert name in out
    assert "文脈選択" in out


def test_unknown_pipeline_name_is_rejected():
    from jev_rag.cli import _build_pipelines
    from jev_rag.config import Settings

    with pytest.raises(SystemExit):
        _build_pipelines(["nope"], Settings())


def test_evaluate_without_an_ingested_corpus_fails_with_a_hint():
    from jev_rag.cli import _load_chunks

    with pytest.raises(SystemExit, match="ingest"):
        _load_chunks()


def test_an_unknown_subcommand_is_rejected():
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_usecases_lists_every_published_use_case(capsys):
    from jev_rag.usecases import USE_CASES

    assert main(["usecases"]) == 0
    out = capsys.readouterr().out
    for use_case in USE_CASES:
        assert use_case.bullet in out


def test_usecases_verbose_adds_the_design_notes(capsys):
    main(["usecases", "--verbose"])
    assert "埋め込みを経路から外し" in capsys.readouterr().out


def _write_corpus(tmp_path, texts):
    import json

    from jev_rag import cli

    cache = tmp_path / "chunks.jsonl"
    cache.write_text(
        "\n".join(
            json.dumps(
                {"chunk_id": f"c{i}", "page": 1, "section": "", "text": t}, ensure_ascii=False
            )
            for i, t in enumerate(texts)
        ),
        encoding="utf-8",
    )
    cli.CHUNK_CACHE = cache
    return cache


def _write_queries(tmp_path, body):
    path = tmp_path / "queries.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_labels_flags_a_rule_that_matches_nothing(tmp_path, capsys, monkeypatch):
    from jev_rag import cli

    monkeypatch.setattr(cli, "CHUNK_CACHE", _write_corpus(tmp_path, ["無関係な本文"] * 5))
    queries = _write_queries(
        tmp_path,
        "queries:\n  - query_id: q1\n    question: q\n    relevant:\n"
        '      all: ["存在しない語"]\n      grade: 3\n',
    )
    main(["labels", "--queries", str(queries)])
    out = capsys.readouterr().out
    assert "q1" in out and "該当なし" in out and "要調整" in out


def test_labels_flags_a_rule_that_matches_almost_the_whole_corpus(tmp_path, capsys, monkeypatch):
    from jev_rag import cli

    monkeypatch.setattr(cli, "CHUNK_CACHE", _write_corpus(tmp_path, ["生成AIの話"] * 10))
    queries = _write_queries(
        tmp_path,
        "queries:\n  - query_id: q1\n    question: q\n    relevant:\n"
        '      all: ["生成AI"]\n      grade: 3\n',
    )
    main(["labels", "--queries", str(queries)])
    assert "過剰マッチ" in capsys.readouterr().out


def test_labels_warns_when_no_query_produces_a_partial_grade(tmp_path, capsys, monkeypatch):
    from jev_rag import cli

    monkeypatch.setattr(cli, "CHUNK_CACHE", _write_corpus(tmp_path, ["生成AIの利用率は26.7%"]))
    queries = _write_queries(
        tmp_path,
        "queries:\n  - query_id: q1\n    question: q\n    relevant:\n"
        '      all: ["生成AI"]\n      grade: 3\n',
    )
    main(["labels", "--queries", str(queries)])
    assert "grade 2" in capsys.readouterr().out


def test_labels_is_quiet_about_a_well_calibrated_query(tmp_path, capsys, monkeypatch):
    from jev_rag import cli

    corpus = ["生成AIの利用率は26.7%", "生成AIの利用は拡大", *["無関係な本文"] * 8]
    monkeypatch.setattr(cli, "CHUNK_CACHE", _write_corpus(tmp_path, corpus))
    queries = _write_queries(
        tmp_path,
        "queries:\n  - query_id: q1\n    question: q\n    rules:\n"
        '      - {all: ["生成AI", "利用"], regex: "[0-9]+(\\\\.[0-9]+)?%", grade: 3}\n'
        '      - {all: ["生成AI", "利用"], grade: 2}\n',
    )
    main(["labels", "--queries", str(queries)])
    out = capsys.readouterr().out
    assert "要調整" not in out and "過剰マッチ" not in out
