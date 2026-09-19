import pytest

from jev_rag.cli import main


def test_demo_runs_without_credentials(capsys):
    assert main(["demo", "--top-k", "3"]) == 0
    out = capsys.readouterr().out
    assert "baseline" in out and "jev_rerank" in out


def test_demo_output_states_that_jev_answers_are_scripted(capsys):
    main(["demo"])
    assert "Jevの性能評価ではない" in capsys.readouterr().out


def test_check_works_against_the_offline_transport(capsys):
    assert main(["check", "--transport", "fake"]) == 0
    assert "noul=" in capsys.readouterr().out


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
