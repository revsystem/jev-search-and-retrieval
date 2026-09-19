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
