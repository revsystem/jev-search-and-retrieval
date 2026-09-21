"""The documented setup is `cp .env.example .env`, so .env has to be read."""

import os

import pytest

from jev_rag.settings_file import load_env_file


def test_values_from_the_file_reach_the_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("AI_GATEWAY_API_KEY=vck_secret\nJEV_TRANSPORT=gateway\n", encoding="utf-8")
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)

    assert load_env_file(env) is True
    assert os.environ["AI_GATEWAY_API_KEY"] == "vck_secret"


def test_the_real_environment_wins_over_the_file(tmp_path, monkeypatch):
    # an exported variable is the more deliberate of the two, and it keeps a
    # stale .env from overriding a one-off run
    env = tmp_path / ".env"
    env.write_text("AI_GATEWAY_API_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "from_shell")

    load_env_file(env)
    assert os.environ["AI_GATEWAY_API_KEY"] == "from_shell"


def test_a_missing_file_is_not_an_error(tmp_path):
    assert load_env_file(tmp_path / "absent") is False


def test_comments_and_blank_lines_are_ignored(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nJEV_TRANSPORT=fake\n", encoding="utf-8")
    monkeypatch.delenv("JEV_TRANSPORT", raising=False)

    assert load_env_file(env) is True
    assert os.environ["JEV_TRANSPORT"] == "fake"


def test_quotes_around_a_value_are_stripped(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('AI_GATEWAY_API_KEY="vck_quoted"\n', encoding="utf-8")
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)

    load_env_file(env)
    assert os.environ["AI_GATEWAY_API_KEY"] == "vck_quoted"


def test_a_malformed_line_is_reported_rather_than_silently_skipped(tmp_path):
    # a key with no value is how a half-pasted secret looks, and running with
    # the default instead of the intended value wastes a whole measurement
    env = tmp_path / ".env"
    env.write_text("AI_GATEWAY_API_KEY\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="1:"):
        load_env_file(env)


def test_the_cli_reads_the_file_before_settings_are_built(monkeypatch, tmp_path):
    from jev_rag import cli

    env = tmp_path / ".env"
    env.write_text("JEV_TRANSPORT=fake\n", encoding="utf-8")
    monkeypatch.setattr(cli, "ENV_FILE", env)
    monkeypatch.delenv("JEV_TRANSPORT", raising=False)

    cli.main(["usecases"])
    assert os.environ["JEV_TRANSPORT"] == "fake"
