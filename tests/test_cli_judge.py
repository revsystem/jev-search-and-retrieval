"""The judge-evaluation commands: preparing the data and checking the synthetic set."""

from tests.test_synthetic import item, write

from jev_rag import cli


def verdicts(entry, fact_for=None):
    fact_for = fact_for or {}
    return {
        "id": entry["id"],
        "verdicts": [
            {
                "doc_id": d["doc_id"],
                "facts_stated": fact_for.get(d["doc_id"], d["contains_facts"]),
                "leak_phrase": False,
            }
            for d in entry["documents"]
        ],
        "question_checks": {
            "needs_multiple_docs": True,
            "non_factoid": True,
            "answerable_from_facts": True,
        },
    }


def test_synthetic_check_passes_when_verification_agrees(tmp_path, capsys):
    entry = item()
    write(tmp_path / "gen", entry)
    write(tmp_path / "verify", verdicts(entry))
    code = cli.main(
        ["synthetic-check", "--gen", str(tmp_path / "gen"), "--verify", str(tmp_path / "verify")]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "bm25" in out and "食い違い 0 件" in out


def test_synthetic_check_fails_on_a_disagreement(tmp_path, capsys):
    entry = item()
    write(tmp_path / "gen", entry)
    write(tmp_path / "verify", verdicts(entry, {"a-001-d02": ["f2"]}))
    code = cli.main(
        ["synthetic-check", "--gen", str(tmp_path / "gen"), "--verify", str(tmp_path / "verify")]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "a-001-d02" in out


def test_synthetic_check_fails_while_questions_are_unverified(tmp_path, capsys):
    write(tmp_path / "gen", item("a-001"), item("a-002"))
    write(tmp_path / "verify", verdicts(item("a-001")))
    code = cli.main(
        ["synthetic-check", "--gen", str(tmp_path / "gen"), "--verify", str(tmp_path / "verify")]
    )
    assert code == 1
    assert "a-002" in capsys.readouterr().out


def test_prepare_judge_downloads_both_public_sets(monkeypatch, tmp_path, capsys):
    from jev_rag import public

    called = {}
    monkeypatch.setattr(public, "prepare_public", lambda raw: called.setdefault("raw", raw) or {})
    monkeypatch.setattr(public, "load_public", lambda raw: {"jragbench": [], "miracl": []})
    assert cli.main(["prepare-judge", "--raw", str(tmp_path)]) == 0
    assert str(called["raw"]) == str(tmp_path)
    assert "MIRACL" in capsys.readouterr().out
