"""Scoring a generated answer against JQaRA's gold answer.

JQaRA carries one gold answer string per question, inherited from JAQKET —
a quiz dataset, so the answers are short proper nouns like 絶対零度. A
generated answer is counted correct when it contains the gold string after
normalisation.

The contestable part is the normalisation, and every number in the end-to-end
comparison rests on it, so the rule is kept in one place with its failure
modes written down.
"""

from jev_rag.answers import contains_answer, normalise_answer, score_answer


def test_identical_answers_match():
    assert contains_answer("絶対零度", ["絶対零度"])


def test_an_answer_embedded_in_a_sentence_matches():
    # the model is asked for the term alone but often writes a sentence
    assert contains_answer("それは絶対零度です。", ["絶対零度"])


def test_an_unrelated_answer_does_not_match():
    assert not contains_answer("摂氏零度", ["絶対零度"])


def test_any_of_several_gold_answers_counts():
    assert contains_answer("天邪鬼だ", ["あまのじゃく", "天邪鬼"])


def test_width_variants_are_normalised():
    assert contains_answer("答えはＡＩです", ["AI"])
    assert contains_answer("１０社", ["10社"])


def test_spaces_are_ignored():
    assert contains_answer("加藤 シゲアキ", ["加藤シゲアキ"])


def test_case_is_ignored():
    assert contains_answer("the answer is ai", ["AI"])


def test_surrounding_punctuation_is_ignored():
    assert contains_answer("「絶対零度」", ["絶対零度"])


def test_normalisation_is_idempotent():
    once = normalise_answer("　Ａ Ｉ　")
    assert normalise_answer(once) == once


def test_an_empty_generation_is_wrong_rather_than_an_error():
    assert not contains_answer("", ["絶対零度"])


def test_a_refusal_is_wrong():
    assert not contains_answer("提供された資料からは判断できません", ["絶対零度"])


def test_scoring_reports_the_matched_answer_for_auditing():
    result = score_answer("それは絶対零度です。", ["絶対零度"])
    assert result.correct is True
    assert result.matched == "絶対零度"


def test_scoring_a_miss_records_no_match():
    result = score_answer("わかりません", ["絶対零度"])
    assert result.correct is False
    assert result.matched is None


def test_a_gold_answer_that_is_a_substring_of_a_longer_wrong_word_still_matches():
    # containment is deliberately generous: JQaRA ships one answer per question
    # and a stricter rule would mark correct-but-differently-worded answers
    # wrong far more often than it catches this case
    assert contains_answer("絶対零度計", ["絶対零度"])
