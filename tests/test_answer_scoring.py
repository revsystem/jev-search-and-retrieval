"""Scoring a generated answer against JQaRA's gold answer.

The rule follows AI王's own normalisation and llm-jp-eval's tagged-span
extraction, because scoring the whole response inflates in ways that differ by
route. Verified on the test split: 3 of 1,667 questions contain the gold
answer in the question itself, and 130 gold answers are one or two characters
after normalisation.
"""

from jev_rag.answers import extract_answer, normalise_answer, score_answer


def test_the_tagged_span_is_what_gets_scored():
    assert extract_answer("考えました。<answer>絶対零度</answer>") == "絶対零度"


def test_a_response_without_the_tag_does_not_parse():
    assert extract_answer("それは絶対零度です。") is None


def test_an_untagged_response_is_wrong_and_recorded_as_unparsed():
    result = score_answer("それは絶対零度です。", ["絶対零度"])
    assert result.parsed is False
    assert result.correct is False


def test_an_exact_answer_is_correct():
    result = score_answer("<answer>絶対零度</answer>", ["絶対零度"])
    assert result.exact is True
    assert result.correct is True
    assert result.matched == "絶対零度"


def test_the_gold_answer_appearing_only_in_prose_is_not_credited():
    # 3 questions in the split name the answer in the question itself, so a
    # model echoing the question would otherwise score correct
    result = score_answer("音読みと訓読みのうち訓読みです<answer>音読み</answer>", ["訓読み"])
    assert result.correct is False


def test_a_hedged_span_is_not_an_exact_match_but_is_recorded_as_containing():
    result = score_answer("<answer>絶対零度または絶対温度</answer>", ["絶対零度"])
    assert result.exact is False
    assert result.contains is True


def test_width_and_case_are_normalised():
    assert score_answer("<answer>ＡＩ</answer>", ["AI"]).exact
    assert score_answer("<answer>ai</answer>", ["AI"]).exact


def test_quotes_and_separators_are_stripped():
    assert score_answer("<answer>「絶対零度」</answer>", ["絶対零度"]).exact
    assert score_answer("<answer>加藤・シゲアキ</answer>", ["加藤シゲアキ"]).exact


def test_surrounding_whitespace_is_ignored():
    assert score_answer("<answer>  絶対零度\n</answer>", ["絶対零度"]).exact


def test_normalisation_is_idempotent():
    once = normalise_answer("　「Ａ・Ｉ」　")
    assert normalise_answer(once) == once


def test_a_wrong_answer_is_wrong():
    result = score_answer("<answer>摂氏零度</answer>", ["絶対零度"])
    assert result.exact is False
    assert result.contains is False


def test_any_of_several_gold_answers_counts():
    assert score_answer("<answer>天邪鬼</answer>", ["あまのじゃく", "天邪鬼"]).exact


def test_a_refusal_is_wrong():
    assert score_answer("<answer>不明</answer>", ["絶対零度"]).correct is False


def test_the_extracted_span_is_kept_for_auditing():
    result = score_answer("<answer>絶対零度</answer>", ["絶対零度"])
    assert result.extracted == "絶対零度"


def test_an_empty_generation_is_wrong_rather_than_an_error():
    assert score_answer("", ["絶対零度"]).correct is False
