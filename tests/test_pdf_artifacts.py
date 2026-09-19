"""Normalisation of the artefacts a real government-whitepaper PDF produces.

Every case here was taken from the extraction of
soumu.go.jp/.../r08/pdf/n1110000.pdf (第1章 個人におけるAI利用, 15 pages).
"""

from jev_rag.documents import chunk_pages, detect_boilerplate, normalise_japanese_text


def test_hair_spaces_are_removed():
    assert normalise_japanese_text("AI 利用    個人") == "AI利用個人"


def test_a_letter_spaced_heading_is_rejoined():
    # 個 人 に お け る A I 利 用  -- decorative tracking, one space per character
    assert normalise_japanese_text("個 人 に お け る A I 利 用") == "個人におけるAI利用"


def test_a_space_beside_japanese_is_typographic_and_removed():
    # "生成 AI サービス" must match a rule looking for 生成AI
    assert normalise_japanese_text("生成 AI サービスの利用経験") == "生成AIサービスの利用経験"
    assert normalise_japanese_text("第 1章") == "第1章"
    assert normalise_japanese_text("令和 7 年版") == "令和7年版"


def test_a_space_between_two_latin_words_is_kept():
    assert normalise_japanese_text("Chat GPT, Gemini") == "Chat GPT, Gemini"


def test_a_line_rendered_twice_for_emphasis_is_collapsed():
    assert normalise_japanese_text("26.726.7") == "26.7"
    assert normalise_japanese_text("個人における AI 利用の現状個人における AI 利用の現状") == (
        "個人におけるAI利用の現状"
    )


def test_a_duplicated_tail_is_collapsed_but_the_numbering_is_kept():
    raw = "1 1\t 個人における生成 AI サービスの利用経験\t 個人における生成 AI サービスの利用経験"
    # the space before 個 is typographic and goes with the others
    assert normalise_japanese_text(raw) == "1 1個人における生成AIサービスの利用経験"


def test_a_short_repetition_is_not_treated_as_duplication():
    # 様々 and 時々 are ordinary Japanese, not a rendering artefact
    assert normalise_japanese_text("様々") == "様々"
    assert normalise_japanese_text("人々が時々") == "人々が時々"


def test_a_mojibake_line_from_a_broken_font_is_dropped():
    raw = "本章では調査結果を概説する。\nୈ 1ਓʹ͓͚Δ AIঢ়\n分析を試みる。"
    assert normalise_japanese_text(raw) == "本章では調査結果を概説する。分析を試みる。"


def test_running_headers_repeated_across_pages_are_detected():
    pages = {n: f"本文{n}である。\n第1節 個人におけるAI利用の現状" for n in range(1, 6)}
    # the typographic space beside 節 is removed before the line is compared
    assert "第1節個人におけるAI利用の現状" in detect_boilerplate(pages)


def test_a_line_on_a_single_page_is_not_boilerplate():
    pages = {1: "固有の本文", 2: "別の本文", 3: "三つ目"}
    assert detect_boilerplate(pages) == frozenset()


def test_boilerplate_is_stripped_from_the_chunk_text():
    footer = "第第 11 節節個人における AI 利用の現状個人における AI 利用の現状"
    pages = {n: f"{n}ページ目の固有の本文である。\n{footer}" for n in range(1, 6)}
    chunks = chunk_pages(pages, chunk_size=700, overlap=120)
    assert all("利用の現状" not in c.text for c in chunks)


def test_a_repeated_heading_footer_becomes_the_section_of_its_page():
    footer = "第第 11 節節個人における AI 利用の現状個人における AI 利用の現状"
    pages = {n: f"{n}ページ目の固有の本文である。\n{footer}" for n in range(1, 6)}
    chunks = chunk_pages(pages, chunk_size=700, overlap=120)
    assert {c.section for c in chunks} == {"第1節 個人におけるAI利用の現状"}


def test_a_chapter_heading_on_the_page_is_still_used_when_there_is_no_footer():
    chunks = chunk_pages({1: "第1章\n本文である。"}, chunk_size=700, overlap=120)
    assert chunks[0].section == "第1章"


def test_chart_axis_noise_repeated_across_pages_is_dropped():
    pages = {n: f"{n}ページ目の本文。\n0 20 40 60 80 100 (%)" for n in range(1, 6)}
    chunks = chunk_pages(pages, chunk_size=700, overlap=120)
    assert all("20 40 60" not in c.text for c in chunks)


def test_a_section_carries_forward_to_pages_that_do_not_repeat_the_header():
    # the running footer sits on alternating pages only
    footer = "第第 11 節節個人における AI 利用の現状個人における AI 利用の現状"
    pages = {
        n: (f"{n}枚目の本文である。\n{footer}" if n % 2 else f"{n}枚目の本文である。")
        for n in range(1, 7)
    }
    sections = {c.page: c.section for c in chunk_pages(pages, chunk_size=700, overlap=120)}
    assert sections[2] == sections[1] == "第1節 個人におけるAI利用の現状"


def test_a_new_section_replaces_the_carried_one():
    pages = {1: "第1章\n本文一。", 2: "本文二。", 3: "第2章\n本文三。", 4: "本文四。"}
    sections = {c.page: c.section for c in chunk_pages(pages, chunk_size=700, overlap=120)}
    assert [sections[p] for p in (1, 2, 3, 4)] == ["第1章", "第1章", "第2章", "第2章"]


def test_nothing_is_carried_before_the_first_heading():
    pages = {1: "前書きの本文。", 2: "第1章\n本文。"}
    sections = {c.page: c.section for c in chunk_pages(pages, chunk_size=700, overlap=120)}
    assert sections[1] == ""
