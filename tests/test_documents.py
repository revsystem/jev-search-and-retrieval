from jev_rag.documents import Chunk, chunk_pages, normalise_japanese_text


def test_normalise_collapses_pdf_layout_noise():
    raw = "第 1 節　 生成 AI の\n動向\n\n\n　ここでは、"
    assert normalise_japanese_text(raw) == "第 1 節 生成 AI の動向\n\nここでは、"


def test_normalise_joins_japanese_lines_without_inserting_spaces():
    assert normalise_japanese_text("生成AIの利用が\n拡大している") == "生成AIの利用が拡大している"


def test_normalise_keeps_a_space_between_latin_words():
    assert normalise_japanese_text("generative AI\nadoption") == "generative AI adoption"


def test_chunking_splits_long_pages_and_keeps_provenance():
    pages = {1: "あ" * 1200}
    chunks = chunk_pages(pages, chunk_size=500, overlap=100)
    assert len(chunks) == 3
    assert all(isinstance(c, Chunk) for c in chunks)
    assert {c.page for c in chunks} == {1}
    assert chunks[0].chunk_id == "p1-c0"


def test_chunks_overlap_by_the_requested_number_of_characters():
    page = "".join(str(i % 10) for i in range(1000))
    chunks = chunk_pages({1: page}, chunk_size=400, overlap=100)
    assert chunks[0].text[-100:] == chunks[1].text[:100]


def test_short_pages_produce_exactly_one_chunk():
    chunks = chunk_pages({7: "短い本文"}, chunk_size=500, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].page == 7
    assert chunks[0].text == "短い本文"


def test_blank_pages_are_skipped():
    assert chunk_pages({1: "   \n\n ", 2: "本文"}, chunk_size=500, overlap=100)[0].page == 2


def test_section_heading_is_carried_forward_into_later_chunks():
    pages = {1: "第2節　データ流通\n" + "本" * 900}
    chunks = chunk_pages(pages, chunk_size=400, overlap=50)
    assert all(c.section == "第2節　データ流通" for c in chunks)


def test_embedding_text_prefixes_provenance_for_retrievability():
    chunk = Chunk(chunk_id="p3-c0", page=3, section="第1節　AI動向", text="本文")
    assert chunk.embedding_text().startswith("第1節　AI動向")
    assert "本文" in chunk.embedding_text()


def test_full_width_latin_is_normalised_so_keyword_rules_can_match():
    # PDF text extraction routinely yields full-width ASCII; without NFKC a rule
    # requiring "生成AI" silently matches nothing in a document that uses 生成ＡＩ.
    assert normalise_japanese_text("生成ＡＩの利用") == "生成AIの利用"


def test_full_width_digits_and_percent_are_normalised():
    assert normalise_japanese_text("２６．７％") == "26.7%"


def test_half_width_katakana_is_widened():
    assert normalise_japanese_text("ﾄﾗﾋｯｸ") == "トラヒック"


def test_normalisation_leaves_ordinary_japanese_untouched():
    assert normalise_japanese_text("生成AIの利用経験は26.7%であった。") == (
        "生成AIの利用経験は26.7%であった。"
    )


def test_the_raw_section_heading_is_kept_verbatim():
    # the heading is read before normalisation, so it keeps its ideographic space
    chunks = chunk_pages({1: "第2節　データ流通\n" + "本" * 200}, chunk_size=400, overlap=50)
    assert chunks[0].section == "第2節　データ流通"
