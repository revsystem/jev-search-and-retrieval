"""PDF loading, Japanese text normalisation and chunking.

The normalisation rules below are not generic tidying: each one was added
against an artefact observed in the extraction of a 総務省 情報通信白書 PDF
(decorative letter spacing, text rendered twice for a bold effect, hair
spaces, a broken-font line, and running headers repeated on every page).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

CHAPTER = re.compile(r"^第\s*\d+\s*[章節部編]")
# The running footer renders every marker character twice: 第第 11 節節 …
DOUBLED_MARKER = re.compile(r"^第第\s*(\d)\1\s*([章節部編])\2\s*(.*)$")
INVISIBLE = re.compile(r"[ ​‌‍﻿­]")
_CJK = re.compile(
    r"[぀-ゟ゠-ヿ㐀-䶿一-鿿ｦ-ﾟ"
    r"、。・「」『』（）［］〔〕【】〜ー]"
)
# Scripts this document is expected to contain; anything else signals a font
# whose glyphs decoded to the wrong code points.
_EXPECTED = re.compile(
    r"[　-〿぀-ゟ゠-ヿ㐀-鿿＀-￯"
    r" -~ -ÿ‐-›←-⇿─-◿]"
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page: int
    section: str
    text: str

    def embedding_text(self) -> str:
        """Text handed to the embedding model. The section keeps the chunk locatable."""
        return f"{self.section}\n{self.text}" if self.section else self.text

    def metadata(self) -> dict[str, object]:
        return {"chunk_id": self.chunk_id, "page": self.page, "section": self.section}


def _is_cjk(char: str) -> bool:
    return bool(_CJK.match(char))


def _is_mojibake(line: str) -> bool:
    if len(line) < 4:
        return False
    unexpected = sum(1 for char in line if not _EXPECTED.match(char))
    return unexpected / len(line) > 0.3


def _unspace_letterspaced(line: str) -> str:
    """A line tracked one space per character is a heading, not prose."""
    tokens = line.split()
    if len(tokens) >= 4 and sum(1 for t in tokens if len(t) == 1) / len(tokens) >= 0.6:
        return "".join(tokens)
    return line


def _drop_typographic_spaces(line: str) -> str:
    """Remove a space that sits next to Japanese; keep spaces between Latin words."""
    out: list[str] = []
    for index, char in enumerate(line):
        if char == " " and 0 < index < len(line) - 1:
            if _is_cjk(line[index - 1]) or _is_cjk(line[index + 1]):
                continue
        out.append(char)
    return "".join(out)


def _collapse_doubled(line: str) -> str:
    """Collapse text the PDF renders twice, keeping any prefix before it.

    A minimum repeat length keeps ordinary Japanese (様々, 人々) intact.
    """
    for size in range(len(line) // 2, 3, -1):
        tail, before = line[-size:], line[:-size]
        if len(set(tail)) < 2:
            # a run of one character (a rule, a filler row) is not doubled text
            continue
        if before.endswith(tail):
            return before[:-size] + tail
        if before.endswith(tail + " "):
            return before[: -size - 1] + " " + tail
    return line


def _clean_line(line: str) -> str | None:
    """Cleaned text, "" for a blank line, or None for a line to delete outright.

    The difference matters: a blank line separates paragraphs, while a deleted
    one (a broken-font line, a running footer) sits inside a sentence that was
    hard-wrapped around it and must not split it.
    """
    line = INVISIBLE.sub("", line)
    line = re.sub(r"[ \t\u3000]+", " ", line).strip()
    if not line:
        return ""
    if _is_mojibake(line):
        return None
    return _collapse_doubled(_drop_typographic_spaces(_unspace_letterspaced(line)))


def detect_boilerplate(
    pages: dict[int, str], min_pages: int = 3, min_ratio: float = 0.2
) -> frozenset[str]:
    """Lines repeating across pages: running headers, footers and chart axes.

    They carry no information for retrieval and, left in, they dominate short
    chunks. The section title usually lives in one of them, so they are
    detected rather than simply discarded.
    """
    seen: dict[str, set[int]] = {}
    for page, raw in pages.items():
        for line in unicodedata.normalize("NFKC", raw).splitlines():
            cleaned = _clean_line(line)
            if cleaned:
                seen.setdefault(cleaned, set()).add(page)  # noqa: E501

    threshold = max(min_pages, int(len(pages) * min_ratio))
    return frozenset(line for line, hits in seen.items() if len(hits) >= threshold)


def normalise_japanese_text(raw: str, drop_lines: frozenset[str] = frozenset()) -> str:
    """Undo PDF layout artefacts and rejoin hard-wrapped Japanese lines.

    NFKC comes first, because PDF text extraction routinely yields full-width
    ASCII and half-width katakana. Without it a keyword rule asking for "生成AI"
    silently matches nothing in a document that renders it 生成ＡＩ, and a figure
    written ２６．７％ is invisible to a rule looking for a percentage.
    """
    cleaned = [_clean_line(line) for line in unicodedata.normalize("NFKC", raw).splitlines()]
    lines = [line for line in cleaned if line is not None and line not in drop_lines]

    paragraphs: list[list[str]] = [[]]
    for line in lines:
        if line:
            paragraphs[-1].append(line)
        elif paragraphs[-1]:
            paragraphs.append([])

    joined = []
    for paragraph in paragraphs:
        if not paragraph:
            continue
        buffer = paragraph[0]
        for line in paragraph[1:]:
            glue = "" if _is_cjk(buffer[-1]) and _is_cjk(line[0]) else " "
            buffer = f"{buffer}{glue}{line}"
        joined.append(buffer)
    return "\n\n".join(joined)


def section_of(raw_page: str, boilerplate: frozenset[str] = frozenset()) -> str:
    """The section a page belongs to.

    A repeated header or footer naming a 章/節 is the most reliable source, so
    it is preferred over a heading found in the body.
    """
    candidates = [
        _clean_line(line) for line in unicodedata.normalize("NFKC", raw_page).splitlines()
    ]
    for line in candidates:
        if line in boilerplate:
            marker = DOUBLED_MARKER.match(line)
            if marker:
                number, kind, title = marker.groups()
                return f"第{number}{kind} {title}".strip()
            if CHAPTER.match(line):
                return line
    for line in candidates:
        if line and line not in boilerplate and CHAPTER.match(line):
            return line
    return ""


def chunk_pages(pages: dict[int, str], chunk_size: int = 700, overlap: int = 120) -> list[Chunk]:
    """Split each page into overlapping windows, carrying page and section provenance."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    boilerplate = detect_boilerplate(pages)
    chunks: list[Chunk] = []
    step = chunk_size - overlap
    # The running footer naming the section appears on alternating pages, so a
    # page without one still belongs to the section that last announced itself.
    section = ""
    for page in sorted(pages):
        section = section_of(pages[page], boilerplate) or section
        text = normalise_japanese_text(pages[page], drop_lines=boilerplate)
        if not text.strip():
            continue
        for index, start in enumerate(range(0, max(len(text), 1), step)):
            window = text[start : start + chunk_size]
            if not window.strip():
                continue
            chunks.append(
                Chunk(chunk_id=f"p{page}-c{index}", page=page, section=section, text=window)
            )
            if start + chunk_size >= len(text):
                break
    return chunks


def load_pdf(path: str | Path) -> dict[int, str]:
    """Read a local PDF into {1-based page number: raw text}."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return {number: (page.extract_text() or "") for number, page in enumerate(reader.pages, 1)}


def download_pdf(url: str, destination: str | Path) -> Path:
    import httpx

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(url, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target
