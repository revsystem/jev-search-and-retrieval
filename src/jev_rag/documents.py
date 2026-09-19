"""PDF loading, Japanese text normalisation and chunking."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^第\s*\d+\s*[章節部編]")
_CJK = re.compile(
    r"[぀-ゟ゠-ヿ㐀-䶿一-鿿ｦ-ﾟ"
    r"、。・「」『』（）［］〔〕【】〜ー]"
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


def normalise_japanese_text(raw: str) -> str:
    """Undo PDF layout artefacts: width variants, hard-wrapped lines, blank runs.

    NFKC first, because PDF text extraction routinely yields full-width ASCII
    and half-width katakana. Without it a keyword rule asking for "生成AI"
    silently matches nothing in a document that renders it 生成ＡＩ, and every
    figure written ２６．７％ is invisible to a rule looking for a percentage.
    Section headings are read before this runs, so they keep their original
    characters.
    """
    raw = unicodedata.normalize("NFKC", raw)
    lines = [re.sub(r"[ \t　]+", " ", line).strip() for line in raw.replace("　", " ").splitlines()]

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


def _section_of(raw_page: str) -> str:
    for line in raw_page.splitlines():
        stripped = line.strip()
        if HEADING.match(stripped):
            return stripped
    return ""


def chunk_pages(pages: dict[int, str], chunk_size: int = 700, overlap: int = 120) -> list[Chunk]:
    """Split each page into overlapping windows, carrying page and section provenance."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for page in sorted(pages):
        section = _section_of(pages[page])
        text = normalise_japanese_text(pages[page])
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
