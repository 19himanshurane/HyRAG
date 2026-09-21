import re
from dataclasses import dataclass

from hyrag.loader import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    chunk_index: int
    heading: str
    page: int | None
    strategy: str
    char_count: int

    def text_for_search(self) -> str:
        """What gets embedded and keyword-indexed: the heading path + the chunk body.
        Headings often hold the most searchable words (like an error code) that the body never repeats."""
        return f"{self.heading}\n\n{self.text}" if self.heading else self.text


def chunk_fixed(doc: Document, size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Cut the whole document into windows of `size` characters, each repeating the last `overlap` characters."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size, or the window never moves forward")

    # 1. Glue all sections into one long string, remembering where each section starts and ends.
    full_text = ""
    section_spans = []  # (start, end, heading, page)
    for section in doc.sections:
        start = len(full_text)
        full_text += section.text + "\n\n"
        section_spans.append((start, len(full_text), section.heading, section.page))

    # 2. Slide a window across the string. Each step moves forward by (size - overlap).
    chunks = []
    step = size - overlap
    for start in range(0, len(full_text), step):
        text = full_text[start : start + size].strip()
        if not text:
            continue

        # 3. The chunk gets the heading/page of whichever section its first character came from.
        heading, page = next((h, p) for s, e, h, p in section_spans if s <= start < e)

        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:fixed:{len(chunks)}",
                doc_id=doc.doc_id,
                source=doc.source,
                text=text,
                chunk_index=len(chunks),
                heading=heading,
                page=page,
                strategy="fixed",
                char_count=len(text),
            )
        )
        if start + size >= len(full_text):
            break  # this window already reached the end
    return chunks


# ---------------------------------------------------------------------------
# Structure-aware: split each section on its own, at the most natural boundary.
# ---------------------------------------------------------------------------

SEPARATORS = ["\n\n", "\n", ". ", " "]  # paragraph, line, sentence, word


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Pack small pieces into chunks up to `size`. A new chunk starts with the last
    whole pieces of the previous one (up to `overlap` chars), so overlap never cuts a word."""
    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and len("".join(current)) + len(piece) > size:
            chunks.append("".join(current))
            while current and (
                len("".join(current)) > overlap or len("".join(current)) + len(piece) > size
            ):
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append("".join(current))
    return chunks


def split_recursive(text: str, size: int, overlap: int, separators: list[str] = SEPARATORS) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size, or the window never moves forward")
    if len(text) <= size:
        return [text]
    if not separators:
        # Nothing natural left to split on (e.g. one giant word): fall back to a hard cut.
        starts = range(0, max(len(text) - overlap, 1), size - overlap)
        return [text[i : i + size] for i in starts]

    sep, finer = separators[0], separators[1:]
    if sep not in text:
        return split_recursive(text, size, overlap, finer)

    # Split *after* each separator so it stays attached (the "." stays on its sentence).
    pieces = [p for p in re.split(f"(?<={re.escape(sep)})", text) if p]

    small_pieces: list[str] = []
    for piece in pieces:
        if len(piece) > size:
            small_pieces.extend(split_recursive(piece, size, overlap, finer))  # the "recursive" part
        else:
            small_pieces.append(piece)
    return _merge(small_pieces, size, overlap)


def chunk_structure(doc: Document, size: int = 800, overlap: int = 120) -> list[Chunk]:
    chunks = []
    for section in doc.sections:  # each section separately, so a chunk never crosses a heading
        for piece in split_recursive(section.text, size, overlap):
            text = piece.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:structure:{len(chunks)}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    text=text,
                    chunk_index=len(chunks),
                    heading=section.heading,
                    page=section.page,
                    strategy="structure",
                    char_count=len(text),
                )
            )
    return chunks
