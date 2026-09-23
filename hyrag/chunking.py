import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hyrag.loader import Document


def _content_id(doc_id: str, strategy: str, heading: str, text: str, seen: Counter) -> str:
    """Chunk ids come from content, not position: hash of what gets embedded (heading + text).
    An edit elsewhere in the document leaves this chunk's id untouched, so the vector store only
    re-embeds chunks that really changed. Identical repeats get a -2, -3 suffix to stay unique."""
    digest = hashlib.sha1(f"{heading}\n\n{text}".encode()).hexdigest()[:12]
    seen[digest] += 1
    suffix = f"-{seen[digest]}" if seen[digest] > 1 else ""
    return f"{doc_id}:{strategy}:{digest}{suffix}"


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
    seen: Counter = Counter()
    step = size - overlap
    for start in range(0, len(full_text), step):
        text = full_text[start : start + size].strip()
        if not text:
            continue

        # 3. The chunk gets the heading/page of whichever section its first character came from.
        heading, page = next((h, p) for s, e, h, p in section_spans if s <= start < e)

        chunks.append(
            Chunk(
                chunk_id=_content_id(doc.doc_id, "fixed", heading, text, seen),
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
    seen: Counter = Counter()
    for section in doc.sections:  # each section separately, so a chunk never crosses a heading
        for piece in split_recursive(section.text, size, overlap):
            text = piece.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=_content_id(doc.doc_id, "structure", section.heading, text, seen),
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


# ---------------------------------------------------------------------------
# Semantic: inside each section, cut where the topic changes between neighbouring units.
# ---------------------------------------------------------------------------


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...  # rows scaled to length 1


def _join_wrapped_lines(lines: list[str]) -> str:
    """PDF lines break wherever the page ran out of width, not where meaning changes: re-join them.
    A line ending in "-" before a lowercase letter is a compound split by wrapping ("hardware-" +
    "protected"), so it joins without a space."""
    out = ""
    for line in (l.strip() for l in lines):
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and line[:1].islower():
            out += line
        else:
            out += " " + line
    return out


def semantic_units(text: str, file_type: str, size: int, min_unit: int = 150) -> list[str]:
    """Split one section into the pieces that get compared. What a "paragraph" is depends on the format:
    PDF = blank-line paragraphs with wrapped lines re-joined; HTML = one line per block (<p>, <li>, row);
    Markdown/text = blank-line paragraphs. Pieces longer than `size` are split at sentence/word boundaries;
    consecutive tiny pieces (table rows, list items) are grouped, because very short text embeds noisily."""
    if file_type == ".pdf":
        raw = [_join_wrapped_lines(p.split("\n")) for p in re.split(r"\n\s*\n", text)]
    elif file_type in (".html", ".htm"):
        raw = text.split("\n")
    else:
        raw = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    for p in (p.strip() for p in raw):
        if p:
            pieces.extend(s.strip() for s in split_recursive(p, size, 0) if s.strip())

    joiner = "\n" if file_type in (".html", ".htm") else "\n\n"
    units: list[str] = []
    for p in pieces:
        if units and len(units[-1]) < min_unit and len(units[-1]) + len(joiner) + len(p) <= size:
            units[-1] = units[-1] + joiner + p  # grow the tiny group
        else:
            units.append(p)
    return units


def chunk_semantic(
    doc: Document,
    embedder: Embedder,
    size: int = 800,
    breakpoint_percentile: float = 25,
    min_chunk: int = 250,
) -> list[Chunk]:
    """Within each section, a new chunk starts where the similarity between two neighbouring units is in
    the lowest `breakpoint_percentile` % for THIS document (a relative threshold: mistral-embed rates even
    unrelated text ~0.65, so no fixed number would mean "different topic"), or when `size` would be exceeded.
    No overlap: a cut here marks a topic change, and repeating text across it would blend the two topics.
    A chunk shorter than `min_chunk` is never closed by a topic break, only by the size cap."""
    joiner = "\n" if doc.file_type in (".html", ".htm") else "\n\n"
    per_section = [semantic_units(s.text, doc.file_type, size) for s in doc.sections]
    flat = [u for units in per_section for u in units]
    if not flat:
        return []
    vectors = embedder.embed(flat)

    # Similarity between each unit and the next one IN THE SAME SECTION (sections never merge).
    sims_per_section, i = [], 0
    for units in per_section:
        v = vectors[i : i + len(units)]
        sims_per_section.append([float(a @ b) for a, b in zip(v, v[1:])])
        i += len(units)
    all_sims = [s for sims in sims_per_section for s in sims]
    threshold = float(np.percentile(all_sims, breakpoint_percentile)) if len(all_sims) >= 4 else None

    chunks: list[Chunk] = []
    seen: Counter = Counter()

    def emit(section, parts: list[str]) -> None:
        text = joiner.join(parts)
        chunks.append(Chunk(
            chunk_id=_content_id(doc.doc_id, "semantic", section.heading, text, seen),
            doc_id=doc.doc_id, source=doc.source, text=text, chunk_index=len(chunks),
            heading=section.heading, page=section.page, strategy="semantic", char_count=len(text),
        ))

    for section, units, sims in zip(doc.sections, per_section, sims_per_section):
        # Characters from each unit to the end of the section: a topic cut must leave a tail that can
        # stand alone. Otherwise a closing label ("Request Process") that looks unlike the paragraph
        # before it becomes a 15-character chunk of its own.
        tail = [len(joiner.join(units[i:])) for i in range(len(units))]
        current = [units[0]] if units else []
        for i, (unit, sim) in enumerate(zip(units[1:], sims), start=1):
            length = len(joiner.join(current))
            too_big = length + len(joiner) + len(unit) > size
            topic_break = (threshold is not None and sim < threshold
                           and length >= min_chunk and tail[i] >= min_chunk)
            if too_big or topic_break:
                emit(section, current)
                current = []
            current.append(unit)
        if current:
            emit(section, current)
    return chunks
