import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED = {".md", ".txt", ".html", ".htm", ".pdf"}
# Bump whenever parsing output changes: stored documents parsed by an older version get re-parsed.
PARSER_VERSION = "2026-09-22.1"
# Refuse inputs that would take minutes and gigabytes (measured: ~90 ms and ~0.1 MB per PDF page).
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 2000


@dataclass
class Section:
    text: str
    heading: str
    page: int | None = None


@dataclass
class Document:
    doc_id: str
    source: str
    file_type: str
    sections: list[Section]


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(?<=\S)[ \t]+", " ", text)  # squeeze spaces *between* words only...
    text = re.sub(r"[ \t]+\n", "\n", text)      # ...and trailing ones, so leading indentation
    text = re.sub(r"\n{3,}", "\n\n", text)      # survives (it matters in code, YAML, Python)
    return text.strip("\n").rstrip()


def decode_text(data: bytes) -> str:
    try:
        text = data.decode("utf-8-sig")  # "-sig" also removes the invisible BOM Windows editors add
    except UnicodeDecodeError:
        # Not valid UTF-8: most likely an older Windows/Word export. Best-effort guess.
        text = data.decode("cp1252", errors="replace")
    # Windows (\r\n) and old Mac (\r) line endings -> \n, so every parser's patterns see one kind.
    # Git on Windows checks files out with \r\n by default.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def make_doc_id(source: str) -> str:
    """`source` is the file's path relative to the corpus root, e.g. "api/README.md",
    so two different README.md files in different folders get different ids."""
    return hashlib.sha1(source.encode()).hexdigest()[:12]


# ---------- one parser per format; each returns a list of Sections ----------

def parse_txt(text: str, title: str) -> list[Section]:
    return [Section(text=clean(text), heading=title)]


def strip_markdown_inline(line: str) -> str:
    line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", line)   # ![alt](img.png)  -> alt
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)    # [text](url)      -> text
    line = re.sub(r"`([^`]+)`", r"\1", line)                # `code`           -> code
    line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)            # **bold**         -> bold
    line = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\1", line)  # *italic* -> italic
    line = re.sub(r"^\s*>\s?", "", line)                    # > blockquote
    line = re.sub(r"^\s*[-*+]\s+", "- ", line)              # normalise bullets
    return line


def split_front_matter(text: str) -> tuple[str | None, str]:
    """Static-site markdown often starts with a YAML block between '---' lines. Return (title, rest)."""
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not match:
        return None, text
    title = re.search(r"^title:\s*[\"']?(.+?)[\"']?\s*$", match.group(1), re.M)
    return (title.group(1) if title else None), text[match.end():]


_SHORTCODE = re.compile(r"\{\{[<%]\s*(/?)([\w-]+)(.*?)\s*[>%]\}\}", re.S)
_ALERTS = {"note": "Note:", "caution": "Caution:", "warning": "Warning:"}
_HEADING_KEYS = {"whatsnext": "What's next", "prerequisites": "Before you begin", "objectives": "Objectives"}


def _shortcode_to_text(m: re.Match) -> str:
    """Hugo shortcodes ({{< name key="value" >}}) render to text on the website; keep that text."""
    closing, name, args = m.group(1), m.group(2), m.group(3)
    attrs = dict(re.findall(r'([\w-]+)="([^"]*)"', args))
    if closing:
        return ""
    if name == "glossary_tooltip":
        return attrs.get("text") or attrs.get("term_id", "").replace("-", " ").replace("_", " ")
    if name in _ALERTS:
        return _ALERTS[name]
    if name == "feature-state":
        if "feature_gate_name" in attrs:
            return f"Feature gate: {attrs['feature_gate_name']}"
        return f"Feature state: {attrs.get('state', '')} {attrs.get('for_k8s_version', '')}".strip()
    if name == "code_sample" and "file" in attrs:
        return f"(example file: {attrs['file']})"
    if name == "heading":
        key = args.strip().strip('"')
        return _HEADING_KEYS.get(key, key)
    if name == "figure":
        return attrs.get("caption", "")
    return attrs.get("text", "")  # param, skew, ...: site variables with no recoverable text


def parse_markdown(text: str, title: str) -> list[Section]:
    front_title, text = split_front_matter(text)
    text = _SHORTCODE.sub(_shortcode_to_text, text)
    # Empty HTML anchors used as link targets: <a id="x" />, <a name="x"></a>. (Placeholders like
    # <name-of-pod> in commands look similar but have no id=/name= attribute, so they survive.)
    text = re.sub(r"<a\s+(?:id|name)=\"[^\"]*\"\s*/?>(?:</a>)?", "", text)
    sections: list[Section] = []
    # e.g. [(0, "Secrets"), (2, "Types of Secret")]; the front-matter title sits at level 0 so it is never popped
    heading_path: list[tuple[int, str]] = [(0, front_title)] if front_title else []
    lines_in_section: list[str] = []
    open_fence: str | None = None  # "```" or "~~~" (or longer) while inside a code block

    def close_section():
        body = clean("\n".join(lines_in_section))
        if body:
            heading = " > ".join(h for _, h in heading_path) or title
            sections.append(Section(text=body, heading=heading))
        lines_in_section.clear()

    for line in text.splitlines():
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence and open_fence is None:
            open_fence = fence.group(1)
            continue  # drop the fence line itself (and its language tag), keep the code inside it
        if open_fence is not None:
            # CommonMark: only a line of the SAME fence character, at least as long, closes the block.
            closes = fence and fence.group(1)[0] == open_fence[0] and len(fence.group(1)) >= len(open_fence) \
                and line.strip() == fence.group(1)
            if closes:
                open_fence = None
            else:
                lines_in_section.append(line)  # code is kept exactly as written
            continue
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            close_section()
            level, heading_text = len(match.group(1)), strip_markdown_inline(match.group(2).strip())
            heading_text = re.sub(r"\s*\{#[^}]*\}\s*$", "", heading_text)  # "Pod phase {#pod-phase}" -> "Pod phase"
            if front_title and level == 1 and heading_text == front_title:
                continue  # the H1 repeats the front-matter title; don't nest "Secrets > Secrets"
            while heading_path and heading_path[-1][0] >= level:
                heading_path.pop()
            heading_path.append((level, heading_text))
        else:
            lines_in_section.append(strip_markdown_inline(line))
    close_section()
    return sections


HTML_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Tags that start a new line. Anything else (span, a, code, b, ...) is inline and flows into the current line.
HTML_BLOCKS = {
    "p", "div", "li", "ul", "ol", "table", "tr", "td", "th", "blockquote", "section",
    "article", "main", "dd", "dt", "dl", "figure", "figcaption", "br", "hr",
}
# Site-specific page furniture that isn't marked up as nav/aside/footer.
HTML_JUNK_SELECTORS = ", ".join([
    "#docComments",  # postgresql.org: "Submit correction" feedback box under every page
])
PERMALINK_MARKS = {"#", "¶", "§", "🔗", ""}


def _cell_text(cell) -> str:
    # Space only where a block (list item, paragraph) ends, so "<code>x</code>)" stays "x)" but
    # "<li>a</li><li>b</li>" becomes "a b" rather than "ab".
    for block in cell.find_all(list(HTML_BLOCKS)):
        block.insert_after(" ")
    return " ".join(cell.get_text().split())


def parse_html(html: str, title: str) -> list[Section]:
    from bs4 import BeautifulSoup, Comment, Doctype, NavigableString

    soup = BeautifulSoup(html, "html.parser")
    if soup.title:
        title = soup.title.get_text(strip=True)
    # <aside> holds sidebars (GitLab's maintainer list and "On this page" box). Some sites use it for
    # inline notes instead; none of our corpus does inside the main content, so we drop it.
    for junk in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]):
        junk.decompose()
    for junk in soup.select(HTML_JUNK_SELECTORS):
        junk.decompose()
    for a in soup.find_all("a", href=True):
        # Permalink markers next to headings: <a href="#section">#</a> (also ¶ in Sphinx/MkDocs).
        if a["href"].startswith("#") and a.get_text(strip=True) in PERMALINK_MARKS:
            a.decompose()

    sections: list[Section] = []
    heading_path: list[tuple[int, str]] = []
    lines_in_section: list[str] = []
    current_line: list[str] = []

    def end_line():
        # Collapse the HTML source's own line breaks/indentation into single spaces.
        line = " ".join("".join(current_line).split())
        if line:
            lines_in_section.append(line)
        current_line.clear()

    def close_section():
        end_line()
        body = clean("\n".join(lines_in_section))
        if body:
            heading = " > ".join(h for _, h in heading_path) or title
            sections.append(Section(text=body, heading=heading))
        lines_in_section.clear()

    # Visit every node exactly once, so text nested like <li><p>x</p></li> is never read twice, and
    # text sitting directly inside a <div> is never skipped. An explicit stack instead of recursion:
    # Python stops recursing at ~1,000 levels, and generated or hostile pages can nest deeper.
    end_of_block = object()  # stack marker: "the block element we entered is finished"
    stack: list = [iter((soup.body or soup).children)]
    while stack:
        top = stack[-1]
        if top is end_of_block:
            stack.pop()
            end_line()
            continue
        child = next(top, None)
        if child is None:
            stack.pop()
            continue
        if isinstance(child, (Comment, Doctype)):
            continue
        if isinstance(child, NavigableString):
            current_line.append(str(child))
        elif child.name in HTML_HEADINGS:
            close_section()
            level = int(child.name[1])
            while heading_path and heading_path[-1][0] >= level:
                heading_path.pop()
            heading_path.append((level, " ".join(child.get_text().split())))
        elif child.name == "pre":
            end_line()
            lines_in_section.append(child.get_text().strip("\n"))  # keep code exactly
        elif child.name == "tr":
            # One line per row, so "23505 | unique_violation" can never be split across two chunks.
            end_line()
            cells = [_cell_text(c) for c in child.find_all(["td", "th"], recursive=False)]
            row = " | ".join(c for c in cells if c)
            if row:
                lines_in_section.append(row)
        elif child.name in HTML_BLOCKS:
            end_line()
            stack.append(end_of_block)
            stack.append(iter(child.children))
        else:
            stack.append(iter(child.children))
    close_section()
    return sections


@dataclass(frozen=True)
class PdfTuning:
    """Every threshold the PDF parser uses, in one place. Tuned on NIST SP 800-63B-4 and 800-61r3;
    change them here (and re-run tests/test_pdf.py) rather than editing numbers inside the logic."""

    margin_band: float = 0.10            # running headers/footers live in the top/bottom 10% of a page...
    repeat_share: float = 0.5            # ...and recur on at least half of the pages
    column_gap_sizes: float = 3.0        # a gap wider than 3 font sizes splits a line into columns (tables, tabs)
    heading_size_slack: float = 0.5      # a heading may be up to 0.5pt smaller than body text
    big_title_ratio: float = 1.15        # text 15% larger than body is a top-level title
    max_heading_words: int = 8           # an unnumbered bold line longer than this is emphasis, not a title
    max_numbered_heading_words: int = 20
    wrap_edge_share: float = 0.9         # a heading line reaching 90% of the text's right edge wrapped
    paragraph_gap_factor: float = 1.5    # a paragraph break is a gap 1.5x the normal line gap...
    paragraph_gap_extra: float = 0.3     # ...or normal + 0.3 font sizes, whichever is larger


PDF_TUNING = PdfTuning()


@dataclass
class PdfLine:
    page: int
    page_height: float
    top: float
    bottom: float
    x1: float  # right end of the line
    text: str
    size: float  # most common font size on the line
    bold: bool   # True only if EVERY character is bold ("Note: some text" is not)
    segments: list[str]  # the line split wherever characters are > 3 font-sizes apart (table columns, tabs)


_NUMBERED_HEADING = re.compile(r"^(?:Appendix\s+[A-Z]\b\.?|(\d+(?:\.\d+)*)\.?)\s+\S")
_DOT_LEADERS = re.compile(r"(?:\.\s?){5,}")                     # table of contents: "Intro ......... 3"
_TABLE_CONTINUED = re.compile(r"^\(?continued (?:on next|from previous) page\)?$", re.I)
_CAPTION = re.compile(r"^(?:Fig\.|Figure|Table)\s*\d", re.I)
_BULLET_ONLY = re.compile(r"^[o•◦▪▫‣∙·]$")  # a bullet symbol on its own line ("-" is NOT here: in tables it means "n/a")
_SECTION_NUMBER = re.compile(r"(?:\d+(?:\.\d+)*\.?|Appendix\s+[A-Z]\.?)")
_ROMAN_PAGE = re.compile(r"\b[ivxlc]+\b$", re.I)                # front-matter page numbers: ii, iv, xii
# Standard document parts: always top-level, never children of the section before them.
_DOC_PARTS = {
    "abstract", "executive summary", "keywords", "audience", "preface", "foreword", "acknowledgments",
    "acknowledgements", "table of contents", "list of tables", "list of figures", "introduction",
    "references", "bibliography", "glossary", "index",
}
# Parts made of entries (terms, acronyms): the bold entries below them are their children.
_ENTRY_LISTS = {"glossary", "index", "acronyms", "abbreviations", "definitions", "terms and definitions"}


def _read_pdf_lines(pdf) -> list[PdfLine]:
    lines = []
    for page in pdf.pages:
        page_number = page.page_number  # the real 1-based number, also when only a page range is open
        for ln in page.extract_text_lines(return_chars=True):
            chars = [c for c in ln["chars"] if c["text"].strip()]
            if not chars or _BULLET_ONLY.match(ln["text"].strip()):
                continue
            segments, last = [[]], None
            for c in ln["chars"]:
                if c["text"].strip():
                    if last is not None and c["x0"] - last["x1"] > PDF_TUNING.column_gap_sizes * c["size"]:
                        segments.append([])
                    last = c
                segments[-1].append(c["text"])
            lines.append(PdfLine(
                page=page_number, page_height=page.height, top=ln["top"], bottom=ln["bottom"], x1=ln["x1"],
                text=ln["text"].strip(),
                size=Counter(round(c["size"], 1) for c in chars).most_common(1)[0][0],
                bold=all("bold" in c["fontname"].lower() for c in chars),
                segments=[s for s in ("".join(seg).strip() for seg in segments) if s],
            ))
        # pdfplumber caches every character object of a page until the PDF is closed (~4 MB/page on
        # NIST docs). We've copied what we need into PdfLine, so free the page now.
        page.close()
    return lines


def _drop_running_headers(lines: list[PdfLine], page_count: int) -> list[PdfLine]:
    """A running header/footer is text in the top or bottom 10% of the page that recurs on at least
    half the pages. Digits are masked so "Page 7" and "Page 8" count as the same line."""
    if page_count < 2:
        return lines

    band = PDF_TUNING.margin_band

    def in_margin(l: PdfLine) -> bool:
        return l.top < band * l.page_height or l.bottom > (1 - band) * l.page_height

    def mask(text: str) -> str:
        return _ROMAN_PAGE.sub("#", re.sub(r"\d+", "#", text))  # "Page 7", "7" and "vii" all -> "#"

    pages_with = Counter()
    for page in {l.page for l in lines}:
        pages_with.update({mask(l.text) for l in lines if l.page == page and in_margin(l)})
    repeated = {t for t, n in pages_with.items() if n >= max(2, page_count * PDF_TUNING.repeat_share)}
    return [l for l in lines if not (in_margin(l) and mask(l.text) in repeated)]


def _heading_level(line: PdfLine, body_size: float, section_depth: int) -> int | None:
    """Return the heading level for this line, or None if it's body text.
    Rules come from studying the NIST PDFs: headings are fully bold and at least body size;
    many are the SAME size as body text, so boldness + numbering matter more than size."""
    t = PDF_TUNING
    if not line.bold or line.size < body_size - t.heading_size_slack:
        return None  # smaller bold text is table headers, captions, footnotes
    if _DOT_LEADERS.search(line.text) or _TABLE_CONTINUED.match(line.text) or _CAPTION.match(line.text):
        return None
    if len(line.segments) > 1:
        # "Type of Secret      Purpose      Reference" is a bold table header row, not a title. The one
        # exception: "3.1<tab>Passwords", a number and a title separated by a tab stop.
        tabbed_number = len(line.segments) == 2 and _SECTION_NUMBER.fullmatch(line.segments[0])
        if not tabbed_number:
            return None
    words = len(line.text.split())
    numbered = _NUMBERED_HEADING.match(line.text)
    if numbered:
        if words > t.max_numbered_heading_words:
            return None
        return numbered.group(1).count(".") + 1 if numbered.group(1) else 1  # "3.1.2" -> 3, "Appendix B" -> 1
    if words > t.max_heading_words or line.text.endswith((".", ",", ";")):
        return None  # a bold sentence for emphasis, not a title
    if line.size > body_size * t.big_title_ratio or line.text.lower() in _DOC_PARTS:
        return 1  # visibly larger, or a standard part like "References": top-level
    return section_depth + 1  # e.g. a glossary term or "Note" title inside the current numbered section


def _usable_pdf_title(meta_title: str | None) -> str | None:
    """PDF metadata titles are often junk ("Microsoft Word - draft3.docx", "untitled")."""
    t = (meta_title or "").strip()
    if not t or t.lower() in {"untitled", "title"} or re.search(r"\.(docx?|pdf|pptx?|indd)$|^microsoft ", t, re.I):
        return None
    return t


def pdf_info(data: bytes) -> tuple[int, str | None]:
    """(page count, usable metadata title). Cheap: characters are only extracted when a page is read."""
    import io

    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        _check_page_count(len(pdf.pages))
        return len(pdf.pages), _usable_pdf_title((pdf.metadata or {}).get("Title"))


def _check_page_count(pages: int) -> None:
    if pages > MAX_PDF_PAGES:
        raise ValueError(f"PDF has {pages} pages; the limit is {MAX_PDF_PAGES}")


def read_pdf_page_range(data: bytes, first: int, last: int) -> list[PdfLine]:
    """Lines of pages first..last (1-based, inclusive). Top-level so worker processes can run it:
    reading pages is ~97% of PDF time and pages are independent, so they can be read in parallel."""
    import io

    import pdfplumber

    with pdfplumber.open(io.BytesIO(data), pages=list(range(first, last + 1))) as pdf:
        return _read_pdf_lines(pdf)


def parse_pdf(data: bytes, title: str) -> list[Section]:
    import io

    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        _check_page_count(len(pdf.pages))
        lines = _read_pdf_lines(pdf)
        page_count = len(pdf.pages)
        title = _usable_pdf_title((pdf.metadata or {}).get("Title")) or title
    return pdf_sections_from_lines(lines, page_count, title)


def pdf_sections_from_lines(lines: list[PdfLine], page_count: int, title: str) -> list[Section]:
    """Everything after reading: needs ALL lines of the document (running headers are found by
    comparing pages; body size and line spacing are document-wide statistics)."""
    import statistics

    lines = [l for l in _drop_running_headers(lines, page_count) if not _TABLE_CONTINUED.match(l.text)]
    if not lines:
        return []

    # Body text size = the size used by the most characters in the document.
    size_weight = Counter()
    for l in lines:
        size_weight[l.size] += len(l.text)
    body_size = size_weight.most_common(1)[0][0]
    # Where full lines of text end; a heading line reaching it probably wrapped onto the next line.
    right_edge = statistics.quantiles([l.x1 for l in lines], n=10)[-1] if len(lines) > 1 else lines[0].x1
    # The document's normal gap between two lines of the same paragraph. A paragraph break is a gap
    # clearly bigger than THIS document's normal spacing (fixed thresholds break on loosely spaced PDFs).
    gaps = [b.top - a.bottom for a, b in zip(lines, lines[1:])
            if a.page == b.page and 0 <= b.top - a.bottom < 2 * body_size]
    normal_gap = statistics.median(gaps) if gaps else 0.0
    paragraph_gap = max(PDF_TUNING.paragraph_gap_factor * normal_gap,
                        normal_gap + PDF_TUNING.paragraph_gap_extra * body_size)

    sections: list[Section] = []
    heading_path: list[tuple[int, str]] = [(0, title)]
    section_depth = 0  # level of the last numbered/large heading; unnumbered bold titles nest under it
    buffer: list[str] = []
    buffer_page: int | None = None
    prev: PdfLine | None = None
    prev_was_heading = False

    def close_section():
        body = clean("\n".join(buffer))
        if body:
            sections.append(Section(text=body, heading=" > ".join(h for _, h in heading_path), page=buffer_page))
        buffer.clear()

    for line in lines:
        new_page = prev is not None and line.page != prev.page
        level = _heading_level(line, body_size, section_depth)
        if level is not None:
            continues_prev_heading = (
                prev_was_heading and prev.page == line.page and abs(prev.size - line.size) < 0.2
                and not _NUMBERED_HEADING.match(line.text) and line.top - prev.bottom < line.size
                # ...and the previous line actually ran out of room, or this one starts mid-sentence.
                # Without this, "Appendix C. Acronyms" swallowed the first acronym "AAL" below it.
                and (prev.x1 >= PDF_TUNING.wrap_edge_share * right_edge or line.text[:1].islower())
            )
            if continues_prev_heading:  # a long title wrapped onto a second line
                lvl, text = heading_path[-1]
                heading_path[-1] = (lvl, f"{text} {line.text}")
            else:
                close_section()
                while heading_path[-1][0] >= level:
                    heading_path.pop()
                heading_path.append((level, line.text))
                if _NUMBERED_HEADING.match(line.text):
                    section_depth = level
                elif level == 1:
                    # After "Glossary", the bold terms are its children. After "Preface" or
                    # "List of Figures", what follows is not.
                    section_depth = 1 if line.text.lower() in _ENTRY_LISTS else 0
            prev, prev_was_heading = line, True
            continue

        if new_page and buffer and buffer[-1].rstrip().endswith((".", "!", "?", ":")):
            # A new page usually starts a new section (so each section has one page number)...
            close_section()
        # ...but a sentence cut off by the page break stays in one piece. Its section keeps the page
        # where the passage STARTS, which is where a reader following the citation should look.
        if not buffer:
            buffer_page = line.page
        elif line.page == prev.page and line.top - prev.bottom > paragraph_gap:
            buffer.append("")  # blank line = paragraph break, which the chunker prefers to split on
        buffer.append(line.text)
        prev, prev_was_heading = line, False

    close_section()
    return sections


# ---------- the single entry point: any file in, a Document out ----------

def load_document(filename: str, data: bytes) -> Document:
    """`filename` should be the path relative to the corpus root (e.g. "api/README.md")."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type {ext!r}; supported: {sorted(SUPPORTED)}")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{filename!r} is {len(data):,} bytes; the limit is {MAX_FILE_BYTES:,}")
    title = Path(filename).stem

    if ext == ".pdf":
        sections = parse_pdf(data, title)
    else:
        text = decode_text(data)
        if ext == ".md":
            sections = parse_markdown(text, title)
        elif ext in (".html", ".htm"):
            sections = parse_html(text, title)
        else:
            sections = parse_txt(text, title)
    return _make_document(filename, ext, sections)


def _make_document(filename: str, ext: str, sections: list[Section]) -> Document:
    if not sections:
        raise ValueError(f"No text could be extracted from {filename!r}")
    return Document(doc_id=make_doc_id(filename), source=filename, file_type=ext, sections=sections)


def list_corpus_files(root: Path) -> list[Path]:
    """The documents in a corpus folder. If root/manifest.json exists it is the source of truth, so
    notes that live next to the documents (licence README, manifest itself) are never ingested as
    content. Without a manifest, every supported file under root counts."""
    manifest = root / "manifest.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        missing = [e["path"] for e in entries if not (root / e["path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"manifest lists files that don't exist: {missing}")
        files = [root / e["path"] for e in entries]
    else:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED]
    # Sort by the POSIX string: sorting Path objects is case-insensitive on Windows but not on
    # Linux, which would give the same corpus a different order on each machine.
    return sorted(files, key=lambda p: p.as_posix())


# ---------- storage: keep raw originals + processed JSON side by side ----------

class DocumentStore:
    def __init__(self, data_dir: Path = Path("data")):
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.index_path = data_dir / "index.json"  # source -> {sha256, parser_version}
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.failures: dict[str, str] = {}  # source -> error, from the last reprocess_all()
        self._index: dict[str, dict] = (
            json.loads(self.index_path.read_text(encoding="utf-8")) if self.index_path.exists() else {}
        )

    def ingest_file(self, path: Path, *, root: Path) -> Document:
        """`root` is the folder the corpus lives in; the file's path inside it is its identity, so
        hr/policy.md and it/policy.md stay two documents. (Required: using only the file name let
        same-named files in different folders silently overwrite each other.)

        A file whose bytes and parser version match what's already stored is not parsed again."""
        source = path.relative_to(root).as_posix()
        data = path.read_bytes()
        t0 = time.perf_counter()
        cached = self._cached(source, data)
        doc = cached or load_document(source, data)  # parse before storing anything
        raw_copy = self._raw_path(source)
        raw_copy.parent.mkdir(parents=True, exist_ok=True)
        if not raw_copy.exists() or raw_copy.read_bytes() != data:
            raw_copy.write_bytes(data)
        self._save_processed(doc, data)
        self._write_index()
        log.info("ingested %s: %d sections, %s in %.2fs", source, len(doc.sections),
                 "unchanged (reused)" if cached else "parsed", time.perf_counter() - t0)
        return doc

    PDF_PAGES_PER_TASK = 8

    def ingest_many(self, paths: list[Path], *, root: Path, workers: int | None = None) -> list[Document]:
        """Ingest many files, parsing the ones that changed in parallel processes. PDFs are split into
        page ranges, so one big PDF also uses every core (reading pages is ~97% of PDF time).
        Files that fail are skipped and recorded in `self.failures`. Scripts calling this on Windows
        need an `if __name__ == "__main__":` guard: worker processes re-import the caller."""
        from concurrent.futures import ProcessPoolExecutor

        self.failures = {}
        t0 = time.perf_counter()
        items = [(p.relative_to(root).as_posix(), p.read_bytes()) for p in paths]
        docs: dict[str, Document] = {s: d for s, data in items if (d := self._cached(s, data))}
        reused = len(docs)
        todo = [(s, data) for s, data in items if s not in docs]
        pdf_meta: dict[str, tuple[int, str]] = {}
        for s, data in todo:
            if Path(s).suffix.lower() == ".pdf":
                try:
                    count, meta_title = pdf_info(data)
                    pdf_meta[s] = (count, meta_title or Path(s).stem)
                except Exception as e:
                    self.failures[s] = f"{type(e).__name__}: {e}"
        if todo:
            with ProcessPoolExecutor(max_workers=workers or os.cpu_count() or 1) as pool:
                futures: dict[str, list] = {}
                for s, data in todo:
                    if s in pdf_meta:
                        count, step = pdf_meta[s][0], self.PDF_PAGES_PER_TASK
                        futures[s] = [pool.submit(read_pdf_page_range, data, a, min(a + step - 1, count))
                                      for a in range(1, count + 1, step)]
                    elif s not in self.failures:
                        futures[s] = [pool.submit(load_document, s, data)]
            for s, parts in futures.items():
                try:
                    if s in pdf_meta:
                        lines = [line for part in parts for line in part.result()]
                        count, title = pdf_meta[s]
                        docs[s] = _make_document(s, ".pdf", pdf_sections_from_lines(lines, count, title))
                    else:
                        docs[s] = parts[0].result()
                except Exception as e:
                    self.failures[s] = f"{type(e).__name__}: {e}"
        for source, data in items:
            if source in docs:
                raw_copy = self._raw_path(source)
                raw_copy.parent.mkdir(parents=True, exist_ok=True)
                if not raw_copy.exists() or raw_copy.read_bytes() != data:
                    raw_copy.write_bytes(data)
                self._save_processed(docs[source], data)
        self._write_index()
        for source, error in self.failures.items():
            log.warning("skipped %s: %s", source, error)
        log.info("ingest_many: %d files, %d parsed, %d unchanged (reused), %d failed in %.1fs",
                 len(items), len(docs) - reused, reused, len(self.failures), time.perf_counter() - t0)
        return [docs[s] for s, _ in items if s in docs]

    def _raw_path(self, source: str) -> Path:
        """Where a document's raw copy lives. `source` may one day come from an uploaded filename, so
        "../../etc/x" must never resolve to a path outside raw_dir."""
        path = (self.raw_dir / source).resolve()
        if not path.is_relative_to(self.raw_dir.resolve()) or path == self.raw_dir.resolve():
            raise ValueError(f"unsafe document path: {source!r}")
        return path

    def delete(self, source: str) -> None:
        """Remove a document everywhere, so it can never be retrieved or cited again."""
        self._raw_path(source).unlink(missing_ok=True)
        (self.processed_dir / f"{make_doc_id(source)}.json").unlink(missing_ok=True)
        if self._index.pop(source, None) is not None:
            self._write_index()

    def reprocess_all(self, force: bool = False) -> list[Document]:
        """Rebuild every processed file from the saved raw originals, with no re-upload needed.
        Unchanged files (same bytes, same PARSER_VERSION) are reused unless `force=True`.

        A file that fails to parse is skipped and recorded in `self.failures`; its last good processed
        version is kept. Processed files whose raw original no longer exists are removed."""
        docs = []
        self.failures = {}
        sources = sorted(p.relative_to(self.raw_dir).as_posix() for p in self.raw_dir.rglob("*") if p.is_file())
        for source in sources:
            data = (self.raw_dir / source).read_bytes()
            try:
                doc = (None if force else self._cached(source, data)) or load_document(source, data)
            except Exception as e:  # one corrupt upload must not stop the whole library from re-indexing
                self.failures[source] = f"{type(e).__name__}: {e}"
                log.warning("skipped %s: %s (kept its last good version, if any)", source, self.failures[source])
                continue
            self._save_processed(doc, data)
            docs.append(doc)
        live_ids = {make_doc_id(s) for s in sources}
        for processed in self.processed_dir.glob("*.json"):
            if processed.stem not in live_ids:
                processed.unlink()  # its raw file was deleted or renamed: don't serve a ghost document
                log.info("removed orphaned processed file %s (raw original is gone)", processed.name)
        live_sources = set(sources)
        self._index = {s: v for s, v in self._index.items() if s in live_sources}
        self._write_index()
        return docs

    def _cached(self, source: str, data: bytes) -> Document | None:
        entry = self._index.get(source)
        processed = self.processed_dir / f"{make_doc_id(source)}.json"
        if (not entry or entry["sha256"] != hashlib.sha256(data).hexdigest()
                or entry["parser_version"] != PARSER_VERSION or not processed.exists()):
            return None
        d = json.loads(processed.read_text(encoding="utf-8"))
        return Document(**{**d, "sections": [Section(**s) for s in d["sections"]]})

    def _save_processed(self, doc: Document, data: bytes) -> None:
        out = self.processed_dir / f"{doc.doc_id}.json"
        out.write_text(json.dumps(asdict(doc), indent=2, ensure_ascii=False), encoding="utf-8")
        self._index[doc.source] = {"sha256": hashlib.sha256(data).hexdigest(), "parser_version": PARSER_VERSION}

    def _write_index(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")  # write-then-rename: a crash never leaves half a file
        tmp.write_text(json.dumps(self._index, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(self.index_path)
