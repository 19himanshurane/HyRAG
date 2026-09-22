import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED = {".md", ".txt", ".html", ".htm", ".pdf"}


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
        return data.decode("utf-8-sig")  # "-sig" also removes the invisible BOM Windows editors add
    except UnicodeDecodeError:
        # Not valid UTF-8: most likely an older Windows/Word export. Best-effort guess.
        return data.decode("cp1252", errors="replace")


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
    inside_code_block = False

    def close_section():
        body = clean("\n".join(lines_in_section))
        if body:
            heading = " > ".join(h for _, h in heading_path) or title
            sections.append(Section(text=body, heading=heading))
        lines_in_section.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            inside_code_block = not inside_code_block
            continue  # drop the fence line itself, keep the code inside it
        if inside_code_block:
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

    def walk(node):
        # Visit every node exactly once, so text nested like <li><p>x</p></li> is never read twice,
        # and text sitting directly inside a <div> is never skipped.
        for child in node.children:
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
                walk(child)
                end_line()
            else:
                walk(child)

    walk(soup.body or soup)
    close_section()
    return sections


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
    for page_number, page in enumerate(pdf.pages, start=1):
        for ln in page.extract_text_lines(return_chars=True):
            chars = [c for c in ln["chars"] if c["text"].strip()]
            if not chars or _BULLET_ONLY.match(ln["text"].strip()):
                continue
            segments, last = [[]], None
            for c in ln["chars"]:
                if c["text"].strip():
                    if last is not None and c["x0"] - last["x1"] > 3 * c["size"]:
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
    return lines


def _drop_running_headers(lines: list[PdfLine], page_count: int) -> list[PdfLine]:
    """A running header/footer is text in the top or bottom 10% of the page that recurs on at least
    half the pages. Digits are masked so "Page 7" and "Page 8" count as the same line."""
    if page_count < 2:
        return lines

    def in_margin(l: PdfLine) -> bool:
        return l.top < 0.1 * l.page_height or l.bottom > 0.9 * l.page_height

    def mask(text: str) -> str:
        return _ROMAN_PAGE.sub("#", re.sub(r"\d+", "#", text))  # "Page 7", "7" and "vii" all -> "#"

    pages_with = Counter()
    for page in {l.page for l in lines}:
        pages_with.update({mask(l.text) for l in lines if l.page == page and in_margin(l)})
    repeated = {t for t, n in pages_with.items() if n >= max(2, page_count / 2)}
    return [l for l in lines if not (in_margin(l) and mask(l.text) in repeated)]


def _heading_level(line: PdfLine, body_size: float, section_depth: int) -> int | None:
    """Return the heading level for this line, or None if it's body text.
    Rules come from studying the NIST PDFs: headings are fully bold and at least body size;
    many are the SAME size as body text, so boldness + numbering matter more than size."""
    if not line.bold or line.size < body_size - 0.5:
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
        if words > 20:
            return None
        return numbered.group(1).count(".") + 1 if numbered.group(1) else 1  # "3.1.2" -> 3, "Appendix B" -> 1
    if words > 8 or line.text.endswith((".", ",", ";")):
        return None  # a bold sentence for emphasis, not a title
    if line.size > body_size * 1.15 or line.text.lower() in _DOC_PARTS:
        return 1  # visibly larger, or a standard part like "References": top-level
    return section_depth + 1  # e.g. a glossary term or "Note" title inside the current numbered section


def _usable_pdf_title(meta_title: str | None) -> str | None:
    """PDF metadata titles are often junk ("Microsoft Word - draft3.docx", "untitled")."""
    t = (meta_title or "").strip()
    if not t or t.lower() in {"untitled", "title"} or re.search(r"\.(docx?|pdf|pptx?|indd)$|^microsoft ", t, re.I):
        return None
    return t


def parse_pdf(data: bytes, title: str) -> list[Section]:
    import io
    import statistics

    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        lines = _read_pdf_lines(pdf)
        page_count = len(pdf.pages)
        title = _usable_pdf_title((pdf.metadata or {}).get("Title")) or title
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
    paragraph_gap = max(1.5 * normal_gap, normal_gap + 0.3 * body_size)

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
                and (prev.x1 >= 0.9 * right_edge or line.text[:1].islower())
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

    if not sections:
        raise ValueError(f"No text could be extracted from {filename!r}")
    return Document(doc_id=make_doc_id(filename), source=filename, file_type=ext, sections=sections)


# ---------- storage: keep raw originals + processed JSON side by side ----------

class DocumentStore:
    def __init__(self, data_dir: Path = Path("data")):
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def ingest_file(self, path: Path, root: Path | None = None) -> Document:
        """`root` is the folder the corpus lives in; the file's path inside it becomes its identity."""
        source = path.relative_to(root).as_posix() if root else path.name
        doc = load_document(source, path.read_bytes())  # parse first: a broken file is never stored
        raw_copy = self.raw_dir / source
        raw_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, raw_copy)
        self._save_processed(doc)
        return doc

    def _save_processed(self, doc: Document) -> None:
        out = self.processed_dir / f"{doc.doc_id}.json"
        out.write_text(json.dumps(asdict(doc), indent=2, ensure_ascii=False), encoding="utf-8")

    def reprocess_all(self) -> list[Document]:
        """Rebuild every processed file from the saved raw originals, with no re-upload needed."""
        docs = []
        for raw in sorted(p for p in self.raw_dir.rglob("*") if p.is_file()):
            doc = load_document(raw.relative_to(self.raw_dir).as_posix(), raw.read_bytes())
            self._save_processed(doc)
            docs.append(doc)
        return docs
