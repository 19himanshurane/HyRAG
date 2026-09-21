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


def parse_markdown(text: str, title: str) -> list[Section]:
    sections: list[Section] = []
    heading_path: list[tuple[int, str]] = []  # e.g. [(1, "VPN Setup"), (2, "Troubleshooting")]
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


def parse_html(html: str, title: str) -> list[Section]:
    from bs4 import BeautifulSoup, Comment, Doctype, NavigableString

    soup = BeautifulSoup(html, "html.parser")
    for junk in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        junk.decompose()
    if soup.title:
        title = soup.title.get_text(strip=True)

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
            elif child.name in HTML_BLOCKS:
                end_line()
                walk(child)
                end_line()
            else:
                walk(child)

    walk(soup.body or soup)
    close_section()
    return sections


def _remove_repeated_edges(pages: list[list[str]], edge: int = 3) -> list[list[str]]:
    """Drop running headers/footers: lines in the top or bottom `edge` lines that recur on at
    least half the pages. Digits are masked so "Page 1" and "Page 2" count as the same line."""
    if len(pages) < 2:
        return pages

    def mask(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip())

    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({mask(l) for l in lines[:edge] + lines[-edge:] if l.strip()})
    repeated = {l for l, c in counts.items() if c >= max(2, len(pages) / 2)}

    def keep(i: int, line: str, n: int) -> bool:
        at_edge = i < edge or i >= n - edge
        return not (at_edge and mask(line) in repeated)

    return [[l for i, l in enumerate(lines) if keep(i, l, len(lines))] for lines in pages]


def parse_pdf(data: bytes, title: str) -> list[Section]:
    import io

    from pypdf import PdfReader

    pages = [clean(p.extract_text() or "").split("\n") for p in PdfReader(io.BytesIO(data)).pages]
    pages = _remove_repeated_edges(pages)

    sections = []
    heading = title
    for page_number, lines in enumerate(pages, start=1):
        text = clean("\n".join(lines))
        if not text:
            continue
        first_line = text.split("\n", 1)[0].strip()
        # A short line not ending in a full stop looks like a title; otherwise the page probably
        # continues the previous page's topic, so we keep the previous heading.
        looks_like_title = (
            len(first_line) <= 80
            and not first_line[:1].islower()  # "days of the month" is a sentence carrying over
            and not first_line.endswith((".", ",", ";", ":"))
        )
        if looks_like_title:
            heading = first_line
        sections.append(Section(text=text, heading=heading, page=page_number))
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
