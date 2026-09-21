import hashlib
import json
import re
import shutil
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
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_doc_id(source: str) -> str:
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


def parse_html(html: str, title: str) -> list[Section]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for junk in soup(["script", "style", "nav", "footer", "header"]):
        junk.decompose()
    if soup.title:
        title = soup.title.get_text(strip=True)

    sections: list[Section] = []
    heading_path: list[tuple[int, str]] = []
    lines_in_section: list[str] = []

    def close_section():
        body = clean("\n".join(lines_in_section))
        if body:
            heading = " > ".join(h for _, h in heading_path) or title
            sections.append(Section(text=body, heading=heading))
        lines_in_section.clear()

    def inline_text(el) -> str:
        # get_text() with no separator joins "Run " + "nimbus-deploy" + ". " exactly as written;
        # split/join then collapses the source file's line breaks and indentation into single spaces.
        return " ".join(el.get_text().split())

    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "td"]):
        if el.name.startswith("h"):
            close_section()
            level = int(el.name[1])
            while heading_path and heading_path[-1][0] >= level:
                heading_path.pop()
            heading_path.append((level, inline_text(el)))
        elif el.name == "pre":
            lines_in_section.append(el.get_text())  # keep code line breaks
        else:
            lines_in_section.append(inline_text(el))
    close_section()
    return sections


def parse_pdf(data: bytes, title: str) -> list[Section]:
    import io

    from pypdf import PdfReader

    sections = []
    for page_number, page in enumerate(PdfReader(io.BytesIO(data)).pages, start=1):
        text = clean(page.extract_text() or "")
        if text:
            first_line = text.split("\n", 1)[0]
            heading = first_line if len(first_line) <= 80 else title
            sections.append(Section(text=text, heading=heading, page=page_number))
    return sections


# ---------- the single entry point: any file in, a Document out ----------

def load_document(filename: str, data: bytes) -> Document:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type {ext!r}; supported: {sorted(SUPPORTED)}")
    title = Path(filename).stem

    if ext == ".pdf":
        sections = parse_pdf(data, title)
    else:
        text = data.decode("utf-8", errors="replace")
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

    def ingest_file(self, path: Path) -> Document:
        doc = load_document(path.name, path.read_bytes())
        shutil.copyfile(path, self.raw_dir / path.name)
        self._save_processed(doc)
        return doc

    def _save_processed(self, doc: Document) -> None:
        out = self.processed_dir / f"{doc.doc_id}.json"
        out.write_text(json.dumps(asdict(doc), indent=2, ensure_ascii=False), encoding="utf-8")

    def reprocess_all(self) -> list[Document]:
        """Rebuild every processed file from the saved raw originals, with no re-upload needed."""
        docs = []
        for raw in sorted(self.raw_dir.iterdir()):
            doc = load_document(raw.name, raw.read_bytes())
            self._save_processed(doc)
            docs.append(doc)
        return docs
