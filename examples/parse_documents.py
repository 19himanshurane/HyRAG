"""Parse the sample documents (Markdown, text, HTML, PDF) and print the sections the loader extracts.

Usage (from the repo root): python -m examples.parse_documents
Originals are saved to data/raw/ and the parsed versions to data/processed/. No API key needed.
"""
import logging
from pathlib import Path

from hyrag.loader import DocumentStore

ROOT = Path("sample_docs")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    store = DocumentStore()
    for path in sorted(ROOT.iterdir()):
        if path.suffix == ".py":
            continue
        doc = store.ingest_file(path, root=ROOT)
        print(f"\n=== {doc.source}  (id={doc.doc_id}, type={doc.file_type}, {len(doc.sections)} sections)")
        for s in doc.sections:
            page = f"  page={s.page}" if s.page else ""
            print(f"  [heading: {s.heading}]{page}")
            print("   " + s.text.replace("\n", "\n   "))


if __name__ == "__main__":
    main()
