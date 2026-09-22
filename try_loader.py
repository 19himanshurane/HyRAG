import logging
from pathlib import Path

from hyrag.loader import DocumentStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
ROOT = Path("sample_docs")
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
