from pathlib import Path

from hyrag.loader import DocumentStore

store = DocumentStore()
for path in sorted(Path("sample_docs").iterdir()):
    if path.suffix == ".py":
        continue
    doc = store.ingest_file(path)
    print(f"\n=== {doc.source}  (id={doc.doc_id}, type={doc.file_type}, {len(doc.sections)} sections)")
    for s in doc.sections:
        page = f"  page={s.page}" if s.page else ""
        print(f"  [heading: {s.heading}]{page}")
        print("   " + s.text.replace("\n", "\n   "))
