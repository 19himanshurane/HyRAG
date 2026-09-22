"""Correctness / robustness probes for Phase 1. Each prints OK or FINDING with the measured evidence.

Usage (from the repo root): python scripts/audit/probe_robustness.py
Runs offline; writes only to a temporary directory.
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from hyrag.chunking import chunk_fixed, chunk_structure, split_recursive
from hyrag.loader import DocumentStore, load_document


def report(name: str, ok: bool, evidence: str) -> None:
    print(f"[{'OK     ' if ok else 'FINDING'}] {name}\n          {evidence}")


# 1. Windows line endings. core.autocrlf=true means a fresh Windows clone gets CRLF markdown.
raw = Path("corpus/engineering/k8s-secrets.md").read_bytes()
lf = load_document("k8s-secrets.md", raw)
crlf = load_document("k8s-secrets.md", raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
report("CRLF markdown parses like LF",
       lf.sections[0].heading == crlf.sections[0].heading and "content_type:" not in crlf.sections[0].text,
       f"LF first heading={lf.sections[0].heading!r} | CRLF first heading={crlf.sections[0].heading!r}, "
       f"front matter leaked into text: {'content_type:' in crlf.sections[0].text}")

# 2. Deeply nested HTML (auto-generated pages, or hostile input) vs. the recursive walk().
for depth in (500, 1200):
    html = "<html><body>" + "<div>" * depth + "deep text" + "</div>" * depth + "</body></html>"
    try:
        d = load_document("nested.html", html.encode())
        report(f"HTML nested {depth} deep", "deep text" in d.sections[0].text, "parsed")
    except RecursionError as e:
        report(f"HTML nested {depth} deep", False, f"RecursionError: {e}")

# 3. '~~~' code fences: a shell comment inside must not become a heading.
md = "# Guide\n\ntext\n\n~~~bash\n# install the agent\napt install x\n~~~\n\nmore text\n"
heads = [s.heading for s in load_document("f.md", md.encode()).sections]
report("~~~ fenced code stays code", not any("install the agent" in h for h in heads), f"headings={heads}")

# 4. One corrupt file in a batch re-ingest.
tmp = Path(tempfile.mkdtemp())
try:
    store = DocumentStore(tmp / "data")
    for p in Path("sample_docs").iterdir():
        if p.suffix != ".py":
            store.ingest_file(p, root=Path("sample_docs"))
    (store.raw_dir / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    try:
        docs = store.reprocess_all()
        report("reprocess_all isolates a bad file", True,
               f"{len(docs)} docs reprocessed, failures recorded: {getattr(store, 'failures', {})}")
    except Exception as e:
        report("reprocess_all isolates a bad file", False,
               f"whole batch aborted on broken.pdf: {type(e).__name__}: {str(e)[:80]}")
    (store.raw_dir / "broken.pdf").unlink()

    # 5. Rename / delete a document: are stale processed files left behind?
    before = {p.name for p in store.processed_dir.iterdir()}
    (store.raw_dir / "leave-policy.txt").rename(store.raw_dir / "leave-policy-2026.txt")
    (store.raw_dir / "deploy-guide.html").unlink()
    store.reprocess_all()
    after = {p.name for p in store.processed_dir.iterdir()}
    live = {json.loads(p.read_text(encoding="utf-8"))["source"] for p in store.processed_dir.iterdir()}
    report("rename/delete leaves no orphans", len(after) == len(list(store.raw_dir.rglob("*.*"))),
           f"raw files={len(list(store.raw_dir.rglob('*.*')))}, processed files={len(after)} "
           f"(was {len(before)}); sources in processed/: {sorted(live)}")

    # 6. Same file name in two folders. (Originally probed with the old default root=None, which
    #    collided; `root` is now required, so this checks the only remaining way to call it.)
    a, b = tmp / "src" / "hr" / "policy.md", tmp / "src" / "it" / "policy.md"
    a.parent.mkdir(parents=True); b.parent.mkdir(parents=True)
    a.write_text("# HR policy\n\n24 vacation days.", encoding="utf-8")
    b.write_text("# IT policy\n\nRotate passwords every 180 days.", encoding="utf-8")
    try:
        da, db = store.ingest_file(a, root=tmp / "src"), store.ingest_file(b, root=tmp / "src")
        report("same name, different folders", da.doc_id != db.doc_id,
               f"doc_id {da.doc_id} vs {db.doc_id}; hr copy starts "
               f"{(store.raw_dir / 'hr' / 'policy.md').read_text(encoding='utf-8').splitlines()[0]!r}")
    except TypeError as e:
        report("same name, different folders", False, f"old signature: {e}")

    # 7. Re-ingesting an unchanged file: is any work skipped?
    big = Path("corpus/security/nist-sp-800-61r3-incident-response.pdf")
    t0 = time.perf_counter(); store.ingest_file(big, root=Path("corpus")); t1 = time.perf_counter()
    store.ingest_file(big, root=Path("corpus")); t2 = time.perf_counter()
    report("unchanged file is not re-parsed", (t2 - t1) < 0.2 * (t1 - t0),
           f"first ingest {t1 - t0:.1f}s, identical re-ingest {t2 - t1:.1f}s")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# 8. Chunk id stability when a paragraph (big enough to add one chunk) is inserted near the top.
base = Path("corpus/engineering/k8s-secrets.md").read_text(encoding="utf-8")
para = "An editor added this paragraph near the top of the page. " * 16  # ~930 chars -> one extra chunk
edited = base.replace("## Uses for Secrets", "## Uses for Secrets\n\n" + para + "\n", 1)
c1 = {c.chunk_id: c.text for c in chunk_structure(load_document("k8s-secrets.md", base.encode()))}
c2 = {c.chunk_id: c.text for c in chunk_structure(load_document("k8s-secrets.md", edited.encode()))}
same_id_diff_text = sum(1 for k in c1 if k in c2 and c1[k] != c2[k])
unchanged = len(set(c1.values()) & set(c2.values()))
report("a one-paragraph edit keeps chunk ids meaningful", same_id_diff_text <= 2,
       f"{same_id_diff_text} of {len(c1)} chunk ids now point to DIFFERENT text, "
       f"while {unchanged} of {len(c1)} chunk texts are unchanged")

# 9. Determinism: same input -> identical output.
pdf = Path("corpus/security/nist-sp-800-61r3-incident-response.pdf").read_bytes()
r1, r2 = load_document("x.pdf", pdf), load_document("x.pdf", pdf)
report("parsing is deterministic", r1 == r2, f"{len(r1.sections)} sections, identical={r1 == r2}")

# 10. Chunker cost on one huge unstructured section (e.g. a 2 MB log pasted into a .txt).
words = ("lorem ipsum dolor sit amet " * 80000)  # ~2.2 MB, spaces only: forces word-level splitting
t0 = time.perf_counter(); n = len(split_recursive(words, 800, 120)); t = time.perf_counter() - t0
report("2.2 MB unstructured text chunks in reasonable time", t < 5, f"{n} chunks in {t:.2f}s")
t0 = time.perf_counter(); n = len(chunk_fixed(load_document("big.txt", words.encode()))); t = time.perf_counter() - t0
report("fixed chunking of the same text", t < 5, f"{n} chunks in {t:.2f}s")

# 11. Non-document files inside the corpus folder are ingested as content.
try:
    from hyrag.loader import list_corpus_files
    listed = {p.as_posix() for p in list_corpus_files(Path("corpus"))}
except ImportError:  # before the fix: everything with a supported extension was ingested
    listed = {p.as_posix() for p in Path("corpus").rglob("*") if p.suffix in {".md", ".html", ".pdf", ".txt"}}
meta = sorted(p for p in listed if p.endswith("README.md"))
report("corpus metadata files are not ingested", not meta,
       f"{len(listed)} documents listed; metadata among them: {meta}")
sys.exit(0)
