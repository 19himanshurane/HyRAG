"""Round-2 audit probes for code added after round 1 (semantic chunking, cache, ingest_many, limits).

Usage (from the repo root): python scripts/audit/probe_round2.py
Offline: uses a deterministic fake embedder and a temp directory; no Mistral calls.
Everything runs under main(): ingest_many starts worker processes, which re-import this file on Windows.
"""
import hashlib
import os
import pickle
import re
import shutil
import tempfile
import threading
import time
import tracemalloc
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from hyrag.chunking import chunk_semantic, semantic_units
from hyrag.loader import MAX_FILE_BYTES, Document, DocumentStore, Section, list_corpus_files, load_document


def report(name: str, ok: bool, evidence: str) -> None:
    print(f"[{'OK     ' if ok else 'FINDING'}] {name}\n          {evidence}")


class FakeEmbedder:
    def embed(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]{4,}", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
            out[i, 0] += 1e-3
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def _try(fn):
    try:
        fn()
        return None
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:90]}"


def probe_semantic_scaling() -> None:
    """1. Semantic chunking cost on one big section (tail lengths are recomputed per unit)."""
    para = "Incident responders document every containment step and share lessons with the owning team."
    for n_units in (500, 2000):
        doc = Document("d", "big.txt", ".txt",
                       [Section(text="\n\n".join(f"{para} Step {i}." for i in range(n_units)), heading="Big")])
        t0 = time.perf_counter()
        chunks = chunk_semantic(doc, FakeEmbedder())
        t = time.perf_counter() - t0
        report(f"semantic chunking of one {n_units}-paragraph section", t < 2,
               f"{len(chunks)} chunks in {t:.2f}s (fake embedder: time is chunking logic only)")


def probe_ingest_many_ipc() -> None:
    """2. What ingest_many actually sends to worker processes (every submit() is recorded and pickled)."""
    from concurrent.futures import ProcessPoolExecutor
    sent: list[int] = []
    real_submit = ProcessPoolExecutor.submit

    def recording_submit(self, fn, *args, **kwargs):
        sent.append(len(pickle.dumps((fn, args, kwargs))))
        return real_submit(self, fn, *args, **kwargs)

    ProcessPoolExecutor.submit = recording_submit
    tmp = Path(tempfile.mkdtemp())
    try:
        nist = Path("corpus/security/nist-sp-800-63b-4-authentication.pdf")
        DocumentStore(tmp / "d").ingest_many([nist], root=Path("corpus"), workers=4)
    finally:
        ProcessPoolExecutor.submit = real_submit
        shutil.rmtree(tmp, ignore_errors=True)
    size = nist.stat().st_size
    report("page-range tasks don't copy the whole PDF each", sum(sent) < size,
           f"NIST 800-63B ({size / 1e6:.1f} MB): {len(sent)} tasks, {sum(sent) / 1e3:,.1f} kB pickled to workers "
           f"in total (round-2 measurement before the fix: 17 tasks x 1.0 MB = 17 MB)")


def probe_size_limit_and_cache_threads() -> None:
    """3. The size limit vs. reading the file; 4. the embedding cache from another thread."""
    tmp = Path(tempfile.mkdtemp())
    try:
        big = tmp / "src" / "huge.txt"
        big.parent.mkdir()
        big.write_bytes(b"a" * (MAX_FILE_BYTES + 20 * 2**20))  # 20 MB over the limit
        store = DocumentStore(tmp / "data")
        tracemalloc.start()
        rejected = _try(lambda: store.ingest_file(big, root=tmp / "src")) is not None
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report("an oversized file is rejected before it is read", peak < 5e6,
               f"rejected={rejected}; peak memory while rejecting a {big.stat().st_size / 1e6:.0f} MB file: "
               f"{peak / 1e6:.1f} MB")

        os.environ.setdefault("MISTRAL_API_KEY", "probe-not-used")
        from hyrag.embeddings import MistralEmbedder
        emb = MistralEmbedder(cache_path=tmp / "cache.sqlite")
        errors: list = []
        th = threading.Thread(target=lambda: errors.append(_try(lambda: emb.cache.get_many(["k"]))))
        th.start()
        th.join()
        report("embedding cache works from another thread", errors == [None], f"result from worker thread: {errors[0]}")
        emb.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def probe_inline_code_comment() -> None:
    """5. HTML-comment stripping vs a comment written inside inline code."""
    doc = load_document("c.md", b"# T\n\nWrite `<!-- note -->` to add a comment in Markdown.\n")
    report("inline code containing <!-- is kept", "<!-- note -->" in doc.sections[0].text,
           f"text: {doc.sections[0].text!r}")


def probe_unicode() -> None:
    """6. Characters a typed query won't match exactly (input to the Step 3 tokenizer)."""
    suspects = {"ﬁ": "fi ligature", "ﬂ": "fl ligature", "­": "soft hyphen", " ": "no-break space",
                "​": "zero-width space", "’": "curly apostrophe", "“": "curly quote", "–": "en dash"}
    dangerous = {"ﬁ", "ﬂ", "­", "​"}
    counts: Counter = Counter()
    for root in (Path("corpus"), Path("sample_docs")):
        for p in list_corpus_files(root):
            doc = load_document(p.relative_to(root).as_posix(), p.read_bytes())
            counts.update(ch for s in doc.sections for ch in s.text + s.heading if ch in suspects)
    nfkc = sum(n for ch, n in counts.items() if unicodedata.normalize("NFKC", ch) != ch)
    report("no ligatures / soft hyphens / zero-width characters (typography is for the Step 3 tokenizer)",
           not (dangerous & set(counts)),
           ", ".join(f"{suspects[ch]} x{n}" for ch, n in counts.most_common()) + f"  (NFKC would rewrite {nfkc})")


def probe_semantic_cost() -> None:
    """7. Semantic chunking embeds units AND (in Step 3) the chunks."""
    k8s = load_document("k8s-pod-lifecycle.md", Path("corpus/engineering/k8s-pod-lifecycle.md").read_bytes())
    unit_chars = sum(len(u) for s in k8s.sections for u in semantic_units(s.text, k8s.file_type, 800))
    chunk_chars = sum(len(c.text_for_search()) for c in chunk_semantic(k8s, FakeEmbedder()))
    report("semantic chunking does not double embedding cost [open question G8]", unit_chars < 0.2 * chunk_chars,
           f"k8s-pod-lifecycle: {unit_chars:,} chars embedded to find cuts + {chunk_chars:,} to index the chunks "
           f"= {(unit_chars + chunk_chars) / chunk_chars:.1f}x the other strategies' cost")


if __name__ == "__main__":
    probe_semantic_scaling()
    probe_ingest_many_ipc()
    probe_size_limit_and_cache_threads()
    probe_inline_code_comment()
    probe_unicode()
    probe_semantic_cost()
