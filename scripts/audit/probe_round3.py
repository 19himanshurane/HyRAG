"""Round-3 audit probes: the index (ChromaDB + BM25 + sync) and near-duplicate detection.

Usage (from the repo root): python scripts/audit/probe_round3.py
Offline: deterministic fake embedder, throwaway directories, no Mistral calls.
"""
import hashlib
import random
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from hyrag.chunking import chunk_structure
from hyrag.dedup import NearDuplicateIndex
from hyrag.index import ChunkIndex
from hyrag.loader import DocumentStore, list_corpus_files, load_document


def report(name: str, ok: bool, evidence: str) -> None:
    print(f"[{'OK     ' if ok else 'FINDING'}] {name}\n          {evidence}")


class FakeEmbedder:
    def __init__(self, model: str = "fake-a"):
        self.model = model

    def embed(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]{4,}", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
            out[i, 0] += 1e-3
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def _try(fn):
    try:
        return fn(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:100]}"


VPN = Path("sample_docs/vpn-setup.md")


def fresh_index(tmp: Path, name: str, embedder=None) -> ChunkIndex:
    idx = ChunkIndex(embedder or FakeEmbedder(), data_dir=tmp / name)
    doc = load_document(VPN.name, VPN.read_bytes())
    idx.index_document(doc, chunk_structure(doc))
    return idx


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        # 1. Searching while Chroma holds an id the table doesn't (the crash state check_sync detects).
        idx = fresh_index(tmp, "p1")
        idx.collection.add(ids=["ghost"], embeddings=[[1.0] + [0.0] * 63], documents=["stale"])
        _, err = _try(lambda: idx.search_dense("ERR_TUNNEL_4012 certificate", k=10))
        report("dense search survives an out-of-sync Chroma", err is None, f"error: {err}")

        # 2. Searching while Chroma is MISSING chunks the table has.
        idx = fresh_index(tmp, "p2")
        some = idx.collection.get(include=[])["ids"][:3]
        idx.collection.delete(ids=some)
        hits, err = _try(lambda: idx.search_dense("vpn", k=10))
        report("dense search survives Chroma missing chunks", err is None,
               f"error: {err}" if err else f"{len(hits)} hits returned for k=10 with {idx.collection.count()} vectors")

        # 3. The index used from another thread (a web server's worker pool).
        idx = fresh_index(tmp, "p3")
        box = []
        t = threading.Thread(target=lambda: box.append(_try(lambda: idx.search_sparse("ERR_TUNNEL_4012"))))
        t.start(); t.join()
        report("index works from another thread", box[0][1] is None, f"error: {box[0][1]}")

        # 4. Two ChunkIndex objects on the same directory: does the second see the first one's writes?
        a = fresh_index(tmp, "p4")
        b = ChunkIndex(FakeEmbedder(), data_dir=tmp / "p4")
        b.search_sparse("vpn")  # b builds its BM25 snapshot now
        extra = load_document("extra.md", b"# Extra\n\nThe quarterly zebrafish audit happens every March.")
        a.index_document(extra, chunk_structure(extra))
        seen = [h.chunk.source for h in b.search_sparse("zebrafish audit")]
        a.remove_document(load_document(VPN.name, VPN.read_bytes()).doc_id)
        _, err = _try(lambda: b.search_sparse("ERR_TUNNEL_4012"))
        report("a second index object stays correct after another one writes", "extra.md" in seen and err is None,
               f"second object found new doc: {'extra.md' in seen}; after the first deleted a doc, "
               f"second object's search: {err or 'ok'}")

        # 5. Is the embedding model recorded, so vectors from two models can't be mixed silently?
        idx = fresh_index(tmp, "p5", FakeEmbedder("model-a"))
        extra2 = load_document("x.md", b"# X\n\nA paragraph embedded by a different model entirely.")

        def reopen_and_add():  # same dimensions, different model
            other = ChunkIndex(FakeEmbedder("model-b"), data_dir=tmp / "p5")
            return other.index_document(extra2, chunk_structure(extra2))

        _, err = _try(reopen_and_add)
        report("an index refuses vectors from a different embedding model", err is not None,
               f"collection metadata: {idx.collection.metadata}; mixing models: {err or 'accepted silently'}")

        # 6. Deleting a document through the supported entry point: does it leave the search index?
        #    (Round 3 found no entry point linking the two: DocumentStore.delete alone left it searchable.
        #    hyrag.pipeline.Pipeline is now the way documents enter and leave.)
        from hyrag.pipeline import Pipeline
        pipe = Pipeline(DocumentStore(tmp / "store"), ChunkIndex(FakeEmbedder(), data_dir=tmp / "p6"))
        pipe.ingest([VPN], root=VPN.parent, workers=1)
        before = [h.chunk.source for h in pipe.index.search_sparse("ERR_TUNNEL_4012")]
        pipe.delete(VPN.name)
        still = [h.chunk.source for h in pipe.index.search_sparse("ERR_TUNNEL_4012")]
        report("a deleted document stops being searchable", bool(before) and not still,
               f"before delete: {before[:1]}; after Pipeline.delete: {still[:1]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 7. Near-duplicate lookup cost as the index grows (corpus chunks, replicated with word noise).
    base = []
    for root in (Path("corpus"), Path("sample_docs")):
        for p in list_corpus_files(root):
            base += [c.text for c in chunk_structure(load_document(p.relative_to(root).as_posix(), p.read_bytes()))]
    rng = random.Random(0)
    for copies in (1, 8):
        texts = [t if k == 0 else " ".join(w for w in t.split() if rng.random() > 0.3) + f" v{k}"
                 for k in range(copies) for t in base]
        near = NearDuplicateIndex()
        t0 = time.perf_counter()
        for i, t in enumerate(texts):
            near.find(t)
            near.add(str(i), t)
        per = (time.perf_counter() - t0) / len(texts)
        report(f"dedup check+add over {len(texts):,} chunks", per < 0.01,
               f"{per * 1000:.2f} ms per chunk, {per * len(texts):.1f} s total")

    # 8. Stale documentation: the README must not hard-code a test count (it went stale in rounds 2 and 3).
    readme = Path("README.md").read_text(encoding="utf-8")
    claimed = re.search(r"(\d+) (?:offline )?tests", readme)
    real = sum(len(re.findall(r"^def test_", p.read_text(encoding="utf-8"), re.M)) for p in Path("tests").glob("test_*.py"))
    report("README has no hard-coded test count to go stale", claimed is None,
           f"README count: {claimed.group(1) if claimed else 'none'}; test functions now: {real}")


if __name__ == "__main__":
    main()
