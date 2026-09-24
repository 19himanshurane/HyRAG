"""Phase 2 Step 3 audit probes: RRF fusion and HybridRetriever.

Usage (from the repo root): python scripts/audit/probe_retrieval.py
Offline except probe 8 (real index, cached query embeddings -> normally 0 API requests).
"""
import hashlib
import re
import shutil
import statistics
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from hyrag.chunking import Chunk, chunk_structure
from hyrag.index import ChunkIndex, Hit
from hyrag.loader import load_document
from hyrag.retrieval import HybridRetriever, RetrievalConfig, reciprocal_rank_fusion


def report(name: str, ok: bool, evidence: str) -> None:
    print(f"[{'OK     ' if ok else 'FINDING'}] {name}\n          {evidence}")


class FakeEmbedder:
    model = "fake"

    def __init__(self, delay: float = 0.0):
        self.delay = delay

    def embed(self, texts):
        time.sleep(self.delay)  # stands in for the Mistral network round trip
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
        return None, f"{type(e).__name__}: {str(e)[:90]}"


def chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", source="d.md", text=cid, chunk_index=0, heading="", page=None,
                 strategy="structure", char_count=len(cid))


VPN = Path("sample_docs/vpn-setup.md")


def build(tmp: Path, name: str, embedder) -> ChunkIndex:
    idx = ChunkIndex(embedder, data_dir=tmp / name)
    doc = load_document(VPN.name, VPN.read_bytes())
    idx.index_document(doc, chunk_structure(doc))
    return idx


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        idx = build(tmp, "a", FakeEmbedder())
        r = HybridRetriever(idx)

        # 1. A typo in the mode.
        res, err = _try(lambda: r.retrieve("ERR_TUNNEL_4012", mode="Hybrid"))
        report("an unknown mode is rejected", err is not None, f"mode='Hybrid' -> {err or f'{len(res)} results, no error'}")

        # 2. Nonsense configs.
        for cfg in (dict(rrf_k=-61), dict(dense_weight=-1.0), dict(dense_k=0), dict(top_k=0)):
            res, err = _try(lambda: HybridRetriever(idx, RetrievalConfig(**cfg)).retrieve("ERR_TUNNEL_4012 vpn"))
            report(f"config {cfg} is rejected up front", err is not None and err.startswith("ValueError"),
                   f"-> {err or f'accepted, {len(res)} results'}")

        # 3. A chunk listed twice in one list (the fusion function is public).
        fused, err = _try(lambda: reciprocal_rank_fusion({"dense": ([Hit(chunk("a"), 1), Hit(chunk("a"), 1)], 1.0)}))
        report("fused score equals the sum of the recorded contributions", abs(fused[0].score - sum(fused[0].contributions.values())) < 1e-12,
               f"score {fused[0].score:.5f}, contributions {fused[0].contributions}, dense_rank {fused[0].dense_rank}")

        # 4. A list name other than dense/sparse.
        fused, err = _try(lambda: reciprocal_rank_fusion({"bm25": ([Hit(chunk("a"), 3.0)], 1.0)}))
        report("an unknown list name is rejected", err is not None,
               f"-> {err}" if err else f"accepted; dense_rank={fused[0].dense_rank}, sparse_rank={fused[0].sparse_rank}, "
               f"stray attribute bm25_rank={getattr(fused[0], 'bm25_rank', None)}")

        # 5. Empty / filler-only queries.
        for q in ("", "   "):
            res, err = _try(lambda: r.retrieve(q))
            report(f"blank query {q!r} is rejected instead of returning arbitrary chunks", err is not None,
                   f"-> {err or f'{len(res)} results (dense-only: {sum(1 for h in res if h.sparse_rank is None)})'}")
        res = r.retrieve("what is the")
        print(f"[INFO   ] filler-only query 'what is the' -> {len(res)} results, all dense-only "
              f"(known limit: left to the Phase 3 confidence score)")

        # 6. Concurrency: does the index lock cover the embedding network call?
        slow = build(tmp, "b", FakeEmbedder(delay=0.0))
        slow.embedder.delay = 0.3
        sr = HybridRetriever(slow)

        def timed(n_threads, fn):
            ts = [threading.Thread(target=fn) for _ in range(n_threads)]
            t0 = time.perf_counter()
            for t in ts: t.start()
            for t in ts: t.join()
            return time.perf_counter() - t0

        one = timed(1, lambda: sr.retrieve("vpn certificate"))
        eight = timed(8, lambda: sr.retrieve("vpn certificate"))
        report("8 concurrent queries overlap their embedding waits", eight < 3 * one,
               f"1 query {one:.2f}s, 8 concurrent {eight:.2f}s (ideal ~{one:.2f}s, fully serialized ~{8 * one:.2f}s)")
        box = {}
        t = threading.Thread(target=lambda: sr.retrieve("vpn certificate", mode="dense")); t.start(); time.sleep(0.05)
        t0 = time.perf_counter(); slow.search_sparse("ERR_TUNNEL_4012"); box["s"] = time.perf_counter() - t0; t.join()
        report("a keyword search is not blocked by another request's embedding call", box["s"] < 0.1,
               f"keyword search took {box['s'] * 1000:.0f} ms while a dense query waited on the network (alone: ~5 ms)")

        # 7. Missing index: HybridRetriever on a directory that was never built.
        import logging
        seen = []
        handler = logging.Handler(); handler.emit = lambda rec: seen.append(rec.getMessage())
        logging.getLogger("hyrag.retrieval").addHandler(handler)
        empty = HybridRetriever(ChunkIndex(FakeEmbedder(), data_dir=tmp / "never-built"))
        res, err = _try(lambda: empty.retrieve("ERR_TUNNEL_4012"))
        report("searching an unbuilt index is distinguishable from 'no match'", bool(seen) or err is not None,
               f"-> {err or f'{len(res)} results'}; warning: {seen[0] if seen else 'none'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 8. Real index: latency and how often sparse-only junk reaches the fused top 5.
    if not Path("data/index/chunks-structure.sqlite").exists():
        print("[SKIP   ] real index not built (run try_index.py)"); return
    from hyrag.embeddings import MistralEmbedder
    qs = ["How do I fix ERR_TUNNEL_4012?", "What does SQLSTATE 23505 mean?",
          "I forgot my login credentials, what can I do?", "Can I deploy to production on a Friday?"]
    with MistralEmbedder() as emb:
        real = HybridRetriever(ChunkIndex(emb))
        for mode in ("sparse", "dense", "hybrid"):
            times = []
            for _ in range(5):
                for q in qs:
                    t0 = time.perf_counter(); real.retrieve(q, mode=mode); times.append(time.perf_counter() - t0)
            report(f"{mode} retrieval latency (cached embeddings)", True,
                   f"median {statistics.median(times) * 1000:.1f} ms, max {max(times) * 1000:.1f} ms")
        for q in qs:
            top = real.retrieve(q)[:5]
            only_sparse = [h for h in top if h.dense_rank is None]
            print(f"          top-5 sparse-only for {q[:35]!r}: {len(only_sparse)} "
                  f"(BM25 scores {[round(h.sparse_score, 1) for h in only_sparse]})")
        print(f"          API requests: {emb.requests_made}")


if __name__ == "__main__":
    main()
