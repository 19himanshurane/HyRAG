"""Phase 2 audit S6: does meaning search return the TRUE top 10? Compares ChunkIndex.search_dense against
brute-force cosine over every vector stored in Chroma, and reports latency.

Usage (from the repo root): PYTHONPATH=. python scripts/audit/probe_recall.py
Needs the real index (python try_index.py). Query embeddings are cached after the first run.
Run it several times: approximate indexes can miss different chunks in different processes.
"""
import time

import numpy as np

from hyrag.embeddings import MistralEmbedder
from hyrag.index import ChunkIndex

QUESTIONS = ["How do I fix ERR_TUNNEL_4012?", "What does SQLSTATE 23505 mean?",
             "I forgot my login credentials, what can I do?", "Can I deploy to production on a Friday?",
             "Why is my pod stuck in Pending?", "What happens when a pod cannot be placed on any machine?"]


def main() -> None:
    with MistralEmbedder() as embedder:
        index = ChunkIndex(embedder)
        got = index.collection.get(include=["embeddings"])
        ids = got["ids"]
        matrix = np.asarray(got["embeddings"], dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        misses, times = [], []
        for q in QUESTIONS:
            qv = embedder.embed([q])[0]
            exact = {ids[i] for i in np.argsort(-(matrix @ qv))[:10]}
            t0 = time.perf_counter()
            found = {h.chunk.chunk_id for h in index.search_dense(q, k=10)}
            times.append(time.perf_counter() - t0)
            misses.append(len(exact - found))
        ok = not any(misses)
        print(f"[{'OK     ' if ok else 'FINDING'}] meaning search returns the exact top 10 ({len(ids):,} vectors)\n"
              f"          missed per question: {misses}; latency median {sorted(times)[len(times) // 2] * 1000:.1f} ms "
              f"(first includes building the in-memory copy: {times[0] * 1000:.0f} ms); API requests {embedder.requests_made}")


if __name__ == "__main__":
    main()
