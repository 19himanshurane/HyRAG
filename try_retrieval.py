"""Hybrid retrieval on the real corpus: meaning search + keyword search merged with Reciprocal Rank Fusion.

Usage: python try_index.py first (builds data/index), then python try_retrieval.py
Only the query embeddings touch the API (and they are cached after the first run).
"""
import re

from hyrag.embeddings import MistralEmbedder
from hyrag.index import ChunkIndex
from hyrag.retrieval import HybridRetriever, RetrievalConfig

# question -> a text snippet that only the chunk we want contains (so we can report where it lands)
QUESTIONS = {
    "How do I fix ERR_TUNNEL_4012?": "ERR_TUNNEL_4012",
    "What does SQLSTATE 23505 mean?": "23505",
    "I forgot my login credentials, what can I do?": None,
    "Can I deploy to production on a Friday?": None,
}


def label(h) -> str:
    where = h.chunk.source.split("/")[-1][:28]
    return f"{where} [{h.chunk.heading[-50:]}]"


def rank_of(hits, needle: str) -> str:
    """Best rank of any chunk containing `needle` as a whole token (so 4012 doesn't match inside 40123)."""
    pattern = re.compile(rf"(?<![\w]){re.escape(needle)}(?![\w])")
    for i, h in enumerate(hits, 1):
        if pattern.search(h.chunk.text_for_search()):
            return str(i)
    return "-"


def main() -> None:
    with MistralEmbedder() as embedder:
        index = ChunkIndex(embedder)
        if index.count() == 0:
            raise SystemExit("data/index is empty: run python try_index.py first")
        print(f"{index.count()} chunks, in sync: {index.check_sync()}")
        for weights in [(0.7, 0.3), (0.5, 0.5)]:
            retriever = HybridRetriever(index, RetrievalConfig(dense_weight=weights[0], sparse_weight=weights[1]))
            print(f"\n=============== weights dense {weights[0]} / sparse {weights[1]} ===============")
            for q, needle in QUESTIONS.items():
                print(f"\nQ: {q}")
                results = {m: retriever.retrieve(q, mode=m) for m in ("dense", "sparse", "hybrid")}
                if needle:
                    print("   target chunk's rank ->  " + ",  ".join(
                        f"{m}: {rank_of(r, needle)}" for m, r in results.items()))
                for h in results["hybrid"][:5]:
                    d = f"d#{h.dense_rank} ({h.dense_score:.3f})" if h.dense_rank else "d#-         "
                    s = f"s#{h.sparse_rank} ({h.sparse_score:.1f})" if h.sparse_rank else "s#-"
                    print(f"   {h.score:.5f}  {d:15} {s:13} {label(h)}")
        print(f"\nAPI requests this run: {embedder.requests_made}")


if __name__ == "__main__":
    main()
