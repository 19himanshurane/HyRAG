"""Hybrid search with and without the reranker: where the right passage ranks before and after.

Usage (from the repo root): python -m examples.hybrid_search      (needs an index: python -m hyrag.seed)
The first run downloads the reranker model (~88 MB) into the Hugging Face cache; after that it loads offline.
"""
import re
import time

from hyrag.embeddings import MistralEmbedder
from hyrag.index import ChunkIndex
from hyrag.rerank import CrossEncoderScorer, download, is_downloaded
from hyrag.retrieval import HybridRetriever

# question -> (field to check, pattern that only the passage holding the answer matches). The heading is
# checked where the body text would also match a table of contents that just lists the section.
QUESTIONS = {
    "How do I fix ERR_TUNNEL_4012?": ("heading", r"Error ERR_TUNNEL_4012$"),
    "What does SQLSTATE 23505 mean?": ("text", r"(?<!\w)23505(?!\w)"),
    "I forgot my login credentials, what can I do?": ("heading", r"Account Recovery"),
    "Can I deploy to production on a Friday?": ("text", r"No deploys on Fridays"),
    "Why is my pod stuck in Pending?": ("heading", r"My pod stays pending$"),
    "What happens when a pod cannot be placed on any machine?": ("heading", r"My pod stays pending$"),  # paraphrase
}


def is_answer(hit, target) -> bool:
    field, pattern = target
    return bool(re.search(pattern, getattr(hit.chunk, field)))


def rank_of(hits, target) -> str:
    return next((str(i) for i, h in enumerate(hits, 1) if is_answer(h, target)), "-")


def main() -> None:
    with MistralEmbedder() as embedder:
        index = ChunkIndex(embedder)
        if index.count() == 0:
            raise SystemExit("data/index is empty: run python -m hyrag.seed first")
        if not is_downloaded():
            print(f"reranker model not cached yet: downloading once -> {download()}")
        scorer = CrossEncoderScorer()
        t0 = time.perf_counter()
        scorer.score("warm up", ["load the model"])
        print(f"{index.count()} chunks; reranker loaded in {time.perf_counter() - t0:.1f}s")
        retriever = HybridRetriever(index, reranker=scorer)

        for q, target in QUESTIONS.items():
            fused = retriever.retrieve(q, rerank_results=False, collection="demo")
            t0 = time.perf_counter()
            reranked = retriever.retrieve(q, collection="demo")
            ms = (time.perf_counter() - t0) * 1000
            print(f"\nQ: {q}\n   answer's rank ->  fused: {rank_of(fused, target)} of {len(fused)},  "
                  f"reranked: {rank_of(reranked, target)} of {len(reranked)}   ({ms:.0f} ms)")
            for h in reranked:
                where = h.chunk.source.split("/")[-1][:26]
                mark = "*" if is_answer(h, target) else " "
                print(f"  {mark} {h.rerank_score:.3f}  was #{h.fused_rank:<2}  {where} [{h.chunk.heading[-55:]}]")
        print(f"\nAPI requests this run: {embedder.requests_made}")


if __name__ == "__main__":
    main()
