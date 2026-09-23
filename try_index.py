"""Ingest the whole corpus through the pipeline (store -> structure-aware chunks -> ChromaDB + BM25), then
compare the two searches.

Usage: python try_index.py
Stored under data/ (git-ignored). Unchanged files are not re-parsed and embeddings are cached, so a re-run
makes no API calls.
"""
import logging
import time
from pathlib import Path

from hyrag.embeddings import MistralEmbedder
from hyrag.index import ChunkIndex, describe
from hyrag.loader import DocumentStore, list_corpus_files
from hyrag.pipeline import Pipeline

QUESTIONS = [
    "How do I fix ERR_TUNNEL_4012?",                      # exact code: keyword search should win
    "What does SQLSTATE 23505 mean?",                     # exact code in a PostgreSQL table
    "I forgot my login credentials, what can I do?",      # paraphrase, no shared words: meaning search should win
    "Can I deploy to production on a Friday?",
]


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    with MistralEmbedder() as embedder:
        pipe = Pipeline(DocumentStore(), ChunkIndex(embedder))
        t0 = time.perf_counter()
        for root in (Path("corpus"), Path("sample_docs")):
            pipe.ingest(list_corpus_files(root), root=root)
        index = pipe.index
        print(f"indexed {index.count()} chunks ({index.duplicate_count()} near-duplicates skipped) in "
              f"{time.perf_counter() - t0:.1f}s, {embedder.requests_made} API requests")
        print(f"in sync: {index.check_sync()}")

        print("\nskipped near-duplicates (copy -> kept original):")
        rows = index.db.execute(
            "SELECT d.source, d.heading, d.text, d.similarity, c.source, c.heading FROM duplicates d "
            "JOIN chunks c ON c.chunk_id = d.duplicate_of ORDER BY d.source").fetchall()
        for src, heading, text, sim, orig_src, orig_heading in rows:
            print(f"   {sim:.2f}  {src.split('/')[-1][:30]} [{heading[-45:]}]")
            print(f"         = {orig_src.split('/')[-1][:30]} [{orig_heading[-45:]}]   {text[:70]!r}")

        for q in QUESTIONS:
            print(f"\nQ: {q}")
            print("   meaning search (dense / Chroma):")
            for hit in index.search_dense(q, k=3):
                print("     " + describe(hit))
            print("   keyword search (sparse / BM25):")
            for hit in index.search_sparse(q, k=3):
                print("     " + describe(hit))


if __name__ == "__main__":  # the pipeline parses in worker processes, which re-import this file on Windows
    main()
