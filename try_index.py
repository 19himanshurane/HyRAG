"""Index the whole corpus (structure-aware chunks) into ChromaDB + BM25, then compare the two searches.

Usage: python try_index.py
Stored under data/index/ (git-ignored). Embeddings are cached, so re-runs make no API calls.
"""
import logging
import time
from pathlib import Path

from hyrag.chunking import chunk_structure
from hyrag.embeddings import MistralEmbedder
from hyrag.index import ChunkIndex, describe
from hyrag.loader import list_corpus_files, load_document

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

QUESTIONS = [
    "How do I fix ERR_TUNNEL_4012?",                      # exact code: keyword search should win
    "What does SQLSTATE 23505 mean?",                     # exact code in a PostgreSQL table
    "I forgot my login credentials, what can I do?",      # paraphrase, no shared words: meaning search should win
    "Can I deploy to production on a Friday?",
]

with MistralEmbedder() as embedder:
    index = ChunkIndex(embedder)
    t0 = time.perf_counter()
    added = kept = 0
    for root in (Path("corpus"), Path("sample_docs")):
        for path in list_corpus_files(root):
            doc = load_document(path.relative_to(root).as_posix(), path.read_bytes())
            result = index.index_document(doc, chunk_structure(doc))
            added, kept = added + result.added, kept + result.kept
    print(f"indexed {index.count()} chunks ({added} embedded now, {kept} unchanged) in {time.perf_counter() - t0:.1f}s, "
          f"{embedder.requests_made} API requests")
    print(f"in sync: {index.check_sync()}")

    for q in QUESTIONS:
        print(f"\nQ: {q}")
        print("   meaning search (dense / Chroma):")
        for hit in index.search_dense(q, k=3):
            print("     " + describe(hit))
        print("   keyword search (sparse / BM25):")
        for hit in index.search_sparse(q, k=3):
            print("     " + describe(hit))
