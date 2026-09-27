"""Build (or update) the search index for one chunking strategy, for the Phase 4 comparison.

Usage (from the repo root): PYTHONPATH=. python scripts/build_index.py --strategy fixed|structure|semantic
Each strategy gets its own chunk table and Chroma collection (data/index/chunks-<strategy>.*). All three use the
same target size (800 characters); fixed and structure overlap 120, semantic cuts on topic changes without
overlap (Phase 1). Parsing is cached by the document store and embeddings by the embedding cache, so a
re-run only pays for text it has not embedded before.
"""
import argparse
import statistics
import time
from pathlib import Path

from hyrag.chunking import chunk_fixed, chunk_semantic, chunk_structure
from hyrag.embeddings import MistralEmbedder
from hyrag.index import ChunkIndex
from hyrag.loader import DocumentStore, list_corpus_files
from hyrag.pipeline import Pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=["fixed", "structure", "semantic"])
    args = ap.parse_args()
    with MistralEmbedder() as embedder:
        chunker = {"fixed": chunk_fixed, "structure": chunk_structure,
                   "semantic": lambda doc: chunk_semantic(doc, embedder)}[args.strategy]
        pipe = Pipeline(DocumentStore(), ChunkIndex(embedder, strategy=args.strategy), chunker=chunker)
        t0 = time.perf_counter()
        for root in (Path("corpus"), Path("sample_docs")):
            pipe.ingest(list_corpus_files(root), root=root)
        index = pipe.index
        sizes = [r[0] for r in index.db.execute("SELECT char_count FROM chunks")]
        print(f"{args.strategy}: {index.count()} chunks ({index.duplicate_count()} near-duplicates skipped), "
              f"size median {statistics.median(sizes):.0f} chars (min {min(sizes)}, max {max(sizes)}), "
              f"{time.perf_counter() - t0:.0f}s, {embedder.requests_made} embedding requests, "
              f"in sync: {not any(index.check_sync().values())}")


if __name__ == "__main__":
    main()
