"""Export the embedding vectors the current index uses into one small cache file to ship with a deployment.

Usage (from the repo root): PYTHONPATH=. python scripts/export_embedding_cache.py [OUT]
                            (default OUT: deploy/embeddings.sqlite)

A hosted copy starts from an empty disk. With this file in place the seed finds every chunk's vector in the
cache and makes no embedding requests; without it, a start would spend several minutes and a few hundred
Mistral requests re-embedding the documents. Re-run it after the documents change.
"""
import sqlite3
import sys
from pathlib import Path

from hyrag.chunking import Chunk
from hyrag.embeddings import EmbeddingCache

MODEL = "mistral-embed"
COLUMNS = "chunk_id, doc_id, source, text, chunk_index, heading, page, strategy, char_count"


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("deploy/embeddings.sqlite")
    chunks = sqlite3.connect("data/index/chunks-structure.sqlite")
    names = COLUMNS.split(", ")
    keys = [EmbeddingCache.key(MODEL, Chunk(**dict(zip(names, row))).text_for_search())
            for row in chunks.execute(f"SELECT {COLUMNS} FROM chunks")]
    source = EmbeddingCache(Path("data/embeddings.sqlite"))
    found = source.get_many(list(set(keys)))
    missing = len(set(keys)) - len(found)
    if missing:
        print(f"{missing} chunk vector(s) are not in data/embeddings.sqlite: rebuild the index first", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    target = EmbeddingCache(out)
    target.put_many(list(found.items()))
    target.db.execute("VACUUM")
    target.close()
    source.close()
    print(f"{len(found)} vectors for {len(keys)} chunks -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
