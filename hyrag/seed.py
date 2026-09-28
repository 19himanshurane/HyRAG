"""Index folders of documents. Docker Compose runs this once before the API starts.

Usage: python -m hyrag.seed [DIR ...]          (default: corpus sample_docs)

Re-running is cheap: unchanged chunks keep their ids and their vectors are cached. Documents keep their folder
in their name ("engineering/k8s-secrets.md"), which the golden set's evidence relies on.

The chunk table and the vectors can live in different volumes, so the seed ends with repair() in case one was
wiped. It exits 1 if a file failed or the index is still out of sync, and Compose then won't start the API.

Environment: MISTRAL_API_KEY, HYRAG_DATA_DIR (data), HYRAG_STRATEGY (structure), HYRAG_CHROMA_URL (optional).
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from hyrag.embeddings import MistralEmbedder
from hyrag.index import ChunkIndex
from hyrag.loader import DocumentStore, list_corpus_files
from hyrag.pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="*", type=Path, default=[Path("corpus"), Path("sample_docs")])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    missing = [d for d in args.dirs if not d.is_dir()]
    if missing:
        print(f"seed: not a folder: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1
    if not os.environ.get("MISTRAL_API_KEY"):
        print("seed: MISTRAL_API_KEY is not set", file=sys.stderr)
        return 1

    data_dir = Path(os.environ.get("HYRAG_DATA_DIR", "data"))
    strategy = os.environ.get("HYRAG_STRATEGY", "structure")
    t0 = time.perf_counter()
    with MistralEmbedder(cache_path=data_dir / "embeddings.sqlite") as embedder:
        index = ChunkIndex(embedder, data_dir=data_dir / "index", strategy=strategy,
                           chroma_url=os.environ.get("HYRAG_CHROMA_URL") or None)
        store = DocumentStore(data_dir)
        pipe = Pipeline(store, index)
        docs, failures = 0, {}
        for root in args.dirs:
            docs += len(pipe.ingest(list_corpus_files(root), root=root))
            failures.update(store.failures)
        repaired = index.repair()
        left = index.check_sync()
        print(f"seed: {docs} documents, {index.count()} chunks ({index.duplicate_count()} near-duplicates skipped), "
              f"{embedder.requests_made} embedding requests, {time.perf_counter() - t0:.0f} s; "
              f"repaired {sum(map(len, repaired.values()))} out-of-sync entries")
    for source, error in failures.items():
        print(f"seed: FAILED {source}: {error}", file=sys.stderr)
    if any(left.values()):
        print(f"seed: index still out of sync: { {k: len(v) for k, v in left.items()} }", file=sys.stderr)
    return 1 if failures or any(left.values()) else 0


if __name__ == "__main__":  # the guard matters: parsing uses worker processes
    sys.exit(main())
