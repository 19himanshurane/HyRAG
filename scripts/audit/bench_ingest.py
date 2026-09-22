"""Compare full-corpus ingest wall time: sequential ingest_file vs parallel ingest_many (cold, no cache).

Usage (from the repo root): python scripts/audit/bench_ingest.py
"""
import shutil
import tempfile
import time
from pathlib import Path

from hyrag.loader import DocumentStore, list_corpus_files

if __name__ == "__main__":  # required on Windows: worker processes re-import this file
    root = Path("corpus")
    files = list_corpus_files(root)
    tmp = Path(tempfile.mkdtemp())
    try:
        seq = DocumentStore(tmp / "seq")
        t0 = time.perf_counter()
        for f in files:
            seq.ingest_file(f, root=root)
        t_seq = time.perf_counter() - t0

        par = DocumentStore(tmp / "par")
        t0 = time.perf_counter()
        docs = par.ingest_many(files, root=root)
        t_par = time.perf_counter() - t0

        t0 = time.perf_counter()
        par.ingest_many(files, root=root)
        t_warm = time.perf_counter() - t0

        same = [d for d in docs] == [seq._cached(d.source, (root / d.source).read_bytes()) for d in docs]
        print(f"{len(files)} files: sequential {t_seq:.1f}s | parallel {t_par:.1f}s ({t_seq / t_par:.1f}x) | "
              f"re-run with nothing changed {t_warm:.2f}s | identical output: {same}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
