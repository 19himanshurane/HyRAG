"""Measure ingestion: per-file wall time, chunk statistics, and optionally cProfile hotspots or peak memory.

Usage (from the repo root):
  python scripts/audit/profile_ingest.py                     # timing + chunk stats for every file
  python scripts/audit/profile_ingest.py --profile nist-sp-800-63b   # cProfile top 10 for one file
  python scripts/audit/profile_ingest.py --memory            # tracemalloc peak per file (slower)
"""
import cProfile
import io
import pstats
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

from hyrag.chunking import chunk_fixed, chunk_structure
from hyrag.loader import list_corpus_files, load_document

FILES = list_corpus_files(Path("corpus")) + list_corpus_files(Path("sample_docs"))


def rel(p: Path) -> str:
    return p.relative_to(p.parts[0]).as_posix()


def stats(values: list[int]) -> str:
    if not values:
        return "-"
    v = sorted(values)
    return f"min {v[0]}, median {int(statistics.median(v))}, p95 {v[int(0.95 * (len(v) - 1))]}, max {v[-1]}"


if "--profile" in sys.argv:
    target = sys.argv[sys.argv.index("--profile") + 1]
    path = next(p for p in FILES if target in p.as_posix())
    data = path.read_bytes()
    prof = cProfile.Profile()
    prof.enable()
    doc = load_document(rel(path), data)
    chunk_structure(doc)
    prof.disable()
    st = pstats.Stats(prof)
    total = st.total_tt
    for key in ("cumulative", "tottime"):
        rows = sorted(st.stats.items(), key=lambda kv: kv[1][3 if key == "cumulative" else 2], reverse=True)[:10]
        print(f"===== {path.as_posix()} — top 10 by {key} (total {total:.1f}s) =====")
        for (file, line, func), (cc, nc, tt, ct, _) in rows:
            print(f"  {ct if key == 'cumulative' else tt:7.2f}s  {100 * (ct if key == 'cumulative' else tt) / total:5.1f}%  "
                  f"calls={nc:<9} {Path(file).name}:{line} {func}")
    sys.exit()

if "--memory" in sys.argv:
    print(f"{'file':52} {'size':>10} {'peak MB':>9}")
    for path in FILES:
        data = path.read_bytes()
        tracemalloc.start()
        load_document(rel(path), data)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"{path.as_posix():52} {len(data):>10,} {peak / 1e6:>9.1f}")
    sys.exit()

total_t, total_chunks = 0.0, {"fixed": 0, "structure": 0}
all_sizes = {"fixed": [], "structure": []}
print(f"{'file':52} {'bytes':>10} {'parse s':>8} {'sections':>8} {'fixed':>6} {'struct':>6} {'<100ch':>6}")
for path in FILES:
    data = path.read_bytes()
    t0 = time.perf_counter()
    doc = load_document(rel(path), data)
    t = time.perf_counter() - t0
    total_t += t
    fx, st = chunk_fixed(doc), chunk_structure(doc)
    total_chunks["fixed"] += len(fx)
    total_chunks["structure"] += len(st)
    all_sizes["fixed"] += [c.char_count for c in fx]
    all_sizes["structure"] += [c.char_count for c in st]
    tiny = sum(c.char_count < 100 for c in st)
    print(f"{path.as_posix():52} {len(data):>10,} {t:>8.2f} {len(doc.sections):>8} {len(fx):>6} {len(st):>6} {tiny:>6}")
print(f"\nTOTAL parse time {total_t:.1f}s for {len(FILES)} files; chunks: fixed={total_chunks['fixed']}, structure={total_chunks['structure']}")
for k, v in all_sizes.items():
    print(f"chunk chars [{k:9}]: {stats(v)}")
