"""Phase 2 Step 4 audit probes: the cross-encoder reranker inside HybridRetriever.

Usage (from the repo root): PYTHONPATH=. python scripts/audit/probe_rerank.py
Needs the reranker model in the local Hugging Face cache (run try_rerank.py once). Probe 6 sends ONE long
query to Mistral (the only API request).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from probe_retrieval import FakeEmbedder, build, report  # noqa: E402

from hyrag.rerank import CrossEncoderScorer  # noqa: E402
from hyrag.retrieval import HybridRetriever, RetrievalConfig  # noqa: E402


def _try(fn):
    try:
        return fn(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:220]}"


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        idx = build(tmp, "a", FakeEmbedder())

        # 1. The reranker fails at request time (model missing, OOM, a bug): does search still answer?
        class Broken:
            def score(self, query, texts):
                raise RuntimeError("model file unreadable")

        res, err = _try(lambda: HybridRetriever(idx, reranker=Broken()).retrieve("ERR_TUNNEL_4012"))
        report("a failing reranker degrades to fused results instead of failing the search", err is None,
               f"-> {err or f'{len(res)} results, rerank_score {[h.rerank_score for h in res]}'}")

        # 2. A scorer that can't load (not cached + offline): how does it fail, and how fast?
        s = CrossEncoderScorer(model="cross-encoder/does-not-exist", revision=None, local_files_only=True)
        t0 = time.perf_counter()
        _, err = _try(lambda: s.score("q", ["t"]))
        first = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, err2 = _try(lambda: s.score("q", ["t"]))
        report("an unloadable model fails with a clear error, and later requests fail fast",
               err is not None and "python -m hyrag.rerank" in err and time.perf_counter() - t0 < 0.1,
               f"first {first:.1f}s -> {err}\n          next request "
               f"{1000 * (time.perf_counter() - t0):.0f} ms -> {err2}")

        # 3. rerank_top_n larger than the candidate pool.
        _, err = _try(lambda: HybridRetriever(idx, RetrievalConfig(top_k=3, rerank_top_n=10), reranker=Broken()))
        report("rerank_top_n > top_k is rejected (it can never return more than top_k)", err is not None,
               f"-> {err or 'accepted'}")

        # 4. Does reranking hold an index lock? A keyword search must not wait for the model.
        scorer = CrossEncoderScorer(local_files_only=True)
        scorer.score("warm", ["up"])
        slow_texts = ["Troubleshooting > Error ERR_TUNNEL_4012\n\n" + "Renew the certificate. " * 60] * 20
        t = threading.Thread(target=lambda: [scorer.score("How do I fix ERR_TUNNEL_4012?", slow_texts) for _ in range(3)])
        t.start(); time.sleep(0.1)
        t0 = time.perf_counter(); idx.search_sparse("ERR_TUNNEL_4012"); dt = time.perf_counter() - t0
        t.join()
        report("keyword search is not blocked while the reranker runs", dt < 0.1, f"{dt * 1000:.0f} ms")

        # 5. Determinism: the same pair scored alone and inside a padded batch.
        q = "How do I fix ERR_TUNNEL_4012?"
        texts = ["Error ERR_TUNNEL_4012\n\nThe client certificate expired."] + [f"filler text {'x ' * i}" for i in range(1, 20)]
        alone = scorer.score(q, texts[:1])[0]
        batched = scorer.score(q, texts)[0]
        again = scorer.score(q, texts)[0]
        report("scores don't depend on batch composition", abs(alone - batched) < 1e-4 and batched == again,
               f"alone {alone:.6f}, in batch {batched:.6f}, repeat {again:.6f} (diff {abs(alone - batched):.2e})")

        # 6. A very long query (a user pastes a log file).
        long_q = "How do I fix ERR_TUNNEL_4012? " + "2026-09-24 12:00:01 vpn-client[331]: tunnel reset by peer, retrying handshake\n" * 600
        tok = scorer.model.tokenizer
        print(f"          long query: {len(long_q):,} chars, {len(tok(long_q)['input_ids']):,} reranker tokens")
        from hyrag.embeddings import MistralEmbedder
        from hyrag.index import ChunkIndex
        with MistralEmbedder(cache_path=None) as emb:
            real = HybridRetriever(ChunkIndex(emb), reranker=scorer)
            res, err = _try(lambda: real.retrieve(long_q))
            report("an over-long query gets a clear error before any API call", err is not None and emb.requests_made == 0,
                   f"-> {err or f'{len(res)} results'}; Mistral requests made: {emb.requests_made}")

        # 7. Memory: what the reranker adds to a server process.
        code = """
import ctypes, ctypes.wintypes as w
class M(ctypes.Structure):
    _fields_ = [("cb", w.DWORD), ("faults", w.DWORD)] + [(n, ctypes.c_size_t) for n in (
        "peak", "ws", "qpp", "qp", "qnpp", "qnp", "pagefile", "peakpagefile")]
ctypes.windll.kernel32.GetCurrentProcess.restype = w.HANDLE  # 64-bit pseudo-handle, not a 32-bit int
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD]
def ws_mb():
    m = M(); m.cb = ctypes.sizeof(M)
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(m), m.cb)
    return m.ws / 2**20
import hyrag.retrieval
before = ws_mb()
from hyrag.rerank import CrossEncoderScorer
CrossEncoderScorer(local_files_only=True).score("q", ["t " * 300] * 20)
print(f"working set {before:.0f} MB before loading, {ws_mb():.0f} MB after loading + scoring 20 pairs")
"""
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": ".", "HF_HUB_OFFLINE": "1"})
        report("reranker memory footprint (resident set size)", True, (out.stdout or out.stderr[-200:]).strip())

        # 8. Loading without HF_HUB_OFFLINE and with an unreachable Hub: does a cached, pinned model need the network?
        code = ("import time; t=time.perf_counter(); from hyrag.rerank import CrossEncoderScorer as C; "
                "C().score('q',['t']); print(f'{time.perf_counter()-t:.1f}s')")
        env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={**env, "PYTHONPATH": ".", "HF_ENDPOINT": "http://127.0.0.1:9"})
        ok = out.returncode == 0
        report("default scorer loads a cached model quickly with the Hub unreachable",
               ok and float(out.stdout.strip().rstrip("s")) < 60,
               out.stdout.strip() if ok else out.stderr.strip().splitlines()[-1][:150])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
