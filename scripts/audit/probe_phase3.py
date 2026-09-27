"""Phase 3 production audit probes (offline part): outages, hangs, concurrency, observability.

Usage (from the repo root): PYTHONPATH=. python scripts/audit/probe_phase3.py
No real API calls: fake models, a fake HTTP transport and a throwaway index.
"""
import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from probe_retrieval import FakeEmbedder, build, report  # noqa: E402

from hyrag import http  # noqa: E402
from hyrag.answer import ask  # noqa: E402
import re  # noqa: E402

from hyrag.citations import BATCH_SCHEMA  # noqa: E402
from hyrag.llm import ChatResult, GroqChat  # noqa: E402
from hyrag.retrieval import HybridRetriever  # noqa: E402


class Writer:
    def __init__(self, text="Renew the certificate in NimbusConnect [1].", fail=None):
        self.text, self.fail = text, fail

    def complete(self, messages, json_schema=None):
        if self.fail:
            raise self.fail
        return ChatResult(self.text, "fake", "stop", 900, 40, 10, 0.3)


class Judge:
    def __init__(self, fail=None):
        self.fail = fail

    def complete(self, messages, json_schema=None):
        if self.fail:
            raise self.fail
        if json_schema is BATCH_SCHEMA:
            user = messages[1]["content"]
            passages = dict(re.findall(r'<passage n="(\d+)">\n(.*?)\n</passage>', user, re.S))
            checks = re.findall(r'<check id="(\d+)" passage="(\d+)">', user)
            body = {"checks": [{"id": int(i), "quote": passages[n].strip().split("\n")[-1], "verdict": "supported",
                                "reason": "ok"} for i, n in checks]}  # whole line: a cut word is rightly rejected
        else:
            body = {"parts": [{"part": "q", "status": "answered"}]}
        return ChatResult(json.dumps(body), "fake", "stop", 300, 20, 0, 0.2)


class Reranker:  # stands in for the cross-encoder: high relevance for everything
    def score(self, query, texts):
        import numpy as np
        return np.full(len(texts), 8.0, dtype=np.float32)


def _try(fn):
    try:
        return fn(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:100]}"


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        idx = build(tmp, "a", FakeEmbedder())
        retriever = HybridRetriever(idx, reranker=Reranker())
        q = "How do I fix ERR_TUNNEL_4012 certificate?"

        # 1. The answering model is down (Groq outage, bad key, quota exhausted).
        res, err = _try(lambda: ask(q, retriever, Writer(fail=RuntimeError("Groq chat request failed after 5 attempts")), Judge()))
        report("writer outage returns a structured error response", err is None and res.status == "error",
               f"-> {err or (res.status + ' / ' + res.code)}")

        # 2. The judge model is down: are answers withheld, and can the reader tell why?
        res, err = _try(lambda: ask(q, retriever, Writer(), Judge(fail=RuntimeError("judge down"))))
        report("judge outage is distinguishable from a bad answer", err is None and res.status == "unchecked",
               f"-> status {res.status if res else err} / {res.code if res else ''}; reason: "
               f"{res.reason[:80] if res else ''!r}")

        # 3. Mistral (query embedding) is down: keyword search is local and still works.
        class DeadEmbedder(FakeEmbedder):
            def embed(self, texts):
                raise RuntimeError("embedding request failed after 5 attempts")
        idx.embedder = DeadEmbedder()
        res, err = _try(lambda: ask(q, retriever, Writer(), Judge()))
        report("embedding outage degrades to keyword search instead of failing",
               err is None and res.status == "answered" and res.degraded == ["dense_search_unavailable"],
               f"-> {err or (res.status + ', degraded ' + str(res.degraded))}")
        idx.embedder = FakeEmbedder()

        # 4. Worst case wall-clock of ONE Groq call when every attempt times out (simulated, sleeps recorded).
        waited = []
        real_sleep = http.time.sleep
        http.time.sleep = waited.append
        try:
            chat = GroqChat.__new__(GroqChat)
            chat.model, chat.reasoning_effort, chat.temperature, chat.seed = "m", None, 0.0, 1
            chat.max_completion_tokens, chat.requests_made, chat._count_lock = 10, 0, threading.Lock()
            defaults = dict(zip(GroqChat.__init__.__code__.co_varnames[1:], GroqChat.__init__.__defaults__))
            chat.attempts, timeout = defaults["attempts"], defaults["timeout"]

            def hang(request):
                raise httpx.ReadTimeout("timed out", request=request)
            chat.client = httpx.Client(transport=httpx.MockTransport(hang), timeout=timeout)
            _try(lambda: chat.complete([{"role": "user", "content": "x"}]))
        finally:
            http.time.sleep = real_sleep
        per_call = chat.requests_made * timeout + sum(waited)
        print(f"          (without a budget, one Groq call: {chat.requests_made} attempts x {timeout:.0f} s + "
              f"{sum(waited):.0f} s backoff = up to {per_call:.0f} s)")

        # 4b. What bounds a QUESTION is its time budget: a Groq server that hangs on every request (it holds
        # each request for as long as that attempt's timeout allows, then times out), budget 3 s.
        def hang_honouring_timeout(request):
            time.sleep(request.extensions["timeout"]["read"])
            raise httpx.ReadTimeout("timed out", request=request)
        chat.client = httpx.Client(transport=httpx.MockTransport(hang_honouring_timeout), timeout=timeout)
        t0 = time.perf_counter()
        res, err = _try(lambda: ask(q, retriever, chat, Judge(), budget_seconds=3))
        took = time.perf_counter() - t0
        report("a question stays within its time budget when the model hangs", err is None and took < 5,
               f"budget 3 s -> returned after {took:.1f} s with {err or res.code}")

        # 5. Observability: can an operator see time, requests and tokens per response?
        res = ask(q, retriever, Writer(), Judge())
        fields = set(res.to_dict())
        report("response carries timings / request counts / token usage",
               {"timings_ms", "usage", "request_id"} <= fields,
               f"request_id {res.request_id}, timings {res.timings_ms}, usage {res.usage}")

        # 6. Concurrency: 8 threads asking at once through shared clients.
        errors, statuses = [], []

        def worker():
            try:
                statuses.append(ask(q, retriever, Writer(), Judge()).status)
            except Exception as e:
                errors.append(repr(e)[:80])
        ts = [threading.Thread(target=worker) for _ in range(8)]
        t0 = time.perf_counter()
        for t in ts: t.start()
        for t in ts: t.join()
        report("8 concurrent ask() calls", not errors and len(set(statuses)) == 1,
               f"{len(statuses)} ok, errors {errors[:1]}, statuses {set(statuses)}, {time.perf_counter() - t0:.2f}s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
