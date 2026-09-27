"""Phase 3 production audit probes (live part): latency, requests and tokens per question, throughput under the
rate limit, and how stable the outcome is when the same question is asked again.

Usage (from the repo root): PYTHONPATH=. python scripts/audit/probe_phase3_live.py [runs]
Real Groq requests: roughly (1 writer + 1-4 judge) per question per run.
"""
import statistics
import sys
import time
from collections import Counter, defaultdict

from hyrag.answer import ask
from hyrag.citations import judge_client
from hyrag.embeddings import MistralEmbedder
from hyrag.index import ChunkIndex
from hyrag.llm import GroqChat
from hyrag.rerank import CrossEncoderScorer
from hyrag.retrieval import HybridRetriever

QUESTIONS = ["How do I fix ERR_TUNNEL_4012?", "What does SQLSTATE 23505 mean?", "Can I deploy to production on a Friday?",
             "Why is my pod stuck in Pending?", "What is GitLab's caregiver sick time policy?",
             "How do I fix ERR_TUNNEL_4012, and how much does the VPN cost per month?"]
TPM_LIMIT = 8000  # tokens per minute per model on this Groq plan (x-ratelimit-limit-tokens)


class Metered:
    """Wraps a chat client to count requests and tokens per question."""

    def __init__(self, chat):
        self.chat, self.calls, self.tokens, self.seconds = chat, 0, 0, 0.0

    def complete(self, messages, json_schema=None):
        r = self.chat.complete(messages, json_schema=json_schema)
        self.calls += 1
        self.tokens += r.prompt_tokens + r.completion_tokens
        self.seconds += r.seconds
        return r


def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    outcomes = defaultdict(list)
    per_q = []
    with MistralEmbedder() as emb, GroqChat() as w, judge_client() as j:
        retriever = HybridRetriever(ChunkIndex(emb), reranker=CrossEncoderScorer())
        retriever.retrieve("warm up the reranker")
        for run in range(runs):
            for q in QUESTIONS:
                writer, judge = Metered(w), Metered(j)
                t0 = time.perf_counter()
                r = ask(q, retriever, writer, judge)
                wall = time.perf_counter() - t0
                outcomes[q].append(f"{r.status}/{r.confidence.level if r.confidence else '-'}")
                per_q.append((wall, writer.calls, writer.tokens, judge.calls, judge.tokens,
                              writer.seconds + judge.seconds))
    walls = [p[0] for p in per_q]
    wt, jt = [p[2] for p in per_q], [p[4] for p in per_q]
    print(f"per question ({len(per_q)} asks): wall median {statistics.median(walls):.1f}s, max {max(walls):.1f}s "
          f"(time inside model calls: median {statistics.median(p[5] for p in per_q):.1f}s)")
    print(f"  requests: writer {statistics.mean(p[1] for p in per_q):.1f}, judge {statistics.mean(p[3] for p in per_q):.1f} "
          f"(max {max(p[3] for p in per_q)})")
    print(f"  tokens: writer mean {statistics.mean(wt):.0f}, judge mean {statistics.mean(jt):.0f}")
    print(f"  throughput ceiling at {TPM_LIMIT} tokens/min per model: writer {TPM_LIMIT / statistics.mean(wt):.1f} q/min, "
          f"judge {TPM_LIMIT / statistics.mean(jt):.1f} q/min -> about "
          f"{min(TPM_LIMIT / statistics.mean(wt), TPM_LIMIT / statistics.mean(jt)):.1f} questions per minute for everyone")
    flips = {q: Counter(o) for q, o in outcomes.items() if len(set(o)) > 1}
    print(f"  stability over {runs} runs: {len(QUESTIONS) - len(flips)}/{len(QUESTIONS)} questions gave the same "
          f"status/level every time")
    for q, c in flips.items():
        print(f"    changed: {q[:60]} -> {dict(c)}")
    print(f"  total requests: writer {w.requests_made}, judge {j.requests_made} (includes 429 retries)")


if __name__ == "__main__":
    main()
