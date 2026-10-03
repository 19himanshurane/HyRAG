"""Ask a question from the command line and print the full response.

Usage (from the repo root):
  python -m examples.ask "How do I fix ERR_TUNNEL_4012?" ["another question" ...]
  python -m examples.ask                                 one question for each kind of response
  python -m examples.ask --collection posthog "How do I book time off?"
  add --json to print the raw response, as the API returns it

Needs an index (python -m hyrag.seed), MISTRAL_API_KEY and GROQ_API_KEY. The reranker model is downloaded on
first use.
"""
import json
import logging
import sys

from hyrag.answer import ask
from hyrag.citations import judge_client
from hyrag.embeddings import MistralEmbedder
from hyrag.index import ChunkIndex
from hyrag.llm import GroqChat
from hyrag.rerank import CrossEncoderScorer, download, is_downloaded
from hyrag.retrieval import HybridRetriever

SHOWCASE = [
    "How do I fix ERR_TUNNEL_4012?",                                             # answered
    "How do I fix ERR_TUNNEL_4012, and how much does the VPN cost per month?",   # partial
    "How do I set up the office printer?",                                       # not found, nothing related
    "What does SQLSTATE 23599 mean?",                                            # not found, a document to check
    "How do I fix a pod stuck in ImagePullBackOff?",                             # the model may improvise
]


def show(r) -> None:
    print(f"\nQ: {r.question}\n   status: {r.status.upper()}  [{r.code}]  ({r.reason})")
    if r.message:
        print(f"   {r.message}")
    if r.answer:
        print("   " + r.answer.replace("\n", "\n   "))
        for s in r.sources:
            print(f"     [{s.n}] {s.source} > {s.section[-60:]}" + (f", page {s.page}" if s.page else ""))
    if r.confidence and r.status != "not_found":
        c = r.confidence
        print(f"   confidence {c.score:.2f} {c.level} (retrieval {c.retrieval:.2f}, citations {c.citations:.2f}, "
              f"completeness {c.completeness:.2f}) {' '.join(c.flags)}")
    for claim in r.flagged_claims:
        print(f"   ! not fully verified: {claim[:100]}")
    for part in r.not_covered:
        print(f"   - not covered by the documents: {part}")
    for s in r.closest:
        print(f"   closest: {s.source} > {s.section[-60:]} (relevance {s.relevance})")
    if r.draft:
        print(f"   (withheld draft: {r.draft[:120]!r})")
    if r.status in ("unverified", "unchecked"):
        for claim in r.verified_claims:
            print(f"   verified anyway: {claim[:100]}")
    if r.search_queries:
        print(f"   searched as: {' | '.join(r.search_queries)}")
    if r.degraded:
        print(f"   degraded: {', '.join(r.degraded)}")
    u = r.usage
    cost = (" (cached)" if r.cached else
            f", writer {u.get('writer_requests', 0)} req / {u.get('writer_tokens', 0)} tok,"
            f" judge {u.get('judge_requests', 0)} req / {u.get('judge_tokens', 0)} tok")
    print(f"   request {r.request_id}: {r.timings_ms.get('total', 0) / 1000:.1f}s{cost}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = sys.argv[1:]
    as_json = "--json" in args
    collection = "demo"  # the showcase questions are about the demo documents
    if "--collection" in args:
        i = args.index("--collection")
        collection = args[i + 1]
        del args[i:i + 2]
    questions = [a for a in args if a != "--json"] or SHOWCASE
    if not is_downloaded():
        download()
    with MistralEmbedder() as emb, GroqChat() as writer, judge_client() as judge:
        index = ChunkIndex(emb)
        if index.count() == 0:
            raise SystemExit("data/index is empty: run python -m hyrag.seed first")
        retriever = HybridRetriever(index, reranker=CrossEncoderScorer(), rewriter=writer)
        for q in questions:
            r = ask(q, retriever, writer, judge, collection=collection)
            print(json.dumps(r.to_dict(), indent=1, ensure_ascii=False)) if as_json else show(r)
        print(f"\nrequests: writer {writer.requests_made}, judge {judge.requests_made}, Mistral {emb.requests_made}")


if __name__ == "__main__":
    main()
