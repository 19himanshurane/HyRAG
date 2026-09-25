"""Grounded answers on the real corpus, with every citation checked.

retrieve (hybrid + rerank) -> answer from the top 5 with [n] citations -> a judge model checks each cited claim.

Usage: python try_index.py first, then python try_generate.py   (needs MISTRAL_API_KEY and GROQ_API_KEY)
Each question costs one Groq request to answer (~1.5-2.5k tokens) plus one judge request per cited claim.
The last case is a deliberately WRONG answer, to show that verification catches a bad citation.
"""
import logging

from hyrag.citations import JUDGE_MODEL, verify
from hyrag.embeddings import MistralEmbedder
from hyrag.generation import GroundedAnswer, generate
from hyrag.index import ChunkIndex
from hyrag.llm import GroqChat
from hyrag.rerank import CrossEncoderScorer, download, is_downloaded
from hyrag.retrieval import HybridRetriever

QUESTIONS = [
    "How do I fix ERR_TUNNEL_4012?",
    "What does SQLSTATE 23505 mean?",
    "I forgot my login credentials, what can I do?",
    "Can I deploy to production on a Friday?",
    "Why is my pod stuck in Pending?",
    "What happens when a pod cannot be placed on any machine?",
    "How do I set up the office printer?",                              # not in the documents
    "How do I fix ERR_TUNNEL_4012, and how much does the VPN cost per month?",  # half answerable
]


def show_verification(answer: GroundedAnswer, judge) -> None:
    report = verify(answer, judge)
    for check in report.checks:
        detail = "; ".join(f"[{j.passage}] {j.verdict}" for j in check.judgements)
        print(f"   {check.status.upper():<11} {check.claim.text[:78]!r}  {detail}")
    if report.flagged:
        print(f"   -> {len(report.flagged)} claim(s) flagged for the reader")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if not is_downloaded():
        download()
    with MistralEmbedder() as embedder, GroqChat() as chat, GroqChat(JUDGE_MODEL, reasoning_effort=None) as judge:
        index = ChunkIndex(embedder)
        if index.count() == 0:
            raise SystemExit("data/index is empty: run python try_index.py first")
        retriever = HybridRetriever(index, reranker=CrossEncoderScorer())
        for q in QUESTIONS:
            a = generate(q, retriever.retrieve(q), chat)
            c = a.chat
            print(f"\nQ: {q}")
            print("A: " + a.text.replace("\n", "\n   "))
            print(f"   cited {a.citations}  invalid {a.invalid_citations}  insufficient={a.insufficient}  "
                  f"ungrounded={a.ungrounded}  truncated={a.truncated}  |  {c.prompt_tokens}+{c.completion_tokens} "
                  f"tokens, {c.seconds:.1f}s")
            show_verification(a, judge)

        # A wrong answer on purpose: 4013's meaning attributed to 4012, citing the 4012 passage.
        wrong = GroundedAnswer(QUESTIONS[0], "ERR_TUNNEL_4012 means the gateway rejected your region [1]. "
                               "Pick your country's gateway under Settings > Gateway [1].",
                               retriever.retrieve(QUESTIONS[0]), citations=[1])
        print(f"\nDELIBERATELY WRONG ANSWER to: {wrong.question}\nA: {wrong.text}")
        show_verification(wrong, judge)
        print(f"\nGroq requests: answers {chat.requests_made}, judge {judge.requests_made}; "
              f"Mistral requests: {embedder.requests_made}")


if __name__ == "__main__":
    main()
