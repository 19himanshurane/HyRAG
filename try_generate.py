"""Grounded answers on the real corpus: retrieve (hybrid + rerank) -> answer from the top 5 with [n] citations.

Usage: python try_index.py first, then python try_generate.py   (needs MISTRAL_API_KEY and GROQ_API_KEY)
Each question costs one Groq request (~1.5-2.5k tokens).
"""
import logging

from hyrag.embeddings import MistralEmbedder
from hyrag.generation import generate
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


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if not is_downloaded():
        download()
    with MistralEmbedder() as embedder, GroqChat() as chat:
        index = ChunkIndex(embedder)
        if index.count() == 0:
            raise SystemExit("data/index is empty: run python try_index.py first")
        retriever = HybridRetriever(index, reranker=CrossEncoderScorer())
        for q in QUESTIONS:
            hits = retriever.retrieve(q)
            a = generate(q, hits, chat)
            c = a.chat
            print(f"\nQ: {q}")
            print("A: " + a.text.replace("\n", "\n   "))
            native = sum(c.text.count(m) for m in ("【", "†"))  # 【 or † in the model's raw words
            print(f"   cited {a.citations}  invalid {a.invalid_citations}  insufficient={a.insufficient}  "
                  f"native-style markers rewritten: {native}")
            print(f"   truncated={a.truncated}  |  {c.prompt_tokens}+{c.completion_tokens} tokens "
                  f"({c.reasoning_tokens} reasoning), {c.seconds:.1f}s")
            for n, h in enumerate(a.passages, 1):
                mark = "cited" if n in a.citations else "     "
                print(f"   [{n}] {mark} {h.chunk.source.split('/')[-1][:28]} [{h.chunk.heading[-55:]}]")
        print(f"\nGroq requests: {chat.requests_made}, Mistral requests: {embedder.requests_made}")


if __name__ == "__main__":
    main()
