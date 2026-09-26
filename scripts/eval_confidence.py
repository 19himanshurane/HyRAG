"""Run the whole answer pipeline on eval/calibration_questions.json and show the confidence it assigns.

Usage (from the repo root): PYTHONPATH=. python scripts/eval_confidence.py
Real requests: per question one answer + one judge request per cited claim + one completeness request.
A good scorer gives answerable questions high confidence and unanswerable ones "not_found" or low.
"""
import json

from hyrag.citations import JUDGE_MODEL, verify
from hyrag.confidence import score_answer
from hyrag.embeddings import MistralEmbedder
from hyrag.generation import GroundedAnswer, generate
from hyrag.index import ChunkIndex
from hyrag.llm import GroqChat
from hyrag.rerank import CrossEncoderScorer
from hyrag.retrieval import HybridRetriever

EXTRA = [("How do I fix ERR_TUNNEL_4012, and how much does the VPN cost per month?", "half")]


def main() -> None:
    qs = [(q["question"], "yes" if q["answerable"] else "no")
          for q in json.load(open("eval/calibration_questions.json", encoding="utf-8"))] + EXTRA
    with MistralEmbedder() as emb, GroqChat() as chat, GroqChat(JUDGE_MODEL, reasoning_effort=None) as judge:
        retriever = HybridRetriever(ChunkIndex(emb), reranker=CrossEncoderScorer())
        rows = []
        for q, answerable in qs:
            a = generate(q, retriever.retrieve(q), chat)
            rows.append((q, answerable, score_answer(a, verify(a, judge), judge)))
        wrong = GroundedAnswer(qs[0][0], "ERR_TUNNEL_4012 means the gateway rejected your region [1]. "
                               "Pick your country's gateway under Settings > Gateway [1].",
                               retriever.retrieve(qs[0][0]), citations=[1])
        rows.append(("(deliberately wrong answer) " + qs[0][0], "wrong", score_answer(wrong, verify(wrong, judge), judge)))
        print(f"{'answerable':>10} {'score':>5} {'level':>9}  retr  cite  comp  flags / question")
        for q, answerable, c in rows:
            print(f"{answerable:>10} {c.score:5.2f} {c.level:>9}  {c.retrieval:4.2f}  {c.citations:4.2f}  "
                  f"{c.completeness:4.2f}  {','.join(c.flags) or '-'} | {q[:55]}")
        print(f"requests: answers {chat.requests_made}, judge {judge.requests_made}, Mistral {emb.requests_made}")


if __name__ == "__main__":
    main()
