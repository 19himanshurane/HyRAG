"""Dense search over the real sample docs, using real Mistral embeddings.

For each question we know which chunk *should* win. We search twice:
  body only       - embed just the chunk text
  heading + body  - embed chunk.text_for_search() (heading path prepended)
and report where the correct chunk landed in each, so we can see whether headings actually help.
"""
from pathlib import Path

import numpy as np

from hyrag.chunking import chunk_structure
from hyrag.embeddings import MistralEmbedder
from hyrag.loader import load_document

# (question, words that identify the correct chunk's heading)
QUESTIONS = [
    ("How do I fix ERR_TUNNEL_4012?", "ERR_TUNNEL_4012"),
    ("What does ERR_TUNNEL_4013 mean?", "ERR_TUNNEL_4013"),
    ("my vpn is really slow", "Slow connection"),
    ("Can I ship to production on a Friday?", "Production"),
    ("How many vacation days do I get?", "leave-policy"),
    ("How long does my password need to be?", "Security Handbook"),
]

chunks = []
for path in sorted(Path("sample_docs").iterdir()):
    if path.suffix != ".py":
        chunks += chunk_structure(load_document(path.name, path.read_bytes()))
print(f"{len(chunks)} chunks from {len({c.source for c in chunks})} documents\n")

embedder = MistralEmbedder()
body_vecs = embedder.embed([c.text for c in chunks])
full_vecs = embedder.embed([c.text_for_search() for c in chunks])
query_vecs = embedder.embed([q for q, _ in QUESTIONS])


def rank_of_correct(scores: np.ndarray, target: str) -> int:
    order = np.argsort(-scores)
    return next(r for r, i in enumerate(order, start=1) if target in chunks[i].heading)


summary = []
for (question, target), qv in zip(QUESTIONS, query_vecs):
    body_scores, full_scores = body_vecs @ qv, full_vecs @ qv  # dot product = cosine (vectors have length 1)
    r_body, r_full = rank_of_correct(body_scores, target), rank_of_correct(full_scores, target)
    summary.append((question, r_body, r_full))

    print(f"Q: {question}")
    for i in np.argsort(-full_scores)[:3]:
        mark = "<- correct" if target in chunks[i].heading else ""
        print(f"   {full_scores[i]:.3f}  [{chunks[i].source} | {chunks[i].heading}]  {mark}")
    print()

print("Rank of the correct chunk (1 = top result):")
print(f"{'question':<42}{'body only':>11}{'heading+body':>14}")
for q, rb, rf in summary:
    print(f"{q:<42}{rb:>11}{rf:>14}")
