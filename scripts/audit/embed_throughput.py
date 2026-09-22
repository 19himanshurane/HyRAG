"""Measure embedding latency/throughput with exactly ONE real Mistral API call (50 real chunks).

Usage (from the repo root): python scripts/audit/embed_throughput.py
Needs MISTRAL_API_KEY in .env. Makes one request; prints latency and projects full-corpus cost in time.
"""
import time
from pathlib import Path

from hyrag.chunking import chunk_structure
from hyrag.embeddings import MistralEmbedder
from hyrag.loader import list_corpus_files, load_document

chunks = chunk_structure(load_document("security/nist-sp-800-61r3-incident-response.pdf",
                                       Path("corpus/security/nist-sp-800-61r3-incident-response.pdf").read_bytes()))
texts = [c.text_for_search() for c in chunks[:50]]
chars = sum(len(t) for t in texts)

emb = MistralEmbedder()
t0 = time.perf_counter()
vectors = emb._request(texts)  # one HTTP request, bypassing batching on purpose
latency = time.perf_counter() - t0
print(f"1 request, {len(texts)} chunks, {chars:,} chars -> {len(vectors)} vectors x {len(vectors[0])} dims in {latency:.2f}s")
print(f"throughput: {len(texts) / latency:.0f} chunks/s, {chars / latency / 1000:.0f}k chars/s")

total = 0
for root in (Path("corpus"), Path("sample_docs")):
    for p in list_corpus_files(root):
        total += len(chunk_structure(load_document(p.relative_to(root).as_posix(), p.read_bytes())))
batches = -(-total // emb.batch_size)
print(f"full corpus: {total} structure chunks = {batches} sequential requests at batch_size={emb.batch_size} "
      f"-> ~{batches * latency * emb.batch_size / len(texts):.0f}s if latency scales with batch size [projection]")
