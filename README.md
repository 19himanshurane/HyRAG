# HyRAG

HyRAG answers questions about internal company docs and cites the exact passage behind each part of the answer.

HyRAG is a question-answering system for a company's internal documentation. You point it at a folder of Markdown, HTML, text and PDF files, and it answers questions in plain English with a numbered citation after each claim, so you can go straight to the passage it used.

Every question runs through two searches: an embedding search that finds passages meaning the same thing as the question even when the wording differs, and a BM25 keyword search that catches the exact strings embeddings tend to miss in technical docs, like error codes, CLI flags and config keys. The two result lists are merged with Reciprocal Rank Fusion, and the best candidates go through a reranker before anything reaches the model.

Most of the work goes into what happens after the answer is written. A second model call checks each citation to see whether the cited passage actually supports the sentence it's attached to, and flags the ones that don't hold up. Each answer also gets a confidence score based on how relevant the retrieved passages were, how many claims have a verified citation, and whether the whole question was covered. When that score is too low, HyRAG says it couldn't find the answer and points you to the documents that came closest, which is more useful than a confident guess.

I'm building it to learn how a RAG pipeline works end to end. The plan is to test it against a hand-written set of 50+ questions (including multi-hop ones and ones with no answer in the docs) and use that to compare three chunking strategies: fixed-size, heading-aware and semantic.

## Status

Work in progress, built one step at a time. The description above is the target design; this list shows what exists today.

- [x] Multi-format loader (Markdown, text, HTML, PDF) with heading and page metadata. PDFs: layout-aware heading detection for single-column documents; two-column layouts and scanned PDFs are not supported yet
- [x] Chunking: fixed-size and heading-aware (recursive), with content-based chunk ids
- [x] Production audit of Phase 1, three rounds ([docs/phase1-audit.md](docs/phase1-audit.md)): 32 findings fixed with before/after measurements; an offline test suite covers every one (run `python -m pytest -q`)
- [x] One pipeline for documents in and out (`hyrag/pipeline.py`): store → chunk → index, and deletions reach both
- [x] Chunking: semantic (topic cuts from neighbour-embedding similarity, relative per-document threshold; whether it beats structure-aware is measured in the eval phase)
- [x] Embeddings + ChromaDB, BM25 index kept in sync (one SQLite chunk table is the source of truth; Chroma is reconciled against it, BM25 is rebuilt from it; `check_sync()` / `repair()`). Meaning search is exact cosine over the stored vectors, not Chroma's approximate HNSW, which missed real top-10 results on this corpus (measured in [docs/phase2-audit.md](docs/phase2-audit.md), S6)
- [x] Near-duplicate detection (by text, not embeddings: ≥90% shared 3-word sequences + identical numbers/identifiers; skipped copies are recorded and promoted back if their original disappears)
- [x] Hybrid retrieval: weighted Reciprocal Rank Fusion of both searches (`hyrag/retrieval.py`), then a local cross-encoder reranker (`hyrag/rerank.py`, ms-marco-MiniLM-L6-v2, pinned revision) that keeps the best 5 of the fused 20. Audited: [docs/phase2-audit.md](docs/phase2-audit.md)
- [ ] Grounded generation with citation verification and confidence scoring
- [ ] Evaluation suite and chunking comparison
- [ ] FastAPI service, dashboard, Docker

## Try it

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python sample_docs/make_sample_pdf.py
python try_loader.py
```

Retrieval (needs `MISTRAL_API_KEY` in `.env`):

```bash
python try_index.py             # parse, chunk and index the corpus into data/index/
python -m hyrag.rerank          # download the reranker model once (~88 MB); it loads offline after that
python try_rerank.py            # hybrid search + reranking on real questions
```

Heads-up on size: `requirements.txt` pulls in PyTorch for the reranker (a ~124 MB CPU wheel on Windows/macOS; on Linux, install the CPU build from https://download.pytorch.org/whl/cpu or pip fetches the multi-GB CUDA one). Loading the reranker takes ~10-13 s and ~590 MB of RAM, so a server should load it once at startup.

`try_loader.py` loads every file in `sample_docs/`, prints the sections it extracted, and saves the originals to `data/raw/` and the cleaned versions to `data/processed/`.
