# HyRAG

HyRAG answers questions about internal company docs and cites the exact passage behind each part of the answer.

HyRAG is a question-answering system for a company's internal documentation. You point it at a folder of Markdown, HTML, text and PDF files, and it answers questions in plain English with a numbered citation after each claim, so you can go straight to the passage it used.

Every question runs through two searches: an embedding search that finds passages meaning the same thing as the question even when the wording differs, and a BM25 keyword search that catches the exact strings embeddings tend to miss in technical docs, like error codes, CLI flags and config keys. The two result lists are merged with Reciprocal Rank Fusion, and the best candidates go through a reranker before anything reaches the model.

Most of the work goes into what happens after the answer is written. A second model call checks each citation to see whether the cited passage actually supports the sentence it's attached to, and flags the ones that don't hold up. Each answer also gets a confidence score based on how relevant the retrieved passages were, how many claims have a verified citation, and whether the whole question was covered. When that score is too low, HyRAG says it couldn't find the answer and points you to the documents that came closest, which is more useful than a confident guess.

I'm building it to learn how a RAG pipeline works end to end. The plan is to test it against a hand-written set of 50+ questions (including multi-hop ones and ones with no answer in the docs) and use that to compare three chunking strategies: fixed-size, heading-aware and semantic.

## Status

Work in progress, built one step at a time. The description above is the target design; this list shows what exists today.

- [x] Multi-format loader (Markdown, text, HTML, PDF) with heading and page metadata
- [x] Chunking: fixed-size and heading-aware (recursive)
- [ ] Chunking: semantic
- [ ] Embeddings + ChromaDB, BM25 index kept in sync
- [ ] Near-duplicate detection
- [ ] Hybrid retrieval with RRF and reranking
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

`try_loader.py` loads every file in `sample_docs/`, prints the sections it extracted, and saves the originals to `data/raw/` and the cleaned versions to `data/processed/`.
