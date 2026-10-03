# HyRAG

HyRAG answers questions about a company's internal documents and shows where each part of the answer came from. Every claim carries a numbered citation, a second model checks that the cited passage really supports it, and when the documents don't contain the answer HyRAG says so instead of guessing.

I built it to learn how a retrieval-augmented generation (RAG) system works end to end, and to find out what it takes to make one you can trust.

## How a question is answered

1. Two searches run over the documents. An embedding search finds passages that mean the same thing as the question. A BM25 keyword search finds exact strings that embeddings blur, such as error codes, flags and config keys.
2. The two result lists are merged by rank (Reciprocal Rank Fusion), and a cross-encoder reranker picks the five best passages.
3. If even the best passage is a poor match, HyRAG stops here and reports "not found", listing the closest documents.
4. Otherwise a model writes an answer from those five passages only, with a citation after each sentence.
5. A second model checks every citation against its passage and has to quote the words that support it. The quote is then looked up in the passage. An answer with an unsupported claim is withheld.
6. The answer gets a confidence score built from retrieval relevance, the share of verified claims, and whether every part of the question was covered.

[docs/architecture.md](docs/architecture.md) walks through each step.

## The documents

The main collection is [PostHog's company handbook](https://posthog.com/handbook): 383 pages on time off, expenses, hiring, on-call and incidents, published by PostHog under the MIT license. It is a real company's internal documentation, which is the use case HyRAG is for.

A second collection holds public technical documents (two NIST security standards, Kubernetes and PostgreSQL pages, four GitLab handbook pages) and four files for a small fictional company. It covers the awkward cases: PDFs, tables of error codes, near-duplicate text.

Each collection is searched on its own, so one company's policy never ends up in an answer about another's. Sources and licenses are in [corpus/README.md](corpus/README.md). HyRAG is not affiliated with PostHog.

## What has been measured

| | Result |
|---|---|
| Heading-aware chunks vs fixed-size chunks, evidence found in the top 5 | 0.83 vs 0.78 (reciprocal rank 0.91 vs 0.67) |
| Citation checker on 35 hand-labelled claim/passage pairs | no false approvals in 18 that should fail; 32 of 35 exactly right |
| Prompt injection planted in a document | hijacked 2 of 10 answers before the fix, 0 of 40 after |
| Time per question (6 questions, 3 runs each) | 5.1 s median, 20.9 s worst |
| Dashboard overhead on top of that | 25 to 45 ms |

There is a hand-written set of 82 evaluation questions, 49 of them held out as a test set that nothing is tuned on. The full end-to-end scores on it are not in yet: the free model tier allows about 115 checked questions a day, and the runs so far went into finding and fixing problems. [docs/evaluation.md](docs/evaluation.md) has the method, the numbers above in detail, and what is still open.

## Run it

You need a [Mistral](https://console.mistral.ai/) API key for embeddings and a [Groq](https://console.groq.com/) API key for the two language models. Both have free tiers.

### With Docker

```bash
cp .env.example .env               # put the two keys in it
docker compose up -d --build
```

Open the dashboard at http://localhost:8501. The API and its documentation are at http://localhost:8000/docs.

The first build takes about 13 minutes, mostly downloading PyTorch. The first start then indexes the documents, which takes about 5 minutes because Mistral's free tier allows 60 embedding requests a minute. Later starts take seconds. Stop it with `docker compose down`; add `-v` to delete the index as well.

### Without Docker

```bash
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements-lock.txt
python -m hyrag.seed               # parse, chunk and index the documents
python -m hyrag.rerank             # download the reranker model once (88 MB)

python -m examples.ask --collection posthog "How do I book time off?"
uvicorn hyrag.api:app --port 8000
streamlit run dashboard/app.py     # with HYRAG_API_URL=http://localhost:8000
python -m pytest -q
```

On Linux, install PyTorch from https://download.pytorch.org/whl/cpu first, or pip fetches the CUDA build, which is several gigabytes. The reranker needs about 590 MB of RAM.

[docs/deployment.md](docs/deployment.md) covers configuration, what each container does and what happens when one of them fails.

## What's in the repository

| Path | Contents |
|---|---|
| `hyrag/` | The library: loading, chunking, the index, retrieval, generation, citation checks, confidence, the API |
| `dashboard/` | The Streamlit page. It only talks to the API |
| `examples/` | Four short scripts: parse documents, compare the two searches, see the reranker's effect, ask a question |
| `scripts/` | Evaluation runners and corpus tools; `scripts/audit/` holds the probes used to find bugs |
| `eval/` | The evaluation questions and the labelled data for the citation checker |
| `corpus/`, `sample_docs/` | The documents |
| `tests/` | The test suite. It runs offline, with fake models |
| `docs/` | How it works, how it's evaluated, how to deploy it, and the main design decisions |

## Limits

- It answers one question at a time. There is no conversation memory, so a follow-up like "and if that doesn't work?" is not understood.
- The free model tier caps it at roughly 115 fully checked questions a day.
- Two-column and scanned PDFs are not supported.
- The reranker was trained on MS MARCO, whose terms are non-commercial. Check that before any commercial use.
- Retrieved passages are sent to Groq and Mistral. That is fine for public documents; a company would need to check its own data policy.
- One API process only. The answer cache lives in memory.

[docs/decisions.md](docs/decisions.md) explains the main design choices and the measurement behind each. The original build notes, kept as they were written, are in [docs/history/](docs/history/).

## License

The code is MIT-licensed (see [LICENSE](LICENSE)). The documents in `corpus/` keep their own licenses.
