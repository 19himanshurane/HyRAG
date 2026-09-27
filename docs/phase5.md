# HyRAG Phase 5: API, dashboard, containers

Three steps from the brief: (1) a FastAPI service, (2) a query dashboard, (3) Docker Compose with a seed script.

---

## Step 1 (2026-09-28): the FastAPI service

**Code:** `hyrag/api.py`. **Run:** `uvicorn hyrag.api:app --port 8000`; the OpenAPI docs are at `/docs` and `/openapi.json`. **Tests:** `tests/test_api.py`: 20 in-process tests with a real index, retriever and document store, and fake models.

### Endpoints

| Endpoint | What it does | Auth |
|---|---|---|
| `POST /v1/ask` | `{question, mode}` → the full structured answer. The response schema is `hyrag.answer.Response`, published in OpenAPI. | `X-API-Key` if `HYRAG_API_KEY` is set |
| `GET /v1/documents` | indexed documents with chunk and duplicate counts (paginated) | same |
| `POST /v1/ingest` | multipart upload of .md/.txt/.html/.pdf files; each is parsed, chunked and indexed in-process. Same filename = a new version | `X-API-Key` = `HYRAG_ADMIN_KEY`; **disabled when unset** |
| `GET /health` | liveness | open |
| `GET /ready` | readiness: index not empty, reranker model downloaded and warmed (503 lists the problems) | open |

### Design decisions
- **HTTP status follows the answer's status.**
  - answered, partial, not_found, unverified and unchecked are all **200**, each with its `status` and `code`. "The documents don't say" is a correct result, not a server failure.
  - Only `error` maps to 5xx: `error.timeout` → 504; outages and `error.quota_exhausted` → 503 (with `Retry-After`).
  - The body always has the same schema, so a client handles one shape.
- **Built once at startup:** the reranker is loaded and warmed in the server's lifespan (~10 s), not on the first user's request. Missing API keys fail the start loudly; a missing index or model is reported by `/ready` instead of crashing.
- **Uploads are parsed in-process**, one batch at a time (`Pipeline.ingest_one`). The batch path starts a process pool even for one worker: on Windows that spawns and re-imports the app per request.
- **Each upload succeeds or fails on its own.** Names must be simple (a path like `../escape.md` is reduced to its name, then checked), extensions are allow-listed, and size is capped (`HYRAG_MAX_UPLOAD_MB`, default 20, never above the loader's 50). Verified per case in the tests.
- **`/v1/ask` returns what the dashboard needs:**
  - `retrieved`: every passage the writer saw, with its dense rank, sparse rank, reranker score and calibrated relevance;
  - `confidence` by dimension;
  - `sources` by `[n]`;
  - `search_queries` if the question was rewritten;
  - `mode`, for the hybrid vs dense-only comparison (answers are cached per mode).
- **Keys are compared in constant time** (`hmac.compare_digest`). CORS is off unless `HYRAG_CORS_ORIGINS` names the dashboard's origin (tested: another origin gets no CORS header).

### Live check (real services, over HTTP, against a copy of `data/`)
- `/health` ok; `/ready` ready with 1,266 chunks; `/v1/documents` lists 16 documents.
- "How do I fix ERR_TUNNEL_4012?" → 200 `answered`, confidence 0.999. The repeat was **served from the cache in 0.3 ms**. `retrieved` shows the project's founding case: 4012 ranked first (rerank 4.95; dense #2, sparse #1), ahead of 4013 (3.12; dense #1).
- "How do I book a meeting room?" → `not_found`; one rewrite (307 tokens), and the writer was never called.
- A blank question → 422. An upload without the key → 401; with it → indexed (1 chunk), and the list grew to 17.
- Asking about the uploaded document hit the **daily quota** of the writer model. The response was a clean `error.quota_exhausted` pointing to the new document, and the server log had the stack trace. This also measured that **gpt-oss-120b has a 200,000 tokens/day limit, like the judge**.

### A leak the live check exposed (fixed)
- **What leaked:** Groq's 429 message names the account's **organization id**. It was copied into exception text and from there into the `reason` field returned to API clients.
- **The fix:** `hyrag.http.provider_text` redacts provider account ids (`org_…`, `proj_…`, `user_…`, `acct_…`) wherever provider errors become messages.
- **Caught by its own test:** the first version of the fix wrote a literal backspace character (a tooling escape slip) where `\b` belonged, so it matched nothing. A redaction test caught it.
- **Swept for the whole class:** every source file was then scanned for hidden characters, and 11 deliberate test inputs (look-alike hyphens, a BOM, no-break spaces) were rewritten as visible `\uXXXX` escapes.

### Dependencies
`fastapi` (MIT), `uvicorn` (BSD-3-Clause) and `python-multipart` (Apache-2.0) are now explicit in `requirements.txt`. **Why it mattered:** the lock file was missing `fastapi` and `python-multipart`; they were only in the development environment, so a fresh install could not have run the API. The lock was regenerated from a clean venv built from `requirements.txt` alone, and all 320 tests pass there.

### Known limits
- **One process:** the answer cache and the ingest lock live in memory. Several server processes would need a shared cache (e.g. Redis) and ingestion moved to one worker or a queue.
- **Deletion is not exposed** over HTTP (`Pipeline.delete` exists). It should come with the same admin auth when needed.
- **Throughput is bounded by the Groq plan**, not the server: ~115 fully checked questions/day (docs/phase3-audit.md).
