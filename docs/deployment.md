# Running and deploying HyRAG

## Docker Compose

```bash
cp .env.example .env               # MISTRAL_API_KEY and GROQ_API_KEY
docker compose up -d --build
```

Four services start in a fixed order, and each waits for the one before it.

| Service | What it does | Starts when |
|---|---|---|
| `chroma` | the ChromaDB server; vectors live in the `chroma-data` volume | first |
| `seed` | indexes `corpus/` and `sample_docs/`, repairs the index, then exits | chroma is healthy |
| `api` | the FastAPI service; chunk table, parsed documents and embedding cache live in the `hyrag-data` volume | the seed exited successfully |
| `dashboard` | the Streamlit page, talking to `http://api:8000` | the API reports ready |

If the seed fails, it exits with an error and Compose never starts the API. That is deliberate: a half-built index would give wrong "not found" answers without any visible error.

Ports are bound to 127.0.0.1 only: 8501 for the dashboard, 8000 for the API. Chroma is not published at all. To expose the service, put a reverse proxy with TLS in front of it. Container logs are capped at three files of 10 MB.

### What to expect

| | Time |
|---|---|
| First build | about 13 minutes, mostly the PyTorch download |
| First start, indexing 399 documents into 7,628 chunks | about 5 minutes |
| Later starts | about 8 seconds; the seed embeds nothing |
| First question after a start | about 6 seconds |

Memory at rest was 405 MB for the API, 62 MB for Chroma and 50 MB for the dashboard. That was measured before PostHog's handbook was added and has not been re-measured since; the API now holds six times as many vectors in memory.

The first start is slow because Mistral's free tier allows 60 embedding requests a minute. The seed waits out the limit and carries on.

### The images

The API image is 2.6 GB. Of that, 770 MB is PyTorch, in its CPU-only build. The default PyTorch package for Linux bundles CUDA and is several gigabytes larger, and the build fails if that one gets installed by mistake. The reranker model is downloaded during the build, so the container never needs the Hugging Face Hub at runtime.

The dashboard image is 776 MB and has neither PyTorch nor the `hyrag` package.

Both containers run as a non-root user. The API keys come from `.env` at runtime and are not baked into any image.

## Configuration

All of it is environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `MISTRAL_API_KEY`, `GROQ_API_KEY` | none | required |
| `HYRAG_DATA_DIR` | `data` | chunk table, parsed documents, embedding cache |
| `HYRAG_CHROMA_URL` | unset | a Chroma server, e.g. `http://chroma:8000`; unset uses an embedded store in the data directory |
| `HYRAG_API_KEY` | unset | if set, `/v1/ask` and `/v1/documents` need the header `X-API-Key` |
| `HYRAG_ADMIN_KEY` | unset | needed for `/v1/ingest`; while unset, uploads are disabled |
| `HYRAG_BUDGET_SECONDS` | `60` | time budget per question |
| `HYRAG_MAX_UPLOAD_MB` | `20` | per-file upload limit |
| `HYRAG_CORS_ORIGINS` | unset | origins allowed to call the API from a browser |
| `HYRAG_STRATEGY` | `structure` | which chunking strategy's index to serve |
| `HYRAG_VISITOR_LIMIT_PER_HOUR` | `0` (off) | questions one visitor may ask per hour |
| `HYRAG_DAILY_LIMIT` | `0` (off) | questions per day in total, across all visitors |
| `HYRAG_API_URL` | `http://localhost:8000` | dashboard only: where the API is |

## Adding documents

Put files under `corpus/` and start the stack again; the seed picks them up. Or upload them to a running API:

```bash
curl -X POST "localhost:8000/v1/ingest?collection=posthog" \
     -H "X-API-Key: $HYRAG_ADMIN_KEY" -F "files=@expenses.md"
```

A file with the same name in the same collection replaces the earlier version. Only the chunks whose text changed are embedded again.

## When something fails

The Chroma and Mistral rows were tested on the running stack. The rows about the models and the time budget were tested with simulated failures.

| What happened | Result |
|---|---|
| Chroma's volume was deleted while the chunk table was kept | The next seed found every vector missing and restored them from the embedding cache without calling Mistral |
| Chroma stopped while the API was running | Questions kept being answered, by keyword search, and were marked degraded. Search vectors have since moved to the local cache, so meaning search should no longer need Chroma at all; that has not been re-tested |
| The API was restarted while Chroma was down | It refused to start with a clear message, and Docker kept restarting it |
| Chroma came back | The API was healthy 21 seconds later, with no manual step |
| Mistral rate-limited the seed | It waited and continued |
| Mistral unreachable during a question | Retrieval fell back to keyword search and the response is marked degraded |
| The writer model unavailable, or its daily quota used up | A structured `error` response, HTTP 503 |
| The citation checker unavailable | The answer is shown with an "unchecked" warning, and confidence is capped at low |
| A question ran past 60 seconds | `error.timeout`, HTTP 504 |

Two of these were bugs first. The seed used to give up on Mistral's rate limit after 15 seconds of retrying, which is shorter than the one-minute window the limit resets in. And the API used to fetch every vector from Chroma on the first question after a start. With 7,628 chunks that took longer than the question's whole time budget, so the first question always failed. The vectors now come from the local cache, at startup.

## Question limits

Every question spends the Groq and Mistral quota of whoever owns the keys, and the free tier covers roughly 115 checked questions a day. Two limits protect that, both off by default.

`HYRAG_VISITOR_LIMIT_PER_HOUR` caps one visitor, and `HYRAG_DAILY_LIMIT` caps everyone together. A question over the limit gets HTTP 429 with a message saying which limit it was and a `Retry-After` header; the model is never called. Answers served from the cache and questions that failed because of an outage don't count, and a visitor at their limit can still get an answer that is already cached.

The dashboard identifies a visitor by a hash of their forwarded address when a proxy supplies one, and by a random id per browser session otherwise. No address is stored. The API takes that id on trust, so the per-visitor limit stops ordinary overuse and not someone who forges ids. The daily total is the one that can't be dodged. Both live in memory and reset when the process restarts.

## One process: Streamlit Community Cloud

Some hosts run a single process and no containers. `deploy/streamlit_app.py` is the entry point for those. On a cold start it indexes the documents, starts the API in a background thread, and then serves the normal dashboard against it on localhost. It sets the limits above to 10 questions per visitor per hour and 100 a day unless the environment says otherwise.

```bash
streamlit run deploy/streamlit_app.py      # locally, with the two keys in .env
```

To deploy on [Streamlit Community Cloud](https://share.streamlit.io):

1. Sign in with GitHub and create an app from this repository, branch `main`, main file `deploy/streamlit_app.py`.
2. Under Advanced settings choose Python 3.12, and paste the two keys into the Secrets box:

   ```toml
   MISTRAL_API_KEY = "..."
   GROQ_API_KEY = "..."
   ```

3. Deploy. The app installs `deploy/requirements.txt`, which sits next to the entry point and pins PyTorch's CPU build.

Three things make this work without a separate vector database or a seed job.

`deploy/embeddings.sqlite` holds the vector of every chunk in the index, 36 MB. The host's disk is empty on every cold start, and with this file the documents are indexed without a single embedding request. Regenerate it with `scripts/export_embedding_cache.py` after the documents change.

Chroma runs embedded, inside the same process, in the data directory.

Documents are parsed one at a time in the app's own process (`python -m hyrag.seed --in-process`). The normal seed uses worker processes, and starting those from inside a running Streamlit app is fragile.

Measured locally from an empty data directory, on a machine short of memory: 179 seconds to a ready page, of which 78 were indexing, with no embedding requests. Once the backend is up, a page loads in 3.6 seconds and a question was answered in 3.6 seconds. Memory was not measured in this mode; the API and dashboard used about 800 MB together when measured separately in Docker, and Community Cloud allows up to 2.7 GB.

What to know about this host: the app sleeps after 12 hours without visitors, and the next visitor waits through a cold start. Whether its installer honours the `--extra-index-url` line in `deploy/requirements.txt` is not confirmed by its documentation; if the build pulls the CUDA build of PyTorch, that line is the place to look.

## Before putting it on the internet

A few things matter that don't on a laptop.

Set the question limits described above. Without them a public demo can spend its whole daily quota in minutes.

The dashboard has no sign-in. If the API requires a key, the dashboard holds it on the server side and every visitor uses it. Put it behind a login proxy.

The API keys are plain environment variables, visible to anyone who can run `docker inspect`. In production they belong in a secrets manager.

Only one API process can run, because the answer cache and the upload lock are in memory.

A public demo would be showing PostHog's handbook. Its license allows that. The page should say that HyRAG is not affiliated with PostHog.
