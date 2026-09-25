# HyRAG Phase 2 audit

## Round 2 (2026-09-25): Step 4, cross-encoder reranking

**Scope:** `hyrag/rerank.py`, the reranking path in `hyrag/retrieval.py`, `tests/test_rerank.py`, `try_rerank.py`, the dependency and lock changes, and a regression pass over round 1. The Step 4 code was audited while it was still uncommitted. **Method:** Phase A was read-only, with measured probes. Every load time came from a fresh process, and the slow cases were run twice: one early 105 s reading and one 29 s reading were machine noise and are not used below. One real Mistral request was made (the over-long query in S3).

```bash
PYTHONPATH=. python scripts/audit/probe_rerank.py      # this round (needs the model cached: python -m hyrag.rerank)
PYTHONPATH=. python scripts/audit/probe_recall.py      # S6: meaning search vs exact brute force (needs the real index)
PYTHONPATH=. python scripts/audit/probe_retrieval.py   # regression: round 1
python -m pytest -q                                    # pytest -m "not model" skips the ~20 s real-model test
```

### Results after fixes

All six fixed (S6 below). 170 tests pass; every probe passes.

| ID | Sev | Before | After | Verified by |
|---|---|---|---|---|
| S1 | High | The default `local_files_only=False` let a cached, pinned model contact the Hub on every load. With the Hub unreachable, loading took **250 s** (reproduced twice), and every reranking request waited behind the load lock. | Offline by default (`local_files_only=True`). The model is fetched once, explicitly: `python -m hyrag.rerank` (`download()`), or automatically by `try_rerank.py` when `is_downloaded()` is false. With the Hub unreachable, loading now takes **10.0 s** (twice), the same as with no network problem. | test (default); probe 8; timing runs in fresh processes |
| S2 | High | A reranker exception failed the whole search, although the fused results were ready. A failed load wasn't remembered, so every request re-tried it (10–27 s each). | `retrieve()` falls back to the fused top `rerank_top_n`, logs the error, and leaves `rerank_score` `None` so callers can see the results weren't reranked. A failed load raises `RerankerUnavailable` (with the "run `python -m hyrag.rerank`" hint) and is re-tried only after 60 s: the next request fails in **0 ms**, down from ~12 s. | 2 tests (fallback; cool-down with a fake clock); probes 1–2 |
| S3 | Med | A 46,830-char query (a pasted log) got **400 Bad Request from Mistral after a request was sent**. The reranker would have silently cut its 18,014 tokens to 512. | `MAX_QUERY_CHARS = 2000`: rejected with a `ValueError` before any search, **0 requests**. | test (limit and limit+1); probe 6 |
| S4 | Low | `top_k=3, rerank_top_n=10` was accepted. | Rejected when a `HybridRetriever` **with a reranker** is built. A first version put the check in `RetrievalConfig`, which broke 5 existing tests that use `top_k=3` with no reranker, a legitimate setup. The check moved to where `rerank_top_n` is actually used. | test; probe 3 |
| S5 | Low | The README didn't mention PyTorch, the model download, or the memory cost. | "Try it" section: index, `python -m hyrag.rerank`, `try_rerank.py`; PyTorch size (and the Linux CUDA-wheel trap); ~10–13 s load and **~590 MB RAM** (measured: working set 30 → 620 MB). | read |

### Checked and fine
- **Reranking holds no index lock:** a keyword search takes 1 ms while the model scores.
- **Deterministic:** a pair scored alone vs inside a padded batch of 20 differs by 2.4e-7, and repeated batches are identical.
- **Model files:** `model.safetensors` only (no pickle), `trust_remote_code` off, revision pinned to a commit hash.
- **Truncation:** 1 of 1,266 chunks exceeds the 512-token window, and questions are now capped (S3).
- **Regression:** round 1's probes pass 15/15. All 165 tests pass. The real demo's ranks are unchanged (4012: fused #2 → #1; credentials: fused #3 → #1).

### S6 (High, found during this round's regression checks): meaning search missed real top-10 results

The Friday question's candidate pool was 15 chunks in some runs and 16 in others. The cause was Chroma's HNSW index, which is approximate. Measured against exact brute-force cosine over all 1,266 vectors:

| Collection | ef_search 100 (default) | ef_search 500 |
|---|---|---|
| `data/index` (graph after Phase 1's many delete/re-add cycles) | missed 1–2 of the exact top 10 for 3 of 6 questions; **which** chunks were missed varied between processes | still missed 1 (the 4012 question) |
| a fresh collection from the same vectors | missed 1–3 | **0 misses** (two runs) |

These were real misses, not near-ties: the credentials question lost the exact #4 (cosine 0.750) while Chroma returned a chunk at 0.739. Two causes: mistral-embed vectors are low-contrast (everything scores 0.65–0.85), so the default search width is too narrow, and the graph degrades with churn.

**Fix (chosen by the user over tuning HNSW):** meaning search ranks by **exact cosine** over an in-memory copy of the stored vectors (`ChunkIndex._nearest` / `_vector_matrix`). Chroma remains the vector store and stays reconciled with the chunk table. The copy is rebuilt on demand and dropped whenever BM25 is: on every write, on `repair()`, and when another process or index object commits (`PRAGMA data_version`). Ids are sorted, so ties break the same way in every process. A query vector with zero or non-finite length returns no results with a warning, instead of ranking by NaN.

| | Before (Chroma HNSW) | After (exact) |
|---|---|---|
| exact top-10 chunks missed, 6 questions × 5 processes | 1–2 in 3 of 6 questions, varying per process | **0** in every process |
| Friday candidate pool across processes | 15 or 16 | **15** every time |
| dense search latency (cached query embedding) | 2–3 ms | **0.7–0.8 ms** (first query per process: +135–450 ms to build the copy) |
| memory | none extra | 4 KB per chunk (5 MB now; ~400 MB at 100k chunks, the point to go back to a tuned approximate index) |

Verified by 5 tests: equals brute force, refreshes after this object's writes, after another object's writes, and after `repair()`, plus the unusable-vector guard. Every probe from rounds 1–2 and Phase 1 round 3 passes. The real demo's ranks are unchanged. Note: comparing the new search with brute force is nearly tautological (both compute the same thing); what it proves is that the approximate path is gone and the in-memory copy matches Chroma.

---|---|---|
| `data/index` (graph after Phase 1's many delete/re-add cycles) | misses 1–2 of the exact top 10 for 3 of 6 questions; **which** chunks are missed varies between processes | still misses 1 (the 4012 question) |
| a fresh collection from the same vectors | misses 1–3 | **0 misses** (two runs) |

These are real misses, not near-ties: the credentials question lost the exact #4 (cosine 0.750) while Chroma returned a chunk at 0.739. Two causes: mistral-embed vectors are low-contrast (everything scores 0.65–0.85), so the default search width is too narrow, and the graph degrades with churn. Exact search over the same vectors takes **0.25 ms** per query (Chroma: 2–3 ms). How to fix it is a design decision and was left to the user: see the checkpoint in the conversation log.

---

## Round 1 (2026-09-24): Step 3, hybrid retrieval (RRF fusion)

**Scope:** `hyrag/retrieval.py`, `tests/test_retrieval.py`, `try_retrieval.py`, plus the index code they depend on (`ChunkIndex.search_dense` / `search_sparse` and its locking). The Step 3 code was audited while it was still uncommitted.

**Method:** Phase A was read-only, with every suspicion measured by a probe before it was ranked. Probes used a fake embedder with an artificial network delay and throwaway directories. Two real Mistral requests were made on purpose: one to check what a blank query does in production, one to time an uncached query.

```bash
PYTHONPATH=. python scripts/audit/probe_retrieval.py   # this round (offline, except the real-index latency check: cached, 0 requests)
PYTHONPATH=. python scripts/audit/probe_round3.py      # regression: index + dedup
python -m pytest -q
```

## Results after fixes

All six were fixed. The two new concurrency tests **fail on the previous `index.py`** and pass now. Every probe from this round and from Phase 1 round 3 passes.

| ID | Sev | Before | After | Verified by |
|---|---|---|---|---|
| R1 | High | `search_dense` held the index lock during the Mistral request, and so did `index_document` / `repair` while embedding. 8 concurrent queries took **2.43 s** against 0.30 s for one (fully serialized). A keyword search took **254 ms** because it waited on another request's embedding. Retries (Retry-After ≤ 60 s) could freeze all searching and indexing. | Two locks. The **state lock** covers only local work: SQLite, BM25, the dedup lookup, Chroma calls. The **writer lock** serializes whole writes (plan, embed, apply) so their plan can't go stale. Embedding runs with no state lock held. Lock order is writer, then state. The embedder's `requests_made` counter got its own lock (`+= 1` isn't atomic). Now 8 concurrent queries take **0.32 s** and the keyword search **1 ms**. | 2 deterministic tests (an embedder that blocks until released: a search must finish while a query or an ingest is stuck in `embed`); probe 6; a stress test with 4 writers and 4 readers × 15 random index/remove ops: 0 errors, in sync afterwards |
| R2 | High | `""` against real Mistral returned **10 arbitrary chunks** (two tables of contents and Acknowledgments) and used an API request. | `retrieve()` raises `ValueError("empty query")`. `search_dense` returns `[]` for a blank query without embedding, matching `search_sparse`. | 3 parametrized tests (the embedder receives nothing); probe 5 |
| R3 | Med | `mode="Hybrid"` returned 0 results and no error. | `ValueError` listing the valid modes. | test; probe 1 |
| R4 | Med | `dense_k=0` crashed inside Chroma (`TypeError: Number of requested results 0`). Negative weights and negative `rrf_k` were accepted and inverted the ranking. `top_k=0` returned nothing. | `RetrievalConfig.__post_init__` checks: counts ≥ 1, weights ≥ 0 and not both 0, `rrf_k` ≥ 0. A list with weight 0 is not searched (no wasted API call). | 6 parametrized tests + 1 zero-weight test; probe 2 |
| R5 | Low | A chunk listed twice in one list scored 0.0325 while its recorded contribution was 0.0161 and its rank 2. An unknown list name created a stray `bm25_rank` attribute. (Not reachable through `HybridRetriever`, but the function is public.) | Only a chunk's first (best) rank in a list counts. List names other than `dense` / `sparse` raise `ValueError`. | 2 tests; probes 3–4 |
| R6 | Low | An index that was never built returned 0 results with no warning, indistinguishable from "no match". | `retrieve()` logs a warning when the index is empty. `try_retrieval.py` stops with "run try_index.py first". | test; probe 7 |

## Checked and fine

- **Latency** (real index, 1,266 chunks, cached embeddings, medians): sparse 1.6 ms, dense 2.3 ms, hybrid 3.8 ms. The first call of each kind is 200–270 ms (BM25 build, Chroma warm-up). An uncached hybrid query is **282 ms**, almost all of it the embedding request.
- **Determinism:** equal fused scores are ordered by chunk id, and the order doesn't depend on which list is passed first.
- **Behaviour unchanged:** after the fixes, `try_retrieval.py` prints the same ranks and scores as before.

## Known limits (by design, handled later)

| Limit | Evidence | Where it is handled |
|---|---|---|
| RRF discards score gaps | `ERR_TUNNEL_4012`: BM25 37.3 vs 17.1 points becomes "each list has one first". At 0.7/0.3 the wrong error wins (0.01631 vs 0.01621). At 0.5/0.5 it's an exact tie, won by chunk id, which is luck. | Step 4, reranker |
| Keyword noise in fusion | "I forgot my login credentials": BM25 has no real match, but a Kubernetes chunk matching filler words ("my", "do") takes the top slot at 0.5/0.5. At 0.7/0.3 no sparse-only hit reaches the top 5 for any of the 4 questions. | Phase 4: weights / BM25 score floor tuned on the golden set, not by hand |
| Off-topic or filler-only questions still get results | `"what is the"` → 5 dense-only results | Phase 3 confidence score |

## Self-review notes

- **R1 was a mistake of mine**, introduced by the Phase 1 round-3 fix H3. That fix made the index correct under threads and was tested for correctness, but never under realistic network delay. The rule now: never hold a lock across network I/O, and measure a concurrency fix with a slow dependency.
- The first version of `try_retrieval.py` looked for the target code only in chunk headings. It reported SQLSTATE 23505 as "not retrieved" when it was ranked #1: the code sits in a table row under the heading "PostgreSQL Error Codes". This was caught before it was reported. The check now matches the code as a whole token in heading + text.
