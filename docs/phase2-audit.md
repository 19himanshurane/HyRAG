# HyRAG Phase 2 audit

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
