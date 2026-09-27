# HyRAG Phase 3 production audit

## Round 1 (2026-09-27): generation, citation checks, confidence, and the `ask()` pipeline

**Why:** Phase 3 had only been validated on 26 hand-picked questions. That validation said nothing about production behaviour: outages, hangs, cost, capacity and concurrency. It also tuned the thresholds on the same questions it reported results on.

**Method:** Phase A was read-only, with every suspicion measured before it was ranked. Offline probes use fake models, a fake HTTP transport and a throwaway index. The live probe ran 6 questions × 3 runs through the real pipeline. After the fixes, every probe was re-run.

```bash
PYTHONPATH=. python scripts/audit/probe_phase3.py            # offline: outages, hangs, budget, observability, concurrency
PYTHONPATH=. python scripts/audit/probe_phase3_live.py 3     # live: latency, requests, tokens, capacity, stability
PYTHONPATH=. python scripts/eval_judge.py openai/gpt-oss-20b low --batch 4   # judge accuracy (32 labeled pairs)
PYTHONPATH=. python scripts/eval_confidence.py               # regression: calibration questions end to end
python -m pytest -q
```

## Results

| ID | Sev | Before | After | Verified by |
|---|---|---|---|---|
| H1 | High | The writer (Groq) failing raised `RuntimeError` out of `ask()` | `status="error"`, `code="error.writer_unavailable"` (or `error.timeout`). It still lists documents worth checking, and the exception is logged with its stack trace (so a programming bug stays visible instead of hiding as "unavailable") | offline probe 1; tests |
| H2 | High | Mistral (the query embedding) failing raised, so a total outage even though BM25 and the reranker run locally | `HybridRetriever.search()` falls back to keyword search + reranking; `degraded=["dense_search_unavailable"]`, and the message says "keyword-only mode" | offline probe 3 (answered, degraded); tests |
| H3 | High | No deadline: one Groq call could take **10.2 min** (5 × 120 s + backoff), and a question made several calls in a row | `hyrag.http.deadline()`: one budget per question (default **60 s**), shared by every HTTP call in it. Each attempt's timeout is cut to the time left; no retry sleeps past the end. 30 s per attempt | offline probe 4b: a server that hangs on every request, budget 3 s → returned in **3.0 s** with `error.timeout`; `tests/test_http.py` |
| H4 | High | Capacity **5.6 questions/min** for everyone; the judge cost more tokens than the writer (one call per claim, instructions repeated); **38 rate-limit retries** in 18 sequential questions | Batched judging: one request checks all of an answer's claims, each passage sent once. `AnswerCache`: a repeated question costs nothing until the index changes. A JSON-schema failure from the model is regenerated (a 400 that is a bad sample, not a bad request) | live probe (table below); judge accuracy unchanged |
| M1 | Med | A judge outage made every answer "unverified", with a reason blaming the answer | New status `unchecked` (`unchecked.checker_unavailable` / `unchecked.timeout`): the answer is shown with a warning (`show_unchecked=True`) or withheld; confidence is capped at low. An outage is no longer presented as evidence that the answer is wrong | offline probe 2; tests |
| M2 | Med | Responses carried no timings, request counts or tokens | `request_id`, `timings_ms` (retrieve / generate / verify / score / total), `usage` (requests and tokens per model), `cached`; one JSON log line per question (the question text is **not** logged, only its length and a hash, because questions can contain personal data) | offline probe 5; tests |
| L1 | Low | One unsupported sentence withheld the whole answer, verified parts included | The answer is still withheld (the safe default), and `verified_claims` lists the sentences that passed | tests |
| L2 | Low | English-only messages | A stable `code` on every response, so a UI can show its own language. The pipeline itself worked in Spanish (answered and verified; the printer question was stopped by the gate) | tests; live check |
| M3 | Med | Thresholds tuned and reported on the same 26 questions | **Carried to Phase 4** as a requirement: a golden set with a held-out split that no threshold is tuned on | — |
| M4 | Med | 1 of 6 questions changed confidence level across 3 identical runs | **Carried to Phase 4**: report spread over repeated runs, and the rate of correct answers withheld, not single numbers | — |

### Live measurements (6 questions × 3 runs, real APIs)

| | Before | After |
|---|---|---|
| Median / worst time per question | 7.3 s / 40.5 s | **5.1 s / 20.9 s** |
| Judge requests per question | 3.2 (up to 8) | **2.1 (up to 3)** |
| Capacity (tokens/min limit ÷ tokens per question) | 5.6 q/min | **4.8 q/min, every answer actually checked** (see the judge section) |
| Same status/level in all 3 runs | 5/6 | 5/6 (the caregiver-policy answer, as before) |

## What went wrong during the fixes (all found by re-measuring)

1. **The judge had to change.**
   - After batching, answers still ended "unchecked". The cause was on the judge's side of Groq: **qwen/qwen3.8-27b has a 1,000 output-tokens-per-minute limit on this plan**, and a request whose output cap exceeds it is refused outright ("Request too large … Limit 1000, Requested 2048").
   - It also spends up to ~800 hidden tokens on a heavy check. `reasoning_effort: "none"` didn't change that (measured).
   - That's about one check a minute. The original choice of Qwen cited its "own rate-limit budget" without measuring that budget.
   - **Switched to openai/gpt-oss-20b:** 0/16 false support in every batched run (7 runs, 29–31/32 exact), no output-rate limit, its own 8,000 tokens/min budget.
   - **The price:** it is the same family as the writer (gpt-oss-120b), so a shared blind spot is possible. Its reasoning tokens also make the judge the capacity limit (4.8 q/min), below the 6.8 q/min Qwen appeared to allow. But that 6.8 was not real: it came from answers silently ending "unchecked".
2. **I cut Groq's attempts from 5 to 3** (for H3), which turned rate-limit bursts into unchecked answers. Reverted: the per-question budget is what bounds the time, not the attempt count.
3. **A "reply cut off" check discarded complete verdicts.** The model's hidden reasoning used up the output budget *after* writing complete, valid JSON. The check now fails only when the JSON is actually broken.
4. **Quote verification rejected correct evidence twice:**
   - A judge joined two real sentences that are not adjacent in the passage, so a correct max_connections answer was withheld. Fix: each quoted sentence is checked on its own, and every one must appear word for word.
   - My first version of that fix also demanded 4+ words, which rejected the exact, short table quote `23505 | unique_violation`. The minimum was removed.
5. **A prompt tweak** ("join separate pieces with ...") fixed its target case in isolation. Over 7 runs it made no measurable difference (29–31/32 with and without), so it was reverted to keep the prompt simpler. I first called it "consistently worse" after 2 vs 4 runs; one more baseline run showed that was noise.

## Hard limits (plan, not code)

This Groq account is on the free on-demand tier. The limits below come from its response headers and error messages (2026-09-27/28):

- **Tokens per DAY, the binding limit:** `openai/gpt-oss-20b` (the judge) allows **200,000 tokens/day**. This was only discovered when the evaluation ran it dry ("Rate limit reached … tokens per day (TPD): Limit 200000, Used 199675 … try again in 7m9s"). At ~1,700 judge tokens per question, that is **about 115 fully checked questions per day**.
  - The writer model (gpt-oss-120b) uses ~1,100 tokens per question; its daily token limit has not been observed yet.
  - **Correction:** this report first said "about 475 checked questions per day", computed from the 1,000 requests/day limit. The token limit binds long before the request limit.
- **Tokens per minute:** 8,000 per model, which allows about 4.8 checked questions per minute in bursts.
- **Requests per day:** 1,000 per model.

When a daily limit is reached, the client now fails at once with `QuotaExhausted`. `ask()` reports `error.quota_exhausted` (writer) or `unchecked.quota_exhausted` (judge) instead of retrying every 60 s against a limit that refills over minutes to hours.

Code can't raise these limits. The options are the answer cache (repeats are free), a paid tier, or self-hosting. The response's `usage` field and the log line make them possible to monitor.

## Known limits
- **The judge's quoting is noisy on long answers:** the caregiver-policy answer (6 claims) scored high once and medium or low in other runs, depending on which claim the judge doubted. It fails safe (withheld or flagged, never falsely verified). Phase 4 must measure how often a *correct* answer is withheld.
- **`show_unchecked=True`** (the default) shows an unchecked answer with a warning; strict deployments should set it to False.
- **The cache is per process.** Several server processes would need a shared cache (e.g. Redis) keyed the same way (question + index version).
