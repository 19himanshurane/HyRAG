# HyRAG Phase 4: evaluation

Three steps from the brief: (1) a golden question set, (2) automated metrics run on every pipeline change, (3) comparing the three chunking strategies on it. Two requirements carried over from the Phase 3 audit shape everything below:
- **M3:** thresholds were tuned and reported on the same questions. The golden set therefore has a held-out test split that nothing is ever tuned on.
- **M4:** results vary between runs. Metrics are therefore reported as a spread over repeated runs, including the rate of correct answers the system withholds.

---

## Step 1 (2026-09-28): the golden set

**Files:**
- `eval/golden_set.json`: 60 questions with their answer key.
- `eval/golden_sample.json`: the random draw of sections.
- `scripts/sample_golden_sections.py`: makes the draw.
- `scripts/check_golden.py`: verifies the answer key and locks the split.

### What each question records
- **`question`**, phrased the way a user would ask, not copied from the section. Copying the section's wording would favour keyword search.
- **`answer`** (a reference) and **`key_facts`**: what a correct answer must contain. Grading against listed facts is more objective than "does this look right".
- **`evidence`**: document, section and an exact quote. It is recorded this way, not as a chunk id, because chunk ids differ between chunking strategies and Step 3 needs one answer key for all three.
- **`expected`**:
  - `answer`;
  - `not_found` (must not invent anything; where noted, stating what *is* documented is acceptable);
  - `clarify_or_cover` (several reasonable readings: ask, or cover each one labelled).

### Composition (60)

| Type | Count | How chosen |
|---|---|---|
| lookup | 34 | sections **drawn at random** (seed 20260928, sections of 40+ words, spread over all 16 documents: more from large ones, at least 1 each), then a question written about whatever was drawn, including dull sections (a patent notice, an abstract) |
| multi_hop | 10 | pairs of documents on a shared topic. Random pairs would not make sensible questions, so these were chosen by hand; this is a known bias. Examples: Nimbus password rules vs NIST 800-63B; Nimbus vs GitLab sick-note rules; a Waiting pod + imagePullSecrets |
| no_answer | 10 | 5 out of scope and 5 near misses. Each near miss sets a trap: a VPN port (the only "port" in the VPN guide is "IT portal"), `statement_timeout` (other PostgreSQL timeouts are documented), the VPN on Linux (Linux appears only in unrelated documents), Nimbus bereavement leave and Nimbus leave notice (only GitLab's policies cover them) |
| ambiguous | 6 | several readings in the corpus: "How long does my password have to be?" (Nimbus 14 characters vs NIST 15), "Do I need approval to deploy?" (staging vs production), and questions too vague to answer ("What does this error mean?") |

### Checks (`scripts/check_golden.py`, all against the parsed corpus)
- **All 86 evidence quotes** occur word for word in their named document, and every named section exists.
- **Every no-answer topic is absent.** Checking *before* labelling mattered: PodDisruptionBudget, pets, health insurance and "badge", all planned as no-answer topics, are in the corpus, and were dropped.
- **No question repeats one the thresholds were tuned on** (the calibration and near-miss sets). Two did: "fix ERR_TUNNEL_4014" repeated the 4099 tuning probe, and a public-holiday deploy question reused the Friday/Saturday template. **Both were replaced**, not reworded to dodge the check. One word-overlap hit ("What does this error mean?") is a documented exception: it tests a question with no error code at all.
- **Split:** assigned once, stratified by type, with a fixed seed. **24 dev / 36 test** (lookup 14/20, multi-hop 4/6, no-answer 4/6, ambiguous 2/4). The test questions are fingerprinted, and the checker fails if any of them changes afterwards (verified by editing one on a copy).

**Rule from here on:** look at, debug and tune on **dev** only. Test is scored at the end, and a test result is never used to change the system.

### Known limits
- **I wrote all 60 questions.** Random section sampling limits *which* topics I picked, but not *how* I phrased them. Questions from someone who has not seen the documents would test more realistically. The file is designed so they can be added (new ids, run the checker).
- 60 questions, 36 in test: a difference of one or two answers is noise. Step 2 reports spreads, not single numbers.
- **Parser observation** (from the random draw): PostgreSQL pages keep their version-navigation line ("Supported Versions: Current (18) / 17 …") as section text. It's page furniture, like the release banner removed in Phase 1. It stays for now because it is the only source for G19; to revisit with the chunking comparison.

---

## Step 3a (2026-09-28): chunking strategies compared on retrieval (dev, no LLM)

**Code:** `scripts/build_index.py --strategy …` (one index per strategy, same 800-char target size) and `scripts/compare_chunking.py` (evidence found by exact quote in the retrieved chunks; the local reranker; no Groq). **Result file:** `eval/results/chunking-retrieval-dev.json`. **Dev split only**: picking a strategy is tuning.

| Strategy | Chunks | Median chars | recall@5 | hit@5 | all@5 | recall@20 | MRR |
|---|---|---|---|---|---|---|---|
| fixed | 884 | 800 | 0.78 | 0.89 | 0.78 | 0.91 | 0.67 |
| **structure** | 1,266 | 528 | **0.83** | **0.94** | **0.83** | **0.96** | **0.91** |
| semantic | 1,307 | 471 | 0.83 | 0.94 | 0.83 | 0.91 | 0.86 |

(18 answerable dev questions, 23 evidence quotes: one quote is 0.04.)

- **Structure vs fixed:** in all 6 questions where they differ, structure ranks the evidence as high or higher (by chance about 1 in 64). Fixed-size chunks cut passages mid-way, so the evidence is diluted or split.
- **Structure vs semantic:** no meaningful difference (G07 rank 1 vs 5, and one quote in the top 20). Semantic also costs more to build (93 embedding requests and 77 s here).
- **So far structure is the default to keep**; the end-to-end comparison (answer quality) is Step 3b.

**Answer-key corrections found on the way (dev only):**
- G07 listed only the Abstract as evidence. The document states its purpose three times (the Abstract, 1.1 Purpose and Scope, and the Appendix C change log), so a correct retrieval was scored as a miss.
- Evidence items can now list `alternatives`, and any one counts. **Disclosure:** the change-log alternative was noticed while reading *structure's* results. It is a genuine answer passage, but the way it was found favoured structure; without G07, structure still leads on rank in the other 5 disagreements.
- **Retrieval scores are a lower bound** wherever a document repeats itself and the answer key names only one place.

**Finding for the gate (not chunking):**
- G38 ("I'm working from home and want to push a change to production on Wednesday afternoon. What do I need?") scores **−5.80** on the reranker for the right section under every strategy. That is below the −3 gate, so `ask()` would answer *not found* without asking the model.
- The gate was calibrated on short, direct questions; conversational phrasing scores lower. This is the M3 risk (a threshold tuned on one question style), caught by the golden set.
- The VPN rule G38 also needs is never retrieved: the question never says "VPN" (a multi-hop retrieval limit).

### The gate across strategies (dev and calibration questions, no LLM)

The retrieval gate (rerank logit −3) was calibrated on **structure** chunks. The same questions under each strategy:

| Strategy | Dev answerable blocked | Calibration: lowest answerable | Calibration: highest unanswerable | Gap for the gate? |
|---|---|---|---|---|
| structure | 1/18 (G38) | +0.08 | −5.83 | yes, wide |
| semantic | 1/18 (G38) | −1.05 | −4.00 | yes, narrower |
| **fixed** | 1/18 (G38) | **−8.73** | −4.96 | **none**: an answerable question scores below unanswerable ones |

Under fixed-size chunks the reranker's confidence no longer separates answerable from unanswerable questions, so no single gate could work well for it. Step 3b therefore compares the strategies with the same gate: that is fair as a product outcome, and the report says so.

Moving the gate to rescue G38 (−5.80) would also admit the most plausible unanswerable calibration question (−5.83): that would be tuning on one dev question. The fix belongs in retrieval (for example rewriting conversational questions into search queries), decided with the full end-to-end numbers.

---

## Step 2 (in progress): running the evaluation, and what running it revealed

The first dev run (24 questions × 3) was stopped after about an hour, having scored only 9 questions. Diagnosing it took three hypotheses, and **the first two were wrong**:

1. *"It hung because I rebuilt the index from another process at the same time."* **Wrong.** A later run hung with no other process touching the index.
2. *"It is hitting the 8,000 tokens-per-minute limit."* **Wrong.** The per-minute budget refills in under a second, as the headers show. A client-side pacing layer was written on this theory, then **removed**, because it fixed a problem that wasn't happening.
3. **Right:** a stack dump from a watchdog showed the process sleeping in a retry. Groq's own 429 message named the limit: **200,000 tokens per day** on the judge model, used up by the day's evaluation work (judge comparisons, grader validation, partial runs).

What changed as a result:
- **`scripts/run_eval.py` is crash-safe.**
  - Every scored question is appended to a `.jsonl` file at once; `--resume` continues a stopped run; `--summarize` rebuilds the report from the file.
  - A `faulthandler` watchdog prints every thread's stack if a question runs far past its budget (this is what found the real cause).
  - If the daily quota runs out, the run stops **without** saving that question and prints the resume command.
  - The first version saved results only at the end, so an hour of work was lost.
- **Daily-limit 429s fail fast** (`QuotaExhausted`) instead of being retried every 60 s. `ask()` has codes for them.
- **Faithfulness only checks claims the pipeline has not already verified** against their citation (those are supported by definition). For a fully verified answer that means no extra request.
- **The production capacity figure was corrected** in docs/phase3-audit.md (~115 checked questions per day, not ~475).

**Budget for evaluation from here:** one dev run of 24 questions costs roughly 60–90k judge tokens, so about 2 runs fit in a day. Runs are scheduled against that budget: resume when the rolling daily window has refilled.

Early observations from the 9 questions scored before the stop (to re-check in the full run, not conclusions):
- G12 and G15 graded not fully correct.
- G14's grading failed.
- G09 was withheld as unverified although its draft was correct.

## Query rewriting for questions the gate would refuse (dev, 2026-09-28)

**Code:** `hyrag/rewrite.py` and `HybridRetriever(rewriter=...)`. A question whose best rerank logit is below −3 (the answer gate) is rewritten by the writer model into 1–3 focused search queries (strict JSON). Each query is retrieved separately; each query's best hit is guaranteed a place in the top 5, then the rest fill by score. Only questions that would otherwise be refused pay for it (one writer request). If the rewrite fails, the original results are kept (`degraded: query_rewrite_failed`). `Response.search_queries` shows what was searched. **Tests:** `tests/test_rewrite.py`.

Measured with the real rewriter and reranker (retrieval only, no judge):

| Question set | Gate result: without → with rewriting |
|---|---|
| dev answerable (18) | 1 blocked → **0 blocked**: G38 rescued |
| dev unanswerable (4) | G45, G46 stay blocked; G50, G54 passed already |
| calibration unanswerable (10) | 10 blocked → **7 blocked**: laptop request, Slack password and on-call pay now reach the writer |

- **Did the next layer hold?** The writer refused all three newly admitted unanswerable questions with the exact "not found" sentence, **6 of 6 runs**.
- **G38 is now answered correctly and cited:** team-lead approval within the Monday–Thursday 10:00–16:00 UTC window, "which includes Wednesday afternoon".
- **Still missed:** G38's VPN requirement. The model folded it into "remote access policy for production deployment" instead of a separate query. Not tuned away: fitting the prompt to one dev question would be overfitting. It is documented as a limit, and multi-hop coverage is measured on test.

**Decision:** on by default in every entry point (`try_ask.py`, `scripts/run_eval.py`, the live smoke test).
- **Benefit:** a wrongly refused answerable question is now answered.
- **Cost:** ~2 extra writer requests for questions below the gate.
- **Risk:** carried by the writer's refusal and citation checks, which held here.

The held-out test set confirms or refutes this.

## Checking the grader on real answers, and the rubric rule it led to (2026-09-28)

The grader passed a synthetic check (69/69 answers of known quality), but real answers are subtler. The first 11 real dev answers were reviewed by hand (`scripts/review_grader.py`):

| Outcome | Questions |
|---|---|
| agree | G02, G09, G11, G14, G17, G20; also G07 and G23, but on those two the grader marked a fact "present" that the answer never states ("CSF 2.0", "authentication_timeout"): too lenient on a detail |
| **disagree: grader too strict** | **G01, G12, G15**: correct answers marked incomplete |

**Agreement on the overall verdict: 8/11 (73%).**

**All 3 disagreements were faults in my answer key, not the grader:**
- G12 ("Will restarting the kubelet kill the pods?"): a key fact required "use kubectl drain", which the question never asked.
- G15: a key fact required "leave an admin note", which the question never asked.
- G01: two key facts restated each other.

**Rule, applied to every item (dev and test):** `key_facts` are only what the question asks, and all of them are required for "correct". Supporting details move to `extra_facts` (kept as context, not required). 24 of 47 items changed; multi-part questions keep every part they ask.

**The test split was relocked with the reason recorded** (`lock_history` in the golden set). This was legitimate because no test result existed yet. `scripts/check_golden.py --relock "reason"` now **refuses** to relock once any test result exists (verified with a dummy test-results file). So "don't tune on test" is enforced by the tooling, not left to discipline.

The 11 answers graded under the old rubric are discarded. The dev run restarts under the new rubric, and a fresh human review of its verdicts gives the reported grader agreement.

## Evaluation throughput on the free plan (2026-09-28)

Both Groq models have a **daily token limit**, and a day of evaluation work exhausts both:
- **gpt-oss-20b** (judge and grader): 200,000 tokens/day, measured from its 429 message. Each refill so far allowed about 11 evaluated questions.
- **gpt-oss-120b** (writer): also exhausted today, during a grader check.

To spread the load, the grader model is now configurable (`scripts/run_eval.py --grader-model`), so grading can run on a different daily budget from the pipeline's judge. gpt-oss-120b as grader agreed with the known grade in **31/31** of the synthetic cases it completed before its own quota ran out. The other 31 cases failed on quota, not on grading. **The check must be completed before switching**, and because 120b also writes the answers, grader agreement on real answers must be re-measured for it (a model grading its own family's output).
