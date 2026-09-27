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
