# Evaluation

This page covers how HyRAG is tested, what has been measured so far, and what hasn't.

The short version: the parts have been measured one by one, and the numbers are below. The full end-to-end score on the held-out questions has not been produced yet, because the free model tier allows about 115 checked questions a day and the runs so far went into finding problems.

## The question set

`eval/golden_set.json` holds 82 hand-written questions. Each has a reference answer, the key facts a correct answer must contain, and the evidence: a document, a section, and an exact quote.

| Type | Count | What it tests |
|---|---|---|
| lookup | 48 | one passage holds the answer |
| multi-hop | 13 | the answer needs two passages, often from two documents |
| no answer | 13 | the documents don't contain it; the right response is "not found" |
| ambiguous | 8 | the documents give different answers depending on what is meant |

60 questions are about the demo documents and 22 about PostHog's handbook. Each question is searched only in its own collection.

Three things were done to keep the set fair.

The lookup questions were written about sections drawn at random with a fixed seed, including dull ones such as a patent notice. Choosing sections by hand would have favoured cases already known to work. The PostHog draw leans towards sales topics because the handbook does.

Every evidence quote is checked word for word against the parsed documents by `scripts/check_golden.py`. So is every "no answer" topic, which must be absent from its collection. Checking before labelling mattered. Pets, health insurance and "badge" were all planned as missing topics and turned out to be in the documents.

The set is split into 33 development questions and 49 test questions. Tuning and debugging use the development questions only. The test questions are fingerprinted, the checker fails if one changes, and it refuses to reset that lock once any test result exists.

Evidence is recorded as a quote and not as a chunk id, because chunk ids differ between chunking strategies and the same answer key has to work for all three.

## What is scored

`hyrag/evaluation.py` scores each question on six things.

| Measure | The question it answers |
|---|---|
| Retrieval | Is the evidence quote among the five passages the writer sees, and among the twenty candidates before reranking? A text comparison; no model involved |
| Correctness | Does the answer contain each key fact? A grader model marks every fact present, missing or contradicted |
| Faithfulness | Is every claim supported by some retrieved passage? |
| Citation accuracy | What share of cited claims passed the citation check? |
| Behaviour | Were the "no answer" questions refused, and the ambiguous ones clarified or answered for each reading? |
| Withheld answers | When an answerable question got no answer, was the withheld draft actually correct? This is what the safety checks cost |

Results are reported as a range over repeated runs. The same question can score differently twice, even with temperature 0 and a fixed seed.

The run is crash-safe: each question's result is written to disk as soon as it is scored, and a stopped run can be resumed. An earlier version saved only at the end and lost an hour of work to a hang.

## Results so far

### Chunking strategies, retrieval only

Measured on the 18 answerable development questions about the demo documents (23 evidence quotes), with no language model involved.

| Strategy | Chunks | Evidence in top 5 | All evidence in top 5 | Evidence in top 20 | Mean reciprocal rank |
|---|---|---|---|---|---|
| fixed size | 884 | 0.78 | 0.78 | 0.91 | 0.67 |
| structure | 1,266 | 0.83 | 0.83 | 0.96 | 0.91 |
| semantic | 1,307 | 0.83 | 0.83 | 0.91 | 0.86 |

Structure-aware chunks rank the evidence as high or higher than fixed-size chunks in all six questions where the two differ. Fixed-size chunks cut passages in the middle, so the evidence gets split or diluted. Semantic chunking performs about the same as structure and costs more to build. With 23 quotes, one quote is worth 0.04, so small gaps here mean little.

One disclosure: an alternative evidence passage for one question was added to the answer key after reading the structure strategy's results. It is a real answer passage, but finding it that way favoured structure. Without that question, structure still leads on rank in the other five.

Fixed-size chunks also break the "not found" gate. With structure chunks, the lowest-scoring answerable calibration question sits at +0.08 and the highest unanswerable one at −5.83, a wide gap to put a threshold in. With fixed-size chunks an answerable question scores −8.73, below unanswerable ones, and no threshold works.

### The citation checker

`eval/judge_pairs.json` has 35 hand-labelled pairs of a claim and a passage. The traps include two error codes swapped, "Monday to Thursday" changed to "Friday", a negation flipped, a true claim cited to the wrong passage, and a passage that instructs the checker to answer "supported".

The number that matters is false support: approving a claim the passage doesn't back. The checker has produced none in any run. The latest run got 32 of the 35 pairs exactly right, and all three misses were on the strict side.

### The gate and the confidence score

On 20 calibration questions, half answerable and half not, the reranker's top score separated the two groups completely. Across two full runs, the answerable ones scored "high" 10 of 10 and 9 of 10 times, and all ten unanswerable ones came back "not found" both times. A deliberately wrong answer scored 0.40, "low".

These 20 questions were also used to set the thresholds, so this shows the design is coherent. It is not evidence of how well it generalises. The held-out test set exists to answer that.

### Rewriting poorly matched questions

One development question was being refused although the documents answer it, because its conversational wording matched badly. With query rewriting on, it is answered correctly. The risk is that rewriting also lets through questions that should be refused. Three unanswerable calibration questions did get past the gate that way, and the writer model declined all three in six of six runs.

### The grader

The grader agreed with the known grade on 69 of 69 synthetic answers. On 11 real answers reviewed by hand it agreed 8 times. All three disagreements turned out to be faults in the answer key: it demanded details the question had never asked for. The rule since then is that key facts are only what the question asks. A fresh human review is due on the next development run.

### PostHog questions, retrieval only

For 7 of the 8 answerable PostHog development questions, all the evidence is in the top five. The miss is a two-part question about a late laptop. Search finds the onboarding page and never looks for the page naming who handles laptops, because query rewriting only runs when the first match is weak, and here the first half matched well.

## What is still open

- The end-to-end run on the 33 development questions, twice, with a human review of the grades.
- The same comparison of chunking strategies measured on answer quality, not only retrieval.
- The final run on the 49 test questions.
- "How many days off do I have to take each year?" came back unverified, although the time-off page says at least 25 days. It needs investigating.

## Limits of this evaluation

I wrote all 82 questions. Random sampling limited which topics I picked, but not how I phrased them. Questions from someone who hasn't read the documents would be a better test.

With 49 test questions, a difference of one or two answers is noise.

The checker and the writer come from the same model family, so they may share blind spots. A checker from a different family was the first choice, but its rate limit on this plan allowed about one check a minute.

## Running it

```bash
PYTHONPATH=. python scripts/check_golden.py                      # verify the answer key
PYTHONPATH=. python scripts/run_eval.py --check-grader           # is the grader trustworthy?
PYTHONPATH=. python scripts/run_eval.py --split dev --runs 2     # score the pipeline
PYTHONPATH=. python scripts/run_eval.py --resume eval/results/<file>.jsonl
PYTHONPATH=. python scripts/compare_chunking.py                  # retrieval only, no LLM
PYTHONPATH=. python scripts/eval_judge.py openai/gpt-oss-20b low --batch 8
```
