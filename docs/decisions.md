# Design decisions

These are the choices that shaped HyRAG, each with the measurement that settled it. Several are reversals: the first version did something reasonable, a measurement showed it was wrong, and it changed.

## Exact vector search, not Chroma's approximate search

Chroma's default search (HNSW) is approximate. On this corpus it missed one or two of the true top ten results for three of six test questions, and which ones it missed changed from run to run. They were real misses: one question lost a passage with similarity 0.750 while a passage at 0.739 was returned.

Two things caused it. The embedding model gives low-contrast scores, with almost everything between 0.65 and 0.85, so the default search width is too narrow. And the search graph degrades as chunks are deleted and re-added.

HyRAG now compares the question with every stored vector in memory. That is exact, gives the same result every time, and takes under a millisecond at this size. It costs 4 KB of memory per chunk. Past roughly 100,000 chunks, about 400 MB, an approximate index becomes the right choice again, with its recall measured first.

## Both searches, merged by rank

Meaning search alone ranked the passage for ERR_TUNNEL_4013 above the one for ERR_TUNNEL_4012 when asked about 4012. To an embedding, the two codes are nearly the same text. Keyword search gets that right and fails on paraphrases.

The two score on scales that can't be compared, cosine similarity against BM25 points, so they are merged by rank. That throws away how large a lead is. For the 4012 question, keyword search preferred the right passage by 37.3 points to 17.1 while meaning search preferred the wrong one by a hair, and rank fusion sees only one first place each. The cross-encoder reranker exists to settle that.

## A local reranker that never touches the network at runtime

The reranker is a small cross-encoder that runs on CPU: about half a second for twenty passages, and 590 MB of RAM.

By default the library checked the Hugging Face Hub every time the model loaded, even with the model already cached. With the Hub unreachable, loading took 250 seconds. The model is now downloaded once, explicitly, and loaded offline after that. If it fails anyway, search returns the merged results without reranking.

## Three layers decide "not found", not one threshold

The brief asked for a retrieval threshold. One threshold isn't enough. "How do I fix ERR_TUNNEL_4099?" asks about a code no document mentions, and scores 0.90 on retrieval because the pages for 4012 and 4013 look relevant.

So there are three layers. The gate stops questions where nothing relevant was found, without calling a model. The writer is told to reply with one fixed sentence when the passages don't contain the answer. And the citation check catches a writer that improvises anyway.

## The checker must quote, and the quote is verified

A model asked "does this passage support this claim?" can be wrong in either direction. So it has to copy the words it relied on, and code looks them up in the passage. A "supported" verdict with a quote that isn't there doesn't count.

That check was too strict twice, and both times it withheld a correct answer. Checkers drop list bullets when quoting two bullet points, so quotes are now compared as word sequences. They also join two sentences that aren't adjacent, so each sentence of a quote is checked on its own. A minimum quote length was tried and removed, because it rejected short and correct table quotes like "23505 | unique_violation".

## The checker sees what the writer saw

A correct answer about "NIST SP 800-61r3" was withheld because the passage only says "this publication". The writer knew which document it was from the passage's source file name. The checker was never shown the source. It gets it now, with a prompt line saying the source identifies the document and is evidence for nothing else. Two trap cases confirm that: the same passage with a different document named, and the right document with a false fact. Both are still rejected.

## Confidence has hard caps

The score is a weighted average of retrieval, verified citations and completeness. An average can hide a red flag. A deliberately wrong answer, with every claim unsupported, first scored 0.50, "medium", because retrieval and completeness were perfect. Caps now sit on top: one unsupported claim limits the score to 0.4, and a partly supported claim rules out "high".

## The model that checks citations

The first choice was a model from a different family than the writer, so the checker wouldn't be grading its own family's output. On this Groq plan that model is limited to 1,000 output tokens a minute, about one check a minute, and real answers kept coming back unchecked. I had picked it for its "separate rate-limit budget" without measuring that budget.

The checker is now gpt-oss-20b. It has produced no false approvals on the labelled set. The cost is that it shares a family with the writer.

## One time budget per question

A single Groq call could take 10.2 minutes in the worst case: five attempts of 120 seconds plus backoff. A question makes several calls. Every question now has one 60-second budget shared by all its network calls. Each attempt's timeout shrinks to what is left, and no retry waits past the end.

Daily quota errors are treated differently. Retrying those is pointless, so they fail at once with their own error code.

## Heading-aware chunks

Compared on retrieval, heading-aware chunks beat fixed-size ones: reciprocal rank 0.91 against 0.67. Fixed-size chunks also made the "not found" gate unworkable. Semantic chunking matched heading-aware chunking and cost more to build. The numbers are in [evaluation.md](evaluation.md).

## Duplicates are found by text, not by embedding

Embedding similarity was the wrong signal in both directions. 633 pairs of chunks scored above 0.95 while sharing a median of 25% of their text, because they start with the same long heading path. And boilerplate copied word for word between two NIST PDFs scored only 0.87 to 0.92.

Two chunks now count as duplicates when they share 90% of their three-word sequences and contain exactly the same numbers and identifiers. The second rule keeps apart two PostgreSQL settings whose descriptions are 91% identical.

## Collections

With one index holding several companies' documents, "How much paid time off do I get per year?" returned three GitLab passages, one from the fictional company and one from PostHog. A question is now answered from one collection. Keyword statistics are kept per collection too, so adding one company's documents changes nothing in another's ranking.

## Search vectors come from the local cache

Meaning search keeps every vector in memory. The API used to fetch them from the Chroma container on the first question after starting. With 7,628 chunks that took 96 seconds in pages and 692 seconds in a single request, on a machine short of memory, so the first question always timed out.

The vectors are also in the embedding cache, a SQLite file in the same data volume. They are read from there now, at startup, and Chroma is asked only for vectors the cache lacks.

## Mistakes worth recording

A fix for thread safety put one lock around every index method, including the one that calls the embedding API. Eight concurrent searches then took 2.43 seconds against 0.30 for one, and a keyword search waited on someone else's network call. The rule since: no lock is held across network I/O, and a concurrency fix is measured with a slow dependency, not only checked for correctness.

A stalled evaluation was diagnosed wrongly twice. First I blamed a concurrent index rebuild, then the per-minute rate limit, and I built a pacing layer for the second guess. The provider's error message had named the real cause all along: the daily token limit. The pacing layer was removed.

A question about "ImagePullBackOff" was labelled unanswerable because that word isn't in the documents. The concept is, under "My pod stays waiting". Topics are now checked by concept before being labelled missing.

The MDX cleaner deleted bare placeholders such as `<source>` from prose, and left some component tags in. A code review caught both.
