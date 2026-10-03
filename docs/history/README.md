# Build notes

These six files are the notes written while HyRAG was being built, kept as they were. They are organised by build phase, record every audit finding with its before-and-after measurement, and use short finding ids (H1, S6 and so on) that only make sense inside each file.

For a description of the system as it is now, read the four documents one level up. These notes are the place to look when you want the full evidence behind a number, or the story of how a bug was found.

Some details in them are out of date. The `try_*.py` scripts they mention became `examples/`, and the documents now include PostHog's handbook.

| File | Covers |
|---|---|
| [phase1-audit.md](phase1-audit.md) | loading, chunking, the index, near-duplicates: three audit rounds |
| [phase2-audit.md](phase2-audit.md) | hybrid retrieval and the reranker |
| [phase3.md](phase3.md) | generation, citation checks, confidence, "not found" |
| [phase3-audit.md](phase3-audit.md) | outages, time budgets, rate limits and cost |
| [phase4.md](phase4.md) | the evaluation set and what running it revealed |
| [phase5.md](phase5.md) | the API, the dashboard and Docker |
