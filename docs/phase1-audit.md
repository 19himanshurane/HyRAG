# HyRAG Phase 1 audit

**Scope:** ingestion (`hyrag/loader.py`), chunking (`hyrag/chunking.py`), embeddings (`hyrag/embeddings.py`), `scripts/`, `try_*.py`.
**Date:** 2026-09-22. **Corpus:** 17 files (12 real public docs, 4 synthetic samples, 1 attribution README).
**Method:** every finding below comes from a command you can re-run from the repo root. Numbers marked *[projection]* are extrapolated from a measurement, not measured directly.

Reproduce everything:

```bash
python scripts/audit/profile_ingest.py                          # timing + chunk stats
python scripts/audit/profile_ingest.py --profile nist-sp-800-63b # cProfile hotspots
python scripts/audit/profile_ingest.py --memory                  # peak memory per file
python scripts/audit/probe_robustness.py                         # correctness probes (offline)
python scripts/audit/embed_throughput.py                         # ONE real Mistral call
```

(Set `PYTHONPATH=.` if running outside an IDE.)

## Results after fixes (Phase B)

All 19 fixable findings are fixed. F20 is an open question that needs Step 3 retrieval to answer, so it stays open. Every fix was followed by a diff against a golden snapshot of all corpus output (parser + chunker) taken before any change. Apart from the intended behaviour changes below, output is byte-identical.

| ID | Before | After | How it's verified |
|---|---|---|---|
| F1 | 562.5 MB peak (800-63B), 230.1 MB (800-61r3) | **13.4 MB / 15.5 MB** | `profile_ingest.py --memory`; test fails on old code (4.0× growth for 8× pages) |
| F2 | CRLF title `k8s-secrets`, YAML in text | CRLF = CR = LF, byte-identical | `test_windows_line_endings_parse_like_unix`, probe 1 |
| F3 | batch aborted on 1 corrupt PDF | skipped, recorded in `store.failures`, last good version kept | 2 store tests, probe 4 |
| F4 | 3 raw vs 5 processed after rename/delete | 3 vs 3; new `DocumentStore.delete()` | test + probe 5 |
| F5 | `hr/policy.md` and `it/policy.md` shared an id | `root` is required (**signature change**, approved); distinct ids | 2 tests, probe 6 |
| F6 | 52/55 ids pointed to different text after an edit | **0/55**; ids = hash(heading + text) (**meaning change**, approved); 0 duplicates in 2,166 ids | 3 tests, probe 8 |
| F7 | no tests | **75 tests**, offline, ~23 s; also pass from `requirements-lock.txt` in a fresh venv | `python -m pytest -q` |
| F8 | identical re-ingest 5.7 s | **0.0 s** (sha256 + `PARSER_VERSION` index in `data/index.json`) | 2 tests, probe 7 |
| F9 | full corpus 19.6-20.4 s sequential; parallel across files only 1.4× | **4.7-4.9 s (3.9-4.2×)**: PDFs split into 8-page tasks across cores; nothing-changed re-run 0.7 s | `bench_ingest.py`, test compares with one-by-one output |
| F10 | `RecursionError` at 1,200 nesting | iterative walk; tested at 5,000 | test + probe 2 |
| F11 | `~~~` code line became a heading | both fence styles; closes only on the same character | 2 tests, probe 3 |
| F12 | `corpus/README.md` ingested | `corpus/manifest.json` decides; README excluded; order is platform-independent | 2 tests, probe 11 |
| F13 | 6/6 deps `>=` | `requirements-lock.txt`: 28 exact pins from a clean venv | fresh venv from the lock: 75 passed |
| F14 | 0 log lines | INFO per file (parsed/reused, seconds), WARNING per skip, batch summary, retry warnings | `test_ingestion_is_logged` |
| F15 | every run re-embeds everything | SQLite cache `data/embeddings.sqlite` keyed by (model, text); repeats in one call sent once | 2 tests (mocked; no extra real API calls) |
| F16 | no limits; **`delete("../../x")` removed a file outside `data/`** (a hole the F4 fix itself introduced, caught here) | 50 MB / 2,000-page limits; every raw path must resolve inside `raw_dir` | 2 tests + reproduction of the old behaviour |
| F17 | `embed([])` shape `(0, 0)` | `(0, 1024)` | test |
| F18 | 10 thresholds inline | `PdfTuning` block, documented; output identical | snapshot 0/16 changed |
| F19 | `.env` read on import; client never closed; `Retry-After` ignored | read in `__init__`; `close()` / `with`; `Retry-After` honoured, capped at 60 s | 3 tests |
| F20 | open question | still open, measure in Step 3 | — |

**Found while fixing:**
- **Path sorting differs by OS.** Sorting `Path` objects is case-insensitive on Windows and case-sensitive on Linux, so the same corpus came out in a different order on each. `list_corpus_files` now sorts by the POSIX string.
- **The F4 fix opened a path-traversal hole.** `delete()` accepted `../` paths. F16 closed it before it was ever committed.

**Out of the brief's file scope (flagged, easy to revert):**
- `try_loader.py`: had to pass `root=` after F5, and gained 2 lines of logging setup.
- `requirements-lock.txt`: a new file, because F13 needs one.

**Unchanged known limitations:** two-column PDFs, scanned PDFs. NIST 800-61r3's empty cover-title line ("NIST Special Publication 800", 2 of 1,672 lines) produces no section, as before.

---

Severity: **P0** blocks production · **P1** must fix · **P2** should fix · **P3** nice to have.
Effort: **S** under 30 min · **M** 1-2 h · **L** half a day or more.

## Ranked findings

| ID | Sev | Area | Finding | Evidence (measured) | Effort |
|---|---|---|---|---|---|
| F1 | P0 | Memory | PDF parsing keeps every page in memory until the file closes | 1 MB / 129-page PDF peaks at **562 MB** (4.4 MB/page); 1,000 pages ≈ 4.4 GB *[projection]* | S |
| F2 | P1 | Correctness | Windows line endings (CRLF) break markdown front matter | Kubernetes doc with CRLF: title lost (`k8s-secrets` instead of `Secrets`), YAML leaks into text. `core.autocrlf=true` here, so a fresh Windows clone hits this | S |
| F3 | P1 | Robustness | One bad file aborts a whole re-ingest | `reprocess_all()` raised `PdfminerException` on one corrupt PDF; no other file was processed after it | S |
| F4 | P1 | Data integrity | Deleted/renamed documents stay in `processed/` | After 1 rename + 1 delete: 3 raw files, **5** processed files (2 orphans still searchable later) | S |
| F5 | P1 | Data integrity | Same file name in two folders silently overwrites | `hr/policy.md` and `it/policy.md` got the same `doc_id`; raw copy now holds only the IT policy | S |
| F6 | P1 | Data integrity | Chunk ids are positions, not content | One added paragraph: **52 of 55** chunk ids now point to different text while **54 of 55** texts are unchanged | M |
| F7 | P1 | Quality | No automated test suite | `tests/` absent; ~25 edge cases exist only as scratch probes; two past fixes broke earlier behaviour | M |
| F8 | P2 | Performance | Unchanged files are fully re-parsed on every ingest | Identical re-ingest of NIST 800-61r3: 6.0 s first, **5.7 s** again (nothing skipped) | M |
| F9 | P2 | Performance | PDF parsing is 82% of ingest time; our code is ~1% of it | 18.0 s of 21.9 s total; cProfile: **96.6%** inside pdfplumber `extract_text_lines` (283k char objects) | M |
| F10 | P2 | Robustness | Deeply nested HTML crashes the parser | 500 nested `<div>`: OK. 1,200: **RecursionError** | S |
| F11 | P2 | Correctness | `~~~` code fences aren't recognised | `# install the agent` inside a `~~~` block became a heading | S |
| F12 | P2 | Correctness | Corpus metadata file ingested as a company document | `corpus/README.md` (license notes) parsed into 3 chunks | S |
| F13 | P2 | Reproducibility | Dependencies unpinned | 6 of 6 requirements use `>=`; a pdfplumber upgrade can silently change parse output and eval numbers | S |
| F14 | P2 | Observability | No logging anywhere in the package | 0 `import logging` in `hyrag/`; no per-file timing, counts or skip reasons | S |
| F15 | P2 | Cost | Embeddings are never cached | Every run re-embeds everything (`try_search.py` embeds all chunks twice); full corpus ≈ 26 s + API cost per run *[projection from 1 call: 50 chunks in 1.00 s]* | M |
| F16 | P2 | Security (latent) | No size limits; raw path built from a filename | No max file size/pages; 1,000-page PDF ≈ 90 s and 4.4 GB *[projection]*. `raw_dir / source` would accept `../` once Phase 5 takes uploaded filenames | S |
| F17 | P3 | API edge case | `embed([])` returns shape `(0, 0)` | `np.vstack` with a real `(n, 1024)` batch raises `ValueError` | S |
| F18 | P3 | Maintainability | 9 PDF thresholds hard-coded, tuned on 2 documents | `loader.py` lines 280-399: 0.1/0.9 margins, 3×size, 1.15, 8/20 words, 0.9 right edge… | S |
| F19 | P3 | Hygiene | Embedder side effects and resource leak | `load_dotenv()` runs on import; `httpx.Client` never closed; retries ignore `Retry-After` *[unverified: no real 429 observed]* | S |
| F20 | P3 | Open question | 8% of chunks are tiny | 105 of 1,278 structure chunks < 100 chars (acronym definitions, one-line intros). Retrieval impact unknown; A/B in Step 3 with the heading-length question | — |

### Checked and fine (no finding)

- **Determinism:** the same PDF parsed twice gives identical output.
- **Chunker complexity:** I suspected the repeated `"".join()` in `_merge` was quadratic. It isn't: the buffer is capped at the chunk size. 2.2 MB of text chunks in 0.68 s (structure) and 0.19 s (fixed).
- **HTML "slowness":** `postgres-config-connection.html` took 1.10 s only because it was the first HTML file (one-time imports plus compiling the CSS selector). Warm, it takes 0.06 s.
- **Secrets:** `.env` has never been committed (`git log --all -- .env` is empty) and is git-ignored.
- **Embeddings:** 50 chunks (33k chars) in one request worked, 1.00 s.

### Not built yet (gaps, not bugs)

Semantic chunking, ChromaDB storage, BM25 index, near-duplicate detection, evaluation set. Phase 1 steps 2-4 cover these.

---

## Details

### F1 (P0): PDF memory grows with page count
**In plain words:** pdfplumber builds a Python object for every character on a page and keeps it cached until the whole PDF is closed. Our 1 MB NIST file needed 562 MB of RAM, and a long manual could run a small server out of memory. The document is just text; the waste is from keeping finished pages around.
**Evidence:** `profile_ingest.py --memory` → 562.5 MB peak (800-63B, 129 pages), 230.1 MB (800-61r3, 48 pages).
**Fix:** call `page.close()` (exists in pdfplumber 0.11.10; it runs `flush_cache()`) right after each page's lines are read in `_read_pdf_lines`. Re-measure peak memory and confirm output is byte-identical.

### F2 (P1): CRLF markdown loses front matter
**In plain words:** Windows text files end lines with two characters (`\r\n`) instead of one. The front-matter regex looks for `---\n`, so on Windows-style files it finds nothing: the Kubernetes title disappears and YAML (`content_type: concept`) ends up inside the searchable text. Git on Windows converts to CRLF on checkout by default, so anyone cloning this repo on Windows gets worse results than we measured.
**Evidence:** `probe_robustness.py` probe 1.
**Fix:** normalise `\r\n` / `\r` to `\n` once in `decode_text()`, so every text format benefits.

### F3 (P1): one broken file stops a batch
**In plain words:** `reprocess_all()` loops over every document and stops at the first exception. In production, one corrupt upload would block re-indexing of the entire library.
**Evidence:** probe 4. A fake `broken.pdf` raised `PdfminerException: No /Root object!` and the batch ended.
**Fix:** catch errors per file, log them, skip the file, and return a list of failures next to the successful documents.

### F4 (P1): orphaned documents after rename/delete
**In plain words:** when a raw file is renamed or deleted, its processed JSON stays behind. Later steps would still search it, so users could get answers cited from a document that no longer exists.
**Evidence:** probe 5. 3 raw files, 5 processed files; `deploy-guide.html` (deleted) and `leave-policy.txt` (renamed) are still present.
**Fix:** `reprocess_all()` removes processed files whose `source` no longer exists in `raw/`; add `DocumentStore.delete(source)`.

### F5 (P1): same-name files overwrite each other
**In plain words:** `ingest_file(path)` without `root` uses only the file name as identity, so `hr/policy.md` and `it/policy.md` collide and the second silently replaces the first. `try_loader.py` calls it this way.
**Evidence:** probe 6. Both got `doc_id 71bc8aae6d3b`, and the raw copy holds the IT text.
**Fix:** refuse to overwrite an existing raw copy whose bytes differ and came from a different origin path, or make `root` required. **Needs your OK:** making `root` required changes a public signature.

### F6 (P1): positional chunk ids
**In plain words:** a chunk id like `doc:structure:17` means "the 17th piece", so adding one paragraph near the top renumbers everything after it. When Step 3 stores vectors under these ids, the store would either keep vectors attached to the wrong text or have to re-embed everything after every edit.
**Evidence:** `probe_robustness.py` probe 8. 52/55 ids changed meaning; 54/55 texts byte-identical.
**Fix:** derive `chunk_id` from content: `sha1(doc_id + strategy + text)`, plus an occurrence counter for repeated text within a document. Keep `chunk_index` as the position. The field name and type stay the same; its meaning changes, so **this needs your OK**.

### F7 (P1): no test suite
**In plain words:** all the edge cases we found (nested HTML, BOM, running headers, glossary nesting…) live in throwaway scripts. Twice, a fix broke an earlier fix and was only caught by chance.
**Fix:** a pytest suite in `tests/` covering every earlier probe, running offline with the Mistral HTTP call mocked.

### F8 (P2): no change detection
**In plain words:** the loader doesn't remember what it already parsed, so re-ingesting an unchanged 48-page PDF costs the full 6 seconds again. With thousands of pages, every re-index repeats hours of identical work.
**Fix:** store `sha256(file bytes)` and a parser version in the processed JSON, and skip parsing when both match.

### F9 (P2): PDF parsing speed
**In plain words:** 82% of ingest time is PDF parsing, and almost all of it is pdfplumber building character objects. Tuning our own code can't help. The practical wins are "don't parse twice" (F8) and parsing several files at once on separate CPU cores.
**Evidence:** cProfile. `extract_text_lines` is 96.6% cumulative; `process_object` alone is 51%.
**Fix options:** (a) F8 caching; (b) a process pool across files, which brings corpus wall time toward the slowest single file (11.6 s) *[projection]*; (c) read characters through pdfminer directly, skipping pdfplumber's per-character dicts. (c) is a larger rewrite; recommend (a) + (b) first.

### F10 (P2): recursion limit in HTML walk
**In plain words:** the HTML walker calls itself once per nesting level, and Python stops at about 1,000 levels. Generated or hostile pages can nest deeper and crash ingestion, which F3 then turns into a failed batch.
**Fix:** replace the recursion with an explicit stack (same output). Verify with the 1,200-deep probe and the existing HTML checks.

### F11 (P2): `~~~` fences
**In plain words:** markdown allows code blocks fenced with `~~~` as well as backticks. We only know backticks, so a shell comment like `# install the agent` inside a `~~~` block becomes a fake heading.
**Fix:** recognise both fence styles, and close a block only with the same fence character that opened it.

### F12 (P2): metadata file ingested
**In plain words:** `corpus/README.md` holds license notes, not company knowledge, but anything that walks `corpus/` treats it as a document. Its text would show up in answers.
**Fix:** use `corpus/manifest.json` as the list of documents to ingest; everything else in the folder is ignored.

### F13 (P2): unpinned dependencies
**In plain words:** `pdfplumber>=0.11` means "any newer version too". A future release could parse PDFs slightly differently and quietly change every evaluation number.
**Fix:** add `requirements-lock.txt` from `pip freeze` (no new dependency) and document installing from it.

### F14 (P2): no logging
**In plain words:** when ingestion goes wrong there is no record of which file, how long it took, or what was skipped.
**Fix:** standard-library `logging`: one INFO line per file (source, sections, seconds) and a WARNING for each skipped or failed file.

### F15 (P2): no embedding cache
**In plain words:** we pay Mistral again for text we've already embedded. It matters most in Step 3 and the Phase 4 evaluation runs, which re-run the pipeline many times.
**Fix:** an on-disk cache keyed by `(model, sha256(text))`. Best built together with Step 3; recommend deferring it until then.

### F16 (P2, latent): input limits and path safety
**In plain words:** nothing stops a 5,000-page PDF or a 500 MB HTML file, and once Phase 5 accepts uploaded filenames, a name like `../../x` could write outside `data/`. It isn't reachable today (`relative_to` blocks it), but it's cheap to guard now.
**Fix:** a maximum byte size and page count, and a check that the resolved raw path stays inside `raw_dir`.

### F17-F19 (P3)
- **F17:** return `np.empty((0, dim))` for empty input.
- **F18:** gather the PDF thresholds into one documented constants block.
- **F19:** load `.env` in scripts rather than on import, add `close()` / context-manager support to the embedder, and honour the `Retry-After` header.

### F20 (open question)
Tiny chunks (8%) and long heading prefixes (median 137-206 chars) may help or hurt retrieval. Measure both in Step 3 with a small labelled query set before changing anything.
