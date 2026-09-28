# Corpus

Unmodified snapshots of public documents, used to test HyRAG on realistic internal-style docs.
They are kept in the repo (rather than downloaded on demand) so evaluation questions written
against them don't go stale when the live pages change. `scripts/fetch_corpus.py` re-downloads
them; `manifest.json` records each file's source URL, license and retrieval date (2026-09-22).

These files are **not** covered by the repository's MIT license. Each keeps its own license:

| Files | Source | License |
|---|---|---|
| `handbook/gitlab-*.html` | [The GitLab Handbook](https://handbook.gitlab.com/), © GitLab Inc. | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `engineering/k8s-*.md` | [Kubernetes Documentation](https://github.com/kubernetes/website), © The Kubernetes Authors | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `engineering/postgres-*.html` | [PostgreSQL Documentation](https://www.postgresql.org/docs/current/), © The PostgreSQL Global Development Group | [PostgreSQL License](https://www.postgresql.org/about/licence/) |
| `security/nist-*.pdf` | NIST Special Publications 800-63B-4 and 800-61r3 | Public domain in the US (17 U.S.C. §105). Reprinted courtesy of the National Institute of Standards and Technology, U.S. Department of Commerce. Not copyrightable in the United States. |
| `posthog/handbook/**` | [PostHog's company handbook](https://posthog.com/handbook), © PostHog Inc., all 383 pages at commit `3b7d39fa` (retrieved 2026-09-27; `fetch_corpus.py --posthog`) | MIT, for the `/contents/` folder of [PostHog/posthog.com](https://github.com/PostHog/posthog.com) only: notice in [`posthog/LICENSE`](posthog/LICENSE). HyRAG is not affiliated with or endorsed by PostHog. |

`posthog/` is its own **collection**: HyRAG answers questions about it from its documents only, never mixed with the others (the demo set).

The GitLab HTML files are large (~2.3 MB each) because every page embeds the handbook's full
navigation menu; the loader strips it.
