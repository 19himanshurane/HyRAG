"""Download the real-document corpus into corpus/ and record where each file came from.

We commit the downloaded snapshot too: live pages (especially the GitLab handbook) change often,
and eval questions written against today's text would silently go stale if we re-downloaded later.

  python scripts/fetch_corpus.py             # everything below (re-downloads the live pages!)
  python scripts/fetch_corpus.py --posthog   # only PostHog's handbook, at its pinned commit
"""
import json
from datetime import date
from pathlib import Path

import httpx

CORPUS = Path("corpus")

DOCS = [
    # --- company handbook / policy (GitLab, CC BY-SA 4.0) ---
    ("https://handbook.gitlab.com/handbook/people-group/time-off-and-absence/",
     "handbook/gitlab-time-off-and-absence.html", "GitLab Handbook", "CC BY-SA 4.0"),
    ("https://handbook.gitlab.com/handbook/people-group/time-off-and-absence/time-off-types/",
     "handbook/gitlab-time-off-types.html", "GitLab Handbook", "CC BY-SA 4.0"),
    ("https://handbook.gitlab.com/handbook/security/",
     "handbook/gitlab-security.html", "GitLab Handbook", "CC BY-SA 4.0"),
    ("https://handbook.gitlab.com/handbook/engineering/infrastructure-platforms/incident-management/",
     "handbook/gitlab-incident-management.html", "GitLab Handbook", "CC BY-SA 4.0"),
    # --- security standards (NIST, US government work: public domain in the US) ---
    ("https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63B-4.pdf",
     "security/nist-sp-800-63b-4-authentication.pdf", "NIST", "Public domain in the US (17 U.S.C. 105); credit NIST"),
    ("https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf",
     "security/nist-sp-800-61r3-incident-response.pdf", "NIST", "Public domain in the US (17 U.S.C. 105); credit NIST"),
    # --- engineering docs (PostgreSQL License; Kubernetes CC BY 4.0) ---
    ("https://www.postgresql.org/docs/current/runtime-config-resource.html",
     "engineering/postgres-config-resource.html", "PostgreSQL Documentation", "PostgreSQL License"),
    ("https://www.postgresql.org/docs/current/runtime-config-connection.html",
     "engineering/postgres-config-connection.html", "PostgreSQL Documentation", "PostgreSQL License"),
    ("https://www.postgresql.org/docs/current/errcodes-appendix.html",
     "engineering/postgres-error-codes.html", "PostgreSQL Documentation", "PostgreSQL License"),
    ("https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/pods/pod-lifecycle.md",
     "engineering/k8s-pod-lifecycle.md", "Kubernetes Documentation", "CC BY 4.0"),
    ("https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/tasks/debug/debug-application/debug-pods.md",
     "engineering/k8s-debug-pods.md", "Kubernetes Documentation", "CC BY 4.0"),
    ("https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/configuration/secret.md",
     "engineering/k8s-secrets.md", "Kubernetes Documentation", "CC BY 4.0"),
]


# --- a real company's internal handbook: PostHog (MIT for the /contents/ folder only; see corpus/posthog/LICENSE).
# Pinned to one commit: the handbook changes daily, and golden questions must keep matching the text.
POSTHOG_REPO = "PostHog/posthog.com"
POSTHOG_COMMIT = "3b7d39fa9a40fe9a099a325e6dd81a082b9e25d1"  # master on 2026-09-27
POSTHOG_PREFIX = "contents/handbook/"
POSTHOG_LICENSE = "MIT (the /contents/ folder of PostHog/posthog.com; notice in corpus/posthog/LICENSE)"


def fetch_posthog(client: httpx.Client) -> list[dict]:
    """Every .md/.mdx page of PostHog's handbook at the pinned commit -> corpus/posthog/handbook/..."""
    tree = client.get(f"https://api.github.com/repos/{POSTHOG_REPO}/git/trees/{POSTHOG_COMMIT}?recursive=1")
    tree.raise_for_status()
    if tree.json().get("truncated"):
        raise RuntimeError("GitHub truncated the file list: fetch the handbook folder's own tree instead")
    paths = sorted(e["path"] for e in tree.json()["tree"] if e["type"] == "blob"
                   and e["path"].startswith(POSTHOG_PREFIX) and e["path"].endswith((".md", ".mdx")))
    entries = []
    for path in paths:
        rel = "posthog/handbook/" + path[len(POSTHOG_PREFIX):]
        resp = client.get(f"https://raw.githubusercontent.com/{POSTHOG_REPO}/{POSTHOG_COMMIT}/{path}")
        resp.raise_for_status()
        out = CORPUS / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(resp.content)
        page = path[len("contents"):].rsplit(".", 1)[0].removesuffix("/index")
        entries.append({"path": rel, "url": f"https://posthog.com{page}", "publisher": "PostHog Handbook",
                        "license": POSTHOG_LICENSE, "retrieved": date.today().isoformat(),
                        "commit": POSTHOG_COMMIT})
    print(f"PostHog handbook: {len(entries)} pages at {POSTHOG_COMMIT[:8]}")
    return entries


def main() -> None:
    import sys

    if sys.argv[1:] == ["--posthog"]:  # refresh only the PostHog pages; the rest stay as they are
        path = CORPUS / "manifest.json"
        kept = [e for e in json.loads(path.read_text(encoding="utf-8")) if not e["path"].startswith("posthog/")]
        with httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": "HyRAG-corpus-fetch"}) as client:
            manifest = kept + fetch_posthog(client)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return
    manifest = []
    with httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": "HyRAG-corpus-fetch"}) as client:
        for url, rel_path, publisher, license_ in DOCS:
            out = CORPUS / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            resp = client.get(url)
            resp.raise_for_status()
            out.write_bytes(resp.content)
            print(f"{len(resp.content):>9,} bytes  {rel_path}")
            manifest.append({"path": rel_path, "url": url, "publisher": publisher,
                             "license": license_, "retrieved": date.today().isoformat()})
    (CORPUS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
