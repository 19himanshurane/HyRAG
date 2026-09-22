"""Download the real-document corpus into corpus/ and record where each file came from.

We commit the downloaded snapshot too: live pages (especially the GitLab handbook) change often,
and eval questions written against today's text would silently go stale if we re-downloaded later.
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


def main() -> None:
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
