"""Pick the sections the golden set's lookup questions are written about, AT RANDOM (fixed seed), so the
questions cover whatever the corpus holds rather than cases the author already knows work.

Usage: python scripts/sample_golden_sections.py   -> eval/golden_sample.json (the draw, kept as a record)
"""
import json
import random
from pathlib import Path

SEED = 20260928
MIN_WORDS = 40  # shorter sections are tables of contents, bare headings or reference lists
ALLOCATION = {  # questions per document: roughly by size, at least one each
    "security/nist-sp-800-63b-4-authentication.pdf": 6, "security/nist-sp-800-61r3-incident-response.pdf": 3,
    "engineering/k8s-pod-lifecycle.md": 3, "handbook/gitlab-incident-management.html": 3,
    "handbook/gitlab-time-off-types.html": 3, "engineering/postgres-config-resource.html": 2,
    "engineering/k8s-secrets.md": 2, "engineering/postgres-config-connection.html": 2,
    "engineering/k8s-debug-pods.md": 2, "vpn-setup.md": 2, "engineering/postgres-error-codes.html": 1,
    "handbook/gitlab-security.html": 1, "handbook/gitlab-time-off-and-absence.html": 1, "deploy-guide.html": 1,
    "leave-policy.txt": 1, "security-handbook.pdf": 1,
}


def main() -> None:
    rng = random.Random(SEED)
    docs = {d["source"]: d for d in (json.load(open(p, encoding="utf-8")) for p in Path("data/processed").glob("*.json"))}
    draw = []
    for source, n in ALLOCATION.items():
        sections = docs[source]["sections"]
        eligible = [i for i, s in enumerate(sections) if len((s.get("text") or "").split()) >= MIN_WORDS]
        if len(eligible) < n:  # the tiny sample documents: take what there is
            eligible = [i for i, s in enumerate(sections) if (s.get("text") or "").strip()]
        for i in rng.sample(eligible, min(n, len(eligible))):
            s = sections[i]
            draw.append({"source": source, "section_index": i, "heading": " > ".join(s.get("heading_path") or [s.get("heading") or ""]),
                         "page": s.get("page"), "text": s["text"]})
    Path("eval/golden_sample.json").write_text(json.dumps({"seed": SEED, "min_words": MIN_WORDS, "sections": draw},
                                                          indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(draw)} sections drawn")


if __name__ == "__main__":
    main()
