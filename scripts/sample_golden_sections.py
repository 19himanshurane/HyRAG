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


POSTHOG_SEED = 20260929  # fixed before any PostHog section was read
POSTHOG_LOOKUPS = 14


def main_posthog() -> None:
    """The PostHog collection (383 pages): too many documents to allocate by hand, so sections are drawn
    uniformly from all of its eligible sections -> eval/golden_sample_posthog.json."""
    rng = random.Random(POSTHOG_SEED)
    pool = []
    for p in sorted(Path("data/processed").glob("*.json")):
        d = json.load(open(p, encoding="utf-8"))
        if d["source"].startswith("posthog/"):
            pool += [(d["source"], i, s) for i, s in enumerate(d["sections"])
                     if len((s.get("text") or "").split()) >= MIN_WORDS]
    pool.sort(key=lambda x: (x[0], x[1]))  # same order on every machine before drawing
    draw = [{"source": src, "section_index": i, "heading": s.get("heading") or "", "text": s["text"]}
            for src, i, s in rng.sample(pool, POSTHOG_LOOKUPS)]
    Path("eval/golden_sample_posthog.json").write_text(
        json.dumps({"seed": POSTHOG_SEED, "min_words": MIN_WORDS, "eligible_sections": len(pool), "sections": draw},
                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(draw)} of {len(pool)} eligible PostHog sections drawn")


if __name__ == "__main__":
    import sys

    main_posthog() if sys.argv[1:] == ["--posthog"] else main()
