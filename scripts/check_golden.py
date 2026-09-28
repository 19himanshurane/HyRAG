"""Check the golden set's answer key against the real documents, then assign (once) and lock the dev/test split.

Usage (from the repo root): PYTHONPATH=. python scripts/check_golden.py
                            PYTHONPATH=. python scripts/check_golden.py --relock "reason"

Checks, all against data/processed (the parsed corpus):
- every evidence quote occurs word for word in its source document, and the named section exists there;
- every no-answer topic (`absence_check`, `absence_check_in`) really is absent;
- no question repeats or closely paraphrases a question the thresholds were tuned on
  (eval/calibration_questions.json, eval/near_miss_questions.json): that would leak tuning data into the test;
- the split: assigned once, stratified by type, with a fixed seed; after that the test questions are
  fingerprinted and this script fails if any of them change. Results are only ever tuned on "dev".
"""
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from hyrag.citations import quote_in_passage
from hyrag.index import DEFAULT_COLLECTION, collection_of

GOLDEN = Path("eval/golden_set.json")
SEED = 20260928
TEST_SHARE = 0.6  # most questions are held out: the test score is the one we report
RELOCK_REASON = sys.argv[sys.argv.index("--relock") + 1] if "--relock" in sys.argv else ""


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def main() -> int:
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    items = data["items"]
    docs = {d["source"]: d for d in (json.loads(p.read_text(encoding="utf-8")) for p in Path("data/processed").glob("*.json"))}
    full = {src: "\n".join(f"{s['heading']}\n{s['text']}" for s in d["sections"]) for src, d in docs.items()}
    problems: list[str] = []

    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")
    for it in items:
        for ev in [e for top in it.get("evidence", []) for e in [top] + top.get("alternatives", [])]:
            src = ev["source"]
            if src not in docs:
                problems.append(f"{it['id']}: unknown source {src}")
                continue
            if not any(ev["section"].lower() in s["heading"].lower() for s in docs[src]["sections"]):
                problems.append(f"{it['id']}: no section matching {ev['section']!r} in {src}")
            if not quote_in_passage(ev["quote"], full[src]):
                problems.append(f"{it['id']}: quote not found in {src}: {ev['quote'][:60]!r}")
        # Absent from the documents the question is searched in: its own collection (no field = demo set, as in
        # hyrag.evaluation). Another company's handbook covering the topic doesn't make it answerable here.
        coll = it.get("collection", DEFAULT_COLLECTION)
        for ev in it.get("evidence", []):
            if collection_of(ev["source"]) != coll:
                problems.append(f"{it['id']}: evidence {ev['source']} is outside its collection {coll!r}")
        for term in it.get("absence_check", []):
            hits = [s for s, t in full.items() if collection_of(s) == coll and term.lower() in t.lower()]
            if hits:
                problems.append(f"{it['id']}: {term!r} is NOT absent: found in {hits}")
        for src, terms in it.get("absence_check_in", {}).items():
            for term in terms:
                if term.lower() in full[src].lower():
                    problems.append(f"{it['id']}: {term!r} is NOT absent from {src}")
        if it["expected"] == "answer" and not it.get("evidence"):
            problems.append(f"{it['id']}: an answerable question needs evidence")

    tuned = [q["question"] for f in ("eval/calibration_questions.json", "eval/near_miss_questions.json")
             for q in json.loads(Path(f).read_text(encoding="utf-8"))]
    for it in items:
        if it.get("overlap_ok"):  # a documented, reviewed exception (the reason is stored in the item)
            continue
        for t in tuned:
            a, b = words(it["question"]), words(t)
            overlap = len(a & b) / max(1, len(a | b))
            if overlap >= 0.6:
                problems.append(f"{it['id']}: too close to a tuning question ({overlap:.0%}): {t!r}")

    if all("split" not in it for it in items):  # assign once, stratified by type
        assign_split(items, random.Random(SEED))
        data["test_fingerprint"] = fingerprint(items)
        GOLDEN.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("split assigned and locked")
    elif any("split" not in it for it in items):
        # Questions added later (e.g. the PostHog collection) get their split the same way, among themselves only:
        # no existing question moves. Test items are added, so the lock needs --relock "reason" below.
        new = [it for it in items if "split" not in it]
        assign_split(new, random.Random(f"{SEED}:{','.join(sorted(it['id'] for it in new))}"))
        GOLDEN.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"split assigned to {len(new)} new item(s)")
    if data.get("test_fingerprint") != fingerprint(items):
        if RELOCK_REASON:
            test_runs = sorted(Path("eval/results").glob("*-test-*.jsonl"))
            if test_runs:  # the whole point of the lock: never change test items after seeing test results
                problems.append(f"refusing to relock: test results already exist ({test_runs[0].name})")
            else:
                data.setdefault("lock_history", []).append({"fingerprint": data.get("test_fingerprint"),
                                                            "replaced_on": time.strftime("%Y-%m-%d"),
                                                            "reason": RELOCK_REASON})
                data["test_fingerprint"] = fingerprint(items)
                GOLDEN.write_text(json.dumps(data, indent=1, ensure_ascii=False) + chr(10), encoding="utf-8")
                print("relocked (no test results existed); reason recorded in lock_history")
        else:
            problems.append("the TEST questions changed after the split was locked (questions, answers or evidence): "
                            "results on this set are no longer a clean held-out measurement "
                            "(if no test results exist yet: --relock \"reason\")")

    counts = Counter((it["type"], it.get("split")) for it in items)
    print(f"{len(items)} items: " + ", ".join(f"{t} {counts[(t, 'dev')]} dev / {counts[(t, 'test')]} test"
                                           for t in ("lookup", "multi_hop", "no_answer", "ambiguous")))
    for p in problems:
        print("PROBLEM:", p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def assign_split(items: list[dict], rng: random.Random) -> None:
    for kind in sorted({it["type"] for it in items}):
        group = [it for it in items if it["type"] == kind]
        rng.shuffle(group)
        n_test = round(len(group) * TEST_SHARE)
        for k, it in enumerate(group):
            it["split"] = "test" if k < n_test else "dev"


def fingerprint(items) -> str:
    test = sorted((it for it in items if it.get("split") == "test"), key=lambda it: it["id"])
    blob = json.dumps([[it["id"], it["question"], it["answer"], it.get("key_facts"), it.get("evidence")] for it in test],
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


if __name__ == "__main__":
    sys.exit(main())
