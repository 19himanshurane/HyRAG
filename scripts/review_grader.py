"""Record whether a person agrees with the grader's verdicts on real answers.

The grader passed a synthetic check (69/69), but real answers are harder: a first look found it marking a fact
"missing" that the answer had stated in other words. The evaluation is only as good as the grader, so its
agreement with a human gets measured and reported.

Usage (from the repo root):
  PYTHONPATH=. python scripts/review_grader.py show  eval/results/<run>.jsonl      # print each graded answer
  PYTHONPATH=. python scripts/review_grader.py record eval/results/<run>.jsonl <ID> <run> agree|disagree "reason"
  PYTHONPATH=. python scripts/review_grader.py report
Reviews are kept in eval/grader_review.json (dev results only: never review test answers to tune anything).
"""
import json
import sys
from pathlib import Path

REVIEWS = Path("eval/grader_review.json")
GOLDEN = Path("eval/golden_set.json")


def records(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows[1:] if r.get("correctness") and r["correctness"].get("score") is not None]


def show(path: Path) -> None:
    gold = {it["id"]: it for it in json.loads(GOLDEN.read_text(encoding="utf-8"))["items"]}
    reviewed = {(r["file"], r["id"], r["run"]) for r in load_reviews()}
    for r in records(path):
        g = r["correctness"]
        mark = "reviewed" if (path.name, r["id"], r["run"]) in reviewed else "TO REVIEW"
        print(f"=== {r['id']} run {r['run']} [{mark}] grader: correct={g['correct']} score={g['score']} "
              f"{'(graded the WITHHELD draft)' if r.get('graded_draft') else ''}")
        print(f"Q: {gold[r['id']]['question']}")
        for fact, verdict in zip(gold[r["id"]]["key_facts"], g["facts"]):
            print(f"   {verdict:12} {fact}")
        print(f"A: {(r.get('answer') or r.get('draft') or '')[:600]}")
        print(f"grader's reason: {g.get('reason', '')[:300]}\n")


def load_reviews() -> list[dict]:
    return json.loads(REVIEWS.read_text(encoding="utf-8")) if REVIEWS.exists() else []


def record(path: Path, qid: str, run: int, verdict: str, reason: str) -> None:
    if verdict not in ("agree", "disagree"):
        raise SystemExit("verdict must be agree or disagree")
    match = [r for r in records(path) if r["id"] == qid and r["run"] == run]
    if not match:
        raise SystemExit(f"no graded record {qid} run {run} in {path}")
    if json.loads(path.read_text(encoding="utf-8").splitlines()[0])["split"] != "dev":
        raise SystemExit("only dev results are reviewed")
    reviews = [r for r in load_reviews() if not (r["file"] == path.name and r["id"] == qid and r["run"] == run)]
    reviews.append({"file": path.name, "id": qid, "run": run, "grader_correct": match[0]["correctness"]["correct"],
                    "human": verdict, "reason": reason})
    REVIEWS.write_text(json.dumps(reviews, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"recorded {qid} run {run}: {verdict}")


def report() -> None:
    reviews = load_reviews()
    if not reviews:
        print("no reviews yet")
        return
    agree = sum(r["human"] == "agree" for r in reviews)
    too_strict = [r for r in reviews if r["human"] == "disagree" and r["grader_correct"] is False]
    too_lenient = [r for r in reviews if r["human"] == "disagree" and r["grader_correct"] is True]
    print(f"grader-human agreement: {agree}/{len(reviews)} ({agree / len(reviews):.0%}); "
          f"too strict {len(too_strict)}, too lenient {len(too_lenient)}")
    for r in too_strict + too_lenient:
        print(f"  {r['id']} run {r['run']} grader_correct={r['grader_correct']}: {r['reason']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "show":
        show(Path(sys.argv[2]))
    elif cmd == "record":
        record(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6] if len(sys.argv) > 6 else "")
    else:
        report()
