"""Run the golden-set evaluation. Tune and debug on DEV only; score TEST once, at the end (docs/phase4.md).

Usage (from the repo root):
  PYTHONPATH=. python scripts/run_eval.py --check-grader            # is the grader trustworthy? (dev items)
  PYTHONPATH=. python scripts/run_eval.py --split dev --runs 2       # score the pipeline, spread over runs
  PYTHONPATH=. python scripts/run_eval.py --resume eval/results/<name>.jsonl   # continue a stopped run
  PYTHONPATH=. python scripts/run_eval.py --summarize eval/results/<name>.jsonl
  options: --strategy structure|fixed|semantic   --limit N   --budget SECONDS

Crash-safe: every question's record is appended to eval/results/<time>-<split>-<strategy>.jsonl the moment it
is scored (one JSON object per line, flushed), and "started <id>" is printed before each question, so a hang
is visible at once and a stopped run loses nothing. (The first version wrote results only at the end; a run
hung on question 10 after an hour and everything was lost.) Do not rebuild an index while a run reads it.

Cost: per question ~1 writer + ~4 checker/grader requests. The judge model's plan allows 1,000 requests/day.
"""
import argparse
import dataclasses
import faulthandler
import json
import sys
import time
from pathlib import Path

from hyrag.citations import judge_client
from hyrag.embeddings import MistralEmbedder
from hyrag.evaluation import Record, evaluate_item, grade_correctness, spread, summarize
from hyrag.index import ChunkIndex
from hyrag.llm import GroqChat
from hyrag.rerank import CrossEncoderScorer
from hyrag.retrieval import HybridRetriever

GOLDEN = Path("eval/golden_set.json")
RESULTS = Path("eval/results")


def load(split: str) -> list[dict]:
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return [it for it in data["items"] if split == "all" or it["split"] == split]


def check_grader(items: list[dict], grader) -> None:
    """Feed the grader answers whose grade we KNOW, and count how often it agrees.
    reference -> correct; first key fact only -> partial; another question's answer -> not correct;
    "I don't know." -> score 0."""
    items = [it for it in items if it["expected"] == "answer" and len(it["key_facts"]) >= 1]
    agree, total, rows = 0, 0, []
    for k, it in enumerate(items):
        other = items[(k + 1) % len(items)]
        cases = [("reference", it["answer"], lambda g: g["correct"] is True),
                 ("unrelated answer", other["answer"], lambda g: g["correct"] is False),
                 ("I don't know", "I don't know.", lambda g: g["score"] == 0.0)]
        if len(it["key_facts"]) >= 2:
            cases.append(("first fact only", it["key_facts"][0] + ".", lambda g: g["correct"] is False and g["score"] > 0))
        for name, text, ok in cases:
            g = grade_correctness(it, text, grader)
            good = g["score"] is not None and ok(g)
            agree += good
            total += 1
            if not good:
                rows.append(f"  {it['id']} {name:17} -> score {g['score']} facts {g['facts']} ({g['reason'][:90]})")
    print(f"grader agreed with the known grade in {agree}/{total} cases")
    print("\n".join(rows))


def read_jsonl(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[0], lines[1:]  # header, records


def report(path: Path) -> None:
    header, rows = read_jsonl(path)
    by_run: dict[int, list[Record]] = {}
    fields = {f.name for f in dataclasses.fields(Record)}
    for row in rows:
        by_run.setdefault(row["run"], []).append(Record(**{k: v for k, v in row.items() if k in fields}))
    summaries = [summarize(recs) for _, recs in sorted(by_run.items())]
    complete = [len(recs) for recs in by_run.values()]
    print(f"{path}: {header['split']} / {header['strategy']}, {len(summaries)} run(s), questions per run {complete}")
    for key, v in spread(summaries).items():
        print(f"  {key:34} mean {v['mean']:<7} range {v['min']}..{v['max']}")
    out = path.with_suffix(".summary.json")
    out.write_text(json.dumps({"header": header, "summaries": summaries, "spread": spread(summaries)}, indent=1),
                   encoding="utf-8")
    print(f"saved {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--strategy", default="structure")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=float, default=180.0)
    ap.add_argument("--check-grader", action="store_true")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--summarize", type=Path)
    args = ap.parse_args()
    if args.summarize:
        report(args.summarize)
        return 0

    if args.resume:
        header, done_rows = read_jsonl(args.resume)
        path = args.resume
        split, strategy, runs, limit = header["split"], header["strategy"], header["runs"], header.get("limit", 0)
        done = {(r["run"], r["id"]) for r in done_rows}
    else:
        split, strategy, runs, limit = args.split, args.strategy, args.runs, args.limit
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}-{split}-{strategy}.jsonl"
        header = {"split": split, "strategy": strategy, "runs": runs, "limit": limit, "budget": args.budget,
                  "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        path.write_text(json.dumps(header) + "\n", encoding="utf-8")
        done = set()
    if split == "test" and not args.check_grader:
        print("Scoring the held-out TEST split. Results must not be used to tune anything.", file=sys.stderr)
    items = load("dev" if args.check_grader else split)
    if limit:
        items = items[:limit]

    with judge_client() as grader:
        if args.check_grader:
            check_grader(items, grader)
            return 0
        with MistralEmbedder() as emb, GroqChat() as writer, judge_client() as judge:
            index = ChunkIndex(emb, strategy=strategy)
            if index.count() == 0:
                raise SystemExit(f"no index for strategy {strategy!r}: build it first")
            retriever = HybridRetriever(index, reranker=CrossEncoderScorer(), rewriter=writer)
            print(f"writing to {path}", flush=True)
            for run in range(1, runs + 1):
                for n, it in enumerate(items, 1):
                    if (run, it["id"]) in done:
                        continue
                    print(f"run {run} {n:>2}/{len(items)} started {it['id']} {time.strftime('%H:%M:%S')}", flush=True)
                    # Watchdog: if one question runs far past its budget, print every thread's stack (repeating),
                    # so a hang shows exactly where it is instead of silence.
                    faulthandler.dump_traceback_later(args.budget + 60, repeat=True, file=sys.stderr)
                    rec = evaluate_item(it, retriever, writer, judge, grader, budget_seconds=args.budget)
                    faulthandler.cancel_dump_traceback_later()
                    # A daily quota running out mid-run makes this record an outage result, not a quality one:
                    # don't save it; stop, and resume the run once the quota has refilled.
                    trail = json.dumps(dataclasses.asdict(rec))
                    if "quota_exhausted" in trail or "QuotaExhausted" in trail:
                        print(f"daily quota exhausted at {it['id']}: stopped without saving it. Resume later with:\n"
                              f"  PYTHONPATH=. python scripts/run_eval.py --resume {path}", flush=True)
                        report(path)
                        return 2
                    with path.open("a", encoding="utf-8") as f:  # one line per question, on disk at once
                        f.write(json.dumps({"run": run, **dataclasses.asdict(rec)}, ensure_ascii=False) + "\n")
                    print(f"run {run} {n:>2}/{len(items)} {it['id']} {it['type']:<9} {rec.status or rec.error[:40]:<11}"
                          f" correct={rec.correctness and rec.correctness['correct']} behaviour="
                          f"{rec.behaviour and rec.behaviour['acceptable']} {rec.seconds}s", flush=True)
    report(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
