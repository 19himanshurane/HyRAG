"""Run the golden-set evaluation. Tune and debug on DEV only; score TEST once, at the end (docs/phase4.md).

Usage (from the repo root):
  PYTHONPATH=. python scripts/run_eval.py --check-grader            # is the grader trustworthy? (dev items)
  PYTHONPATH=. python scripts/run_eval.py --split dev --runs 3       # score the pipeline, spread over 3 runs
  options: --strategy structure|fixed|semantic   --limit N   --budget SECONDS
Results: eval/results/<time>-<split>-<strategy>.json (every record + per-run summaries + spread).

Cost: per question ~1 writer + ~4 checker/grader requests. The judge model's plan allows 1,000 requests/day.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from hyrag.citations import judge_client
from hyrag.embeddings import MistralEmbedder
from hyrag.evaluation import evaluate_item, grade_correctness, spread, summarize, to_json
from hyrag.index import ChunkIndex
from hyrag.llm import GroqChat
from hyrag.rerank import CrossEncoderScorer
from hyrag.retrieval import HybridRetriever

GOLDEN = Path("eval/golden_set.json")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--strategy", default="structure")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=float, default=180.0)
    ap.add_argument("--check-grader", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.check_grader:
        print("Scoring the held-out TEST split. Results must not be used to tune anything.", file=sys.stderr)
    items = load("dev" if args.check_grader else args.split)
    if args.limit:
        items = items[:args.limit]

    with judge_client() as grader:
        if args.check_grader:
            check_grader(items, grader)
            return 0
        with MistralEmbedder() as emb, GroqChat() as writer, judge_client() as judge:
            index = ChunkIndex(emb, strategy=args.strategy)
            if index.count() == 0:
                raise SystemExit(f"no index for strategy {args.strategy!r}: build it first")
            retriever = HybridRetriever(index, reranker=CrossEncoderScorer())
            runs, summaries = [], []
            for run in range(args.runs):
                records = []
                for n, it in enumerate(items, 1):
                    rec = evaluate_item(it, retriever, writer, judge, grader, budget_seconds=args.budget)
                    records.append(rec)
                    print(f"run {run + 1} {n:>2}/{len(items)} {it['id']} {it['type']:<9} {rec.status or rec.error[:40]:<11}"
                          f" correct={rec.correctness and rec.correctness['correct']} behaviour="
                          f"{rec.behaviour and rec.behaviour['acceptable']}", flush=True)
                summaries.append(summarize(records))
                runs.append(to_json(records))
    result = {"split": args.split, "strategy": args.strategy, "runs": args.runs, "questions": len(items),
              "summaries": summaries, "spread": spread(summaries), "records": runs}
    out = Path("eval/results") / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.split}-{args.strategy}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {out}")
    for key, v in result["spread"].items():
        print(f"  {key:34} mean {v['mean']:<7} range {v['min']}..{v['max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
