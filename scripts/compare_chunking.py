"""Compare the three chunking strategies on RETRIEVAL (Phase 4 Step 3a). No LLM calls: evidence is found by exact
quote in the retrieved chunks, and the reranker runs locally. Dev split only: choosing a strategy is tuning.

Usage (from the repo root): PYTHONPATH=. python scripts/compare_chunking.py [--split dev]
Needs each strategy's index (scripts/build_index.py --strategy ...).

Metrics per strategy, over the answerable questions' evidence quotes:
- recall@5: share of evidence quotes found in the top 5 (what the writer sees);
- hit@5: share of questions with at least one evidence quote in the top 5;
- all@5: share of questions with ALL their evidence in the top 5 (multi-hop needs both documents);
- recall@20: share found among the top 20 fused candidates before reranking (what the reranker had to work with);
- MRR: mean of 1/rank of the first evidence quote in the top 5 (0 when absent).
A quote split across two chunks counts as missing: that is a real cost of a strategy that cuts mid-passage.
"""
import argparse
import json
import statistics
from pathlib import Path

from hyrag.embeddings import MistralEmbedder
from hyrag.evaluation import evidence_found
from hyrag.index import DEFAULT_COLLECTION, ChunkIndex
from hyrag.rerank import CrossEncoderScorer
from hyrag.retrieval import HybridRetriever

STRATEGIES = ("fixed", "structure", "semantic")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    args = ap.parse_args()
    items = [it for it in json.loads(Path("eval/golden_set.json").read_text(encoding="utf-8"))["items"]
             if it["split"] == args.split and it["expected"] == "answer" and it.get("evidence")]
    scorer = CrossEncoderScorer()
    rows, per_question = {}, {}
    with MistralEmbedder() as emb:
        for strategy in STRATEGIES:
            index = ChunkIndex(emb, strategy=strategy)
            retriever = HybridRetriever(index, reranker=scorer)
            r5, r20, hit, allin, rr = [], [], [], [], []
            for it in items:
                coll = it.get("collection", DEFAULT_COLLECTION)  # as in hyrag.evaluation.evaluate_item
                top5 = evidence_found(it["evidence"], retriever.search(it["question"], collection=coll).hits)
                top20 = evidence_found(it["evidence"],
                                       retriever.search(it["question"], rerank_results=False, collection=coll).hits)
                r5 += [x is not None for x in top5]
                r20 += [x is not None for x in top20]
                hit.append(any(x is not None for x in top5))
                allin.append(all(x is not None for x in top5))
                first = min((x for x in top5 if x is not None), default=None)
                rr.append(1 / first if first else 0.0)
                per_question.setdefault(it["id"], {})[strategy] = top5
            sizes = [r[0] for r in index.db.execute("SELECT char_count FROM chunks")]
            rows[strategy] = {"chunks": index.count(), "median_chars": statistics.median(sizes),
                              "recall@5": statistics.mean(r5), "hit@5": statistics.mean(hit),
                              "all@5": statistics.mean(allin), "recall@20": statistics.mean(r20), "MRR": statistics.mean(rr)}
        print(f"{len(items)} answerable {args.split} questions, {sum(len(it['evidence']) for it in items)} evidence quotes; "
              f"Mistral requests {emb.requests_made}\n")
    print(f"{'strategy':10} {'chunks':>6} {'median':>6}  recall@5  hit@5  all@5  recall@20   MRR")
    for s, m in rows.items():
        print(f"{s:10} {m['chunks']:>6} {m['median_chars']:>6.0f}  {m['recall@5']:8.2f} {m['hit@5']:6.2f} {m['all@5']:6.2f}"
              f" {m['recall@20']:10.2f} {m['MRR']:6.2f}")
    print("\nquestions where the strategies disagree (evidence rank in top 5; - = not found):")
    for qid, by in per_question.items():
        shown = {s: [x if x is not None else "-" for x in ranks] for s, ranks in by.items()}
        if len({str(v) for v in shown.values()}) > 1:
            print(f"  {qid}: " + "  ".join(f"{s} {v}" for s, v in shown.items()))
    out = Path("eval/results") / f"chunking-retrieval-{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "metrics": rows, "per_question": per_question}, indent=1), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
