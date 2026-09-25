"""How good is an LLM judge at citation verification? Scores judge models on eval/judge_pairs.json
(hand-labeled claim/passage pairs: supported, partial, unsupported).

Usage (from the repo root): PYTHONPATH=. python scripts/eval_judge.py <model> [reasoning_effort|none]
Real Groq requests: one per pair (~32).

What matters most is FALSE SUPPORT: a judge saying "supported" for a claim that isn't lets a bad citation
through. Too strict (supported claims flagged) is annoying; too lenient defeats the purpose.
"""
import json
import sys
import time
from collections import Counter

from hyrag.citations import judge_pair
from hyrag.llm import GroqChat


def main() -> None:
    model = sys.argv[1]
    effort = None if len(sys.argv) > 2 and sys.argv[2] == "none" else (sys.argv[2] if len(sys.argv) > 2 else "low")
    pairs = json.load(open("eval/judge_pairs.json", encoding="utf-8"))
    confusion: Counter = Counter()
    mistakes = []
    t0 = time.perf_counter()
    with GroqChat(model=model, reasoning_effort=effort, max_completion_tokens=2048) as judge:
        for p in pairs:
            j = judge_pair(p["claim"], p["passage"], judge)
            confusion[(p["label"], j.verdict)] += 1
            if j.verdict != p["label"]:
                mistakes.append(f"  #{p['id']:>2} {p['label']:>11} -> {j.verdict:<11} ({p['note']}) {j.reason[:90]}")
        requests = judge.requests_made
    n = len(pairs)
    correct = sum(v for (label, got), v in confusion.items() if label == got)
    false_support = sum(v for (label, got), v in confusion.items() if label != "supported" and got == "supported")
    missed = sum(v for (label, got), v in confusion.items() if label == "supported" and got != "supported")
    unverified = sum(v for (_, got), v in confusion.items() if got in ("unverified", "error"))
    print(f"{model} (effort={effort}): exact {correct}/{n}, FALSE SUPPORT {false_support}/{n - 16}, "
          f"missed support {missed}/16, unverified/error {unverified}, {time.perf_counter() - t0:.0f}s, "
          f"{requests} requests")
    print("\n".join(mistakes))


if __name__ == "__main__":
    main()
