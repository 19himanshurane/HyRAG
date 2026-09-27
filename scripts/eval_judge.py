"""How good is an LLM judge at citation verification? Scores judge models on eval/judge_pairs.json
(hand-labeled claim/passage pairs: supported, partial, unsupported).

Usage (from the repo root): PYTHONPATH=. python scripts/eval_judge.py <model> [reasoning_effort|none] [--batch N]
Without --batch: one request per pair (judge_pair). With --batch N: N pairs per request (judge_batch, what
verify() uses), grouped in file order so batches mix passages - including the one that tells the judge to
answer "supported".

What matters most is FALSE SUPPORT: a judge saying "supported" for a claim that isn't lets a bad citation
through. Too strict (supported claims flagged) is annoying; too lenient defeats the purpose.
"""
import json
import sys
import time
from collections import Counter

from hyrag.citations import judge_batch, judge_pair
from hyrag.llm import GroqChat


class Metered:
    def __init__(self, chat):
        self.chat, self.tokens, self.calls = chat, 0, 0

    def complete(self, messages, json_schema=None):
        r = self.chat.complete(messages, json_schema=json_schema)
        self.tokens += r.prompt_tokens + r.completion_tokens
        self.calls += 1
        return r


def main() -> None:
    args = [a for a in sys.argv[1:]]
    batch = 0
    if "--batch" in args:
        i = args.index("--batch")
        batch = int(args[i + 1])
        del args[i:i + 2]
    model = args[0]
    effort = None if len(args) > 1 and args[1] == "none" else (args[1] if len(args) > 1 else "low")
    pairs = json.load(open("eval/judge_pairs.json", encoding="utf-8"))
    confusion: Counter = Counter()
    mistakes = []
    t0 = time.perf_counter()
    with GroqChat(model=model, reasoning_effort=effort, max_completion_tokens=4096) as chat:
        judge = Metered(chat)
        if batch:
            verdicts = []
            for k in range(0, len(pairs), batch):
                group = pairs[k:k + batch]
                # each pair gets its own passage number, even when two pairs share a passage text
                texts = {n: p["passage"] for n, p in enumerate(group, 1)}
                sources = {n: p.get("source", "") for n, p in enumerate(group, 1)}  # as verify() passes them
                verdicts += judge_batch([(p["claim"], n) for n, p in enumerate(group, 1)], texts, judge, sources)
        else:
            verdicts = [judge_pair(p["claim"], p["passage"], judge, p.get("source", "")) for p in pairs]
        requests = chat.requests_made
    for p, j in zip(pairs, verdicts):
        confusion[(p["label"], j.verdict)] += 1
        if j.verdict != p["label"]:
            mistakes.append(f"  #{p['id']:>2} {p['label']:>11} -> {j.verdict:<11} ({p['note']}) {j.reason[:90]}")
    n = len(pairs)
    n_supported = sum(p["label"] == "supported" for p in pairs)
    correct = sum(v for (label, got), v in confusion.items() if label == got)
    false_support = sum(v for (label, got), v in confusion.items() if label != "supported" and got == "supported")
    missed = sum(v for (label, got), v in confusion.items() if label == "supported" and got != "supported")
    unverified = sum(v for (_, got), v in confusion.items() if got in ("unverified", "error"))
    mode = f"batch {batch}" if batch else "one pair per call"
    print(f"{model} ({mode}): exact {correct}/{n}, FALSE SUPPORT {false_support}/{n - n_supported}, missed support "
          f"{missed}/{n_supported}, unverified/error {unverified}, {judge.calls} calls, {judge.tokens} tokens "
          f"({judge.tokens / n:.0f}/pair), {time.perf_counter() - t0:.0f}s, {requests} HTTP requests")
    print("\n".join(mistakes))


if __name__ == "__main__":
    main()
