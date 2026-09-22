"""Run every corpus file through the loader and print health signals, so parser problems show up as numbers.

Usage: python scripts/inspect_corpus.py [path-substring]   e.g.  python scripts/inspect_corpus.py k8s
"""
import re
import sys
import time
from collections import Counter
from pathlib import Path

from hyrag.loader import SUPPORTED, load_document

ROOT = Path("corpus")
filter_ = sys.argv[1] if len(sys.argv) > 1 else ""

SIGNALS = {
    "hugo_shortcodes": re.compile(r"\{\{[<%].*?[>%]\}\}"),        # {{< note >}} left in markdown
    "front_matter": re.compile(r"^(title|weight|reviewers|content_type):", re.M),
    "hyphen_breaks": re.compile(r"[a-z]-\n[a-z]"),                 # "config-\nuration" from PDFs
    "html_leftovers": re.compile(r"</?[a-z]+[^>]*>|&[a-z]+;"),
    "dot_leaders": re.compile(r"\.{5,}"),                          # table-of-contents "......... 12"
}

for path in sorted(p for p in ROOT.rglob("*") if p.suffix.lower() in SUPPORTED):
    rel = path.relative_to(ROOT).as_posix()
    if filter_ not in rel:
        continue
    t0 = time.perf_counter()
    try:
        doc = load_document(rel, path.read_bytes())
    except Exception as e:
        print(f"\n### {rel}\n   FAILED: {type(e).__name__}: {e}")
        continue
    secs = time.perf_counter() - t0
    all_text = "\n".join(s.text for s in doc.sections)
    lines = [l.strip() for l in all_text.split("\n") if l.strip()]
    repeated = [(l, c) for l, c in Counter(lines).most_common(5) if c > 3 and len(l) > 3]
    sizes = sorted(len(s.text) for s in doc.sections)

    print(f"\n### {rel}   ({path.stat().st_size:,} bytes -> {len(all_text):,} chars of text, {secs:.1f}s)")
    print(f"   sections: {len(doc.sections)}   section chars: min {sizes[0]}, median {sizes[len(sizes)//2]}, max {sizes[-1]}")
    hits = {name: len(rx.findall(all_text)) for name, rx in SIGNALS.items()}
    print("   noise signals: " + ", ".join(f"{k}={v}" for k, v in hits.items() if v) if any(hits.values()) else "   noise signals: none")
    if repeated:
        print("   most repeated lines: " + " | ".join(f"{l[:40]!r} x{c}" for l, c in repeated))
    headings = list(dict.fromkeys(s.heading for s in doc.sections))
    print(f"   distinct headings: {len(headings)}; first 6:")
    for h in headings[:6]:
        print(f"      - {h[:110]}")
