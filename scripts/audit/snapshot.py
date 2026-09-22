"""Save or compare a "golden" snapshot of parser + chunker output, to prove a change altered nothing unintended.

Usage (from the repo root):
  python scripts/audit/snapshot.py save    <dir>
  python scripts/audit/snapshot.py compare <dir>
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

from hyrag.chunking import chunk_fixed, chunk_structure
from hyrag.loader import list_corpus_files, load_document

FILES = list_corpus_files(Path("corpus")) + list_corpus_files(Path("sample_docs"))


def snapshot(path: Path) -> dict:
    doc = load_document(path.relative_to(path.parts[0]).as_posix(), path.read_bytes())
    return {
        "document": asdict(doc),
        "fixed": [c.text for c in chunk_fixed(doc)],
        "structure": [(c.heading, c.page, c.text) for c in chunk_structure(doc)],
    }


mode, out = sys.argv[1], Path(sys.argv[2])
if mode == "save":
    out.mkdir(parents=True, exist_ok=True)
    for p in FILES:
        name = p.as_posix().replace("/", "__") + ".json"
        (out / name).write_text(json.dumps(snapshot(p), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {len(FILES)} snapshots to {out}")
else:
    changed = 0
    for p in FILES:
        f = out / (p.as_posix().replace("/", "__") + ".json")
        if not f.exists():
            print(f"NEW      {p.as_posix()}")
            continue
        old, new = json.loads(f.read_text(encoding="utf-8")), json.loads(json.dumps(snapshot(p), ensure_ascii=False))
        diffs = [k for k in ("document", "fixed", "structure") if old[k] != new[k]]
        if diffs:
            changed += 1
            print(f"CHANGED  {p.as_posix()}  ({', '.join(diffs)})")
    print(f"{changed} of {len(FILES)} files differ from the snapshot")
