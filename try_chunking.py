"""Usage: python try_chunking.py [fixed|structure] [filename]"""
import sys
from pathlib import Path

from hyrag.chunking import chunk_fixed, chunk_structure
from hyrag.loader import load_document

# Our sample docs are tiny, so we use small numbers here to make the cuts visible.
SIZE, OVERLAP = 150, 30
strategy = sys.argv[1] if len(sys.argv) > 1 else "structure"
filename = sys.argv[2] if len(sys.argv) > 2 else "vpn-setup.md"

path = Path("sample_docs") / filename
doc = load_document(path.name, path.read_bytes())
chunker = {"fixed": chunk_fixed, "structure": chunk_structure}[strategy]
chunks = chunker(doc, size=SIZE, overlap=OVERLAP)

print(f"{doc.source}: {len(chunks)} chunks (strategy={strategy}, size={SIZE}, overlap={OVERLAP})\n")
for c in chunks:
    print(f"--- chunk {c.chunk_index}  [{c.char_count} chars]  heading: {c.heading}")
    print("|" + c.text.replace("\n", "\n|") + "|\n")
