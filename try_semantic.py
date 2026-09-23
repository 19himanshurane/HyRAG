"""Semantic vs structure-aware chunking on real corpus documents, with real Mistral embeddings.

Usage: python try_semantic.py
Embeddings are cached in data/embeddings.sqlite, so a second run makes no API calls.
"""
import logging
import statistics
from pathlib import Path

import numpy as np

from hyrag.chunking import chunk_semantic, chunk_structure, semantic_units
from hyrag.embeddings import MistralEmbedder
from hyrag.loader import load_document

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
DOCS = [
    "engineering/k8s-pod-lifecycle.md",
    "handbook/gitlab-time-off-types.html",
    "security/nist-sp-800-61r3-incident-response.pdf",
]


def sizes(chunks) -> str:
    s = sorted(c.char_count for c in chunks)
    return f"{len(chunks):4} chunks, median {int(statistics.median(s)):4} chars, min {s[0]:3}"


with MistralEmbedder() as embedder:
    for rel in DOCS:
        doc = load_document(rel, (Path("corpus") / rel).read_bytes())
        before = embedder.requests_made
        semantic = chunk_semantic(doc, embedder)
        structure = chunk_structure(doc)
        print(f"\n=== {rel}")
        print(f"   structure: {sizes(structure)}")
        print(f"   semantic : {sizes(semantic)}   ({embedder.requests_made - before} API requests)")

    # Where did semantic chunking cut, and are those real topic changes? For each document, the
    # clearest (lowest-similarity) topic cut inside a section, with the text on both sides.
    for rel in DOCS[:2]:
        doc = load_document(rel, (Path("corpus") / rel).read_bytes())
        best = None
        for section in doc.sections:
            units = semantic_units(section.text, doc.file_type, 800)
            if len(units) < 2:
                continue
            v = embedder.embed(units)
            for i in range(len(units) - 1):
                sim = float(v[i] @ v[i + 1])
                if best is None or sim < best[0]:
                    best = (sim, section.heading, units[i], units[i + 1])
        sim, heading, left, right = best
        print(f"\n=== lowest neighbour similarity in {rel}: {sim:.3f}")
        print(f"   section: {heading.split(' > ', 1)[-1]}")
        print(f"   before the cut: ...{left[-160:]!r}")
        print(f"   after the cut : {right[:160]!r}...")
