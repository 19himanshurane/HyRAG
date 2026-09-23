"""Near-duplicate detection for chunks, by TEXT, not by embedding.

Measured on the corpus (docs/phase1-audit.md has the method), cosine similarity of our embeddings is the
wrong signal for "the same text twice":
- it over-flags: 633 chunk pairs score > 0.95, almost all different sections of one document (median text
  overlap 25%), because the embedded text starts with a long shared heading path;
- it misses: NIST boilerplate copied word for word between two PDFs scores only 0.87-0.92, since each copy
  sits under a different document title.

So two chunks are duplicates when their BODY texts share >= 90% of their 3-word sequences AND contain
exactly the same "fact tokens" (numbers and identifiers). The second rule keeps bgwriter_flush_after and
backend_flush_after apart: their descriptions are 91% identical, but they are different settings.
"""
import re
import unicodedata
from collections import defaultdict

_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_IDENTIFIER = re.compile(r"[a-z0-9]+(?:[_\-.][a-z0-9]+)+|\w*\d\w*")


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_FOLD).lower()


def shingles(text: str, n: int = 3) -> frozenset[str]:
    """All runs of n consecutive words. Two texts share most of them only if the wording is nearly the same."""
    words = re.findall(r"\w+", _normalise(text))
    if len(words) <= n:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))


def fact_tokens(text: str) -> frozenset[str]:
    """Numbers and identifiers (anything with a digit, or joined by _ - .): the parts of a sentence where a
    one-token difference changes the fact ("backend_flush_after", "4012", "30 days", "3.1.5")."""
    return frozenset(_IDENTIFIER.findall(_normalise(text)))


class NearDuplicateIndex:
    """In-memory lookup: given a text, find an already-indexed chunk whose body is (nearly) the same."""

    def __init__(self, threshold: float = 0.9, min_chars: int = 120):
        self.threshold = threshold
        # Bodies shorter than this are never deduplicated: glossary entries like "See cybersecurity incident."
        # repeat word for word, but their headings (the term being defined) are what make them different.
        self.min_chars = min_chars
        self._shingles: dict[str, frozenset[str]] = {}
        self._facts: dict[str, frozenset[str]] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)

    def add(self, chunk_id: str, text: str) -> None:
        if len(text) < self.min_chars:
            return
        s = shingles(text)
        self._shingles[chunk_id], self._facts[chunk_id] = s, fact_tokens(text)
        for g in s:
            self._postings[g].add(chunk_id)

    def remove(self, chunk_id: str) -> None:
        for g in self._shingles.pop(chunk_id, ()):
            self._postings[g].discard(chunk_id)
        self._facts.pop(chunk_id, None)

    def find(self, text: str) -> tuple[str, float] | None:
        """(chunk_id, similarity) of the closest duplicate, or None."""
        if len(text) < self.min_chars:
            return None
        s, facts = shingles(text), fact_tokens(text)
        shared: dict[str, int] = defaultdict(int)
        for g in s:
            for cid in self._postings.get(g, ()):
                shared[cid] += 1
        best: tuple[str, float] | None = None
        for cid, n in shared.items():
            similarity = n / len(s | self._shingles[cid])  # Jaccard: shared / all distinct 3-word sequences
            if similarity >= self.threshold and self._facts[cid] == facts:
                if best is None or similarity > best[1] or (similarity == best[1] and cid < best[0]):
                    best = (cid, similarity)
        return best
