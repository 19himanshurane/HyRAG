"""Two searchable indexes over the same chunks, kept in sync:

- dense  (ChromaDB): nearest embedding vectors -> finds text that MEANS the same as the question
- sparse (BM25):     keyword scoring            -> finds text containing the question's exact words

One source of truth: a SQLite `chunks` table. Chroma is reconciled against it and BM25 is rebuilt from it,
so the two indexes can never disagree about which chunks exist. `check_sync()` proves it; `repair()` fixes
Chroma if a crash ever left it behind.
"""
import json
import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hyrag.chunking import Chunk, Embedder
from hyrag.loader import Document

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer: decides which words can match. Documents and queries go through the same function.
# ---------------------------------------------------------------------------

# Typography a typed query won't contain (audit G7): fold it so "don’t" matches "don't".
_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
# A word, or an identifier joined by _ - . ("err_tunnel_4012", "nimbus-deploy", "800-63b", "3.1.3.1").
_TOKEN = re.compile(r"[a-z0-9]+(?:[_\-.][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i if in into is it its of on or our s so t that the their "
    "then there these this to was we were what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words for BM25. An identifier is kept whole AND split into parts, so a search for the exact
    code "ERR_TUNNEL_4012" matches strongly (one rare token) and "tunnel 4012" still matches it."""
    text = unicodedata.normalize("NFKC", text).translate(_FOLD).lower()  # NFKC: no-break space -> space, ﬁ -> fi
    tokens: list[str] = []
    for match in _TOKEN.finditer(text):
        word = match.group(0)
        parts = [p for p in re.split(r"[_\-.]", word) if p]
        if len(parts) > 1:
            tokens.append(word)
        tokens.extend(p for p in parts if p not in STOPWORDS)
    return tokens


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    chunk: Chunk
    score: float  # dense: cosine similarity (1 = same meaning); sparse: BM25 score (unbounded, higher = better)


@dataclass
class IndexResult:
    doc_id: str
    added: int    # chunks embedded and stored
    removed: int  # chunks no longer in the document
    kept: int     # unchanged chunks: not re-embedded


_COLUMNS = ("chunk_id", "doc_id", "source", "text", "chunk_index", "heading", "page", "strategy", "char_count")


class ChunkIndex:
    """All chunks of ONE chunking strategy, searchable by meaning (Chroma) and by keywords (BM25)."""

    def __init__(self, embedder: Embedder, data_dir: Path = Path("data/index"), strategy: str = "structure"):
        import chromadb

        self.embedder = embedder
        self.strategy = strategy
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(data_dir / f"chunks-{strategy}.sqlite")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, source TEXT, "
            "text TEXT, chunk_index INTEGER, heading TEXT, page INTEGER, strategy TEXT, char_count INTEGER)"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc_id)")
        client = chromadb.PersistentClient(path=str(data_dir / "chroma"),
                                           settings=chromadb.Settings(anonymized_telemetry=False))
        # embedding_function=None: we always pass our own Mistral vectors. (With a function set, Chroma would
        # silently embed any text we forgot to give a vector for with a different model.)
        self.collection = client.get_or_create_collection(
            f"chunks-{strategy}", embedding_function=None, configuration={"hnsw": {"space": "cosine"}})
        self._bm25 = None
        self._bm25_ids: list[str] = []

    # ----- writing -----

    def index_document(self, doc: Document, chunks: list[Chunk]) -> IndexResult:
        """Make the index hold exactly `chunks` for this document. Chunk ids come from content, so only
        chunks whose text changed are embedded; removed ones disappear from BOTH indexes."""
        existing = set(self._ids_for(doc.doc_id))
        new = {c.chunk_id: c for c in chunks}
        to_add = [c for cid, c in new.items() if cid not in existing]
        to_remove = sorted(existing - set(new))
        kept = [c for cid, c in new.items() if cid in existing]

        # Embed first: if the API fails, nothing has been changed yet.
        vectors = self.embedder.embed([c.text_for_search() for c in to_add]) if to_add else None

        with self.db:  # one transaction: all of it or none of it
            if to_remove:
                self.db.executemany("DELETE FROM chunks WHERE chunk_id = ?", [(i,) for i in to_remove])
            self.db.executemany(f"INSERT OR REPLACE INTO chunks VALUES ({','.join('?' * len(_COLUMNS))})",
                                [self._row(c) for c in to_add + kept])  # kept: position/page may have moved
        if to_remove:
            self.collection.delete(ids=to_remove)
        if to_add:
            self.collection.add(ids=[c.chunk_id for c in to_add], embeddings=vectors.tolist(),
                                documents=[c.text for c in to_add], metadatas=[self._meta(c) for c in to_add])
        if kept:
            self.collection.update(ids=[c.chunk_id for c in kept], metadatas=[self._meta(c) for c in kept])
        self._bm25 = None  # rebuilt from the table on the next keyword search
        log.info("indexed %s: %d added, %d removed, %d unchanged", doc.source, len(to_add), len(to_remove), len(kept))
        return IndexResult(doc.doc_id, len(to_add), len(to_remove), len(kept))

    def remove_document(self, doc_id: str) -> int:
        ids = self._ids_for(doc_id)
        with self.db:
            self.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        if ids:
            self.collection.delete(ids=ids)
        self._bm25 = None
        return len(ids)

    # ----- searching -----

    def search_dense(self, query: str, k: int = 10) -> list[Hit]:
        if self.count() == 0:
            return []
        qv = self.embedder.embed([query])[0]
        res = self.collection.query(query_embeddings=[qv.tolist()], n_results=min(k, self.count()),
                                    include=["distances"])
        ids, distances = res["ids"][0], res["distances"][0]
        chunks = self._chunks(ids)
        return [Hit(chunks[i], 1.0 - d) for i, d in zip(ids, distances)]  # cosine distance = 1 - similarity

    def search_sparse(self, query: str, k: int = 10) -> list[Hit]:
        bm25 = self._bm25_index()
        tokens = tokenize(query)
        if bm25 is None or not tokens:
            return []
        scores = bm25.get_scores(tokens)
        top = [i for i in np.argsort(-scores, kind="stable")[:k] if scores[i] > 0]  # 0 = no query word present
        chunks = self._chunks([self._bm25_ids[i] for i in top])
        return [Hit(chunks[self._bm25_ids[i]], float(scores[i])) for i in top]

    # ----- consistency -----

    def check_sync(self) -> dict[str, list[str]]:
        """Ids present in one place but not the other. Empty lists everywhere = in sync."""
        table = set(self._all_ids())
        chroma = set(self.collection.get(include=[])["ids"])
        return {"missing_in_chroma": sorted(table - chroma), "extra_in_chroma": sorted(chroma - table)}

    def repair(self) -> dict[str, list[str]]:
        """Bring Chroma back in line with the table (after a crash between the two writes). BM25 needs no
        repair: it is always rebuilt from the table. Re-embedding is cheap because vectors are cached."""
        problems = self.check_sync()
        if problems["extra_in_chroma"]:
            self.collection.delete(ids=problems["extra_in_chroma"])
        if problems["missing_in_chroma"]:
            chunks = self._chunks(problems["missing_in_chroma"])
            ordered = [chunks[i] for i in problems["missing_in_chroma"]]
            vectors = self.embedder.embed([c.text_for_search() for c in ordered])
            self.collection.add(ids=[c.chunk_id for c in ordered], embeddings=vectors.tolist(),
                                documents=[c.text for c in ordered], metadatas=[self._meta(c) for c in ordered])
        if any(problems.values()):
            log.warning("repaired index: %s", {k: len(v) for k, v in problems.items()})
        return problems

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ----- internals -----

    def _bm25_index(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            rows = self.db.execute(f"SELECT {','.join(_COLUMNS)} FROM chunks ORDER BY chunk_id").fetchall()
            if not rows:
                return None
            chunks = [self._chunk(r) for r in rows]
            self._bm25_ids = [c.chunk_id for c in chunks]
            # Same text as the dense side embeds: heading path + body (headings hold codes the body never repeats).
            self._bm25 = BM25Okapi([tokenize(c.text_for_search()) for c in chunks])
        return self._bm25

    def _ids_for(self, doc_id: str) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))]

    def _all_ids(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT chunk_id FROM chunks")]

    def _chunks(self, ids: list[str]) -> dict[str, Chunk]:
        found: dict[str, Chunk] = {}
        for i in range(0, len(ids), 500):
            part = ids[i : i + 500]
            rows = self.db.execute(f"SELECT {','.join(_COLUMNS)} FROM chunks WHERE chunk_id IN "
                                   f"({','.join('?' * len(part))})", part)
            found.update({r[0]: self._chunk(r) for r in rows})
        return found

    @staticmethod
    def _row(c: Chunk) -> tuple:
        return (c.chunk_id, c.doc_id, c.source, c.text, c.chunk_index, c.heading, c.page, c.strategy, c.char_count)

    @staticmethod
    def _chunk(row: tuple) -> Chunk:
        return Chunk(**dict(zip(_COLUMNS, row)))

    @staticmethod
    def _meta(c: Chunk) -> dict:
        # Chroma metadata can't hold None, so "no page" (non-PDF sources) is stored as -1.
        return {"doc_id": c.doc_id, "source": c.source, "chunk_index": c.chunk_index, "heading": c.heading,
                "page": c.page if c.page is not None else -1, "strategy": c.strategy, "char_count": c.char_count}


def describe(hit: Hit) -> str:
    """One-line human-readable summary of a search hit (used by demos)."""
    where = f"p{hit.chunk.page}" if hit.chunk.page else hit.chunk.source
    return f"{hit.score:7.3f}  [{where} | {hit.chunk.heading[-70:]}]  {json.dumps(hit.chunk.text[:60])}"
