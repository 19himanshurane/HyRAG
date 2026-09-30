"""Meaning search (vectors in ChromaDB) and keyword search (BM25) over the same chunks.

A SQLite `chunks` table is the source of truth. BM25 is rebuilt from it and Chroma is reconciled against it,
so the two can't disagree about which chunks exist. check_sync() compares them; repair() fixes Chroma after a
crash between the two writes.
"""
import functools
import json
import logging
import math
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from rank_bm25 import BM25Okapi

from hyrag.chunking import Chunk, Embedder
from hyrag.dedup import NearDuplicateIndex
from hyrag.loader import Document

log = logging.getLogger(__name__)

# A Chroma server that stops answering must not hang searches (its calls run under the index's state lock).
# The chromadb HTTP client is built with NO timeout, so one is set on it here. 30 s covers loading every
# vector after a write (~1,300 x 1024 floats as JSON, well under a second on a local network).
CHROMA_TIMEOUT_SECONDS = 30.0
CHROMA_CONNECT_SECONDS = 5.0

# ----- collections: each company's documents, searched on their own -----
# One index can hold several companies' documents, but each question is answered from one company's only.
# Mixed, "How much paid time off do I get per year?" pulled GitLab, Nimbus and PostHog passages into one top 5.
# A document's collection is its first folder if that's a named collection ("posthog/handbook/..." is
# "posthog"); everything else is the demo set.
NAMED_COLLECTIONS = ("posthog",)
DEFAULT_COLLECTION = "demo"
COLLECTIONS = (DEFAULT_COLLECTION, *NAMED_COLLECTIONS)


def collection_of(source: str) -> str:
    first = source.split("/", 1)[0]
    return first if "/" in source and first in NAMED_COLLECTIONS else DEFAULT_COLLECTION


def check_collection(collection: str | None) -> None:
    if collection is not None and collection not in COLLECTIONS:
        raise ValueError(f"unknown collection {collection!r}; expected one of {COLLECTIONS}")

# ---------------------------------------------------------------------------
# Tokenizer: decides which words can match. Documents and queries go through the same function.
# ---------------------------------------------------------------------------

# Fold typography nobody types, so "don’t" in a document matches "don't" in a question.
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


class LuceneBM25(BM25Okapi):
    """BM25 with the word weight Lucene and Elasticsearch use: log(1 + (N - n + 0.5) / (n + 0.5)).

    rank_bm25's Okapi weight, log((N - n + 0.5) / (n + 0.5)), is 0 for a word found in exactly half the
    chunks and negative above that; its "epsilon floor" is a fraction of the AVERAGE weight, which is itself
    negative in a small index. Measured here: a 4-chunk index gave "err" weight 0, and a 1-chunk index could
    never match anything. Adding 1 inside the log keeps every weight positive and keeps the same order."""

    def _calc_idf(self, nd):
        for word, doc_count in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - doc_count + 0.5) / (doc_count + 0.5))
        self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0


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
    added: int           # chunks embedded and stored
    removed: int         # chunks no longer in the document
    kept: int            # unchanged chunks: not re-embedded
    duplicates: int = 0  # near-duplicates of an already-indexed chunk: flagged and skipped, not embedded


class EmbeddingModelMismatch(RuntimeError):
    """The index was built with a different embedding model: its vectors and the new ones aren't comparable."""


def _locked(method):
    """Run the method under the index's state lock: a web server calls one index from many threads, and
    SQLite, BM25 and the duplicate lookup are not safe to use concurrently. Hold it only for local work,
    never across a network call (an embedding request can take seconds with retries, and every search
    would queue behind it)."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _writer(method):
    """Run a write under the writer lock, so two writes can't interleave between planning a change and
    applying it. Searches don't take this lock, so they keep running while a write waits on the embedding API.
    Lock order: writer lock, then state lock; never the reverse."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._write_lock:
            return method(self, *args, **kwargs)

    return wrapper


_COLUMNS = ("chunk_id", "doc_id", "source", "text", "chunk_index", "heading", "page", "strategy", "char_count")
_CHUNK_TABLE = ", ".join(f"{c} {t}" for c, t in zip(_COLUMNS, (
    "TEXT PRIMARY KEY", "TEXT NOT NULL", "TEXT", "TEXT", "INTEGER", "TEXT", "INTEGER", "TEXT", "INTEGER")))


def _chroma_client(data_dir: Path, url: str | None):
    import chromadb
    import httpx

    settings = chromadb.Settings(anonymized_telemetry=False)
    if not url:
        return chromadb.PersistentClient(path=str(data_dir / "chroma"), settings=settings)
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"Chroma URL must look like http://host:port, got {url!r}")
    # The client's own first heartbeat has no timeout either, so a server that accepts connections but never
    # answers would hang startup: check it here first, with one. Unreachable -> fails now, not on a search.
    base = f"{parts.scheme}://{parts.netloc}"
    try:
        httpx.get(f"{base}/api/v2/heartbeat", timeout=CHROMA_CONNECT_SECONDS).raise_for_status()
    except httpx.HTTPError as e:
        raise ConnectionError(f"Chroma at {base} is not answering ({type(e).__name__})") from e
    client = chromadb.HttpClient(host=parts.hostname, port=parts.port or (443 if parts.scheme == "https" else 8000),
                                 ssl=parts.scheme == "https", settings=settings)
    session = getattr(getattr(client, "_server", None), "_session", None)
    if not isinstance(session, httpx.Client):  # a chromadb upgrade moved it: refuse to run with no timeout
        raise RuntimeError("cannot set a timeout on the chromadb HTTP client (its internals changed)")
    session.timeout = httpx.Timeout(CHROMA_TIMEOUT_SECONDS, connect=CHROMA_CONNECT_SECONDS)
    return client


class ChunkIndex:
    """All chunks of one chunking strategy, searchable by meaning (Chroma) and by keywords (BM25)."""

    def __init__(self, embedder: Embedder, data_dir: Path = Path("data/index"), strategy: str = "structure",
                 dedup: bool = True, chroma_url: str | None = None):
        """chroma_url (e.g. "http://chroma:8000"): use a Chroma server instead of the embedded store in
        data_dir/chroma. The chunk table stays in data_dir either way: it is the source of truth."""

        self.embedder = embedder
        self.strategy = strategy
        self.dedup = dedup
        self._lock = threading.RLock()        # state: SQLite, BM25, duplicate lookup, Chroma calls
        self._write_lock = threading.RLock()  # whole writes, including their embedding calls
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(data_dir / f"chunks-{strategy}.sqlite", check_same_thread=False)
        self.db.execute(f"CREATE TABLE IF NOT EXISTS chunks ({_CHUNK_TABLE})")
        self.db.execute("CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc_id)")
        # Chunks skipped as near-duplicates keep their full row, so they can be promoted back into the index
        # if the chunk they duplicate disappears (its document deleted or edited).
        self.db.execute(f"CREATE TABLE IF NOT EXISTS duplicates ({_CHUNK_TABLE}, duplicate_of TEXT NOT NULL, "
                        "similarity REAL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS duplicates_of ON duplicates (duplicate_of)")
        self._near: NearDuplicateIndex | None = None
        client = _chroma_client(data_dir, chroma_url)
        # embedding_function=None: we always pass our own Mistral vectors. (With a function set, Chroma would
        # silently embed any text we forgot to give a vector for with a different model.)
        self.collection = client.get_or_create_collection(
            f"chunks-{strategy}", embedding_function=None, configuration={"hnsw": {"space": "cosine"}})
        self._check_embedding_model()
        self._bm25: dict | None = None  # collection (None = all) -> (BM25, its chunk ids); see _bm25_index
        self._vectors: tuple[list[str], np.ndarray, np.ndarray] | None = None  # exact-search snapshot of Chroma
        self._seen_version = self._data_version()
        self._writes = 0  # this object's own committed writes (data_version only counts OTHER connections')

    def _check_embedding_model(self) -> None:
        """Vectors from two models live in different spaces: comparing them gives meaningless similarities,
        with no error. So the collection remembers which model built it, and refuses any other."""
        model = getattr(self.embedder, "model", None) or type(self.embedder).__name__
        recorded = (self.collection.metadata or {}).get("embedding_model")
        if recorded is None:
            if self.collection.count():
                log.warning("index %s has vectors but no recorded model; assuming %s", self.collection.name, model)
            self.collection.modify(metadata={"embedding_model": model})
        elif recorded != model:
            raise EmbeddingModelMismatch(
                f"index {self.collection.name!r} was built with {recorded!r}, not {model!r}: "
                f"rebuild it (delete its folder and re-index) to switch models")

    def _data_version(self) -> int:
        # Changes whenever ANOTHER connection commits to this database (not on our own commits).
        return self.db.execute("PRAGMA data_version").fetchone()[0]

    def _drop_stale_caches(self) -> None:
        """BM25 and the duplicate lookup are in-memory snapshots of the table. If another process or index
        object wrote to it since we built them, rebuild them instead of serving (and crashing on) old data."""
        version = self._data_version()
        if version != self._seen_version:
            self._bm25, self._near, self._vectors, self._seen_version = None, None, None, version

    # ----- writing -----

    @_writer
    def index_document(self, doc: Document, chunks: list[Chunk]) -> IndexResult:
        """Make the index hold exactly `chunks` for this document. Chunk ids come from content, so only
        chunks whose text changed are embedded; removed ones disappear from BOTH indexes. A new chunk whose
        body nearly repeats an indexed chunk (see hyrag.dedup) is flagged in `duplicates` and skipped."""
        result = self._sync(doc.doc_id, chunks)
        log.info("indexed %s: %d added, %d duplicates skipped, %d removed, %d unchanged",
                 doc.source, result.added, result.duplicates, result.removed, result.kept)
        return result

    @_writer
    def remove_document(self, doc_id: str) -> int:
        return self._sync(doc_id, []).removed

    @_locked
    def also_appears_in(self, chunk_id: str) -> list[Chunk]:
        """Where else this chunk's text occurs (skipped copies). For citations: "also in NIST SP 800-61r3"."""
        rows = self.db.execute(f"SELECT {','.join(_COLUMNS)} FROM duplicates WHERE duplicate_of = ? "
                               "ORDER BY source, chunk_index", (chunk_id,)).fetchall()
        return [self._chunk(r) for r in rows]

    @_locked
    def doc_ids(self) -> set[str]:
        """Every document with anything in this index (indexed chunks or skipped duplicates)."""
        rows = self.db.execute("SELECT doc_id FROM chunks UNION SELECT doc_id FROM duplicates")
        return {r[0] for r in rows}

    @_locked
    def duplicate_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM duplicates").fetchone()[0]

    def _sync(self, doc_id: str, chunks: list[Chunk]) -> IndexResult:
        """Plan under the state lock, embed with no state lock held, apply under the state lock. The caller
        holds the writer lock, so no other write in this process can change the plan in between."""
        with self._lock:
            plan = self._plan(doc_id, chunks)
        to_add = plan[1]
        try:
            # Embed before writing anything: if the API fails, nothing has changed.
            vectors = self.embedder.embed([c.text_for_search() for c in to_add]) if to_add else None
        except Exception:
            with self._lock:
                self._near = None  # the in-memory lookup was edited for writes that never happened
            raise
        with self._lock:
            return self._apply(doc_id, *plan, vectors)

    def _plan(self, doc_id: str, chunks: list[Chunk]):
        self._drop_stale_caches()
        existing = set(self._ids_for(doc_id))
        new = {c.chunk_id: c for c in chunks}
        to_remove = sorted(existing - set(new))
        kept = [c for cid, c in new.items() if cid in existing]
        # Skipped copies (in other documents) of chunks we are about to remove need a new home, or their
        # text would vanish from the index although their own document still has it.
        orphans = self._duplicates_of(to_remove)
        candidates = [c for cid, c in new.items() if cid not in existing] + [c for c in orphans if c.doc_id != doc_id]

        near = self._near_index()
        to_add: list[Chunk] = []
        skipped: list[tuple[Chunk, str, float]] = []
        try:
            for rid in to_remove:
                near.remove(rid)
            for c in candidates:
                hit = near.find(c.text) if self.dedup else None
                if hit:
                    skipped.append((c, *hit))
                else:
                    to_add.append(c)
                    near.add(c.chunk_id, c.text)
        except Exception:
            self._near = None
            raise
        return to_remove, to_add, kept, skipped, orphans

    def _apply(self, doc_id, to_remove, to_add, kept, skipped, orphans, vectors) -> IndexResult:
        with self.db:  # one transaction: all of it or none of it
            if to_remove:
                self.db.executemany("DELETE FROM chunks WHERE chunk_id = ?", [(i,) for i in to_remove])
            # This document's old duplicate records are recomputed above; promoted orphans leave the table.
            self.db.execute("DELETE FROM duplicates WHERE doc_id = ?", (doc_id,))
            self.db.executemany("DELETE FROM duplicates WHERE chunk_id = ?", [(c.chunk_id,) for c in orphans])
            self.db.executemany(f"INSERT OR REPLACE INTO chunks VALUES ({','.join('?' * len(_COLUMNS))})",
                                [self._row(c) for c in to_add + kept])  # kept: position/page may have moved
            self.db.executemany(f"INSERT OR REPLACE INTO duplicates VALUES ({','.join('?' * (len(_COLUMNS) + 2))})",
                                [(*self._row(c), of, round(sim, 4)) for c, of, sim in skipped])
        if to_remove:
            self.collection.delete(ids=to_remove)
        if to_add:
            self.collection.add(ids=[c.chunk_id for c in to_add], embeddings=vectors.tolist(),
                                documents=[c.text for c in to_add], metadatas=[self._meta(c) for c in to_add])
        if kept:
            self.collection.update(ids=[c.chunk_id for c in kept], metadatas=[self._meta(c) for c in kept])
        self._bm25 = None  # rebuilt from the table on the next keyword search
        self._writes += 1
        self._vectors = None  # and the exact-search matrix from Chroma on the next meaning search
        return IndexResult(doc_id, len(to_add), len(to_remove), len(kept), len(skipped))

    def _near_index(self) -> NearDuplicateIndex:
        if self._near is None:
            self._near = NearDuplicateIndex()
            for cid, text in self.db.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id"):
                self._near.add(cid, text)
        return self._near

    def _duplicates_of(self, chunk_ids: list[str]) -> list[Chunk]:
        found: list[Chunk] = []
        for i in range(0, len(chunk_ids), 500):
            part = chunk_ids[i : i + 500]
            rows = self.db.execute(f"SELECT {','.join(_COLUMNS)} FROM duplicates WHERE duplicate_of IN "
                                   f"({','.join('?' * len(part))})", part)
            found.extend(self._chunk(r) for r in rows)
        return found

    # ----- searching -----

    def search_dense(self, query: str, k: int = 10, collection: str | None = None) -> list[Hit]:
        """collection: search only that company's chunks (None = all of them)."""
        check_collection(collection)
        if not query.strip() or self.count() == 0:  # a blank query would still embed and "match" something
            return []
        qv = self.embedder.embed([query])[0]  # network call: deliberately outside the state lock
        with self._lock:
            return self._nearest(qv, k, collection)

    def _nearest(self, qv: np.ndarray, k: int, collection: str | None = None) -> list[Hit]:
        """Exact cosine top-k over every stored vector. Chroma's approximate HNSW search missed 1-2 of the true
        top 10 for half the test questions, and differently each run; a plain matrix product is exact and takes
        well under a millisecond here. Past ~100k chunks (~400 MB of vectors) an approximate index makes sense
        again, with its recall measured."""
        self._drop_stale_caches()
        ids, matrix, labels = self._vector_matrix()
        if not ids:  # emptied while we were embedding
            return []
        norm = float(np.linalg.norm(qv))
        if not np.isfinite(norm) or norm == 0:  # NaN scores would rank chunks arbitrarily
            log.warning("query embedding is unusable (norm %s); no meaning-search results", norm)
            return []
        sims = matrix @ (qv / norm)
        eligible = len(ids)
        if collection is not None:
            # Mask before taking the top k, or other companies' chunks could fill all k slots. The scores
            # that remain don't change.
            inside = labels == collection
            sims = np.where(inside, sims, -np.inf)
            eligible = int(inside.sum())
        k = min(k, eligible)
        if k == 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.lexsort((top, -sims[top]))]  # by similarity, ties by position (ids are sorted): stable
        found = [ids[i] for i in top]
        chunks = self._chunks(found)
        self._warn_missing(found, chunks)  # ids the table doesn't have (drift) are skipped, not fatal
        return [Hit(chunks[ids[i]], float(sims[i])) for i in top if ids[i] in chunks]

    def _vector_matrix(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """(ids sorted, normalised vectors, each chunk's collection) for every chunk in the table.

        Vectors come from the embedder's local cache when it has them, and from Chroma only for the rest.
        Fetching all 7,628 from a Chroma server took 96 s in pages and 692 s in one request on a
        memory-starved machine, which made the first question after a restart time out; the cache is a local
        SQLite read. A chunk with no vector anywhere is left out of meaning search, with a warning."""
        if self._vectors is None:
            rows = self.db.execute(f"SELECT {','.join(_COLUMNS)} FROM chunks ORDER BY chunk_id").fetchall()
            chunks = [self._chunk(r) for r in rows]
            vectors = self._stored_vectors(chunks)
            kept = [c for c in chunks if c.chunk_id in vectors]
            if len(kept) < len(chunks):
                log.warning("%d chunk(s) have no stored vector and are left out of meaning search; "
                            "run check_sync() / repair()", len(chunks) - len(kept))
            if not kept:
                self._vectors = ([], np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=object))
            else:
                matrix = np.stack([vectors[c.chunk_id] for c in kept]).astype(np.float32)
                matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
                labels = np.array([collection_of(c.source) for c in kept], dtype=object)
                self._vectors = ([c.chunk_id for c in kept], matrix, labels)
        return self._vectors

    def _stored_vectors(self, chunks: list[Chunk]) -> dict[str, np.ndarray]:
        cached = getattr(self.embedder, "cached", None)  # test embedders have no cache
        found: dict[str, np.ndarray] = {}
        if cached:
            for c, v in zip(chunks, cached([c.text_for_search() for c in chunks])):  # what was embedded
                if v is not None:
                    found[c.chunk_id] = v
        rest = [c.chunk_id for c in chunks if c.chunk_id not in found]
        for i in range(0, len(rest), 500):
            got = self.collection.get(ids=rest[i:i + 500], include=["embeddings"])
            found.update(zip(got["ids"], (np.asarray(e, dtype=np.float32) for e in got["embeddings"])))
        return found

    @_locked
    def warm(self) -> None:
        """Build the in-memory search data now (vectors, and keyword indexes per collection) instead of on the
        first question, which then paid for all of it inside its time budget."""
        self._drop_stale_caches()
        self._vector_matrix()
        for collection in (None, *COLLECTIONS):
            self._bm25_index(collection)

    @_locked
    def search_sparse(self, query: str, k: int = 10, collection: str | None = None) -> list[Hit]:
        """collection: search only that company's chunks (None = all of them), scored with that collection's
        own word statistics, so adding another company's documents never changes this one's ranking."""
        check_collection(collection)
        self._drop_stale_caches()
        bm25, bm25_ids = self._bm25_index(collection)
        tokens = tokenize(query)
        if bm25 is None or not tokens:
            return []
        scores = bm25.get_scores(tokens)
        top = [i for i in np.argsort(-scores, kind="stable")[:k] if scores[i] > 0]  # 0 = no query word present
        ids = [bm25_ids[i] for i in top]
        chunks = self._chunks(ids)
        self._warn_missing(ids, chunks)
        return [Hit(chunks[cid], float(scores[i])) for cid, i in zip(ids, top) if cid in chunks]

    @staticmethod
    def _warn_missing(ids: list[str], found: dict) -> None:
        missing = [i for i in ids if i not in found]
        if missing:
            log.warning("search skipped %d result(s) missing from the chunk table (%s...); "
                        "run check_sync() / repair()", len(missing), missing[0])

    # ----- consistency -----

    @_locked
    def check_sync(self) -> dict[str, list[str]]:
        """Ids present in one place but not the other. Empty lists everywhere = in sync."""
        table = set(self._all_ids())
        chroma = set(self.collection.get(include=[])["ids"])
        originals = {r[0] for r in self.db.execute("SELECT DISTINCT duplicate_of FROM duplicates")}
        return {"missing_in_chroma": sorted(table - chroma), "extra_in_chroma": sorted(chroma - table),
                "duplicates_of_missing_chunks": sorted(originals - table)}

    @_writer
    def repair(self) -> dict[str, list[str]]:
        """Bring Chroma back in line with the table (after a crash between the two writes), and promote
        skipped duplicates whose original is gone. BM25 needs no repair: it is always rebuilt from the table.
        Re-embedding is cheap because vectors are cached."""
        problems = self.check_sync()
        with self._lock:
            orphaned_docs = [r[0] for r in self.db.execute(
                "SELECT DISTINCT doc_id FROM duplicates WHERE duplicate_of NOT IN (SELECT chunk_id FROM chunks)")]
        for doc_id in orphaned_docs:
            # Re-syncing the document with its current chunks re-checks its skipped copies from scratch.
            with self._lock:
                current = self._chunks(self._ids_for(doc_id))
                rows = self.db.execute(f"SELECT {','.join(_COLUMNS)} FROM duplicates WHERE doc_id = ?", (doc_id,))
                skipped = [self._chunk(r) for r in rows]
            self._sync(doc_id, list(current.values()) + skipped)
        if problems["extra_in_chroma"]:
            with self._lock:
                self.collection.delete(ids=problems["extra_in_chroma"])
        if problems["missing_in_chroma"]:
            with self._lock:
                chunks = self._chunks(problems["missing_in_chroma"])
            ordered = [chunks[i] for i in problems["missing_in_chroma"] if i in chunks]  # may have changed since
            if ordered:
                vectors = self.embedder.embed([c.text_for_search() for c in ordered])  # no state lock held
                with self._lock:
                    self.collection.add(ids=[c.chunk_id for c in ordered], embeddings=vectors.tolist(),
                                        documents=[c.text for c in ordered],
                                        metadatas=[self._meta(c) for c in ordered])
        with self._lock:
            self._vectors = None  # Chroma changed underneath the exact-search snapshot
            self._writes += 1
        if any(problems.values()):
            log.warning("repaired index: %s", {k: len(v) for k, v in problems.items()})
        return problems

    @_locked
    def version(self) -> tuple:
        """Changes whenever the indexed content may have changed (a write by this object or by any other
        connection). Caches of answers built from this index key on it."""
        return (self.strategy, self._writes, self._data_version())

    @_locked
    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ----- internals -----

    def _bm25_index(self, collection: str | None = None) -> tuple["LuceneBM25 | None", list[str]]:
        """(BM25 over the collection's chunks, their ids in the same order); built on first use, one per
        collection (None = all chunks). Dropped on every write, like the other in-memory snapshots."""
        if self._bm25 is None:
            self._bm25 = {}
        if collection not in self._bm25:
            rows = self.db.execute(f"SELECT {','.join(_COLUMNS)} FROM chunks ORDER BY chunk_id").fetchall()
            chunks = [c for c in map(self._chunk, rows)
                      if collection is None or collection_of(c.source) == collection]
            # Same text as the dense side embeds: heading path + body (headings hold codes the body never repeats).
            bm25 = LuceneBM25([tokenize(c.text_for_search()) for c in chunks]) if chunks else None
            self._bm25[collection] = (bm25, [c.chunk_id for c in chunks])
        return self._bm25[collection]

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
