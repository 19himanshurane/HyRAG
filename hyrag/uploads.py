"""Visitor uploads: a visitor's own document, answered from on its own and held only in memory.

An uploaded file must not join the shared index. There it would be searchable by every visitor, each upload
would rebuild the keyword index and the vector matrix for all the built-in chunks and change the version the
answer cache keys on, and the near-duplicate check could skip one of its passages because someone else's file
already holds the same text. So each upload gets a small index of its own (UploadIndex), and the usual
retriever, reranker, writer and checker run against that.

Nothing is stored. The parsed passages and their vectors live in this process until the upload expires, is
removed, or is pushed out by newer ones; they are embedded without the on-disk embedding cache. The text does
leave the machine: passages go to the embedding API, and the ones retrieved for a question go to the chat
models.

The upload id is a random token and the only way to reach a document, so it is the secret. The visitor id is
only used for the limits, and several people behind one office address can share it.

The limits are there because a public copy runs on someone's API keys and a small machine: file size and page
count bound parsing time and memory, the passage count bounds what one upload costs to embed, and the hourly
and daily counts bound the total. One file is read at a time for the same reason.
"""
import logging
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hyrag.chunking import Chunk, Embedder, chunk_structure
from hyrag.index import Hit, LuceneBM25, tokenize
from hyrag.limits import AskLimiter
from hyrag.loader import MAX_FILE_BYTES, Document, NoTextError, load_document, pdf_info

log = logging.getLogger(__name__)

UPLOAD_TYPES = (".pdf", ".md", ".txt")
# Below this a file is a scan with a stray watermark or page number, not a document to answer from.
MIN_TEXT_CHARS = 200
BUSY_WAIT_SECONDS = 20.0  # how long an upload waits for the one being read before it is told to retry
_UNSAFE = re.compile(r"[^\w.\- ]")  # the name is shown on the page and put in the model's prompt

NO_TEXT = ("No text could be read from this file. If it is a scanned PDF, its pages are images, and HyRAG reads "
           "text only.")


@dataclass(frozen=True)
class UploadPolicy:
    max_bytes: int = 5 * 1024 * 1024
    max_pages: int = 50              # PDFs: about 90 ms of parsing a page
    max_chunks: int = 400            # passages, embedded 32 to a request
    ttl_seconds: float = 3600.0
    per_visitor_per_hour: int = 5    # 0 turns a limit off
    per_day: int = 50
    max_live: int = 20               # uploads held at once, over all visitors (a few MB each at most)
    max_live_per_visitor: int = 3

    def __post_init__(self):
        if not 0 < self.max_bytes <= MAX_FILE_BYTES:
            raise ValueError(f"max_bytes must be between 1 and the loader's {MAX_FILE_BYTES:,}")
        if min(self.max_pages, self.max_chunks, self.max_live, self.max_live_per_visitor) < 1 or self.ttl_seconds <= 0:
            raise ValueError("page, passage, live-upload and time limits must be positive")


class UploadRejected(Exception):
    """The file can't be taken. `status` is the HTTP status that says why; the message is for the visitor."""

    def __init__(self, message: str, status: int = 422, retry_after: int | None = None):
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


class UploadIndex:
    """One uploaded document, searchable by meaning and by keywords like hyrag.index.ChunkIndex, but in memory
    only and never changing. It has the methods the retriever and ask() use, and no others."""

    strategy = "structure"

    def __init__(self, upload_id: str, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder):
        """vectors: one row per chunk, each scaled to length 1 (what the embedder returns)."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.upload_id, self.chunks, self.embedder = upload_id, chunks, embedder
        self._matrix = np.asarray(vectors, dtype=np.float32)
        # Same text as the dense side embeds: heading path + body.
        self._bm25 = LuceneBM25([tokenize(c.text_for_search()) for c in chunks]) if chunks else None

    def count(self) -> int:
        return len(self.chunks)

    def version(self) -> tuple:
        return ("upload", self.upload_id)  # never equal to a ChunkIndex version or to another upload's

    def search_dense(self, query: str, k: int = 10, collection: str | None = None) -> list[Hit]:
        self._no_collection(collection)
        if not query.strip() or not self.chunks:
            return []
        qv = self.embedder.embed([query])[0]
        norm = float(np.linalg.norm(qv))
        if not np.isfinite(norm) or norm == 0:  # NaN scores would rank chunks arbitrarily
            log.warning("query embedding is unusable (norm %s); no meaning-search results", norm)
            return []
        sims = self._matrix @ (qv / norm)
        top = np.lexsort((np.arange(len(sims)), -sims))[:k]  # by similarity, ties by position: stable
        return [Hit(self.chunks[i], float(sims[i])) for i in top]

    def search_sparse(self, query: str, k: int = 10, collection: str | None = None) -> list[Hit]:
        self._no_collection(collection)
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top = [i for i in np.argsort(-scores, kind="stable")[:k] if scores[i] > 0]  # 0 = no query word present
        return [Hit(self.chunks[i], float(scores[i])) for i in top]

    @staticmethod
    def _no_collection(collection: str | None) -> None:
        if collection is not None:
            raise ValueError("an uploaded document is not part of a collection")


@dataclass
class Upload:
    upload_id: str
    visitor: str
    filename: str
    pages: int | None  # PDFs only
    characters: int
    index: UploadIndex
    created: float


def display_name(filename: str) -> str:
    """The file's own name, reduced to characters that are safe to show and to put in a prompt."""
    name = Path((filename or "").replace("\\", "/")).name
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    if not ext and stem.startswith("."):  # a file called just ".pdf": pathlib reads that as a name, not a type
        stem, ext = "", stem.lower()
    stem = " ".join(_UNSAFE.sub("_", stem).split())[:80].strip(" .") or "document"
    return stem + _UNSAFE.sub("", ext)


class UploadStore:
    def __init__(self, embedder: Embedder, policy: UploadPolicy = UploadPolicy(),
                 chunker: Callable[[Document], list[Chunk]] = chunk_structure,
                 clock: Callable[[], float] = time.monotonic):
        """embedder: one WITHOUT an on-disk cache, or uploaded text's vectors would outlive the upload."""
        self.embedder, self.policy, self.chunker, self.clock = embedder, policy, chunker, clock
        self._limiter = AskLimiter(policy.per_visitor_per_hour, policy.per_day, clock)
        self._uploads: dict[str, Upload] = {}  # in the order they arrived: the first is the oldest
        self._lock = threading.Lock()          # the dict
        self._busy = threading.Lock()          # one file is parsed and embedded at a time

    def add(self, visitor: str, filename: str, data: bytes) -> Upload:
        """Parse, chunk and embed one file, and hold it. Raises UploadRejected with the reason if it can't."""
        p = self.policy
        name = display_name(filename)
        if Path(name).suffix not in UPLOAD_TYPES:
            raise UploadRejected(f"Only {', '.join(UPLOAD_TYPES)} files can be uploaded.", 415)
        if len(data) > p.max_bytes:
            raise UploadRejected(f"The file is larger than {p.max_bytes / (1024 * 1024):g} MB.", 413)
        blocked = self._limiter.check(visitor)
        if blocked:
            which, wait = blocked
            raise UploadRejected(
                f"This demo allows {p.per_visitor_per_hour} uploads per visitor per hour. Try again in about "
                f"{max(1, round(wait / 60))} min." if which == "visitor" else
                "This demo has taken all the uploads it can for today. Please come back later.", 429, wait)
        if not self._busy.acquire(timeout=BUSY_WAIT_SECONDS):
            raise UploadRejected("Another document is being read right now. Try again in a moment.", 503, 15)
        try:
            started = time.perf_counter()
            doc, chunks, pages = self._read(name, data)
            try:
                vectors = self.embedder.embed([c.text_for_search() for c in chunks])
            except Exception as e:
                log.warning("embedding an upload failed (%s: %s)", type(e).__name__, str(e)[:120])
                raise UploadRejected("The document couldn't be indexed right now. Please try again shortly.",
                                     503, 60) from e
        finally:
            self._busy.release()
        upload_id = secrets.token_urlsafe(16)
        upload = Upload(upload_id, visitor, name, pages, sum(len(s.text) for s in doc.sections),
                        UploadIndex(upload_id, chunks, vectors, self.embedder), self.clock())
        with self._lock:
            self._prune()
            # Make room: this visitor's oldest first, then the oldest of anyone's.
            mine = [u.upload_id for u in self._uploads.values() if u.visitor == visitor]
            for old in mine[: max(0, len(mine) - (p.max_live_per_visitor - 1))]:
                del self._uploads[old]
            while len(self._uploads) >= p.max_live:
                del self._uploads[next(iter(self._uploads))]
            self._uploads[upload_id] = upload
        self._limiter.record(visitor)
        # The name is not logged: a file name can be personal ("Jane Doe CV.pdf").
        log.info("upload indexed: %s pages, %d chunks, %d characters, %.1f s", pages if pages is not None else "-",
                 len(chunks), upload.characters, time.perf_counter() - started)
        return upload

    def _read(self, name: str, data: bytes) -> tuple[Document, list[Chunk], int | None]:
        p = self.policy
        pages = None
        if name.endswith(".pdf"):
            try:
                pages, _ = pdf_info(data)  # cheap: no page is read yet
            except Exception as e:
                log.info("upload is not a readable PDF (%s)", type(e).__name__)
                raise UploadRejected("This file couldn't be read as a PDF. It may be damaged or "
                                     "password-protected.") from e
            if pages > p.max_pages:
                raise UploadRejected(f"This PDF has {pages} pages; the limit is {p.max_pages}.")
        try:
            doc = load_document(name, data)
        except NoTextError as e:
            raise UploadRejected(NO_TEXT) from e
        except Exception as e:
            log.info("parsing an upload failed (%s: %s)", type(e).__name__, str(e)[:120])
            raise UploadRejected("This file couldn't be read. It may be damaged.") from e
        characters = sum(len(s.text) for s in doc.sections)
        if characters < MIN_TEXT_CHARS:
            raise UploadRejected(f"Only {characters} characters of text could be read from this file, too little "
                                 "to answer questions from. If it is a scanned PDF, its pages are images, and "
                                 "HyRAG reads text only.")
        chunks = self.chunker(doc)
        if len(chunks) > p.max_chunks:
            raise UploadRejected(f"This document is too long: it splits into {len(chunks)} passages and the "
                                 f"limit is {p.max_chunks}.")
        return doc, chunks, pages

    def get(self, upload_id: str) -> Upload | None:
        with self._lock:
            self._prune()
            return self._uploads.get(upload_id)

    def remove(self, upload_id: str) -> bool:
        with self._lock:
            return self._uploads.pop(upload_id, None) is not None

    def seconds_left(self, upload: Upload) -> int:
        return max(0, int(upload.created + self.policy.ttl_seconds - self.clock()))

    def count(self) -> int:
        with self._lock:
            self._prune()
            return len(self._uploads)

    def _prune(self) -> None:
        cutoff = self.clock() - self.policy.ttl_seconds
        for upload_id in [i for i, u in self._uploads.items() if u.created <= cutoff]:
            del self._uploads[upload_id]
