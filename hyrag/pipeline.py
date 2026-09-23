"""The single way documents enter and leave HyRAG: store -> chunk -> index, and deletions reach both.

Before this module, ingestion (DocumentStore) and search (ChunkIndex) were two islands: a document deleted
from the store stayed searchable, so answers could cite a withdrawn policy (audit H1). Everything that adds,
changes or removes a document should go through a Pipeline.
"""
import logging
from collections.abc import Callable
from pathlib import Path

from hyrag.chunking import Chunk, chunk_structure
from hyrag.index import ChunkIndex
from hyrag.loader import Document, DocumentStore, make_doc_id

log = logging.getLogger(__name__)

Chunker = Callable[[Document], list[Chunk]]


class Pipeline:
    def __init__(self, store: DocumentStore, index: ChunkIndex, chunker: Chunker = chunk_structure):
        self.store = store
        self.index = index
        self.chunker = chunker

    def ingest(self, paths: list[Path], *, root: Path, workers: int | None = None) -> list[Document]:
        """Parse (in parallel), store, chunk and index. Files that fail are in `self.store.failures`.
        Windows callers need an `if __name__ == "__main__":` guard (ingest_many uses worker processes)."""
        docs = self.store.ingest_many(paths, root=root, workers=workers)
        for doc in docs:
            self._index(doc)
        return docs

    def delete(self, source: str) -> None:
        """Remove a document from the store AND the index, so it can never be retrieved or cited again."""
        self.store.delete(source)
        removed = self.index.remove_document(make_doc_id(source))
        log.info("deleted %s: %d chunks removed from the index", source, removed)

    def reprocess(self, force: bool = False) -> list[Document]:
        """Re-parse everything the store holds (e.g. after a parser upgrade), re-index what changed, and drop
        documents the store no longer has from the index. A document whose re-parse FAILED keeps its index
        entries: the store keeps its last good version too, and deleting it would lose working answers."""
        docs = self.store.reprocess_all(force=force)
        for doc in docs:
            self._index(doc)
        keep = {d.doc_id for d in docs} | {make_doc_id(s) for s in self.store.failures}
        for stale in sorted(self.index.doc_ids() - keep):
            removed = self.index.remove_document(stale)
            log.info("removed document %s from the index (no longer in the store): %d chunks", stale, removed)
        return docs

    def _index(self, doc: Document) -> None:
        chunks = self.chunker(doc)
        wrong = {c.strategy for c in chunks} - {self.index.strategy}
        if wrong:  # e.g. semantic chunks written into the structure index would mix two strategies' results
            raise ValueError(f"chunker produced {sorted(wrong)} chunks for the {self.index.strategy!r} index")
        self.index.index_document(doc, chunks)
