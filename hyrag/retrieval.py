"""Hybrid retrieval: run meaning search and keyword search, then merge the two ranked lists with
Reciprocal Rank Fusion (RRF).

Why ranks and not scores: the two searches score on incompatible scales (cosine similarity 0..1 vs
unbounded BM25 points), so adding scores is meaningless. RRF uses only each chunk's POSITION in each list:

    fused(chunk) = sum over lists of  weight / (rrf_k + rank)          (rank starts at 1)

Agreement between the lists wins; a chunk found by only one list still counts. rrf_k = 60 comes from the
original paper (Cormack, Clarke & Buettcher, SIGIR 2009) and damps the top ranks: 1/61 vs 1/62 is a small
gap, so no single list's #1 dominates.

Known limit, measured on our corpus: because only positions count, the SIZE of a gap is thrown away. For
"How do I fix ERR_TUNNEL_4012?" BM25 prefers 4012 decisively (37.3 vs 17.1 points) while dense search puts
4013 first by a hair; RRF sees only "each list has one of them first". That is the reranker's job.
"""
import logging
from dataclasses import dataclass, field
from typing import Literal, get_args

from hyrag.chunking import Chunk
from hyrag.index import ChunkIndex, Hit

log = logging.getLogger(__name__)

Mode = Literal["hybrid", "dense", "sparse"]
LISTS = ("dense", "sparse")  # the ranked lists fusion knows how to record on a FusedHit


@dataclass(frozen=True)
class RetrievalConfig:
    dense_k: int = 10          # candidates taken from meaning search (brief: start with 10)
    sparse_k: int = 10         # candidates taken from keyword search
    dense_weight: float = 0.7  # brief's suggested starting point; tune on the evaluation set, not by hand
    sparse_weight: float = 0.3
    rrf_k: int = 60
    top_k: int = 20            # how many fused results to return (the reranker's input size)

    def __post_init__(self):
        # Fail when the config is built, not deep inside Chroma or as a silently inverted ranking.
        for name in ("dense_k", "sparse_k", "top_k"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("weights can't be negative: a negative weight turns a list's best hit into its worst")
        if self.dense_weight + self.sparse_weight == 0:
            raise ValueError("at least one weight must be above 0")
        if self.rrf_k < 0:
            raise ValueError(f"rrf_k can't be negative (rank 1 with rrf_k=-1 divides by zero), got {self.rrf_k}")


@dataclass
class FusedHit:
    chunk: Chunk
    score: float                       # fused RRF score (only meaningful for ordering)
    dense_rank: int | None = None      # 1-based position in meaning search, None if not retrieved there
    sparse_rank: int | None = None
    dense_score: float | None = None   # cosine similarity
    sparse_score: float | None = None  # BM25 points
    contributions: dict[str, float] = field(default_factory=dict)  # how much each list added to `score`


def reciprocal_rank_fusion(ranked: dict[str, tuple[list[Hit], float]], rrf_k: int = 60) -> list[FusedHit]:
    """Merge ranked lists. `ranked` maps a list name ("dense", "sparse") to (hits in rank order, weight)."""
    unknown = set(ranked) - set(LISTS)
    if unknown:
        raise ValueError(f"unknown list name(s) {sorted(unknown)}; expected {LISTS}")
    fused: dict[str, FusedHit] = {}
    for name, (hits, weight) in ranked.items():
        for rank, hit in enumerate(hits, start=1):
            entry = fused.setdefault(hit.chunk.chunk_id, FusedHit(chunk=hit.chunk, score=0.0))
            if name in entry.contributions:  # listed twice in one list: only its best (first) rank counts
                continue
            contribution = weight / (rrf_k + rank)
            entry.score += contribution
            entry.contributions[name] = contribution
            setattr(entry, f"{name}_rank", rank)
            setattr(entry, f"{name}_score", hit.score)
    # Ties (e.g. two chunks swapped between the lists under equal weights) break by chunk id: deterministic.
    return sorted(fused.values(), key=lambda h: (-h.score, h.chunk.chunk_id))


class HybridRetriever:
    def __init__(self, index: ChunkIndex, config: RetrievalConfig = RetrievalConfig()):
        self.index = index
        self.config = config

    def retrieve(self, query: str, mode: Mode = "hybrid") -> list[FusedHit]:
        """Ranked candidates for `query`. mode="dense" / "sparse" run one search alone (same output shape),
        which is what the dashboard's hybrid-vs-dense comparison needs."""
        if mode not in get_args(Mode):
            raise ValueError(f"unknown mode {mode!r}; expected one of {get_args(Mode)}")
        if not query.strip():
            raise ValueError("empty query")  # meaning search would still return its 'nearest' chunks
        if self.index.count() == 0:
            log.warning("retrieving from an empty index (strategy %r): nothing has been indexed yet",
                        self.index.strategy)
            return []
        c = self.config
        dense_w = c.dense_weight if mode == "hybrid" else float(mode == "dense")
        sparse_w = c.sparse_weight if mode == "hybrid" else float(mode == "sparse")
        lists: dict[str, tuple[list[Hit], float]] = {}
        if dense_w > 0:  # a zero-weight list can't change the order: don't pay for its search (or API call)
            lists["dense"] = (self.index.search_dense(query, k=c.dense_k), dense_w)
        if sparse_w > 0:
            lists["sparse"] = (self.index.search_sparse(query, k=c.sparse_k), sparse_w)
        return reciprocal_rank_fusion(lists, rrf_k=c.rrf_k)[: c.top_k]
