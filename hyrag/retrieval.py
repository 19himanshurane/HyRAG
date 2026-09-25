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
4013 first by a hair; RRF sees only "each list has one of them first". That is the reranker's job:
with a reranker set, the fused top `top_k` are re-scored by a cross-encoder (hyrag/rerank.py) and the best
`rerank_top_n` returned.
"""
import logging
from dataclasses import dataclass, field
from typing import Literal, Protocol, get_args

import numpy as np

from hyrag.chunking import Chunk
from hyrag.index import ChunkIndex, Hit

log = logging.getLogger(__name__)

Mode = Literal["hybrid", "dense", "sparse"]
LISTS = ("dense", "sparse")  # the ranked lists fusion knows how to record on a FusedHit
# A question, not a document. Longer input (a pasted log file) got a 400 from Mistral AFTER a paid request, and
# the reranker would cut it to its 512-token window anyway. ~2,000 chars = 300-500 words.
MAX_QUERY_CHARS = 2000


class Scorer(Protocol):
    """Anything that scores texts against a query, higher = more relevant (hyrag.rerank.CrossEncoderScorer)."""

    def score(self, query: str, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class RetrievalConfig:
    dense_k: int = 10          # candidates taken from meaning search (brief: start with 10)
    sparse_k: int = 10         # candidates taken from keyword search
    dense_weight: float = 0.7  # brief's suggested starting point; tune on the evaluation set, not by hand
    sparse_weight: float = 0.3
    rrf_k: int = 60
    top_k: int = 20            # how many fused results to return (the reranker's input size)
    rerank_top_n: int = 5      # how many the reranker keeps (what the LLM will see in Phase 3)

    def __post_init__(self):
        # Fail when the config is built, not deep inside Chroma or as a silently inverted ranking.
        for name in ("dense_k", "sparse_k", "top_k", "rerank_top_n"):
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
    fused_rank: int | None = None      # 1-based position after fusion (before any reranking)
    rerank_score: float | None = None  # cross-encoder relevance, set only when reranked


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
    ordered = sorted(fused.values(), key=lambda h: (-h.score, h.chunk.chunk_id))
    for position, hit in enumerate(ordered, start=1):
        hit.fused_rank = position
    return ordered


def rerank(query: str, hits: list[FusedHit], scorer: Scorer, top_n: int) -> list[FusedHit]:
    """Re-order `hits` by the scorer's relevance and keep the best `top_n`. The scorer reads the same text
    both searches indexed (heading path + body): the heading often holds what the question names."""
    if not hits:
        return []
    scores = scorer.score(query, [h.chunk.text_for_search() for h in hits])
    if len(scores) != len(hits):
        raise ValueError(f"scorer returned {len(scores)} scores for {len(hits)} texts")
    for hit, s in zip(hits, scores):
        hit.rerank_score = float(s)
    # Equal relevance keeps the fused order (stable sort), so ties are deterministic.
    return sorted(hits, key=lambda h: -h.rerank_score)[:top_n]


class HybridRetriever:
    def __init__(self, index: ChunkIndex, config: RetrievalConfig = RetrievalConfig(),
                 reranker: Scorer | None = None):
        # Checked here, not in RetrievalConfig: without a reranker, rerank_top_n is unused and a small top_k is fine.
        if reranker is not None and config.rerank_top_n > config.top_k:
            raise ValueError(f"rerank_top_n ({config.rerank_top_n}) can't exceed top_k ({config.top_k}): "
                             f"the reranker only sees top_k candidates")
        self.index = index
        self.config = config
        self.reranker = reranker

    def retrieve(self, query: str, mode: Mode = "hybrid", rerank_results: bool = True) -> list[FusedHit]:
        """Ranked candidates for `query`. mode="dense" / "sparse" run one search alone (same output shape),
        which is what the dashboard's hybrid-vs-dense comparison needs. With a reranker (and
        rerank_results=True) the fused top_k are re-scored and the best rerank_top_n returned."""
        if mode not in get_args(Mode):
            raise ValueError(f"unknown mode {mode!r}; expected one of {get_args(Mode)}")
        if not query.strip():
            raise ValueError("empty query")  # meaning search would still return its 'nearest' chunks
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query is {len(query):,} characters; the limit is {MAX_QUERY_CHARS:,}")
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
        candidates = reciprocal_rank_fusion(lists, rrf_k=c.rrf_k)[: c.top_k]
        if self.reranker is None or not rerank_results:
            return candidates
        try:
            return rerank(query, candidates, self.reranker, c.rerank_top_n)
        except Exception:
            # The fused list is a good answer on its own: a broken reranker shouldn't take search down.
            # rerank_score stays None on every hit, so callers (Phase 3 confidence) can see it wasn't reranked.
            log.exception("reranker failed; returning the fused top %d instead", c.rerank_top_n)
            for hit in candidates:
                hit.rerank_score = None
            return candidates[: c.rerank_top_n]
