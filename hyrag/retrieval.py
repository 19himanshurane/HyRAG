"""Hybrid retrieval: meaning search and keyword search, merged with Reciprocal Rank Fusion, then reranked.

RRF adds weight / (rrf_k + rank) over the two lists. It uses ranks because cosine similarity and BM25 points
aren't on comparable scales. rrf_k = 60 is the value from the original paper (Cormack et al., SIGIR 2009).

Ranks throw away how big a lead is. For "How do I fix ERR_TUNNEL_4012?" BM25 prefers 4012 by 37.3 to 17.1
while meaning search puts 4013 first by a hair, and RRF only sees one first place each. The cross-encoder
(rerank.py) re-scores the fused top_k to settle cases like that.
"""
import logging
from dataclasses import dataclass, field
from typing import Literal, Protocol, get_args

import numpy as np

from hyrag.chunking import Chunk
from hyrag.index import ChunkIndex, Hit, check_collection

log = logging.getLogger(__name__)

Mode = Literal["hybrid", "dense", "sparse"]
LISTS = ("dense", "sparse")  # the ranked lists fusion knows how to record on a FusedHit
# Longer input (a pasted log file) got a 400 from Mistral after being billed, and the reranker only reads
# 512 tokens anyway.
MAX_QUERY_CHARS = 2000


class Scorer(Protocol):
    """Anything that scores texts against a query, higher = more relevant (hyrag.rerank.CrossEncoderScorer)."""

    def score(self, query: str, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class RetrievalConfig:
    dense_k: int = 10          # candidates from meaning search
    sparse_k: int = 10         # candidates from keyword search
    dense_weight: float = 0.7  # tune these on the evaluation set, not by hand
    sparse_weight: float = 0.3
    rrf_k: int = 60
    top_k: int = 20            # fused results passed to the reranker
    rerank_top_n: int = 5      # what the writer gets to see
    # A question whose best rerank logit is below this gets rewritten into focused queries (rewrite.py). It's
    # the same threshold as the answer gate, so the questions that would be refused get a second try.
    rewrite_below_logit: float = -3.0

    def __post_init__(self):
        # Fail here rather than deep inside Chroma, or with a silently inverted ranking.
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
                 reranker: Scorer | None = None, rewriter=None):
        # Only matters with a reranker; without one, rerank_top_n is unused.
        if reranker is not None and config.rerank_top_n > config.top_k:
            raise ValueError(f"rerank_top_n ({config.rerank_top_n}) can't exceed top_k ({config.top_k}): "
                             f"the reranker only sees top_k candidates")
        self.index = index
        self.config = config
        self.reranker = reranker
        self.rewriter = rewriter  # a chat model (hyrag.llm.GroqChat) for query rewriting, or None

    def retrieve(self, query: str, mode: Mode = "hybrid", rerank_results: bool = True,
                 collection: str | None = None) -> list[FusedHit]:
        """Ranked candidates for `query`. mode "dense" or "sparse" runs one search alone, for the dashboard's
        comparison. collection limits the search to one company's documents (None searches all)."""
        return self.search(query, mode, rerank_results, collection).hits

    def search(self, query: str, mode: Mode = "hybrid", rerank_results: bool = True,
               collection: str | None = None) -> "Retrieval":
        """Like retrieve(), plus what had to be skipped (`degraded`) and any rewritten queries (`queries`)."""
        check_collection(collection)  # before any paid call
        first = self._search_one(query, mode, rerank_results, collection)
        c = self.config
        top = first.hits[0].rerank_score if first.hits else None
        if (self.rewriter is None or mode != "hybrid" or not rerank_results or top is None
                or top >= c.rewrite_below_logit):
            return first
        from hyrag.rewrite import rewrite_query

        usage: dict[str, int] = {}
        queries = rewrite_query(query, self.rewriter, usage)
        if not queries:
            return Retrieval(first.hits, first.degraded + ["query_rewrite_failed"], usage=usage)
        results = [self._search_one(q[:MAX_QUERY_CHARS], mode, rerank_results, collection) for q in queries]
        degraded = sorted(set(first.degraded).union(*(r.degraded for r in results)))
        return Retrieval(merge_rewritten([r.hits for r in results] + [first.hits], c.rerank_top_n), degraded, queries,
                         usage)

    def _search_one(self, query: str, mode: Mode = "hybrid", rerank_results: bool = True,
                    collection: str | None = None) -> "Retrieval":
        """One search: dense + sparse, fused, reranked.

        Only meaning search needs the network (Mistral embeds the query), so if that fails in hybrid mode we
        carry on with keyword search and the reranker instead of failing the question."""
        if mode not in get_args(Mode):
            raise ValueError(f"unknown mode {mode!r}; expected one of {get_args(Mode)}")
        if not query.strip():
            raise ValueError("empty query")  # meaning search would still return its 'nearest' chunks
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query is {len(query):,} characters; the limit is {MAX_QUERY_CHARS:,}")
        if self.index.count() == 0:
            log.warning("retrieving from an empty index (strategy %r): nothing has been indexed yet",
                        self.index.strategy)
            return Retrieval([], ["empty_index"])
        c = self.config
        degraded: list[str] = []
        dense_w = c.dense_weight if mode == "hybrid" else float(mode == "dense")
        sparse_w = c.sparse_weight if mode == "hybrid" else float(mode == "sparse")
        lists: dict[str, tuple[list[Hit], float]] = {}
        if dense_w > 0:  # a zero-weight list can't change the order, so skip the API call
            try:
                lists["dense"] = (self.index.search_dense(query, k=c.dense_k, collection=collection), dense_w)
            except Exception as e:
                if mode == "dense":  # nothing to fall back to
                    raise
                log.warning("meaning search unavailable (%s: %s); using keyword search only",
                            type(e).__name__, str(e)[:120])
                degraded.append("dense_search_unavailable")
        if sparse_w > 0:
            lists["sparse"] = (self.index.search_sparse(query, k=c.sparse_k, collection=collection), sparse_w)
        candidates = reciprocal_rank_fusion(lists, rrf_k=c.rrf_k)[: c.top_k]
        if self.reranker is None or not rerank_results:
            return Retrieval(candidates, degraded)
        try:
            return Retrieval(rerank(query, candidates, self.reranker, c.rerank_top_n), degraded)
        except Exception:
            # The fused list is usable on its own, so a broken reranker shouldn't take search down.
            # rerank_score stays None, which tells the confidence score nothing was reranked.
            log.exception("reranker failed; returning the fused top %d instead", c.rerank_top_n)
            for hit in candidates:
                hit.rerank_score = None
            return Retrieval(candidates[: c.rerank_top_n], degraded + ["rerank_unavailable"])


@dataclass
class Retrieval:
    hits: list[FusedHit]
    degraded: list[str] = field(default_factory=list)  # parts skipped: dense_search_unavailable, rerank_unavailable
    queries: list[str] = field(default_factory=list)   # the rewritten search queries, when the question was rewritten
    usage: dict[str, int] = field(default_factory=dict)  # rewrite_requests / rewrite_tokens


def merge_rewritten(ranked_lists: list[list[FusedHit]], top_n: int) -> list[FusedHit]:
    """Merge several queries' results into one top_n.

    Each query's best hit gets a place first, since a multi-part question needs evidence for every part (one
    query alone found either the deploy rules or the VPN rule, never both). The rest follow by reranker score.
    A chunk found twice keeps its best score."""
    best: dict[str, FusedHit] = {}
    for hits in ranked_lists:
        for h in hits:
            kept = best.get(h.chunk.chunk_id)
            if kept is None or (h.rerank_score or float("-inf")) > (kept.rerank_score or float("-inf")):
                best[h.chunk.chunk_id] = h
    score = lambda h: h.rerank_score if h.rerank_score is not None else float("-inf")
    leaders = {hits[0].chunk.chunk_id for hits in ranked_lists[:-1] if hits}  # each rewritten query's best
    first = sorted((best[cid] for cid in leaders), key=score, reverse=True)
    rest = sorted((h for cid, h in best.items() if cid not in leaders), key=score, reverse=True)
    return (first + rest)[:top_n]
