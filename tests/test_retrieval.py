import pytest

from hyrag.chunking import Chunk
from hyrag.index import ChunkIndex, Hit
from hyrag.retrieval import HybridRetriever, RetrievalConfig, reciprocal_rank_fusion

from .test_index import CountingEmbedder, vpn_doc


def chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", source="d.md", text=cid, chunk_index=0, heading="", page=None,
                 strategy="structure", char_count=len(cid))


def hits(*ids: str) -> list[Hit]:
    return [Hit(chunk(i), score=1.0) for i in ids]


def test_rrf_formula_and_agreement_wins():
    fused = reciprocal_rank_fusion({"dense": (hits("a", "b", "c"), 1.0), "sparse": (hits("c", "d"), 1.0)}, rrf_k=60)
    assert fused[0].chunk.chunk_id == "c"  # in both lists: 1/63 + 1/61 beats "a" at 1/61 alone
    assert fused[0].score == pytest.approx(1 / 63 + 1 / 61)
    assert (fused[0].dense_rank, fused[0].sparse_rank) == (3, 1)
    assert {h.chunk.chunk_id: (h.dense_rank, h.sparse_rank) for h in fused}["d"] == (None, 2)


def test_weights_scale_each_lists_contribution():
    fused = reciprocal_rank_fusion({"dense": (hits("a"), 0.7), "sparse": (hits("b"), 0.3)}, rrf_k=60)
    assert [h.chunk.chunk_id for h in fused] == ["a", "b"]
    assert fused[0].contributions == {"dense": pytest.approx(0.7 / 61)}


def test_rank_swap_ties_under_equal_weights_and_dense_wins_at_07_03():
    # The ERR_TUNNEL_4012 situation: each list puts a different one of two chunks first.
    lists = lambda wd, ws: {"dense": (hits("4013", "4012"), wd), "sparse": (hits("4012", "4013"), ws)}
    equal = reciprocal_rank_fusion(lists(0.5, 0.5))
    assert equal[0].score == pytest.approx(equal[1].score)  # RRF sees positions only: a perfect tie
    assert reciprocal_rank_fusion(lists(0.7, 0.3))[0].chunk.chunk_id == "4013"  # the gap BM25 saw is gone


def test_ties_break_deterministically():
    once = reciprocal_rank_fusion({"dense": (hits("x", "y"), 1.0), "sparse": (hits("y", "x"), 1.0)})
    again = reciprocal_rank_fusion({"sparse": (hits("y", "x"), 1.0), "dense": (hits("x", "y"), 1.0)})
    assert [h.chunk.chunk_id for h in once] == [h.chunk.chunk_id for h in again]


@pytest.fixture
def retriever(tmp_path):
    index = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i")
    index.index_document(*vpn_doc())
    return HybridRetriever(index, RetrievalConfig(top_k=3))


def test_modes(retriever):
    hybrid = retriever.retrieve("ERR_TUNNEL_4012 certificate renew")
    assert len(hybrid) <= 3 and any(h.dense_rank and h.sparse_rank for h in hybrid)
    dense = retriever.retrieve("ERR_TUNNEL_4012 certificate renew", mode="dense")
    assert all(h.sparse_rank is None and h.dense_rank for h in dense)
    sparse = retriever.retrieve("ERR_TUNNEL_4012", mode="sparse")
    assert sparse[0].chunk.heading.endswith("ERR_TUNNEL_4012") and all(h.dense_rank is None for h in sparse)


def test_empty_index_returns_nothing(tmp_path):
    empty = HybridRetriever(ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "e"))
    assert empty.retrieve("anything") == []


# ----- audit fixes (docs/phase2-audit.md) -----

@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_is_rejected_without_searching(retriever, query):
    before = len(retriever.index.embedder.texts)
    with pytest.raises(ValueError, match="empty query"):
        retriever.retrieve(query)
    assert retriever.index.search_dense(query) == []  # the index itself doesn't embed a blank query either
    assert len(retriever.index.embedder.texts) == before


def test_unknown_mode_is_rejected(retriever):
    with pytest.raises(ValueError, match="unknown mode"):
        retriever.retrieve("vpn", mode="Hybrid")


@pytest.mark.parametrize("bad", [dict(dense_k=0), dict(sparse_k=-1), dict(top_k=0), dict(dense_weight=-0.1),
                                 dict(dense_weight=0, sparse_weight=0), dict(rrf_k=-1)])
def test_config_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        RetrievalConfig(**bad)


def test_zero_weight_list_is_not_searched(tmp_path):
    index = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i")
    index.index_document(*vpn_doc())
    before = len(index.embedder.texts)
    hits = HybridRetriever(index, RetrievalConfig(dense_weight=0, sparse_weight=1)).retrieve("ERR_TUNNEL_4012")
    assert hits and len(index.embedder.texts) == before and all(h.dense_rank is None for h in hits)


def test_fusion_counts_a_chunk_once_per_list():
    fused = reciprocal_rank_fusion({"dense": (hits("a", "b", "a"), 1.0)})
    a = next(h for h in fused if h.chunk.chunk_id == "a")
    assert a.dense_rank == 1 and a.score == pytest.approx(1 / 61) == pytest.approx(sum(a.contributions.values()))


def test_fusion_rejects_unknown_list_names():
    with pytest.raises(ValueError, match="unknown list"):
        reciprocal_rank_fusion({"bm25": (hits("a"), 1.0)})


def test_empty_index_warns(tmp_path, caplog):
    empty = HybridRetriever(ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "e"))
    assert empty.retrieve("anything") == []
    assert "empty index" in caplog.text
