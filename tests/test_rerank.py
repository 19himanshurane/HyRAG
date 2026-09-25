import numpy as np
import pytest

from hyrag.index import ChunkIndex
from hyrag.retrieval import FusedHit, HybridRetriever, RetrievalConfig, rerank

from .test_index import CountingEmbedder, vpn_doc
from .test_retrieval import chunk


class KeywordScorer:
    """Fake cross-encoder: +1 per query word found in the text. Records what it was asked to score."""

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, texts):
        self.calls.append((query, texts))
        words = query.lower().replace("?", "").split()
        return np.array([sum(w in t.lower() for w in words) for t in texts], dtype=np.float32)


def fused(*ids: str) -> list[FusedHit]:
    return [FusedHit(chunk(i), score=1.0 / n, fused_rank=n) for n, i in enumerate(ids, 1)]


def test_rerank_reorders_by_score_and_keeps_top_n():
    out = rerank("beta gamma", fused("alpha", "beta", "beta gamma"), KeywordScorer(), top_n=2)
    assert [h.chunk.chunk_id for h in out] == ["beta gamma", "beta"]
    assert [h.fused_rank for h in out] == [3, 2]  # where each came from stays visible
    assert [h.rerank_score for h in out] == [2.0, 1.0]


def test_rerank_ties_keep_fused_order():
    out = rerank("zzz", fused("c", "a", "b"), KeywordScorer(), top_n=3)
    assert [h.chunk.chunk_id for h in out] == ["c", "a", "b"]


def test_rerank_scores_the_heading_too():
    hit = FusedHit(chunk("body"), score=1.0)
    hit.chunk.heading = "Troubleshooting > Error ERR_TUNNEL_4012"
    scorer = KeywordScorer()
    rerank("ERR_TUNNEL_4012", [hit], scorer, top_n=1)
    assert scorer.calls[0][1] == ["Troubleshooting > Error ERR_TUNNEL_4012\n\nbody"]


def test_rerank_empty_and_bad_scorer():
    assert rerank("q", [], KeywordScorer(), top_n=5) == []

    class Broken:
        def score(self, query, texts):
            return np.zeros(len(texts) - 1)

    with pytest.raises(ValueError, match="scores for"):
        rerank("q", fused("a", "b"), Broken(), top_n=5)


def test_fusion_records_fused_rank(tmp_path):
    index = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i")
    index.index_document(*vpn_doc())
    hits = HybridRetriever(index).retrieve("ERR_TUNNEL_4012 certificate")
    assert [h.fused_rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.rerank_score is None for h in hits)  # no reranker: nothing pretends to be reranked


def test_retriever_reranks_fused_candidates(tmp_path):
    index = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i")
    index.index_document(*vpn_doc())
    scorer = KeywordScorer()
    r = HybridRetriever(index, RetrievalConfig(top_k=4, rerank_top_n=2), reranker=scorer)
    out = r.retrieve("ERR_TUNNEL_4012 certificate")
    assert len(out) == 2 and all(h.rerank_score is not None for h in out)
    assert len(scorer.calls[0][1]) <= 4  # only the fused top_k are scored
    assert out[0].rerank_score >= out[1].rerank_score
    assert len(r.retrieve("ERR_TUNNEL_4012 certificate", rerank_results=False)) > 2  # the comparison toggle
    assert len(scorer.calls) == 1


def test_rerank_top_n_is_validated():
    with pytest.raises(ValueError, match="rerank_top_n"):
        RetrievalConfig(rerank_top_n=0)


def _model_cached() -> bool:
    from hyrag.rerank import is_downloaded

    return is_downloaded()


@pytest.mark.model
@pytest.mark.skipif(not _model_cached(), reason="reranker model not downloaded (run try_rerank.py once)")
def test_real_cross_encoder_tells_4012_from_4013():
    from hyrag.rerank import CrossEncoderScorer

    scorer = CrossEncoderScorer(local_files_only=True)  # offline: fails instead of downloading
    s = scorer.score("How do I fix ERR_TUNNEL_4012?", [
        "Troubleshooting > Error ERR_TUNNEL_4013\n\nThe VPN gateway is unreachable. Check your network.",
        "Troubleshooting > Error ERR_TUNNEL_4012\n\nThe client certificate has expired. Renew it in the portal.",
        "Deployment Guide > Production\n\nNo deploys on Fridays.",
    ])
    assert s.shape == (3,) and s[1] > s[0] > s[2]
    assert scorer.score("q", []).shape == (0,)


# ----- audit round 2 fixes (docs/phase2-audit.md) -----

class RaisingScorer:
    def score(self, query, texts):
        raise RuntimeError("model file unreadable")


def test_failing_reranker_falls_back_to_fused_order(tmp_path, caplog):
    index = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i")
    index.index_document(*vpn_doc())
    r = HybridRetriever(index, RetrievalConfig(top_k=4, rerank_top_n=2), reranker=RaisingScorer())
    out = r.retrieve("ERR_TUNNEL_4012 certificate")
    assert [h.fused_rank for h in out] == [1, 2]
    assert all(h.rerank_score is None for h in out)  # visibly NOT reranked
    assert "reranker failed" in caplog.text and "model file unreadable" in caplog.text


def test_failed_load_is_remembered_then_retried(monkeypatch):
    import hyrag.rerank as rr

    attempts = []

    class Unloadable(rr.CrossEncoderScorer):
        def _load(self):
            attempts.append(1)
            raise OSError("not in the local cache")

    now = [1000.0]
    monkeypatch.setattr(rr.time, "monotonic", lambda: now[0])
    s = Unloadable()
    for _ in range(3):
        with pytest.raises(rr.RerankerUnavailable, match="python -m hyrag.rerank|failed to load recently"):
            s.score("q", ["t"])
    assert len(attempts) == 1  # the two later requests failed fast instead of re-loading
    now[0] += rr.RETRY_LOAD_AFTER_SECONDS + 1
    with pytest.raises(rr.RerankerUnavailable):
        s.score("q", ["t"])
    assert len(attempts) == 2  # retried once the cool-down passed


def test_scorer_never_downloads_by_default():
    from hyrag.rerank import CrossEncoderScorer

    assert CrossEncoderScorer().local_files_only is True


def test_overlong_query_is_rejected_before_any_search(tmp_path):
    from hyrag.retrieval import MAX_QUERY_CHARS

    index = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i")
    index.index_document(*vpn_doc())
    before = len(index.embedder.texts)
    with pytest.raises(ValueError, match="limit"):
        HybridRetriever(index).retrieve("x" * (MAX_QUERY_CHARS + 1))
    assert len(index.embedder.texts) == before
    assert HybridRetriever(index).retrieve("ERR_TUNNEL_4012 " + "x" * (MAX_QUERY_CHARS - 16))  # at the limit: fine


def test_rerank_top_n_cannot_exceed_top_k_when_reranking(tmp_path):
    index = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i")
    with pytest.raises(ValueError, match="can't exceed top_k"):
        HybridRetriever(index, RetrievalConfig(top_k=3, rerank_top_n=10), reranker=KeywordScorer())
    HybridRetriever(index, RetrievalConfig(top_k=3))  # no reranker: rerank_top_n is unused, so it's allowed
