import json

import numpy as np

from hyrag.index import ChunkIndex
from hyrag.llm import ChatResult
from hyrag.loader import load_document
from hyrag.chunking import chunk_structure
from hyrag.retrieval import HybridRetriever, merge_rewritten
from hyrag.rewrite import MAX_QUERIES, rewrite_query

from .test_generation import hit
from .test_index import CountingEmbedder, vpn_doc

DEPLOY = b"# Deployment Guide\n\n## Production\n\nProduction deploys require approval from one team lead.\n"


class WordReranker:
    """Fake cross-encoder: high logit when the query's words appear in the text, very low otherwise."""

    def score(self, query, texts):
        words = [w for w in query.lower().replace("?", "").split() if len(w) > 3]
        return np.array([8.0 if words and sum(w in t.lower() for w in words) >= max(1, len(words) // 2) else -8.0
                         for t in texts], dtype=np.float32)


class Rewriter:
    def __init__(self, queries=None, fail=False):
        self.queries, self.fail, self.calls = queries or [], fail, 0

    def complete(self, messages, json_schema=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model down")
        return ChatResult(json.dumps({"queries": self.queries}), "fake", "stop", 1, 1, 0, 0.0)


def index_with_both(tmp_path):
    index = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i")
    index.index_document(*vpn_doc())
    deploy = load_document("deploy-guide.md", DEPLOY)
    index.index_document(deploy, chunk_structure(deploy))
    return index


# ----- rewrite_query -----

def test_rewrite_query_cleans_dedupes_and_caps():
    queries = rewrite_query("q", Rewriter(["  production  deploy approval ", "Production deploy approval", "vpn rules",
                                          "a", "b"]))
    assert queries == ["production deploy approval", "vpn rules", "a"] and len(queries) <= MAX_QUERIES


def test_rewrite_failure_returns_nothing():
    assert rewrite_query("q", Rewriter(fail=True)) == []


# ----- merge -----

def test_merge_gives_each_query_its_best_hit_then_fills_by_score():
    a, b, c, d = hit("a"), hit("b"), hit("c"), hit("d")
    a.rerank_score, b.rerank_score, c.rerank_score, d.rerank_score = 9.0, 8.5, 2.0, 7.0
    # query 1 finds a, b; query 2's best is only c (2.0); the original found d
    merged = merge_rewritten([[a, b], [c], [d]], top_n=3)
    assert [h.chunk.text for h in merged] == ["a", "c", "b"]  # c kept for query 2 although d scores higher


def test_merge_keeps_a_chunk_once_with_its_best_score():
    x1, x2 = hit("x"), hit("x")
    x1.rerank_score, x2.rerank_score = 1.0, 6.0
    merged = merge_rewritten([[x1], [x2], []], top_n=5)
    assert len(merged) == 1 and merged[0].rerank_score == 6.0


# ----- search -----

def test_a_question_that_matches_well_is_not_rewritten(tmp_path):
    rewriter = Rewriter(["anything"])
    r = HybridRetriever(index_with_both(tmp_path), reranker=WordReranker(), rewriter=rewriter)
    result = r.search("ERR_TUNNEL_4012 certificate expired")
    assert rewriter.calls == 0 and result.queries == []


def test_a_low_scoring_question_is_rewritten_and_both_parts_are_found(tmp_path):
    rewriter = Rewriter(["production deploy approval", "VPN remote employees"])
    r = HybridRetriever(index_with_both(tmp_path), reranker=WordReranker(), rewriter=rewriter)
    result = r.search("I'm at home and want to ship something on Wednesday, what do I need?")
    assert rewriter.calls == 1 and result.queries == ["production deploy approval", "VPN remote employees"]
    texts = " ".join(h.chunk.text_for_search() for h in result.hits)
    assert "team lead" in texts and "VPN" in texts
    assert result.hits[0].rerank_score > 0  # the gate now sees a confident match


def test_a_failed_rewrite_keeps_the_original_results(tmp_path):
    r = HybridRetriever(index_with_both(tmp_path), reranker=WordReranker(), rewriter=Rewriter(fail=True))
    result = r.search("I'm at home and want to ship something on Wednesday, what do I need?")
    assert result.degraded == ["query_rewrite_failed"] and result.hits


def test_without_a_rewriter_nothing_changes(tmp_path):
    r = HybridRetriever(index_with_both(tmp_path), reranker=WordReranker())
    assert r.search("I'm at home and want to ship something on Wednesday, what do I need?").queries == []


def test_ask_reports_the_queries_it_used(tmp_path):
    from hyrag.answer import ask
    from .test_answer import FakeJudge
    from .test_generation import FakeChat

    r = HybridRetriever(index_with_both(tmp_path), reranker=WordReranker(),
                        rewriter=Rewriter(["production deploy approval"]))
    resp = ask("I'm at home and want to ship something on Wednesday, what do I need?", r,
               FakeChat("Get approval from one team lead [1]."), FakeJudge())
    assert resp.search_queries == ["production deploy approval"] and resp.status != "not_found"


def test_the_rewrite_request_is_counted_in_the_response_cost(tmp_path):
    from hyrag.answer import ask
    from .test_answer import FakeJudge
    from .test_generation import FakeChat

    r = HybridRetriever(index_with_both(tmp_path), reranker=WordReranker(),
                        rewriter=Rewriter(["production deploy approval"]))
    resp = ask("I'm at home and want to ship something on Wednesday, what do I need?", r,
               FakeChat("Get approval from one team lead [1]."), FakeJudge())
    assert resp.usage["rewrite_requests"] == 1 and resp.usage["rewrite_tokens"] == 2 and resp.usage["writer_requests"] == 1
