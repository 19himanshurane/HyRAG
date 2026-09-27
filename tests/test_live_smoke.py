"""Live smoke tests: the real Groq, Mistral and reranker, end to end. About 6 requests in total.

Run before a deploy:  pytest tests/test_live_smoke.py --live
Skipped in a plain `pytest` run. They prove the wiring works against the real services; the golden-set
evaluation (scripts/run_eval.py) measures answer QUALITY. Assertions are about behaviour that must hold on
every run (status, source, refusal), not about exact wording, which the models vary.
"""
import os
import uuid

import numpy as np
import pytest
from dotenv import load_dotenv

pytestmark = pytest.mark.live


def _need(condition: bool, why: str) -> None:
    if not condition:
        pytest.skip(why)


@pytest.fixture(scope="module")
def services():
    load_dotenv()
    _need(bool(os.environ.get("GROQ_API_KEY")), "GROQ_API_KEY is not set")
    _need(bool(os.environ.get("MISTRAL_API_KEY")), "MISTRAL_API_KEY is not set")
    from hyrag.citations import judge_client
    from hyrag.embeddings import MistralEmbedder
    from hyrag.index import ChunkIndex
    from hyrag.llm import GroqChat
    from hyrag.rerank import CrossEncoderScorer, is_downloaded
    from hyrag.retrieval import HybridRetriever

    _need(is_downloaded(), "reranker model not downloaded: python -m hyrag.rerank")
    emb, writer, judge = MistralEmbedder(), GroqChat(), judge_client()
    index = ChunkIndex(emb)
    _need(index.count() > 0, "data/index is empty: python try_index.py")
    retriever = HybridRetriever(index, reranker=CrossEncoderScorer())
    retriever.search("warm up the reranker")  # load the model once, outside any time budget
    yield {"emb": emb, "writer": writer, "judge": judge, "retriever": retriever}
    emb.close(), writer.close(), judge.close()


def test_writer_model_answers(services):
    r = services["writer"].complete([{"role": "user", "content": "Reply with the single word: ready"}])
    assert r.text.strip() and r.finish_reason in ("stop", "length") and r.prompt_tokens > 0


def test_judge_model_returns_schema_valid_json(services):
    import json
    schema = {"type": "object", "additionalProperties": False, "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    r = services["judge"].complete([{"role": "user", "content": "Is 2 greater than 1? Answer as JSON."}], json_schema=schema)
    assert json.loads(r.text)["ok"] is True


def test_embedding_service_returns_normalised_vectors(services):
    from hyrag.embeddings import MistralEmbedder
    with MistralEmbedder(cache_path=None) as fresh:  # no cache: this must reach the API
        v = fresh.embed([f"live smoke test {uuid.uuid4().hex}"])
        assert fresh.requests_made == 1
    assert v.shape == (1, 1024) and abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-3


def test_answerable_question_end_to_end(services):
    from hyrag.answer import ask
    r = ask("How do I fix ERR_TUNNEL_4012?", services["retriever"], services["writer"], services["judge"])
    assert r.status in ("answered", "partial"), (r.status, r.code, r.reason)
    assert any(s.source == "vpn-setup.md" for s in r.sources)
    assert "renew" in r.answer.lower()
    assert r.request_id and r.timings_ms["total"] > 0 and r.usage["writer_requests"] >= 1


def test_unanswerable_question_is_refused_without_calling_the_writer(services):
    from hyrag.answer import ask
    r = ask("How do I set up the office printer?", services["retriever"], services["writer"], services["judge"])
    assert (r.status, r.code) == ("not_found", "not_found.no_relevant_documents")
    assert r.usage["writer_requests"] == 0 and r.answer is None


def test_time_budget_holds_on_the_real_network_path(services):
    from hyrag.answer import ask
    r = ask("How do I fix ERR_TUNNEL_4012?", services["retriever"], services["writer"], services["judge"],
            budget_seconds=0.001)
    assert r.status == "error" and r.code in ("error.timeout", "error.retrieval_unavailable"), (r.code, r.reason)
    assert r.usage["writer_requests"] == 0  # the budget ran out before anything was sent
