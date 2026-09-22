import json

import httpx
import numpy as np
import pytest

from hyrag import embeddings
from hyrag.embeddings import MistralEmbedder, cosine_similarity


def fake_mistral(responses: list, seen: list):
    """A MockTransport that returns queued (status, headers) or embeds each input as a deterministic vector."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if responses:
            status, headers = responses.pop(0)
            if status != 200:
                return httpx.Response(status, headers=headers, json={"error": "x"})
        data = [{"index": i, "embedding": [float(len(t)), 1.0, 0.0]} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": list(reversed(data))})  # out of order on purpose

    return httpx.MockTransport(handler)


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(embeddings.time, "sleep", waited.append)
    return waited


@pytest.fixture
def embedder(monkeypatch, sleeps):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(embeddings, "load_dotenv", lambda *a, **k: None)  # never read the real .env in tests

    def build(responses=None, seen=None, **kw):
        kw.setdefault("cache_path", None)
        e = MistralEmbedder(**kw)
        e.client = httpx.Client(transport=fake_mistral(responses or [], seen if seen is not None else []))
        return e

    return build


def test_batches_order_and_normalisation(embedder):
    seen = []
    e = embedder(seen=seen, batch_size=2)
    v = e.embed(["a", "bbb", "cc"])
    assert [len(b["input"]) for b in seen] == [2, 1]
    assert v.shape == (3, 3)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert v[1][0] > v[0][0]  # results were re-sorted by "index", not taken in response order


def test_retries_on_429_and_5xx_then_succeeds(embedder):
    seen = []
    e = embedder(responses=[(429, {}), (503, {})], seen=seen)
    assert e.embed(["x"]).shape == (1, 3)
    assert len(seen) == 3


def test_client_errors_fail_fast(embedder):
    seen = []
    e = embedder(responses=[(401, {})], seen=seen)
    with pytest.raises(httpx.HTTPStatusError):
        e.embed(["x"])
    assert len(seen) == 1


def test_retry_after_header_is_honoured_and_capped(embedder, sleeps):
    e = embedder(responses=[(429, {"Retry-After": "3"}), (429, {"Retry-After": "999"}), (503, {})])
    e.embed(["x"])
    assert sleeps == [3.0, 60.0, 4.0]  # server's value, capped at 60, then normal backoff (attempt 3 -> 2**2)


def test_empty_input_has_the_embedding_width(embedder):
    e = embedder()
    empty = e.embed([])
    assert empty.shape == (0, 1024)
    assert np.vstack([empty, np.ones((2, 1024), dtype=np.float32)]).shape == (2, 1024)


def test_cache_only_pays_for_new_text(embedder, tmp_path):
    seen = []
    e = embedder(seen=seen, cache_path=tmp_path / "cache.sqlite")
    first = e.embed(["a", "bb", "a"])                      # "a" twice in one call: sent once
    assert [b["input"] for b in seen] == [["a", "bb"]]
    again = e.embed(["bb", "ccc", "a"])                    # only "ccc" is new
    assert [b["input"] for b in seen] == [["a", "bb"], ["ccc"]]
    assert np.allclose(again[0], first[1]) and np.allclose(again[2], first[0])
    e.close()

    seen2 = []
    reopened = embedder(seen=seen2, cache_path=tmp_path / "cache.sqlite")  # survives a restart
    assert np.allclose(reopened.embed(["a", "bb", "ccc"]), np.vstack([first[0], first[1], again[1]]))
    assert seen2 == []


def test_cache_is_per_model(embedder, tmp_path):
    seen = []
    embedder(seen=seen, cache_path=tmp_path / "c.sqlite").embed(["a"])
    embedder(seen=seen, cache_path=tmp_path / "c.sqlite", model="other-model").embed(["a"])
    assert len(seen) == 2


def test_embedder_closes_its_connection(embedder):
    with embedder() as e:
        pass
    assert e.client.is_closed


def test_importing_the_module_does_not_read_env_files():
    import importlib
    import inspect
    src = inspect.getsource(importlib.import_module("hyrag.embeddings"))
    module_level = [l for l in src.splitlines() if l.startswith("load_dotenv(")]
    assert module_level == []


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr(embeddings, "load_dotenv", lambda *a, **k: None, raising=False)
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        MistralEmbedder()


def test_cosine_similarity():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 2.0])) == pytest.approx(0.0)
