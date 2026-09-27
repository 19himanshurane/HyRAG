"""The HTTP API, end to end in-process: a real index, retriever and document store; fake models."""
import pytest
from fastapi.testclient import TestClient

from hyrag.api import Services, Settings, create_app
from hyrag.chunking import chunk_structure
from hyrag.http import DeadlineExceeded, QuotaExhausted
from hyrag.index import ChunkIndex
from hyrag.loader import DocumentStore, load_document
from hyrag.pipeline import Pipeline
from hyrag.retrieval import HybridRetriever

from .test_answer import FakeJudge, _Failing
from .test_generation import FakeChat
from .test_index import CountingEmbedder, vpn_doc
from .test_rewrite import DEPLOY, WordReranker


@pytest.fixture
def services(tmp_path):
    index = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "index")
    index.index_document(*vpn_doc())
    deploy = load_document("deploy-guide.md", DEPLOY)
    index.index_document(deploy, chunk_structure(deploy))
    retriever = HybridRetriever(index, reranker=WordReranker())
    return Services(retriever, FakeChat("Renew the certificate in NimbusConnect [1]."), FakeJudge(),
                    Pipeline(DocumentStore(tmp_path), index))


def client(services, **settings) -> TestClient:
    return TestClient(create_app(services, Settings(**settings)))


# ----- ops -----

def test_health_and_ready(services):
    with client(services) as c:
        assert c.get("/health").json() == {"status": "ok"}
        r = c.get("/ready")
        assert r.status_code == 200 and r.json()["ready"] and r.json()["chunks"] > 0
    services.problems.append("reranker model not downloaded")
    with client(services) as c:
        r = c.get("/ready")
        assert r.status_code == 503 and r.json()["problems"] == ["reranker model not downloaded"]


def test_openapi_documents_the_answer_schema(services):
    with client(services) as c:
        spec = c.get("/openapi.json").json()
    assert {"/v1/ask", "/v1/documents", "/v1/ingest"} <= set(spec["paths"])
    assert {"status", "code", "confidence", "sources", "retrieved"} <= set(spec["components"]["schemas"]["Response"]["properties"])


# ----- /v1/ask -----

def test_ask_returns_the_structured_answer(services):
    with client(services) as c:
        r = c.post("/v1/ask", json={"question": "How do I fix ERR_TUNNEL_4012 certificate expired?"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "answered"
    assert r.headers["X-Request-ID"] == body["request_id"]
    assert body["sources"][0]["source"] == "vpn-setup.md" and body["confidence"]["level"] in ("high", "medium")
    assert body["retrieved"] and body["retrieved"][0]["rerank_score"] is not None and body["mode"] == "hybrid"


def test_ask_can_run_dense_only_for_comparison(services):
    with client(services) as c:
        body = c.post("/v1/ask", json={"question": "ERR_TUNNEL_4012 certificate expired", "mode": "dense"}).json()
    assert body["mode"] == "dense" and all(p["sparse_rank"] is None for p in body["retrieved"])


def test_not_found_is_a_normal_200_answer(services):
    with client(services) as c:
        r = c.post("/v1/ask", json={"question": "Where do employees park the company bicycles overnight?"})
    assert r.status_code == 200 and r.json()["status"] == "not_found"


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x" * 2001}, {"question": "   "},
                                     {"question": "q", "mode": "vector"}, {}])
def test_bad_requests_are_422(services, payload):
    with client(services) as c:
        assert c.post("/v1/ask", json=payload).status_code == 422


@pytest.mark.parametrize("error, status, code", [
    (RuntimeError("Groq chat request failed after 5 attempts"), 503, "error.writer_unavailable"),
    (QuotaExhausted("daily limit reached"), 503, "error.quota_exhausted"),
    (DeadlineExceeded("budget used up"), 504, "error.timeout"),
])
def test_outages_map_to_5xx_with_the_same_body(services, error, status, code):
    services.writer = _Failing(error)
    with client(services) as c:
        r = c.post("/v1/ask", json={"question": "How do I fix ERR_TUNNEL_4012 certificate expired?"})
    assert r.status_code == status and r.json()["code"] == code and r.json()["status"] == "error"
    if code == "error.quota_exhausted":
        assert r.headers["Retry-After"] == "3600"


def test_api_key_when_configured(services):
    with client(services, api_key="s3cret") as c:
        assert c.post("/v1/ask", json={"question": "q"}).status_code == 401
        assert c.get("/v1/documents").status_code == 401
        assert c.get("/v1/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/health").status_code == 200  # ops endpoints stay open for health checks


# ----- documents -----

def test_documents_lists_sources_with_chunk_counts(services):
    with client(services) as c:
        body = c.get("/v1/documents").json()
        page = c.get("/v1/documents", params={"limit": 1, "offset": 1}).json()
    assert body["total"] == 2 and [d["source"] for d in body["documents"]] == ["deploy-guide.md", "vpn-setup.md"]
    assert all(d["chunks"] > 0 for d in body["documents"])
    assert page["total"] == 2 and [d["source"] for d in page["documents"]] == ["vpn-setup.md"]


def test_ingest_is_disabled_without_an_admin_key(services):
    with client(services) as c:
        r = c.post("/v1/ingest", files={"files": ("a.md", b"# A\n\ntext", "text/markdown")})
    assert r.status_code == 403


def test_ingest_needs_the_right_admin_key(services):
    with client(services, admin_key="admin") as c:
        r = c.post("/v1/ingest", files={"files": ("a.md", b"# A\n\ntext", "text/markdown")},
                   headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_ingest_indexes_a_new_document_and_it_becomes_searchable(services):
    doc = b"# Printer Guide\n\nThe office printer is configured from the Settings app under Devices."
    with client(services, admin_key="admin") as c:
        r = c.post("/v1/ingest", files=[("files", ("printer-guide.md", doc, "text/markdown"))],
                   headers={"X-API-Key": "admin"})
        assert r.status_code == 200
        result = r.json()["files"][0]
        assert result["status"] == "indexed" and result["chunks"] >= 1
        assert "printer-guide.md" in [d["source"] for d in c.get("/v1/documents").json()["documents"]]
        found = services.retriever.search("office printer configured Settings Devices").hits
        assert found[0].chunk.source == "printer-guide.md"


def test_each_bad_upload_fails_alone(services):
    good = ("files", ("notes.md", b"# Notes\n\nFine.", "text/markdown"))
    bad = [("files", ("../escape.md", b"# X", "text/markdown")), ("files", ("run.exe", b"MZ", "application/octet-stream")),
           ("files", ("broken.pdf", b"not a pdf at all", "application/pdf")),
           ("files", ("huge.md", b"x" * 2048, "text/markdown"))]
    with client(services, admin_key="admin", max_upload_bytes=1024) as c:
        files = c.post("/v1/ingest", files=[good] + bad, headers={"X-API-Key": "admin"}).json()["files"]
    by_name = {f["filename"]: f for f in files}
    assert by_name["notes.md"]["status"] == "indexed"
    assert by_name["escape.md"]["status"] == "failed"      # the path part is stripped, then the name is checked
    assert by_name["run.exe"]["status"] == "failed" and "end in one of" in by_name["run.exe"]["error"]
    assert by_name["broken.pdf"]["status"] == "failed" and by_name["broken.pdf"]["error"]
    assert by_name["huge.md"]["status"] == "failed" and "larger than" in by_name["huge.md"]["error"]


def test_cors_only_for_configured_origins(services):
    with client(services, cors_origins=["http://localhost:8501"]) as c:
        ok = c.options("/v1/ask", headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "POST"})
        other = c.options("/v1/ask", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8501"
    assert "access-control-allow-origin" not in other.headers
