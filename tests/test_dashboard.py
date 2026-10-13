"""The dashboard, clicked through with Streamlit's AppTest against a REAL HTTP API (uvicorn in a thread) that uses a
real index and fake models. Covers what the brief asks the dashboard to show, the error states, and the speed
guarantee: once an answer is on screen, redrawing the page never calls the API again."""
import socket
import threading
import time
from pathlib import Path

import pytest
import streamlit as st
import uvicorn
from streamlit.testing.v1 import AppTest

from hyrag.api import Settings, create_app
from hyrag.uploads import UploadPolicy, UploadStore

from .conftest import make_pdf
from .test_api import services  # noqa: F401  (the fixture: real index + fake models)
from .test_index import CountingEmbedder
from .test_uploads import BIKES, Clock, parking_pdf

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")  # AppTest resolves relative paths against this file


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(services, monkeypatch, settings):  # noqa: F811
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(services, settings), port=port, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    monkeypatch.setenv("HYRAG_API_URL", f"http://127.0.0.1:{port}")
    st.cache_data.clear()
    st.cache_resource.clear()  # the client is cached per process: don't reuse one pointing at another test's server
    yield services
    server.should_exit = True
    thread.join(5)


@pytest.fixture
def api(services, monkeypatch):  # noqa: F811
    yield from _serve(services, monkeypatch, Settings())


@pytest.fixture
def limited_api(services, monkeypatch):  # noqa: F811
    yield from _serve(services, monkeypatch, Settings(visitor_limit=1))


@pytest.fixture
def uploads_api(services, monkeypatch):  # noqa: F811
    services.clock = Clock()
    services.uploads = UploadStore(CountingEmbedder(), UploadPolicy(max_bytes=64 * 1024), clock=services.clock)
    services.sent = []  # every file that reached the API
    add = services.uploads.add
    services.uploads.add = lambda visitor, name, data: (services.sent.append(name), add(visitor, name, data))[1]
    yield from _serve(services, monkeypatch, Settings())


def page() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30)


def ask(at: AppTest, question: str, compare: bool = False) -> AppTest:
    at.text_area[0].input(question)
    if compare:
        at.checkbox[0].check()
    return at.button[0].click().run()


def all_text(at: AppTest) -> str:
    parts = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.subheader]
    parts += [e.value for kind in ("success", "info", "warning", "error") for e in getattr(at, kind)]
    return "\n".join(str(p) for p in parts)


def test_page_loads_and_shows_the_api_is_ready(api):
    at = page().run()
    assert not at.exception
    assert any("API ready" in s.value for s in at.sidebar.success)


def test_asking_shows_answer_citations_sources_and_confidence(api):
    at = ask(page().run(), "How do I fix ERR_TUNNEL_4012 certificate expired?")
    assert not at.exception
    text = all_text(at)
    assert "Answered from the documents" in text
    assert "(#hybrid-source-1)" in text                     # the [1] in the answer links to its source
    assert any(h.value.startswith("[1] vpn-setup.md") for h in at.subheader)
    assert {m.label for m in at.metric} >= {"Confidence", "Retrieval", "Citations verified", "Completeness"}


def test_compare_shows_hybrid_and_dense_side_by_side(api):
    at = ask(page().run(), "How do I fix ERR_TUNNEL_4012 certificate expired?", compare=True)
    text = all_text(at)
    assert "**Hybrid**" in text and "**Dense only**" in text
    assert "(#hybrid-source-1)" in text and "(#dense-source-1)" in text  # each column's citations stay separate


def test_not_found_is_shown_as_such(api):
    at = ask(page().run(), "Where do employees park the company bicycles overnight?")
    assert "Not found in the documents" in all_text(at)


def test_redrawing_after_an_answer_never_calls_the_api_again(api):
    at = ask(page().run(), "How do I fix ERR_TUNNEL_4012 certificate expired?")
    calls = len(api.writer.calls)
    for _ in range(3):
        at.run()  # a rerun without pressing Ask (what any widget interaction would cause)
    assert len(api.writer.calls) == calls and "Answered from the documents" in all_text(at)


def test_a_visitor_over_the_limit_sees_why(limited_api):
    at = ask(page().run(), "How do I fix ERR_TUNNEL_4012 certificate expired?")
    assert "Answered from the documents" in all_text(at)
    at = ask(at, "And how do I fix ERR_TUNNEL_4013 gateway region?")  # same browser session: same visitor
    assert not at.exception
    assert any("1 question per visitor per hour" in e.value for e in at.error)
    assert len(limited_api.writer.calls) == 1  # the second question never reached the model


def test_empty_question_is_caught_before_any_request(api):
    at = ask(page().run(), "   ")
    assert any("Type a question first" in w.value for w in at.warning) and api.writer.calls == []


def test_api_down_is_explained_not_a_crash(monkeypatch):
    monkeypatch.setenv("HYRAG_API_URL", f"http://127.0.0.1:{_free_port()}")  # nothing listens there
    st.cache_data.clear()
    st.cache_resource.clear()
    at = ask(page().run(), "How do I fix ERR_TUNNEL_4012?")
    assert not at.exception
    assert any("API not ready" in e.value for e in at.sidebar.error)
    assert any("unreachable" in e.value for e in at.error)


# ----- a visitor's own document -----

def upload(at: AppTest, name: str = "parking.pdf", data: bytes | None = None) -> AppTest:
    at.file_uploader[0].upload(name, data if data is not None else parking_pdf(), "application/pdf")
    return at.run()


def test_no_upload_box_when_the_api_has_uploads_off(api):
    at = page().run()
    assert not at.exception and len(at.file_uploader) == 0
    assert at.selectbox[0].label == "Answer from" and at.selectbox[0].value == "demo"


def test_uploading_a_document_and_asking_about_it(uploads_api):
    at = page().run()
    assert "don't upload anything confidential" in all_text(at) and "64 KB" not in all_text(at)
    at = upload(at)
    assert not at.exception and len(at.file_uploader) == 0  # the box gives way to the document
    assert at.selectbox[0].value == "__upload__"            # and answers now come from it
    assert "parking.pdf" in all_text(at) and "kept for about 60 more min" in all_text(at)
    at = ask(at, BIKES)
    text = all_text(at)
    assert "Answered from the documents" in text and "Answered from: Your document: parking.pdf" in text
    assert any(h.value.startswith("[1] parking.pdf") for h in at.subheader)
    assert uploads_api.uploads.count() == 1 and uploads_api.sent == ["parking.pdf"]  # reruns never upload again


def test_the_built_in_documents_stay_one_click_away(uploads_api):
    at = upload(page().run())
    at.selectbox[0].select("demo").run()
    at = ask(at, BIKES)
    assert "Not found in the documents" in all_text(at)  # the shared documents don't have the visitor's file


def test_removing_the_document_brings_the_box_back(uploads_api):
    at = upload(page().run())
    at.sidebar.button[0].click().run()  # "Remove my document"
    assert not at.exception and uploads_api.uploads.count() == 0
    assert len(at.file_uploader) == 1 and at.file_uploader[0].value is None
    assert at.selectbox[0].value == "demo"


def test_a_refused_file_says_why_and_is_not_sent_again(uploads_api):
    at = upload(page().run(), "scan.pdf", make_pdf([[]]))
    assert not at.exception and any("scanned PDF" in e.value for e in at.sidebar.error)
    at.run()
    at.run()
    assert uploads_api.sent == ["scan.pdf"] and any("scanned PDF" in e.value for e in at.sidebar.error)
    at = upload(at, "big.pdf", b"x" * (64 * 1024 + 1))  # over the limit: stopped before it is sent
    assert any("larger than" in e.value for e in at.sidebar.error) and uploads_api.sent == ["scan.pdf"]


def test_an_expired_document_is_explained_and_no_longer_offered(uploads_api):
    at = upload(page().run())
    uploads_api.clock.now += 3601  # the API has dropped it; this page doesn't know yet
    at = ask(at, BIKES)
    assert not at.exception and any("Upload it again" in e.value for e in at.error)
    at.run()
    assert at.selectbox[0].value == "demo" and len(at.file_uploader) == 1
