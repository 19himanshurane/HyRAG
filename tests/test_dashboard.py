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

from .test_api import services  # noqa: F401  (the fixture: real index + fake models)

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")  # AppTest resolves relative paths against this file


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def api(services, monkeypatch):  # noqa: F811
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(services, Settings()), port=port, log_level="warning", access_log=False))
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
