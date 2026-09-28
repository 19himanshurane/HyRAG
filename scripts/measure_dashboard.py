"""How much time does the dashboard itself add? Measured separately from the API.

Usage (from the repo root): PYTHONPATH=. python scripts/measure_dashboard.py
Runs the real dashboard (Streamlit AppTest) against a local API with a real index and instant fake models, so
the time measured is the dashboard and HTTP, not the models. AppTest adds some overhead of its own, so treat the
numbers as upper bounds. Browser rendering isn't included.
"""
import os
import socket
import statistics
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st
import uvicorn
from streamlit.testing.v1 import AppTest

from hyrag.api import Services, Settings, create_app
from hyrag.chunking import chunk_structure
from hyrag.index import ChunkIndex
from hyrag.loader import DocumentStore, load_document
from hyrag.pipeline import Pipeline
from hyrag.retrieval import HybridRetriever
from tests.test_answer import FakeJudge
from tests.test_generation import FakeChat
from tests.test_index import CountingEmbedder, vpn_doc
from tests.test_rewrite import DEPLOY, WordReranker

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
Q = "How do I fix ERR_TUNNEL_4012 certificate expired?"


def ms(xs):
    return f"median {statistics.median(xs) * 1000:.0f} ms, max {max(xs) * 1000:.0f} ms"


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    index = ChunkIndex(CountingEmbedder(), data_dir=tmp / "index")
    index.index_document(*vpn_doc())
    deploy = load_document("deploy-guide.md", DEPLOY)
    index.index_document(deploy, chunk_structure(deploy))
    services = Services(HybridRetriever(index, reranker=WordReranker()), FakeChat("Renew the certificate [1]."),
                        FakeJudge(), Pipeline(DocumentStore(tmp), index))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(services, Settings()), port=port, log_level="warning", access_log=False))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    os.environ["HYRAG_API_URL"] = f"http://127.0.0.1:{port}"
    st.cache_data.clear()
    st.cache_resource.clear()

    loads, asks, redraws, compares, api_times = [], [], [], [], []
    for _ in range(5):
        st.cache_data.clear()
        at = AppTest.from_file(APP, default_timeout=30)
        t0 = time.perf_counter(); at.run(); loads.append(time.perf_counter() - t0)
        at.text_area[0].input(Q)
        t0 = time.perf_counter(); at.button[0].click().run(); asks.append(time.perf_counter() - t0)
        api_times.append(float(next(c.value for c in at.caption if "API" in c.value and " s ·" in c.value)
                               .split("API ")[1].split(" s")[0]))
        for _ in range(5):
            t0 = time.perf_counter(); at.run(); redraws.append(time.perf_counter() - t0)
        at.text_area[0].input(Q + " ")
        at.checkbox[0].check()
        t0 = time.perf_counter(); at.button[0].click().run(); compares.append(time.perf_counter() - t0)
    writer_calls = len(services.writer.calls)
    print(f"first load of the page (status + document list):  {ms(loads)}")
    print(f"press Ask -> answer on screen:                      {ms(asks)}  (of which the API call: "
          f"median {statistics.median(api_times) * 1000:.0f} ms)")
    print(f"redraw with an answer on screen (no API call):      {ms(redraws)}")
    print(f"press Ask with the dense-only comparison (2 calls): {ms(compares)}")
    print(f"API answer requests made: {writer_calls} (5 single + 10 compared; redraws made none)")
    server.should_exit = True


if __name__ == "__main__":
    main()
