"""HyRAG as one Streamlit app, for hosts that run a single process (Streamlit Community Cloud).

Run locally:  streamlit run deploy/streamlit_app.py     (needs MISTRAL_API_KEY and GROQ_API_KEY)

On a cold start it indexes the documents and starts the HTTP API in a background thread, once per process.
After that every visitor's page is the normal dashboard (dashboard/app.py) talking to that API on localhost.
The index is built from deploy/embeddings.sqlite, so no embedding requests are needed to start.
"""
import os
import runpy
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PER_VISITOR_PER_HOUR = "10"
QUESTIONS_PER_DAY = "100"  # the free model tier covers about 115 checked questions a day


@st.cache_resource(show_spinner="Starting HyRAG. A cold start indexes the documents and takes a minute or two.")
def backend() -> str:
    """Index the documents and start the API once per process. Returns the API's address."""
    os.chdir(ROOT)  # the seed reads corpus/ and sample_docs/ relative to the repository root
    sys.path.insert(0, str(ROOT))
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # local runs; a hosted copy has no .env and uses its secrets box
    for key in ("MISTRAL_API_KEY", "GROQ_API_KEY"):  # the host's secrets box, if the environment lacks them
        if not os.environ.get(key):
            try:
                os.environ[key] = st.secrets[key]
            except Exception:
                raise RuntimeError(f"{key} is not set: add it to the app's secrets") from None
    data = Path(os.environ.setdefault("HYRAG_DATA_DIR", str(ROOT / "data")))
    data.mkdir(parents=True, exist_ok=True)
    if not (data / "embeddings.sqlite").exists():
        shutil.copy(ROOT / "deploy" / "embeddings.sqlite", data / "embeddings.sqlite")
    os.environ.setdefault("HYRAG_VISITOR_LIMIT_PER_HOUR", QUESTIONS_PER_VISITOR_PER_HOUR)
    os.environ.setdefault("HYRAG_DAILY_LIMIT", QUESTIONS_PER_DAY)

    from hyrag import seed
    from hyrag.rerank import download, is_downloaded

    if seed.main(["--in-process"]) != 0:
        raise RuntimeError("indexing the documents failed: see the app's log")
    if not is_downloaded():
        download()

    import uvicorn

    from hyrag.api import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="hyrag-api").start()
    deadline = time.monotonic() + 300  # startup loads the reranker and the search data
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the API did not start within 5 minutes")
        time.sleep(0.2)
    return f"http://127.0.0.1:{port}"


os.environ["HYRAG_API_URL"] = backend()
sys.path.insert(0, str(ROOT / "dashboard"))  # the dashboard imports its neighbours `client` and `safety`
runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
