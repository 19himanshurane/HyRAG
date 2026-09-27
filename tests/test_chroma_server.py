"""The index against a REAL Chroma server (what Docker Compose runs), started from the installed chromadb
package, which is the same version as the pinned server image. Skipped if the `chroma` command is missing."""
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from hyrag import index as index_module
from hyrag.index import ChunkIndex, _chroma_client
from hyrag.retrieval import HybridRetriever

from .test_index import CountingEmbedder, vpn_doc

CHROMA = shutil.which("chroma") or shutil.which("chroma", path=str(Path(sys.executable).parent))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ChromaServer:
    def __init__(self, path: Path):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc = subprocess.Popen([CHROMA, "run", "--path", str(path), "--host", "127.0.0.1", "--port",
                                      str(self.port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if httpx.get(f"{self.url}/api/v2/heartbeat", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.2)
        self.stop()
        raise RuntimeError("chroma server did not start")

    def stop(self) -> None:
        self.proc.kill()
        self.proc.wait(10)


@pytest.fixture
def chroma(tmp_path):
    if not CHROMA:
        pytest.skip("the chroma server command is not installed")
    server = ChromaServer(tmp_path / "chroma-server")
    yield server
    server.stop()


def test_server_index_gives_the_same_results_as_the_embedded_one(tmp_path, chroma):
    embedded = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "local")
    remote = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "remote", chroma_url=chroma.url)
    for ix in (embedded, remote):
        ix.index_document(*vpn_doc())
    for query in ("certificate expired", "gateway region", "slow connection"):
        a, b = embedded.search_dense(query, k=5), remote.search_dense(query, k=5)
        assert [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b]
        assert [h.score for h in a] == pytest.approx([h.score for h in b], abs=1e-6)
    assert not any(remote.check_sync().values())


def test_a_restarted_api_finds_what_the_seed_wrote(tmp_path, chroma):
    """Compose runs the seed and the API as two processes sharing the data volume and the Chroma server."""
    seed = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "index", chroma_url=chroma.url)
    seed.index_document(*vpn_doc())
    api = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "index", chroma_url=chroma.url)
    assert api.count() == seed.count() > 0
    assert api.search_dense("certificate expired", k=1)[0].chunk.text.startswith("This error means")


def test_a_wiped_chroma_volume_is_detected_and_repaired(tmp_path, chroma):
    ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "index", chroma_url=chroma.url).index_document(*vpn_doc())
    fresh = ChromaServer(tmp_path / "empty-chroma")  # same chunk table, a Chroma that lost its data
    try:
        ix = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "index", chroma_url=fresh.url)
        assert len(ix.check_sync()["missing_in_chroma"]) == ix.count()
        ix.repair()
        assert not any(ix.check_sync().values())
        assert ix.search_dense("certificate expired", k=1)
    finally:
        fresh.stop()


def test_the_http_client_has_a_timeout(tmp_path, chroma):
    session = _chroma_client(tmp_path, chroma.url)._server._session
    assert session.timeout.read == index_module.CHROMA_TIMEOUT_SECONDS
    assert session.timeout.connect == index_module.CHROMA_CONNECT_SECONDS


def test_chroma_dying_mid_run_degrades_to_keyword_search(tmp_path, chroma):
    ix = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "index", chroma_url=chroma.url)
    ix.index_document(*vpn_doc())
    chroma.stop()
    ix._vectors = None  # as after any write: the next meaning search must reload the vectors from Chroma
    t0 = time.perf_counter()
    result = HybridRetriever(ix).search("ERR_TUNNEL_4012 certificate")
    assert time.perf_counter() - t0 < 10
    assert "dense_search_unavailable" in result.degraded
    assert result.hits[0].chunk.text.startswith("This error means your device certificate has expired")


def test_unreachable_chroma_fails_at_startup(tmp_path):
    with pytest.raises(ConnectionError, match="not answering"):
        ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "index", chroma_url=f"http://127.0.0.1:{_free_port()}")


def test_a_chroma_that_never_answers_fails_at_startup_instead_of_hanging(tmp_path, monkeypatch):
    monkeypatch.setattr(index_module, "CHROMA_CONNECT_SECONDS", 0.5)
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen()  # accepts connections, never replies
    held = []
    threading.Thread(target=lambda: held.append(silent.accept()), daemon=True).start()
    try:
        t0 = time.perf_counter()
        with pytest.raises(ConnectionError):
            _chroma_client(tmp_path, f"http://127.0.0.1:{silent.getsockname()[1]}")
        assert time.perf_counter() - t0 < 5
    finally:
        silent.close()


@pytest.mark.parametrize("url", ["chroma:8000", "ftp://chroma:8000", "http://"])
def test_a_malformed_chroma_url_is_refused(tmp_path, url):
    with pytest.raises(ValueError, match="Chroma URL"):
        _chroma_client(tmp_path, url)
