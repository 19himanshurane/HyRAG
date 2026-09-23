from pathlib import Path

import pytest

from hyrag.chunking import chunk_fixed
from hyrag.index import ChunkIndex
from hyrag.loader import DocumentStore
from hyrag.pipeline import Pipeline

from .test_index import VPN, CountingEmbedder

POLICY = "# Leave\n\nFull-time employees receive 24 days of paid annual leave per calendar year."


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def setup(tmp_path):
    src = tmp_path / "src"
    files = [write(src / "vpn-setup.md", VPN), write(src / "hr" / "leave.md", POLICY)]
    pipe = Pipeline(DocumentStore(tmp_path / "data"), ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "index"))
    pipe.ingest(files, root=src, workers=1)
    return pipe, src


def sources(pipe, query):
    return [h.chunk.source for h in pipe.index.search_sparse(query)]


def test_ingest_stores_and_indexes(setup):
    pipe, _ = setup
    assert sources(pipe, "ERR_TUNNEL_4012")[0] == "vpn-setup.md"
    assert sources(pipe, "annual leave") == ["hr/leave.md"]
    assert (pipe.store.raw_dir / "hr" / "leave.md").exists()


def test_deleting_a_document_removes_it_from_the_index(setup):  # audit H1
    pipe, _ = setup
    pipe.delete("vpn-setup.md")
    assert sources(pipe, "ERR_TUNNEL_4012") == []
    assert not (pipe.store.raw_dir / "vpn-setup.md").exists()
    assert sources(pipe, "annual leave") == ["hr/leave.md"]


def test_reprocess_drops_documents_the_store_lost(setup):
    pipe, _ = setup
    (pipe.store.raw_dir / "hr" / "leave.md").unlink()  # the raw original disappeared (e.g. removed on disk)
    pipe.reprocess()
    assert sources(pipe, "annual leave") == []
    assert sources(pipe, "ERR_TUNNEL_4012")[0] == "vpn-setup.md"


def test_reprocess_keeps_a_document_whose_reparse_failed(setup):
    pipe, _ = setup
    (pipe.store.raw_dir / "hr" / "leave.md").write_text("   ", encoding="utf-8")  # now unparseable
    pipe.reprocess(force=True)
    assert "hr/leave.md" in pipe.store.failures
    assert sources(pipe, "annual leave") == ["hr/leave.md"]  # last good version still answers questions


def test_reingesting_an_edited_file_updates_the_index(setup):
    pipe, src = setup
    write(src / "hr" / "leave.md", POLICY.replace("24 days", "30 days"))
    pipe.ingest([src / "hr" / "leave.md"], root=src, workers=1)
    [hit] = pipe.index.search_sparse("annual leave")
    assert "30 days" in hit.chunk.text


def test_chunker_must_match_the_index_strategy(tmp_path):
    src = tmp_path / "src"
    f = write(src / "a.md", POLICY)
    pipe = Pipeline(DocumentStore(tmp_path / "d"), ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i"),
                    chunker=chunk_fixed)  # fixed chunks into the "structure" index
    with pytest.raises(ValueError, match="fixed"):
        pipe.ingest([f], root=src, workers=1)
