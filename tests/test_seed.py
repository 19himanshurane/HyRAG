"""The seed job Docker Compose runs before the API: re-runnable, strict about failures, keeps folder names."""
import sqlite3

import pytest

from hyrag import seed

from .test_index import VPN, CountingEmbedder


class FakeEmbedder(CountingEmbedder):
    """Stands in for MistralEmbedder: same constructor keyword and context-manager use, no network."""
    made: list["FakeEmbedder"] = []

    def __init__(self, cache_path=None):
        super().__init__()
        self.requests_made = 0
        FakeEmbedder.made.append(self)

    def embed(self, texts):
        self.requests_made += 1
        return super().embed(texts)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def env(tmp_path, monkeypatch):
    FakeEmbedder.made = []
    monkeypatch.setattr(seed, "MistralEmbedder", FakeEmbedder)
    monkeypatch.setattr(seed, "load_dotenv", lambda: None)
    monkeypatch.setenv("MISTRAL_API_KEY", "test")
    monkeypatch.setenv("HYRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("HYRAG_CHROMA_URL", raising=False)
    docs = tmp_path / "docs"
    (docs / "it").mkdir(parents=True)
    (docs / "it" / "vpn-setup.md").write_text(VPN, encoding="utf-8")
    return tmp_path, docs


def sources(tmp_path) -> set[str]:
    db = sqlite3.connect(tmp_path / "data" / "index" / "chunks-structure.sqlite")
    return {r[0] for r in db.execute("SELECT DISTINCT source FROM chunks")}


def test_seed_indexes_documents_keeping_their_folder(env, capsys):
    tmp_path, docs = env
    assert seed.main([str(docs)]) == 0
    assert sources(tmp_path) == {"it/vpn-setup.md"}  # the name the golden set's evidence uses
    assert "1 documents" in capsys.readouterr().out


def test_running_the_seed_again_embeds_nothing(env):
    tmp_path, docs = env
    assert seed.main([str(docs)]) == 0
    assert seed.main([str(docs)]) == 0
    assert FakeEmbedder.made[1].texts == []


def test_a_file_that_fails_fails_the_seed(env, capsys):
    tmp_path, docs = env
    (docs / "broken.pdf").write_bytes(b"not a pdf")
    assert seed.main([str(docs)]) == 1
    assert "FAILED broken.pdf" in capsys.readouterr().err
    assert sources(tmp_path) == {"it/vpn-setup.md"}  # the good file still went in


def test_missing_folder_or_key_is_refused_before_any_work(env, monkeypatch, capsys):
    tmp_path, docs = env
    assert seed.main([str(tmp_path / "nope")]) == 1
    monkeypatch.delenv("MISTRAL_API_KEY")
    assert seed.main([str(docs)]) == 1
    assert FakeEmbedder.made == []
