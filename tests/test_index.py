import pytest

from hyrag.chunking import chunk_structure
from hyrag.index import ChunkIndex, tokenize
from hyrag.loader import load_document

from .test_semantic import BagOfWordsEmbedder


class CountingEmbedder(BagOfWordsEmbedder):
    """Records every text sent for embedding, so tests can prove what was (not) re-embedded."""

    def __init__(self):
        super().__init__()
        self.texts: list[str] = []

    def embed(self, texts):
        self.texts.extend(texts)
        return super().embed(texts)


VPN = """# VPN Setup Guide

All employees must connect through the Nimbus VPN when working remotely.

## Troubleshooting

### Error ERR_TUNNEL_4012

This error means your device certificate has expired. Open NimbusConnect and click Renew.

### Error ERR_TUNNEL_4013

The gateway rejected your region. Open Settings and pick the gateway for your country.

### Slow connection

Switch the gateway to the region closest to you, for example eu-west-gw.
"""


def vpn_doc(text: str = VPN):
    doc = load_document("vpn-setup.md", text.encode())
    return doc, chunk_structure(doc)


@pytest.fixture
def index(tmp_path):
    return ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "index")


# ----- tokenizer -----

def test_identifiers_are_kept_whole_and_split_into_parts():
    assert tokenize("Fix ERR_TUNNEL_4012 now") == ["fix", "err_tunnel_4012", "err", "tunnel", "4012", "now"]
    assert "nimbus-deploy" in tokenize("run nimbus-deploy --env staging")
    assert "3.1.3.1" in tokenize("see 3.1.3.1. Out-of-Band") and "band" in tokenize("Out-of-Band")


def test_typography_is_folded_so_typed_queries_match():  # audit G7
    assert tokenize("don’t use “smart” quotes – ever") == tokenize("don't use \"smart\" quotes - ever")
    assert tokenize("no break") == ["no", "break"]
    assert tokenize("the ﬁle") == ["file"]  # NFKC turns the ﬁ ligature into f + i


def test_stopwords_and_case():
    assert tokenize("The VPN and THE Gateway") == ["vpn", "gateway"]


# ----- indexing and search -----

def test_both_indexes_hold_the_same_chunks(index):
    doc, chunks = vpn_doc()
    result = index.index_document(doc, chunks)
    assert result.added == len(chunks) == index.count()
    assert not any(index.check_sync().values())


def test_keyword_search_ranks_the_exact_error_code_first(index):
    index.index_document(*vpn_doc())
    hits = index.search_sparse("How do I fix ERR_TUNNEL_4012?", k=3)
    assert hits[0].chunk.heading.endswith("Error ERR_TUNNEL_4012")
    # The near-miss code shares "err" and "tunnel", so it ranks second, clearly lower. (With rank_bm25's
    # original Okapi weight those words, present in exactly half of these 4 chunks, weighed 0 and 4013 vanished.)
    assert hits[1].chunk.heading.endswith("Error ERR_TUNNEL_4013") and hits[1].score < hits[0].score


def test_dense_search_returns_similarities(index):
    index.index_document(*vpn_doc())
    hits = index.search_dense("my vpn is slow, which gateway region should I switch to", k=2)
    assert len(hits) == 2 and 0 < hits[1].score <= hits[0].score <= 1
    assert hits[0].chunk.heading.endswith("Slow connection")


def test_reindexing_an_edited_document_only_embeds_what_changed(index):
    doc, chunks = vpn_doc()
    index.index_document(doc, chunks)
    before = len(index.embedder.texts)
    edited = VPN.replace("click Renew.", "click Renew, then restart the client.")
    doc2, chunks2 = vpn_doc(edited)
    result = index.index_document(doc2, chunks2)
    assert (result.added, result.removed, result.kept) == (1, 1, len(chunks) - 1)
    assert len(index.embedder.texts) - before == 1  # one changed chunk re-embedded, not the whole document
    assert "restart the client" in index.search_sparse("restart client", k=1)[0].chunk.text
    assert index.check_sync()["missing_in_chroma"] == [] and index.count() == len(chunks2)


def test_removing_a_document_removes_it_from_both_indexes(index):
    doc, chunks = vpn_doc()
    index.index_document(doc, chunks)
    assert index.remove_document(doc.doc_id) == len(chunks)
    assert index.count() == 0 and index.collection.count() == 0
    assert index.search_sparse("ERR_TUNNEL_4012") == [] and index.search_dense("vpn") == []


def test_index_survives_a_restart(tmp_path):
    first = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "idx")
    first.index_document(*vpn_doc())
    reopened = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "idx")
    assert reopened.count() == first.count()
    assert reopened.search_sparse("ERR_TUNNEL_4013", k=1)[0].chunk.heading.endswith("ERR_TUNNEL_4013")
    assert reopened.search_dense("certificate expired renew", k=1)[0].chunk.heading.endswith("ERR_TUNNEL_4012")


def test_crash_between_writes_is_detected_and_repaired(index):
    doc, chunks = vpn_doc()
    index.index_document(doc, chunks)
    lost = chunks[1].chunk_id
    index.collection.delete(ids=[lost])  # as if the process died after the table write, before Chroma's
    index.collection.add(ids=["ghost"], embeddings=[[1.0] * 256], documents=["stale"])
    assert index.check_sync() == {"missing_in_chroma": [lost], "extra_in_chroma": ["ghost"], "duplicates_of_missing_chunks": []}
    index.repair()
    assert not any(index.check_sync().values())


def test_failed_embedding_changes_nothing(index):
    doc, chunks = vpn_doc()

    class Broken:
        def embed(self, texts):
            raise RuntimeError("Mistral is down")

    index.embedder = Broken()
    with pytest.raises(RuntimeError):
        index.index_document(doc, chunks)
    assert index.count() == 0 and index.collection.count() == 0


def test_search_skips_results_missing_from_the_table(index, caplog):  # audit H2
    index.index_document(*vpn_doc())
    index.collection.add(ids=["ghost"], embeddings=[[1.0] * 256], documents=["stale"])  # crash-state drift
    hits = index.search_dense("certificate expired renew", k=10)
    assert hits and all(h.chunk.chunk_id != "ghost" for h in hits)
    assert any("check_sync" in r.getMessage() for r in caplog.records)


def test_index_is_safe_to_use_from_many_threads(index):  # audit H3
    from concurrent.futures import ThreadPoolExecutor
    index.index_document(*vpn_doc())
    queries = ["ERR_TUNNEL_4012", "gateway region", "employees remotely", "certificate"] * 5
    with ThreadPoolExecutor(max_workers=8) as pool:
        sparse = list(pool.map(index.search_sparse, queries))
        dense = list(pool.map(index.search_dense, queries))
    assert all(sparse[:4]) and all(dense)


def test_a_second_index_object_sees_the_other_ones_writes(tmp_path):  # audit H4
    a = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "shared")
    doc, chunks = vpn_doc()
    a.index_document(doc, chunks)
    b = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "shared")
    assert b.search_sparse("ERR_TUNNEL_4012")  # b builds its BM25 snapshot now
    extra = load_document("extra.md", b"# Extra\n\nThe quarterly zebrafish audit happens every March.")
    a.index_document(extra, chunk_structure(extra))
    assert [h.chunk.source for h in b.search_sparse("zebrafish audit")] == ["extra.md"]
    a.remove_document(doc.doc_id)
    assert b.search_sparse("ERR_TUNNEL_4012") == []  # no stale hit, no crash


def test_an_index_refuses_a_different_embedding_model(tmp_path):  # audit H5
    from hyrag.index import EmbeddingModelMismatch

    class Other(CountingEmbedder):
        model = "some-other-model"

    ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i").index_document(*vpn_doc())
    with pytest.raises(EmbeddingModelMismatch, match="rebuild"):
        ChunkIndex(Other(), data_dir=tmp_path / "i")
    assert ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i").count() > 0  # the right model still opens it


def test_pages_round_trip_through_chroma_metadata(index):
    doc, chunks = vpn_doc()
    index.index_document(doc, chunks)
    meta = index.collection.get(ids=[chunks[0].chunk_id], include=["metadatas"])["metadatas"][0]
    assert meta["page"] == -1 and meta["strategy"] == "structure"  # Chroma can't store None
    assert index.search_sparse("employees remotely", k=1)[0].chunk.page is None  # read back from the table
