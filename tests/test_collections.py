"""Collections: one company's documents each. A question is answered from ONE company's documents only, and
another company's documents never change this one's results (not even its keyword statistics)."""
import pytest
from fastapi.testclient import TestClient

from hyrag.answer import AnswerCache, ask
from hyrag.api import Services, Settings, create_app
from hyrag.chunking import chunk_structure
from hyrag.index import ChunkIndex, collection_of
from hyrag.loader import DocumentStore, load_document
from hyrag.pipeline import Pipeline
from hyrag.retrieval import HybridRetriever

from .test_answer import FakeJudge
from .test_generation import FakeChat
from .test_index import CountingEmbedder, vpn_doc
from .test_rewrite import WordReranker

# Another company's page about the SAME error, worded to match the question even better than ours.
OTHER = """# IT help

## Error ERR_TUNNEL_4012

ERR_TUNNEL_4012 certificate expired: ERR_TUNNEL_4012 means the ERR_TUNNEL_4012 certificate expired. Ask IT in Slack.
"""


def posthog_doc():
    doc = load_document("posthog/handbook/it/vpn.md", OTHER.encode())
    return doc, chunk_structure(doc)


@pytest.fixture
def index(tmp_path):
    ix = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "index")
    ix.index_document(*vpn_doc())
    ix.index_document(*posthog_doc())
    return ix


def sources(hits) -> set[str]:
    return {h.chunk.source for h in hits}


def test_a_documents_collection_is_its_named_first_folder():
    assert collection_of("posthog/handbook/people/time-off.md") == "posthog"
    assert collection_of("vpn-setup.md") == "demo"
    assert collection_of("engineering/k8s-secrets.md") == "demo"  # a folder, but not a named collection
    assert collection_of("posthog.md") == "demo"                  # a file name is not a folder


def test_each_search_stays_inside_its_collection(index):
    q = "ERR_TUNNEL_4012 certificate expired"
    for search in (index.search_dense, index.search_sparse):
        assert sources(search(q, k=10, collection="demo")) == {"vpn-setup.md"}
        assert sources(search(q, k=10, collection="posthog")) == {"posthog/handbook/it/vpn.md"}
        assert sources(search(q, k=10)) == {"vpn-setup.md", "posthog/handbook/it/vpn.md"}  # None = all


def test_other_companies_cannot_crowd_out_the_top_k(index):
    """The filter runs BEFORE the top k is taken: with k=1 the other company's closer match must not take the
    only slot and leave this collection with nothing."""
    q = "ERR_TUNNEL_4012 certificate expired"
    assert index.search_sparse(q, k=1)[0].chunk.source.startswith("posthog/")  # it IS the better match overall
    assert sources(index.search_sparse(q, k=1, collection="demo")) == {"vpn-setup.md"}
    assert len(index.search_dense(q, k=1, collection="demo")) == 1


def test_another_companys_documents_never_change_this_ones_scores(tmp_path):
    alone = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "alone")
    alone.index_document(*vpn_doc())
    shared = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "shared")
    shared.index_document(*vpn_doc())
    shared.index_document(*posthog_doc())
    q = "ERR_TUNNEL_4012 certificate expired"
    for search in ("search_sparse", "search_dense"):
        a = getattr(alone, search)(q, k=10)
        b = getattr(shared, search)(q, k=10, collection="demo")
        assert [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b]
        assert [h.score for h in a] == pytest.approx([h.score for h in b])  # BM25 statistics per collection too


def test_an_unknown_collection_is_refused_before_any_paid_call(index):
    with pytest.raises(ValueError, match="unknown collection"):
        index.search_dense("anything", collection="acme")
    assert index.embedder.texts[-1:] != ["anything"]  # refused before the query was embedded
    with pytest.raises(ValueError, match="unknown collection"):
        HybridRetriever(index).search("anything", collection="acme")


def test_the_retriever_and_ask_answer_from_one_collection(index):
    retriever = HybridRetriever(index, reranker=WordReranker())
    hits = retriever.search("ERR_TUNNEL_4012 certificate expired", collection="demo").hits
    assert sources(hits) == {"vpn-setup.md"}
    r = ask("How do I fix ERR_TUNNEL_4012 certificate expired?", retriever, FakeChat("Renew it [1]."), FakeJudge(),
            collection="posthog")
    assert r.collection == "posthog" and {p.source for p in r.retrieved} == {"posthog/handbook/it/vpn.md"}


def test_one_companys_cached_answer_is_never_served_to_another(index):
    retriever, cache = HybridRetriever(index, reranker=WordReranker()), AnswerCache()
    q = "How do I fix ERR_TUNNEL_4012 certificate expired?"
    first = ask(q, retriever, FakeChat("Renew it [1]."), FakeJudge(), cache=cache, collection="demo")
    other = ask(q, retriever, FakeChat("Ask IT [1]."), FakeJudge(), cache=cache, collection="posthog")
    again = ask(q, retriever, FakeChat("unused"), FakeJudge(), cache=cache, collection="demo")
    assert not other.cached and other.sources[0].source.startswith("posthog/")
    assert again.cached and again.sources == first.sources


# ----- the API -----

@pytest.fixture
def services(index, tmp_path):
    return Services(HybridRetriever(index, reranker=WordReranker()), FakeChat("Renew it [1]."), FakeJudge(),
                    Pipeline(DocumentStore(tmp_path / "store"), index))


def test_api_lists_collections_and_answers_from_one(services):
    with TestClient(create_app(services, Settings())) as c:
        listed = {x["name"]: x for x in c.get("/v1/collections").json()}
        assert listed["demo"]["documents"] == 1 and listed["posthog"]["documents"] == 1
        body = c.post("/v1/ask", json={"question": "ERR_TUNNEL_4012 certificate expired", "collection": "demo"}).json()
        assert body["collection"] == "demo" and {p["source"] for p in body["retrieved"]} == {"vpn-setup.md"}
        assert c.post("/v1/ask", json={"question": "x", "collection": "acme"}).status_code == 422
        docs = c.get("/v1/documents", params={"collection": "posthog"}).json()
        assert [d["source"] for d in docs["documents"]] == ["posthog/handbook/it/vpn.md"]


def test_an_upload_goes_into_the_named_collection(services):
    with TestClient(create_app(services, Settings(admin_key="k"))) as c:
        r = c.post("/v1/ingest", params={"collection": "posthog"}, headers={"X-API-Key": "k"},
                   files={"files": ("expenses.md", b"# Expenses\n\nClaim expenses in Brex within 30 days.\n")})
        assert r.json()["files"][0]["status"] == "indexed"
        docs = c.get("/v1/documents", params={"collection": "posthog"}).json()
    assert "posthog/expenses.md" in [d["source"] for d in docs["documents"]]
