"""Visitor uploads: the in-memory store on its own, then through the HTTP API with a real shared index next to it."""
import numpy as np
import pytest

from hyrag.uploads import NO_TEXT, UploadPolicy, UploadRejected, UploadStore, display_name

from .conftest import body_lines, make_pdf
from .test_api import client, services  # noqa: F401  (the fixture: real index + fake models)
from .test_index import CountingEmbedder

BIKES = "Where do employees park the company bicycles overnight?"  # the shared index has no answer to this


def parking_pdf(pages: int = 1) -> bytes:
    return make_pdf([[(15, 20, "Parking Policy", "B", 16)]
                     + body_lines(32, 8, "Company bicycles are parked overnight in the basement cage, row {i}.")
                     for _ in range(pages)])


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def store(clock=None, **policy) -> UploadStore:
    return UploadStore(CountingEmbedder(), UploadPolicy(**policy), clock=clock or Clock())


def refusal(s: UploadStore, name: str, data: bytes, visitor: str = "v") -> UploadRejected:
    with pytest.raises(UploadRejected) as caught:
        s.add(visitor, name, data)
    return caught.value


# ----- the store -----

def test_an_uploaded_pdf_is_searchable_by_meaning_and_by_keywords():
    s = store()
    held = s.add("v", "parking.pdf", parking_pdf(pages=2))
    assert held.pages == 2 and held.characters > 200 and held.index.count() >= 1
    assert s.get(held.upload_id) is held and len(held.upload_id) >= 20
    for search in (held.index.search_dense, held.index.search_sparse):
        hits = search("bicycles parked overnight basement", k=3)
        assert hits and hits[0].chunk.source == "parking.pdf" and hits[0].chunk.page in (1, 2)
        assert hits == sorted(hits, key=lambda h: -h.score)
    assert held.index.search_sparse("zebra") == [] and held.index.search_dense("   ") == []
    with pytest.raises(ValueError):
        held.index.search_sparse("bicycles", collection="demo")


def test_each_upload_has_its_own_version_so_answers_can_never_be_shared():
    s = store()
    a, b = s.add("v", "a.pdf", parking_pdf()), s.add("v", "a.pdf", parking_pdf())
    assert a.index.version() != b.index.version() and a.upload_id != b.upload_id


@pytest.mark.parametrize("name, data, status, says", [
    ("run.exe", b"MZ", 415, "Only .pdf"),
    ("notes.html", b"<p>hello</p>", 415, "Only .pdf"),
    ("big.txt", b"x" * 2049, 413, "larger than"),
    ("broken.pdf", b"not a pdf at all", 422, "couldn't be read as a PDF"),
    ("scan.pdf", make_pdf([[], []]), 422, "scanned PDF"),
    ("stamp.pdf", make_pdf([[(15, 20, "COPY", "", 11)]]), 422, "Only 4 characters"),
    ("long.pdf", parking_pdf(pages=3), 422, "3 pages; the limit is 2"),
    ("long.txt", b"The quick brown fox. " * 90, 422, "passages and the limit is 1"),
])
def test_a_file_that_cannot_be_used_is_refused_with_the_reason(name, data, status, says):
    s = store(max_bytes=2048 if name == "big.txt" else 5 * 1024 * 1024, max_pages=2, max_chunks=1)
    e = refusal(s, name, data)
    assert e.status == status and says in str(e)
    assert s.count() == 0 and s.embedder.texts == []  # nothing kept, nothing sent for embedding


def test_no_text_message_explains_scans():
    assert "images" in NO_TEXT and str(refusal(store(), "scan.pdf", make_pdf([[]]))) == NO_TEXT


def test_an_upload_expires():
    clock = Clock()
    s = store(clock, ttl_seconds=600)
    held = s.add("v", "parking.pdf", parking_pdf())
    clock.now += 599
    assert s.get(held.upload_id) is held and s.seconds_left(held) == 1
    clock.now += 1
    assert s.get(held.upload_id) is None and s.count() == 0


def test_remove_drops_it_at_once():
    s = store()
    held = s.add("v", "parking.pdf", parking_pdf())
    assert s.remove(held.upload_id) and s.get(held.upload_id) is None and not s.remove(held.upload_id)


def test_upload_limits_per_visitor_and_per_day_count_only_uploads_that_were_taken():
    clock = Clock()
    s = store(clock, per_visitor_per_hour=2, per_day=3, max_live_per_visitor=5)
    refusal(s, "broken.pdf", b"nope")  # refused: costs the visitor nothing
    s.add("v", "a.pdf", parking_pdf())
    s.add("v", "b.pdf", parking_pdf())
    e = refusal(s, "c.pdf", parking_pdf())
    assert e.status == 429 and "2 uploads per visitor per hour" in str(e) and 0 < e.retry_after <= 3601
    s.add("w", "c.pdf", parking_pdf())  # someone else still can, until the day's total is used
    e = refusal(s, "d.pdf", parking_pdf(), visitor="x")
    assert e.status == 429 and "for today" in str(e)
    clock.now += 86401
    s.add("v", "e.pdf", parking_pdf())


def test_room_is_made_by_dropping_the_visitors_oldest_then_anyones_oldest():
    s = store(max_live=3, max_live_per_visitor=2, per_visitor_per_hour=0, per_day=0)
    a1, a2 = s.add("a", "1.pdf", parking_pdf()), s.add("a", "2.pdf", parking_pdf())
    a3 = s.add("a", "3.pdf", parking_pdf())
    assert s.get(a1.upload_id) is None and s.get(a2.upload_id) and s.get(a3.upload_id)
    b1, c1 = s.add("b", "1.pdf", parking_pdf()), s.add("c", "1.pdf", parking_pdf())
    assert s.count() == 3 and s.get(a2.upload_id) is None and s.get(b1.upload_id) and s.get(c1.upload_id)


def test_an_embedding_outage_is_a_503_and_keeps_nothing():
    class Down:
        def embed(self, texts):
            raise RuntimeError("embedding request failed after 5 attempts")

    s = UploadStore(Down(), UploadPolicy(per_visitor_per_hour=1))
    e = refusal(s, "parking.pdf", parking_pdf())
    assert e.status == 503 and e.retry_after and "failed after" not in str(e) and s.count() == 0
    s.embedder = CountingEmbedder()
    assert s.add("v", "parking.pdf", parking_pdf())  # the failed attempt didn't use up the visitor's one upload


def test_a_second_upload_waits_for_the_first_and_is_told_to_retry(monkeypatch):
    monkeypatch.setattr("hyrag.uploads.BUSY_WAIT_SECONDS", 0.01)
    s = store()
    with s._busy:  # another file is being read
        e = refusal(s, "parking.pdf", parking_pdf())
    assert e.status == 503 and e.retry_after == 15


@pytest.mark.parametrize("given, shown", [
    ("My Résumé (final).PDF", "My Résumé _final_.pdf"),
    ("C:\\Users\\me\\Desktop\\plan.pdf", "plan.pdf"),
    ("../../etc/passwd.txt", "passwd.txt"),
    ('x" section="y"><b>.pdf', "x_ section__y___b_.pdf"),  # can't close the prompt's passage tag or add markup
    (".pdf", "document.pdf"),
    ("", "document"),
])
def test_file_names_are_reduced_to_safe_characters(given, shown):
    assert display_name(given) == shown


def test_a_query_that_embeds_to_nothing_finds_nothing_instead_of_ranking_by_nan():
    held = store().add("v", "parking.pdf", parking_pdf())
    held.index.embedder = type("Zero", (), {"embed": lambda self, texts: np.zeros((1, 256), dtype=np.float32)})()
    assert held.index.search_dense("bicycles") == []


# ----- through the API -----

@pytest.fixture
def uploading(services):  # noqa: F811
    services.uploads = UploadStore(CountingEmbedder())
    return services


def send(c, name="parking.pdf", data=None, visitor="v1"):
    return c.post("/v1/uploads", files={"file": (name, data if data is not None else parking_pdf(), "application/pdf")},
                  headers={"X-Visitor-Id": visitor})


def test_upload_then_ask_answers_from_that_document(uploading):
    with client(uploading) as c:
        assert c.post("/v1/ask", json={"question": BIKES}).json()["status"] == "not_found"
        info = send(c).json()
        assert info["filename"] == "parking.pdf" and info["pages"] == 1 and info["chunks"] >= 1
        assert 3590 <= info["expires_in_seconds"] <= 3600
        body = c.post("/v1/ask", json={"question": BIKES, "upload_id": info["upload_id"]}).json()
    assert body["status"] == "answered" and body["sources"][0]["source"] == "parking.pdf"
    assert {p["source"] for p in body["retrieved"]} == {"parking.pdf"} and body["collection"] is None


def test_an_upload_never_reaches_the_shared_index_or_the_answer_cache(uploading):
    from hyrag.answer import AnswerCache

    uploading.cache = AnswerCache()
    version, chunks = uploading.index.version(), uploading.index.count()
    with client(uploading) as c:
        info = send(c).json()
        for _ in range(2):
            body = c.post("/v1/ask", json={"question": BIKES, "upload_id": info["upload_id"]}).json()
            assert body["status"] == "answered" and not body["cached"]
        assert "parking.pdf" not in [d["source"] for d in c.get("/v1/documents").json()["documents"]]
        assert c.post("/v1/ask", json={"question": BIKES}).json()["status"] == "not_found"  # others can't see it
    assert uploading.index.version() == version and uploading.index.count() == chunks
    assert all("parking" not in str(response) for _, response in uploading.cache._items.values())


def test_two_uploads_are_answered_separately(uploading):
    printers = make_pdf([[(15, 20, "Printers", "B", 16)] + body_lines(
        32, 8, "The office printer is set up from the Settings app under Devices, step {i}.")])
    with client(uploading) as c:
        bikes, prints = send(c).json()["upload_id"], send(c, "printers.pdf", printers, "v2").json()["upload_id"]
        body = c.post("/v1/ask", json={"question": BIKES, "upload_id": prints}).json()
        assert {p["source"] for p in body["retrieved"]} == {"printers.pdf"} and "parking.pdf" not in str(body)
        body = c.post("/v1/ask", json={"question": BIKES, "upload_id": bikes}).json()
        assert {p["source"] for p in body["retrieved"]} == {"parking.pdf"} and "printers.pdf" not in str(body)


def test_the_relevance_gate_is_skipped_only_when_the_writer_sees_the_whole_document(uploading):
    unrelated = "Which certifications does the author hold?"  # the fake reranker scores every passage very low
    long_doc = "\n\n".join(f"# Rule {i}\n\n" + f"Company bicycles are parked overnight in cage {i}. " * 12
                           for i in range(8)).encode()
    with client(uploading) as c:
        short = send(c).json()
        body = c.post("/v1/ask", json={"question": unrelated, "upload_id": short["upload_id"]}).json()
        assert short["chunks"] <= 5 and body["code"] != "not_found.no_relevant_documents"
        assert len(uploading.writer.calls) == 1  # the writer was asked and decides for itself
        long = send(c, "rules.md", long_doc).json()
        body = c.post("/v1/ask", json={"question": unrelated, "upload_id": long["upload_id"]}).json()
        assert long["chunks"] > 5 and body["code"] == "not_found.no_relevant_documents"
        assert len(uploading.writer.calls) == 1  # gated as usual: the model was not asked


def test_uploads_are_off_unless_configured(services):  # noqa: F811
    with client(services) as c:
        assert c.get("/v1/uploads/limits").json()["enabled"] is False
        assert send(c).status_code == 403
        assert c.post("/v1/ask", json={"question": BIKES, "upload_id": "anything"}).status_code == 404
        assert c.delete("/v1/uploads/anything").status_code == 204


def test_limits_endpoint_says_what_is_accepted(uploading):
    with client(uploading) as c:
        assert c.get("/v1/uploads/limits").json() == {
            "enabled": True, "max_mb": 5.0, "max_pages": 50, "minutes_kept": 60, "types": [".pdf", ".md", ".txt"]}


def test_unknown_expired_and_deleted_uploads_are_404(uploading):
    clock = Clock()
    uploading.uploads = UploadStore(CountingEmbedder(), clock=clock)
    with client(uploading) as c:
        r = c.post("/v1/ask", json={"question": BIKES, "upload_id": "no-such-upload"})
        assert r.status_code == 404 and "Upload it again" in r.json()["detail"]
        first, second = send(c).json()["upload_id"], send(c).json()["upload_id"]
        assert c.delete(f"/v1/uploads/{first}").status_code == 204
        assert c.post("/v1/ask", json={"question": BIKES, "upload_id": first}).status_code == 404
        clock.now += 3601
        assert c.post("/v1/ask", json={"question": BIKES, "upload_id": second}).status_code == 404
    assert uploading.writer.calls == []  # none of these reached the model


def test_a_collection_and_an_upload_together_are_rejected(uploading):
    with client(uploading) as c:
        upload_id = send(c).json()["upload_id"]
        r = c.post("/v1/ask", json={"question": BIKES, "upload_id": upload_id, "collection": "demo"})
    assert r.status_code == 422


def test_refusals_keep_their_status_and_reason_over_http(uploading):
    uploading.uploads = UploadStore(CountingEmbedder(), UploadPolicy(max_bytes=4096, per_visitor_per_hour=1))
    with client(uploading) as c:
        assert send(c, "run.exe", b"MZ").status_code == 415
        r = send(c, "big.pdf", b"x" * 5000)
        assert r.status_code == 413 and "larger than" in r.json()["detail"]
        r = send(c, "scan.pdf", make_pdf([[]]))
        assert r.status_code == 422 and r.json()["detail"] == NO_TEXT
        assert send(c).status_code == 200
        r = send(c)
        assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0


def test_questions_about_an_upload_count_against_the_question_limit(uploading):
    with client(uploading, visitor_limit=1) as c:
        upload_id = send(c).json()["upload_id"]
        ask = {"question": BIKES, "upload_id": upload_id}
        assert c.post("/v1/ask", json=ask, headers={"X-Visitor-Id": "v1"}).status_code == 200
        assert c.post("/v1/ask", json=ask, headers={"X-Visitor-Id": "v1"}).status_code == 429


def test_uploads_need_the_api_key_when_one_is_configured(uploading):
    with client(uploading, api_key="s3cret") as c:
        assert send(c).status_code == 401 and c.get("/v1/uploads/limits").status_code == 401
        assert c.delete("/v1/uploads/x").status_code == 401
