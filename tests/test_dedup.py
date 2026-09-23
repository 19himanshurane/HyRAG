import pytest

from hyrag.chunking import chunk_structure
from hyrag.dedup import NearDuplicateIndex, fact_tokens
from hyrag.index import ChunkIndex
from hyrag.loader import load_document

from .test_index import CountingEmbedder

# Real boilerplate from our corpus: printed word for word in BOTH NIST publications.
NOTICE = ("This publication has been developed by NIST in accordance with its statutory responsibilities under the "
          "Federal Information Security Modernization Act (FISMA) of 2014, 44 U.S.C. 3551 et seq., Public Law "
          "(P.L.) 113-283. NIST is responsible for developing information security standards and guidelines.")
BGWRITER = ("bgwriter_flush_after (integer): Whenever more than this amount of data has been written by the background "
            "writer, attempt to force the OS to issue these writes to the underlying storage. Doing so will limit "
            "the amount of dirty data in the kernel's page cache.")
BACKEND = BGWRITER.replace("bgwriter_flush_after", "backend_flush_after").replace(
    "by the background writer", "by a single backend")


def md(name, body):
    doc = load_document(name, body.encode())
    return doc, chunk_structure(doc)


@pytest.fixture
def index(tmp_path):
    return ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "index")


# ----- the rules -----

def test_identical_text_is_a_duplicate_and_only_long_texts_tolerate_a_changed_word():
    near = NearDuplicateIndex()
    near.add("a", NOTICE)
    assert near.find(NOTICE) == ("a", 1.0)
    # One changed word breaks ~3 of the ~50 three-word sequences of a 50-word paragraph: similarity ~0.89,
    # below 0.9, so it is kept as distinct (the safe direction: we may keep a copy, never drop a fact).
    assert near.find(NOTICE.replace("developing", "writing")) is None
    long_text = " ".join([NOTICE, BGWRITER.replace("bgwriter_flush_after (integer): ", ""), NOTICE.lower()])
    near.add("long", long_text)
    cid, sim = near.find(long_text.replace("developing", "writing", 1))
    assert cid == "long" and 0.9 <= sim < 1.0


def test_different_settings_with_near_identical_descriptions_are_not_duplicates():
    near = NearDuplicateIndex(threshold=0.7)  # even with a loose threshold...
    near.add("bg", BGWRITER)
    assert fact_tokens(BGWRITER) != fact_tokens(BACKEND)
    assert near.find(BACKEND) is None  # ...a different identifier means a different fact


def test_a_changed_number_is_not_a_duplicate():
    near = NearDuplicateIndex()
    near.add("a", NOTICE)
    assert near.find(NOTICE.replace("of 2014", "of 2018")) is None


def test_short_bodies_are_never_deduplicated():
    near = NearDuplicateIndex()
    near.add("term1", "See cybersecurity incident.")  # glossary: the heading (the term) is what differs
    assert near.find("See cybersecurity incident.") is None


# ----- inside the index -----

def test_copy_in_a_second_document_is_skipped_recorded_and_not_embedded(index):
    first = index.index_document(*md("nist-a.md", f"# Authority\n\n{NOTICE}"))
    before = len(index.embedder.texts)
    second = index.index_document(*md("nist-b.md", f"# Authority\n\n{NOTICE}\n\n## Scope\n\n{BGWRITER}"))
    assert (first.added, second.added, second.duplicates) == (1, 1, 1)
    assert len(index.embedder.texts) - before == 1  # only the new, non-duplicate chunk was embedded
    kept = index.search_sparse("statutory responsibilities FISMA", k=1)[0].chunk
    assert kept.source == "nist-a.md"
    assert [c.source for c in index.also_appears_in(kept.chunk_id)] == ["nist-b.md"]
    assert not any(index.check_sync().values())


def test_deleting_the_original_promotes_the_copy(index):
    doc_a, chunks_a = md("nist-a.md", f"# Authority\n\n{NOTICE}")
    index.index_document(doc_a, chunks_a)
    index.index_document(*md("nist-b.md", f"# Authority\n\n{NOTICE}"))
    assert index.count() == 1 and index.duplicate_count() == 1
    index.remove_document(doc_a.doc_id)
    hits = index.search_sparse("statutory responsibilities FISMA", k=1)
    assert hits and hits[0].chunk.source == "nist-b.md"  # the text is still findable, now from document b
    assert index.duplicate_count() == 0 and not any(index.check_sync().values())


def test_editing_the_original_away_promotes_the_copy(index):
    index.index_document(*md("nist-a.md", f"# Authority\n\n{NOTICE}"))
    index.index_document(*md("nist-b.md", f"# Authority\n\n{NOTICE}"))
    index.index_document(*md("nist-a.md", f"# Authority\n\n{BGWRITER}"))  # a's notice replaced by other text
    sources = {h.chunk.source for h in index.search_sparse("statutory responsibilities FISMA", k=3)}
    assert "nist-b.md" in sources and not any(index.check_sync().values())


def test_reindexing_is_idempotent(index):
    index.index_document(*md("nist-a.md", f"# Authority\n\n{NOTICE}"))
    doc_b, chunks_b = md("nist-b.md", f"# Authority\n\n{NOTICE}")
    index.index_document(doc_b, chunks_b)
    before = len(index.embedder.texts)
    again = index.index_document(doc_b, chunks_b)
    assert (again.added, again.duplicates) == (0, 1) and index.duplicate_count() == 1
    assert len(index.embedder.texts) == before


def test_duplicates_are_still_caught_after_a_restart(tmp_path):  # audit H6
    ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i").index_document(*md("nist-a.md", f"# Authority\n\n{NOTICE}"))
    reopened = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i")  # duplicate lookup rebuilt from the table
    result = reopened.index_document(*md("nist-b.md", f"# Authority\n\n{NOTICE}"))
    assert (result.added, result.duplicates) == (0, 1) and reopened.embedder.texts == []


def test_a_repeat_inside_one_document_is_skipped(index):
    result = index.index_document(*md("guide.md", f"# A\n\n{NOTICE}\n\n## B\n\n{NOTICE}"))
    assert (result.added, result.duplicates) == (1, 1)


def test_dedup_can_be_switched_off(tmp_path):
    index = ChunkIndex(CountingEmbedder(), data_dir=tmp_path / "i", dedup=False)
    index.index_document(*md("nist-a.md", f"# Authority\n\n{NOTICE}"))
    assert index.index_document(*md("nist-b.md", f"# Authority\n\n{NOTICE}")).duplicates == 0
    assert index.count() == 2


def test_repair_promotes_copies_whose_original_vanished(index):
    doc_a, chunks_a = md("nist-a.md", f"# Authority\n\n{NOTICE}")
    index.index_document(doc_a, chunks_a)
    index.index_document(*md("nist-b.md", f"# Authority\n\n{NOTICE}"))
    with index.db:  # simulate a crash that lost the original row but not the duplicate record
        index.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_a.doc_id,))
    index.collection.delete(ids=[chunks_a[0].chunk_id])
    index._near = None
    assert index.check_sync()["duplicates_of_missing_chunks"] == [chunks_a[0].chunk_id]
    index.repair()
    assert not any(index.check_sync().values())
    assert index.search_sparse("statutory responsibilities FISMA", k=1)[0].chunk.source == "nist-b.md"
