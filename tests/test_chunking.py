import pytest

from hyrag.chunking import chunk_fixed, chunk_structure, split_recursive
from hyrag.loader import Document, Section


def doc_of(*sections: tuple[str, str]) -> Document:
    return Document(doc_id="d1", source="d.md", file_type=".md",
                    sections=[Section(text=t, heading=h) for h, t in sections])


def test_fixed_windows_overlap():
    doc = doc_of(("H", "abcdefghij" * 30))
    chunks = chunk_fixed(doc, size=100, overlap=20)
    assert all(len(c.text) <= 100 for c in chunks)
    assert chunks[0].text[-20:] == chunks[1].text[:20]


def test_structure_chunks_never_cross_sections():
    doc = doc_of(("A", "alpha " * 50), ("B", "beta " * 50))
    for c in chunk_structure(doc, size=120, overlap=20):
        assert ("alpha" in c.text) != ("beta" in c.text)
        assert c.heading == ("A" if "alpha" in c.text else "B")


def test_split_prefers_paragraph_then_sentence_boundaries():
    text = "First paragraph sentence one. Sentence two.\n\nSecond paragraph here."
    assert split_recursive(text, 50, 10) == ["First paragraph sentence one. Sentence two.\n\n",
                                             "Second paragraph here."]


def test_hard_cut_fallback_has_no_redundant_tail():
    word = "Supercalifragilisticexpialidocious" * 3
    pieces = split_recursive(word, 60, 15)
    assert len(pieces) == 2
    assert "".join(split_recursive(word, 60, 0)) == word


@pytest.mark.parametrize("fn", [chunk_fixed, chunk_structure])
def test_overlap_must_be_smaller_than_size(fn):
    with pytest.raises(ValueError, match="overlap must be smaller"):
        fn(doc_of(("H", "x" * 500)), size=10, overlap=20)


def test_text_for_search_prepends_heading():
    c = chunk_structure(doc_of(("VPN > Error ERR_TUNNEL_4012", "Renew the certificate.")))[0]
    assert c.text_for_search() == "VPN > Error ERR_TUNNEL_4012\n\nRenew the certificate."
    assert c.text == "Renew the certificate."


@pytest.mark.parametrize("fn", [chunk_fixed, chunk_structure])
def test_chunk_ids_are_unique_even_for_repeated_text(fn):
    doc = doc_of(("H", "same line\n\n" * 3 + "tail"), ("H", "same line"))
    ids = [c.chunk_id for c in fn(doc, size=12, overlap=2)]
    assert len(ids) == len(set(ids))


def test_editing_one_section_keeps_other_chunk_ids():
    before = doc_of(("A", "alpha " * 40), ("B", "beta " * 40), ("C", "gamma " * 40))
    after = doc_of(("A", "alpha " * 40), ("B", "beta " * 40 + "and a brand new sentence. " * 10), ("C", "gamma " * 40))
    ids_before = {c.chunk_id: c.text for c in chunk_structure(before, size=120, overlap=20)}
    ids_after = {c.chunk_id: c.text for c in chunk_structure(after, size=120, overlap=20)}
    kept = {k for k in ids_before if k in ids_after}
    assert all(ids_before[k] == ids_after[k] for k in kept)            # a kept id always means the same text
    assert {ids_before[k] for k in kept} >= {c.text for c in chunk_structure(before, size=120, overlap=20)
                                             if "beta" not in c.text}   # sections A and C kept every id


def test_changing_the_heading_changes_the_id():
    a = chunk_structure(doc_of(("Old title", "body text")))[0]
    b = chunk_structure(doc_of(("New title", "body text")))[0]
    assert a.chunk_id != b.chunk_id  # what gets embedded changed, so it must be re-embedded


def test_large_unstructured_text_chunks_quickly():
    import time
    t0 = time.perf_counter()
    pieces = split_recursive("lorem ipsum dolor sit amet " * 40000, 800, 120)
    assert time.perf_counter() - t0 < 5
    assert all(len(p) <= 800 for p in pieces)
