import hashlib
import re

import numpy as np
import pytest

from hyrag.chunking import chunk_semantic, chunk_structure, semantic_units
from hyrag.loader import Document, Section


class BagOfWordsEmbedder:
    """Deterministic offline stand-in: texts sharing CONTENT words get similar vectors. Short filler words
    ("the", "and", "so") are skipped; counting them made two deploy paragraphs look less alike than a
    password paragraph and a deploy paragraph, which no real embedding model would do."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]{4,}", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        return out / np.linalg.norm(out, axis=1, keepdims=True)


PASSWORD = "Passwords must be long, passwords must be unique, and password managers help users keep every password safe."
DEPLOY = "Deploys to production need a lead approval, deploys run on weekdays, and a failed deploy is rolled back quickly."


def doc_of(sections, file_type=".md"):
    return Document(doc_id="d1", source="d" + file_type, file_type=file_type,
                    sections=[Section(text=t, heading=h) for h, t in sections])


def test_pdf_units_rejoin_wrapped_lines_and_hyphenated_words():
    text = "Authenticators must be hardware-\nprotected and\nnon-exportable.\n\nA second paragraph."
    assert semantic_units(text, ".pdf", 800, min_unit=0) == [
        "Authenticators must be hardware-protected and non-exportable.",
        "A second paragraph.",
    ]


def test_html_lines_are_units_and_tiny_ones_are_grouped():
    rows = "\n".join(f"2350{i} | violation_{i}" for i in range(6))
    units = semantic_units(rows, ".html", 800, min_unit=60)
    assert all("\n" in u or len(u) >= 60 for u in units[:-1])  # tiny rows were grouped...
    assert "\n".join(units) == rows                            # ...without losing or reordering any


PASSWORD_PARAS = [
    "Passwords must be at least fifteen characters long, and a password manager should generate every password so users never reuse a password across services.",
    "Password rotation is not required on a schedule; a password is only changed when there is evidence the password was exposed, guessed, or phished by an attacker.",
    "Every stored password is salted and hashed, and the password verifier rate-limits failed password attempts so that online guessing of a password stays impractical.",
]
DEPLOY_PARAS = [
    "Production deploys need approval from one team lead, and every deploy runs through the staging pipeline first so a broken build never reaches customers.",
    "Deploys are only allowed on weekdays between ten and four, because a deploy late on a Friday leaves nobody around to roll back a failed release quickly.",
    "A failed deploy is rolled back automatically, and the deploy owner writes a short review so the next release avoids repeating the same deploy mistake.",
]


def test_cuts_where_the_topic_changes():
    doc = doc_of([("Policies", "\n\n".join(PASSWORD_PARAS + DEPLOY_PARAS))])
    semantic = chunk_semantic(doc, BagOfWordsEmbedder(), size=800, min_chunk=250)
    assert [("assword" in c.text, "eploy" in c.text) for c in semantic] == [(True, False), (False, True)]
    # the structure-aware baseline packs to the size limit and mixes both topics in one chunk
    assert any("assword" in c.text and "eploy" in c.text for c in chunk_structure(doc, size=800, overlap=0))


def test_a_short_closing_line_is_not_cut_off_as_its_own_chunk():
    body = "\n\n".join(PASSWORD_PARAS + DEPLOY_PARAS + ["Request Process"])
    chunks = chunk_semantic(doc_of([("A", body)]), BagOfWordsEmbedder(), size=800, min_chunk=250)
    assert min(c.char_count for c in chunks) >= 250
    assert chunks[-1].text.endswith("Request Process")


def test_tiny_documents_fall_back_to_size_only():
    # Fewer than 4 neighbour similarities: a percentile over them means nothing, so no topic cuts.
    doc = doc_of([("A", "\n\n".join([PASSWORD_PARAS[0], DEPLOY_PARAS[0]]))])
    assert len(chunk_semantic(doc, BagOfWordsEmbedder(), size=800, min_chunk=0)) == 1


def test_never_crosses_sections_and_never_repeats_text():
    doc = doc_of([("A", "\n\n".join([PASSWORD] * 3)), ("B", "\n\n".join([DEPLOY] * 3))])
    chunks = chunk_semantic(doc, BagOfWordsEmbedder())
    for heading in ("A", "B"):
        own = [c.text for c in chunks if c.heading == heading]
        section_text = doc.sections[0 if heading == "A" else 1].text
        assert "\n\n".join(own) == section_text  # no overlap, nothing lost
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_size_cap_holds_even_without_topic_changes():
    doc = doc_of([("A", "\n\n".join([PASSWORD] * 20))])
    chunks = chunk_semantic(doc, BagOfWordsEmbedder(), size=400)
    assert len(chunks) > 1 and all(c.char_count <= 400 for c in chunks)


def test_one_embedding_call_per_document():
    emb = BagOfWordsEmbedder()
    chunk_semantic(doc_of([("A", PASSWORD), ("B", DEPLOY), ("C", PASSWORD)]), emb)
    assert emb.calls == 1  # all units batched together (the real embedder then batches by 32 and caches)


@pytest.mark.parametrize("file_type", [".pdf", ".html", ".md"])
def test_empty_sections_are_fine(file_type):
    assert chunk_semantic(doc_of([], file_type), BagOfWordsEmbedder()) == []
