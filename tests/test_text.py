import pytest

from hyrag.loader import clean, load_document


def test_cp1252_file_decodes_without_replacement_chars():
    doc = load_document("w.txt", "Don’t share passwords – ever.".encode("cp1252"))
    assert doc.sections[0].text == "Don’t share passwords – ever."


def test_clean_squeezes_spaces_but_keeps_indentation():
    assert clean("a   b\n    indented   x  \n\n\n\nc") == "a b\n    indented x\n\nc"


def test_txt_is_one_section_titled_by_filename():
    doc = load_document("leave-policy.txt", b"Leave\n\n24 days.")
    assert [(s.heading, s.text) for s in doc.sections] == [("leave-policy", "Leave\n\n24 days.")]


def test_unsupported_and_empty_files_are_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        load_document("x.docx", b"...")
    with pytest.raises(ValueError, match="No text"):
        load_document("empty.md", b"   \n")


def test_doc_id_depends_on_path_not_just_name():
    a = load_document("docs/README.md", b"# A\n\nalpha")
    b = load_document("api/README.md", b"# B\n\nbeta")
    assert a.doc_id != b.doc_id
