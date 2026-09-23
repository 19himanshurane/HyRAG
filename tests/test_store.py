import json
from pathlib import Path

import pytest

from hyrag.loader import DocumentStore


def write(path: Path, text: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    return path


@pytest.fixture
def store(tmp_path):
    return DocumentStore(tmp_path / "data")


def test_ingest_keeps_raw_and_processed(store, tmp_path):
    src = tmp_path / "src"
    doc = store.ingest_file(write(src / "guide.md", "# Guide\n\nHello."), root=src)
    assert (store.raw_dir / "guide.md").read_text(encoding="utf-8") == "# Guide\n\nHello."
    saved = json.loads((store.processed_dir / f"{doc.doc_id}.json").read_text(encoding="utf-8"))
    assert saved["source"] == "guide.md"


def test_reprocess_rebuilds_from_raw(store, tmp_path):
    src = tmp_path / "src"
    store.ingest_file(write(src / "a.md", "# A\n\nx"), root=src)
    store.ingest_file(write(src / "b.txt", "y"), root=src)
    assert sorted(d.source for d in store.reprocess_all()) == ["a.md", "b.txt"]


def test_root_is_required(store, tmp_path):
    with pytest.raises(TypeError):
        store.ingest_file(write(tmp_path / "a.md", "# A\n\nx"))


def test_same_name_in_two_folders_stays_two_documents(store, tmp_path):
    src = tmp_path / "src"
    hr = store.ingest_file(write(src / "hr" / "policy.md", "# HR\n\n24 vacation days."), root=src)
    it = store.ingest_file(write(src / "it" / "policy.md", "# IT\n\nRotate passwords."), root=src)
    assert hr.doc_id != it.doc_id
    assert (store.raw_dir / "hr" / "policy.md").read_text(encoding="utf-8").startswith("# HR")
    assert sorted(d.source for d in store.reprocess_all()) == ["hr/policy.md", "it/policy.md"]


def test_one_corrupt_file_does_not_stop_the_batch(store, tmp_path):
    src = tmp_path / "src"
    store.ingest_file(write(src / "a.md", "# A\n\nx"), root=src)
    write(store.raw_dir / "broken.pdf", b"%PDF-1.4 not really a pdf")
    docs = store.reprocess_all()
    assert [d.source for d in docs] == ["a.md"]
    assert list(store.failures) == ["broken.pdf"]


def test_failed_reparse_keeps_last_good_version(store, tmp_path):
    src = tmp_path / "src"
    doc = store.ingest_file(write(src / "a.md", "# A\n\nx"), root=src)
    write(store.raw_dir / "a.md", b"   ")  # raw file becomes unparseable (no text)
    store.reprocess_all()
    assert "a.md" in store.failures
    assert (store.processed_dir / f"{doc.doc_id}.json").exists()


def test_renamed_and_deleted_documents_leave_no_orphans(store, tmp_path):
    src = tmp_path / "src"
    for name in ("keep.md", "old-name.md", "gone.md"):
        store.ingest_file(write(src / name, f"# {name}\n\ntext"), root=src)
    (store.raw_dir / "old-name.md").rename(store.raw_dir / "new-name.md")
    (store.raw_dir / "gone.md").unlink()
    store.reprocess_all()
    sources = sorted(json.loads(p.read_text(encoding="utf-8"))["source"] for p in store.processed_dir.glob("*.json"))
    assert sources == ["keep.md", "new-name.md"]


def test_manifest_decides_what_is_a_document(tmp_path):
    from hyrag.loader import list_corpus_files
    write(tmp_path / "docs" / "a.md", "# A\n\nx")
    write(tmp_path / "README.md", "licence notes, not content")
    assert [p.name for p in list_corpus_files(tmp_path)] == ["README.md", "a.md"]  # no manifest: everything
    write(tmp_path / "manifest.json", json.dumps([{"path": "docs/a.md"}]))
    assert list_corpus_files(tmp_path) == [tmp_path / "docs" / "a.md"]
    write(tmp_path / "manifest.json", json.dumps([{"path": "docs/missing.md"}]))
    with pytest.raises(FileNotFoundError, match="missing.md"):
        list_corpus_files(tmp_path)


def test_real_corpus_manifest_excludes_readme():
    from hyrag.loader import list_corpus_files
    from .conftest import CORPUS
    files = list_corpus_files(CORPUS)
    assert len(files) == 12 and not any(p.name == "README.md" for p in files)


def test_unchanged_file_is_not_parsed_again(store, tmp_path, monkeypatch):
    import hyrag.loader as loader
    src = tmp_path / "src"
    f = write(src / "a.md", "# A\n\nfirst version")
    first = store.ingest_file(f, root=src)

    calls = []
    real = loader.load_document
    monkeypatch.setattr(loader, "load_document", lambda *a: calls.append(a[0]) or real(*a))
    assert store.ingest_file(f, root=src) == first            # same bytes: served from the stored copy
    assert DocumentStore(store.raw_dir.parent).ingest_file(f, root=src) == first  # survives a restart
    assert calls == []

    write(f, "# A\n\nsecond version")                         # changed bytes: parsed again
    assert "second version" in store.ingest_file(f, root=src).sections[0].text
    monkeypatch.setattr(loader, "PARSER_VERSION", "future")   # parser upgraded: parsed again
    store.ingest_file(f, root=src)
    assert calls == ["a.md", "a.md"]


def test_reprocess_force_reparses_everything(store, tmp_path, monkeypatch):
    import hyrag.loader as loader
    src = tmp_path / "src"
    store.ingest_file(write(src / "a.md", "# A\n\nx"), root=src)
    calls = []
    real = loader.load_document
    monkeypatch.setattr(loader, "load_document", lambda *a: calls.append(a[0]) or real(*a))
    store.reprocess_all()
    assert calls == []
    store.reprocess_all(force=True)
    assert calls == ["a.md"]


def test_ingest_many_matches_one_by_one_and_isolates_failures(tmp_path):
    from .conftest import body_lines, make_pdf
    src = tmp_path / "src"
    pages = [[(15, 20, f"{i}. Chapter {i}", "B", 11), *body_lines(30, 8)] for i in range(1, 12)]  # 11 pages -> 2 tasks
    write(src / "manual.pdf", make_pdf(pages, header="Nimbus - Confidential", footer=True))
    write(src / "guide.md", "# Guide\n\nHello.")
    write(src / "broken.pdf", b"%PDF-1.4 not a pdf")
    files = sorted(src.iterdir(), key=lambda p: p.name)

    one_by_one = DocumentStore(tmp_path / "a")
    expected = [one_by_one.ingest_file(f, root=src) for f in files if f.name != "broken.pdf"]
    many = DocumentStore(tmp_path / "b")
    got = many.ingest_many(files, root=src, workers=2)
    assert got == expected
    assert list(many.failures) == ["broken.pdf"]
    assert [s.page for s in got[1].sections][-1] == 11


def test_ingest_many_reuses_unchanged_files(tmp_path, caplog):
    import logging
    src = tmp_path / "src"
    files = [write(src / "a.md", "# A\n\nx"), write(src / "b.txt", "y")]
    store = DocumentStore(tmp_path / "d")
    first = store.ingest_many(files, root=src, workers=1)
    caplog.set_level(logging.INFO, logger="hyrag")
    assert store.ingest_many(files, root=src, workers=1) == first
    assert any("2 files, 0 parsed, 2 unchanged (reused)" in r.getMessage() for r in caplog.records)


def test_paths_cannot_escape_the_data_folder(store, tmp_path):
    victim = write(tmp_path / "outside.txt", "must survive")
    for bad in ("../../outside.txt", str(victim), ""):
        with pytest.raises(ValueError, match="unsafe"):
            store.delete(bad)
    assert victim.exists()


def test_oversized_inputs_are_refused(monkeypatch):
    import hyrag.loader as loader
    from .conftest import body_lines, make_pdf
    monkeypatch.setattr(loader, "MAX_FILE_BYTES", 100)
    with pytest.raises(ValueError, match="limit"):
        loader.load_document("big.md", b"# x\n\n" + b"a" * 200)
    monkeypatch.setattr(loader, "MAX_FILE_BYTES", 10**9)
    monkeypatch.setattr(loader, "MAX_PDF_PAGES", 2)
    with pytest.raises(ValueError, match="3 pages"):
        loader.load_document("long.pdf", make_pdf([body_lines(20, 2)] * 3))


def test_ingestion_is_logged(store, tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="hyrag")
    src = tmp_path / "src"
    f = write(src / "a.md", "# A\n\nx")
    store.ingest_file(f, root=src)
    store.ingest_file(f, root=src)
    write(store.raw_dir / "broken.pdf", b"%PDF-1.4 not a pdf")
    store.reprocess_all()
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("ingested a.md: 1 sections, parsed") for m in messages)
    assert any(m.startswith("ingested a.md: 1 sections, unchanged (reused)") for m in messages)
    assert any(m.startswith("skipped broken.pdf: PdfminerException") for m in messages)


def test_oversized_files_are_refused_without_being_read(store, tmp_path, monkeypatch):
    import hyrag.loader as loader
    src = tmp_path / "src"
    big = write(src / "big.md", "# Big\n\n" + "x" * 500)
    ok = write(src / "ok.md", "# Ok\n\nfine")
    monkeypatch.setattr(loader, "MAX_FILE_BYTES", 100)
    real_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail(f"read {self.name}") if self.name == "big.md"
                        else real_read(self))
    with pytest.raises(ValueError, match="limit"):
        store.ingest_file(big, root=src)
    docs = store.ingest_many([big, ok], root=src, workers=1)
    assert [d.source for d in docs] == ["ok.md"] and "big.md" in store.failures


def test_pdf_page_range_reads_the_same_from_path_or_bytes(tmp_path):
    from hyrag.loader import read_pdf_page_range
    from .conftest import body_lines, make_pdf
    pdf = write(tmp_path / "m.pdf", make_pdf([body_lines(20, 3) for _ in range(4)]))
    assert read_pdf_page_range(str(pdf), 2, 3) == read_pdf_page_range(pdf.read_bytes(), 2, 3)
    assert {l.page for l in read_pdf_page_range(str(pdf), 2, 3)} == {2, 3}


def test_delete_removes_raw_and_processed(store, tmp_path):
    src = tmp_path / "src"
    doc = store.ingest_file(write(src / "a.md", "# A\n\nx"), root=src)
    store.delete("a.md")
    assert not (store.raw_dir / "a.md").exists()
    assert not (store.processed_dir / f"{doc.doc_id}.json").exists()
