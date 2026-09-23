import re

from hyrag.loader import load_document

from .conftest import body_lines, make_pdf


def headings(doc):
    return [s.heading for s in doc.sections]


def test_running_header_and_page_footer_are_removed():
    pages = [[(15, 30, "Expense Policy", "B", 14), *body_lines(45, 5)] for _ in range(3)]
    doc = load_document("e.pdf", make_pdf(pages, header="Nimbus Inc. - Confidential", footer=True))
    text = "\n".join(s.text for s in doc.sections)
    assert "Confidential" not in text and "Page " not in text


def test_numbered_bold_heading_levels():
    page = [(15, 20, "3. Requirements", "B", 11), *body_lines(28, 2),
            (15, 45, "3.1. Passwords", "B", 11), *body_lines(53, 2),
            (15, 70, "3.1.1. Length", "B", 11), *body_lines(78, 6)]
    doc = load_document("n.pdf", make_pdf([page]))
    assert headings(doc) == ["n > 3. Requirements", "n > 3. Requirements > 3.1. Passwords",
                             "n > 3. Requirements > 3.1. Passwords > 3.1.1. Length"]


def test_tab_separated_numbered_heading():
    page = [(15, 20, "3.1", "B", 11), (60, 20, "Passwords", "B", 11), *body_lines(30, 6)]
    assert headings(load_document("t.pdf", make_pdf([page]))) == ["t > 3.1 Passwords"]


def test_bold_table_header_row_and_caption_are_not_headings():
    page = [(15, 20, "2. Section", "B", 11),
            (15, 30, "Type", "B", 11), (80, 30, "Purpose", "B", 11), (140, 30, "Reference", "B", 11),
            (15, 40, "Fig. 1. Life cycle model", "B", 11), *body_lines(50, 6)]
    assert headings(load_document("tb.pdf", make_pdf([page]))) == ["tb > 2. Section"]


LONG = "Body text that runs across the whole width of the page so the right edge is known xx."


def test_wrapped_heading_is_merged_but_a_short_heading_does_not_swallow_the_next_line():
    page = [(15, 20, "4. Authenticator Lifecycle Management And Other Very Long Words", "B", 14),
            (15, 27, "Continued Title Line", "B", 14),                   # the title wrapped here
            *[(15, 40 + i * 6, LONG, "", 11) for i in range(4)],
            (15, 70, "Appendix C. Acronyms", "B", 11),                  # short: it did NOT wrap...
            (15, 76, "AAL", "B", 11),                                    # ...so this is a separate entry
            (15, 82, "Authentication Assurance Level", "", 11),
            *[(15, 90 + i * 6, LONG, "", 11) for i in range(4)]]
    hs = headings(load_document("w.pdf", make_pdf([page])))
    assert "w > 4. Authenticator Lifecycle Management And Other Very Long Words Continued Title Line" in hs
    assert "w > Appendix C. Acronyms > AAL" in hs


def test_bold_sentences_and_overlong_numbered_lines_are_not_headings():
    page = [(15, 20, "1. Scope", "B", 11),
            (15, 30, "This sentence is bold only for emphasis.", "B", 11),
            (15, 38, "A bold line with far too many words to ever be a real title here", "B", 11),
            (15, 46, "2. " + " ".join(["word"] * 22), "B", 11),
            *[(15, 56 + i * 6, LONG, "", 11) for i in range(6)]]
    assert headings(load_document("b.pdf", make_pdf([page]))) == ["b > 1. Scope"]


def test_standalone_glossary_nests_its_terms():
    page = [(15, 20, "Glossary", "B", 11), (15, 30, "authenticator", "B", 11), *body_lines(38, 1),
            (15, 48, "verifier", "B", 11), *body_lines(56, 6)]
    assert headings(load_document("g.pdf", make_pdf([page]))) == ["g > Glossary > authenticator",
                                                                  "g > Glossary > verifier"]


def test_sentence_cut_by_page_break_stays_in_one_section():
    p1 = [(15, 20, "1. Intro", "B", 11), *body_lines(30, 4), (15, 60, "this sentence continues on the", "", 11)]
    p2 = [(15, 20, "next page and ends here.", "", 11), *body_lines(30, 4)]
    doc = load_document("c.pdf", make_pdf([p1, p2]))
    joined = [s for s in doc.sections if "continues on the" in s.text][0]
    assert "next page and ends here." in joined.text and joined.page == 1


def test_peak_memory_does_not_grow_with_page_count():
    import tracemalloc

    def peak(n_pages: int) -> int:
        data = make_pdf([body_lines(20, 12) for _ in range(n_pages)])
        tracemalloc.start()
        load_document("m.pdf", data)
        _, p = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return p

    small, large = peak(3), peak(24)
    assert large < 2 * small, f"8x the pages used {large / small:.1f}x the memory"


def test_real_nist_63b_structure(nist_63b):
    hs = list(dict.fromkeys(headings(nist_63b)))
    root = "Digital Identity Guidelines: Authentication and Authenticator Management"
    assert all(h.startswith(root) for h in hs)
    assert len(hs) > 300
    text = "\n".join(s.text for s in nist_63b.sections)
    assert not re.search(r"^NIST SP 800-63B-4 Digital Identity", text, re.M)   # running header gone
    assert not re.search(r"\n[ivxlc]+$", text)                                 # roman page numbers gone
    p31 = [s for s in nist_63b.sections if s.page == 31]
    assert any(s.heading.endswith("3.1.3.1. Out-of-Band Authenticators") for s in p31)
    assert not any(re.search(r"Type of Secret|> Fig\. |List of Figures > ", h) for h in hs)
    assert any(h.endswith("Appendix D. Glossary > activation") for h in hs)


def test_real_nist_61r3_structure(nist_61r3):
    hs = list(dict.fromkeys(headings(nist_61r3)))
    assert any(h.endswith("2. Incident Response as Part of Cybersecurity Risk Management > "
                          "2.2. Incident Response Roles and Responsibilities") for h in hs)
    assert not any(h.split(" > ")[-1].isdigit() for h in hs)                  # page numbers aren't headings
