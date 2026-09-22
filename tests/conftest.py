from pathlib import Path

import pytest
from fpdf import FPDF

from hyrag.loader import load_document

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "corpus"


def make_pdf(pages: list[list[tuple]], header: str | None = None, footer: bool = False) -> bytes:
    """Build a small PDF. Each page is a list of (x_mm, y_mm, text, font_style, size) tuples."""

    class Doc(FPDF):
        def header(self):
            if header:
                self.set_font("Helvetica", size=8)
                self.set_xy(15, 8)
                self.cell(0, 5, header)

        def footer(self):
            if footer:
                self.set_font("Helvetica", size=8)
                self.set_xy(15, -12)
                self.cell(0, 5, f"Page {self.page_no()}")

    pdf = Doc()
    pdf.set_auto_page_break(False)
    for items in pages:
        pdf.add_page()
        for x, y, text, style, size in items:
            pdf.set_font("Helvetica", style, size)
            pdf.set_xy(x, y)
            pdf.cell(0, 6, text)
    return bytes(pdf.output())


def body_lines(start_y: float, n: int, text: str = "Body text line number {i} of the section.", x: float = 15):
    return [(x, start_y + i * 6, text.format(i=i), "", 11) for i in range(n)]


@pytest.fixture(scope="session")
def nist_63b():
    path = CORPUS / "security" / "nist-sp-800-63b-4-authentication.pdf"
    return load_document("security/nist-sp-800-63b-4-authentication.pdf", path.read_bytes())


@pytest.fixture(scope="session")
def nist_61r3():
    path = CORPUS / "security" / "nist-sp-800-61r3-incident-response.pdf"
    return load_document("security/nist-sp-800-61r3-incident-response.pdf", path.read_bytes())
