"""Generates security-handbook.pdf so we have a real PDF to test the loader with."""
from pathlib import Path

from fpdf import FPDF

pages = [
    ("Security Handbook", "Passwords must be at least 14 characters long.\nPasswords rotate every 180 days."),
    ("Laptop Security", "Laptops must use full-disk encryption.\nReport a lost laptop to security@nimbus.example within 1 hour."),
]

pdf = FPDF()
for title, body in pages:
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, body)
pdf.output(str(Path(__file__).parent / "security-handbook.pdf"))
