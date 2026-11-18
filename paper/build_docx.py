"""Build an APA7-style readable .docx from RESEARCHGATE_MANUSCRIPT.md."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, Twips


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "RESEARCHGATE_MANUSCRIPT.md"
OUT = ROOT / "Kaya_CrossAsset_Anomaly_Detection.docx"


def set_run_font(run, *, bold=False, italic=False, size=12):
    run.bold = bold
    run.italic = italic
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)


def style_paragraph(p, *, first_line=False, hanging=False, space_after=8, space_before=0, center=False):
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if first_line:
        pf.first_line_indent = Inches(0.5)
    if hanging:
        pf.left_indent = Inches(0.5)
        pf.first_line_indent = Inches(-0.5)


def add_runs_with_markup(paragraph, text: str, *, size=12):
    # Handle **bold**, *italic*, `code`, and plain text; strip simple LaTeX delimiters for Word.
    text = text.replace(r"\(", "").replace(r"\)", "").replace(r"\[", "").replace(r"\]", "")
    text = text.replace(r"\mathrm{rec}", "rec").replace(r"\mathrm{feat}", "feat")
    text = text.replace(r"\ell", "ℓ").replace(r"\cdot", "·")
    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            run = paragraph.add_run(text[pos : m.start()])
            set_run_font(run, size=size)
        token = m.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            set_run_font(run, bold=True, size=size)
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            set_run_font(run, size=size)
            run.font.name = "Courier New"
        else:
            run = paragraph.add_run(token[1:-1])
            set_run_font(run, italic=True, size=size)
        pos = m.end()
    if pos < len(text):
        run = paragraph.add_run(text[pos:])
        set_run_font(run, size=size)


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", c) for c in row):
            rows.append(row)
        i += 1
    return rows, i


def add_table(doc: Document, rows: list[list[str]]):
    if not rows:
        return
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    for r_idx, row in enumerate(rows):
        for c_idx, cell in enumerate(row):
            cell_p = table.rows[r_idx].cells[c_idx].paragraphs[0]
            cell_p.clear()
            add_runs_with_markup(cell_p, cell, size=10)
            for run in cell_p.runs:
                run.font.name = "Times New Roman"
            if r_idx == 0:
                for run in cell_p.runs:
                    run.bold = True
    doc.add_paragraph()


def build() -> Path:
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines()
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    in_code = False
    in_refs = False
    i = 0
    while i < len(lines):
        line = lines[i]
        raw = line.rstrip()

        if raw.startswith("```"):
            in_code = not in_code
            i += 1
            continue

        if in_code:
            p = doc.add_paragraph()
            style_paragraph(p, space_after=2)
            run = p.add_run(raw)
            set_run_font(run, size=10)
            run.font.name = "Courier New"
            i += 1
            continue

        if not raw.strip():
            i += 1
            continue

        if raw.strip() == "---":
            i += 1
            continue

        if raw.startswith("# "):
            p = doc.add_paragraph()
            style_paragraph(p, center=True, space_after=12, space_before=0)
            add_runs_with_markup(p, raw[2:].strip(), size=14)
            for run in p.runs:
                run.bold = True
            i += 1
            continue

        if raw.startswith("## "):
            title = raw[3:].strip()
            in_refs = title.lower() == "references"
            p = doc.add_paragraph()
            style_paragraph(p, space_before=18, space_after=10)
            add_runs_with_markup(p, title, size=12)
            for run in p.runs:
                run.bold = True
            i += 1
            continue

        if raw.startswith("### "):
            p = doc.add_paragraph()
            style_paragraph(p, space_before=12, space_after=6)
            add_runs_with_markup(p, raw[4:].strip(), size=12)
            for run in p.runs:
                run.bold = True
                run.italic = True
            i += 1
            continue

        if raw.strip().startswith("|"):
            rows, i = parse_table(lines, i)
            add_table(doc, rows)
            continue

        if raw.startswith("- ") or raw.startswith("1. ") or re.match(r"^\d+\. ", raw):
            content = re.sub(r"^(-\s+|\d+\.\s+)", "", raw)
            p = doc.add_paragraph(style="List Number" if re.match(r"^\d+\. ", raw) else "List Bullet")
            p.clear()
            style_paragraph(p, space_after=4)
            add_runs_with_markup(p, content, size=12)
            i += 1
            continue

        # Author / meta lines before Abstract: keep centered lightly
        if i < 20 and not raw.startswith("##"):
            p = doc.add_paragraph()
            style_paragraph(p, center=True, space_after=4)
            add_runs_with_markup(p, raw.lstrip("*").rstrip("*").strip() if raw.startswith("*(") else raw, size=12)
            i += 1
            continue

        p = doc.add_paragraph()
        if in_refs:
            style_paragraph(p, hanging=True, space_after=8)
        else:
            style_paragraph(p, first_line=True, space_after=8)
        add_runs_with_markup(p, raw, size=12)
        i += 1

    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
