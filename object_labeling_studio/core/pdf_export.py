"""
Renders one Gemini conversation record to a PDF — the "backing
document" record_store.py's PDF path points at, and what the Archive
screen opens.

Two sources feed the same template:
  - API records (the normal path): prompt_text/response_text come
    straight from gemini_client.describe_material()'s return value —
    the actual text sent/received, not a reconstruction.
  - Pasted records (backfilling an old, manually-shared conversation):
    the Archive screen's "Paste Conversation" import lets you paste
    whatever text you copied out of the shared chat yourself. There is
    no official API to fetch a shared Gemini conversation's content
    (checked before building this — Google doesn't expose one, and the
    unofficial scrapers that exist need a full headless browser and
    aren't reliable enough to build on), so this is the deliberate,
    ToS-safe alternative: you copy it, this formats it.

Needs fpdf2 (`pip install fpdf2`) — NOT already a dependency of the
main app, since PDF generation is specific to this tool.
"""

import os
from datetime import datetime

from fpdf import FPDF

_MARGIN = 15
# fpdf2 2.8.x's multi_cell() defaults to leaving the cursor at the
# RIGHT margin after writing (new_x=XPos.RIGHT), not the left one like
# older fpdf/fpdf2 releases — without passing this explicitly on every
# call, the very next multi_cell() has ~0 width left to work with and
# raises "Not enough horizontal space to render a single character".
# Passing these two on every call below is what actually fixes that,
# not a cosmetic choice.
_NEWLINE = {"new_x": "LMARGIN", "new_y": "NEXT"}


class _RecordPDF(FPDF):
    def header(self):
        pass  # no running header — title is written once at the top instead

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def export_record_to_pdf(record: dict, output_path: str) -> str:
    """
    Writes `record` (see core/record_store.py for the exact shape) to
    `output_path` as a formatted PDF. Overwrites if a file already
    exists there. Returns output_path.
    """
    pdf = _RecordPDF(format="A4")
    pdf.set_margins(_MARGIN, _MARGIN, _MARGIN)
    pdf.set_auto_page_break(auto=True, margin=_MARGIN)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(0, 9, f"{record.get('object', '(unnamed object)')}", **_NEWLINE)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    created_at = record.get("created_at")
    created_str = created_at.strftime("%Y-%m-%d %H:%M") if isinstance(created_at, datetime) else str(created_at or "")
    pdf.multi_cell(0, 6, f"Object ID: {record.get('object_id', '')}    |    Generated: {created_str}    |    "
                          f"Source: {record.get('source', 'api')}", **_NEWLINE)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(0, 8, "Material / Part", **_NEWLINE)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, f"Part: {record.get('part', 'overall')}", **_NEWLINE)
    pdf.multi_cell(0, 6, f"Material: {record.get('material', '')}", **_NEWLINE)
    if record.get("color"):
        pdf.multi_cell(0, 6, f"Color: {record['color']}", **_NEWLINE)
    pdf.multi_cell(0, 6, f"Confidence: {record.get('confidence', '')}", **_NEWLINE)
    if record.get("notes"):
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 6, f"Notes: {record['notes']}", **_NEWLINE)
    pdf.ln(4)

    image_path = record.get("image_path")
    if image_path and os.path.exists(image_path):
        try:
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 8, "Reference Photo", **_NEWLINE)
            pdf.image(image_path, w=100)
            pdf.ln(4)
        except Exception as e:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(180, 0, 0)
            pdf.multi_cell(0, 6, f"(Could not embed photo: {e})", **_NEWLINE)
            pdf.set_text_color(20, 20, 20)
            pdf.ln(2)

    if record.get("share_code"):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(0, 6, f"Gemini share code: {record['share_code']}", **_NEWLINE)
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 8, "Conversation", **_NEWLINE)

    prompt_text = record.get("prompt_text", "")
    response_text = record.get("response_text", "")
    pasted_text = record.get("pasted_text", "")

    if pasted_text:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(0, 5, "(Pasted in from a manually-shared conversation — not captured via the API.)",
                       **_NEWLINE)
        pdf.ln(1)
        pdf.set_font("Courier", "", 9)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(0, 5, pasted_text, **_NEWLINE)
    else:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(40, 40, 120)
        pdf.multi_cell(0, 6, "Prompt:", **_NEWLINE)
        pdf.set_font("Courier", "", 9)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(0, 5, prompt_text, **_NEWLINE)
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(40, 120, 40)
        pdf.multi_cell(0, 6, "Response:", **_NEWLINE)
        pdf.set_font("Courier", "", 9)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(0, 5, response_text, **_NEWLINE)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    return output_path
