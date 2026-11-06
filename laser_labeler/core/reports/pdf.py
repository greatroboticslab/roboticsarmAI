"""One PDF per image: overlays, the full AI conversation, the user's answers, the final labels (spec section 8).

The PDF is always rebuilt from three sources, so it can never drift from the approved result:
the saved transcript (the AI conversation and the user's answers), the current label set,
and a small addenda file listing every edit made in review. Nothing here talks to a model.
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from core.camera import load_image
from core.labeler.transcript import load_transcript
from core.models import ImageLabelSet
from core.reports.sidecar import safe_id
from core.schema.class_names import ClassNameError, encode_object_class, resolve_color
from core.tools.crop import downscale_to_max
from core.tools.overlay import draw_boxes

MAX_TEXT_CHARS = 6000  # per message; the full text stays in the transcript JSON
OVERLAY_MAX_PX = 1400
CONTENT_WIDTH = A4[0] - 30 * mm
KEY_METRICS = ("peak_intensity_norm", "mean_intensity_norm", "core_area_px", "scatter_ratio", "halo_radius_px",
               "confidence")

_styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_styles["BodyText"], fontSize=8.5, leading=11, wordWrap="CJK", alignment=TA_LEFT)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#444444"))
MONO = ParagraphStyle("mono", parent=BODY, fontName="Courier", fontSize=7.5, leading=9.5)
H1 = ParagraphStyle("h1", parent=_styles["Heading1"], fontSize=16, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=_styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)


# ---------------------------------------------------------------- paths and addenda
def pdf_path_for(data_dir: str | Path, image_id: str) -> Path:
    return Path(data_dir) / "pdfs" / f"{safe_id(image_id)}.pdf"


def addenda_path_for(data_dir: str | Path, image_id: str) -> Path:
    return Path(data_dir) / "pdfs" / f"{safe_id(image_id)}.addenda.json"


def load_addenda(path: str | Path) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def add_addendum(path: str | Path, lines: Sequence[str], status: str, note: str = "") -> list[dict]:
    """Record one edit (a list of plain-words changes). The PDF shows these after the AI conversation."""
    path = Path(path)
    items = load_addenda(path)
    items.append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": status,
                  "changes": list(lines), "note": note})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
    tmp.replace(path)
    return items


# ---------------------------------------------------------------- small helpers
def _text(value: object) -> str:
    """Escaped, Latin-1 safe text for a Paragraph (the built-in fonts have no wider alphabet)."""
    s = str(value)
    s = s.encode("cp1252", "replace").decode("cp1252")
    return escape(s).replace("\n", "<br/>")


def _clip(value: str) -> str:
    if len(value) <= MAX_TEXT_CHARS:
        return value
    return value[:MAX_TEXT_CHARS] + f"\n... [{len(value) - MAX_TEXT_CHARS} more characters are in the transcript file]"


def class_label(label_set: ImageLabelSet, kind: str, item, placeholder: str = "na") -> str:
    """The Roboflow class name of a box, or a readable fallback when its values are not valid tokens."""
    try:
        if kind == "laser":
            return label_set.laser_class_name(item, placeholder)
        color = resolve_color(item.color, label_set.is_grayscale, placeholder)
        return encode_object_class(item.general_object, item.material, color, item.incidence)
    except ClassNameError:
        return f"(invalid class: {item.kind if kind == 'laser' else item.general_object})"


def _fit_image(source, max_width: float, max_height: float) -> Image:
    """A reportlab Image scaled to fit inside the box, keeping its proportions."""
    with PILImage.open(source if isinstance(source, (str, Path)) else io.BytesIO(source)) as im:
        w, h = im.size
    if hasattr(source, "seek"):
        source.seek(0)
    scale = min(max_width / w, max_height / h)
    return Image(source if isinstance(source, (str, Path)) else io.BytesIO(source), width=w * scale, height=h * scale)


def overlay_png(label_set: ImageLabelSet, image_path: str | Path, placeholder: str = "na") -> bytes:
    """The image with every final box drawn and labeled."""
    rgb = load_image(image_path).rgb
    shown, factor = downscale_to_max(rgb, OVERLAY_MAX_PX)
    items = [(ls.id + ":" + ls.kind, ls.box) for ls in label_set.laser_boxes]
    items += [(o.id + ":" + o.general_object, o.box) for o in label_set.object_boxes]
    drawn = draw_boxes(shown, items, factor)
    buf = io.BytesIO()
    PILImage.fromarray(drawn).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- sections
def _table(rows: list[list], widths: Sequence[float], header: bool = True) -> Table:
    data = [[Paragraph(_text(c), SMALL) if not hasattr(c, "wrap") else c for c in row] for row in rows]
    table = Table(data, colWidths=list(widths), repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")))
    table.setStyle(TableStyle(style))
    return table


def _summary_block(ls: ImageLabelSet) -> list:
    rows = [["Image", f"{ls.image_name}  (id {ls.image_id})"], ["Status", ls.status],
            ["Prompt version", ls.prompt_version or "-"], ["Provider / model", f"{ls.provider or '-'} / {ls.model or '-'}"],
            ["Run mode", ls.run_mode or "-"], ["Camera", f"{ls.camera_id}{' (grayscale)' if ls.is_grayscale else ''}"],
            ["Image size", f"{ls.image_width} x {ls.image_height} px"]]
    if ls.notes:
        rows.append(["Notes", ls.notes])
    return [_table(rows, [35 * mm, CONTENT_WIDTH - 35 * mm], header=False)]


def _labels_block(ls: ImageLabelSet, placeholder: str) -> list:
    out: list = [Paragraph("Final labels", H2)]
    lasers = [["id", "class", "box x,y,w,h", "order", "from"] + ["peak", "area px", "scatter", "conf"]]
    for box in ls.laser_boxes:
        m = box.metrics
        lasers.append([box.id, class_label(ls, "laser", box, placeholder), list(box.box), box.order, box.from_id or "-",
                       f"{m.get('peak_intensity_norm', 0):.2f}", f"{m.get('core_area_px', 0):.0f}",
                       f"{m.get('scatter_ratio', 0):.2f}", f"{m.get('confidence', 0):.2f}"])
    widths = [10, 50, 28, 9, 10, 12, 14, 13, 12]
    scale = CONTENT_WIDTH / (sum(widths) * mm)
    out += [Paragraph("Laser boxes", BODY),
            _table(lasers if len(lasers) > 1 else [lasers[0], ["(none)"] + [""] * 8], [w * mm * scale for w in widths])]
    objects = [["id", "class", "box x,y,w,h", "lasers", "conf"]]
    for obj in ls.object_boxes:
        objects.append([obj.id, class_label(ls, "object", obj, placeholder), list(obj.box),
                        ", ".join(obj.laser_ids) or "-", f"{obj.confidence:.2f}"])
    widths = [10, 80, 30, 18, 12]
    scale = CONTENT_WIDTH / (sum(widths) * mm)
    out += [Spacer(1, 4), Paragraph("Object boxes", BODY),
            _table(objects if len(objects) > 1 else [objects[0], ["(none)", "", "", "", ""]],
                   [w * mm * scale for w in widths])]
    if ls.questions:
        rows = [["question", "answer"]] + [[q.text, q.answer if q.answer is not None else "(unanswered)"]
                                            for q in ls.questions]
        out += [Spacer(1, 4), Paragraph("Questions", BODY), _table(rows, [CONTENT_WIDTH * 0.6, CONTENT_WIDTH * 0.4])]
    return out


def _entry_block(entry: dict, image_dir: Path | None) -> list:
    stamp = entry.get("time", "")[11:19] + "Z" if entry.get("time") else ""
    kind = entry.get("type")
    if kind != "message":
        label = {"answer": "USER ANSWER", "reviewer_note": "REVIEWER NOTE", "error": "ERROR", "fallback": "FALLBACK",
                 "invalid_reply": "INVALID REPLY"}.get(kind, str(kind).upper())
        return [Paragraph(f"<b>{_text(stamp)} {label}</b>  {_text(_clip(entry.get('text', '')))}", BODY), Spacer(1, 3)]
    role = entry.get("role", "?")
    head = f"<b>{_text(stamp)} {role.upper()}</b>" + (f" (tool {_text(entry['tool_name'])})" if entry.get("tool_name") else "")
    top: list = [Paragraph(head, BODY)]
    if entry.get("text"):
        top.append(Paragraph(_text(_clip(entry["text"])), MONO if role in ("assistant", "tool") else BODY))
    rest: list = []
    for call in entry.get("tool_calls") or []:
        rest.append(Paragraph(f"<b>tool call</b> {_text(call['name'])} {_text(json.dumps(call['arguments']))}", MONO))
    for name in entry.get("images") or []:
        path = (image_dir / name) if image_dir else None
        if path is not None and path.is_file():
            rest.append(_fit_image(str(path), 85 * mm, 60 * mm))
        else:
            rest.append(Paragraph(f"[image {_text(name)} not found]", SMALL))
    short = len(entry.get("text") or "") < 800  # a long message may split across pages
    return ([KeepTogether(top)] if short else top) + rest + [Spacer(1, 3)]


def _conversation_block(transcript: dict | None, image_dir: Path | None) -> list:
    out: list = [Paragraph("AI conversation", H2)]
    if not transcript:
        return out + [Paragraph("No transcript was saved for this image.", BODY)]
    usage = transcript.get("usage") or {}
    out.append(Paragraph(f"Tokens used: {usage.get('input_tokens', 0)} in, {usage.get('output_tokens', 0)} out.", SMALL))
    for entry in transcript.get("entries", []):
        out += _entry_block(entry, image_dir)
    return out


def _addenda_block(addenda: Sequence[dict]) -> list:
    out: list = [Paragraph("Edits after the AI proposal", H2)]
    for n, item in enumerate(addenda, 1):
        out.append(Paragraph(f"<b>Edit {n}</b>  {_text(item.get('time', ''))}  (status now: {_text(item.get('status', ''))})", BODY))
        for line in item.get("changes", []) or ["(no label changes recorded)"]:
            out.append(Paragraph("- " + _text(line), BODY))
        if item.get("note"):
            out.append(Paragraph("Note: " + _text(item["note"]), BODY))
        out.append(Spacer(1, 4))
    return out


# ---------------------------------------------------------------- the document
def build_pdf(label_set: ImageLabelSet, image_path: str | Path, out_path: str | Path, *,
              transcript_path: str | Path | None = None, addenda: Sequence[dict] = (), placeholder: str = "na") -> Path:
    """Write the conversation PDF. Missing transcripts or crop images are noted in the PDF, never fatal."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    transcript, image_dir = None, None
    tpath = Path(transcript_path or label_set.transcript_path or "")
    if transcript_path or label_set.transcript_path:
        try:
            transcript = load_transcript(tpath)
            image_dir = tpath.with_suffix("")
        except (OSError, ValueError):
            transcript = None
    story: list = [Paragraph(f"Laser label report: {_text(label_set.image_name)}", H1)]
    story += _summary_block(label_set)
    story.append(Paragraph("Image with final labels", H2))
    try:
        story.append(_fit_image(overlay_png(label_set, image_path, placeholder), CONTENT_WIDTH, 150 * mm))
    except (OSError, ValueError) as exc:
        story.append(Paragraph(f"The image could not be drawn: {_text(exc)}", BODY))
    story += _labels_block(label_set, placeholder)
    if addenda:
        story += _addenda_block(addenda)
    story += [PageBreak()] + _conversation_block(transcript, image_dir)
    tmp = out_path.with_suffix(".pdf.tmp")
    doc = SimpleDocTemplate(str(tmp), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm,
                            bottomMargin=15 * mm, title=f"Laser label report {label_set.image_id}",
                            author="Laser Labeler")
    doc.build(story, onFirstPage=_footer(label_set), onLaterPages=_footer(label_set))
    tmp.replace(out_path)
    return out_path


def _footer(ls: ImageLabelSet):
    def draw(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(15 * mm, 8 * mm, _footer_text(ls))
        canvas.drawRightString(A4[0] - 15 * mm, 8 * mm, f"page {doc.page}")
        canvas.restoreState()
    return draw


def _footer_text(ls: ImageLabelSet) -> str:
    return re.sub(r"[^\x20-\x7e]", "?", f"{ls.image_id}  prompt {ls.prompt_version or '-'}  {ls.provider}/{ls.model}")
