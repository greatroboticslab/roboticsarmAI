"""
pdf_label_parser.py

Parses the Gemini-conversation-transcript PDFs stored in dataset/pdf_for_labels.

Each PDF is a chat transcript: an initial labeling prompt/response, optionally
followed by one or more human correction turns (e.g. "thats a pen ands its
plastic rubber near the tip and transparent plastic"). Corrections are
appended to the END of the document (chronological order == top-to-bottom
document order), so the physically LAST "Response:" turn in the PDF is
always the most current, authoritative label -- any responses above it are
superseded and must be ignored.

This module walks the document turn-by-turn starting from the BOTTOM
("Response:" occurrences, latest first) and works its way UP, only falling
back to an earlier turn if the most recent one can't be parsed at all (e.g.
a PDF text-extraction artifact truncated it). This means a later correction
always wins over an earlier machine-generated guess.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

_RESPONSE_MARKER_RE = re.compile(r"Response:")
_USER_PROMPT_MARKER_RE = re.compile(r"User prompt:")

# One entry inside the JSON "materials": [ ... ] array. Written to tolerate
# literal newlines (from PDF line-wrapping) between fields via \s*, and to
# not care what comes after the closing brace (so a corrupted/truncated
# trailing "notes" field elsewhere in the same turn doesn't break this).
_MATERIAL_ENTRY_RE = re.compile(
    r'\{\s*"part"\s*:\s*"(?P<part>.*?)"\s*,\s*'
    r'"material"\s*:\s*"(?P<material>.*?)"\s*,\s*'
    r'"color"\s*:\s*"(?P<color>.*?)"\s*,\s*'
    r'"confidence"\s*:\s*"(?P<confidence>.*?)"\s*\}',
    re.DOTALL,
)

_OBJECT_FIELD_RE = re.compile(r'"object"\s*:\s*"(?P<object>.*?)"\s*,\s*"materials"', re.DOTALL)

# Fallback: the one-line plain text summary, e.g.
# "Object: ballpoint pen | Material: plastic, rubber | Color: red, clear"
_PLAIN_LINE_RE = re.compile(
    r"Object:\s*(?P<object>.+?)\s*\|\s*Material:\s*(?P<material>.+?)"
    r"(?:\s*\|\s*Color:\s*(?P<color>.+?))?\s*$"
)


@dataclass
class MaterialEntry:
    part: str
    material: str
    color: str = ""
    confidence: str = ""


@dataclass
class PdfLabel:
    pdf_id: str
    object_name: str
    materials: list[MaterialEntry] = field(default_factory=list)
    num_turns_found: int = 0
    turn_index_used: int = 0  # 0 = bottom-most / most recent turn, 1 = one turn up, etc.
    used_fallback: bool = False
    source_path: str = ""


def _extract_full_text(pdf_path: Path) -> str:
    """Extract text from every page, concatenated top-to-bottom (chronological order)."""
    chunks = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def _normalize_quotes(s: str) -> str:
    return (
        s.replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2018", "'").replace("\u2019", "'")
    )


def _split_into_response_turns(text: str) -> list[str]:
    """
    Split the transcript into the text following each 'Response:' marker, up
    to (but not including) the next 'User prompt:' marker or end of doc.
    Returned in DOCUMENT ORDER (oldest turn first, most recent turn last).
    """
    turns = []
    resp_matches = list(_RESPONSE_MARKER_RE.finditer(text))
    for m in resp_matches:
        start = m.end()
        next_prompt = _USER_PROMPT_MARKER_RE.search(text, start)
        end = next_prompt.start() if next_prompt else len(text)
        turns.append(text[start:end])
    return turns


def _dedupe_materials(materials: list[MaterialEntry]) -> list[MaterialEntry]:
    """Collapse multiple parts sharing the same (material, color) -- data.yaml
    classes are keyed on material+color, not on individual object parts."""
    seen: dict[tuple[str, str], MaterialEntry] = {}
    for m in materials:
        key = (m.material.lower().strip(), m.color.lower().strip())
        if key not in seen:
            seen[key] = m
    return list(seen.values())


def _try_parse_turn(turn_text: str) -> Optional[tuple[str, list[MaterialEntry]]]:
    """Attempt to pull (object, materials) out of one 'Response:' turn's text."""
    obj_match = _OBJECT_FIELD_RE.search(turn_text)
    material_matches = list(_MATERIAL_ENTRY_RE.finditer(turn_text))
    if obj_match and material_matches:
        obj_name = re.sub(r"\s+", " ", obj_match.group("object")).strip()
        if obj_name.startswith("<"):
            return None  # placeholder schema text, not a real answer
        materials = [
            MaterialEntry(
                part=re.sub(r"\s+", " ", mm.group("part")).strip(),
                material=re.sub(r"\s+", " ", mm.group("material")).strip(),
                color=re.sub(r"\s+", " ", mm.group("color")).strip(),
                confidence=mm.group("confidence").strip(),
            )
            for mm in material_matches
        ]
        return obj_name, _dedupe_materials(materials)

    # Fall back to the plain-text summary line within this turn, in case the
    # JSON portion of this specific turn is unrecoverable.
    plain_matches = list(_PLAIN_LINE_RE.finditer(turn_text))
    if plain_matches:
        pm = plain_matches[-1]
        obj_name = pm.group("object").strip()
        mats = [x.strip() for x in pm.group("material").split(",") if x.strip()]
        colors = [x.strip() for x in pm.group("color").split(",")] if pm.group("color") else []
        materials = [
            MaterialEntry(part="overall", material=mat, color=(colors[i] if i < len(colors) else ""))
            for i, mat in enumerate(mats)
        ]
        return obj_name, _dedupe_materials(materials)

    return None


def parse_pdf_label(pdf_path: Path) -> PdfLabel:
    """
    Extract the MOST RECENT (bottom-of-document) label for one object PDF.

    Walks 'Response:' turns from the BOTTOM of the transcript upward, using
    the first turn that yields a usable (object, materials) pair. In the
    normal case that's simply the last turn in the document -- i.e. the
    newest correction. We only walk further up if the bottom turn's text
    was mangled by PDF extraction (e.g. a truncated trailing field) and
    can't be parsed at all.
    """
    pdf_id = pdf_path.stem
    text = _normalize_quotes(_extract_full_text(pdf_path))
    turns = _split_into_response_turns(text)  # oldest -> newest

    for steps_up, turn_text in enumerate(reversed(turns)):
        result = _try_parse_turn(turn_text)
        if result is not None:
            obj_name, materials = result
            return PdfLabel(
                pdf_id=pdf_id,
                object_name=obj_name,
                materials=materials,
                num_turns_found=len(turns),
                turn_index_used=steps_up,
                used_fallback=False,
                source_path=str(pdf_path),
            )

    raise ValueError(f"Could not extract any label from {pdf_path}")


def canonical_class_names(label: PdfLabel) -> list[str]:
    """
    Build the class-name string(s) for one PDF's final label, in the same
    convention used by dataset/data.yaml:
        "Object <object> - Material <material> - Pdfname <pdf_id>"
        "Object <object> - Material <material> - Color <color> - Pdfname <pdf_id>"
    One class name is produced per distinct (material, color) pair.
    """
    names = []
    for m in label.materials:
        if m.color:
            name = f"Object {label.object_name} - Material {m.material} - Color {m.color} - Pdfname {label.pdf_id}"
        else:
            name = f"Object {label.object_name} - Material {m.material} - Pdfname {label.pdf_id}"
        names.append(name)
    return names


if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1])
    lbl = parse_pdf_label(p)
    print(f"pdf_id={lbl.pdf_id} turns_found={lbl.num_turns_found} used_turn_from_bottom={lbl.turn_index_used} fallback={lbl.used_fallback}")
    print(f"object={lbl.object_name!r}")
    for m in lbl.materials:
        print(f"  part={m.part!r} material={m.material!r} color={m.color!r} confidence={m.confidence!r}")
    print("canonical class names:")
    for n in canonical_class_names(lbl):
        print(f"  - {n}")
