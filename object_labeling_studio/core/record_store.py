"""
Owns the "one record per material+color combination" data model — the
thing that turns ONE Gemini API call (see core/gemini_client.
describe_material) into potentially SEVERAL individually-trackable,
individually-shareable records, each with its own backing PDF.

Deliberately one API call, split many ways — not one API call PER
combination. Gemini already returns every material/part/color combo it
found in a single structured response; calling it again per-combo would
just burn quota for information already sitting in the first response.

STORAGE
-------
Metadata lives in Mongo (gemini_records collection — see
mongo_client.save_gemini_record and friends). Each record's PDF lives
on local disk under config.GEMINI_RECORDS_ROOT/<object_id>/<safe
material>_<safe color>.pdf — the two are kept in sync: the Mongo
document's "pdf_path" field always points at the actual file
core/pdf_export.py wrote.

Roboflow cannot store the PDF itself — there is no file/document
attachment capability anywhere in Roboflow's API or Asset Library
(checked before building this; their storage model is images + scalar
metadata + tags + annotations only). So the arrangement is: PDFs +
full detail live here (Mongo + local disk); Roboflow's own per-image
metadata gets a pointer back — a list of this object's record ids and
a short human-readable summary — so anyone looking at the Roboflow
image can trace back to exactly which local records back it up, even
though the files themselves live outside Roboflow. See
screens/annotate_screen.py for where that pointer actually gets
written.

LABELED VS UNLABELED
---------------------
has_records_for_object() is what "already labeled" vs "not yet
labeled" means throughout this tool's screens — an object with zero
gemini_records is unlabeled; one or more means it's been through this
at least once. Screens filter on this explicitly (a three-way choice:
Unlabeled / Already labeled / All) rather than silently hiding
already-processed objects with no way back to them, since revisiting
an object (a Gemini re-run, editing a share code, regenerating a PDF)
is a normal, expected thing to want to do here, not an edge case.
"""

import os
import re
import uuid
from datetime import datetime
from typing import List, Optional

from vision.storage import mongo_client

from object_labeling_studio.config import GEMINI_RECORDS_ROOT
from object_labeling_studio.core.pdf_export import export_record_to_pdf


def _safe_filename_part(text: str) -> str:
    """Turns a material/color string into something safe to use in a
    filename — collapses anything that isn't alphanumeric/space/hyphen
    into underscores, so "black & orange" doesn't break on a
    filesystem that dislikes '&'."""
    cleaned = re.sub(r"[^a-zA-Z0-9 \-]", "", text).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "unspecified"


def has_records_for_object(object_id: str) -> bool:
    """True if this object has been through Gemini labeling at least
    once (one or more gemini_records exist for it) — see module
    docstring's "LABELED VS UNLABELED" section."""
    return len(mongo_client.list_gemini_records_for_object(object_id)) > 0


def create_records_from_gemini_result(object_id: str, object_name: str, image_path: str,
                                       gemini_result: dict) -> List[dict]:
    """
    Splits ONE gemini_client.describe_material() result into one record
    per {part, material, color} combination in its "materials" list
    (falls back to a single "overall" record if materials came back
    empty, so an object never ends up with zero records just because
    Gemini didn't break it into parts), writes each one's PDF, and
    saves each to Mongo.

    Every record gets its own random id, and shares the SAME
    prompt_text/response_text (the one real API call this all came
    from) and the same reference photo — only part/material/color/
    confidence actually differ between records for the same object.

    Returns the list of saved record dicts (in the same shape stored
    in Mongo, including their "_id" and "pdf_path").
    """
    materials = gemini_result.get("materials") or [{
        "part": "overall", "material": gemini_result.get("object", "") or "unknown",
        "color": "", "confidence": "",
    }]

    saved = []
    for m in materials:
        record_id = f"{object_id}_{uuid.uuid4().hex[:8]}"
        pdf_filename = f"{_safe_filename_part(m['material'])}_{_safe_filename_part(m.get('color') or 'na')}.pdf"
        pdf_path = os.path.abspath(os.path.join(GEMINI_RECORDS_ROOT, object_id, pdf_filename))

        record = {
            "_id": record_id,
            "object_id": object_id,
            "object": object_name,
            "part": m.get("part", "overall"),
            "material": m.get("material", ""),
            "color": m.get("color", ""),
            "confidence": m.get("confidence", ""),
            "notes": gemini_result.get("notes", ""),
            "labels": gemini_result.get("labels", []),
            "prompt_text": gemini_result.get("prompt_text", ""),
            "response_text": gemini_result.get("response_text", ""),
            "image_path": image_path,
            "pdf_path": pdf_path,
            "share_code": "",
            "source": "api",
            "created_at": datetime.now(),
        }
        export_record_to_pdf(record, pdf_path)
        mongo_client.save_gemini_record(record)
        saved.append(record)
    return saved


def create_record_from_pasted_conversation(object_id: str, object_name: str, part: str, material: str,
                                            color: str, pasted_text: str,
                                            image_path: str = None) -> dict:
    """
    Backfill path for an OLD, manually-shared conversation — no API
    call happens here at all. You paste whatever text you copied out
    of the shared chat yourself (see core/pdf_export.py's docstring for
    why this exists instead of an automated fetch), and it's formatted
    into the exact same PDF template and filed the exact same way a
    fresh API record would be, so the archive doesn't end up with two
    inconsistent record shapes depending on where something came from.
    """
    record_id = f"{object_id}_{uuid.uuid4().hex[:8]}"
    pdf_filename = f"{_safe_filename_part(material)}_{_safe_filename_part(color or 'na')}.pdf"
    pdf_path = os.path.abspath(os.path.join(GEMINI_RECORDS_ROOT, object_id, pdf_filename))

    record = {
        "_id": record_id,
        "object_id": object_id,
        "object": object_name,
        "part": part or "overall",
        "material": material,
        "color": color,
        "confidence": "",
        "notes": "",
        "labels": [],
        "prompt_text": "",
        "response_text": "",
        "pasted_text": pasted_text,
        "image_path": image_path or "",
        "pdf_path": pdf_path,
        "share_code": "",
        "source": "pasted",
        "created_at": datetime.now(),
    }
    export_record_to_pdf(record, pdf_path)
    mongo_client.save_gemini_record(record)
    return record


def set_share_code(record_id: str, share_code: str) -> dict:
    """Saves a manually-provided Gemini share code onto one record AND
    regenerates its PDF so the code actually appears in the document,
    not just in Mongo. Returns the updated record."""
    mongo_client.update_gemini_record_share_code(record_id, share_code.strip())
    record = mongo_client.get_gemini_record(record_id)
    if record:
        export_record_to_pdf(record, record["pdf_path"])
    return record


def records_for_object(object_id: str) -> List[dict]:
    return mongo_client.list_gemini_records_for_object(object_id)


def all_records(limit: int = 500) -> List[dict]:
    return mongo_client.list_all_gemini_records(limit=limit)


def roboflow_metadata_pointer(object_id: str) -> dict:
    """
    Builds the small pointer dict written into Roboflow's per-image
    metadata (see screens/annotate_screen.py) — NOT the full record
    data, since Roboflow can't store the PDF or the full conversation
    text anyway (see module docstring). Just enough to find the real
    detail later: which record ids to look up here, and a short
    human-readable summary for anyone glancing at Roboflow itself.
    """
    records = records_for_object(object_id)
    if not records:
        return {}
    summary = "; ".join(f"{r['part']}: {r['material']}" + (f" ({r['color']})" if r.get("color") else "")
                         for r in records)
    return {
        "gemini_record_ids": [r["_id"] for r in records],
        "gemini_material_summary": summary,
    }
