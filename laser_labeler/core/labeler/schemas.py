"""JSON Schemas that constrain model replies on servers that support structured output.

Used in pipeline mode only (no tools in that call). When the whole taxonomy is matched, object and material
are limited to the listed values plus "unknown", so a small model cannot invent a value; "unknown" is caught
later and turned into a question for the user. Otherwise they are free text, so the model can name a new object
(the app reuses a listed word when one fits, and adds or asks as the taxonomy match setting says).
"""
from __future__ import annotations

from core.models import INCIDENCES, LASER_KINDS
from core.schema.taxonomy import Taxonomy

UNKNOWN = "unknown"


def _text(null: bool = False) -> dict:
    return {"type": ["string", "null"] if null else "string"}


def _object(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _question() -> dict:
    return _object({"id": _text(), "text": _text(), "target_id": _text(True), "field": _text(True),
                    "choices": {"type": "array", "items": _text()}, "allow_taxonomy_add": {"type": "boolean"}})


def _surface(tax: Taxonomy, strict: bool = True) -> dict:
    return _object({
        "general_object": {"enum": [*tax.objects, UNKNOWN]} if strict else _text(),
        "material": {"enum": [*tax.materials, UNKNOWN]} if strict else _text(),
        "finish": _text(True),
        "incidence": {"enum": list(INCIDENCES)},
        "confidence": {"type": "number"},
    })


def pipeline_schema(candidate_ids: list[str], tax: Taxonomy, strict: bool = True) -> dict:
    """Schema for the pipeline-mode reply (see prompts/parts/06_output_format.pipeline.md)."""
    candidate = _object({
        "id": {"enum": list(candidate_ids)},
        "is_laser": {"type": "boolean"},
        "kind": {"enum": [*LASER_KINDS, None]},
        "order": {"type": ["integer", "null"]},
        "from_id": {"enum": [*candidate_ids, None]},
        "object": {"anyOf": [{"type": "null"}, _surface(tax, strict)]},
        "notes": _text(),
    })
    return _object({"status": {"enum": ["ok", "needs_input"]},
                    "candidates": {"type": "array", "items": candidate},
                    "questions": {"type": "array", "items": _question()}})
