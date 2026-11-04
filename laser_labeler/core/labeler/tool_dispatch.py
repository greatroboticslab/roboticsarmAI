"""The tools the AI can call in agent mode (spec 5.1). Coordinates are in shown-image pixels.

Tool failures come back as text ("error: ...") so the model can correct itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from core.labeler.context import LabelContext
from core.models import Question
from core.providers.base import ImagePart, ToolCall, ToolSpec
from core.tools.color import sample_color
from core.tools.crop import crop_zoom, encode_png
from core.tools.geometry import clamp_box
from core.tools.measure_laser import measure_laser
from core.tools.tighten_box import tighten_box

_BOX = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
        "description": "[x, y, width, height] in the pixels of the image you were shown"}
TAXONOMY_PARTS = ("objects", "materials", "colors", "laser_events", "incidence", "all")


def _spec(name: str, description: str, props: dict | None = None, required: list[str] | None = None) -> ToolSpec:
    return ToolSpec(name, description, {"type": "object", "properties": props or {}, "required": required or []})


TOOL_SPECS = [
    _spec("detect_laser", "Find laser spots (ids c1, c2, ...) and visible beam lines. Call this first."),
    _spec("crop_zoom", "Return an enlarged full-resolution crop around a box.", {"box": _BOX}, ["box"]),
    _spec("tighten_box", "Snap a box to the edges of the same-colored region around it.", {"box": _BOX}, ["box"]),
    _spec("sample_color", "Dominant color of the pixels in a box (ignores laser glare).", {"box": _BOX}, ["box"]),
    _spec("measure_laser", "Preview the numbers the app will compute for a laser box.",
          {"box": _BOX, "kind": {"type": "string"}}, ["box"]),
    _spec("lookup_taxonomy", "List the allowed values.",
          {"what": {"type": "string", "enum": list(TAXONOMY_PARTS)}}, ["what"]),
    _spec("ask_user", "Park a question for the user. Does not block; keep working.",
          {"question": {"type": "string"}, "target_id": {"type": "string"}, "field": {"type": "string"},
           "choices": {"type": "array", "items": {"type": "string"}},
           "allow_taxonomy_add": {"type": "boolean"}}, ["question"]),
]


@dataclass
class ToolResult:
    text: str
    images: list[ImagePart] = field(default_factory=list)
    is_error: bool = False


def _error(message: str) -> ToolResult:
    return ToolResult(f"error: {message}", is_error=True)


def _json(data) -> ToolResult:
    return ToolResult(json.dumps(data))


def _original_box(ctx: LabelContext, args: dict):
    box = args.get("box")
    if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
        raise ValueError("'box' must be [x, y, w, h] numbers")
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError("box width and height must be positive")
    return clamp_box(ctx.to_original(box), ctx.image.width, ctx.image.height)


def _detect(ctx: LabelContext, args: dict) -> ToolResult:
    det = ctx.detection
    return _json({
        "image_size": [ctx.shown.shape[1], ctx.shown.shape[0]],
        "candidates": [{"id": c.id, "box": ctx.to_shown(c.box), "peak": round(c.peak, 3), "area_px": c.area,
                        "saturated_fraction": round(c.saturated_fraction, 2)} for c in det.candidates],
        "beam_lines": [{"from": ctx.to_shown((ln.x1, ln.y1, 0, 0))[:2], "to": ctx.to_shown((ln.x2, ln.y2, 0, 0))[:2]}
                       for ln in det.lines],
        "note": "no laser found: ask the user where it first hits" if not det.candidates else "",
    })


def _crop(ctx: LabelContext, args: dict) -> ToolResult:
    crop = crop_zoom(ctx.rgb, _original_box(ctx, args))
    region = ctx.to_shown(crop.region)
    return ToolResult(f"crop of region {region} (shown pixels), enlarged x{crop.scale / ctx.factor:.1f}",
                      [ImagePart(encode_png(crop.image))])


def _tighten(ctx: LabelContext, args: dict) -> ToolResult:
    res = tighten_box(ctx.rgb, _original_box(ctx, args))
    return _json({"box": ctx.to_shown(res.box), "confidence": res.confidence,
                  "may_extend_further": res.hit_window_edge, "changed": res.changed})


def _color(ctx: LabelContext, args: dict) -> ToolResult:
    s = sample_color(ctx.rgb, _original_box(ctx, args), is_grayscale=ctx.image.is_grayscale,
                     allowed_colors=ctx.taxonomy.colors, placeholder=ctx.color_placeholder)
    return _json({"color": s.name, "confidence": round(s.confidence, 2), "rgb": list(s.rgb)})


def _measure(ctx: LabelContext, args: dict) -> ToolResult:
    metrics = measure_laser(ctx.rgb, _original_box(ctx, args), ctx.stats, ctx.image.is_grayscale,
                            kind=str(args.get("kind", "other")), v=ctx.intensity)
    keep = ("peak_intensity", "core_area_px", "aspect_ratio", "scatter_ratio", "halo_radius_px",
            "clipped_fraction", "confidence")
    return _json({k: round(metrics[k], 4) for k in keep})


def _taxonomy(ctx: LabelContext, args: dict) -> ToolResult:
    what = args.get("what")
    if what not in TAXONOMY_PARTS:
        raise ValueError(f"'what' must be one of {', '.join(TAXONOMY_PARTS)}")
    data = ctx.taxonomy.to_dict()
    return _json(data if what == "all" else {what: data[what]})


def _ask(ctx: LabelContext, args: dict) -> ToolResult:
    text = args.get("question")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("'question' is required")
    ctx.questions.append(Question(
        id=f"qa{len(ctx.questions) + 1}", text=text.strip(), target_id=args.get("target_id"),
        field_name=args.get("field"), choices=[str(c) for c in args.get("choices") or []],
        allow_taxonomy_add=bool(args.get("allow_taxonomy_add", False))))
    return ToolResult("question parked for the user. Keep going and give your final answer.")


_HANDLERS = {"detect_laser": _detect, "crop_zoom": _crop, "tighten_box": _tighten, "sample_color": _color,
             "measure_laser": _measure, "lookup_taxonomy": _taxonomy, "ask_user": _ask}


def run_tool(ctx: LabelContext, call: ToolCall) -> ToolResult:
    handler = _HANDLERS.get(call.name)
    if handler is None:
        return _error(f"unknown tool '{call.name}'. Tools: {', '.join(_HANDLERS)}")
    if not isinstance(call.arguments, dict) or "_raw" in call.arguments:
        return _error("arguments were not valid JSON")
    try:
        return handler(ctx, call.arguments)
    except (ValueError, TypeError, KeyError) as exc:
        return _error(str(exc))
