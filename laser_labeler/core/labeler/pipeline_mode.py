"""Pipeline mode (spec 5.2): the app finds and measures; the model only classifies.

Default for local models. One model call per image: an overview with every
candidate boxed, plus a zoomed crop per candidate, answered with strict JSON.
"""
from __future__ import annotations

from core.labeler.context import LabelContext
from core.labeler.conversation import CostGuard, call_model, repair_loop
from core.labeler.finalize import RawResult
from core.labeler.parse_reply import PipelineReply, parse_pipeline_reply
from core.labeler.schemas import pipeline_schema
from core.labeler.transcript import Transcript
from core.models import Question
from core.providers.base import ImagePart, Message, VisionProvider
from core.tools.crop import crop_zoom, encode_png
from core.tools.geometry import expand_box
from core.tools.overlay import draw_boxes
from core.tools.tighten_box import tighten_box


def _user_message(ctx: LabelContext, candidates) -> Message:
    items = [(c.id, c.box) for c in candidates]
    images = [ImagePart(encode_png(draw_boxes(ctx.shown, items, ctx.factor)))]
    lines = [f"Camera: {'grayscale' if ctx.image.is_grayscale else 'color'}. Image size shown: "
             f"{ctx.shown.shape[1]}x{ctx.shown.shape[0]} px.",
             "Image 1 is the whole image with each candidate laser spot boxed and labeled."]
    for n, cand in enumerate(candidates, start=2):
        crop = crop_zoom(ctx.rgb, expand_box(cand.box, 12, ctx.image.width, ctx.image.height), pad_fraction=1.0)
        images.append(ImagePart(encode_png(crop.image)))
        lines.append(f"Image {n} is a zoomed crop around {cand.id} (peak brightness {cand.peak:.2f}, "
                     f"{cand.area} px core, {cand.saturated_fraction:.0%} saturated).")
    if ctx.user_note:
        lines.append(f"Note from the reviewer about this image (follow it): {ctx.user_note}")
    lines.append("Classify every candidate and reply with the JSON object described above.")
    return Message("user", "\n".join(lines), images=images)


def _no_laser(ctx: LabelContext) -> RawResult:
    note = "No laser spot was found by image processing."
    return RawResult(questions=[Question("q_no_laser", note + " Click where the beam first hits.", None, "hit_point")],
                     notes=note)


def _build(ctx: LabelContext, candidates, reply: PipelineReply) -> RawResult:
    by_id = {c.id: c for c in candidates}
    raw = RawResult(questions=list(reply.questions))
    lasers = {v["id"]: v for v in reply.candidates if v["is_laser"]}
    ranked = sorted(lasers.values(), key=lambda v: (v["order"] if isinstance(v["order"], int) else 99,
                                                     -by_id[v["id"]].peak))
    for position, verdict in enumerate(ranked):
        cand = by_id[verdict["id"]]
        lid = "l" + cand.id[1:]
        from_id = verdict.get("from_id") if isinstance(verdict.get("from_id"), str) else None
        raw.laser_boxes.append({"id": lid, "kind": verdict["kind"], "box": cand.box, "order": position,
                                "from_id": "l" + from_id[1:] if from_id in lasers else None,
                                "notes": verdict["notes"]})
        obj = verdict["object"]
        if obj:
            region = tighten_box(ctx.rgb, expand_box(cand.box, 2 * max(cand.box[2:]), ctx.image.width, ctx.image.height))
            raw.object_boxes.append({
                "id": "o" + cand.id[1:], "box": region.box, "laser_ids": [lid], "color": None,
                "general_object": obj["general_object"],
                "material": obj["material"], "finish": obj.get("finish"), "incidence": obj["incidence"],
                "confidence": min(obj.get("confidence") or 0.5, max(region.confidence, 0.3) + 0.2),
                "notes": "" if region.changed else "box not tightened"})
    made = {o["id"] for o in raw.object_boxes}
    for q in raw.questions:  # candidate ids in questions become the box ids the app created
        if q.target_id in by_id:
            q.target_id = ("o" if "o" + q.target_id[1:] in made else "l") + q.target_id[1:]
    dropped = [cid for cid in by_id if cid not in lasers]
    raw.notes = f"candidates the model rejected as not laser: {', '.join(dropped)}" if dropped else ""
    return raw


def run_pipeline(ctx: LabelContext, provider: VisionProvider, system_prompt: str, transcript: Transcript, *,
                 max_repairs: int = 2, guard: CostGuard | None = None, structured: bool = False) -> RawResult:
    candidates = ctx.detection.candidates[:ctx.max_candidates]
    if not candidates:
        return _no_laser(ctx)
    messages = [Message("system", system_prompt), _user_message(ctx, candidates)]
    for message in messages:
        transcript.add_message(message)
    ids = [c.id for c in candidates]
    schema = pipeline_schema(ids, ctx.taxonomy) if structured else None
    first = call_model(provider, messages, transcript, None, guard, schema)
    reply, repairs = repair_loop(provider, messages, transcript, first,
                                 lambda text: parse_pipeline_reply(text, ids), max_repairs, guard, schema)
    raw = _build(ctx, candidates, reply)
    if repairs:
        raw.notes = (raw.notes + f" reply needed {repairs} repair(s).").strip()
    return raw
