"""Turn raw proposals (original-pixel boxes) into a checked ImageLabelSet.

The app, not the AI, supplies: laser measurements, pixel-based color, the table
rule, taxonomy checks and the questions that follow from them (spec section 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.labeler.components import apply_components
from core.labeler.context import LabelContext
from core.labeler.table_rule import apply_table_rule
from core.models import INCIDENCES, ImageLabelSet, LaserBox, ObjectBox, Question
from core.tools.color import sample_color
from core.tools.geometry import clamp_box, iou
from core.tools.measure_laser import measure_laser


@dataclass
class RawResult:
    """Proposals in ORIGINAL pixel coordinates."""
    laser_boxes: list[dict] = field(default_factory=list)
    object_boxes: list[dict] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    notes: str = ""


def _measure_lasers(ctx: LabelContext, raw: list[dict]) -> list[LaserBox]:
    lasers: list[LaserBox] = []
    centers: dict[str, tuple[float, float]] = {}
    first_total = None
    for item in sorted(raw, key=lambda d: d["order"]):
        prev = centers.get(item.get("from_id") or "")
        box = clamp_box(item["box"], ctx.image.width, ctx.image.height)
        metrics = measure_laser(ctx.rgb, box, ctx.stats, ctx.image.is_grayscale, kind=item["kind"],
                                first_total=first_total, prev_center=prev, lines=ctx.detection.lines,
                                v=ctx.intensity)
        if item["kind"] == "first_incidence" and first_total is None and metrics["total_light"] > 0:
            first_total = metrics["total_light"]
        if "center_x" in metrics:
            centers[item["id"]] = (metrics["center_x"], metrics["center_y"])
        lasers.append(LaserBox(item["id"], item["kind"], box, int(item["order"]), item.get("from_id"),
                               metrics, item.get("notes", "")))
    return lasers


def _color_for(ctx: LabelContext, item: dict, box, color_conf: float, questions: list[Question]) -> str:
    if ctx.image.is_grayscale:
        return ctx.color_placeholder
    sample = sample_color(ctx.rgb, box, allowed_colors=ctx.taxonomy.colors, placeholder=ctx.color_placeholder)
    if sample.name and sample.confidence >= color_conf:
        return sample.name
    given = item.get("color")
    if isinstance(given, str) and given in ctx.taxonomy.colors:
        return given
    questions.append(Question(f"qc_{item['id']}", "The color of this region is ambiguous. Which is it?",
                              item["id"], "color", list(ctx.taxonomy.colors), True))
    return sample.name or "unknown"


def _taxonomy_questions(ctx: LabelContext, obj: ObjectBox, color_asked: bool) -> list[Question]:
    problems = ctx.taxonomy.check_object(obj.general_object, obj.sub_object, obj.material, obj.color,
                                         obj.incidence, ctx.color_placeholder)
    problems = [p for p in problems if not (color_asked and p.startswith("color"))]
    return [Question(f"qt_{obj.id}_{i}", f"{p}. Pick an allowed value or add it to the taxonomy.",
                     obj.id, None, [], True) for i, p in enumerate(problems)]


def _objects(ctx: LabelContext, raw: list[dict], color_conf: float, low_conf: float, questions: list[Question]):
    objects = []
    for item in raw:
        box = clamp_box(item["box"], ctx.image.width, ctx.image.height)
        incidence = item["incidence"]
        if incidence not in INCIDENCES:
            questions.append(Question(f"qi_{item['id']}", f"Incidence '{incidence}' is not valid.",
                                      item["id"], "incidence", list(INCIDENCES)))
            incidence = "other"
        asked = len(questions)
        color = _color_for(ctx, item, box, color_conf, questions)
        obj = ObjectBox(item["id"], box, item["general_object"], item["sub_object"], item["material"],
                        color, incidence,
                        list(item.get("laser_ids", [])), item.get("finish"), item["confidence"], item.get("notes", ""))
        questions.extend(_taxonomy_questions(ctx, obj, len(questions) > asked))
        if obj.confidence < low_conf:
            questions.append(Question(f"ql_{obj.id}", f"Low confidence ({obj.confidence:.2f}) on this box. "
                                      "Please check the object, material and color.", obj.id))
        objects.append(obj)
    return objects


def merge_duplicate_objects(objects: list[ObjectBox], min_iou: float = 0.6) -> list[ObjectBox]:
    """Two spots on the same surface region give one object box listing both lasers."""
    kept: list[ObjectBox] = []
    for obj in objects:
        twin = next((k for k in kept if iou(k.box, obj.box) >= min_iou and
                     (k.general_object, k.sub_object, k.material, k.color, k.incidence) ==
                     (obj.general_object, obj.sub_object, obj.material, obj.color, obj.incidence)), None)
        if twin:
            twin.laser_ids = sorted(set(twin.laser_ids) | set(obj.laser_ids))
            twin.confidence = min(twin.confidence, obj.confidence)
        else:
            kept.append(obj)
    return kept


def finalize(ctx: LabelContext, raw: RawResult, *, image_id: str, camera_id: str, table_rule: str,
             color_confidence: float = 0.25, question_confidence: float = 0.4, scheme: str = "full") -> ImageLabelSet:
    questions = list(raw.questions)
    lasers = _measure_lasers(ctx, raw.laser_boxes)
    if scheme == "components":  # each spot becomes its parts: core, halo, beam
        lasers = apply_components(lasers, ctx.detection.lines, ctx.image.width, ctx.image.height,
                                  intensity=ctx.intensity, noise=ctx.stats.noise_level)
    objects = _objects(ctx, raw.object_boxes, color_confidence, question_confidence, questions)
    merged = merge_duplicate_objects(objects)
    objects = apply_table_rule(merged, table_rule)
    removed = {o.id for o in merged} - {o.id for o in objects}  # dropped by the table rule
    known = {o.id for o in objects} | {ls.id for ls in lasers}
    questions = [q for q in questions if q.target_id not in removed]
    for q in questions:
        if q.target_id is not None and q.target_id not in known:
            q.target_id = None  # never lose a model question over a bad reference
    if not lasers and not any(q.id == "q_no_laser" for q in questions):
        questions.append(Question("q_no_laser", "No laser was found. Click where the beam first hits.", None, "hit_point"))
    return ImageLabelSet(
        image_id=image_id, image_name=Path(ctx.image.path).name, camera_id=camera_id,
        is_grayscale=ctx.image.is_grayscale, background_level=ctx.stats.background_level,
        exposure=ctx.image.exposure, laser_boxes=lasers, object_boxes=objects, questions=questions,
        status="needs_input" if questions else "pending", image_width=ctx.image.width,
        image_height=ctx.image.height, notes=raw.notes)
