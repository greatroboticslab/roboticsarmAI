"""Turn raw proposals (original-pixel boxes) into a checked ImageLabelSet.

The app, not the AI, supplies: laser measurements, pixel-based color, the table
rule, taxonomy checks and the questions that follow from them (spec section 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.labeler.components import apply_components
from core.labeler.laser_size import grow_laser_boxes
from core.labeler.snap import snap_lasers
from core.labeler.context import LabelContext
from core.labeler.table_rule import apply_table_rule
from core.schema.class_names import normalise_token
from core.schema.taxonomy import match_entry
from core.models import INCIDENCES, ImageLabelSet, LaserBox, ObjectBox, Question
from core.tools.color import sample_color
from core.tools.geometry import clamp_box, expand_box, iou
from core.tools.measure_laser import measure_laser
from core.tools.tighten_box import tighten_box


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
    if ctx.taxonomy_match != "all" and isinstance(given, str) and normalise_token(given, ""):
        return match_entry(given, ctx.taxonomy.colors) or normalise_token(given)  # a listed color if one fits
    questions.append(Question(f"qc_{item['id']}", "The color of this region is ambiguous. Which is it?",
                              item["id"], "color", list(ctx.taxonomy.colors), True))
    return sample.name or "unknown"


MOVED_CONFIDENCE = 0.3  # a laser box the app had to move is never more certain than this


NOT_A_NAME = {"unknown", "unsure", "none", "null", "na", "n/a", "unidentified"}


def _resolve(ctx: LabelContext, kind: str, raw: str) -> str:
    """The taxonomy word for a model's object or material. A listed entry is reused; otherwise the text is made a
    token and, with the match turned off, added to the lists (the app saves them; see label_image)."""
    entries = ctx.taxonomy.objects if kind == "object" else ctx.taxonomy.materials
    hit = match_entry(raw, entries)
    if hit:
        return hit
    token = normalise_token(raw)
    if (ctx.taxonomy_match == "off" and normalise_token(raw, "") and token not in entries
            and str(raw).strip().lower() not in NOT_A_NAME):
        entries.append(token)  # also seen by the next image's prompt
        ctx.taxonomy_added.append((kind, token))
    return token


def _taxonomy_questions(ctx: LabelContext, obj: ObjectBox, color_asked: bool) -> list[Question]:
    if ctx.taxonomy_match == "off":
        if obj.general_object in NOT_A_NAME:  # the model could not name it: that is worth a question
            return [Question(f"qt_{obj.id}_0", "The model could not tell what this object is. Type its name.",
                             obj.id, "general_object", [], True)]
        return []  # nothing else is asked about the lists
    problems = ctx.taxonomy.check_object(obj.general_object, obj.material, obj.color, obj.incidence,
                                         ctx.color_placeholder, only_object=ctx.taxonomy_match != "all")
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
        general, material = item["general_object"], item["material"]
        if ctx.taxonomy_match != "all":  # reuse a listed word where one fits; free text must be a legal token
            general, material = _resolve(ctx, "object", general), _resolve(ctx, "material", material)
        obj = ObjectBox(item["id"], box, general, material,
                        color, incidence,
                        list(item.get("laser_ids", [])), item.get("finish"), item["confidence"], item.get("notes", ""))
        questions.extend(_taxonomy_questions(ctx, obj, len(questions) > asked))
        if obj.confidence < low_conf:
            questions.append(Question(f"ql_{obj.id}", f"Low confidence ({obj.confidence:.2f}) on this box. "
                                      "Please check the object, material and color.", obj.id))
        objects.append(obj)
    return objects


def _spot_on_box(object_box, spot_box, margin: float = 0.25) -> bool:
    """True if the spot's centre is inside the object box, allowing a margin (a spot can sit on an edge)."""
    x, y, w, h = object_box
    cx, cy = spot_box[0] + spot_box[2] / 2, spot_box[1] + spot_box[3] / 2
    return x - margin * w <= cx <= x + w * (1 + margin) and y - margin * h <= cy <= y + h * (1 + margin)


def replace_misplaced_objects(ctx: LabelContext, items: list[dict], lasers: list[LaserBox]) -> list[dict]:
    """An object box must hold the laser spot that hits it. A box that misses every spot it lists is wrong (a model
    that drew it in the wrong coordinate system, say): the app finds the object around the spot itself and marks the
    box low-confidence so a person checks it."""
    spots = {ls.id: ls for ls in lasers}
    out = []
    for item in items:
        hit = [spots[i] for i in item.get("laser_ids", []) if i in spots]
        if hit and not any(_spot_on_box(item["box"], s.box) for s in hit):
            spot = hit[0]
            region = tighten_box(ctx.rgb, expand_box(spot.box, 2 * max(spot.box[2:]), ctx.image.width, ctx.image.height))
            note = "The model's box did not contain its laser spot, so the app searched around the spot instead."
            item = {**item, "box": region.box, "confidence": min(item.get("confidence") or 0.5, 0.3),
                    "notes": f"{item.get('notes', '')} {note}".strip()}
        out.append(item)
    return out


def merge_duplicate_objects(objects: list[ObjectBox], min_iou: float = 0.6) -> list[ObjectBox]:
    """Two spots on the same surface region give one object box listing both lasers."""
    kept: list[ObjectBox] = []
    for obj in objects:
        twin = next((k for k in kept if iou(k.box, obj.box) >= min_iou and
                     (k.general_object, k.material, k.color, k.incidence) ==
                     (obj.general_object, obj.material, obj.color, obj.incidence)), None)
        if twin:
            twin.laser_ids = sorted(set(twin.laser_ids) | set(obj.laser_ids))
            twin.confidence = min(twin.confidence, obj.confidence)
        else:
            kept.append(obj)
    return kept


def finalize(ctx: LabelContext, raw: RawResult, *, image_id: str, camera_id: str, table_rule: str,
             color_confidence: float = 0.25, question_confidence: float = 0.4, scheme: str = "full",
             laser_object_fraction: float = 0.0) -> ImageLabelSet:
    questions = list(raw.questions)
    laser_items, snaps = snap_lasers(raw.laser_boxes, ctx.detection.candidates)
    lasers = _measure_lasers(ctx, laser_items)
    notes = raw.notes
    for snap in snaps:  # a laser box far from every spot: moved onto the nearest spot, or left, and flagged either way
        laser = next(ls for ls in lasers if ls.id == snap.laser_id)
        laser.metrics["confidence"] = min(laser.metrics.get("confidence", 0.0), MOVED_CONFIDENCE)
        far = f"about {snap.distance:.0f} px from the nearest laser spot the app found"
        if snap.moved:
            text = (f"The model's box for this laser was {far}, so the app moved it onto that spot. Check that it is "
                    "the right place. If not, use 'Fit selected box by hand'.")
            laser.notes = (laser.notes + " Box moved onto the nearest detected spot.").strip()
        else:
            text = (f"The model's box for this laser was {far} and every spot was already used, so it was left where "
                    "the model put it. Check it, and use 'Fit selected box by hand' if it is wrong.")
        questions.append(Question(f"qm_{snap.laser_id}", text, snap.laser_id))
    if snaps:
        moved = sum(s.moved for s in snaps)
        notes = (notes + f" {moved} laser box(es) were far from any laser spot and were moved onto the nearest one"
                 + (f"; {len(snaps) - moved} could not be moved." if moved < len(snaps) else ".")).strip()
    if scheme in ("components", "merged"):  # direct hit = component1, every bounce = component2
        lasers = apply_components(lasers)
    items = replace_misplaced_objects(ctx, raw.object_boxes, lasers)
    objects = _objects(ctx, items, color_confidence, question_confidence, questions)
    merged = merge_duplicate_objects(objects)
    objects = apply_table_rule(merged, table_rule)
    removed = {o.id for o in merged} - {o.id for o in objects}  # dropped by the table rule
    grow_laser_boxes(lasers, objects, laser_object_fraction, ctx.image.width, ctx.image.height)
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
        status="needs_input" if questions else "pending", scheme=scheme, image_width=ctx.image.width,
        image_height=ctx.image.height, notes=notes, pixel_space="upright")
