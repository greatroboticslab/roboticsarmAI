"""Everything the review screen does to a label set, without any GUI code.

The queue database is the source of truth for an image's status; the sidecar JSON is the source of truth
for its boxes. Every method here keeps the two, the manifest, the measurements CSV and the PDF in step.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from core.schema.orientation import to_upright
from core.camera import load_image
from core.labeler.context import build_context
from core.labeler.components import SCHEMES, component_for_kind
from core.labeler.finalize import _measure_lasers
from core.labeler.transcript import append_event
from core.models import INCIDENCES, LASER_KINDS, ImageLabelSet, LaserBox, ObjectBox
from core.prompts.examples import record_corrections
from core.queue_db import ALLOWED, InvalidTransition, QueueDB, QueueRow
from core.reports import manifest
from core.reports.measurements_csv import update_measurements
from core.reports.pdf import add_addendum, addenda_path_for, build_pdf, load_addenda, pdf_path_for
from core.reports.sidecar import read_sidecar, safe_id, sidecar_path, write_sidecar
from core.review.diffs import OBJECT_FIELDS, Change, diff_label_sets
from core.schema.class_names import COMPONENT_NAMES, ClassNameError, check_token
from core.schema.taxonomy import Taxonomy, TaxonomyError, add_entry, load_taxonomy
from core.tools.color import sample_color
from core.tools.geometry import clamp_box

HIT_SEARCH_RADIUS = 40  # a clicked hit point snaps to a detected spot this close
DEFAULT_HIT_BOX = 15
TAXONOMY_KIND = {"material": "material", "color": "color", "general_object": "object"}


class ReviewError(ValueError):
    """A review action that cannot be done; the message is written for the person at the screen."""


@dataclass
class SaveResult:
    status: str
    changes: list[Change]
    warnings: list[str]


def next_id(prefix: str, existing: Iterable[str]) -> str:
    taken = set(existing)
    n = 1
    while f"{prefix}{n}" in taken:
        n += 1
    return f"{prefix}{n}"


class ReviewService:
    def __init__(self, db: QueueDB, data_dir: str | Path, taxonomy_dir: str | Path, settings: dict) -> None:
        self.db, self.data_dir = db, Path(data_dir)
        self.taxonomy_dir, self.settings = Path(taxonomy_dir), settings

    # ------------------------------------------------------------ paths and loading
    @property
    def placeholder(self) -> str:
        return self.settings["labeling"]["color_placeholder"]

    @property
    def sidecars(self) -> Path:
        return self.data_dir / "sidecars"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.csv"

    @property
    def corrections_path(self) -> Path:
        return self.data_dir / "corrections.jsonl"

    def taxonomy(self) -> Taxonomy:
        return load_taxonomy(self.taxonomy_dir)  # reloaded each time so "add to taxonomy" shows up at once

    def row(self, image_id: str) -> QueueRow:
        row = self.db.get(image_id)
        if row is None:
            raise ReviewError(f"{image_id} is not in the queue")
        return row

    def load(self, image_id: str) -> ImageLabelSet:
        """The label set as the reviewer should see it: boxes from the sidecar, status and PDF from the queue."""
        row = self.row(image_id)
        path = Path(row.sidecar_path) if row.sidecar_path else sidecar_path(self.sidecars, image_id)
        if not path.is_file():
            raise ReviewError(f"{row.name} has no labels yet (status: {row.status})")
        ls = read_sidecar(path)
        to_upright(ls, row.path)  # an older set was measured before photos were turned upright
        ls.status = row.status  # type: ignore[assignment]
        ls.pdf_path = row.pdf_path or ls.pdf_path
        return ls

    # ------------------------------------------------------------ saving
    def _check_move(self, row: QueueRow, status: str | None) -> None:
        if status and status != row.status and status not in ALLOWED[row.status]:
            raise ReviewError(f"{row.name} is {row.status}, so it cannot become {status}")

    def _persist(self, row: QueueRow, ls: ImageLabelSet, status: str | None = None) -> None:
        self._check_move(row, status)
        ls.status = status or row.status  # type: ignore[assignment]
        path = write_sidecar(ls, self.sidecars)
        old = next((r for r in manifest.read_manifest(self.manifest_path) if r["image_id"] == ls.image_id), None)
        manifest.update_manifest(ls, self.manifest_path, str(path), old["labeled_at"] if old else None)
        update_measurements(ls, self.data_dir / "measurements.csv")
        self.db.record_review(row.image_id, ls, status, str(path))

    # ------------------------------------------------------------ editing
    def save_edit(self, image_id: str, edited: ImageLabelSet, note: str = "") -> SaveResult:
        """Store the reviewer's version. Moved laser boxes are re-measured; the change list goes to the PDF
        addendum and the corrections file. Approved or pending images become `edited`."""
        row = self.row(image_id)
        before = self.load(image_id)
        warnings: list[str] = []
        self._remeasure_moved(row, before, edited, warnings)
        changes = diff_label_sets(before, edited)
        if not changes:
            return SaveResult(row.status, [], warnings)
        status = row.status
        if status in ("pending", "approved"):
            status = "edited"
        elif status == "needs_input" and not any(q.answer is None for q in edited.questions):
            status = "pending"
        self._persist(row, edited, status)
        lines = [c.describe() for c in changes]
        add_addendum(addenda_path_for(self.data_dir, image_id), lines, status, note)
        record_corrections(self.corrections_path, image_id, row.name, changes, edited.prompt_version)
        warnings += self._try_pdf(image_id) if status in ("edited", "approved") else []
        return SaveResult(status, changes, warnings)

    def _remeasure_moved(self, row: QueueRow, before: ImageLabelSet, after: ImageLabelSet, warnings: list[str]) -> None:
        old = {b.id: b for b in before.laser_boxes}
        moved = [b.id for b in after.laser_boxes if b.id in old and tuple(old[b.id].box) != tuple(b.box)]
        moved += [b.id for b in after.laser_boxes if b.id not in old and not b.metrics]
        if not moved:
            return
        try:
            measure_boxes(after, row.path, self.settings, moved)
        except (OSError, ValueError) as exc:
            warnings.append(f"Could not re-measure the moved laser box ({exc}); old numbers were kept.")

    def approve(self, image_id: str) -> list[str]:
        row = self.row(image_id)
        if row.status == "needs_input":
            raise ReviewError("This image still has questions. Answer them first, or reject it.")
        ls = self.load(image_id)
        self._add_new_words(ls)
        self._persist(row, ls, "approved")
        return self._try_pdf(image_id)

    def clear_taxonomy_questions(self) -> tuple[int, list[str]]:
        """Images stuck on "X is not in taxonomy" questions: add each named object and material to the lists and mark
        those questions answered. A question about an object the model could not name ("unknown") stays open, and an
        image with no questions left goes back to `pending`. Returns (images freed, words added)."""
        from core.labeler.finalize import NOT_A_NAME
        freed, added = 0, []
        for row in self.db.list(["needs_input"]):
            if not (row.sidecar_path and Path(row.sidecar_path).is_file()):
                continue
            ls = self.load(row.image_id)
            changed = False
            for q in ls.questions:
                if q.answer is not None or not q.id.startswith("qt_"):
                    continue
                obj = next((o for o in ls.object_boxes if o.id == q.target_id), None)
                if obj is None or obj.general_object in NOT_A_NAME:
                    continue
                for kind, value, known in (("object", obj.general_object, self.taxonomy().objects),
                                           ("material", obj.material, self.taxonomy().materials)):
                    if value and value not in known and value not in NOT_A_NAME:
                        try:
                            add_entry(self.taxonomy_dir, kind, value)
                            added.append(f"{kind} {value}")
                        except (TaxonomyError, ClassNameError):
                            pass
                q.answer = "added to taxonomy"
                changed = True
            if changed:
                status = "needs_input" if any(q.answer is None for q in ls.questions) else "pending"
                self._persist(row, ls, status)
                freed += status == "pending"
                if row.transcript_path and Path(row.transcript_path).is_file():
                    append_event(row.transcript_path, "answer", "taxonomy questions cleared: new words added to the lists")
        return freed, added

    def reject(self, image_id: str) -> None:
        row = self.row(image_id)
        ls = self.load(image_id) if row.sidecar_path else None
        if ls is None:
            self.db.set_status(image_id, "rejected")
        else:
            self._persist(row, ls, "rejected")

    def reopen(self, image_id: str) -> None:
        """Undo an approval or rejection: the image goes back to `pending`."""
        row = self.row(image_id)
        self._persist(row, self.load(image_id), "pending")

    def bulk_approve(self, min_confidence: float) -> list[str]:
        """Approve sure images with no open questions. PDFs are built later, when needed (see ensure_pdfs)."""
        ids = self.db.bulk_approve(min_confidence)
        for image_id in ids:
            row = self.row(image_id)
            ls = self.load(image_id)
            self._persist(row, ls, None)
        return ids

    # ------------------------------------------------------------ questions
    def answer_question(self, image_id: str, question_id: str, answer: str, *, add_to_taxonomy: bool = False) -> list[str]:
        """Apply an answer to its box (when the question names one), record it, and resume the image when none are open."""
        row = self.row(image_id)
        ls = self.load(image_id)
        before = self.load(image_id)
        question = next((q for q in ls.questions if q.id == question_id), None)
        if question is None:
            raise ReviewError(f"question {question_id} not found")
        answer = answer.strip()
        if not answer:
            raise ReviewError("Please choose or type an answer.")
        warnings: list[str] = []
        if question.field_name == "hit_point":
            self._answer_hit_point(row, ls, answer)
        elif question.field_name and question.target_id:
            warnings += self._apply_field(ls, question.target_id, question.field_name, answer, add_to_taxonomy)
        question.answer = answer
        status = row.status
        if status == "needs_input" and not any(q.answer is None for q in ls.questions):
            status = "pending"
        elif status in ("pending", "approved") and question.field_name:
            status = "edited" if status == "approved" else status
        self._persist(row, ls, status)
        if row.transcript_path and Path(row.transcript_path).is_file():
            append_event(row.transcript_path, "answer", f"{question.text} -> {answer}")
        changes = diff_label_sets(before, ls)
        if changes:
            add_addendum(addenda_path_for(self.data_dir, image_id), [c.describe() for c in changes], status,
                         f"answer to: {question.text}")
            record_corrections(self.corrections_path, image_id, row.name, changes, ls.prompt_version)
        return warnings

    def _answer_hit_point(self, row: QueueRow, ls: ImageLabelSet, answer: str) -> None:
        """The answer is one or more places, "x,y" separated by ";". Each is only a place where a laser is."""
        try:
            points = [tuple(float(v) for v in part.split(",")) for part in answer.split(";") if part.strip()]
            if not points or any(len(p) != 2 for p in points):
                raise ValueError
        except ValueError:
            raise ReviewError("Click on the image where a laser is (the answer should look like 120,85, "
                              "or several places separated by ;).") from None
        add_lasers_at_points(ls, row.path, self.settings, clean_points(points))

    def _apply_field(self, ls: ImageLabelSet, target_id: str, field: str, value: str, add_to_taxonomy: bool) -> list[str]:
        obj = next((o for o in ls.object_boxes if o.id == target_id), None)
        laser = next((b for b in ls.laser_boxes if b.id == target_id), None)
        warnings: list[str] = []
        if obj is not None and field in OBJECT_FIELDS:
            if field == "incidence" and value not in INCIDENCES:
                raise ReviewError(f"Incidence must be one of: {', '.join(INCIDENCES)}.")
            try:
                if field != "finish":
                    check_token(value, field.replace("_", " "))
            except ClassNameError as exc:
                raise ReviewError(str(exc)) from None
            if field in TAXONOMY_KIND:
                warnings += self._taxonomy_for(obj, field, value, add_to_taxonomy)
            setattr(obj, field, value)
        elif laser is not None and field in ("kind", "event"):
            if value not in LASER_KINDS:
                raise ReviewError(f"Laser event must be one of: {', '.join(LASER_KINDS)}.")
            laser.kind = value  # type: ignore[assignment]
        return warnings

    def _taxonomy_for(self, obj: ObjectBox, field: str, value: str, add: bool) -> list[str]:
        tax = self.taxonomy()
        known = {"material": tax.materials, "color": [*tax.colors, self.placeholder],
                 "general_object": tax.objects}[field]
        if value in known:
            return []
        if add:  # the reviewer ticked "add to taxonomy": always honoured
            try:
                add_entry(self.taxonomy_dir, TAXONOMY_KIND[field], value)
            except (TaxonomyError, ClassNameError) as exc:
                raise ReviewError(str(exc)) from None
            return []
        if self._match_mode() == "off" or (field != "general_object" and self._only_object()):
            return []  # not matched to the lists (off: a new word is added when the label is approved)
        return [f"{field.replace('_', ' ')} '{value}' is not in your taxonomy lists."]

    def _match_mode(self) -> str:
        return self.settings["labeling"].get("taxonomy_match", "object")

    def _only_object(self) -> bool:
        return self._match_mode() != "all"

    def _add_new_words(self, ls: ImageLabelSet) -> None:
        """Taxonomy match off: a new object or material on an approved label goes onto the lists."""
        if self._match_mode() != "off":
            return
        tax = self.taxonomy()
        for o in ls.object_boxes:
            for kind, value, known in (("object", o.general_object, tax.objects), ("material", o.material, tax.materials)):
                if value and value not in known:
                    try:
                        tax = add_entry(self.taxonomy_dir, kind, value)
                    except (TaxonomyError, ClassNameError):
                        pass

    def taxonomy_problems(self, ls: ImageLabelSet) -> list[str]:
        """Values on this label set that are not in the taxonomy lists (soft problems; approval still works)."""
        tax = self.taxonomy()
        out = []
        if self._match_mode() == "off":
            return out
        for o in ls.object_boxes:
            out += [f"{o.id}: {p}" for p in tax.check_object(o.general_object, o.material, o.color, o.incidence,
                                                              self.placeholder, only_object=self._only_object())]
        return out

    def add_to_taxonomy(self, kind: str, value: str) -> Taxonomy:
        try:
            return add_entry(self.taxonomy_dir, kind, value)
        except (TaxonomyError, ClassNameError) as exc:
            raise ReviewError(str(exc)) from None

    # ------------------------------------------------------------ re-run and PDF
    def rerun(self, image_id: str, note: str = "", points: Iterable = ()) -> None:
        """Label this image again, passing the reviewer's note and marked places to the model. The old answer is replaced.

        `points` are places (x, y in original pixels) where the reviewer says a laser is. They are only places to
        check: the labeler measures each spot from the pixels and the model decides what it is."""
        row = self.row(image_id)
        self._check_move(row, "queued")
        kept = clean_points(points)  # refuse bad input before anything is changed
        if note.strip():
            set_rerun_note(self.data_dir, image_id, note)
        set_rerun_points(self.data_dir, image_id, kept)
        self.db.requeue([image_id])

    def rebuild_pdf(self, image_id: str) -> Path:
        row = self.row(image_id)
        ls = self.load(image_id)
        out = build_pdf(ls, row.path, pdf_path_for(self.data_dir, image_id), transcript_path=row.transcript_path or None,
                        addenda=load_addenda(addenda_path_for(self.data_dir, image_id)), placeholder=self.placeholder)
        ls.pdf_path = str(out)
        self._persist(row, ls, None)
        return out

    def _try_pdf(self, image_id: str) -> list[str]:
        try:
            self.rebuild_pdf(image_id)
            return []
        except Exception as exc:  # a PDF problem must never block approving a label
            return [f"The PDF could not be built ({exc}). Use Rebuild PDF to try again."]

    def pdf_is_current(self, row: QueueRow) -> bool:
        pdf = Path(row.pdf_path) if row.pdf_path else None
        side = Path(row.sidecar_path) if row.sidecar_path else None
        return bool(pdf and side and pdf.is_file() and side.is_file() and pdf.stat().st_mtime >= side.stat().st_mtime - 1)

    def ensure_pdfs(self, image_ids: Iterable[str]) -> list[str]:
        """Build the PDF of every listed image that has none or an out-of-date one. Returns problems."""
        problems = []
        for image_id in image_ids:
            if not self.pdf_is_current(self.row(image_id)):
                problems += [f"{image_id}: {w}" for w in self._try_pdf(image_id)]
        return problems

    # ------------------------------------------------------------ upload support
    def approved_sets(self) -> list[ImageLabelSet]:
        sets = []
        for row in self.db.list(["approved", "edited"]):
            if row.sidecar_path and Path(row.sidecar_path).is_file():
                sets.append(self.load(row.image_id))
        return sets

    def local_image_ids(self) -> list[str]:
        """Approved images that came from a local file: they have no Roboflow image id to send labels to."""
        return [r.image_id for r in self.db.list(["approved", "edited"]) if r.source != "roboflow"]

    def mark_uploaded(self, image_ids: Iterable[str]) -> int:
        count = 0
        for image_id in image_ids:
            self.db.set_status(image_id, "uploaded")
            manifest.set_status(self.manifest_path, image_id, "uploaded")
            count += 1
        return count


# ---------------------------------------------------------------- re-run notes
def _notes_file(data_dir: str | Path) -> Path:
    return Path(data_dir) / "rerun_notes.json"


def _read_notes(data_dir: str | Path) -> dict[str, str]:
    try:
        data = json.loads(_notes_file(data_dir).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def set_rerun_note(data_dir: str | Path, image_id: str, note: str) -> None:
    notes = _read_notes(data_dir)
    notes[image_id] = note.strip()
    path = _notes_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notes, indent=2), encoding="utf-8")


def _points_file(data_dir: str | Path) -> Path:
    return Path(data_dir) / "rerun_points.json"


def _read_points(data_dir: str | Path) -> dict[str, list[list[float]]]:
    try:
        data = json.loads(_points_file(data_dir).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def clean_points(points: Iterable) -> list[tuple[float, float]]:
    """Points as (x, y) floats in original pixels; anything that is not a finite pair is refused."""
    out: list[tuple[float, float]] = []
    for point in points:
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            raise ReviewError("A marked place must be an x, y pair of numbers.") from None
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ReviewError("A marked place must be an x, y pair of numbers.")
        out.append((x, y))
    return out


def set_rerun_points(data_dir: str | Path, image_id: str, points: Iterable) -> None:
    """Remember the places a reviewer marked ("a laser is here") for this image's next run."""
    marks = _read_points(data_dir)
    kept = clean_points(points)
    if kept:
        marks[image_id] = [[x, y] for x, y in kept]
    else:
        marks.pop(image_id, None)
    path = _points_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marks, indent=2), encoding="utf-8")


def take_rerun_points(data_dir: str | Path, image_id: str) -> list[tuple[float, float]]:
    """The marked places waiting for this image, removed so they are used once."""
    marks = _read_points(data_dir)
    found = marks.pop(image_id, [])
    if found:
        _points_file(data_dir).write_text(json.dumps(marks, indent=2), encoding="utf-8")
    try:
        return clean_points(found)
    except ReviewError:
        return []


def take_rerun_note(data_dir: str | Path, image_id: str) -> str:
    """The pending note for this image, removed so it is used once."""
    notes = _read_notes(data_dir)
    note = notes.pop(image_id, "")
    if note:
        _notes_file(data_dir).write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return note


# ---------------------------------------------------------------- box editing helpers (in memory)
def measure_boxes(ls: ImageLabelSet, image_path: str | Path, settings: dict, only_ids: Iterable[str] | None = None) -> None:
    """Re-measure laser boxes from the pixels. Boxes not in `only_ids` keep their numbers."""
    image = load_image(image_path)
    taxonomy_stub = Taxonomy()
    ctx = build_context(image, taxonomy_stub, max_image_px=settings["provider"]["max_image_px"],
                        color_placeholder=settings["labeling"]["color_placeholder"])
    raw = [{"id": b.id, "kind": b.kind, "box": b.box, "order": b.order, "from_id": b.from_id, "notes": b.notes}
           for b in ls.laser_boxes]
    wanted = set(only_ids) if only_ids is not None else None
    for fresh in _measure_lasers(ctx, raw):
        target = next(b for b in ls.laser_boxes if b.id == fresh.id)
        if wanted is None or fresh.id in wanted:
            target.metrics, target.box = fresh.metrics, fresh.box
    if not ls.image_width:
        ls.image_width, ls.image_height = image.width, image.height


def add_laser_box(ls: ImageLabelSet, image_path: str | Path, settings: dict, box, kind: str = "first_incidence",
                  component: str | None = None) -> LaserBox:
    """A reviewer-drawn laser box. With the components or merged scheme its class follows its event:
    a direct hit (first_incidence) is component1, any bounce is component2."""
    if kind not in LASER_KINDS:
        raise ReviewError(f"Laser event must be one of: {', '.join(LASER_KINDS)}.")
    if component is not None and component not in COMPONENT_NAMES:
        raise ReviewError(f"Component must be one of: {', '.join(COMPONENT_NAMES)}.")
    if component is None and settings["labeling"].get("scheme", "full") in ("components", "merged"):
        component = component_for_kind(kind)
    width, height = ls.image_width or 10**6, ls.image_height or 10**6
    previous = max(ls.laser_boxes, key=lambda b: b.order, default=None)
    new = LaserBox(next_id("l", (b.id for b in ls.laser_boxes)), kind, clamp_box(box, width, height),  # type: ignore[arg-type]
                   previous.order + 1 if previous else 0,
                   previous.id if previous and kind != "first_incidence" else None,
                   component=component)
    ls.laser_boxes.append(new)
    measure_boxes(ls, image_path, settings, [new.id])
    # it hits the smallest object box its centre is inside; under "merged" that is what puts the object in its label
    cx, cy = new.box[0] + new.box[2] / 2, new.box[1] + new.box[3] / 2
    inside = [o for o in ls.object_boxes if o.box[0] <= cx <= o.box[0] + o.box[2] and o.box[1] <= cy <= o.box[1] + o.box[3]]
    if inside:
        min(inside, key=lambda o: o.box[2] * o.box[3]).laser_ids.append(new.id)
    return new


def apply_scheme(ls: ImageLabelSet, scheme: str) -> bool:
    """Switch an already labeled image to `scheme` without calling the model. Returns True if anything changed.

    Spots get component1 / component2 from their event under components and merged, and lose it under full."""
    if scheme not in SCHEMES:
        raise ReviewError(f"Label scheme must be one of: {', '.join(SCHEMES)}.")
    before = (ls.scheme, [b.component for b in ls.laser_boxes])
    ls.scheme = scheme
    for b in ls.laser_boxes:
        b.component = component_for_kind(b.kind) if scheme in ("components", "merged") else None
    return before != (ls.scheme, [b.component for b in ls.laser_boxes])


def add_lasers_at_points(ls: ImageLabelSet, image_path: str | Path, settings: dict,
                         points: Sequence[tuple[float, float]]) -> list[LaserBox]:
    """A laser box for each place a reviewer clicked. A click only says where a laser is, not how big it is: each click
    snaps to a detected spot when one is close (its measured box is used) and otherwise gets a small square.
    The first click is taken as the direct hit and the others as "other" (change the event in the editor).
    Two clicks on the same spot make one box."""
    image = load_image(image_path)
    ctx = build_context(image, Taxonomy(), max_image_px=settings["provider"]["max_image_px"],
                        max_candidates=settings["run"]["max_candidates"])
    ls.image_width, ls.image_height = ls.image_width or image.width, ls.image_height or image.height
    made: list[LaserBox] = []
    boxes: list[tuple] = []
    half = DEFAULT_HIT_BOX / 2
    for x, y in points:
        best, best_d = None, HIT_SEARCH_RADIUS
        for cand in ctx.detection.candidates:
            d = ((cand.center[0] - x) ** 2 + (cand.center[1] - y) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = cand, d
        box = tuple(best.box) if best else (x - half, y - half, DEFAULT_HIT_BOX, DEFAULT_HIT_BOX)
        if any(_overlap(box, other) > 0.5 for other in boxes):
            continue
        boxes.append(box)
        kind = "first_incidence" if not made else "other"
        made.append(add_laser_box(ls, image_path, settings, box, kind))
    ls.questions = [q for q in ls.questions if q.id != "q_no_laser" or q.answer is not None]
    return made


def add_laser_at_point(ls: ImageLabelSet, image_path: str | Path, settings: dict, x: float, y: float) -> LaserBox:
    """A laser box where the user clicked: snapped to a detected spot if one is close, else a small square."""
    made = add_lasers_at_points(ls, image_path, settings, [(x, y)])
    if made:
        return made[0]
    raise ReviewError("There is already a laser box there.")


def _overlap(a, b) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def add_object_box(ls: ImageLabelSet, image_path: str | Path, taxonomy: Taxonomy, box, settings: dict) -> ObjectBox:
    """A reviewer-drawn object box: first taxonomy values as a start, color from the pixels, lasers whose centre is inside."""
    width, height = ls.image_width or 10**6, ls.image_height or 10**6
    clamped = clamp_box(box, width, height)
    placeholder = settings["labeling"]["color_placeholder"]
    first_object = next(iter(taxonomy.objects), "other")
    color = placeholder
    if not ls.is_grayscale:
        sample = sample_color(load_image(image_path).rgb, clamped, allowed_colors=taxonomy.colors, placeholder=placeholder)
        color = sample.name or (taxonomy.colors[0] if taxonomy.colors else placeholder)
    x, y, w, h = clamped
    inside = [b.id for b in ls.laser_boxes if x <= b.box[0] + b.box[2] / 2 <= x + w and y <= b.box[1] + b.box[3] / 2 <= y + h]
    obj = ObjectBox(next_id("o", (o.id for o in ls.object_boxes)), clamped, first_object,  # type: ignore[arg-type]
                    (taxonomy.materials or ["other"])[0], color,
                    "first", inside, None, 1.0, "added by reviewer")
    ls.object_boxes.append(obj)
    return obj


def delete_box(ls: ImageLabelSet, box_id: str) -> None:
    ls.object_boxes = [o for o in ls.object_boxes if o.id != box_id]
    ls.laser_boxes = [b for b in ls.laser_boxes if b.id != box_id]
    for o in ls.object_boxes:
        o.laser_ids = [i for i in o.laser_ids if i != box_id]
    for b in ls.laser_boxes:
        if b.from_id == box_id:
            b.from_id = None
