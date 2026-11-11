"""Core dataclasses (build spec section 12). Everything else imports these."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

from core.schema.class_names import (COMPONENT_NAMES, encode_component_class, encode_laser_class, encode_merged_class,
                                     encode_object_class, resolve_color)

LaserKind = Literal["first_incidence", "reflection", "refraction", "scatter", "other"]
Status = Literal[
    "queued", "labeling", "needs_input", "failed",
    "pending", "approved", "edited", "rejected", "uploaded",
]
Box = tuple[int, int, int, int]  # x, y, w, h in pixels

LASER_KINDS: tuple[str, ...] = get_args(LaserKind)
STATUSES: tuple[str, ...] = get_args(Status)
INCIDENCES: tuple[str, ...] = ("first", "reflected", "refracted", "scattered", "other")


def as_box(value: Any) -> Box:
    """Coerce a list/tuple of four numbers into an int box tuple."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"box must have 4 numbers (x, y, w, h), got {value!r}")
    return tuple(int(round(v)) for v in value)  # type: ignore[return-value]


@dataclass
class LaserBox:
    id: str
    kind: LaserKind
    box: Box
    order: int  # position along the beam path
    from_id: str | None = None  # previous laser box in the path
    metrics: dict[str, float] = field(default_factory=dict)  # section 4, raw and normalized
    notes: str = ""
    # Label schemes "components" and "merged": component1 = direct hit, component2 = bounce. When set it is the
    # Roboflow class; `kind` stays as information about the event.
    component: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in LASER_KINDS:
            raise ValueError(f"unknown laser kind {self.kind!r}")
        if self.component is not None and self.component not in COMPONENT_NAMES:
            raise ValueError(f"unknown laser component {self.component!r}")
        if self.order < 0:
            raise ValueError("order must be >= 0")
        self.box = as_box(self.box)

    @property
    def event_name(self) -> str:
        """Class-name form of the kind, e.g. first_incidence -> first-incidence."""
        return self.kind.replace("_", "-")

    @property
    def class_name(self) -> str:
        """The Roboflow class name of this box."""
        return encode_component_class(self.component) if self.component else encode_laser_class(self.event_name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LaserBox":
        return cls(**data)


@dataclass
class ObjectBox:
    id: str
    box: Box
    general_object: str  # "the object", for example pen
    material: str
    color: str  # "na" (or the configured placeholder) on grayscale
    incidence: str  # first / reflected / refracted / scattered / other
    laser_ids: list[str] = field(default_factory=list)  # laser boxes that hit this region
    finish: str | None = None
    confidence: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.incidence not in INCIDENCES:
            raise ValueError(f"unknown incidence {self.incidence!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.box = as_box(self.box)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectBox":
        data = {k: v for k, v in data.items() if k != "sub_object"}  # older sidecars had a sub-object; it is gone
        return cls(**data)


@dataclass
class Question:
    """A question parked for the user. Never blocks the run (spec section 5, step 8)."""
    id: str
    text: str
    target_id: str | None = None  # box id the question is about
    field_name: str | None = None  # attribute name, e.g. "material"
    choices: list[str] = field(default_factory=list)
    allow_taxonomy_add: bool = False  # offer "add to taxonomy"
    answer: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Question":
        return cls(**data)


@dataclass
class ImageLabelSet:
    image_id: str
    image_name: str
    camera_id: str
    is_grayscale: bool
    background_level: float
    exposure: dict[str, float] | None = None
    laser_boxes: list[LaserBox] = field(default_factory=list)
    object_boxes: list[ObjectBox] = field(default_factory=list)
    prompt_version: str = ""
    provider: str = ""
    model: str = ""
    transcript_path: str = ""
    pdf_path: str | None = None
    status: Status = "queued"
    # Not in the spec's dataclass, but COCO export needs the image size.
    image_width: int = 0
    image_height: int = 0
    questions: list[Question] = field(default_factory=list)
    run_mode: str = ""  # "pipeline" or "agent": the mode that produced this set
    notes: str = ""
    scheme: str = "full"  # label scheme it was made with: full | components | merged
    pixel_space: str = "stored"  # "upright" = boxes are on the picture turned upright by its EXIF orientation (as Roboflow shows it); "stored" = older sets, measured before turning
    scheme_version: int = 2  # 1 = components were core/halo/beam parts (migrated on load), 2 = direct hit / bounce

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")

    def object_for_laser(self, laser: "LaserBox") -> "ObjectBox | None":
        """The object box a laser spot lands on: the one that lists it, else the one its centre is inside (the
        smallest), so a spot nobody linked still gets its object in a merged name."""
        linked = next((o for o in self.object_boxes if laser.id in o.laser_ids), None)
        if linked is not None:
            return linked
        cx, cy = laser.box[0] + laser.box[2] / 2, laser.box[1] + laser.box[3] / 2
        inside = [o for o in self.object_boxes
                  if o.box[0] <= cx <= o.box[0] + o.box[2] and o.box[1] <= cy <= o.box[1] + o.box[3]]
        return min(inside, key=lambda o: o.box[2] * o.box[3]) if inside else None

    def laser_class_name(self, laser: "LaserBox", color_placeholder: str = "na") -> str:
        """The Roboflow class of a laser box. Under the "merged" scheme a part that lands on an object is named
        after both, e.g. obj_pen-ballpoint_rubber_red_first__component1; otherwise it is the plain laser class."""
        obj = self.object_for_laser(laser) if self.scheme == "merged" and laser.component else None
        if obj is None:
            return laser.class_name
        color = resolve_color(obj.color, self.is_grayscale, color_placeholder)
        return encode_merged_class(
            encode_object_class(obj.general_object, obj.material, color, obj.incidence), laser.component)

    def merged_object_ids(self) -> set[str]:
        """Object boxes whose information is carried by laser part boxes (so they are not exported separately)."""
        if self.scheme != "merged":
            return set()
        found = (self.object_for_laser(b) for b in self.laser_boxes if b.component)
        return {o.id for o in found if o is not None}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageLabelSet":
        data = dict(data)
        if "scheme_version" not in data:
            _migrate_parts(data)
            data["scheme_version"] = 2
        data["laser_boxes"] = [LaserBox.from_dict(d) for d in data.get("laser_boxes", [])]
        data["object_boxes"] = [ObjectBox.from_dict(d) for d in data.get("object_boxes", [])]
        data["questions"] = [Question.from_dict(d) for d in data.get("questions", [])]
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "ImageLabelSet":
        return cls.from_dict(json.loads(text))


def _migrate_parts(data: dict[str, Any]) -> None:
    """Old components/merged sidecars had halo (component2) and beam (component3) boxes next to each core.

    Works on the raw dict, because component3 is no longer a valid class. The halo and beam boxes are dropped,
    questions about them move to the core they belonged to, and every remaining spot gets direct hit / bounce
    from its kind.
    """
    if data.get("scheme") not in ("components", "merged"):
        return
    lasers = [dict(d) for d in data.get("laser_boxes", [])]
    parts = {d["id"]: d for d in lasers if d.get("component") in ("component2", "component3") and d.get("from_id")}
    for d in lasers:
        if d.get("component") is not None:
            direct = str(d.get("kind", "")).replace("-", "_") == "first_incidence"
            d["component"] = "component1" if direct else "component2"
    data["laser_boxes"] = [d for d in lasers if d["id"] not in parts]
    if parts:
        data["object_boxes"] = [{**o, "laser_ids": [i for i in o.get("laser_ids", []) if i not in parts]}
                                for o in data.get("object_boxes", [])]
        data["questions"] = [{**q, "target_id": parts[q["target_id"]]["from_id"]} if q.get("target_id") in parts else q
                             for q in data.get("questions", [])]
