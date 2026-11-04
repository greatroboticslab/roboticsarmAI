"""Core dataclasses (build spec section 12). Everything else imports these."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

from core.schema.class_names import COMPONENT_NAMES, encode_component_class, encode_laser_class

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
    # Label scheme "components": which part of a laser spot this box is (component1 = core, component2 = halo,
    # component3 = beam). When set, it is the Roboflow class; `kind` stays as information about the event.
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
    def is_part(self) -> bool:
        """A halo or beam box: it belongs to the core box named by from_id and is not measured as a spot."""
        return self.component in ("component2", "component3")

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
    general_object: str
    sub_object: str
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

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageLabelSet":
        data = dict(data)
        data["laser_boxes"] = [LaserBox.from_dict(d) for d in data.get("laser_boxes", [])]
        data["object_boxes"] = [ObjectBox.from_dict(d) for d in data.get("object_boxes", [])]
        data["questions"] = [Question.from_dict(d) for d in data.get("questions", [])]
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "ImageLabelSet":
        return cls.from_dict(json.loads(text))
