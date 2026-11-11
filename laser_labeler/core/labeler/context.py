"""Per-image working state shared by the tools, agent mode and pipeline mode.

The AI sees a downscaled copy of the image. Every coordinate that crosses the
AI boundary is in "shown" pixels; everything stored or measured is in original
pixels. `factor` = shown size / original size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from core.camera import ImageData, ImageStats, image_stats, intensity_map
from core.labeler.coords import resolve_space, to_shown_px
from core.models import Question
from core.schema.taxonomy import Taxonomy
from core.tools.crop import downscale_to_max
from core.tools.laser_detect import Detection, apply_marks, detect_laser


@dataclass
class LabelContext:
    image: ImageData
    stats: ImageStats
    taxonomy: Taxonomy
    shown: np.ndarray
    factor: float
    color_placeholder: str = "na"
    max_candidates: int = 6
    crop_px: int = 384  # longest side of the zoomed crop sent with each candidate (pipeline mode)
    taxonomy_match: str = "object"  # "off": reuse matches, add new values; "object": only the object is matched; "all": every value
    taxonomy_added: list[tuple[str, str]] = field(default_factory=list)  # ("object" | "material", value) added in "off" mode
    questions: list[Question] = field(default_factory=list)  # filled by the ask_user tool
    user_note: str = ""  # reviewer's note when an image is re-run
    hint_points: list[tuple[float, float]] = field(default_factory=list)  # places the reviewer clicked: "a laser is here" (original px)
    coordinates: str = "auto"  # the space the model's boxes are written in: auto | pixels | thousand | fraction
    coordinates_used: str = "pixels"  # what the last reply was read as (set by `from_model`)
    coordinates_why: str = ""
    _detection: Detection | None = None
    _intensity: np.ndarray | None = None

    @property
    def rgb(self) -> np.ndarray:
        return self.image.rgb

    @property
    def intensity(self) -> np.ndarray:
        if self._intensity is None:
            self._intensity = intensity_map(self.rgb)
        return self._intensity

    @property
    def detection(self) -> Detection:
        if self._detection is None:
            self._detection = detect_laser(self.rgb, self.stats, max_candidates=self.max_candidates)
            if self.hint_points:  # the reviewer pointed at lasers: they are looked at first and always shown to the model
                marked = apply_marks(self._detection, self.rgb, self.stats, self.hint_points)
                self.max_candidates = max(self.max_candidates, min(12, len(marked)))
        return self._detection

    def to_original(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        return tuple(v / self.factor for v in box)  # type: ignore[return-value]

    def read_model_boxes(self, laser_boxes: Sequence[Sequence[float]], other_boxes: Sequence[Sequence[float]]) -> None:
        """Decide which space this reply's boxes are in (see core/labeler/coords.py); `from_model` then uses it."""
        spots = [(c.box[0] * self.factor + c.box[2] * self.factor / 2.0, c.box[1] * self.factor + c.box[3] * self.factor / 2.0)
                 for c in self.detection.candidates]
        self.coordinates_used, self.coordinates_why = resolve_space(
            laser_boxes, other_boxes, spots, self.shown.shape[1], self.shown.shape[0], self.coordinates)

    def from_model(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        """A box as the model wrote it -> original pixels."""
        return self.to_original(to_shown_px(box, self.coordinates_used, self.shown.shape[1], self.shown.shape[0]))

    def to_shown(self, box: Sequence[float]) -> list[int]:
        return [int(round(v * self.factor)) for v in box]


def build_context(image: ImageData, taxonomy: Taxonomy, *, max_image_px: int = 1024,
                  color_placeholder: str = "na", max_candidates: int = 6, crop_px: int = 384) -> LabelContext:
    shown, factor = downscale_to_max(image.rgb, max_image_px)
    return LabelContext(image, image_stats(image.rgb), taxonomy, shown, factor,
                        color_placeholder, max_candidates, crop_px)
