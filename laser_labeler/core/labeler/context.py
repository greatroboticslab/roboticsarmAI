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
from core.models import Question
from core.schema.taxonomy import Taxonomy
from core.tools.crop import downscale_to_max
from core.tools.laser_detect import Detection, detect_laser


@dataclass
class LabelContext:
    image: ImageData
    stats: ImageStats
    taxonomy: Taxonomy
    shown: np.ndarray
    factor: float
    color_placeholder: str = "na"
    max_candidates: int = 6
    questions: list[Question] = field(default_factory=list)  # filled by the ask_user tool
    user_note: str = ""  # reviewer's note when an image is re-run
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
        return self._detection

    def to_original(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        return tuple(v / self.factor for v in box)  # type: ignore[return-value]

    def to_shown(self, box: Sequence[float]) -> list[int]:
        return [int(round(v * self.factor)) for v in box]


def build_context(image: ImageData, taxonomy: Taxonomy, *, max_image_px: int = 1024,
                  color_placeholder: str = "na", max_candidates: int = 6) -> LabelContext:
    shown, factor = downscale_to_max(image.rgb, max_image_px)
    return LabelContext(image, image_stats(image.rgb), taxonomy, shown, factor,
                        color_placeholder, max_candidates)
