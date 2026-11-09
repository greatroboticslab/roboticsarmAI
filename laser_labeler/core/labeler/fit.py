"""Fit the request to a small context window.

Local servers (Ollama's default window is 4096 tokens) refuse a request that is too long and say how long it was.
When that happens the labeler retries with a smaller request: fewer and smaller crops first, then a smaller whole
image. What worked is remembered for the rest of the run, so only the first image pays for the retries.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RequestSize:
    max_image_px: int
    max_candidates: int
    crop_px: int

    def describe(self) -> str:
        return f"whole image {self.max_image_px}px, {self.max_candidates} spot(s), crops {self.crop_px}px"


class FitMemory:
    """The smaller size that fitted, shared by the images of one run."""
    def __init__(self) -> None:
        self.size: RequestSize | None = None


def shrink(size: RequestSize) -> RequestSize | None:
    """One step smaller, or None when already at the floor (the instructions alone are then most of the window)."""
    if size.crop_px > 192 or size.max_candidates > 3:
        return replace(size, crop_px=min(size.crop_px, max(192, int(size.crop_px * 0.67))),
                       max_candidates=min(size.max_candidates, 3))
    if size.max_image_px > 512:
        return replace(size, max_image_px=max(512, int(size.max_image_px * 0.75)))
    if size.max_candidates > 2 or size.crop_px > 128:
        return replace(size, max_candidates=min(size.max_candidates, 2), crop_px=min(size.crop_px, 128))
    if size.max_image_px > 384:
        return replace(size, max_image_px=384)
    if size.max_candidates > 1:
        return replace(size, max_candidates=1)
    return None
