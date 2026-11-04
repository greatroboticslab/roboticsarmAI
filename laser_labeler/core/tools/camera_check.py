"""camera_check: does background brightness drift per camera? (spec section 4.2)"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from core.camera import camera_id_for, image_stats, load_image

DRIFT_CV = 0.10  # background varying by more than 10% of its mean is flagged


@dataclass
class CameraReport:
    camera_id: str
    images: int
    background_mean: float
    background_std: float
    background_min: float
    background_max: float
    variation: float  # std / mean
    noise_mean: float
    with_exposure_data: int
    drifting: bool


def camera_check(
    paths: Sequence[str | Path],
    tags_by_path: Mapping[str, Sequence[str]] | None = None,
    *,
    sample: int | None = 50,
    seed: int = 0,
    source: str = "tag",
    tag_prefix: str = "camera:",
    drift_cv: float = DRIFT_CV,
) -> list[CameraReport]:
    """Measure background level per camera over a random sample of images."""
    chosen = list(paths)
    if sample and len(chosen) > sample:
        chosen = random.Random(seed).sample(chosen, sample)
    groups: dict[str, list[tuple[float, float, bool]]] = {}
    for path in chosen:
        image = load_image(path)
        cam = camera_id_for(path, (tags_by_path or {}).get(str(path), ()),
                            is_grayscale=image.is_grayscale, source=source, tag_prefix=tag_prefix)
        stats = image_stats(image.rgb)
        groups.setdefault(cam, []).append((stats.background_level, stats.noise_level, image.exposure is not None))
    return [_report(cam, rows, drift_cv) for cam, rows in sorted(groups.items())]


def _report(cam: str, rows, drift_cv: float) -> CameraReport:
    bg = np.array([r[0] for r in rows])
    mean = float(bg.mean())
    variation = float(bg.std() / mean) if mean > 0 else 0.0
    return CameraReport(cam, len(rows), mean, float(bg.std()), float(bg.min()), float(bg.max()), variation,
                        float(np.mean([r[1] for r in rows])), sum(r[2] for r in rows),
                        len(rows) > 1 and variation > drift_cv)


def format_report(reports: Sequence[CameraReport]) -> str:
    lines = ["camera       images  bg mean  bg std  variation  exposure data  verdict"]
    for r in reports:
        verdict = "DRIFTING: lock exposure/gain if you can" if r.drifting else "steady"
        lines.append(f"{r.camera_id:<12} {r.images:>6}  {r.background_mean:>7.3f}  {r.background_std:>6.3f}  "
                     f"{r.variation:>8.1%}  {r.with_exposure_data:>13}  {verdict}")
    return "\n".join(lines)
