"""Image loading, grayscale detection, camera id, exposure and background level (spec 4.1, 4.2).

Intensity is the brightest colour channel scaled to 0..1 (a red or green laser
spot still reads as bright). Pictures are kept as RGB uint8 arrays; grayscale
files are expanded to three equal channels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import math

import cv2
import numpy as np
from PIL import Image

EXIF_IFD = 0x8769
EXIF_FIELDS = {0x829A: "exposure_time", 0x829D: "f_number", 0x8827: "iso", 0x9201: "shutter_speed_value"}
GRAY_TOLERANCE = 2  # max channel difference (0..255) still counted as grayscale


@dataclass
class ImageData:
    path: str
    rgb: np.ndarray  # H x W x 3 uint8
    is_grayscale: bool
    exposure: dict[str, float] | None

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


@dataclass(frozen=True)
class ImageStats:
    background_level: float  # median intensity, 0..1
    noise_level: float  # robust spread of intensity, 0..1


def intensity_map(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel intensity 0..1: the brightest channel."""
    return rgb.max(axis=2).astype(np.float32) / 255.0


def is_grayscale_array(rgb: np.ndarray, tolerance: int = GRAY_TOLERANCE) -> bool:
    if rgb.ndim == 2:
        return True
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return bool(np.abs(r - g).max() <= tolerance and np.abs(g - b).max() <= tolerance)


def _local_noise(v: np.ndarray) -> float:
    """Sensor noise from what is left after removing the picture's own structure: each pixel minus the mean of
    its 3x3 neighbourhood. A textured table or a lit object spreads the whole image's brightness out, which says
    nothing about noise; this does not see it. The 3x3 mean takes 1/9 of a pixel's variance with it, hence the factor."""
    if min(v.shape) < 3:
        return float(np.std(v))
    k = np.ones((3, 3), np.float32) / 9.0
    residual = v - cv2.filter2D(v.astype(np.float32), -1, k, borderType=cv2.BORDER_REPLICATE)
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))) / math.sqrt(8.0 / 9.0))


def image_stats(rgb: np.ndarray) -> ImageStats:
    """Background = median brightness. The laser covers few pixels, so the median ignores it.
    Noise comes from pixel-to-pixel differences, not from the spread of the whole picture."""
    v = intensity_map(rgb)
    return ImageStats(background_level=float(np.median(v)), noise_level=max(_local_noise(v), 1.0 / 255.0))


def _to_uint8(pil: Image.Image) -> np.ndarray:
    arr = np.asarray(pil)
    if arr.dtype == np.uint16:
        return (arr // 257).astype(np.uint8)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def read_exposure(pil: Image.Image) -> dict[str, float] | None:
    """EXIF exposure data when present (webcam captures usually have none)."""
    try:
        exif = pil.getexif()
        merged = {**dict(exif), **dict(exif.get_ifd(EXIF_IFD))}
    except Exception:  # damaged EXIF is not worth failing an image over
        return None
    found = {}
    for tag, name in EXIF_FIELDS.items():
        if tag in merged:
            try:
                found[name] = float(merged[tag])
            except (TypeError, ValueError):
                continue
    return found or None


def load_image(path: str | Path) -> ImageData:
    with Image.open(path) as pil:
        pil.load()
        exposure = read_exposure(pil)
        mode_gray = pil.mode in ("L", "LA", "I;16", "I")
        arr = _to_uint8(pil if pil.mode in ("L", "I;16", "I") else pil.convert("RGB"))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return ImageData(str(path), arr, mode_gray or is_grayscale_array(arr), exposure)


def camera_id_for(
    path: str | Path,
    tags: Iterable[str] = (),
    *,
    is_grayscale: bool = False,
    source: str = "tag",
    tag_prefix: str = "camera:",
    default: str = "unknown",
) -> str:
    """Grayscale images are their own camera; otherwise a Roboflow tag or the folder name."""
    if is_grayscale:
        return "grayscale"
    if source == "tag":
        for tag in tags:
            if tag.startswith(tag_prefix) and len(tag) > len(tag_prefix):
                return tag[len(tag_prefix):]
    elif source == "folder":
        return Path(path).parent.name or default
    return default
