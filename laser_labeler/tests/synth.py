"""Synthetic test images with known spots, halos and objects."""
from __future__ import annotations

import numpy as np


def scene(w=320, h=240, bg=40, noise=1.5, seed=0, gray=False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = bg + rng.normal(0, noise, (h, w))
    img = np.repeat(base[..., None], 3, axis=2)
    if not gray:
        img += rng.normal(0, 3.0, (h, w, 3))
    return img


def rect(img, x, y, w, h, color):
    img[y:y + h, x:x + w] = np.array(color, float)


def spot(img, cx, cy, sigma=3.0, amp=0.8, sigma_y=None, halo_amp=0.0, halo_sigma=12.0,
         color=(1.0, 0.25, 0.25)):
    """Add a Gaussian spot (+ optional broad halo). amp is a fraction of 255 added to the brightest channel."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    sy = sigma_y or sigma
    core = amp * np.exp(-(((xx - cx) ** 2) / (2 * sigma ** 2) + ((yy - cy) ** 2) / (2 * sy ** 2)))
    halo = halo_amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * halo_sigma ** 2)))
    light = (core + halo) * 255.0
    for c in range(3):
        img[..., c] += light * color[c]


def finish(img) -> np.ndarray:
    return np.clip(np.round(img), 0, 255).astype(np.uint8)
