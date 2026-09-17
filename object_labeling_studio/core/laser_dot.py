"""
[WIRED] Locates a laser pointer's dot in a photo, for auto-placing a
bounding box at that spot on Roboflow (see
vision.storage.roboflow_export.upload_yolo_box_annotation) instead of
manually clicking there in Roboflow's UI every time.

HOW THIS WORKS — AND WHY IT'S A HEURISTIC, NOT A CERTAINTY
-------------------------------------------------------------
There is no reliable, universal "find the laser dot" algorithm — a
focused laser point typically overwhelms (saturates/blows out) the
camera sensor at its exact center, appearing as a small, extremely
bright, often near-white spot regardless of the laser's actual color,
with a colored halo around it. This module looks for exactly that:
a small, near-maximum-brightness blob, isolated from other similarly
bright regions.

This WILL occasionally get it wrong — a specular highlight off a shiny
surface, a reflection, or a bright background light can look the same
way to this heuristic. There is no calibration data in this codebase
tying the robot's known laser-pointing geometry to camera pixel space
(that would need a separate hand-eye calibration step this project
doesn't have), so this is the practical alternative: look at the pixels
themselves rather than trying to compute where the dot "should" be.

Every caller of this module MUST treat its result as a starting guess
to show the user for confirmation/adjustment, never as ground truth to
upload unreviewed — see scripts/object_labeling_studio.py, which shows
the detected point overlaid on the photo and lets the user drag it
before anything is sent to Roboflow.
"""

from __future__ import annotations

from typing import Optional

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def _require_cv2():
    if not _CV2_AVAILABLE:
        raise RuntimeError("opencv-python (cv2) is required for laser dot detection.")


def detect_laser_dot(image_path: str, box_size_fraction: float = 0.06) -> Optional[dict]:
    """
    Looks for a small, very bright, isolated blob in the image — the
    telltale signature of a focused laser dot hitting a surface (see
    module docstring for why this is a heuristic, not a certainty).

    box_size_fraction: how big (as a fraction of image width) the
        returned bounding box around the detected point should be —
        the dot itself is usually only a few pixels, but a training
        annotation needs a box with real area, not a single point.

    Returns None if no confident candidate blob was found at all (the
    caller should fall back to asking the user to click manually), or:

        {"cx_norm": float, "cy_norm": float,      # box CENTER, 0-1
         "w_norm": float, "h_norm": float,        # box size, 0-1
         "confidence": "high" | "low"}            # see below

    "confidence" is "high" only when exactly one clearly-brightest
    small blob was found with no close competitor; "low" when a
    candidate exists but there was ambiguity (e.g. more than one
    similarly bright small region) — callers should visually flag
    "low" results more insistently for the user to double-check.
    """
    _require_cv2()
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Near-maximum brightness only — a focused laser dot blows out the
    # sensor; this threshold deliberately misses anything merely
    # "bright," only catching pixels at or near full saturation.
    _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = h * w
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        # A laser dot is small — reject anything covering too large a
        # fraction of the frame (a blown-out window, an overexposed
        # light fixture, etc. would otherwise pass the brightness test
        # too, but span a much bigger area than a focused dot does).
        if area <= 0 or area > image_area * 0.01:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        # Reject anything too elongated to plausibly be a round dot
        # (a bright thin reflection/glare streak, for instance).
        aspect = max(cw, ch) / max(1, min(cw, ch))
        if aspect > 2.5:
            continue
        cx, cy = x + cw / 2, y + ch / 2
        candidates.append((area, cx, cy))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    best_area, best_cx, best_cy = candidates[0]

    # Confidence: "high" only if there's no other candidate within 2x
    # the top one's area — two similarly-bright small blobs means this
    # can't tell which one (if either) is the actual laser dot.
    confidence = "high"
    if len(candidates) > 1 and candidates[1][0] > best_area * 0.5:
        confidence = "low"

    box_w_px = box_size_fraction * w
    box_h_px = box_size_fraction * h
    return {
        "cx_norm": best_cx / w,
        "cy_norm": best_cy / h,
        "w_norm": box_w_px / w,
        "h_norm": box_h_px / h,
        "confidence": confidence,
    }
