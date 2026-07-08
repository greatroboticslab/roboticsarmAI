"""
[WIRED] USB camera capture via OpenCV.

Works for any standard UVC-class USB webcam (the vast majority of USB
cameras) using nothing but a device index - no vendor SDK needed. If your
camera turns out to need a specialized SDK (e.g. an industrial/machine-
vision camera rather than a plain webcam), swap the cv2.VideoCapture
calls below for that SDK's frame-grab call; everything calling
capture_station_frame()/capture_wrist_frame() elsewhere stays the same.

SETUP
-----
    pip install opencv-python

Then find your camera indices - plug in one camera at a time and run:

    python -m vision.camera.capture

This opens each index 0-4 briefly and reports which ones produce a frame,
so you can confirm STATION_CAMERA_INDEX / WRIST_CAMERA_INDEX in
vision/config.py before relying on them in the real pipeline.
"""

from __future__ import annotations
import os
import uuid
import threading

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from vision.config import (
    STATION_CAMERA_INDEX,
    WRIST_CAMERA_INDEX,
    CAMERA_FRAME_WIDTH,
    CAMERA_FRAME_HEIGHT,
)

IMAGES_ROOT = "images/objects"

# Lazily-opened, cached VideoCapture handles - opened once, reused across
# calls rather than reopening the device every capture (slow + some UVC
# cameras don't like being reopened rapidly).
_capture_handles = {}

# CONCURRENCY NOTE: main.py has two independent triggers that both call
# into this module from their own background thread - the full
# pickup-and-photograph pipeline, and the standalone "Capture Photo"
# button. cv2.VideoCapture is not safe to .read() from two threads at
# once on the same handle (can corrupt frames or crash the backend). A
# per-camera-index lock below serializes access to a given camera while
# still letting station and wrist cameras (different indices) be used
# independently.
_capture_locks = {}
_locks_guard = threading.Lock()


def _get_lock(index: int) -> threading.Lock:
    with _locks_guard:
        if index not in _capture_locks:
            _capture_locks[index] = threading.Lock()
        return _capture_locks[index]


def _require_cv2():
    if not _CV2_AVAILABLE:
        raise ImportError(
            "opencv-python is not installed. Run: pip install opencv-python"
        )


def _get_handle(index: int):
    _require_cv2()
    if index not in _capture_handles:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {index}. Run "
                f"`python -m vision.camera.capture` to list working indices, "
                f"and confirm it isn't already in use by another program."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
        _capture_handles[index] = cap
    return _capture_handles[index]


def _capture_from_index(index: int):
    lock = _get_lock(index)
    with lock:
        cap = _get_handle(index)
        ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(
            f"Camera at index {index} did not return a frame. Check the "
            f"USB connection and that no other program has it open."
        )
    return frame


def capture_station_frame():
    """[WIRED] Grab a single frame from the fixed station camera."""
    return _capture_from_index(STATION_CAMERA_INDEX)


def capture_wrist_frame():
    """[WIRED] Grab a single frame from the wrist-mounted camera."""
    return _capture_from_index(WRIST_CAMERA_INDEX)


def save_image(frame, sample_id: str, source: str, view_index: int = 0) -> str:
    """
    [WIRED] Persist a captured frame to disk and return its path.
    Mirrors 4DAI's Server/main.py image storage layout
    (images/<category>/<sample_id>/<id>.jpg) so the two systems'
    on-disk data stays consistent.
    """
    _require_cv2()
    sample_dir = ensure_sample_dir(sample_id)
    image_path = os.path.join(sample_dir, f"{source}_{view_index}.jpg")
    ok = cv2.imwrite(image_path, frame)
    if not ok:
        raise IOError(f"Failed to write image to {image_path}")
    return image_path


def release_all():
    """Release all opened camera handles. Call on app shutdown."""
    for cap in _capture_handles.values():
        cap.release()
    _capture_handles.clear()


def new_sample_id() -> str:
    """[WIRED] Helper - no hardware dependency."""
    return str(uuid.uuid4())


def ensure_sample_dir(sample_id: str) -> str:
    """[WIRED] Helper - creates and returns the folder for a sample's images."""
    path = os.path.join(IMAGES_ROOT, sample_id)
    os.makedirs(path, exist_ok=True)
    return path


def list_camera_indices(max_index: int = 5):
    """
    Utility: probe indices 0..max_index-1 and report which ones produce a
    frame. Run directly: `python -m vision.camera.capture`
    """
    _require_cv2()
    working = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                working.append(i)
        cap.release()
    return working


if __name__ == "__main__":
    print("Probing camera indices 0-4 ...")
    found = list_camera_indices()
    if found:
        print(f"Working camera indices: {found}")
        print("Set STATION_CAMERA_INDEX / WRIST_CAMERA_INDEX in vision/config.py accordingly.")
    else:
        print("No working cameras found. Check USB connections and try again.")
