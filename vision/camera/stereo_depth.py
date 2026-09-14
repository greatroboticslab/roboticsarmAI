"""
Stereo depth-map generation for "extract lenses" cameras (e.g. the
See3CAM_Stereo) — turns a Left/Right pair from vision.camera.capture's
lens extraction into a depth map, with an optional calibration workflow
for accuracy.

IMPORTANT HONESTY NOTE: the See3CAM_Stereo (and UVC stereo cameras
generally) have NO onboard depth hardware — unlike e.g. an Intel
RealSense, which computes depth on its own ASIC and exposes it as its
own stream, this camera is a pure pair of image sensors. There is no
generic OpenCV/UVC property to ask an arbitrary camera "do you have a
native depth stream" — that's vendor-SDK territory, not something plain
OpenCV can discover. So depth here is ALWAYS computed in software using
OpenCV's StereoSGBM — the standard, well-established open-source stereo
matching algorithm — after a Left/Right pair has been produced by lens
extraction. has_native_depth_support() below is a clear hook for future
vendor-SDK-specific depth support should a camera that actually has it
ever get added, but it always returns False today because no such
integration exists — there's no camera plugged into this that this code
could have tested against for real onboard depth.

CALIBRATION: raw, uncalibrated stereo images have lens distortion and
aren't perfectly aligned along epipolar lines, which makes disparity/
depth noisy and geometrically inaccurate. A calibration workflow here
(show a checkerboard to both lenses from several angles, run OpenCV's
standard cv2.calibrateCamera/stereoCalibrate/stereoRectify pipeline)
computes rectification maps that correct for both, saved per camera
name so it only needs to be done once (until the camera's physical
mounting or lens changes). Depth computation works without calibration
too (falls back to the raw, unrectified images) — it's just less
accurate — calibration is optional, not required.
"""

import json
import os

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def _require_cv2():
    if not _CV2_AVAILABLE:
        raise ImportError(
            "opencv-python (cv2) is required for stereo depth features but is not "
            "installed. Run: pip install opencv-python"
        )


CALIBRATION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "camera_calibration",
)

DEFAULT_CHECKERBOARD_SIZE = (9, 6)  # internal corners, not squares — standard OpenCV convention
DEFAULT_SQUARE_SIZE_MM = 25.0


def has_native_depth_support(camera_name: str) -> bool:
    """Always False today — see module docstring. This exists as a
    deliberate, clearly-named hook: if a camera with genuine onboard
    depth hardware and a real SDK integration is ever added, THIS is
    where that check would go, and compute_depth_map() below would
    prefer it over the software fallback. Right now nothing sets it to
    True for any camera."""
    return False


def _safe_name(camera_name: str) -> str:
    return "".join(c for c in camera_name if c.isalnum() or c in "-_") or "camera"


def _calib_path(camera_name: str) -> str:
    return os.path.join(CALIBRATION_DIR, f"{_safe_name(camera_name)}_stereo_calibration.json")


def has_calibration(camera_name: str) -> bool:
    return os.path.exists(_calib_path(camera_name))


def load_calibration(camera_name: str):
    """Returns the saved calibration dict for `camera_name` (rectification
    maps + image size + reprojection error), or None if none is saved."""
    path = _calib_path(camera_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        _require_cv2()
        for key in ("map1x", "map1y", "map2x", "map2y"):
            data[key] = np.array(data[key], dtype=np.float32)
        # Q is only present in calibrations saved after metric depth
        # support was added — older calibration files won't have it,
        # so this stays optional rather than raising, and
        # disparity_to_depth_mm() below treats its absence as "no
        # metric depth available for this camera" (falls back to
        # disparity-only) rather than an error.
        if "Q" in data:
            data["Q"] = np.array(data["Q"], dtype=np.float64)
        return data
    except Exception as e:
        print(f"[STEREO DEPTH] Could not load calibration for '{camera_name}': {e}")
        return None


def clear_calibration(camera_name: str) -> None:
    path = _calib_path(camera_name)
    if os.path.exists(path):
        os.remove(path)


# In-memory accumulation state for the calibration wizard (Add Calibration
# Image / Finish Calibration buttons on the Camera tab) — deliberately not
# persisted to disk; a half-finished calibration session isn't meaningful
# to resume across an app restart, and starting over is cheap.
_calib_state: dict = {}


def start_calibration_session(camera_name: str) -> None:
    """Clears any in-progress calibration image collection for this
    camera so 'Add Calibration Image' starts a fresh batch."""
    _calib_state[camera_name] = {"objpoints": [], "imgpoints_l": [], "imgpoints_r": [], "image_size": None}


def calibration_progress(camera_name: str) -> int:
    state = _calib_state.get(camera_name)
    return len(state["imgpoints_l"]) if state else 0


def add_calibration_image(camera_name: str, left_frame, right_frame,
                           checkerboard_size=DEFAULT_CHECKERBOARD_SIZE,
                           square_size_mm: float = DEFAULT_SQUARE_SIZE_MM) -> dict:
    """
    Attempts to find checkerboard corners in both frames of one Left/
    Right pair (see vision.camera.capture's lens extraction — this is
    meant to be called with two of the extracted lens images) and, if
    found in BOTH, accumulates them toward the running calibration.

    The camera itself stays completely still/mounted for the whole
    process — this is standard practice for stereo calibration and is
    what this workflow assumes throughout. Print a checkerboard pattern
    (checkerboard_size default is 9x6 INTERNAL corners — a 10x7-square
    board) and move ONLY THE BOARD to several different positions,
    angles, and distances in front of the stationary camera, calling
    this once per position; run_calibration() once enough images (10+)
    are collected.

    Returns {"found": bool, "count": int, "message": str}.
    """
    _require_cv2()
    state = _calib_state.setdefault(
        camera_name, {"objpoints": [], "imgpoints_l": [], "imgpoints_r": [], "image_size": None})

    gray_l = _to_gray_u8(left_frame)
    gray_r = _to_gray_u8(right_frame)

    found_l, corners_l = cv2.findChessboardCorners(gray_l, checkerboard_size)
    found_r, corners_r = cv2.findChessboardCorners(gray_r, checkerboard_size)

    if not (found_l and found_r):
        missing = "left" if not found_l else "right"
        return {
            "found": False, "count": len(state["imgpoints_l"]),
            "message": f"Checkerboard not found in the {missing} view — reposition it "
                       f"(fully visible, flat, well-lit, not too close/far) and try again.",
        }

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
    corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    state["objpoints"].append(objp)
    state["imgpoints_l"].append(corners_l)
    state["imgpoints_r"].append(corners_r)
    state["image_size"] = (gray_l.shape[1], gray_l.shape[0])

    return {
        "found": True, "count": len(state["imgpoints_l"]),
        "message": f"Checkerboard found in both views — {len(state['imgpoints_l'])} "
                   f"calibration image(s) collected so far.",
    }


def run_calibration(camera_name: str, min_images: int = 10) -> dict:
    """
    Runs OpenCV's standard stereo calibration pipeline (calibrateCamera
    per lens, then stereoCalibrate + stereoRectify for the pair) over
    every image collected via add_calibration_image() since the last
    start_calibration_session(), and saves the resulting rectification
    maps for compute_depth_map() to use. Requires at least `min_images`
    (default 10) checkerboard detections — fewer than that produces an
    unreliable calibration, so this refuses rather than saving a bad
    one silently.　Returns {"ok": bool, "message": str}.
    """
    _require_cv2()
    state = _calib_state.get(camera_name)
    got = len(state["imgpoints_l"]) if state else 0
    if not state or got < min_images:
        return {
            "ok": False,
            "message": f"Need at least {min_images} calibration images with the "
                       f"checkerboard found in both views — only have {got}. Keep using "
                       f"'Add Calibration Image' from different angles/distances.",
        }

    image_size = state["image_size"]
    objpoints, imgpoints_l, imgpoints_r = state["objpoints"], state["imgpoints_l"], state["imgpoints_r"]

    try:
        _, K_l, D_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, image_size, None, None)
        _, K_r, D_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, image_size, None, None)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        reproj_error, K_l, D_l, K_r, D_r, R, T, _, _ = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r, K_l, D_l, K_r, D_r, image_size,
            criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_l, D_l, K_r, D_r, image_size, R, T, alpha=0)

        map1x, map1y = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, image_size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, image_size, cv2.CV_32FC1)
    except Exception as e:
        return {"ok": False, "message": f"Calibration failed: {e}"}

    data = {
        "image_size": list(image_size),
        "map1x": map1x.tolist(), "map1y": map1y.tolist(),
        "map2x": map2x.tolist(), "map2y": map2y.tolist(),
        # Q: the disparity-to-depth reprojection matrix stereoRectify()
        # already computes above — saved so disparity_to_depth_mm()
        # below can turn a disparity map into an actual METRIC depth
        # map (real millimeters, using the real calibrated baseline
        # between the two lenses) instead of only a relative "brighter=
        # closer" disparity visualization with no real-world scale.
        "Q": Q.tolist(),
        "reprojection_error": float(reproj_error),
    }
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    with open(_calib_path(camera_name), "w") as f:
        json.dump(data, f)

    _calib_state.pop(camera_name, None)
    quality = "good" if reproj_error < 1.0 else ("okay" if reproj_error < 2.0 else "poor - consider redoing")
    return {
        "ok": True,
        "message": f"Calibration complete for '{camera_name}' — reprojection error "
                   f"{reproj_error:.3f}px ({quality}; lower is better, under ~1.0px is "
                   f"generally good). Saved and will be used automatically for depth maps.",
    }


def _to_gray_u8(frame):
    """Normalizes any frame (color or mono, any bit depth) to a plain
    8-bit single-channel grayscale image for stereo matching/checkerboard
    detection."""
    _require_cv2()
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.shape[2] >= 3 else frame[:, :, 0]
    if frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return frame


def compute_disparity(camera_name: str, left_frame, right_frame):
    """
    Core stereo matching step, factored out so both the disparity
    VISUALIZATION and the METRIC depth conversion below build on the
    exact same underlying computation rather than two separate
    (and potentially inconsistent) StereoSGBM runs.

    Rectifies the pair first if a matching calibration is saved for
    `camera_name` (see load_calibration), same as before. Returns the
    RAW float32 disparity array (in pixels, NOT normalized to 0-255 —
    that normalization is a visualization detail, and normalizing here
    would throw away the actual scale reprojectImageTo3D() needs to
    convert to real millimeters), or None if the pair isn't usable.
    """
    _require_cv2()
    if left_frame is None or right_frame is None:
        return None

    gray_l = _to_gray_u8(left_frame)
    gray_r = _to_gray_u8(right_frame)

    calib = load_calibration(camera_name)
    if calib is not None and calib.get("image_size") == [gray_l.shape[1], gray_l.shape[0]]:
        gray_l = cv2.remap(gray_l, calib["map1x"], calib["map1y"], cv2.INTER_LINEAR)
        gray_r = cv2.remap(gray_r, calib["map2x"], calib["map2y"], cv2.INTER_LINEAR)

    # StereoSGBM with reasonable general-purpose defaults. numDisparities
    # must be a positive multiple of 16; blockSize odd, typically 3-11.
    block_size = 7
    num_disparities = 128
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0


def _visualize(array, colorize: bool = False):
    """
    Normalizes any single-channel float array (disparity in pixels, or
    depth in mm) to an 8-bit image for saving/display, then optionally
    false-colors it with a colormap (near/far mapped to a color
    gradient — e.g. blue-to-red — the same idea as how a RealSense
    viewer or most depth-camera tools show depth, rather than a flat
    grayscale gradient that's harder to read at a glance).

    IMPORTANT: this "color" is a VISUALIZATION AID ONLY — it is not,
    and cannot be, the scene's real color. A disparity/depth map has
    exactly one value per pixel (a distance), not three (R/G/B); there
    is no camera in this pipeline that captures real per-pixel color to
    show here even if one wanted to overlay it. Colorizing just makes
    that one value's variation across the image easier to read than a
    grayscale gradient does — it's the same "colorized" convention used
    by essentially every depth-camera/LIDAR visualization tool.

    Returns an 8-bit grayscale (colorize=False) or BGR (colorize=True)
    array, or None if `array` is None.
    """
    _require_cv2()
    if array is None:
        return None
    normalized = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)
    u8 = np.uint8(normalized)
    if colorize:
        return cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return u8


def disparity_to_depth_mm(camera_name: str, disparity):
    """
    Converts a raw disparity array (see compute_disparity) into an
    actual METRIC depth map — real distance in millimeters at every
    pixel — using the saved calibration's Q matrix (the disparity-to-
    depth reprojection matrix cv2.stereoRectify() computes from the
    real, measured baseline between the two lenses during calibration).

    This is what makes the difference between "brighter looks closer"
    (disparity alone — relative, uncalibrated, arbitrary units) and
    "this pixel is approximately 812mm away" (actual depth — requires
    knowing the real physical distance between the two lenses, which
    only calibration provides). Without a saved calibration that
    includes Q (see run_calibration — older calibrations predating this
    won't have it), there's no way to know the real-world scale, so
    this returns None rather than guessing.

    Returns a float32 array (mm, one value per pixel; pixels with no
    valid disparity are set to 0) or None if no usable calibration is
    saved for `camera_name`.
    """
    _require_cv2()
    if disparity is None:
        return None
    calib = load_calibration(camera_name)
    if calib is None or "Q" not in calib:
        return None
    points_3d = cv2.reprojectImageTo3D(disparity, calib["Q"])
    depth_mm = points_3d[:, :, 2].astype(np.float32)
    invalid = (disparity <= 0) | ~np.isfinite(depth_mm) | (depth_mm < 0)
    depth_mm[invalid] = 0
    return depth_mm


def compute_depth_map(camera_name: str, left_frame, right_frame, colorize: bool = False):
    """
    Computes a disparity VISUALIZATION from a Left/Right pair — kept
    for backward compatibility with existing callers (this is the
    original/simplest entry point: one image out, no calibration
    required). For the fuller picture (raw disparity AND, when
    calibrated, real metric depth in mm) see compute_depth_map_and_
    metric() below, which is what vision.camera.capture's
    _apply_extraction actually uses now.

    Prefers a native/hardware depth source if has_native_depth_support()
    is ever True for this camera (see that function's docstring — always
    False today, for any camera); otherwise (always, currently) uses
    OpenCV's StereoSGBM via compute_disparity() above.

    If a saved calibration exists for this camera AND matches the
    current image size, the Left/Right images are rectified (lens-
    undistorted + epipolar-aligned) first, giving a substantially more
    accurate and consistent result. Without calibration, this still
    computes a usable disparity map directly from the raw images, but
    it's a RELATIVE grayscale visualization (brighter = closer), not a
    calibrated metric distance — true metric depth needs the physical
    baseline distance between the two lenses (see disparity_to_depth_mm),
    which calibration provides and an uncalibrated map doesn't have.

    colorize: False (default) returns an 8-bit single-channel grayscale
        image (brighter = closer); True returns a false-colored BGR
        image instead — see _visualize()'s docstring for why this is a
        visualization aid, not real captured color.

    Returns the visualization array, or None if the pair doesn't look
    usable for stereo matching.
    """
    if has_native_depth_support(camera_name):
        # Hook for a future vendor-SDK depth integration — nothing
        # implements this path today (see module docstring), so
        # has_native_depth_support() always returns False and this
        # branch is unreachable in practice right now.
        raise NotImplementedError(
            "has_native_depth_support() returned True but no native depth backend "
            "is actually implemented yet — this is a placeholder for future work."
        )
    disparity = compute_disparity(camera_name, left_frame, right_frame)
    return _visualize(disparity, colorize=colorize)


def compute_depth_map_and_metric(camera_name: str, left_frame, right_frame, colorize: bool = False):
    """
    The fuller version of compute_depth_map() above: computes disparity
    ONCE (see compute_disparity), then produces BOTH outputs from it —
    "convert into a disparity, and then [if possible] a depth map", the
    two distinct artifacts a stereo pair can produce:

      1. A disparity visualization — always produced if the pair is
         usable at all; relative "brighter/more saturated = closer",
         no real-world units, no calibration required.
      2. A metric depth visualization — ONLY produced when a
         calibration with a saved Q matrix exists for `camera_name`
         (see disparity_to_depth_mm) — visualizes actual real-world
         distance in mm rather than raw disparity magnitude, which
         means it correctly accounts for this camera's real baseline/
         focal length instead of just "more shifted pixels = closer,
         who knows how close." None when uncalibrated — there's no way
         to fabricate real-world scale without it, so this doesn't
         pretend to.

    colorize applies to both outputs identically — see _visualize()'s
    docstring for why this "color" is a false-color visualization aid,
    not real captured scene color.

    Returns (disparity_vis, depth_vis_or_None) — either element can
    still be None if the pair wasn't usable for stereo matching at all
    (disparity_vis) or no calibration was available (depth_vis).
    """
    if has_native_depth_support(camera_name):
        # Hook for a future vendor-SDK depth integration — see
        # compute_depth_map()'s matching check above; always
        # unreachable today (has_native_depth_support() always
        # returns False for every camera right now).
        raise NotImplementedError(
            "has_native_depth_support() returned True but no native depth backend "
            "is actually implemented yet — this is a placeholder for future work."
        )
    disparity = compute_disparity(camera_name, left_frame, right_frame)
    if disparity is None:
        return None, None
    disparity_vis = _visualize(disparity, colorize=colorize)
    depth_mm = disparity_to_depth_mm(camera_name, disparity)
    depth_vis = _visualize(depth_mm, colorize=colorize) if depth_mm is not None else None
    return disparity_vis, depth_vis
