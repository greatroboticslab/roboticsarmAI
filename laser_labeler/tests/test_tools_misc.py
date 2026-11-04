import numpy as np
import pytest
from PIL import Image

from core.camera import (camera_id_for, image_stats, intensity_map, is_grayscale_array, load_image)
from core.tools.camera_check import camera_check, format_report
from core.tools.color import sample_color
from core.tools.crop import crop_zoom, downscale_to_max, encode_png
from core.tools.geometry import clamp_box, iou
from core.tools.overlay import draw_boxes
from core.tools.tighten_box import tighten_box
from tests.synth import finish, rect, scene, spot


# ---- tighten_box
def object_scene():
    img = scene(bg=40)
    rect(img, 100, 80, 80, 60, (150, 30, 30))  # red object
    spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
    return finish(img)


def test_tighten_finds_object_edges():
    res = tighten_box(object_scene(), (120, 90, 40, 40))
    assert iou(res.box, (100, 80, 80, 60)) > 0.9
    assert res.confidence >= 0.8 and not res.hit_window_edge


def test_tighten_shrinks_oversized_box():
    res = tighten_box(object_scene(), (60, 40, 160, 140), margin=0.1)
    assert res.box[2] <= 110 or res.hit_window_edge or res.confidence < 0.8


def test_tighten_falls_back_on_tiny_region():
    rgb = finish(scene())
    rng = np.random.default_rng(1)
    rgb[...] = rng.integers(0, 200, rgb.shape, dtype=np.uint8)  # pure texture
    box = (100, 100, 40, 40)
    res = tighten_box(rgb, box)
    assert res.box == box and not res.changed and res.confidence < 0.5


# ---- color
def test_sample_color_names():
    rgb = object_scene()
    s = sample_color(rgb, (105, 85, 30, 20))
    assert s.name == "red" and s.confidence > 0.15
    assert s.rgb[0] > s.rgb[1] + 80


def test_sample_color_ignores_laser_glare():
    rgb = object_scene()
    s = sample_color(rgb, (120, 90, 40, 40))  # contains the white laser spot
    assert s.name == "red"


def test_sample_color_grayscale_is_placeholder():
    g = finish(scene(gray=True))
    s = sample_color(g, (10, 10, 20, 20), is_grayscale=True, placeholder="na")
    assert s.name == "na" and 20 < s.gray_level < 70


def test_sample_color_respects_allowed_names():
    s = sample_color(object_scene(), (105, 85, 30, 20), allowed_colors=["blue", "green"])
    assert s.name in ("blue", "green")
    assert sample_color(object_scene(), (105, 85, 30, 20), allowed_colors=["clear"]).name is None


# ---- camera
def test_grayscale_detection():
    assert is_grayscale_array(finish(scene(gray=True)))
    assert not is_grayscale_array(object_scene())


def test_load_image_roundtrip_and_modes(tmp_path):
    rgb = object_scene()
    Image.fromarray(rgb).save(tmp_path / "c.png")
    Image.fromarray(rgb[..., 0]).save(tmp_path / "g.png")  # mode L
    color, gray = load_image(tmp_path / "c.png"), load_image(tmp_path / "g.png")
    assert not color.is_grayscale and gray.is_grayscale
    assert gray.rgb.shape[2] == 3 and color.exposure is None
    assert (color.width, color.height) == (320, 240)


def test_camera_ids():
    assert camera_id_for("a.jpg", ["split:train", "camera:logi"]) == "logi"
    assert camera_id_for("a.jpg", [], is_grayscale=True) == "grayscale"
    assert camera_id_for("/x/cam2/a.jpg", [], source="folder") == "cam2"
    assert camera_id_for("a.jpg", []) == "unknown"


def test_background_level_ignores_laser():
    img = scene(bg=60)
    base = image_stats(finish(img)).background_level
    spot(img, 100, 100, amp=0.9)
    assert abs(image_stats(finish(img)).background_level - base) < 0.01


def test_camera_check_flags_drift(tmp_path):
    for i, bg in enumerate([30, 60, 90, 120]):
        Image.fromarray(finish(scene(bg=bg))).save(tmp_path / f"drift_{i}.png")
    steady = tmp_path / "steady"
    steady.mkdir()
    for i in range(4):
        Image.fromarray(finish(scene(bg=80, seed=i))).save(steady / f"s_{i}.png")
    drift_paths = sorted(tmp_path.glob("drift_*.png"))
    tags = {str(p): ["camera:webcam"] for p in drift_paths}
    tags.update({str(p): ["camera:stable"] for p in steady.glob("*.png")})
    reports = {r.camera_id: r for r in camera_check([*drift_paths, *steady.glob("*.png")], tags, sample=None)}
    assert reports["webcam"].drifting and not reports["stable"].drifting
    assert "DRIFTING" in format_report(list(reports.values()))


# ---- crop, overlay, geometry
def test_crop_zoom_and_scale():
    rgb = object_scene()
    crop = crop_zoom(rgb, (130, 100, 20, 20))
    assert max(crop.image.shape[:2]) == 384 and crop.scale > 1
    assert crop.region[0] <= 130 and crop.region[0] + crop.region[2] >= 150


def test_downscale_and_encode():
    big = np.zeros((1000, 2000, 3), np.uint8)
    small, f = downscale_to_max(big, 500)
    assert small.shape[:2] == (250, 500) and f == 0.25
    assert downscale_to_max(big, 0)[1] == 1.0
    assert encode_png(small)[:4] == b"\x89PNG"


def test_overlay_does_not_mutate_input():
    rgb = object_scene()
    before = rgb.copy()
    out = draw_boxes(rgb, [("a", (10, 10, 30, 30))])
    assert (rgb == before).all() and (out != rgb).any()


def test_clamp_box_and_iou():
    assert clamp_box((-5, -5, 20, 20), 100, 100) == (0, 0, 15, 15)
    assert clamp_box((95, 95, 20, 20), 100, 100) == (95, 95, 5, 5)
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
