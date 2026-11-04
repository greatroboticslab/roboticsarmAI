import math

import numpy as np

from core.camera import image_stats
from core.tools.laser_detect import BeamLine
from core.tools.measure_laser import angle_into, measure_laser
from tests.synth import finish, scene, spot

BOX = (90, 90, 20, 20)


def measure(img, box=BOX, gray=False, **kw):
    rgb = finish(img)
    return measure_laser(rgb, box, image_stats(rgb), gray, **kw)


def make(**kw):
    img = scene()
    spot(img, 100, 100, **kw)
    return img


def test_known_spot_numbers():
    m = measure(make(sigma=3.0, amp=0.5, color=(1, 1, 1)))
    expected_area = math.pi * (1.177 * 3.0) ** 2  # half-maximum disc of a sigma=3 gaussian
    assert 0.7 * expected_area < m["core_area_px"] < 1.3 * expected_area
    assert abs(m["peak_intensity"] - (0.16 + 0.5)) < 0.03
    assert 1.0 <= m["aspect_ratio"] < 1.25
    assert 0.3 < m["scatter_ratio"] < 0.6  # a clean gaussian already reads ~0.45; compare relatively
    assert m["clipped_fraction"] == 0.0
    assert 0.5 < m["confidence"] <= 1.0
    assert abs(m["center_x"] - 100) < 1 and abs(m["center_y"] - 100) < 1


def test_background_normalisation():
    m = measure(make(amp=0.5, color=(1, 1, 1)))
    assert abs(m["background_level"] - 40 / 255) < 0.02  # max-channel intensity reads a little high
    expected = (m["peak_intensity"] - m["background_level"]) / (1 - m["background_level"])
    assert abs(m["peak_intensity_norm"] - expected) < 1e-6
    assert m["mean_intensity"] < m["peak_intensity"]


def test_aspect_ratio_of_elongated_spot():
    m = measure(make(sigma=6.0, sigma_y=3.0, amp=0.6, color=(1, 1, 1)), box=(85, 90, 30, 20))
    assert 1.7 < m["aspect_ratio"] < 2.3


def test_halo_raises_scatter_and_radius():
    clean = measure(make(sigma=3, amp=0.6, color=(1, 1, 1)))
    hazy = measure(make(sigma=3, amp=0.6, halo_amp=0.12, halo_sigma=14, color=(1, 1, 1)))
    assert hazy["scatter_ratio"] > clean["scatter_ratio"] + 0.1
    assert hazy["halo_radius_px"] > clean["halo_radius_px"] + 5


def test_saturated_spot_reports_clipping():
    m = measure(make(sigma=4, amp=3.0, color=(1, 1, 1)))
    assert m["clipped_fraction"] > 0.3
    assert m["confidence"] < measure(make(sigma=4, amp=0.5, color=(1, 1, 1)))["confidence"]


def test_reflectivity_relative_to_first_box():
    first = measure(make(sigma=3, amp=0.8, color=(1, 1, 1)), kind="first_incidence")
    assert first["reflectivity"] == 1.0
    dim = measure(make(sigma=3, amp=0.3, color=(1, 1, 1)), kind="reflection", first_total=first["total_light"])
    assert 0.1 < dim["reflectivity"] < 0.7
    assert "reflectivity" not in measure(make(), kind="reflection")


def test_empty_box_gives_zero_confidence():
    m = measure(scene(), box=(10, 10, 20, 20))
    assert m["confidence"] == 0.0 and m["core_area_px"] == 0.0


def test_deterministic():
    img = make(sigma=3, amp=0.5)
    assert measure(img) == measure(img.copy())


def test_color_and_gray_level():
    m = measure(make(amp=0.6, color=(1.0, 0.2, 0.2)))
    assert m["color_r"] > m["color_g"] + 40 and "gray_level" not in m
    g = scene(gray=True)
    spot(g, 100, 100, amp=0.6, color=(1, 1, 1))
    mg = measure(g, gray=True)
    assert "gray_level" in mg and "color_r" not in mg and mg["gray_level"] > 100


def test_angle_from_previous_point_and_line():
    assert abs(angle_into((10, 10), prev_center=(0, 10)) - 0) < 1e-6        # moving right
    assert abs(angle_into((10, 0), prev_center=(10, 10)) - 90) < 1e-6       # moving up
    line = BeamLine(0, 100, 90, 100)
    assert abs(angle_into((100, 100), lines=[line]) - 0) < 1e-6
    assert angle_into((100, 100)) is None
    assert angle_into((300, 300), lines=[line]) is None                       # line too far away
    m = measure(make(amp=0.6), prev_center=(60.0, 100.0))
    assert abs(m["angle_deg"]) < 2 or abs(m["angle_deg"] - 360) < 2
