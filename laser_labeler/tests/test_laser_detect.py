import numpy as np

from core.tools.laser_detect import detect_laser
from tests.synth import finish, scene, spot


def test_finds_two_spots_brightest_first():
    img = scene()
    spot(img, 80, 100, sigma=3, amp=0.9)
    spot(img, 220, 150, sigma=3, amp=0.45)
    det = detect_laser(finish(img))
    assert len(det.candidates) == 2
    first, second = det.candidates
    assert first.peak > second.peak
    assert abs(first.center[0] - 80) < 2 and abs(first.center[1] - 100) < 2
    assert abs(second.center[0] - 220) < 2
    assert first.id == "c1" and second.id == "c2"


def test_no_laser_gives_nothing():
    det = detect_laser(finish(scene()))
    assert det.candidates == [] and det.lines == []


def test_saturated_flat_top_is_one_candidate():
    img = scene()
    spot(img, 100, 100, sigma=5, amp=3.0, color=(1, 1, 1))
    det = detect_laser(finish(img))
    assert len(det.candidates) == 1
    assert det.candidates[0].saturated_fraction > 0.5


def test_streak_is_a_line_not_a_spot():
    img = scene(w=400)
    rgb = finish(img)
    rgb[100:103, 40:360] = (230, 60, 60)
    det = detect_laser(rgb)
    assert det.candidates == []
    assert det.lines and det.lines[0].length > 250


def test_spot_at_end_of_beam():
    img = scene(w=400)
    spot(img, 340, 101, sigma=3, amp=0.9)
    rgb = finish(img)
    rgb[100:103, 40:340] = (150, 40, 40)
    det = detect_laser(rgb)
    assert any(abs(c.center[0] - 340) < 4 for c in det.candidates)
    assert det.lines


def test_huge_bright_region_is_not_a_spot():
    rgb = finish(scene())
    rgb[20:200, 20:300] = 200
    det = detect_laser(rgb)
    assert det.candidates == []


def test_works_on_grayscale():
    img = scene(gray=True)
    spot(img, 150, 120, sigma=3, amp=0.7, color=(1, 1, 1))
    det = detect_laser(finish(img))
    assert len(det.candidates) == 1
