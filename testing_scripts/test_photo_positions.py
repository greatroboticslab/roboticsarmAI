"""Unit tests for vision/services/photo_positions.py (no GUI, no robot).

Run from the repo root:   python -m unittest testing_scripts.test_photo_positions -v
"""
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.services import photo_positions as pp


class PhotoPositionsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = pp._PATH
        pp._PATH = os.path.join(self._tmp.name, "photo_positions.json")

    def tearDown(self):
        pp._PATH = self._orig
        self._tmp.cleanup()

    def test_make_position_derives_xy_from_joints(self):
        p = pp.make_position("a", 0, 0, 100, 0)
        self.assertAlmostEqual(p["x"], 400.0)
        self.assertAlmostEqual(p["y"], 0.0)

    def test_make_position_keeps_supplied_xy(self):
        p = pp.make_position("a", 10, 20, 100, 0, x=1.5, y=2.5)
        self.assertEqual((p["x"], p["y"]), (1.5, 2.5))

    def test_rejects_blank_name_and_out_of_range(self):
        with self.assertRaises(ValueError):
            pp.make_position("  ", 0, 0, 100, 0)
        for bad in [(200, 0, 100, 0), (0, 200, 100, 0), (0, 0, 300, 0), (0, 0, 100, 400),
                    (float("nan"), 0, 100, 0)]:
            with self.assertRaises(ValueError, msg=str(bad)):
                pp.make_position("a", *bad)

    def test_fk_inverts_main_ikinematics_geometry(self):
        # IK for (250, 50): D = (x^2+y^2-L1^2-L2^2)/(2 L1 L2)
        x, y = 250.0, 50.0
        d = (x * x + y * y - 200 ** 2 - 200 ** 2) / (2 * 200 * 200)
        t2 = math.atan2(math.sqrt(1 - d * d), d)
        t1 = math.atan2(y, x) - math.atan2(200 * math.sin(t2), 200 + 200 * math.cos(t2))
        fx, fy = pp.fk_xy(math.degrees(t1), math.degrees(t2))
        self.assertAlmostEqual(fx, x, places=6)
        self.assertAlmostEqual(fy, y, places=6)

    def test_save_load_roundtrip_and_order(self):
        items = [pp.make_position(n, j1, 10, 120, 5) for n, j1 in (("b", 1), ("a", 2), ("c", 3))]
        pp.save(items)
        self.assertEqual([p["name"] for p in pp.load()], ["b", "a", "c"])

    def test_load_missing_or_corrupt_file(self):
        self.assertEqual(pp.load(), [])
        with open(pp._PATH, "w") as f:
            f.write("{not json")
        self.assertEqual(pp.load(), [])
        with open(pp._PATH, "w") as f:
            f.write('{"not": "a list"}')
        self.assertEqual(pp.load(), [])

    def test_load_drops_invalid_records_but_keeps_good_ones(self):
        with open(pp._PATH, "w") as f:
            f.write('[{"name":"ok","j1":0,"j2":0,"z":100,"j4":0},'
                    '{"name":"bad","j1":999,"j2":0,"z":100,"j4":0},{"name":"missing"}]')
        self.assertEqual([p["name"] for p in pp.load()], ["ok"])

    def test_unique_and_default_names(self):
        existing = [pp.make_position(n, 0, 0, 100, 0) for n in ("shelf", "Shelf_2", "pos_1")]
        self.assertEqual(pp.unique_name("shelf", existing), "shelf_3")
        self.assertEqual(pp.unique_name("new", existing), "new")
        self.assertEqual(pp.next_default_name(existing), "pos_4")

    def test_enabled_defaults_on_and_old_files_load_as_on(self):
        self.assertTrue(pp.make_position("a", 0, 0, 100, 0)["enabled"])
        with open(pp._PATH, "w") as f:   # file written before the toggle existed
            f.write('[{"name":"old","j1":0,"j2":0,"z":100,"j4":0}]')
        self.assertTrue(pp.load()[0]["enabled"])

    def test_enabled_flag_persists(self):
        items = [pp.make_position("a", 0, 0, 100, 0), pp.make_position("b", 1, 0, 100, 0, enabled=False)]
        pp.save(items)
        self.assertEqual([p["enabled"] for p in pp.load()], [True, False])

    def test_enabled_positions_keeps_order_and_skips_off(self):
        items = [pp.make_position(n, 0, 0, 100, 0, enabled=e)
                 for n, e in (("a", True), ("b", False), ("c", True))]
        self.assertEqual([p["name"] for p in pp.enabled_positions(items)], ["a", "c"])

    def test_toggle_all_on_goes_off_otherwise_goes_on(self):
        items = [pp.make_position(n, 0, 0, 100, 0) for n in "abc"]
        self.assertFalse(pp.toggle(items, [0, 1]))           # both on -> off
        self.assertEqual([p["enabled"] for p in items], [False, False, True])
        self.assertTrue(pp.toggle(items, [1, 2]))            # mixed -> all on
        self.assertEqual([p["enabled"] for p in items], [False, True, True])
        self.assertIsNone(pp.toggle(items, [9]))             # nothing valid
        self.assertEqual(pp.set_enabled(items, range(3), True), 1)
        self.assertEqual(pp.set_enabled(items, [-1, 7], False), 0)


if __name__ == "__main__":
    unittest.main()
