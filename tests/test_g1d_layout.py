"""Run without Isaac Sim: python3 -m unittest discover -s tests -p test_g1d_layout.py."""

import copy
import importlib.util
import math
from pathlib import Path
import random
import unittest


spec = importlib.util.spec_from_file_location(
    "g1d_layout",
    Path(__file__).parents[1] / "tasks/g1d_tasks/sort_shapes/layout.py",
)
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.cfg = layout.load_settings()

    def test_legacy_controller_is_default(self):
        self.assertFalse(self.cfg["robot"]["controller_alignment"])

    def test_seeded_layouts_fit_without_object_or_bin_overlap(self):
        self.cfg["randomization"]["enabled"] = True
        objects = layout.objects(self.cfg)
        clearance = self.cfg["randomization"]["clearance"]
        tx, ty = self.cfg["table"]["center_xy"]
        sx, sy = self.cfg["table"]["size_xy"]
        for seed in range(50):
            poses = layout.sample_layout(self.cfg, random.Random(seed))
            self.assertEqual(poses, layout.sample_layout(self.cfg, random.Random(seed)))
            self.assertEqual(len(poses), len(objects))
            for i, obj in enumerate(objects):
                x, y, z, yaw = poses[obj["name"]]
                self.assertGreaterEqual(x - obj["radius"], tx - sx / 2 + clearance)
                self.assertLessEqual(x + obj["radius"], tx + sx / 2 - clearance)
                self.assertGreaterEqual(y - obj["radius"], ty - sy / 2 + clearance)
                self.assertLessEqual(y + obj["radius"], ty + sy / 2 - clearance)
                self.assertGreater(z - obj["size"][2] / 2, self.cfg["table"]["height"])
                for other in objects[i + 1:]:
                    ox, oy, *_ = poses[other["name"]]
                    self.assertGreaterEqual(math.hypot(x - ox, y - oy), obj["radius"] + other["radius"] + clearance)
                for box in self.cfg["bins"]:
                    x0, x1, y0, y1 = layout.bin_bounds(box)
                    self.assertGreaterEqual(math.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1)),
                                            obj["radius"] + clearance)

    def test_fixed_mode_and_variable_counts(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["randomization"]["enabled"] = False
        # Keep this count test independent of the user's default task layout.
        cfg["randomization"]["x_range"] = [0.48, 0.82]
        cfg["randomization"]["y_range"] = [-0.30, 0.30]
        cfg["cubes"]["count"] = 0
        cfg["cylinders"]["count"] = 8
        a = layout.sample_layout(cfg, random.Random(1))
        self.assertEqual(a, layout.sample_layout(cfg, random.Random(987)))
        self.assertEqual(len(a), 8)
        cfg["cylinders"]["count"] = 0
        self.assertEqual(layout.sample_layout(cfg, random.Random(0)), {})

    def test_different_seeds_change_random_layouts(self):
        self.cfg["randomization"]["enabled"] = True
        self.assertNotEqual(layout.sample_layout(self.cfg, random.Random(1)),
                            layout.sample_layout(self.cfg, random.Random(2)))

    def test_impossible_dimensions_are_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["cubes"]["size"] = [2, 2, 2]
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            layout.sample_layout(cfg, random.Random(1))


if __name__ == "__main__":
    unittest.main()
