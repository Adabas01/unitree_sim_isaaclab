"""CPU-only checks for evaluation configuration and geometric predicates."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch


ROOT = Path(__file__).parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_module(
    "g1d_evaluation_config",
    ROOT / "tasks/g1d_tasks/sort_shapes/evaluation/config.py",
)
predicates = load_module(
    "g1d_evaluation_predicates",
    ROOT / "tasks/g1d_tasks/sort_shapes/evaluation/predicates.py",
)
state_machine = load_module(
    "g1d_evaluation_state_machine",
    ROOT / "tasks/g1d_tasks/sort_shapes/evaluation/state_machine.py",
)


class EvaluationConfigTests(unittest.TestCase):
    def test_default_protocol_is_normalized_and_resolves_inside_repository(self):
        protocol = config.load_evaluation_settings()
        self.assertAlmostEqual(sum(protocol["stages"][name]["weight"] for name in config.STAGE_NAMES), 1.0)
        self.assertEqual(config.evaluation_output_path(protocol), ROOT / "logs/g1d_evaluation/episodes.jsonl")

    def test_invalid_hysteresis_is_rejected(self):
        protocol = config.load_evaluation_settings()
        protocol["stages"]["reach"]["exit_distance_m"] = 0.01
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "exit_distance_m"):
                config.load_evaluation_settings(path)


class EvaluationPredicateTests(unittest.TestCase):
    def setUp(self):
        self.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.pos = torch.tensor([[0.0, 0.0, 0.0]])

    def test_cube_surface_distance(self):
        cube = {"kind": "cube", "size": [0.1, 0.2, 0.3]}
        points = torch.tensor([[[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]]])
        distance = predicates.point_surface_distance(points, self.pos, self.quat, cube)
        torch.testing.assert_close(distance, torch.tensor([[0.0, 0.05]]))

    def test_bin_predicate_uses_complete_object_extent(self):
        cube = {"kind": "cube", "size": [0.1, 0.1, 0.1]}
        box = {"center_xy": [0.0, 0.0], "inner_size": [0.3, 0.3, 0.2], "wall_thickness": 0.01}
        inside = predicates.fully_inside_bin(
            torch.tensor([[0.0, 0.0, 0.76]]), self.quat, cube, box, table_height=0.7, margin=0.0
        )
        outside = predicates.fully_inside_bin(
            torch.tensor([[0.12, 0.0, 0.76]]), self.quat, cube, box, table_height=0.7, margin=0.0
        )
        self.assertTrue(bool(inside.item()))
        self.assertFalse(bool(outside.item()))


class EvaluationStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.achieved = torch.zeros((1, 1, 3), dtype=torch.bool)
        self.dwell = torch.zeros((1, 1, 3))
        self.required = torch.tensor([0.10, 0.10, 0.10])

    def advance(self, conditions):
        self.achieved, self.dwell, newly = state_machine.advance_ordered_stages(
            self.achieved,
            self.dwell,
            torch.tensor([[conditions]], dtype=torch.bool),
            self.required,
            0.05,
        )
        return newly

    def test_stages_require_order_and_continuous_dwell(self):
        self.advance([False, True, True])
        self.assertEqual(self.dwell.tolist(), [[[0.0, 0.0, 0.0]]])

        self.advance([True, True, True])
        self.advance([False, True, True])
        self.assertEqual(self.achieved.tolist(), [[[False, False, False]]])
        self.assertEqual(self.dwell[0, 0, 0].item(), 0.0)

        self.advance([True, True, True])
        newly = self.advance([True, True, True])
        self.assertEqual(self.achieved.tolist(), [[[True, False, False]]])
        self.assertTrue(bool(newly[0, 0, 0]))

        self.advance([False, True, True])
        self.assertEqual(self.achieved.tolist(), [[[True, True, False]]])
        self.advance([False, False, True])
        self.assertEqual(self.achieved.tolist(), [[[True, True, True]]])

    def test_zero_dwell_does_not_credit_a_false_condition(self):
        required = torch.zeros(3)
        achieved, _, newly = state_machine.advance_ordered_stages(
            self.achieved,
            self.dwell,
            torch.zeros_like(self.achieved),
            required,
            0.05,
        )
        self.assertFalse(bool(achieved.any()))
        self.assertFalse(bool(newly.any()))

    def test_task_success_dwell_uses_current_task_state(self):
        dwell = torch.tensor([0.20, 0.20])
        dwell, complete = state_machine.advance_condition_dwell(
            torch.tensor([True, False]), dwell, required_dwell_s=0.25, dt=0.05
        )
        torch.testing.assert_close(dwell, torch.tensor([0.25, 0.0]))
        self.assertEqual(complete.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
