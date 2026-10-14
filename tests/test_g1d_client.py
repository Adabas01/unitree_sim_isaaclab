"""Check outgoing G1 commands from both workspace clients without opening DDS."""

import importlib.util
import logging
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC


class Published(Exception):
    """Stop the client's infinite publisher loop after inspecting its first packet."""


class ClientAlignmentTests(unittest.TestCase):
    def test_real_and_simulated_publishers_apply_the_same_limiter(self):
        workspace = Path(__file__).resolve().parents[2]
        paths = (
            workspace / "unitree_lerobot/unitree_lerobot/eval_robot/robot_control/robot_arm.py",
            workspace / "xr_teleoperate/teleop/robot_control/robot_arm.py",
        )
        for index, path in enumerate(paths):
            spec = importlib.util.spec_from_file_location(f"g1d_test_client_{index}", path)
            module = importlib.util.module_from_spec(spec)
            # The simulator environment's logging_mp API differs from the clients'
            # runtime. Ordinary logging is sufficient for this command-only test.
            with patch.dict("sys.modules", {"logging_mp": logging}):
                spec.loader.exec_module(module)
            for maximum_delta in (0.03, 0.8):
                results = []
                for simulation in (False, True):
                    with self.subTest(client=str(path), simulation=simulation, delta=maximum_delta):
                        # Skip __init__: it opens network channels and starts background threads.
                        client = object.__new__(module.G1_29_ArmController)
                        current = np.linspace(-0.1, 0.1, 14)
                        delta = np.linspace(-maximum_delta, maximum_delta, 14)
                        client.q_target = current + delta
                        client.tauff_target = np.linspace(-0.2, 0.2, 14)
                        client.waist_yaw_target = 0.15
                        client.ctrl_lock = threading.Lock()
                        client.motion_mode, client.simulation_mode = False, simulation
                        client.controller_alignment = True
                        client.control_dt, client.arm_velocity_limit = 1 / 250, 20.0
                        client.get_current_dual_arm_q = lambda: current.copy()
                        client.msg = unitree_hg_msg_dds__LowCmd_()
                        client.crc = CRC()

                        def capture(msg):
                            q = np.array([msg.motor_cmd[i].q for i in range(15, 29)])
                            tau = np.array([msg.motor_cmd[i].tau for i in range(15, 29)])
                            np.testing.assert_allclose(tau, client.tauff_target)
                            self.assertEqual(msg.motor_cmd[12].q, client.waist_yaw_target)
                            self.assertEqual(msg.crc, CRC().Crc(msg))
                            results.append(q)
                            raise Published

                        client.lowcmd_publisher = SimpleNamespace(Write=capture)
                        with self.assertRaises(Published):
                            client._ctrl_motor_state()
                np.testing.assert_allclose(results[0], results[1])
                expected = current + delta / max(maximum_delta / (20 / 250), 1.0)
                np.testing.assert_allclose(results[0], expected)

    def test_legacy_simulation_bypasses_host_target_limiter(self):
        path = (Path(__file__).resolve().parents[2]
                / "unitree_lerobot/unitree_lerobot/eval_robot/robot_control/robot_arm.py")
        spec = importlib.util.spec_from_file_location("g1d_test_legacy_client", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"logging_mp": logging}):
            spec.loader.exec_module(module)
        client = object.__new__(module.G1_29_ArmController)
        current = np.zeros(14)
        requested = np.linspace(-0.8, 0.8, 14)
        client.q_target = requested
        client.tauff_target = np.zeros(14)
        client.waist_yaw_target = 0.0
        client.ctrl_lock = threading.Lock()
        client.motion_mode = False
        client.simulation_mode = True
        client.controller_alignment = False
        client.control_dt, client.arm_velocity_limit = 1 / 250, 20.0
        client.get_current_dual_arm_q = lambda: current.copy()
        client.msg = unitree_hg_msg_dds__LowCmd_()
        client.crc = CRC()

        def capture(msg):
            actual = np.array([msg.motor_cmd[i].q for i in range(15, 29)])
            np.testing.assert_allclose(actual, requested)
            raise Published

        client.lowcmd_publisher = SimpleNamespace(Write=capture)
        with self.assertRaises(Published):
            client._ctrl_motor_state()


if __name__ == "__main__":
    unittest.main()
