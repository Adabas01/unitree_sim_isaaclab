"""CPU-only protocol checks; create no DDS channels and launch no simulator."""

import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
from unitree_sdk2py.utils.crc import CRC
from dds.g1d_bridge import G1DBridge
from robots.g1d_joints import (
    ARM_JOINTS,
    FINGER_JOINTS,
    GRIPPER_OPEN,
    binary_gripper_targets,
    joint_state,
)


class SDKMappingTests(unittest.TestCase):
    def setUp(self):
        names = list(reversed([*ARM_JOINTS, *FINGER_JOINTS, "torso_Joint"]))
        positions = torch.zeros((1, len(names)))
        for name in FINGER_JOINTS:
            positions[:, names.index(name)] = GRIPPER_OPEN
        self.robot = SimpleNamespace(joint_names=names, data=SimpleNamespace(joint_pos=positions))
        self.env = SimpleNamespace(scene={"robot": self.robot})
        self.bridge = object.__new__(G1DBridge)
        for key, value in dict(env=self.env, robot=self.robot, timeout=0.5, _error=None,
                               _lock=threading.Lock(), _commands={}, _fresh=set(),
                               rejected_commands=0, accepted_commands=0).items():
            setattr(self.bridge, key, value)
        self.bridge._yaw_id = names.index("torso_Joint")
        self.bridge._held_action = joint_state(self.env).clone()
        self.bridge._held_yaw = positions[:, self.bridge._yaw_id].clone()
        self.clock = patch("dds.g1d_bridge.time.monotonic", return_value=100.0)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)

    def command(self):
        msg = unitree_hg_msg_dds__LowCmd_()
        for i in range(15, 29):
            msg.motor_cmd[i].mode = 1
            msg.motor_cmd[i].q = (i - 14) / 10
        msg.motor_cmd[12].mode = 1
        msg.motor_cmd[12].q = 0.2
        return msg

    def send(self, msg, topic="rt/lowcmd"):
        msg.crc = CRC().Crc(msg)
        self.bridge._arm_command(msg, topic)

    def grip(self, side, q):
        msg = MotorCmds_()
        msg.cmds.append(unitree_go_msg_dds__MotorCmd_())
        msg.cmds[0].q = q
        self.bridge._gripper_command(msg, side)

    def test_named_joint_mapping_and_torso_slot(self):
        self.send(self.command())
        action = self.bridge.action()
        torch.testing.assert_close(action[0, :7], torch.arange(1, 8) / 10)
        torch.testing.assert_close(action[0, 8:15], torch.arange(8, 15) / 10)
        self.assertAlmostEqual(float(self.env._g1d_torso_yaw_target[0]), 0.2)
        self.assertEqual(float(action[0, 7]), 1.0)

    def test_crc_nonfinite_and_mode_rejected(self):
        valid = self.command()
        self.send(valid)
        initial = self.bridge.action().clone()
        valid.crc ^= 1
        self.bridge._arm_command(valid, "rt/lowcmd")
        invalid = self.command()
        invalid.motor_cmd[15].q = float("nan")
        self.send(invalid)
        invalid = self.command()
        invalid.mode_pr = 1
        self.send(invalid)
        self.assertEqual(self.bridge.rejected_commands, 3)
        torch.testing.assert_close(initial, self.bridge.action())

    def test_arm_sdk_weight(self):
        msg = self.command()
        msg.motor_cmd[29].q = 0.5
        self.send(msg, "rt/arm_sdk")
        self.assertAlmostEqual(float(self.bridge.action()[0, 0]), 0.05)
        msg.motor_cmd[29].q = 0.0
        self.send(msg, "rt/arm_sdk")
        self.assertAlmostEqual(float(self.bridge.action()[0, 0]), 0.0)

    def test_gripper_units_and_sides(self):
        self.grip("left", 0.0)
        self.grip("right", 5.4)
        action = self.bridge.action()
        self.assertEqual(float(action[0, 7]), 0.0)
        self.assertEqual(float(action[0, 15]), 1.0)
        self.grip("left", 2.7)
        self.grip("right", -2.0)
        action = self.bridge.action()
        self.assertEqual(float(action[0, 7]), 0.5)
        self.assertEqual(float(action[0, 15]), 0.0)

    def test_watchdog_latches_once_and_streams_expire_independently(self):
        self.send(self.command())
        self.bridge.action()
        self.now.return_value = 100.2
        self.grip("left", 0)
        self.bridge.action()
        index = self.robot.joint_names.index(ARM_JOINTS[0])
        self.robot.data.joint_pos[0, index] = 0.3
        self.now.return_value = 100.6
        a = self.bridge.action()
        self.assertAlmostEqual(float(a[0, 0]), 0.3)
        self.assertEqual(float(a[0, 7]), 0.0)
        self.robot.data.joint_pos[0, index] = 0.4
        self.now.return_value = 100.65
        self.assertAlmostEqual(float(self.bridge.action()[0, 0]), 0.3)


class BinaryGripperTests(unittest.TestCase):
    def test_thresholds_and_deadband(self):
        previous = torch.tensor([[1.0, 0.0]])
        targets = binary_gripper_targets(
            torch.tensor([[0.2, 0.8]]), previous, close_threshold=0.45, open_threshold=0.55
        )
        torch.testing.assert_close(targets, torch.tensor([[0.0, 1.0]]))

        held = binary_gripper_targets(
            torch.tensor([[0.5, 0.5]]), targets, close_threshold=0.45, open_threshold=0.55
        )
        torch.testing.assert_close(held, targets)


if __name__ == "__main__":
    unittest.main()
