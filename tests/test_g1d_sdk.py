"""CPU-only protocol checks; create no DDS channels and launch no simulator."""

import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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
        stiffness = torch.full_like(positions, 80.0)
        damping = torch.full_like(positions, 3.0)
        actuator = SimpleNamespace(
            joint_indices=slice(None), is_implicit_model=True,
            stiffness=stiffness.clone(), damping=damping.clone(),
        )
        self.robot = SimpleNamespace(
            joint_names=names,
            data=SimpleNamespace(joint_pos=positions, default_joint_stiffness=stiffness,
                                 default_joint_damping=damping),
            actuators={"test": actuator},
            write_joint_stiffness_to_sim=Mock(), write_joint_damping_to_sim=Mock(),
            set_joint_velocity_target=Mock(), set_joint_effort_target=Mock(),
        )
        self.env = SimpleNamespace(scene={"robot": self.robot})
        self.bridge = object.__new__(G1DBridge)
        for key, value in dict(env=self.env, robot=self.robot, timeout=0.5, _error=None,
                               _lock=threading.Lock(), _commands={}, _fresh=set(),
                               _ignore_commands_until=0.0,
                               rejected_commands=0, accepted_commands=0,
                               controller_alignment=True).items():
            setattr(self.bridge, key, value)
        self.bridge._yaw_id = names.index("torso_Joint")
        self.bridge._arm_ids = [names.index(n) for n in ARM_JOINTS]
        self.bridge._init_motor_targets()
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
            msg.motor_cmd[i].kp = 80.0
            msg.motor_cmd[i].kd = 3.0
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

    def test_legacy_mode_uses_positions_and_ignores_other_motor_fields(self):
        self.bridge.controller_alignment = False
        msg = self.command()
        msg.motor_cmd[15].kp = -1.0
        msg.motor_cmd[15].dq = float("nan")
        msg.motor_cmd[15].tau = float("inf")
        self.send(msg)
        action = self.bridge.action()
        self.assertAlmostEqual(float(action[0, 0]), 0.1)
        self.robot.write_joint_stiffness_to_sim.assert_not_called()
        self.robot.write_joint_damping_to_sim.assert_not_called()
        self.robot.set_joint_velocity_target.assert_not_called()
        self.robot.set_joint_effort_target.assert_not_called()

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

    def test_commands_are_ignored_during_post_reset_hold(self):
        self.bridge._ignore_commands_until = 101.0
        self.send(self.command())
        self.grip("left", 0.0)
        self.assertEqual(self.bridge._commands, {})

        self.now.return_value = 101.01
        self.send(self.command())
        self.grip("left", 0.0)
        self.assertEqual(set(self.bridge._commands), {"arms", "left"})

    def test_sdk_gains_velocity_and_feedforward_follow_named_joints(self):
        msg = self.command()
        msg.motor_cmd[15].kp = 63.0
        msg.motor_cmd[15].kd = 2.5
        msg.motor_cmd[15].dq = 0.4
        msg.motor_cmd[15].tau = 1.2
        msg.motor_cmd[28].kp = 40.0
        msg.motor_cmd[12].kp = 300.0
        msg.motor_cmd[12].kd = 3.0
        msg.motor_cmd[12].tau = -0.2
        self.send(msg)
        self.bridge.action()
        kp = self.robot.write_joint_stiffness_to_sim.call_args.args[0]
        kd = self.robot.write_joint_damping_to_sim.call_args.args[0]
        dq = self.robot.set_joint_velocity_target.call_args.args[0]
        tau = self.robot.set_joint_effort_target.call_args.args[0]
        self.assertEqual(kp[0, [0, 13, 14]].tolist(), [63.0, 40.0, 300.0])
        self.assertEqual(float(kd[0, 0]), 2.5)
        self.assertAlmostEqual(float(dq[0, 0]), 0.4)
        self.assertAlmostEqual(float(tau[0, 0]), 1.2)
        self.assertAlmostEqual(float(tau[0, 14]), -0.2)
        self.assertEqual(self.robot.set_joint_effort_target.call_args.kwargs["joint_ids"],
                         [*self.bridge._arm_ids, self.bridge._yaw_id])
        model = self.robot.actuators["test"]
        self.assertEqual(float(model.stiffness[0, self.bridge._arm_ids[0]]), 63.0)
        self.assertEqual(float(model.damping[0, self.bridge._arm_ids[0]]), 2.5)
        self.bridge.action()
        self.assertEqual(self.robot.write_joint_stiffness_to_sim.call_count, 1)

    def test_motor_fields_reject_invalid_values_without_replacing_command(self):
        self.send(self.command())
        original = self.bridge._commands["arms"]
        for field, value in (("dq", float("nan")), ("tau", float("inf")),
                             ("kp", -1.0), ("kd", -1.0), ("kp", float("nan"))):
            msg = self.command()
            setattr(msg.motor_cmd[15], field, value)
            self.send(msg)
            self.assertIs(self.bridge._commands["arms"], original)
        self.assertEqual(self.bridge.rejected_commands, 5)

    def test_zero_gains_and_weighted_effort_then_timeout_restore_hold(self):
        msg = self.command()
        msg.motor_cmd[15].kp = msg.motor_cmd[15].kd = 0.0
        msg.motor_cmd[15].dq = 0.8
        msg.motor_cmd[15].tau = 2.0
        msg.motor_cmd[29].q = 0.5
        self.send(msg, "rt/arm_sdk")
        self.bridge.action()
        self.assertEqual(float(self.robot.write_joint_stiffness_to_sim.call_args.args[0][0, 0]), 0)
        self.assertAlmostEqual(float(self.robot.set_joint_velocity_target.call_args.args[0][0, 0]), 0.4)
        self.assertEqual(float(self.robot.set_joint_effort_target.call_args.args[0][0, 0]), 1)
        self.now.return_value = 100.6
        self.bridge.action()
        torch.testing.assert_close(self.robot.write_joint_stiffness_to_sim.call_args.args[0],
                                   self.bridge._default_kp)
        self.assertEqual(self.robot.set_joint_effort_target.call_args.args[0].abs().sum(), 0)
        self.assertEqual(self.robot.set_joint_velocity_target.call_args.args[0].abs().sum(), 0)

    def test_inactive_joint_clears_previous_feedforward_and_restores_gains(self):
        msg = self.command()
        msg.motor_cmd[15].kp = 20.0
        msg.motor_cmd[15].tau = 2.0
        self.send(msg)
        self.bridge.action()
        msg.motor_cmd[15].mode = 0
        msg.motor_cmd[15].tau = float("nan")  # Inactive fields are not consumed.
        self.send(msg)
        self.bridge.action()
        self.assertEqual(float(self.robot.write_joint_stiffness_to_sim.call_args.args[0][0, 0]), 80)
        self.assertEqual(float(self.robot.set_joint_effort_target.call_args.args[0][0, 0]), 0)

    def test_reset_clears_motor_commands_and_restores_gains(self):
        msg = self.command()
        msg.motor_cmd[15].kp = 0
        msg.motor_cmd[15].tau = 2
        self.send(msg)
        self.bridge.action()
        self.bridge.reset_hold = 1.1
        self.bridge.update_state = Mock()
        self.bridge.reset()
        self.assertEqual(self.bridge._commands, {})
        torch.testing.assert_close(self.robot.write_joint_stiffness_to_sim.call_args.args[0],
                                   self.bridge._default_kp)
        self.assertEqual(self.robot.set_joint_effort_target.call_args.args[0].abs().sum(), 0)


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
