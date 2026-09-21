"""G1-D joint conventions shared by the simulator and the SDK adapter."""

import torch

ARM_SUFFIXES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
ARM_JOINTS = [f"{side}_{joint}_joint" for side in ("left", "right") for joint in ARM_SUFFIXES]
FINGER_JOINTS = [f"{side}_dex1_finger_joint_{i}" for side in ("left", "right") for i in (1, 2)]
GRIPPER_CLOSED = -0.02
GRIPPER_OPEN = 0.0245


def binary_gripper_targets(commands, previous, close_threshold, open_threshold):
    """Quantize normalized commands with a deadband that retains the previous state."""
    commands = commands.clamp(0, 1)
    return torch.where(
        commands <= close_threshold,
        torch.zeros_like(commands),
        torch.where(commands >= open_threshold, torch.ones_like(commands), previous),
    )


def joint_indices(robot):
    if not hasattr(robot, "_g1d_indices"):
        names = robot.joint_names
        robot._g1d_indices = ([names.index(n) for n in ARM_JOINTS], [names.index(n) for n in FINGER_JOINTS])
    return robot._g1d_indices


def joint_state(env):
    """[left arm 7 radians, left grip 0..1, right arm 7 radians, right grip 0..1]."""
    robot = env.scene["robot"]
    arms, fingers = joint_indices(robot)
    q = robot.data.joint_pos
    grip = ((q[:, fingers].reshape(-1, 2, 2).mean(-1) - GRIPPER_CLOSED)
            / (GRIPPER_OPEN - GRIPPER_CLOSED)).clamp(0, 1)
    return torch.cat((q[:, arms[:7]], grip[:, :1], q[:, arms[7:]], grip[:, 1:]), dim=1)
