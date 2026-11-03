"""G1-D joint conventions shared by the simulator and the SDK adapter."""

import torch

ARM_SUFFIXES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
ARM_JOINTS = [f"{side}_{joint}_joint" for side in ("left", "right") for joint in ARM_SUFFIXES]
FINGER_JOINTS = [f"{side}_dex1_finger_joint_{i}" for side in ("left", "right") for i in (1, 2)]
GRIPPER_CLOSED = -0.02  # finger joint limits [m]
GRIPPER_OPEN = 0.0245
SDK_GRIPPER_OPEN = 5.4  # raw Dex1 units: 0 closed ... 5.4 open
# Pad gap of the simulated fingers at the lower joint limit; it grows by 2 mm per mm of finger travel.
SIM_GAP_AT_CLOSED_M = 0.0058
# Real Dex1-1, measured with a caliper (master-thesis eval/sim2real/gripper_calib.py, run 20261102-165026,
# mean of both grippers): pad gap = REAL_GAP_AT_RAW0_M + REAL_GAP_PER_RAW_M * raw.
REAL_GAP_AT_RAW0_M = 0.00028
REAL_GAP_PER_RAW_M = 0.01608


def grip_to_finger(fraction):
    """Finger joint target [m] for a gripper command (raw / 5.4), giving the real gripper's pad gap.

    Near 0 the target lies below the joint limit (the real pads close to 0.3 mm, the simulated ones only to
    5.8 mm), so a fully closed gripper presses against its limit or against what it holds.
    """
    gap = REAL_GAP_AT_RAW0_M + REAL_GAP_PER_RAW_M * SDK_GRIPPER_OPEN * fraction
    return GRIPPER_CLOSED + (gap - SIM_GAP_AT_CLOSED_M) / 2


def finger_to_grip(q):
    """Gripper reading (raw / 5.4, clamped to 0..1) for a finger joint position, as the real gripper reports it."""
    gap = SIM_GAP_AT_CLOSED_M + 2 * (q - GRIPPER_CLOSED)
    return ((gap - REAL_GAP_AT_RAW0_M) / (REAL_GAP_PER_RAW_M * SDK_GRIPPER_OPEN)).clamp(0, 1)


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
    grip = finger_to_grip(q[:, fingers].reshape(-1, 2, 2).mean(-1))
    return torch.cat((q[:, arms[:7]], grip[:, :1], q[:, arms[7:]], grip[:, 1:]), dim=1)
