"""Named joint control, RGB observations and reproducible sorting scene resets."""

import math
import random

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from robots.g1d_joints import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    binary_gripper_targets,
    joint_indices,
    joint_state,
)
from .camera_lens import LensWarp
from .layout import objects, sample_layout
from .evaluation.config import STAGE_NAMES
from .evaluation.predicates import fully_inside_bin


class G1DSortAction(ActionTerm):
    """Absolute arm angles and coupled grippers; chassis, lift and torso hold their defaults."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._arm_ids, self._finger_ids = joint_indices(self._asset)
        self._raw = joint_state(env).clone()
        self._targets = self._asset.data.default_joint_pos.clone()
        self._gripper_control = env.cfg.sorting["robot"]["gripper_control"]
        threshold_midpoint = (
            self._gripper_control["close_threshold"]
            + self._gripper_control["open_threshold"]
        ) / 2
        self._binary_grip = (self._raw[:, [7, 15]] >= threshold_midpoint).float()

    @property
    def action_dim(self):
        return 16

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._targets

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("G1-D action contains NaN or infinity")
        self._raw.copy_(actions)
        self._targets.copy_(self._asset.data.default_joint_pos)
        arm_targets = torch.cat((actions[:, :7], actions[:, 8:15]), dim=1)
        limits = self._asset.data.soft_joint_pos_limits[:, self._arm_ids]
        self._targets[:, self._arm_ids] = torch.clamp(arm_targets, limits[..., 0], limits[..., 1])
        grip_command = actions[:, [7, 15]].clamp(0, 1)
        if self._gripper_control["mode"] == "binary":
            self._binary_grip = binary_gripper_targets(
                grip_command,
                self._binary_grip,
                self._gripper_control["close_threshold"],
                self._gripper_control["open_threshold"],
            )
            grip_command = self._binary_grip
        grip = GRIPPER_CLOSED + grip_command * (GRIPPER_OPEN - GRIPPER_CLOSED)
        self._targets[:, self._finger_ids] = grip.repeat_interleave(2, dim=1)
        if hasattr(self._env, "_g1d_torso_yaw_target"):
            yaw_id = self._asset.joint_names.index("torso_Joint")
            limits = self._asset.data.soft_joint_pos_limits[:, yaw_id]
            self._targets[:, yaw_id] = self._env._g1d_torso_yaw_target.clamp(limits[:, 0], limits[:, 1])

    def apply_actions(self):
        self._asset.set_joint_position_target(self._targets)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        measured = joint_state(self._env)
        self._raw[env_ids] = measured[env_ids]
        self._targets[env_ids] = self._asset.data.default_joint_pos[env_ids]
        threshold_midpoint = (
            self._gripper_control["close_threshold"]
            + self._gripper_control["open_threshold"]
        ) / 2
        self._binary_grip[env_ids] = (measured[env_ids][:, [7, 15]] >= threshold_midpoint).float()


@configclass
class G1DSortActionCfg(ActionTermCfg):
    class_type: type = G1DSortAction
    asset_name: str = "robot"


CAMERA_SETTINGS = {"front_camera": "head", "left_wrist_camera": "left_wrist", "right_wrist_camera": "right_wrist"}


def camera_rgb(env, name):
    """Unnormalized uint8 RGB, with an environment dimension. Cameras with calibrated intrinsics are rendered as
    a virtual pinhole and warped into the calibrated lens here (camera_lens.py)."""
    rgb = env.scene[name].data.output["rgb"][..., :3]
    camera = env.cfg.sorting["cameras"].get(CAMERA_SETTINGS.get(name), {})
    if "intrinsics" not in camera:
        return rgb.clone()
    warps = env.__dict__.setdefault("_g1d_lens_warps", {})
    if name not in warps:
        warps[name] = LensWarp(camera["intrinsics"])
    return warps[name](rgb)


def object_poses(env):
    values = []
    for obj in objects(env.cfg.sorting):
        data = env.scene[obj["name"]].data
        values.append(torch.cat((data.root_pos_w - env.scene.env_origins, data.root_quat_w), dim=1))
    return torch.cat(values, dim=1) if values else torch.empty((env.num_envs, 0), device=env.device)


def reset_objects(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    # The environment seeds Python's RNG; reset(seed=N) reproduces the same layout.
    layouts = [sample_layout(env.cfg.sorting, random) for _ in env_ids]
    for obj in objects(env.cfg.sorting):
        asset = env.scene[obj["name"]]
        state = asset.data.default_root_state[env_ids].clone()
        for row, layout in enumerate(layouts):
            x, y, z, yaw = layout[obj["name"]]
            state[row, :7] = torch.tensor([x, y, z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], device=env.device)
        state[:, :3] += env.scene.env_origins[env_ids]
        state[:, 7:] = 0.0
        asset.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)


def apply_robot_payloads(env, env_ids):
    """Add the point masses of robot.payloads to their links (mass, centre of mass and inertia).

    Used for load the imported URDF lacks, e.g. the wrist camera, its mount and cables. Each payload is
    a mass at a point given in the link frame; the link's centre of mass moves to the combined one and
    its inertia is shifted with the parallel-axis theorem.
    """
    payloads = env.cfg.sorting["robot"]["payloads"]
    if not payloads:
        return
    robot = env.scene["robot"]
    view = robot.root_physx_view
    ids = torch.arange(env.scene.num_envs, device="cpu")
    masses, coms, inertias = view.get_masses(), view.get_coms().clone(), view.get_inertias().clone()
    eye = torch.eye(3, dtype=coms.dtype)

    def parallel_axis(mass, offset):  # inertia of a point mass about a point `offset` away
        return mass[:, None, None] * ((offset * offset).sum(-1)[:, None, None] * eye - offset[:, :, None] * offset[:, None, :])

    for payload in payloads:
        if payload["body"] not in robot.body_names:
            raise ValueError(f"robot.payloads: no body {payload['body']!r}; bodies are {robot.body_names}")
        b = robot.body_names.index(payload["body"])
        m0, c0 = masses[:, b].clone(), coms[:, b, :3].clone()
        m, p = payload["mass_kg"], torch.tensor(payload["com_m"], dtype=coms.dtype).expand_as(c0)
        c1 = (m0[:, None] * c0 + m * p) / (m0 + m)[:, None]
        # PhysX gives the inertia in the centre-of-mass frame (orientation: quaternion x, y, z, w).
        x, y, z, w = coms[:, b, 3:].unbind(-1)
        rot = torch.stack((
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ), dim=-1).reshape(-1, 3, 3)
        inertia_body = rot @ inertias[:, b].reshape(-1, 3, 3) @ rot.transpose(1, 2)
        inertia_body = (inertia_body + parallel_axis(m0, c0 - c1)
                        + parallel_axis(torch.full_like(m0, m), p - c1))
        inertias[:, b] = (rot.transpose(1, 2) @ inertia_body @ rot).reshape(-1, 9)
        masses[:, b] = m0 + m
        coms[:, b, :3] = c1
        print(f"[G1-D] Payload {m:.3f} kg on {payload['body']} at {payload['com_m']}: link mass "
              f"{float(m0[0]):.3f} -> {float(m0[0] + m):.3f} kg", flush=True)
    view.set_masses(masses, ids)
    view.set_coms(coms, ids)
    view.set_inertias(inertias, ids)


def sorted_fraction(env):
    """Geometric reward: fraction fully inside the target bin and moving slowly."""
    items = objects(env.cfg.sorting)
    score = torch.zeros(env.num_envs, device=env.device)
    boxes = {box["name"]: box for box in env.cfg.sorting["bins"]}
    table_z = env.cfg.sorting["table"]["height"]
    for obj in items:
        data = env.scene[obj["name"]].data
        pos = data.root_pos_w - env.scene.env_origins
        box = boxes[obj["target_bin"]]
        inside = fully_inside_bin(pos, data.root_quat_w, obj, box, table_z)
        inside &= data.root_lin_vel_w.norm(dim=-1) < 0.03
        inside &= data.root_ang_vel_w.norm(dim=-1) < 0.2
        score += inside.float()
    return score / max(1, len(items))


def evaluation_stages(env):
    """Monotonic per-object stage flags for privileged inspection only."""
    evaluator = getattr(env, "g1d_evaluation", None)
    count = len(objects(env.cfg.sorting)) * len(STAGE_NAMES)
    if evaluator is None:
        return torch.zeros((env.num_envs, count), device=env.device)
    return evaluator.achieved.float().flatten(start_dim=1)


def evaluation_summary(env):
    """Stage rates, PSR, success, drops and wrong-bin events."""
    evaluator = getattr(env, "g1d_evaluation", None)
    if evaluator is None:
        return torch.zeros((env.num_envs, len(STAGE_NAMES) + 4), device=env.device)
    return torch.cat((
        evaluator.stage_rates,
        evaluator.task_score[:, None],
        evaluator.success.float()[:, None],
        evaluator.drop_count.sum(dim=1, keepdim=True).float(),
        evaluator.wrong_bin_count.sum(dim=1, keepdim=True).float(),
    ), dim=1)
