"""Named joint control, RGB observations and reproducible sorting scene resets."""

import math
import random

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from robots.g1d_joints import GRIPPER_CLOSED, GRIPPER_OPEN, joint_indices, joint_state
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
        grip = GRIPPER_CLOSED + actions[:, [7, 15]].clamp(0, 1) * (GRIPPER_OPEN - GRIPPER_CLOSED)
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
        self._raw[env_ids] = joint_state(self._env)[env_ids]
        self._targets[env_ids] = self._asset.data.default_joint_pos[env_ids]


@configclass
class G1DSortActionCfg(ActionTermCfg):
    class_type: type = G1DSortAction
    asset_name: str = "robot"


def camera_rgb(env, name):
    """Unnormalized uint8 RGB, with an environment dimension."""
    return env.scene[name].data.output["rgb"][..., :3].clone()


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
