"""Stateful, per-object stage evaluation updated once per control step."""

from collections.abc import Sequence

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply

from robots.g1d_joints import joint_state
from ..layout import objects
from .config import STAGE_NAMES
from .predicates import center_over_bin, fully_inside_bin, point_surface_distance, world_extent
from .state_machine import advance_condition_dwell, advance_ordered_stages


REACH, GRASP, LIFT, TRANSPORT, PLACE = range(len(STAGE_NAMES))
HAND_NAMES = ("left", "right")


class StageEvaluator(ManagerTermBase):
    """Track ordered milestones without exposing simulator state to the policy."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.settings = env.cfg.sorting
        self.protocol = env.cfg.evaluation
        self.items = objects(self.settings)
        self.num_objects = len(self.items)
        self.num_stages = len(STAGE_NAMES)
        self.enabled = self.protocol["enabled"]
        self.weights = torch.tensor(
            [self.protocol["stages"][name]["weight"] for name in STAGE_NAMES],
            device=env.device,
            dtype=torch.float32,
        )
        self.dwell_required = torch.tensor(
            [self.protocol["stages"][name]["dwell_s"] for name in STAGE_NAMES],
            device=env.device,
            dtype=torch.float32,
        )
        shape = (env.num_envs, self.num_objects, self.num_stages)
        self.raw = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.achieved = torch.zeros_like(self.raw)
        self.dwell_s = torch.zeros(shape, device=env.device)
        self.first_time_s = torch.full(shape, -1.0, device=env.device)
        self.grasp_hand = torch.full((env.num_envs, self.num_objects), -1, dtype=torch.int8, device=env.device)
        self.drop_count = torch.zeros((env.num_envs, self.num_objects), dtype=torch.int32, device=env.device)
        self.wrong_bin_count = torch.zeros_like(self.drop_count)
        self._drop_active = torch.zeros((env.num_envs, self.num_objects), dtype=torch.bool, device=env.device)
        self._wrong_bin_active = torch.zeros_like(self._drop_active)
        self.task_success_dwell_s = torch.zeros(env.num_envs, device=env.device)
        self.success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.episode_serial = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.initial_object_pos = torch.full((env.num_envs, self.num_objects, 3), torch.nan, device=env.device)
        self.geometric_sorted_fraction = torch.zeros(env.num_envs, device=env.device)
        object_shape = (env.num_envs, self.num_objects)
        self.minimum_reach_distance_m = torch.full(object_shape, torch.inf, device=env.device)
        self.maximum_opposing_contact_force_n = torch.zeros(object_shape, device=env.device)
        self.maximum_lift_clearance_m = torch.full(object_shape, -torch.inf, device=env.device)
        self.ever_over_target = torch.zeros(object_shape, dtype=torch.bool, device=env.device)
        self.ever_inside_target = torch.zeros_like(self.ever_over_target)
        self.inside_target = torch.zeros_like(self.ever_over_target)

        robot = env.scene["robot"]
        self._finger_body_ids = [
            [robot.body_names.index(f"{side}_dex1_finger_link_{index}") for index in (1, 2)]
            for side in HAND_NAMES
        ]
        self._pinch_point_offset = torch.tensor(
            self.protocol["stages"]["reach"]["pinch_point_offset_m"],
            dtype=torch.float32,
            device=env.device,
        )
        env.g1d_evaluation = self

    @property
    def per_object_score(self):
        if self.num_objects == 0:
            return torch.empty((self.num_envs, 0), device=self.device)
        return (self.achieved.float() * self.weights).sum(-1) / self.weights.sum()

    @property
    def task_score(self):
        if self.num_objects == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return self.per_object_score.mean(-1)

    @property
    def stage_rates(self):
        if self.num_objects == 0:
            return torch.zeros((self.num_envs, self.num_stages), device=self.device)
        return self.achieved.float().mean(1)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.raw[env_ids] = False
        self.achieved[env_ids] = False
        self.dwell_s[env_ids] = 0
        self.first_time_s[env_ids] = -1
        self.grasp_hand[env_ids] = -1
        self.drop_count[env_ids] = 0
        self.wrong_bin_count[env_ids] = 0
        self._drop_active[env_ids] = False
        self._wrong_bin_active[env_ids] = False
        self.task_success_dwell_s[env_ids] = 0
        self.success[env_ids] = False
        self.steps[env_ids] = 0
        self.episode_serial[env_ids] += 1
        self.initial_object_pos[env_ids] = torch.nan
        self.geometric_sorted_fraction[env_ids] = 0
        self.minimum_reach_distance_m[env_ids] = torch.inf
        self.maximum_opposing_contact_force_n[env_ids] = 0
        self.maximum_lift_clearance_m[env_ids] = -torch.inf
        self.ever_over_target[env_ids] = False
        self.ever_inside_target[env_ids] = False
        self.inside_target[env_ids] = False

    def _finger_forces(self):
        """Return filtered normal-force magnitudes as (env, hand, finger, object)."""
        result = torch.zeros(
            (self.num_envs, len(HAND_NAMES), 2, self.num_objects), device=self.device
        )
        for hand_index, side in enumerate(HAND_NAMES):
            for finger_index in range(2):
                sensor = self._env.scene[f"{side}_finger_{finger_index + 1}_contact"]
                matrix = sensor.data.force_matrix_w
                if matrix is None or matrix.shape[1] != 1 or matrix.shape[2] != self.num_objects:
                    shape = None if matrix is None else tuple(matrix.shape)
                    raise RuntimeError(
                        f"Unexpected {side} Dex1 contact matrix {shape}; expected "
                        f"({self.num_envs}, 1, {self.num_objects}, 3)"
                    )
                result[:, hand_index, finger_index] = matrix[:, 0].norm(dim=-1)
        return result

    def __call__(self, env):
        if not self.enabled:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.num_objects == 0:
            self.steps += 1
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        robot = env.scene["robot"]
        world_pos = torch.stack([env.scene[item["name"]].data.root_pos_w for item in self.items], dim=1)
        quat = torch.stack([env.scene[item["name"]].data.root_quat_w for item in self.items], dim=1)
        lin_speed = torch.stack(
            [env.scene[item["name"]].data.root_lin_vel_w.norm(dim=-1) for item in self.items], dim=1
        )
        ang_speed = torch.stack(
            [env.scene[item["name"]].data.root_ang_vel_w.norm(dim=-1) for item in self.items], dim=1
        )
        local_pos = world_pos - env.scene.env_origins[:, None, :]
        starting = self.steps == 0
        self.initial_object_pos[starting] = local_pos[starting]

        pinch_positions = []
        for body_ids in self._finger_body_ids:
            body_pos = robot.data.body_pos_w[:, body_ids]
            body_quat = robot.data.body_quat_w[:, body_ids]
            offset = self._pinch_point_offset.view(1, 1, 3).expand_as(body_pos)
            world_offset = quat_apply(
                body_quat.reshape(-1, 4), offset.reshape(-1, 3)
            ).reshape_as(body_pos)
            pinch_positions.append((body_pos + world_offset).mean(dim=1))
        pinch_pos = torch.stack(pinch_positions, dim=1)
        distances = torch.stack(
            [point_surface_distance(pinch_pos, world_pos[:, i], quat[:, i], item)
             for i, item in enumerate(self.items)],
            dim=2,
        )
        reach_cfg = self.protocol["stages"]["reach"]
        closest = distances.min(dim=1).values
        self.minimum_reach_distance_m = torch.minimum(self.minimum_reach_distance_m, closest)
        reach_limit = torch.where(
            self.raw[:, :, REACH],
            torch.full_like(closest, reach_cfg["exit_distance_m"]),
            torch.full_like(closest, reach_cfg["enter_distance_m"]),
        )
        reached = closest <= reach_limit

        finger_forces = self._finger_forces()
        grasp_cfg = self.protocol["stages"]["grasp"]
        both_fingers = (finger_forces >= grasp_cfg["minimum_contact_force_n"]).all(dim=2)
        opposing_contact_force = finger_forces.min(dim=2).values.max(dim=1).values
        self.maximum_opposing_contact_force_n = torch.maximum(
            self.maximum_opposing_contact_force_n, opposing_contact_force
        )
        gripper_open = joint_state(env)[:, [7, 15]]
        hand_grasp = both_fingers & (
            gripper_open[:, :, None] <= grasp_cfg["maximum_gripper_open_fraction"]
        )
        grasped = hand_grasp.any(dim=1)

        extent_z = torch.stack(
            [world_extent(quat[:, i], item)[:, 2] for i, item in enumerate(self.items)], dim=1
        )
        bottom_clearance = local_pos[:, :, 2] - extent_z - self.settings["table"]["height"]
        self.maximum_lift_clearance_m = torch.maximum(
            self.maximum_lift_clearance_m, bottom_clearance
        )
        lifted = bottom_clearance >= self.protocol["stages"]["lift"]["table_clearance_m"]

        boxes = {box["name"]: box for box in self.settings["bins"]}
        transport = torch.zeros_like(reached)
        target_geometry = torch.zeros_like(reached)
        wrong_geometry = torch.zeros_like(reached)
        for i, item in enumerate(self.items):
            target = boxes[item["target_bin"]]
            transport[:, i] = center_over_bin(
                local_pos[:, i], target, self.protocol["stages"]["transport"]["bin_xy_margin_m"]
            )
            target_geometry[:, i] = fully_inside_bin(
                local_pos[:, i], quat[:, i], item, target, self.settings["table"]["height"],
                self.protocol["stages"]["place"]["bin_margin_m"],
            )
            for name, box in boxes.items():
                if name != item["target_bin"]:
                    wrong_geometry[:, i] |= fully_inside_bin(
                        local_pos[:, i], quat[:, i], item, box, self.settings["table"]["height"],
                        self.protocol["stages"]["place"]["bin_margin_m"],
                    )

        place_cfg = self.protocol["stages"]["place"]
        self.ever_over_target |= transport
        self.ever_inside_target |= target_geometry
        self.inside_target.copy_(target_geometry)
        stable = (lin_speed <= place_cfg["maximum_linear_speed_m_s"]) & (
            ang_speed <= place_cfg["maximum_angular_speed_rad_s"]
        )
        release_contact = (finger_forces >= place_cfg["release_contact_force_n"]).any(dim=2).any(dim=1)
        selected_hand = self.grasp_hand.long().clamp_min(0)
        selected_open = torch.gather(gripper_open, 1, selected_hand)
        released = ~release_contact
        if place_cfg["require_release"]:
            released &= (self.grasp_hand >= 0) & (
                selected_open >= place_cfg["minimum_gripper_open_fraction"]
            )
        placed = target_geometry & stable & released
        wrong_bin = wrong_geometry & stable & released
        self.geometric_sorted_fraction = (target_geometry & stable).float().mean(dim=1)

        conditions = torch.stack((reached, grasped, lifted, transport, placed), dim=-1)
        episode_time = env.episode_length_buf.float() * env.step_dt
        self.achieved, self.dwell_s, newly_achieved = advance_ordered_stages(
            self.achieved,
            self.dwell_s,
            conditions,
            self.dwell_required,
            env.step_dt,
        )
        for stage_index in range(self.num_stages):
            self.first_time_s[:, :, stage_index] = torch.where(
                newly_achieved[:, :, stage_index],
                episode_time[:, None].expand_as(newly_achieved[:, :, stage_index]),
                self.first_time_s[:, :, stage_index],
            )
            if stage_index == GRASP:
                hand_strength = finger_forces.sum(dim=2)
                chosen = hand_strength.argmax(dim=1).to(torch.int8)
                self.grasp_hand = torch.where(
                    newly_achieved[:, :, stage_index], chosen, self.grasp_hand
                )

        dropped = self.achieved[:, :, LIFT] & ~grasped & ~target_geometry & (
            bottom_clearance <= self.protocol["failures"]["drop_clearance_m"]
        )
        self.drop_count += (dropped & ~self._drop_active).to(self.drop_count.dtype)
        self.wrong_bin_count += (wrong_bin & ~self._wrong_bin_active).to(self.wrong_bin_count.dtype)
        self._drop_active.copy_(dropped)
        self._wrong_bin_active.copy_(wrong_bin)
        self.raw.copy_(conditions)
        self.steps += 1

        terminal_placed = target_geometry & stable
        if self.protocol["task_success"]["require_release"]:
            terminal_placed &= ~release_contact
        all_sorted = terminal_placed.all(dim=1)
        self.task_success_dwell_s, self.success = advance_condition_dwell(
            all_sorted,
            self.task_success_dwell_s,
            self.protocol["task_success"]["dwell_s"],
            env.step_dt,
        )
        if self.protocol["task_success"]["terminate_episode"]:
            return self.success.clone()
        return torch.zeros_like(self.success)
