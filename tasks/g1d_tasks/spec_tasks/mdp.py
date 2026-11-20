"""Spec objects placed in the torso frame, and the scene observations of the spec tasks."""

import math

import torch
from isaaclab.utils.math import quat_mul

TORSO_LINK = "torso_link"
# isaaclab_biman_sim's hand links: grasp = both fingers of one hand, touch = any of the three
HAND_LINKS = {side: [f"{side}_dex1_finger_link_1", f"{side}_dex1_finger_link_2", f"{side}_dex1_base_link"]
              for side in ("left", "right")}


def object_key(name):
    return f"obj_{name}"


def cache_torso_frame(env, env_ids):
    """Startup: the torso origin and heading in each env frame, the frame the specs place objects in. Taken with the
    robot at its start pose; the torso does not move during an episode (only torso yaw could, and every reset
    restores it), so later resets reuse it."""
    robot = env.scene["robot"]
    body = robot.body_names.index(TORSO_LINK)
    w, x, y, z = robot.data.body_quat_w[:, body].unbind(-1)
    env.spec_torso_pos = (robot.data.body_pos_w[:, body] - env.scene.env_origins).clone()
    env.spec_torso_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).clone()
    p = env.spec_torso_pos[0].tolist()
    table_z = env.cfg.sorting["table"]["height"]
    print(f"[G1-D spec] Torso frame: origin ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m, heading "
          f"{math.degrees(float(env.spec_torso_yaw[0])):.2f} deg; table top {table_z:.4f} m = "
          f"{table_z - p[2]:+.4f} m from the torso origin", flush=True)


def spec_poses(env, env_ids):
    """{object: (positions (n, 3), wxyz quaternions (n, 4))} in the env frames: the spec's torso-frame layout."""
    pos_t, yaw = env.spec_torso_pos[env_ids], env.spec_torso_yaw[env_ids]
    c, s = torch.cos(yaw), torch.sin(yaw)
    q_yaw = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
    table_z = env.cfg.sorting["table"]["height"]
    poses = {}
    for o in env.cfg.task_spec["objects"]:
        x, y = (float(v) for v in o["xy_torso"])
        qx, qy, qz, qw = (float(v) for v in o["orientation_torso_xyzw"])
        pos = torch.stack((pos_t[:, 0] + c * x - s * y, pos_t[:, 1] + s * x + c * y,
                           torch.full_like(yaw, table_z + float(o["centre_z_above_table"]))), -1)
        quat = quat_mul(q_yaw, torch.tensor([qw, qx, qy, qz], device=yaw.device).expand_as(q_yaw))
        poses[o["name"]] = pos, quat
    return poses


def place_objects(env, env_ids):
    """Reset: every spec object at its authored pose, at rest."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    for name, (pos, quat) in spec_poses(env, env_ids).items():
        asset = env.scene[object_key(name)]
        asset.write_root_pose_to_sim(torch.cat((pos + env.scene.env_origins[env_ids], quat), -1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)


reset_objects = place_objects


def object_poses(env):
    values = []
    for o in env.cfg.task_spec["objects"]:
        data = env.scene[object_key(o["name"])].data
        values.append(torch.cat((data.root_pos_w - env.scene.env_origins, data.root_quat_w), dim=1))
    return torch.cat(values, dim=1)


def task_progress(env):
    """Fraction of the rubric's stages latched, (num_envs,) (privileged; for inspection, not for the policy)."""
    evaluator = getattr(env, "g1d_evaluation", None)
    if evaluator is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.as_tensor(evaluator.rubric.progress(), dtype=torch.float32, device=env.device)


def task_progress_obs(env):
    return task_progress(env)[:, None]
