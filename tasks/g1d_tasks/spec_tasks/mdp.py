"""Spec objects placed in the torso frame, and the scene observations of the spec tasks."""

import math

import numpy as np
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


# Layout perturbations (cfg.perturbation = {"kind", "seed"}), from ponimatkin/chiral biman/perturb.py at 554fb5e:
#   Default   every reset puts the objects at the spec's poses (chiral's evaluation default).
#   VB-POSE   every reset re-places the main objects (role "main") anywhere in REALM's G1 / G1-D perturbation zone in
#             the torso frame, clear of the other objects (3.5 cm) and of the start-pose hands, turned about the
#             vertical by N(0, pi) (N(0, 0.15) for a lid / door task) from their spec orientation. Draws come from
#             numpy default_rng([seed, draw]), draw = this process's reset count, in chiral's order, so a seed gives
#             chiral's layouts wherever the keep-outs agree. No free spot: the spec spot, still turned (as chiral).
PERTURBATIONS = ("Default", "VB-POSE")
ZONE_TORSO = (0.24, 0.44, -0.22, 0.22)  # x_min, x_max, y_min, y_max [m]
MIN_SEP = 0.035
HAND_KEEPOUT_R = 0.05  # around each Dex1 link and a point 7 cm ahead of it (the fingers)
HAND_REACH = 0.07


def start_hands_torso(env):
    """(num_envs, links, 2) torso-frame xy of the Dex1 links at the arm start pose (cached, VB-POSE keep-outs). Called
    during a reset, after reset_scene_to_default wrote the start pose: the link poses are recomputed from it."""
    if getattr(env, "spec_hands_torso", None) is None:
        robot = env.scene["robot"]
        env.sim.physics_sim_view.update_articulations_kinematic()
        links = robot.root_physx_view.get_link_transforms()[:, :, :3].clone()  # (N, bodies, 3), world
        ids = [robot.body_names.index(link) for side in HAND_LINKS.values() for link in side]
        rel = links[:, ids].to(env.device) - env.scene.env_origins[:, None, :] - env.spec_torso_pos[:, None, :]
        c, s = torch.cos(env.spec_torso_yaw)[:, None], torch.sin(env.spec_torso_yaw)[:, None]
        xy = torch.stack((c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]), -1)
        env.spec_hands_torso = xy.cpu().numpy()
        print("[G1-D spec] VB-POSE hand keep-outs (torso xy, start pose): "
              + ", ".join(f"({x:.3f}, {y:.3f})" for x, y in env.spec_hands_torso[0]), flush=True)
    return env.spec_hands_torso


def _free_spot(rng, discs, r, zone, tries=4000):
    # the centre's range per axis (an object wider than the zone stays on its middle line)
    rx = (zone[0] + r, zone[1] - r) if zone[1] - zone[0] > 2 * r else ((zone[0] + zone[1]) / 2,) * 2
    ry = (zone[2] + r, zone[3] - r) if zone[3] - zone[2] > 2 * r else ((zone[2] + zone[3]) / 2,) * 2
    for _ in range(tries):
        xy = np.array([rng.uniform(*rx), rng.uniform(*ry)])
        if all(np.linalg.norm(xy - c) >= r + rc + MIN_SEP for c, rc in discs):
            return xy
    return None


def vb_pose(env, env_ids, poses, seed):
    """Re-place the main objects in `poses` (spec_poses output, changed in place); returns the layout per env."""
    spec = env.cfg.task_spec
    rng = np.random.default_rng([seed, env.spec_perturb_draws])
    draw = env.spec_perturb_draws
    env.spec_perturb_draws += 1
    hands = start_hands_torso(env)
    lid = spec["task_type"] in ("open_lid", "close_lid")
    mains = [o for o in spec["objects"] if o["role"] == "main"]
    layouts = {}
    for row, i in enumerate(env_ids.tolist()):
        discs = [(np.asarray(o["xy_torso"], float), max(float(o["bounding_box"][0]), float(o["bounding_box"][1])) / 2)
                 for o in spec["objects"] if o["role"] != "main"]
        for h in hands[i]:
            discs += [(h, HAND_KEEPOUT_R), (h + np.array([HAND_REACH, 0.0]), HAND_KEEPOUT_R)]
        c, s = math.cos(float(env.spec_torso_yaw[i])), math.sin(float(env.spec_torso_yaw[i]))
        placed = {}
        for o in mains:
            r = float(np.hypot(float(o["bounding_box"][0]), float(o["bounding_box"][1]))) / 2  # turning circle
            xy = _free_spot(rng, discs, r, ZONE_TORSO)
            yaw = float(rng.normal(0.0, 0.15 if lid else np.pi))
            free = xy is not None
            if not free:
                xy = np.asarray(o["xy_torso"], float)
            pos, quat = poses[o["name"]]
            pos[row, 0] = env.spec_torso_pos[i, 0] + c * xy[0] - s * xy[1]
            pos[row, 1] = env.spec_torso_pos[i, 1] + s * xy[0] + c * xy[1]
            q = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=quat.device)
            quat[row] = quat_mul(q[None], quat[row][None])[0]
            discs.append((xy, r))
            placed[o["name"]] = {"xy_torso": [round(float(v), 4) for v in xy], "yaw": round(yaw, 4), "free_spot": free}
        layouts[i] = {"kind": "VB-POSE", "seed": seed, "draw": draw, "objects": placed}
    return layouts


def place_objects(env, env_ids):
    """Reset: every spec object at its authored pose, at rest; with cfg.perturbation VB-POSE the main objects at a
    new draw. The layout of each env's current episode is kept in env.spec_layout (written to its record)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    perturbation = getattr(env.cfg, "perturbation", None) or {"kind": "Default", "seed": 0}
    if not hasattr(env, "spec_layout"):
        env.spec_layout = [None] * env.num_envs
        env.spec_perturb_draws = 0
    poses = spec_poses(env, env_ids)
    if perturbation["kind"] == "VB-POSE":
        layouts = vb_pose(env, env_ids, poses, int(perturbation["seed"]))
    elif perturbation["kind"] == "Default":
        layouts = {i: {"kind": "Default"} for i in env_ids.tolist()}
    else:
        raise ValueError(f"Unknown perturbation {perturbation['kind']!r}; known: {PERTURBATIONS}")
    for i, layout in layouts.items():
        env.spec_layout[i] = layout
    fixed = {o["name"] for o in env.cfg.task_spec["objects"] if o.get("fixed_base")}
    # objects fed later (key `feed_at_s`, see feed_objects) wait at their parking spot
    for o in env.cfg.task_spec["objects"]:
        if "feed_at_s" in o:
            poses[o["name"]] = _parked_pose(env, env_ids, o, poses[o["name"]][1])
    env.spec_fed = [set() for _ in range(env.num_envs)] if not hasattr(env, "spec_fed") else env.spec_fed
    for i in env_ids.tolist():
        env.spec_fed[i] = set()
    for name, (pos, quat) in poses.items():
        asset = env.scene[object_key(name)]
        asset.write_root_pose_to_sim(torch.cat((pos + env.scene.env_origins[env_ids], quat), -1), env_ids=env_ids)
        if name not in fixed:  # kinematic bodies take no velocity (PhysX error otherwise)
            asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)


reset_objects = place_objects


def _parked_pose(env, env_ids, o, quat):
    """Where an object with `feed_at_s` waits: at `park_xy_torso` (torso frame) on the floor, `park_z` [m] above it
    (default: half its height), out of the cameras' sight when the spot is behind the robot."""
    pos_t, yaw = env.spec_torso_pos[env_ids], env.spec_torso_yaw[env_ids]
    c, s = torch.cos(yaw), torch.sin(yaw)
    x, y = (float(v) for v in o["park_xy_torso"])
    z = float(o.get("park_z", float(o["bounding_box"][2]) / 2 + 0.005))
    return torch.stack((pos_t[:, 0] + c * x - s * y, pos_t[:, 1] + s * x + c * y, torch.full_like(yaw, z)), -1), quat


def feed_objects(env, env_ids=None):
    """Interval event (ours, not in chiral): an object with the spec key `feed_at_s` is moved from its parking spot to
    its spec pose once the episode is that many seconds old, at rest. This is how further boxes come onto the
    conveyor while the policy keeps running (the conveyor recordings' cell: an operator puts the next box on the belt)."""
    late = [o for o in env.cfg.task_spec["objects"] if "feed_at_s" in o]
    if not late or not hasattr(env, "spec_fed"):
        return
    t = env.episode_length_buf.float() * env.step_dt
    for i in range(env.num_envs):
        due = [o for o in late if o["name"] not in env.spec_fed[i] and float(t[i]) >= float(o["feed_at_s"])]
        if not due:
            continue
        ids = torch.tensor([i], device=env.device)
        poses = spec_poses(env, ids)
        for o in due:
            pos, quat = poses[o["name"]]
            asset = env.scene[object_key(o["name"])]
            asset.write_root_pose_to_sim(torch.cat((pos + env.scene.env_origins[ids], quat), -1), env_ids=ids)
            asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=ids)
            env.spec_fed[i].add(o["name"])
        print(f"[G1-D spec] fed at {float(t[i]):.1f} s: " + ", ".join(o["name"] for o in due), flush=True)


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
