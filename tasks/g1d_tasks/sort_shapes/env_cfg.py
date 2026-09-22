"""Configurable tabletop sorting environment for a parked G1-D."""

import math
import random

import torch
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_matrix

from robots.g1d import robot_config
from . import mdp
from .evaluation import EvaluationRecorderManagerCfg, StageEvaluator, load_evaluation_settings
from .layout import load_settings, objects, sample_layout
from .visuals import apply_robot_materials


def camera_rotation(forward):
    """Quaternion in Isaac's world camera convention: local +X forward, +Z up."""
    x = torch.tensor(forward, dtype=torch.float64)
    x /= x.norm()
    y = torch.linalg.cross(torch.tensor([0.0, 0.0, 1.0], dtype=x.dtype), x)
    y /= y.norm()
    z = torch.linalg.cross(x, y)
    return tuple(quat_from_matrix(torch.stack((x, y, z), dim=-1)).tolist())


def camera_config(path, config, resolution):
    return CameraCfg(
        prim_path=path, update_period=0.0, width=resolution[0], height=resolution[1], data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=config["focal_length"], horizontal_aperture=config["horizontal_aperture"],
            clipping_range=(0.01, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(config["position"]), rot=camera_rotation(config["forward"]), convention="world",
        ),
    )


def build_scene(settings, evaluation):
    room = settings["room"]
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=max(room["size"][:2]) + 1, replicate_physics=True)
    material = sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8, restitution=0.0)

    def box(name, size, position, color):
        setattr(scene, name, AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=tuple(size), collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=0.002, rest_offset=0.0,
                ),
                physics_material=material,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), roughness=0.7),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(position)),
        ))

    rx, ry, rz = room["size"]
    box("floor", (rx, ry, 0.10), (0.5, 0, -0.05), (0.19, 0.21, 0.23))
    box("ceiling", (rx, ry, 0.10), (0.5, 0, rz + 0.05), (0.65, 0.65, 0.63))
    for side in (-1, 1):
        suffix = "minus" if side == -1 else "plus"
        box(f"wall_x_{suffix}", (0.12, ry, rz), (0.5 + side * rx / 2, 0, rz / 2), (0.54, 0.56, 0.55))
        box(f"wall_y_{suffix}", (rx, 0.12, rz), (0.5, side * ry / 2, rz / 2), (0.47, 0.50, 0.53))
        box(f"wall_trim_{suffix}", (rx, 0.03, 0.12), (0.5, side * (ry / 2 - 0.07), 0.06), (0.08, 0.10, 0.12))
    # A storage rack and workbench provide an indoor manipulation-lab background.
    back = 0.5 + rx / 2 - 0.42
    for y in (-0.85, 0.85):
        box(f"shelf_post_{'l' if y > 0 else 'r'}", (0.05, 0.05, 1.9), (back, y, 0.95), (0.09, 0.12, 0.15))
    for level in range(4):
        z = 0.15 + level * 0.48
        box(f"shelf_{level}", (0.55, 1.85, 0.035), (back, 0, z), (0.17, 0.20, 0.22))
        for column in (-1, 1):
            box(f"stored_box_{level}_{'l' if column > 0 else 'r'}", (0.32, 0.46, 0.26),
                (back, column * 0.48, z + 0.15), (0.33 + level * 0.025, 0.24, 0.14))
    cabinet_y = -ry / 2 + 0.35
    box("cabinet", (1.8, 0.55, 0.80), (0.1, cabinet_y, 0.40), (0.24, 0.29, 0.32))
    box("cabinet_top", (1.86, 0.6, 0.04), (0.1, cabinet_y, 0.82), (0.40, 0.42, 0.43))
    for i in range(3):
        box(f"cabinet_handle_{i}", (0.18, 0.025, 0.025), (-0.5 + i * 0.6, cabinet_y + 0.285, 0.65), (0.05, 0.06, 0.07))
    scene.robot = robot_config(settings["robot"])
    table = settings["table"]
    tx, ty = table["center_xy"]
    sx, sy = table["size_xy"]
    height, thickness = table["height"], table["thickness"]
    box("table_top", (sx, sy, thickness), (tx, ty, height - thickness / 2), (0.68, 0.64, 0.54))
    for i, (x, y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        box(f"table_leg_{i}", (0.035, 0.035, height - thickness),
            (tx + x * (sx / 2 - 0.035), ty + y * (sy / 2 - 0.035), (height - thickness) / 2),
            (0.18, 0.18, 0.19))
    for bin_cfg in settings["bins"]:
        name, color = bin_cfg["name"], bin_cfg["color"]
        x, y = bin_cfg["center_xy"]
        bx, by, bz = bin_cfg["inner_size"]
        t = bin_cfg["wall_thickness"]
        box(f"bin_{name}_floor", (bx + 2 * t, by + 2 * t, t), (x, y, height + t / 2), color)
        for sign, side in ((-1, "minus"), (1, "plus")):
            box(f"bin_{name}_x_{side}", (t, by + 2 * t, bz),
                (x + sign * (bx + t) / 2, y, height + t + bz / 2), color)
            box(f"bin_{name}_y_{side}", (bx, t, bz),
                (x, y + sign * (by + t) / 2, height + t + bz / 2), color)
    seed = settings["randomization"]["fixed_seed"]
    if seed == -1:
        seed = None
    poses = sample_layout(settings, random.Random(seed))
    items = objects(settings)
    for obj in items:
        shape_args = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=obj["mass"]),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(obj["color"]), roughness=0.5),
        )
        if obj["kind"] == "cube":
            shape = sim_utils.CuboidCfg(size=tuple(obj["size"]), **shape_args)
        else:
            shape = sim_utils.CylinderCfg(radius=obj["radius"], height=obj["size"][2], **shape_args)
        x, y, z, yaw = poses[obj["name"]]
        setattr(scene, obj["name"], RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{obj['name']}", spawn=shape,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, z), rot=(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))),
        ))

    if evaluation["enabled"] and items:
        object_paths = [f"{{ENV_REGEX_NS}}/{obj['name']}" for obj in items]
        for side in ("left", "right"):
            for finger in (1, 2):
                setattr(scene, f"{side}_finger_{finger}_contact", ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_dex1_finger_link_{finger}",
                    update_period=0.0,
                    history_length=0,
                    filter_prim_paths_expr=object_paths,
                    debug_vis=False,
                ))

    cameras = settings["cameras"]
    resolution = cameras["width"], cameras["height"]
    scene.front_camera = camera_config("{ENV_REGEX_NS}/Robot/head_link/front_camera", cameras["head"], resolution)
    for side in ("left", "right"):
        setattr(scene, f"{side}_wrist_camera", camera_config(
            f"{{ENV_REGEX_NS}}/Robot/{side}_dex1_base_link/{side}_wrist_camera", cameras[f"{side}_wrist"], resolution,
        ))
    scene.world_camera = camera_config("{ENV_REGEX_NS}/world_camera", dict(
        position=[1.8, 1.6, 1.8], forward=[-1.4, -1.6, -0.9], focal_length=12.0, horizontal_aperture=27.0,
    ), resolution)
    scene.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=room["ambient_intensity"]))
    for i, (x, y, scale) in enumerate(((0.7, 0.0, 1.0), (-1.0, -1.1, 0.6))):
        box(f"light_panel_{i}", (0.95, 0.65, 0.025), (x, y, rz - 0.04), (0.85, 0.85, 0.82))
        setattr(scene, f"ceiling_light_{i}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/ceiling_light_{i}",
            spawn=sim_utils.DiskLightCfg(radius=0.5, intensity=room["ceiling_intensity"] * scale,
                                         color=(1.0, 0.97, 0.93)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, rz - 0.08)),
        ))
    return scene


@configclass
class ActionsCfg:
    joint_targets = mdp.G1DSortActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=mdp.joint_state)
        head = ObservationTermCfg(func=mdp.camera_rgb, params={"name": "front_camera"})
        left_wrist = ObservationTermCfg(func=mdp.camera_rgb, params={"name": "left_wrist_camera"})
        right_wrist = ObservationTermCfg(func=mdp.camera_rgb, params={"name": "right_wrist_camera"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SceneCfg(ObservationGroupCfg):
        object_poses = ObservationTermCfg(func=mdp.object_poses)
        sorted_fraction = ObservationTermCfg(func=mdp.sorted_fraction)
        evaluation_stages = ObservationTermCfg(func=mdp.evaluation_stages)
        evaluation_summary = ObservationTermCfg(func=mdp.evaluation_summary)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy = PolicyCfg()
    scene_state = SceneCfg()


@configclass
class EventsCfg:
    robot_materials = EventTermCfg(func=apply_robot_materials, mode="startup")
    reset_robot_and_objects = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")
    randomize_objects = EventTermCfg(func=mdp.reset_objects, mode="reset")


@configclass
class RewardsCfg:
    sorted_fraction = RewardTermCfg(func=mdp.sorted_fraction, weight=1.0)


@configclass
class TerminationsCfg:
    success = TerminationTermCfg(func=StageEvaluator)
    time_out = TerminationTermCfg(func=base_mdp.time_out, time_out=True)


@configclass
class G1DSortEnvCfg(ManagerBasedRLEnvCfg):
    sorting: dict = load_settings()
    evaluation: dict = load_evaluation_settings()
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    recorders = EvaluationRecorderManagerCfg()

    def __post_init__(self):
        self.scene = build_scene(self.sorting, self.evaluation)

        env_seed = self.sorting["randomization"]["fixed_seed"]
        self.seed = None if env_seed == -1 else env_seed

        self.decimation = 4
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_ccd = self.sorting["robot"]["collisions"]["continuous_collision_detection"]
        self.sim.render.enable_direct_lighting = True
        self.sim.render.enable_shadows = True
        self.sim.render.enable_global_illumination = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.antialiasing_mode = "DLAA"
        self.sim.physx.bounce_threshold_velocity = 0.05
        self.episode_length_s = self.evaluation["episode_timeout_s"]
        self.num_rerenders_on_reset = 1
        self.viewer.eye = (1.8, 1.6, 1.8)
        self.viewer.lookat = (0.4, 0.0, 0.9)
