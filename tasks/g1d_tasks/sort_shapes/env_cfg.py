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
from isaaclab.utils.math import quat_from_euler_xyz, quat_from_matrix

from robots.g1d import robot_config
from . import mdp
from .calibration_targets import target_assets
from .camera_lens import isaac_intrinsic_matrix, virtual_camera
from .cube_spawner import ChamferedCubeCfg, RoundCylinderCfg
from .evaluation import EvaluationRecorderManagerCfg, StageEvaluator, load_evaluation_settings
from .layout import load_settings, objects, sample_layout
from .visuals import apply_robot_materials


def camera_rotation(forward, up=(0.0, 0.0, 1.0)):
    """Quaternion in Isaac's world camera convention: local +X forward, +Z up (roll from `up`)."""
    x = torch.tensor(forward, dtype=torch.float64)
    x /= x.norm()
    y = torch.linalg.cross(torch.tensor(up, dtype=x.dtype), x)
    y /= y.norm()
    z = torch.linalg.cross(x, y)
    return tuple(quat_from_matrix(torch.stack((x, y, z), dim=-1)).tolist())


def camera_config(path, config, resolution):
    """Pinhole camera from focal_length/horizontal_aperture, or, with calibrated intrinsics, the centred virtual
    pinhole camera whose renders mdp.camera_rgb warps into the calibrated lens (camera_lens.py)."""
    if "intrinsics" in config:
        focal, width, height = virtual_camera(config["intrinsics"])
        spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            isaac_intrinsic_matrix(focal, width, height), width, height, clipping_range=(0.01, 20.0), focal_length=1.0,
        )
        resolution = width, height
    else:
        spawn = sim_utils.PinholeCameraCfg(
            focal_length=config["focal_length"], horizontal_aperture=config["horizontal_aperture"],
            clipping_range=(0.01, 20.0),
        )
    return CameraCfg(
        prim_path=path, update_period=0.0, width=resolution[0], height=resolution[1], data_types=["rgb"],
        spawn=spawn,
        offset=CameraCfg.OffsetCfg(
            pos=tuple(config["position"]), rot=camera_rotation(config["forward"], config.get("up", (0.0, 0.0, 1.0))),
            convention="world",
        ),
    )


def scene_material():
    return sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8, restitution=0.0)


def add_box(scene, name, size, position, color):
    """A static collision box (room, furniture, table, bins)."""
    setattr(scene, name, AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=tuple(size), collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0,
            ),
            physics_material=scene_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color), roughness=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(position)),
    ))


def build_workspace(settings):
    """Room, robot (with wrist camera mounts and calibration targets) and table: everything but the task's
    objects and the cameras/lights (add_cameras_and_lights). Shared with the spec tasks (g1d_tasks/spec_tasks)."""
    room = settings["room"]
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=max(room["size"][:2]) + 1, replicate_physics=True)

    def box(name, size, position, color):
        add_box(scene, name, size, position, color)

    rx, ry, rz = room["size"]
    # room.floor_color / wall_color (optional): the look of another room; room.furniture false: no rack and cabinet;
    # room.props: further static boxes [{name, size, position, color}] (e.g. the surroundings of the conveyor cell)
    wall_colors = [tuple(room["wall_color"])] * 2 if "wall_color" in room else [(0.54, 0.56, 0.55), (0.47, 0.50, 0.53)]
    box("floor", (rx, ry, 0.10), (0.5, 0, -0.05), tuple(room.get("floor_color", (0.19, 0.21, 0.23))))
    box("ceiling", (rx, ry, 0.10), (0.5, 0, rz + 0.05), (0.65, 0.65, 0.63))
    for side in (-1, 1):
        suffix = "minus" if side == -1 else "plus"
        box(f"wall_x_{suffix}", (0.12, ry, rz), (0.5 + side * rx / 2, 0, rz / 2), wall_colors[0])
        box(f"wall_y_{suffix}", (rx, 0.12, rz), (0.5, side * ry / 2, rz / 2), wall_colors[1])
        box(f"wall_trim_{suffix}", (rx, 0.03, 0.12), (0.5, side * (ry / 2 - 0.07), 0.06), (0.08, 0.10, 0.12))
    # A storage rack and workbench provide an indoor manipulation-lab background.
    back = 0.5 + rx / 2 - 0.42
    for prop in room.get("props", []):
        box(f"prop_{prop['name']}", prop["size"], prop["position"], prop["color"])
    for y in (-0.85, 0.85) if room.get("furniture", True) else ():
        box(f"shelf_post_{'l' if y > 0 else 'r'}", (0.05, 0.05, 1.9), (back, y, 0.95), (0.09, 0.12, 0.15))
    for level in range(4) if room.get("furniture", True) else ():
        z = 0.15 + level * 0.48
        box(f"shelf_{level}", (0.55, 1.85, 0.035), (back, 0, z), (0.17, 0.20, 0.22))
        for column in (-1, 1):
            box(f"stored_box_{level}_{'l' if column > 0 else 'r'}", (0.32, 0.46, 0.26),
                (back, column * 0.48, z + 0.15), (0.33 + level * 0.025, 0.24, 0.14))
    cabinet_y = -ry / 2 + 0.35
    if room.get("furniture", True):
        box("cabinet", (1.8, 0.55, 0.80), (0.1, cabinet_y, 0.40), (0.24, 0.29, 0.32))
        box("cabinet_top", (1.86, 0.6, 0.04), (0.1, cabinet_y, 0.82), (0.40, 0.42, 0.43))
    for i in range(3) if room.get("furniture", True) else ():
        box(f"cabinet_handle_{i}", (0.18, 0.025, 0.025), (-0.5 + i * 0.6, cabinet_y + 0.285, 0.65), (0.05, 0.06, 0.07))
    scene.robot = robot_config(settings["robot"])
    # Wrist camera mounts (missing in the imported URDF): boxes fixed to each Dex1 base link, pose given for
    # the left gripper (position, roll/pitch/yaw) and mirrored for the right. Visual and collision only; their
    # mass is not added here (the link keeps its explicit mass), the measured load belongs in robot.payloads.
    collisions = settings["robot"]["collisions"]
    for side, mirror in (("left", 1.0), ("right", -1.0)):
        for part in settings["robot"]["camera_mount"]:
            x, y, z = part["position"]
            roll, pitch, yaw = part["rpy"]
            rotation = quat_from_euler_xyz(*(torch.tensor([v]) for v in (mirror * roll, pitch, mirror * yaw)))[0]
            setattr(scene, f"{side}_camera_mount_{part['name']}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_dex1_base_link/camera_mount_{part['name']}",
                spawn=sim_utils.CuboidCfg(
                    size=tuple(part["size"]),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True, contact_offset=collisions["contact_offset_m"],
                        rest_offset=collisions["rest_offset_m"],
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(part["color"]), roughness=0.6),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x, mirror * y, z), rot=tuple(rotation.tolist())),
            ))
    # ChArUco board / ArUco markers at their calibrated poses (visual only), for sim-vs-real image comparison.
    for name, parent, usd_path, position, quat in target_assets(settings["calibration_targets"]):
        setattr(scene, name, AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{parent}/{name}",
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(position), rot=tuple(quat)),
        ))
    table = settings["table"]
    tx, ty = table["center_xy"]
    sx, sy = table["size_xy"]
    height, thickness = table["height"], table["thickness"]
    # table.color (optional): the real evaluation table is dark; the default is the original light top
    top_color = tuple(table.get("color", (0.68, 0.64, 0.54)))
    if "belt_velocity_xy" in table:
        # a conveyor instead of a table: the top is a kinematic body whose surface moves at table.belt_velocity_xy
        # [m/s], between two fixed side rails (table.rail_width, default 0.04 m; outside size_xy, tops flush) on a body
        from .belt import BeltCfg
        vx, vy = table["belt_velocity_xy"]
        scene.table_top = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/table_top",
            spawn=BeltCfg(
                size=(sx, sy, thickness), surface_velocity=(float(vx), float(vy), 0.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
                physics_material=scene_material(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=top_color, roughness=0.6),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(tx, ty, height - thickness / 2)),
        )
        rail, frame = float(table.get("rail_width", 0.04)), tuple(table.get("frame_color", (0.62, 0.63, 0.64)))
        along_y = abs(vy) >= abs(vx)
        for side in (-1, 1):
            offset = side * ((sx if along_y else sy) / 2 + rail / 2)
            box(f"belt_rail_{'minus' if side < 0 else 'plus'}", (rail, sy, 0.10) if along_y else (sx, rail, 0.10),
                (tx + offset, ty, height - 0.05) if along_y else (tx, ty + offset, height - 0.05), frame)
        box("belt_body", (sx, sy, 0.10), (tx, ty, height - thickness - 0.05), (0.25, 0.26, 0.27))
    else:
        box("table_top", (sx, sy, thickness), (tx, ty, height - thickness / 2), top_color)
    for i, (x, y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        box(f"table_leg_{i}", (0.035, 0.035, height - thickness),
            (tx + x * (sx / 2 - 0.035), ty + y * (sy / 2 - 0.035), (height - thickness) / 2),
            (0.18, 0.18, 0.19))
    return scene


def add_cameras_and_lights(scene, settings):
    """Head and wrist cameras (manual or calibrated profile), the world camera, and the room lights."""
    room = settings["room"]
    rz = room["size"][2]
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
        add_box(scene, f"light_panel_{i}", (0.95, 0.65, 0.025), (x, y, rz - 0.04), (0.85, 0.85, 0.82))
        setattr(scene, f"ceiling_light_{i}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/ceiling_light_{i}",
            spawn=sim_utils.DiskLightCfg(radius=0.5, intensity=room["ceiling_intensity"] * scale,
                                         color=(1.0, 0.97, 0.93)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, rz - 0.08)),
        ))


def build_scene(settings, evaluation):
    scene = build_workspace(settings)
    material = scene_material()
    height = settings["table"]["height"]

    def box(name, size, position, color):
        add_box(scene, name, size, position, color)

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
        if obj["kind"] == "cube" and settings["cubes"].get("face_flat"):
            # the real cubes: chamfered edges and corners, a hole in each face (scene keys face_flat, hole_diameter)
            shape = ChamferedCubeCfg(size=obj["size"][0], face_flat=settings["cubes"]["face_flat"],
                                     hole_diameter=settings["cubes"].get("hole_diameter", 0.0), **shape_args)
        elif obj["kind"] == "cube":
            shape = sim_utils.CuboidCfg(size=tuple(obj["size"]), **shape_args)
        elif settings["cylinders"].get("mesh_sections"):
            # a round cylinder mesh (scene key cylinders.mesh_sections); the USD cylinder renders with about ten sides
            shape = RoundCylinderCfg(radius=obj["radius"], height=obj["size"][2],
                                     sections=int(settings["cylinders"]["mesh_sections"]), **shape_args)
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

    add_cameras_and_lights(scene, settings)
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
    robot_payloads = EventTermCfg(func=mdp.apply_robot_payloads, mode="startup")
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
        apply_sim_settings(self)


def apply_sim_settings(cfg):
    """30 Hz control over 120 Hz physics, rendering and viewer settings (shared with the spec tasks)."""
    cfg.decimation = 4
    cfg.sim.dt = 1 / 120
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physx.enable_ccd = cfg.sorting["robot"]["collisions"]["continuous_collision_detection"]
    cfg.sim.render.enable_direct_lighting = True
    cfg.sim.render.enable_shadows = True
    cfg.sim.render.enable_global_illumination = True
    cfg.sim.render.enable_ambient_occlusion = True
    cfg.sim.render.enable_reflections = True
    cfg.sim.render.antialiasing_mode = "DLAA"
    cfg.sim.physx.bounce_threshold_velocity = 0.05
    cfg.episode_length_s = cfg.evaluation["episode_timeout_s"]
    cfg.num_rerenders_on_reset = 1
    cfg.viewer.eye = (1.8, 1.6, 1.8)
    cfg.viewer.lookat = (0.4, 0.0, 0.9)
