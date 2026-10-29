"""Scene settings and conservative placement sampling, independent of Isaac Sim.

All lengths are metres. Scene coordinates are relative to the parked chassis:
+X forward, +Y left, +Z up. Bins are axis aligned and fixed to the table.
"""

import json
import math
from pathlib import Path
import random
import re

from robots.g1d_joints import ARM_SUFFIXES


DEFAULT_SCENE = Path(__file__).parent / "configs" / "scene.json"
CAMERAS = ("head", "left_wrist", "right_wrist")
JOINT_DYNAMICS_KEYS = ("static_friction_nm", "dynamic_friction_nm", "viscous_friction_nm_s_per_rad", "armature_kg_m2")


def _numbers(values, count, label, positive=False):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{label} must contain {count} numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or (positive and v <= 0) for v in values):
        raise ValueError(f"Invalid {label}: {values}")


def objects(settings):
    result = []
    for group, kind in (("cubes", "cube"), ("cylinders", "cylinder")):
        cfg = settings[group]
        size = cfg["size"] if kind == "cube" else [2 * cfg["radius"], 2 * cfg["radius"], cfg["height"]]
        radius = math.hypot(size[0], size[1]) / 2 if kind == "cube" else cfg["radius"]
        result.extend(dict(name=f"{kind}_{i:03d}", kind=kind, size=size, radius=radius,
                           mass=cfg["mass"], color=cfg["color"], target_bin=cfg["target_bin"])
                      for i in range(cfg["count"]))
    return result


def bin_bounds(box):
    x, y = box["center_xy"]
    sx, sy, _ = box["inner_size"]
    t = box["wall_thickness"]
    return (x - sx / 2 - t, x + sx / 2 + t, y - sy / 2 - t, y + sy / 2 + t)


def load_settings(path=DEFAULT_SCENE, camera_profile=None):
    """Validated scene settings; camera_profile overrides cameras.profile (see _cameras)."""
    cfg = json.loads(Path(path).read_text())
    cfg.setdefault("room", {"size": [6.0, 6.0, 3.0], "ambient_intensity": 100.0, "ceiling_intensity": 2500.0})
    _numbers(cfg["room"]["size"], 3, "room.size", True)
    _numbers([cfg["room"]["ambient_intensity"], cfg["room"]["ceiling_intensity"]], 2, "room light intensity", True)
    if any(v < 4 for v in cfg["room"]["size"][:2]) or cfg["room"]["size"][2] < 2.5:
        raise ValueError("The room must be at least 4 × 4 × 2.5 metres")
    table = cfg["table"]
    _numbers(table["center_xy"], 2, "table.center_xy")
    _numbers(table["size_xy"], 2, "table.size_xy", True)
    _numbers([table["height"], table["thickness"]], 2, "table height/thickness", True)
    if table["height"] <= table["thickness"]:
        raise ValueError("Table height must exceed its thickness")
    robot = cfg["robot"]
    robot.setdefault("controller_alignment", False)
    if type(robot["controller_alignment"]) is not bool:
        raise ValueError("robot.controller_alignment must be boolean")
    _numbers([robot["base_height"], robot["torso_pitch"], robot["torso_yaw"]], 3, "robot pose")
    _numbers(robot["lift_positions"], 2, "robot.lift_positions")
    robot.setdefault("arm_positions", [0.0] * 14)
    _numbers(robot["arm_positions"], 14, "robot.arm_positions")
    gripper = robot.setdefault("gripper_actuator", {})
    if not isinstance(gripper, dict):
        raise ValueError("robot.gripper_actuator must be an object")
    defaults = {
        "velocity_limit_m_s": 0.4,
        "stiffness_n_m": 1000.0,
        "damping_n_s_m": 20.0,
        "effort_limit_n": 20.0,
    }
    for key, value in defaults.items():
        gripper.setdefault(key, value)
    _numbers(
        [gripper[key] for key in defaults],
        len(defaults),
        "robot.gripper_actuator values",
        True,
    )
    gripper_control = robot.setdefault("gripper_control", {})
    if not isinstance(gripper_control, dict):
        raise ValueError("robot.gripper_control must be an object")
    gripper_control.setdefault("mode", "continuous")
    gripper_control.setdefault("close_threshold", 0.45)
    gripper_control.setdefault("open_threshold", 0.55)
    if gripper_control["mode"] not in ("continuous", "binary"):
        raise ValueError("robot.gripper_control.mode must be 'continuous' or 'binary'")
    close_threshold = gripper_control["close_threshold"]
    open_threshold = gripper_control["open_threshold"]
    _numbers([close_threshold, open_threshold], 2, "robot.gripper_control thresholds")
    if not 0 <= close_threshold <= open_threshold <= 1:
        raise ValueError("Gripper thresholds must satisfy 0 <= close <= open <= 1")
    if any(not 0 <= q <= 0.21 for q in robot["lift_positions"]):
        raise ValueError("Each lift position must be within [0, 0.21] metres")
    if not -0.04363 <= robot["torso_pitch"] <= 2.3562 or not -2.7053 <= robot["torso_yaw"] <= 2.7053:
        raise ValueError("Torso pose exceeds the URDF joint limits")
    collisions = robot.setdefault("collisions", {})
    if not isinstance(collisions, dict):
        raise ValueError("robot.collisions must be an object")
    collisions.setdefault("self_collisions", True)
    collisions.setdefault("continuous_collision_detection", True)
    collisions.setdefault("contact_offset_m", 0.002)
    collisions.setdefault("rest_offset_m", 0.0)
    for key in ("self_collisions", "continuous_collision_detection"):
        if type(collisions[key]) is not bool:
            raise ValueError(f"robot.collisions.{key} must be boolean")
    _numbers(
        [collisions["contact_offset_m"], collisions["rest_offset_m"]],
        2,
        "robot collision offsets",
    )
    if not 0 <= collisions["rest_offset_m"] <= collisions["contact_offset_m"]:
        raise ValueError("Robot collision offsets must satisfy 0 <= rest_offset_m <= contact_offset_m")
    payloads = robot.setdefault("payloads", [])
    if not isinstance(payloads, list):
        raise ValueError("robot.payloads must be a list")
    for payload in payloads:
        if not isinstance(payload, dict) or not isinstance(payload.get("body"), str):
            raise ValueError("Each robot payload needs a body name, mass_kg and com_m")
        _numbers([payload.get("mass_kg")], 1, "payload mass_kg")
        _numbers(payload.get("com_m"), 3, "payload com_m")
        if payload["mass_kg"] < 0:
            raise ValueError("Payload masses must be nonnegative")
    _camera_mount(robot.setdefault("camera_mount", []), "robot.camera_mount")
    dynamics = robot.setdefault("joint_dynamics", {})
    if not isinstance(dynamics, dict):
        raise ValueError("robot.joint_dynamics must be an object")
    for key, values in dynamics.items():
        if key not in JOINT_DYNAMICS_KEYS:
            raise ValueError(f"Unknown robot.joint_dynamics key {key!r}; expected {JOINT_DYNAMICS_KEYS}")
        if not isinstance(values, dict) or set(values) != set(ARM_SUFFIXES):
            raise ValueError(f"robot.joint_dynamics.{key} needs one value per arm joint: {ARM_SUFFIXES}")
        _numbers([values[name] for name in ARM_SUFFIXES], 7, f"robot.joint_dynamics.{key}")
        if any(values[name] < 0 for name in ARM_SUFFIXES):
            raise ValueError(f"robot.joint_dynamics.{key} values must be nonnegative")
    boxes = cfg["bins"]
    names = [box["name"] for box in boxes]
    if len(boxes) != 2 or len(set(names)) != 2:
        raise ValueError("Specify two bins with unique names")
    tx, ty = table["center_xy"]
    sx, sy = table["size_xy"]
    for box in boxes:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", box["name"]):
            raise ValueError("Bin names must use lower-case letters, digits and underscores")
        _numbers(box["center_xy"], 2, "bin center")
        _numbers(box["inner_size"], 3, "bin inner_size", True)
        _numbers([box["wall_thickness"]], 1, "bin wall_thickness", True)
        _numbers(box["color"], 3, "bin color")
        x0, x1, y0, y1 = bin_bounds(box)
        if x0 < tx - sx / 2 or x1 > tx + sx / 2 or y0 < ty - sy / 2 or y1 > ty + sy / 2:
            raise ValueError(f"{box['name']} extends beyond the tabletop")
    a, b = map(bin_bounds, boxes)
    if a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]:
        raise ValueError("The bins overlap")
    for group in ("cubes", "cylinders"):
        obj = cfg[group]
        if type(obj["count"]) is not int or not 0 <= obj["count"] <= 100:
            raise ValueError(f"{group}.count must be an integer from 0 to 100")
        _numbers(obj["size"] if group == "cubes" else [obj["radius"], obj["height"]],
                 3 if group == "cubes" else 2, f"{group} dimensions", True)
        _numbers([obj["mass"]], 1, f"{group}.mass", True)
        _numbers(obj["color"], 3, f"{group}.color")
        if obj["target_bin"] not in names:
            raise ValueError(f"Unknown target bin for {group}")
    rand = cfg["randomization"]
    if type(rand["enabled"]) is not bool or type(rand["fixed_seed"]) is not int:
        raise ValueError("randomization.enabled must be boolean and fixed_seed must be an integer")
    for key in ("x_range", "y_range", "yaw_range"):
        _numbers(rand[key], 2, key)
        if rand[key][0] > rand[key][1]:
            raise ValueError(f"{key} is reversed")
    _numbers([rand["clearance"]], 1, "clearance")
    if rand["clearance"] < 0:
        raise ValueError("clearance must be nonnegative")
    _cameras(cfg, camera_profile)
    _calibration_targets(cfg.setdefault("calibration_targets", {}))
    # Fail before starting simulation if the requested layout cannot be packed.
    sample_layout(cfg, random.Random(rand["fixed_seed"]))
    return cfg


def _camera_mount(parts, label):
    if not isinstance(parts, list):
        raise ValueError(f"{label} must be a list of parts")
    for part in parts:
        if not isinstance(part, dict) or not re.fullmatch(r"[a-z][a-z0-9_]*", str(part.get("name"))):
            raise ValueError(f"Each {label} part needs a lower-case name, size, position and color")
        _numbers(part.get("size"), 3, f"{label} size", True)
        _numbers(part.get("position"), 3, f"{label} position")
        _numbers(part.setdefault("rpy", [0.0, 0.0, 0.0]), 3, f"{label} rpy")
        _numbers(part.get("color"), 3, f"{label} color")
    if len({part["name"] for part in parts}) != len(parts):
        raise ValueError(f"{label} part names must be unique")


def _unit(vector, label):
    _numbers(vector, 3, label)
    norm = math.sqrt(sum(v * v for v in vector))
    if norm < 1e-6:
        raise ValueError(f"{label} must not be zero")
    return [v / norm for v in vector]


def _cameras(cfg, override):
    """Select the active camera profile and validate it.

    cameras = {"width", "height", "profile": name, "profiles": {name: {"head", "left_wrist", "right_wrist",
    optional "camera_mount" (replaces robot.camera_mount), optional "source"}}}. A scene without "profiles"
    (head/left_wrist/right_wrist directly under cameras) is a single profile named "default". The active
    profile's cameras are copied to cameras.head/left_wrist/right_wrist.

    Camera: position and forward (parent link frame), optional up (roll; default: the parent's +Z), and
    either a pinhole lens (focal_length, horizontal_aperture) or calibrated OpenCV intrinsics (camera_lens.py).
    """
    cameras = cfg["cameras"]
    for key in ("width", "height"):
        if type(cameras[key]) is not int or cameras[key] <= 0:
            raise ValueError(f"cameras.{key} must be a positive integer")
    profiles = cameras.get("profiles")
    if profiles is None:
        profiles = {"default": {name: cameras[name] for name in CAMERAS}}
        cameras.setdefault("profile", "default")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("cameras.profiles must map profile names to camera sets")
    name = override or cameras.get("profile")
    if name not in profiles:
        raise ValueError(f"Unknown camera profile {name!r}; available: {sorted(profiles)}")
    cameras["profile"] = name
    profile = profiles[name]
    for camera_name in CAMERAS:
        camera = profile[camera_name]
        _numbers(camera["position"], 3, f"{camera_name} camera position")
        forward = _unit(camera["forward"], f"{camera_name} camera forward")
        up = _unit(camera.get("up", [0.0, 0.0, 1.0]), f"{camera_name} camera up")
        if abs(sum(f * u for f, u in zip(forward, up))) > 0.99:
            raise ValueError(f"{camera_name} camera forward must not be parallel to its up vector")
        intrinsics = camera.get("intrinsics")
        if intrinsics is None:
            _numbers([camera["focal_length"], camera["horizontal_aperture"]], 2, "camera lens", True)
        else:
            if intrinsics.get("model") not in ("pinhole", "fisheye"):
                raise ValueError(f"{camera_name} intrinsics.model must be 'pinhole' or 'fisheye'")
            K = intrinsics.get("K")
            if not isinstance(K, list) or len(K) != 3:
                raise ValueError(f"{camera_name} intrinsics.K must be 3x3")
            for row in K:
                _numbers(row, 3, f"{camera_name} intrinsics.K row")
            D = intrinsics.get("D")
            count = len(D) if isinstance(D, list) else 0
            if (intrinsics["model"] == "fisheye" and count != 4) or (
                    intrinsics["model"] == "pinhole" and count not in (4, 5, 8, 12, 14)):
                raise ValueError(f"{camera_name} intrinsics.D has the wrong number of coefficients")
            _numbers(D, count, f"{camera_name} intrinsics.D")
            if list(intrinsics.get("size", [])) != [cameras["width"], cameras["height"]]:
                raise ValueError(f"{camera_name} intrinsics.size must equal cameras width/height")
            _numbers([intrinsics.setdefault("render_scale", 1.0)], 1, f"{camera_name} render_scale", True)
        cameras[camera_name] = camera
    if "camera_mount" in profile:
        _camera_mount(profile["camera_mount"], f"cameras.profiles.{name}.camera_mount")
        cfg["robot"]["camera_mount"] = profile["camera_mount"]


def _pose(entry, label):
    if not isinstance(entry, dict) or not re.fullmatch(r"[a-z][a-z0-9_]*", str(entry.get("parent"))):
        raise ValueError(f"{label} needs a parent link name, position and quat_wxyz")
    _numbers(entry.get("position"), 3, f"{label} position")
    _numbers(entry.get("quat_wxyz"), 4, f"{label} quat_wxyz")
    if abs(math.sqrt(sum(v * v for v in entry["quat_wxyz"])) - 1) > 1e-3:
        raise ValueError(f"{label} quat_wxyz must be a unit quaternion")


def _calibration_targets(targets):
    """Optional ChArUco board and ArUco markers (visual only) for comparing sim and real camera images.

    {"dictionary": "DICT_5X5_100", "board": {pose, "squares": [cols, rows], "square_m", "marker_ratio"},
     "markers": [{pose, "id", "size_m"}]}; pose = "parent" link, "position", "quat_wxyz". Frames as in OpenCV:
    board origin at its top-left corner, x right, y down (seen from the front); marker origin at its centre,
    x right, y up, z towards the viewer. Sizes are the black squares' sides.
    """
    if not isinstance(targets, dict):
        raise ValueError("calibration_targets must be an object")
    if not targets:
        return
    if not re.fullmatch(r"DICT_[0-9A-Z_]+", str(targets.get("dictionary"))):
        raise ValueError("calibration_targets.dictionary must name an OpenCV ArUco dictionary, e.g. DICT_5X5_100")
    board = targets.get("board")
    if board is not None:
        _pose(board, "calibration_targets.board")
        squares = board.get("squares")
        if not isinstance(squares, list) or len(squares) != 2 or any(type(v) is not int or v < 2 for v in squares):
            raise ValueError("calibration_targets.board.squares must be two integers >= 2")
        _numbers([board.get("square_m"), board.get("marker_ratio")], 2, "calibration_targets.board size", True)
        if board["marker_ratio"] >= 1:
            raise ValueError("calibration_targets.board.marker_ratio must be below 1")
    markers = targets.setdefault("markers", [])
    if not isinstance(markers, list):
        raise ValueError("calibration_targets.markers must be a list")
    for marker in markers:
        _pose(marker, "calibration_targets marker")
        if type(marker.get("id")) is not int or marker["id"] < 0:
            raise ValueError("calibration_targets marker id must be a nonnegative integer")
        _numbers([marker.get("size_m")], 1, "calibration_targets marker size_m", True)


def sample_layout(cfg, rng):
    """Return non-overlapping poses; reject layouts that cannot fit instead of piling objects up."""
    if not cfg["randomization"]["enabled"]:
        rng = random.Random(cfg["randomization"]["fixed_seed"])
    rand, table = cfg["randomization"], cfg["table"]
    tx, ty = table["center_xy"]
    sx, sy = table["size_xy"]
    clearance = rand["clearance"]
    placed = []
    result = {}
    # Place larger footprints first, with bounded full-layout retries.
    ordered = sorted(objects(cfg), key=lambda obj: obj["radius"], reverse=True)
    for _ in range(20):
        placed.clear()
        result.clear()
        for obj in ordered:
            r = obj["radius"]
            xmin = max(rand["x_range"][0], tx - sx / 2 + r + clearance)
            xmax = min(rand["x_range"][1], tx + sx / 2 - r - clearance)
            ymin = max(rand["y_range"][0], ty - sy / 2 + r + clearance)
            ymax = min(rand["y_range"][1], ty + sy / 2 - r - clearance)
            if xmin > xmax or ymin > ymax:
                raise ValueError(f"{obj['name']} cannot fit in the spawn region")
            for _ in range(500):
                x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
                if any(math.hypot(x - px, y - py) < r + pr + clearance for px, py, pr in placed):
                    continue
                blocked = False
                for box in cfg["bins"]:
                    x0, x1, y0, y1 = bin_bounds(box)
                    dx = max(x0 - x, 0, x - x1)
                    dy = max(y0 - y, 0, y - y1)
                    blocked |= math.hypot(dx, dy) < r + clearance
                if blocked:
                    continue
                yaw = rng.uniform(*rand["yaw_range"])
                result[obj["name"]] = (x, y, table["height"] + obj["size"][2] / 2 + 0.002, yaw)
                placed.append((x, y, r))
                break
            else:
                break
        else:
            return dict(result)
    raise ValueError("Cannot fit the requested objects: reduce counts/sizes or enlarge the spawn region")
