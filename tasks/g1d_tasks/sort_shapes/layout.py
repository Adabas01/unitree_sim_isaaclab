"""Scene settings and conservative placement sampling, independent of Isaac Sim.

All lengths are metres. Scene coordinates are relative to the parked chassis:
+X forward, +Y left, +Z up. Bins are axis aligned and fixed to the table.
"""

import json
import math
from pathlib import Path
import random
import re


DEFAULT_SCENE = Path(__file__).parent / "configs" / "scene.json"


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


def load_settings(path=DEFAULT_SCENE):
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
    if any(not 0 <= q <= 0.21 for q in robot["lift_positions"]):
        raise ValueError("Each lift position must be within [0, 0.21] metres")
    if not -0.04363 <= robot["torso_pitch"] <= 2.3562 or not -2.7053 <= robot["torso_yaw"] <= 2.7053:
        raise ValueError("Torso pose exceeds the URDF joint limits")
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
    cameras = cfg["cameras"]
    for key in ("width", "height"):
        if type(cameras[key]) is not int or cameras[key] <= 0:
            raise ValueError(f"cameras.{key} must be a positive integer")
    for name in ("head", "left_wrist", "right_wrist"):
        camera = cameras[name]
        _numbers(camera["position"], 3, f"{name} camera position")
        _numbers(camera["forward"], 3, f"{name} camera forward")
        if math.hypot(*camera["forward"][:2]) < 1e-6:
            raise ValueError("Camera forward must not be parallel to local Z")
        _numbers([camera["focal_length"], camera["horizontal_aperture"]], 2, "camera lens", True)
    # Fail before starting simulation if the requested layout cannot be packed.
    sample_layout(cfg, random.Random(rand["fixed_seed"]))
    return cfg


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
