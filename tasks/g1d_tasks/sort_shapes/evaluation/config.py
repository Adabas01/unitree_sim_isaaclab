"""Load and validate the stage-evaluation protocol."""

import json
import math
from pathlib import Path


STAGE_NAMES = ("reach", "grasp", "lift", "transport", "place")
DEFAULT_EVALUATION = Path(__file__).parents[1] / "configs" / "evaluation.json"
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _number(mapping, key, label, minimum=0.0, maximum=None):
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}.{key} must be a finite number")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{label}.{key} must be {suffix}")
    return float(value)


def load_evaluation_settings(path=DEFAULT_EVALUATION):
    """Return a validated evaluation protocol loaded from JSON."""
    path = Path(path)
    cfg = json.loads(path.read_text())
    if cfg.get("version") != 1:
        raise ValueError("evaluation.version must be 1")
    if type(cfg.get("enabled")) is not bool:
        raise ValueError("evaluation.enabled must be boolean")
    _number(cfg, "episode_timeout_s", "evaluation", minimum=0.001)

    stages = cfg.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(STAGE_NAMES):
        raise ValueError(f"evaluation.stages must contain exactly: {', '.join(STAGE_NAMES)}")
    for name in STAGE_NAMES:
        stage = stages[name]
        if not isinstance(stage, dict):
            raise ValueError(f"evaluation.stages.{name} must be an object")
        _number(stage, "dwell_s", f"stages.{name}")
        _number(stage, "weight", f"stages.{name}")
    if sum(stages[name]["weight"] for name in STAGE_NAMES) <= 0:
        raise ValueError("At least one stage weight must be positive")

    enter = _number(stages["reach"], "enter_distance_m", "stages.reach")
    exit_distance = _number(stages["reach"], "exit_distance_m", "stages.reach")
    if exit_distance < enter:
        raise ValueError("stages.reach.exit_distance_m must be at least enter_distance_m")
    pinch_offset = stages["reach"].setdefault("pinch_point_offset_m", [0.11, 0.0, 0.0])
    if (
        not isinstance(pinch_offset, list)
        or len(pinch_offset) != 3
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in pinch_offset)
    ):
        raise ValueError("stages.reach.pinch_point_offset_m must contain three finite numbers")
    _number(stages["grasp"], "minimum_contact_force_n", "stages.grasp")
    _number(stages["grasp"], "maximum_gripper_open_fraction", "stages.grasp", maximum=1.0)
    _number(stages["lift"], "table_clearance_m", "stages.lift")
    _number(stages["transport"], "bin_xy_margin_m", "stages.transport")
    place = stages["place"]
    _number(place, "bin_margin_m", "stages.place")
    _number(place, "maximum_linear_speed_m_s", "stages.place")
    _number(place, "maximum_angular_speed_rad_s", "stages.place")
    _number(place, "release_contact_force_n", "stages.place")
    _number(place, "minimum_gripper_open_fraction", "stages.place", maximum=1.0)
    if type(place.get("require_release")) is not bool:
        raise ValueError("stages.place.require_release must be boolean")

    success = cfg.get("task_success")
    if not isinstance(success, dict) or type(success.get("terminate_episode")) is not bool:
        raise ValueError("task_success.terminate_episode must be boolean")
    success.setdefault("require_release", True)
    if type(success["require_release"]) is not bool:
        raise ValueError("task_success.require_release must be boolean")
    _number(success, "dwell_s", "task_success")
    failures = cfg.get("failures")
    if not isinstance(failures, dict):
        raise ValueError("evaluation.failures must be an object")
    _number(failures, "drop_clearance_m", "failures")

    logging = cfg.get("logging")
    if not isinstance(logging, dict) or type(logging.get("enabled")) is not bool:
        raise ValueError("logging.enabled must be boolean")
    if not isinstance(logging.get("path"), str) or not logging["path"].strip():
        raise ValueError("logging.path must be a non-empty string")
    if not isinstance(logging.get("run_label", ""), str):
        raise ValueError("logging.run_label must be a string")
    return cfg


def evaluation_output_path(cfg):
    """Resolve a configured log path relative to the repository root."""
    path = Path(cfg["logging"]["path"]).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
