"""JSONL episode records of the spec tasks (one line per episode, written before its reset)."""

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
import time
import json

from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from ..sort_shapes.evaluation.config import evaluation_output_path
from ..sort_shapes.evaluation.recorder import _hash


class SpecRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.output_path = evaluation_output_path(env.cfg.evaluation)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.episode_numbers = [0] * env.num_envs
        self._last_episode_serial = [-1] * env.num_envs
        self._wall_start = [time.monotonic()] * env.num_envs
        env.g1d_evaluation_recorder = self

    def reset(self, env_ids: Sequence[int] | None = None):
        now = time.monotonic()
        for env_id in range(self._env.num_envs) if env_ids is None else [int(i) for i in env_ids]:
            self._wall_start[env_id] = now

    def _reason(self, env_id, fallback):
        if bool(self._env.g1d_evaluation.success[env_id].item()):
            return "success"
        manager = getattr(self._env, "termination_manager", None)
        if manager is not None and "time_out" in manager.active_terms:
            if bool(manager.get_term("time_out")[env_id].item()):
                return "timeout"
        return fallback

    def _record(self, env_id, fallback_reason):
        evaluator = getattr(self._env, "g1d_evaluation", None)
        if evaluator is None or int(evaluator.steps[env_id].item()) == 0:
            return
        serial = int(evaluator.episode_serial[env_id].item())
        if self._last_episode_serial[env_id] == serial:
            return
        env, spec, rubric = self._env, self._env.cfg.task_spec, evaluator.rubric
        steps = int(evaluator.steps[env_id].item())
        stages = [{"name": name, "achieved": bool(rubric.done[env_id, k]),
                   "first_step": int(evaluator.first_step[env_id, k]),
                   "first_time_s": (float(evaluator.first_step[env_id, k] * env.step_dt)
                                    if evaluator.first_step[env_id, k] >= 0 else None)}
                  for k, name in enumerate(rubric.stages)]
        frame = evaluator.frame(env)
        objects = [{"name": o["name"], "role": o["role"], "shape": o["shape"], "placeholder": o["placeholder"],
                    "asset_key": o.get("asset_key"), "asset_problem": o.get("asset_problem"),
                    "initial_pose_wxyz": evaluator.initial_pose[o["name"]][env_id].tolist(),
                    "final_pose_wxyz": [*frame["pose"][o["name"]][0][env_id].tolist(),
                                        *frame["pose"][o["name"]][1][env_id].tolist()]}
                   for o in spec["objects"]]
        record = {
            "schema_version": 1,
            "task": spec["name"],
            "suite": spec.get("suite"),
            "task_type": spec["task_type"],
            "prompt": spec["prompt"],
            "spec_file": Path(spec["spec_file"]).name,
            "spec_hash": spec["spec_hash"],
            "run_id": self.run_id,
            "run_label": env.cfg.evaluation["logging"].get("run_label", ""),
            "episode_id": f"{self.run_id}-env{env_id}-episode{self.episode_numbers[env_id]:06d}",
            "environment_id": env_id,
            "reason": self._reason(env_id, fallback_reason),
            "success": bool(evaluator.success[env_id].item()),
            "progress": float(rubric.progress()[env_id]),
            "perturbation": (getattr(env, "spec_layout", None) or [None] * env.num_envs)[env_id],
            "placed_fraction": float(rubric.placed_fraction()[env_id]),
            "stages": stages,
            "unsupported_stages": rubric.unsupported,
            "duration_s": float(steps * env.step_dt),
            "wall_duration_s": float(time.monotonic() - self._wall_start[env_id]),
            "control_steps": steps,
            "step_dt_s": float(env.step_dt),
            "seed": env.cfg.seed,
            "scene_config_hash": _hash(env.cfg.sorting),
            "camera_profile": env.cfg.sorting["cameras"]["profile"],
            "controller_alignment": env.cfg.sorting["robot"]["controller_alignment"],
            "command_timing": getattr(env, "g1d_command_timing", None),
            "evaluation_config_hash": _hash({k: v for k, v in env.cfg.evaluation.items() if k != "logging"}),
            "objects": objects,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._last_episode_serial[env_id] = serial
        self.episode_numbers[env_id] += 1
        self._wall_start[env_id] = time.monotonic()
        latched = [s["name"] for s in stages if s["achieved"]]
        print(f"[G1-D spec] {spec['name']} {record['reason']}: stages {latched or 'none'} of {rubric.stages}, "
              f"placed {record['placed_fraction']:.2f} -> {self.output_path}", flush=True)

    def write_pending(self, reason="closed", env_ids: Sequence[int] | None = None):
        if not self._env.cfg.evaluation["logging"]["enabled"]:
            return
        for env_id in range(self._env.num_envs) if env_ids is None else [int(i) for i in env_ids]:
            self._record(env_id, reason)

    def record_pre_reset(self, env_ids):
        self.write_pending("manual_reset", env_ids)
        return None, None

    def close(self, file_path):
        self.write_pending("closed")


@configclass
class SpecRecorderTermCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = SpecRecorder


@configclass
class SpecRecorderManagerCfg(RecorderManagerBaseCfg):
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset: bool = False
    export_in_close: bool = False
    evaluation = SpecRecorderTermCfg()
