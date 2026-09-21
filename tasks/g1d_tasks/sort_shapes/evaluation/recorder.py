"""Human-readable JSONL episode summaries for stage evaluation."""

from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json

from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from .config import STAGE_NAMES, evaluation_output_path


def _hash(mapping):
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class EvaluationRecorder(RecorderTerm):
    """Write one compact record before every reset and for a bounded run on close."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.output_path = evaluation_output_path(env.cfg.evaluation)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.episode_numbers = [0] * env.num_envs
        self._last_episode_serial = [-1] * env.num_envs
        env.g1d_evaluation_recorder = self

    def _reason(self, env_id, fallback):
        evaluator = getattr(self._env, "g1d_evaluation", None)
        if evaluator is not None and bool(evaluator.success[env_id].item()):
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
        episode_serial = int(evaluator.episode_serial[env_id].item())
        if self._last_episode_serial[env_id] == episode_serial:
            return
        scene = self._env.scene
        origin = scene.env_origins[env_id]
        object_records = []
        per_object_score = evaluator.per_object_score
        for object_index, item in enumerate(evaluator.items):
            data = scene[item["name"]].data
            hand_index = int(evaluator.grasp_hand[env_id, object_index].item())
            object_records.append({
                "name": item["name"],
                "kind": item["kind"],
                "target_bin": item["target_bin"],
                "initial_position": evaluator.initial_object_pos[env_id, object_index].cpu().tolist(),
                "final_position": (data.root_pos_w[env_id] - origin).cpu().tolist(),
                "final_quaternion_wxyz": data.root_quat_w[env_id].cpu().tolist(),
                "grasp_hand": None if hand_index < 0 else ("left", "right")[hand_index],
                "stages": {
                    name: {
                        "achieved": bool(evaluator.achieved[env_id, object_index, stage_index].item()),
                        "first_time_s": float(evaluator.first_time_s[env_id, object_index, stage_index].item()),
                    }
                    for stage_index, name in enumerate(STAGE_NAMES)
                },
                "partial_success": float(per_object_score[env_id, object_index].item()),
                "drop_count": int(evaluator.drop_count[env_id, object_index].item()),
                "wrong_bin_count": int(evaluator.wrong_bin_count[env_id, object_index].item()),
            })

        record = {
            "schema_version": 1,
            "run_id": self.run_id,
            "run_label": self._env.cfg.evaluation["logging"].get("run_label", ""),
            "episode_id": f"{self.run_id}-env{env_id}-episode{self.episode_numbers[env_id]:06d}",
            "environment_id": env_id,
            "reason": self._reason(env_id, fallback_reason),
            "success": bool(evaluator.success[env_id].item()),
            "partial_success": float(evaluator.task_score[env_id].item()),
            "geometric_sorted_fraction": float(evaluator.geometric_sorted_fraction[env_id].item()),
            "stage_rates": {
                name: float(evaluator.stage_rates[env_id, stage_index].item())
                for stage_index, name in enumerate(STAGE_NAMES)
            },
            "duration_s": float(evaluator.steps[env_id].item() * self._env.step_dt),
            "control_steps": int(evaluator.steps[env_id].item()),
            "step_dt_s": float(self._env.step_dt),
            "seed": self._env.cfg.seed,
            "scene_config_hash": _hash(self._env.cfg.sorting),
            "evaluation_config_hash": _hash({
                key: value for key, value in self._env.cfg.evaluation.items() if key != "logging"
            }),
            "objects": object_records,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._last_episode_serial[env_id] = episode_serial
        self.episode_numbers[env_id] += 1
        print(
            f"[G1-D evaluation] {record['reason']}: PSR={record['partial_success']:.3f}, "
            f"sorted={record['geometric_sorted_fraction']:.3f} -> {self.output_path}",
            flush=True,
        )

    def write_pending(self, reason="closed", env_ids: Sequence[int] | None = None):
        if not self._env.cfg.evaluation["logging"]["enabled"]:
            return
        if env_ids is None:
            env_ids = range(self._env.num_envs)
        elif hasattr(env_ids, "tolist"):
            env_ids = env_ids.tolist()
        for env_id in env_ids:
            self._record(int(env_id), reason)

    def record_pre_reset(self, env_ids):
        self.write_pending("manual_reset", env_ids)
        return None, None

    def close(self, file_path):
        self.write_pending("closed")


@configclass
class EvaluationRecorderTermCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = EvaluationRecorder


@configclass
class EvaluationRecorderManagerCfg(RecorderManagerBaseCfg):
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset: bool = False
    export_in_close: bool = False
    evaluation = EvaluationRecorderTermCfg()
