"""Per-step evaluation of a spec task: the rubric (rubric.py) over a snapshot of poses, boxes and contacts."""

from collections.abc import Sequence

import numpy as np
import torch
from isaaclab.managers import ManagerTermBase

from .mdp import HAND_LINKS, object_key
from .rubric import CONTACT_N, Rubric
from .spec import aabb, local_points, object_by_role, usd_local_points


def _np(tensor):
    return tensor.detach().cpu().numpy().copy()


class SpecEvaluator(ManagerTermBase):
    """Termination term: True once the rubric's final stage has latched (if task_success.terminate_episode)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.spec = env.cfg.task_spec
        self.protocol = env.cfg.evaluation
        self.enabled = self.protocol["enabled"]
        self.rubric = Rubric(self.spec, env.num_envs)
        self.objects = {o["name"]: o for o in self.spec["objects"]}
        # Registered USD assets: boxes from their real geometry (as chiral), not from the spec box's corners.
        self.points = {name: usd_local_points(f"/World/envs/env_0/{object_key(name)}") if o["shape"] == "usd"
                       else local_points(o) for name, o in self.objects.items()}
        for name, o in self.objects.items():
            if o["shape"] == "usd":
                extent = self.points[name].max(0) - self.points[name].min(0)
                print(f"[G1-D spec] {name}: geometry extent {np.round(extent, 4).tolist()} m "
                      f"(spec box {[float(v) for v in o['bounding_box']]})", flush=True)
        self.has_main_contact = bool(object_by_role(self.spec, "main") and object_by_role(self.spec, "target"))
        robot = env.scene["robot"]
        self.finger_ids = {side: [robot.body_names.index(link) for link in links[:2]]
                           for side, links in HAND_LINKS.items()}
        n = env.num_envs
        self.steps = torch.zeros(n, dtype=torch.long, device=env.device)
        self.success = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.episode_serial = torch.zeros(n, dtype=torch.long, device=env.device)
        self.first_step = np.full((n, len(self.rubric.stages)), -1)
        self.initial_pose = {name: np.full((n, 7), np.nan) for name in self.objects}
        if self.rubric.unsupported:
            print(f"[G1-D spec] Stages without a check (always False): {self.rubric.unsupported}", flush=True)
        env.g1d_evaluation = self

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        rows = np.arange(self.num_envs)[env_ids] if isinstance(env_ids, slice) else np.asarray(
            env_ids.tolist() if hasattr(env_ids, "tolist") else env_ids, dtype=int)
        self.rubric.reset(rows)
        self.first_step[rows] = -1
        for pose in self.initial_pose.values():
            pose[rows] = np.nan
        self.steps[env_ids] = 0
        self.success[env_ids] = False
        self.episode_serial[env_ids] += 1

    def frame(self, env):
        """The rubric's inputs for this step, in the env frames."""
        origins = env.scene.env_origins
        pose = {}
        for name in self.objects:
            data = env.scene[object_key(name)].data
            pose[name] = _np(data.root_pos_w - origins), _np(data.root_quat_w)
        boxes = {name: aabb(self.objects[name], self.points[name], *pose[name]) for name in self.objects}
        contact = {}
        for links in HAND_LINKS.values():
            for link in links:
                forces = env.scene[f"contact_{link}"].data.force_matrix_w        # (N, 1, objects, 3), normal forces
                contact[link] = np.linalg.norm(_np(forces[:, 0]), axis=-1) > CONTACT_N
        main_contact = None
        if self.has_main_contact:
            forces = env.scene["contact_main"].data.force_matrix_w                 # (N, 1, [target, table], 3)
            main_contact = np.linalg.norm(_np(forces[:, 0]), axis=-1) > CONTACT_N
        body_pos = _np(env.scene["robot"].data.body_pos_w - origins[:, None, :])
        return {
            "pose": pose, "aabb": boxes, "contact": contact, "main_contact": main_contact, "main_contact_n_target": 1,
            "contact_cols": {name: k for k, name in enumerate(self.objects)}, "hand_links": HAND_LINKS,
            "fingers": {side: [body_pos[:, b] for b in ids] for side, ids in self.finger_ids.items()},
        }

    def __call__(self, env):
        if not self.enabled:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        frame = self.frame(env)
        steps = _np(self.steps)
        for i in np.flatnonzero(steps == 0):
            self.rubric.set_start(i, frame)
            for name, (pos, quat) in frame["pose"].items():
                self.initial_pose[name][i] = np.r_[pos[i], quat[i]]
        done = self.rubric.update(frame)
        newly = done & (self.first_step < 0)
        self.first_step[newly] = np.broadcast_to(steps[:, None], newly.shape)[newly]
        self.success = torch.as_tensor(self.rubric.success(), device=self.device)
        self.steps += 1
        if self.protocol["task_success"]["terminate_episode"]:
            return self.success.clone()
        return torch.zeros_like(self.success)
