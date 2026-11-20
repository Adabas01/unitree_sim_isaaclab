"""Stage rubric of the spec tasks: a port of ponimatkin/isaaclab_biman_sim g1sim/success.py (commit 4d0554c), REALM's
task-progression semantics with its G1 hardening, so that a success here means what it means there.

A task's stages (stages.yaml, by task_type) latch in order: a stage is checked only once all earlier ones have
latched, and stays latched. It latches after holding for `stage_hold_steps` consecutive control steps (default 1;
the final stage 15 = 0.5 s at 30 Hz). LIFT_* count only while the main object is grasped (`lift_requires_grasp`);
placements use the strict checks (`place_check`: inside / on / seated_strict) and are never credited while held.

Grasp = both finger links of one Dex1 in contact with the object; touch = a finger or the Dex1 palm in contact;
contact = filtered normal force > CONTACT_N. Stages without a check here (articulated props, fluids, hanging) report
False and are listed in `unsupported`.

The checks read a per-step `frame` (evaluator.py): object poses and boxes in the env frame, contact flags per hand
link and object, and the main object's contacts with the target and the table.
"""

import numpy as np

from .spec import object_by_role, stages

CONTACT_N = 0.05          # N: a resting 40 g block presses 0.4 N
REACH_DIST = 0.10
FINAL_HOLD_STEPS = 15


def _rot_angle(q1, q2):
    """Angle (rad) between two wxyz quaternions."""
    d = abs(float(np.dot(q1, q2)) / (np.linalg.norm(q1) * np.linalg.norm(q2)))
    return 2.0 * np.arccos(min(1.0, d))


def _up_axis(q):
    w, x, y, z = q
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


class Rubric:
    def __init__(self, spec, num_envs):
        self.spec = spec
        self.num_envs = num_envs
        self.task_type = spec["task_type"]
        self.stages = stages(spec)
        self.opts = dict(spec.get("rubric") or {})
        self.main = object_by_role(spec, "main")
        self.target = object_by_role(spec, "target")
        holds = dict(self.opts.get("stage_hold_steps") or {})
        holds.setdefault(self.stages[-1], FINAL_HOLD_STEPS)
        self.holds = holds
        self.lift_requires_grasp = self.opts.get("lift_requires_grasp", True)
        self.checks = {
            "REACH": self._reach, "GRASP": self._grasp, "TOUCH": self._touch, "GRASP_BOTH": self._grasp_both,
            "LIFT_SLIGHT": lambda i: self._lift(i, 0.05, 0.01), "LIFT_LARGE": lambda i: self._lift(i, 0.10, 0.075),
            "ROTATED": self._rotated, "MOVE_CLOSE": self._move_close,
            "PLACE_INTO": self._place, "PLACE_ONTO": self._place_onto, "UPRIGHT": self._upright,
            "ON_TARGET": self._on_target, "PYRAMID": self._pyramid, "PLACE_ALL": self._place_all,
            "CAPPED": self._capped,
        }
        self.unsupported = [s for s in self.stages if s not in self.checks]
        self.f = None
        self.done = np.zeros((num_envs, len(self.stages)), bool)
        self.hold = np.zeros((num_envs, len(self.stages)), int)
        self.pos0 = np.zeros((num_envs, 3))
        self.quat0 = np.tile([1.0, 0.0, 0.0, 0.0], (num_envs, 1))

    # -- bookkeeping -------------------------------------------------------------------------------------------------
    def reset(self, env_ids):
        self.done[env_ids] = False
        self.hold[env_ids] = 0

    def set_start(self, i, frame):
        """The main object's pose at the episode's first evaluated step (the LIFT / ROTATED reference)."""
        if self.main:
            self.pos0[i], self.quat0[i] = frame["pose"][self.main][0][i], frame["pose"][self.main][1][i]

    def update(self, frame):
        """One control step: latch stages in order. Returns the (num_envs, stages) latched flags."""
        self.f = frame
        for i in range(self.num_envs):
            for k, stage in enumerate(self.stages):
                if self.done[i, k]:
                    continue
                fn = self.checks.get(stage)
                ok = bool(fn(i)) if fn is not None else False
                if stage.startswith("LIFT") and self.lift_requires_grasp:
                    ok = ok and self._grasp(i)
                self.hold[i, k] = self.hold[i, k] + 1 if ok else 0
                if self.hold[i, k] >= self.holds.get(stage, 1):
                    self.done[i, k] = True
                    continue
                break
        return self.done.copy()

    def success(self):
        return self.done[:, -1].copy()

    def progress(self):
        return self.done.mean(1)

    def placed_fraction(self):
        """put_all: fraction of the placements done, per env (partial credit for multi-object tasks)."""
        placements = self.opts.get("placements") or []
        if not placements or self.f is None:
            return self.success().astype(float)
        return np.array([np.mean([self._placed_named(i, p) for p in placements]) for i in range(self.num_envs)])

    # -- frame access ------------------------------------------------------------------------------------------------
    def _pos(self, name):
        return self.f["pose"][name][0]

    def _quat(self, name):
        return self.f["pose"][name][1]

    def _box(self, name):
        return self.f["aabb"][name]

    # -- contacts ----------------------------------------------------------------------------------------------------
    def _touch_obj(self, i, name):
        col = self.f["contact_cols"][name]
        return any(self.f["contact"][link][i, col] for links in self.f["hand_links"].values() for link in links)

    def _grasp_obj(self, i, name):
        col = self.f["contact_cols"][name]
        return any(all(self.f["contact"][link][i, col] for link in links[:2]) for links in self.f["hand_links"].values())

    # -- checks ------------------------------------------------------------------------------------------------------
    def _reach(self, i):
        if self.task_type in ("close_lid", "open_lid", "open_drawer", "close_drawer"):
            return self._touch(i)
        p = self._pos(self.main)[i]
        near = any(np.linalg.norm(f[i] - p) < REACH_DIST for fingers in self.f["fingers"].values() for f in fingers)
        return near or self._touch(i)

    def _grasp(self, i):
        return self._grasp_obj(i, self.main)

    def _grasp_both(self, i):
        """Both hands hold the main object (each with both fingers in contact)."""
        col = self.f["contact_cols"][self.main]
        return all(all(self.f["contact"][link][i, col] for link in links[:2]) for links in self.f["hand_links"].values())

    def _touch(self, i):
        return self._touch_obj(i, self.main)

    def _lift(self, i, dist, lift):
        p, p0 = self._pos(self.main)[i], self.pos0[i]
        return p[2] - p0[2] > lift and np.linalg.norm(p - p0) > dist

    def _rotated(self, i):
        return _rot_angle(self.quat0[i], self._quat(self.main)[i]) > 1.1

    def _move_close(self, i):
        p = self._pos(self.main)[i]
        if self.opts.get("move_close_to") == "aabb":
            lo, hi = self._box(self.target)[i]
            return float(np.linalg.norm(p - np.clip(p, lo, hi))) < 0.125
        return float(np.linalg.norm(p - self._pos(self.target)[i])) < 0.125

    def _on_top_of_target(self, i):
        olo, _ = self._box(self.main)[i]
        _, thi = self._box(self.target)[i]
        main_contact = self.f.get("main_contact")
        touching = bool(main_contact[i, :self.f["main_contact_n_target"]].any()) if main_contact is not None else False
        return touching and olo[2] >= thi[2] - 0.02

    def _in_footprint(self, i, wall=0.01):
        lo, hi = self._box(self.target)[i]
        olo, ohi = self._box(self.main)[i]
        c = (olo + ohi) / 2
        return bool(np.all(c[:2] >= lo[:2] + wall) and np.all(c[:2] <= hi[:2] - wall))

    def _place(self, i):
        if self._grasp(i):
            return False
        mode = self.opts.get("place_check")
        lo, hi = self._box(self.target)[i]
        olo, ohi = self._box(self.main)[i]
        c = (olo + ohi) / 2
        rim = float(self.opts.get("rim_margin", 0.02))
        if mode == "on":
            return self._in_footprint(i) and self._on_top_of_target(i)
        if mode == "seated_strict":
            return self._in_footprint(i) and lo[2] - 0.01 <= olo[2] < hi[2] - rim
        if mode == "inside":
            return self._in_footprint(i) and olo[2] >= lo[2] - 0.01 and c[2] < hi[2] - rim
        inside = self._in_footprint(i, 0.0) and olo[2] >= lo[2] - 0.01 and c[2] < hi[2]
        return inside or (self._in_footprint(i, 0.0) and self._on_top_of_target(i))

    def _place_onto(self, i):
        return not self._grasp(i) and self._in_footprint(i, 0.0) and self._on_top_of_target(i)

    def _upright(self, i):
        return _up_axis(self._quat(self.main)[i])[2] > np.cos(np.radians(20.0)) and not self._grasp(i)

    def _on_target(self, i):
        lo, hi = self._box(self.target)[i]
        p = self._pos(self.main)[i]
        return bool(np.all(p[:2] >= lo[:2]) and np.all(p[:2] <= hi[:2])) and not self._grasp(i)

    def _inside_named(self, i, obj, target):
        """`obj` inside `target` (the strict `inside` test), not held."""
        lo, hi = self._box(target)[i]
        olo, ohi = self._box(obj)[i]
        c = (olo + ohi) / 2
        rim = float(self.opts.get("rim_margin", 0.02))
        return (not self._grasp_obj(i, obj) and bool(np.all(c[:2] >= lo[:2] + 0.01) and np.all(c[:2] <= hi[:2] - 0.01))
                and olo[2] >= lo[2] - 0.01 and c[2] < hi[2] - rim)

    def _on_named(self, i, obj, target):
        """`obj` resting on top of `target`, not held: its centre over the target's footprint and its bottom within
        2 cm of the target's top."""
        lo, hi = self._box(target)[i]
        olo, ohi = self._box(obj)[i]
        c = (olo + ohi) / 2
        return (not self._grasp_obj(i, obj) and bool(np.all(c[:2] >= lo[:2]) and np.all(c[:2] <= hi[:2]))
                and hi[2] - 0.02 <= olo[2] <= hi[2] + 0.02)

    def _placed_named(self, i, placement):
        """One rubric.placements entry, [object, target] or [object, target, inside | on] (default: place_check)."""
        obj, target = placement[0], placement[1]
        mode = placement[2] if len(placement) > 2 else self.opts.get("place_check", "inside")
        return self._on_named(i, obj, target) if mode == "on" else self._inside_named(i, obj, target)

    def _capped(self, i):
        """The cap where it sits on the pen (rubric.cap_offset in the pen's frame, within cap_tol) with their long
        axes within 15 deg."""
        from .spec import quat_matrix_wxyz
        pp, pq = self._pos(self.main)[i], self._quat(self.main)[i]
        cp, cq = self._pos(self.target)[i], self._quat(self.target)[i]
        Rp, Rc = quat_matrix_wxyz(pq), quat_matrix_wxyz(cq)
        off = np.asarray(self.opts["cap_offset"], float)
        near = np.linalg.norm(cp - (pp + Rp @ off)) < float(self.opts.get("cap_tol", 0.012))
        return near and abs(float(np.dot(Rp[:, 1], Rc[:, 1]))) > np.cos(np.radians(15))

    def _place_all(self, i):
        return all(self._placed_named(i, p) for p in self.opts["placements"])

    def _pyramid(self, i):
        spec = self.opts.get("pyramid") or {}
        top, (b1, b2) = spec["top"], spec["bottom"]
        pt, p1, p2 = (self._pos(n)[i] for n in (top, b1, b2))
        tops = [self._box(n)[i, 1, 2] for n in (b1, b2)]
        top_bottom = self._box(top)[i, 0, 2]
        mid = (p1[:2] + p2[:2]) / 2
        return (top_bottom > min(tops) - 0.03 and np.linalg.norm(pt[:2] - mid) < 0.06
                and np.linalg.norm(p1[:2] - p2[:2]) < 0.15)
