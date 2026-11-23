"""Write the in-distribution task specs of suite G1_DATA: two Unitree G1_Dex1 tasks of the lab mid-train mixture laid
out as their recordings show them (unlike the REALM layouts of the G1 suite). Numbers from
`tools/recording_layouts.py` on the fetched episodes (2026-11-20; ~/adam_unitree/datasets/unitree_reference/):

- `events`: where the grippers closed and opened (forward kinematics of the recorded joints, torso frame): where
  objects were picked and put, and from the grasp heights the table height in the torso frame. The release points
  are the gripper ~6-8 cm above a container, not the container centre.
- the first head frames, back-projected with chiral's G1 head camera after `fitcam`: sizes, counts and arrangement.

Positions come from the forward kinematics: the policy outputs joint angles, and the same angles put our G1-D's
grippers at (nearly) the same torso-frame points, so the objects go where the recorded arms picked and put them. The
camera back-projection puts the containers 5-10 cm further away than the releases allow (Collect_Eggs episode 0: tray
front edge x 0.48, releases into its front cells at x 0.41; Find_Green_Blocks: bin front edge 0.45, releases 0.41);
the projected gripper does land on the real one at grasps and releases (the ray direction is right, the depth along
it depends on the camera height, which `fitcam` does not fit), so the camera is used for sizes only.

    python tools/author_g1_data_specs.py      # writes tasks/g1d_tasks/spec_tasks/specs/G1_DATA/*.yaml

Run the specs with the table and start pose of their header (`--table-height`, `--start-pose`); the torso origin
of our G1-D is 0.8455 m above the floor.
"""

import math
from pathlib import Path

import numpy as np
import yaml

OUT = Path(__file__).resolve().parents[1] / "tasks" / "g1d_tasks" / "spec_tasks" / "specs" / "G1_DATA"
TORSO_Z = 0.8455   # our G1-D torso origin above the floor (sim printout "[G1-D spec] Torso frame")
TABLE_Z = 0.07     # table top in the torso frame: Collect_Eggs grasps of eggs lying on the table put the gripper centre
                   # at 0.08-0.10 (egg centre ~0.016 above the table); the same for Find_Green_Blocks (same robot squat
                   # and table; its grasps come from the top of a brick pile: lowest fingertip 0.083 min). +-2 cm.


def quat_yaw(yaw):
    return [0.0, 0.0, round(math.sin(yaw / 2), 4), round(math.cos(yaw / 2), 4)]


def obj(name, role, asset, box, xy, quat, z, **extra):
    return {"name": name, "role": role, "asset": asset, "bounding_box": [round(float(v), 4) for v in box],
            "xy_torso": [round(float(v), 4) for v in xy], "orientation_torso_xyzw": quat,
            "centre_z_above_table": round(float(z), 4), **extra}


def write(name, header, spec):
    text = "".join(f"# {line}\n" if line else "#\n" for line in header.strip().splitlines())
    text += yaml.safe_dump(spec, sort_keys=False, width=120, default_flow_style=None)
    (OUT / f"{name}.yaml").write_text(text)
    print(f"wrote {OUT / name}.yaml ({len(spec['objects'])} objects)")


def start_pose(values):
    arms = values[0:7] + values[8:15]
    return ",".join(f"{v:.3f}" for v in arms)


def collect_eggs():
    # Tray: the releases into its free cells (16 episodes, 60 releases) span x 0.35-0.46, y -0.03..+0.12 (10-90 %),
    # median (0.41, 0.03): the free cells are mostly in its front row. A 3 x 5 pulp tray (long side left-right in the
    # head view; 0.20-0.26 m wide in the frames), 45 mm cells: front row at x 0.41, centre x 0.455.
    cx, cy, pitch = 0.455, 0.03, 0.045
    tray = obj("tray", "target", {"source": "generated", "kind": "open_box", "rgba": [0.93, 0.66, 0.72, 1.0]},
               [3 * pitch + 0.02, 5 * pitch + 0.02, 0.03], [cx, cy], quat_yaw(0.0), 0.015, fixed_base=True)
    # SimReady chicken egg (REALM's egg), recoloured: the recorded toy eggs are light peach, the model's texture brown
    egg_asset = {"source": "behavior1k", "category": "egg", "model": "qopwgx", "rgba": [0.95, 0.76, 0.58, 1.0]}
    egg_box = [0.043, 0.033, 0.033]  # toy eggs, ~43 mm long (one per 45 mm tray cell)
    empty = {(0, 1), (0, 2), (0, 4), (1, 3)}  # (row from the robot, column from the right): the 4 free cells
    objects, filled = [], 0
    for row in range(3):
        for col in range(5):
            if (row, col) in empty:
                continue
            filled += 1
            x, y = cx + (row - 1) * pitch, cy + (col - 2) * pitch
            objects.append(obj(f"tray_egg_{filled}", "distractor", egg_asset, egg_box, [x, y],
                               [0.0, 0.7071, 0.0, 0.7071], 0.006 + egg_box[0] / 2, fixed_base=True))  # upright
    # Loose eggs: four (11 in the tray + 4 = 15), where the recorded grasps outside the tray cluster: front-left
    # (0.31-0.38, +0.07..+0.20), left (0.45-0.49, +0.20..+0.23), front (0.27-0.34, -0.03..+0.04), right
    # (0.36-0.50, -0.07..-0.26); episode 0's eggs back-project to the same four places.
    loose = [(0.35, 0.20, 0.4), (0.48, 0.22, 1.6), (0.29, -0.01, -0.9), (0.42, -0.17, 0.2)]
    eggs = [obj(f"egg_{k + 1}", "main" if k == 0 else "distractor", egg_asset, egg_box, xy[:2], quat_yaw(xy[2]),
                egg_box[1] / 2 + 0.001) for k, xy in enumerate(loose)]
    spec = {
        "name": "collect_eggs_unitree", "suite": "G1_DATA",
        "source": "huggingface.co/datasets/unitreerobotics/G1_Dex1_Collect_Eggs (16 episodes, grasp/release FK + first frames)",
        "task_type": "put_all",
        "prompt": "Collect the eggs and place them into the tray.",
        "nouns": {"object": "eggs", "target": "tray", "verb": "Collect"},
        "rubric": {"place_check": "inside", "rim_margin": 0.0, "move_close_to": "aabb", "lift_requires_grasp": True,
                   "placements": [[f"egg_{k + 1}", "tray"] for k in range(len(eggs))]},
        "objects": eggs + [tray] + objects,
        "prompt_variants": {},
    }
    pose = [-0.121, 0.761, 0.33, -0.031, -0.531, 0.32, -1.026, 0.988, -0.212, -0.802, -0.163, -0.155, 0.464, 0.294, 1.057, 0.995]
    header = f"""
In-distribution layout of the mid-train dataset G1_Dex1_Collect_Eggs (1000 episodes; prompt verbatim, the first of its
five phrasings), written by tools/author_g1_data_specs.py. As recorded: a 3 x 5 pulp egg tray in the middle (pink in
the first half of the episodes, blue later), 11 eggs standing in it, 4 loose eggs lying in front of it and beside
it; the robot puts the loose ones into the free cells. Positions from the recorded grasps (loose eggs) and releases
(tray) of 16 episodes (forward kinematics, torso frame); sizes and counts from the first head frames. The tray and
its eggs are fixed (kinematic); an egg counts when it lies inside the tray, not on top of other eggs.
Run with: --table-height {TORSO_Z + TABLE_Z:.4f} (table {TABLE_Z:+.3f} m in the torso frame, from the lowest grasps)
and --start-pose={start_pose(pose)} (median first frame of the 16 episodes). Torso frame: x forward, y left, metres.
"""
    write("collect_eggs_unitree", header, spec)


def find_green_blocks(variant="1green", seed=0):
    """variant "1green" (find_green_blocks_unitree, since 2026-11-23 14:00): one green brick near the middle of a flat
    pile, as recorded. "2green" (find_green_blocks_unitree_2green): the first version (two green bricks at the bottom
    of a three-layer pile, both needed), kept because the runs of 2026-11-20/23 used it (spec hash e21f2ffd7055e2e5
    under the name find_green_blocks_unitree); same seed and order, so its layout is unchanged."""
    # Pile: the grasps (15 episodes, 35 grasps) span x 0.31-0.48, y -0.15..+0.12 (10-90 %), median (0.41, -0.02);
    # ~0.17 x 0.23 m. In the first frames ~30 red and yellow DUPLO-size bricks (2x2, 2x4), mostly one layer, with one
    # green brick: each recorded episode drops at most one brick over the bin (12 of 15 once), the last frames show a
    # single green brick in the bin, and the episodes last 5-12 s with 1-4 grasps.
    # Bin: each episode's last release (the green brick, dropped ~6 cm above the rim) median (0.41, -0.31),
    # x 0.36-0.52, y -0.44..-0.12; a teal parts bin ~0.16 m wide in the frames.
    cx, cy, rx, ry = 0.40, 0.0, 0.085, 0.115  # pile centre and half extents
    h = 0.0192
    sizes = {"2x2": (0.0318, 0.0318), "2x4": (0.0636, 0.0318)}
    colours = {"red": [0.80, 0.08, 0.08, 1.0], "yellow": [0.98, 0.72, 0.10, 1.0], "green": [0.15, 0.62, 0.22, 1.0]}
    if variant == "2green":
        greens = 2
        bottom = [("green", "2x2"), ("green", "2x2")] + [("red", "2x2")] * 5 + [("red", "2x4")] * 2 + \
                 [("yellow", "2x2")] * 3 + [("yellow", "2x4")] * 2
        middle = [("red", "2x2")] * 4 + [("red", "2x4")] * 2 + [("yellow", "2x2")] * 3 + [("yellow", "2x4")] * 1
        top = [("red", "2x2")] * 2 + [("yellow", "2x2")] * 2
        scales = (1.0, 0.85, 0.55)
    else:
        greens = 1
        bottom = [("green", "2x2")] + [("red", "2x2")] * 5 + [("red", "2x4")] * 2 + [("yellow", "2x2")] * 4 + \
                 [("yellow", "2x4")] * 2
        middle = [("red", "2x2")] * 5 + [("red", "2x4")] * 1 + [("yellow", "2x2")] * 3 + [("yellow", "2x4")] * 1
        top = [("red", "2x2")] * 1 + [("yellow", "2x2")] * 1
        scales = (1.0, 0.85, 0.5)

    def corners(x, y, yaw, size, margin=0.002):
        hx, hy = size[0] / 2 + margin, size[1] / 2 + margin
        c, s_ = math.cos(yaw), math.sin(yaw)
        return np.array([(x + c * dx - s_ * dy, y + s_ * dx + c * dy) for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))])

    def overlap(a, b):  # separating axis test of two convex quadrilaterals
        for poly in (a, b):
            for k in range(4):
                edge = poly[(k + 1) % 4] - poly[k]
                axis = np.array([-edge[1], edge[0]])
                pa, pb = a @ axis, b @ axis
                if pa.max() < pb.min() or pb.max() < pa.min():
                    return False
        return True

    class NoFit(Exception):
        pass

    def place(items, layer, scale, centre_first=0):
        placed, out = [], []
        for k, (colour, kind) in enumerate(items):
            for _ in range(20000):
                a = rng.uniform(0, 2 * math.pi)
                d = rng.uniform(0.0, 0.35) if k < centre_first else math.sqrt(rng.uniform(0, 1))
                x, y, yaw = cx + scale * rx * d * math.cos(a), cy + scale * ry * d * math.sin(a), rng.uniform(-math.pi, math.pi)
                quad = corners(x, y, yaw, sizes[kind])
                if not any(overlap(quad, other) for other in placed):
                    break
            else:
                raise NoFit
            placed.append(quad)
            out.append((colour, kind, x, y, yaw, layer))
        return out

    # random sequential packing: the first draw that fits (attempt 0 = numpy seed `seed`, later ones [seed, attempt])
    for attempt in range(100):
        rng = np.random.default_rng(seed if attempt == 0 else [seed, attempt])
        try:
            bricks = place(bottom, 0, scales[0], centre_first=greens) + place(middle, 1, scales[1]) + \
                     place(top, 2, scales[2])
            break
        except NoFit:
            continue
    else:
        raise RuntimeError("pile does not fit")
    draw = f"seed {seed}" if attempt == 0 else f"seed [{seed}, {attempt}]"
    objects, count = [], {}
    for colour, kind, x, y, yaw, layer in bricks:
        count[colour] = count.get(colour, 0) + 1
        name = f"{colour}_{count[colour]}"
        z = h / 2 + 0.0005 + layer * (h + 0.004)  # upper layers drop ~4 mm onto the one below and settle
        objects.append(obj(name, "main" if name == "green_1" else "distractor",
                           {"source": "primitive", "primitive_type": "Cube", "rgba": colours[colour]},
                           [*sizes[kind], h], [x, y], quat_yaw(yaw), z))
    bin_ = obj("bin", "target", {"source": "generated", "kind": "open_box", "rgba": [0.08, 0.62, 0.58, 1.0]},
               [0.12, 0.16, 0.08], [0.42, -0.32], quat_yaw(0.0), 0.04)
    name = "find_green_blocks_unitree" + ("_2green" if variant == "2green" else "")
    spec = {
        "name": name, "suite": "G1_DATA",
        "source": "huggingface.co/datasets/unitreerobotics/G1_Dex1_Find_Green_Blocks (15 episodes, grasp/release FK + first frames)",
        "task_type": "put_all" if variant == "2green" else "put",
        "prompt": "Find the green blocks in the pile and put them into the box.",
        "nouns": {"object": "green blocks", "target": "box", "verb": "put"},
        "rubric": ({"place_check": "inside", "move_close_to": "aabb", "lift_requires_grasp": True,
                    "placements": [["green_1", "bin"], ["green_2", "bin"]]} if variant == "2green" else
                   {"place_check": "inside", "move_close_to": "aabb", "lift_requires_grasp": True}),
        "objects": objects + [bin_],
        "prompt_variants": {},
    }
    pose = [-0.106, 0.585, 0.397, 0.017, -0.416, 0.059, -0.808, 0.995, -0.123, -0.712, -0.201, -0.038, 0.455, 0.065, 1.088, 0.991]
    if variant == "2green":
        layout = """FIRST VERSION, NOT AS RECORDED (kept for the runs of 2026-11-20/23, which used it under the name
find_green_blocks_unitree, spec hash e21f2ffd7055e2e5): 28 bricks in three layers (14, 10, 4; the upper ones drop a
few mm and settle), 2 green 2x2 bricks in the bottom layer near the middle, both needed in the bin. The recordings
have one green brick and a flatter pile (find_green_blocks_unitree)."""
    else:
        layout = """Here 26 bricks, mostly one layer (14, then 10 and 2 on top; the upper ones drop a few mm and settle), the
one green 2x2 brick in the bottom layer near the middle, partly covered; success = it lies in the bin (task type
put, as REALM's find_green_blocks)."""
    header = f"""
In-distribution layout of the mid-train dataset G1_Dex1_Find_Green_Blocks (490 episodes; prompt verbatim), written by
tools/author_g1_data_specs.py (pile drawn with numpy {draw}). As recorded: a loose pile of ~30 red and yellow
DUPLO-size bricks (2x2 and 2x4, 19 mm high), mostly one layer, with one green brick in it, a teal parts bin to the
right; the robot moves a few bricks aside if needed and puts the green one into the bin (episodes 5-12 s).
{layout} Pile region from the recorded grasps, bin from each episode's last release (15 episodes, forward
kinematics, torso frame); brick sizes and counts from the first head frames. The real bin has a sloped front; this
one is a plain open box.
Run with: --table-height {TORSO_Z + TABLE_Z:.4f} (table {TABLE_Z:+.3f} m in the torso frame, from the Collect_Eggs
grasps; see author_g1_data_specs.py) and --start-pose={start_pose(pose)} (median first frame of the 15 episodes).
Torso frame: x forward, y left, metres.
"""
    write(name, header, spec)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    collect_eggs()
    find_green_blocks("2green")
    find_green_blocks("1green")
