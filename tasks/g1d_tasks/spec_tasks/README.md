# G1-D spec tasks

Task specs from ponimatkin's [chiral](https://github.com/ponimatkin/chiral) (formerly isaaclab_biman_sim; specs
copied at `4d0554c` and, since 2026-11-19, `554fb5e`, named in each file's header) on this repo's G1-D workspace: the `sort_shapes` room, table, robot, controller, payload, joint
dynamics, cameras and DDS bridge stay as they are; only the objects and the success check come from the spec.

```bash
python run_g1d_sort.py --device cpu --enable_cameras --sdk --dds-interface lo --stream-cameras --task-spec stack_rgy_blocks
```

Gym id `Isaac-Spec-Task-G1D-Dex1-Joint`, config `G1DSpecEnvCfg(sorting=<scene.json settings>, task_spec=load_spec(...))`.

| File | Role |
|---|---|
| `specs/<suite>/<task>.yaml` | Copied specs, unchanged below a header naming the source commit |
| `stages.yaml` | Stage lists by task type (isaaclab_biman_sim `task_progressions.yaml`) |
| `spec.py` | Loading and validation; object shapes, boxes and spawn configs (physical properties as there) |
| `rubric.py` | Port of isaaclab_biman_sim `g1sim/success.py`: same stages, thresholds and holds |
| `evaluator.py` | Termination term: the rubric's inputs per control step (poses, boxes, contacts) |
| `mdp.py` | Torso frame (taken at startup), object placement at every reset, observations |
| `recorder.py` | One JSONL record per episode (`logs/g1d_spec_tasks/<task>.jsonl`) |

## Placement

Specs give each object's position in the robot's torso frame (`xy_torso`, x forward, y left, level frame at the
torso origin) and its centre height above the table top (`centre_z_above_table`). Here the torso origin and
heading are read from the robot at its start pose (startup event; printed as `[G1-D spec] Torso frame`), and
the table top is `table.height` of the scene. Every reset puts the objects back at these poses; there is no
layout randomisation yet.

The specs were authored for isaaclab_biman_sim's robots and table placements: the G1 suites (`G1`, `UNIFOLM`,
`GRASP_BENCH`) for the legged G1's recordings (table ~0.03 m above the torso origin, robot leaning ~0.15 rad
forward), `G1D_DATA` for the g1d-sort recordings (table ~0.10 m below). The default scene's table is 0.0855 m
below the torso origin (0.8455 m). For the G1 suites run with `--table-height 0.8755 --table-x 0.59` and an arm
start pose that clears the raised table (`--start-pose`, e.g. the dataset's home pose): with the default zero
pose the forearms rest on it, and at the default table x its front edge presses on the robot's body (the
printed torso origin moves by 2 mm). The forward lean is not reproduced. With the default table and zero pose,
objects can sit close to the hands (in `stack_rgy_blocks` the green block is partly under the left gripper).

Start pose: chiral evaluates every G1 / G1_ST / DROID10 task from `UNITREE_START_POSE` (biman/robots/g1_spec.py:
median first frame of 320 episodes of eight tabletop G1_Dex1 datasets of the mid-train mixture). Same pose here:
`--start-pose=-0.03,0.68,0.37,-0.07,-0.46,0.20,-0.87,-0.14,-0.71,-0.30,-0.11,0.45,0.21,0.96` (and the same
values as the client's `--home`).

## Objects

- Primitives (`Cube`, `Cylinder`, `Sphere`; density 450 kg/m^3) and generated open boxes (0.25 kg, 6 mm walls; USD
  written to `tmp/generated_usd/`), friction 1.0 / 0.8, contact offset 4 mm.
- Registered BEHAVIOR-1K objects (`source: behavior1k`, since 2026-11-19), as chiral's `biman/assets.py` spawns them:
  `assets/objects/registry.yaml` maps `category/model` to a single-rigid-body wrapper USD around a SimReady /
  Palatial / ForgeCAD asset (convex-hull colliders, friction 1.0 / 0.8 in the wrapper), scaled per axis to the spec's
  `bounding_box` (registry `native_extent`), mass from the registry or 500 kg/m^3 over the collider volume. The
  asset root is outside git: `$G1D_CHIRAL_ASSETS`, default `~/adam_unitree/sim_assets/chiral_554fb5e` (chiral's
  `assets/objects`, `assets/MANIFEST.csv` and specs exported at `554fb5e` with `git archive`). The original assets
  (public Hugging Face datasets, CC-BY-4.0) are downloaded once:
  ```bash
  python tools/fetch_spec_assets.py --all          # or task names; 711 MB for the specs here
  ```
  The rubric's object boxes come from the spawned geometry (mesh vertices, `spec.py` `usd_local_points`, port of
  chiral's `_local_points`; since 2026-11-20), so a rolled apple keeps a tight box. On 2026-11-19 the corners of the
  spec box were used instead, which rejected rolled round objects on a plate.
- Objects without a loadable asset (not in the registry, not on Hugging Face, articulated) need
  `--spec-placeholders`: chiral's stand-in of the same box (open box for containers, cylinder for plates and round
  objects, box otherwise), flagged `placeholder` with its `asset_problem` in every record.

| Task (suite) | Objects | Stand-ins with `--spec-placeholders` |
|---|---|---|
| put_fruit_on_plate, collect_eggs, find_green_blocks, put_pens_in_box (G1) | real assets / primitives | none |
| store_camera (G1) | | **camera** (main object; chiral's Nikon is a local Objaverse conversion) -> box |
| put_apple_in_basket, banana_on_yellow_plate, apple_on_green_plate, battery_in_green_box, rotor_in_blue_box (G1_ST) | | none |
| glue_stick_in_box (G1_ST) | | lamp (articulated) and marker (local asset) -> boxes |
| put_green_block_into_bowl, pick_water_bottle (DROID10) | | none |
| pick_spoon (DROID10) | | beaker (not in chiral's registry) -> open box |

## Rubric

Stages latch in order and stay latched; the final stage must hold 15 control steps (0.5 s), LIFT stages count
only while grasped, placements never while held. Grasp = both fingers of one Dex1 with normal force > 0.05 N
on the object; touch = a finger or the palm. Stages without a check (articulated props, fluids, hanging) are
always False and listed as `unsupported_stages`. `progress` = fraction of stages latched; `placed_fraction` =
fraction of `rubric.placements` satisfied at the end (multi-object tasks).
