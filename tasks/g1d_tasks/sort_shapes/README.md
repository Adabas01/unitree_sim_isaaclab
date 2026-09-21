# G1-D sorting scene

Task ID: `Isaac-Sort-Shapes-G1D-Dex1-Joint`.

The scene contains a parked G1-D with Dex1 grippers, one table, green cubes,
red cylinders, and green/brown open bins. Each bin has a floor and four collision
walls. The current default is two cubes and one cylinder. The robot holds its initial arm,
lift and torso pose in preview mode. The work area is inside a room with a
storage rack, cabinet and ceiling lights. Robot materials distinguish the body,
motors, wheels and grippers; these are approximate appearance overrides, not
manufacturer textures. Direct lighting, shadows and indirect illumination are
enabled explicitly because the default Isaac Lab preset disables direct light.

## Run

```bash
conda activate unitree_sim_env
cd /home/humanoid/adam_unitree/unitree_sim_isaaclab

python run_g1d_sort.py --device cpu --enable_cameras
```

Use `run_g1d_sort.py` for this task. It now has its own G1-D SDK adapter; the
existing `sim_main.py` launcher remains for the other robot tasks. The OpenPI
client can reuse the real-robot evaluator with `--sim true`; see below.
Close the application window or press Ctrl+C in the terminal to stop.

## SDK control

Start the simulation with arm, torso-yaw and gripper DDS enabled:

```bash
conda activate unitree_sim_env
export CYCLONEDDS_HOME=/home/humanoid/adam_unitree/setup_unitree_sim/cyclonedds-install
cd /home/humanoid/adam_unitree/unitree_sim_isaaclab
python run_g1d_sort.py --device cpu --enable_cameras --sdk --dds-interface lo
```

In a second terminal, run a small movement using the actual Unitree Python SDK:

```bash
conda activate unitree_sim_env
export CYCLONEDDS_HOME=/home/humanoid/adam_unitree/setup_unitree_sim/cyclonedds-install
cd /home/humanoid/adam_unitree/unitree_sim_isaaclab
python tools/g1d_sdk_demo.py --interface lo
```

The demo moves both wrists and torso yaw slightly, closes each gripper in turn,
and returns to the starting pose. Both processes use **DDS domain 1**. `lo`
keeps DDS on this computer; for a client on another machine, select the actual
network interface in both processes. Existing G1 arm clients use motor slots
15–28 and waist-yaw slot 12. Missing leg slots are zero in the published state.

| Topic | Direction | Message |
| --- | --- | --- |
| `rt/lowcmd` | Command | `unitree_hg/LowCmd_`, valid CRC, `mode_pr=0`, active motors `mode=1` |
| `rt/arm_sdk` | Command | Same message; motor 29 `q` supplies the 0–1 blend weight |
| `rt/lowstate` | State | `unitree_hg/LowState_`, valid CRC, published at approximately 100 Hz |
| `rt/dex1/left/cmd`, `rt/dex1/right/cmd` | Command | `unitree_go/MotorCmds_`, one motor each |
| `rt/dex1/left/state`, `rt/dex1/right/state` | State | `unitree_go/MotorStates_`, one motor each |
| `rt/reset_pose/cmd` | Command | `std_msgs/String_`: `"1"` randomizes objects, `"2"` resets the whole task |

SDK gripper positions are **0=closed, 5.4=open**. The adapter converts these to
the imported robot's finger travel. It uses named joints, not USD joint order.
It holds the initial pose until commands arrive. Each command stream has its
own timeout (default 0.5 s, `--command-timeout`); on expiry, the measured position
is latched and held. A whole-task reset clears cached commands.

This is a **position-control SDK bridge**, like the existing simulator's arm
action path: command `q` is used, while per-packet `kp`, `kd`, `dq` and `tau` do
not replace the simulation's configured gains. It does not emulate the real
motor firmware, full dynamic IMU measurements, or AGV wheel/lift RPCs. The base,
lift and torso pitch remain parked; torso yaw can be commanded through slot 12.

For head/wrist camera streaming, add `--stream-cameras` to the simulator command.
It uses `teleimager/cam_config_server.yaml` and the existing ZMQ ports 55555
(head), 55556 (left), 55557 (right), plus configured WebRTC services. SDK motor
control works independently of camera streaming.

## OpenPI policy smoke test

The policy server, simulator, and evaluator run as three processes. Start the
simulator with both DDS and camera streaming:

```bash
conda activate unitree_sim_env
export CYCLONEDDS_HOME=/home/humanoid/adam_unitree/setup_unitree_sim/cyclonedds-install
cd /home/humanoid/adam_unitree/unitree_sim_isaaclab
python run_g1d_sort.py --device cpu --enable_cameras \
    --sdk --dds-interface lo --stream-cameras
```

Start the policy server from the `ichrpi` environment. When it shares a GPU
with Isaac Sim, disable JAX's large up-front allocation:

```bash
cd /home/humanoid/adam_unitree/ichrpi
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=pi05_g1d_sort_lora \
    --policy.dir=/home/humanoid/adam_unitree/checkpoints/g1d-openpi-pr-2026-09-03/pi05_g1d_sort_lora/g1d_sort_lora/4999
```

First query the policy without applying its outputs:

```bash
conda activate unitree_lerobot
export CYCLONEDDS_HOME=/home/humanoid/adam_unitree/setup_unitree_sim/cyclonedds-install
cd /home/humanoid/adam_unitree/unitree_lerobot
PYTHONNOUSERSITE=1 python unitree_lerobot/eval_robot/eval_g1_openpi_simple.py \
    --sim true --policy-host 127.0.0.1 --policy-port 8000 \
    --image-host 127.0.0.1 --network-interface lo \
    --frequency 30 --max-steps 300 \
    --execute-actions false --execute-grippers false \
    --gripper-scale 5.4 --home-before-start false
```

Type `s` at the prompt. Once it prints finite `[30, 16]` action chunks without
any missing-image errors, repeat with `--max-steps 0 --execute-actions true
--execute-grippers true` to control the simulated arms and grippers. The
simulator and evaluator use DDS domain 1; the policy connection and camera
streams use TCP on localhost.

## Evaluation

The environment evaluates every object with five ordered, monotonic stages:

| Stage | Condition |
| --- | --- |
| Reach | Either gripper's pinch point stays within the configured distance of the object surface |
| Grasp | Both opposing fingers of one gripper contact the same object while the gripper is sufficiently closed |
| Lift | The bottom of the object stays above the tabletop by the configured clearance |
| Transport | The object centre moves over its target bin opening |
| Place | The whole object is inside the target bin, moving slowly, released, and remains there for the dwell time |

Each stage begins accumulating dwell time only after the previous stage was
credited. A broken condition resets that stage's dwell timer. Credited stages
remain credited for the rest of the episode. The partial success rate (PSR) is
the weighted mean of credited stages over all configured objects. Binary success
requires every object to complete `place`, followed by the task success dwell.
The evaluator also counts drops after lifting and stable placements in a wrong
bin. These metrics use simulator state and contact sensors and are available
only in `scene_state`; they are not policy inputs.

Thresholds, dwell times, weights, timeout, automatic success termination, and
logging are configured in [configs/evaluation.json](configs/evaluation.json).
The defaults are initial engineering thresholds and should be calibrated using
real rollouts before reporting a simulation-to-real comparison.

Start a labeled evaluation run like this:

```bash
python run_g1d_sort.py --device cpu --enable_cameras \
    --sdk --dds-interface lo --stream-cameras \
    --evaluation-run-label pi05_g1d_sort_lora-4999
```

Each completed, timed-out, manually reset, or closed episode is appended as one
JSON object to `logs/g1d_evaluation/episodes.jsonl`. The record contains binary
success, task and per-object PSR, stage completion times, geometric sorting
fraction, drops, wrong-bin events, duration, seed, and scene/evaluation config
hashes. `rt/reset_pose/cmd` value `"1"` closes the current evaluation episode,
randomizes the objects, and restarts its timeout without moving the robot.

Inspect the latest episode:

```bash
tail -n 1 logs/g1d_evaluation/episodes.jsonl | python -m json.tool
```

Summarize all episodes, grouped by the run labels used for different policy
checkpoints:

```bash
python tools/summarize_g1d_evaluation.py \
    logs/g1d_evaluation/episodes.jsonl
```

Use `--format json` for a machine-readable aggregate or `--group-by run_id` to
separate simulator launches. Alternative protocols and log files can be passed
with `--evaluation-config` and `--evaluation-output`.

To demonstrate randomized resets and save camera views:

```bash
python run_g1d_sort.py --device cpu --enable_cameras \
    --seed 7 --reset-every 10 --capture-dir /tmp/g1d-views
```

`--reset-every` is measured in simulated seconds. After 60 control steps, the
capture folder contains four PNGs plus the settings/seed/object poses in
`scene.json`. Without a reset interval the layout stays in place until the
configured episode timeout is reached (120 seconds by default). Physics
continues while the scene is playing.

For a bounded headless run:

```bash
python run_g1d_sort.py --device cpu --enable_cameras \
    --headless --steps 90 --capture-dir /tmp/g1d-views
```

## Change the scene

Edit [configs/scene.json](configs/scene.json) and restart the launcher. Lengths are metres,
masses are kilograms, and angles are radians. The coordinate axes are +X in
front of the robot, +Y to its left, and +Z upward.

| Setting | Meaning |
| --- | --- |
| `cubes.count`, `cylinders.count` | Object counts; zero is supported |
| `cubes.size` | X/Y/Z dimensions; equal values produce a cube |
| `cylinders.radius`, `cylinders.height` | Cylinder dimensions |
| `table.height` | Height of the upper tabletop surface |
| `bins[].center_xy` | Box position on the tabletop |
| `bins[].inner_size` | Usable inner X/Y dimensions and wall height above the bin floor |
| `bins[].wall_thickness` | Thickness of the floor and walls |
| `randomization.enabled` | New layout on each reset, or repeat a fixed layout |
| `randomization.fixed_seed` | Layout seed when randomization is disabled |
| `randomization.x_range`, `y_range`, `yaw_range` | Allowed object-center positions and upright yaw rotations |
| `randomization.clearance` | Minimum object-to-object, object-to-bin and table-edge separation |
| `robot.arm_positions` | Initial left-arm 7 values followed by right-arm 7 values, in radians |
| `robot.lift_positions` | Extension of the two lift joints, each from 0 to 0.21 m |
| `robot.torso_pitch`, `torso_yaw` | Parked torso pose |
| `room.size` | Room X/Y/Z dimensions |
| `room.ambient_intensity`, `ceiling_intensity` | Ambient fill and overhead lighting |

An alternative configuration can be selected with
`--scene-config /path/to/scene.json`. Counts/dimensions are set before constructing
the physics scene. Reset changes only object poses and velocities. The sampler
checks whole object footprints, including cube yaw, and excludes both boxes;
impossible packing requests fail with an error. Keep the spawn region clear of
the robot when moving the table or changing its initial pose.

The same `--seed` reproduces the sequence of requested layouts on the same
setup. Physics/rendering are not guaranteed to be bitwise deterministic.

## Cameras

The USD imported from URDF did not contain camera sensors. This task adds them
at scene creation; the original USD is unchanged.

| Sensor | Parent frame | Use |
| --- | --- | --- |
| `front_camera` | `Robot/head_link` | Full tabletop view |
| `left_wrist_camera` | `Robot/left_dex1_base_link` | Moves with left gripper |
| `right_wrist_camera` | `Robot/right_dex1_base_link` | Moves with right gripper |
| `world_camera` | Environment root | Overview for inspection |

All four produce 640×480 RGB images. Change resolution in `cameras.width/height`.
Each head/wrist camera has editable `position`, `forward`, `focal_length`, and
`horizontal_aperture` fields. Position and viewing direction are expressed in
the parent link frame. Image up is derived from that frame's +Z direction.
These are initial viewing positions, not calibrated replicas of the real
mounts, lenses, distortion or exposure. Camera sensors do not add a physical
housing or mass to the robot. Match the real camera extrinsics/intrinsics before
interpreting policy evaluation as representative of the real robot.

## Task interface

The Gym action and `observations["policy"]["state"]` both have 16 elements:

```text
left arm (7), left gripper (1), right arm (7), right gripper (1)
```

Each arm is ordered shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw.
Arm commands are absolute radians. Grippers use 0=closed, 1=open and couple the
two fingers on each side. This imported URDF opens at +0.0245 m and closes at
-0.020 m. The old simulator's Dex1 USD conversion helper uses a different
convention and is not reused here. Arm targets are clamped to joint limits;
gripper inputs are clamped to [0,1]. Non-finite actions are rejected.

The chassis is fixed to the world; wheels, lift and torso pitch have fixed
position targets. No AGV driving interface is implemented in this task. Robot
configuration and named joints are in [robots/g1d.py](../../../robots/g1d.py).

Policy observations also contain raw uint8 RGB under `head`, `left_wrist`, and
`right_wrist`; these are batched tensors. The separate `scene_state` group has
object poses, geometric sorting progress, flattened per-object stage flags, and
an evaluation summary. Objects are ordered cubes first, then cylinders, using
`cube_000`, `cylinder_000`, etc. The evaluation summary is ordered as reach,
grasp, lift, transport and place rates, then task PSR, binary success, total
drops and total wrong-bin events.

The reward measures the fraction of objects fully inside their target box and
moving slowly. Cubes target the brown box and cylinders the green box by default
(`target_bin` is configurable), matching the OpenPI evaluator prompt. The reward
remains a simple geometric training signal; the separate evaluator supplies the
ordered contact-aware metrics and success termination.

Actuator gains and friction are initial simulation settings. Self-collisions
are disabled; the original URDF has no collisions on the two moving lift links.
The task does not yet establish realistic grasp performance or motor-level
equivalence with the physical robot.

## Checks

Placement checks (no simulator needed):

```bash
python3 -m unittest discover -s tests -p test_g1d_layout.py
```

Layout, geometric predicates, ordered stage progression, protocol mapping, CRC
rejection, gripper units, blend weight and command-timeout checks (CPU only,
simulator environment required, no DDS connections created):

```bash
python -m unittest discover -s tests -p 'test_g1d_*.py'
```

Bounded physics/camera check, including different counts/dimensions and two
environments:

```bash
source /home/humanoid/adam_unitree/setup_unitree_sim/activate.sh
python /home/humanoid/adam_unitree/setup_unitree_sim/check_g1d_sort.py \
    --headless --device cpu --enable_cameras
```

An automated loopback SDK/ZMQ check starts its own simulator, runs the movement
demo against both command topics, receives all three camera streams and stops:

```bash
python /home/humanoid/adam_unitree/setup_unitree_sim/check_g1d_sdk.py
```

Validation so far: the first scene ran for 90 steps with four rendered views;
the room/SDK build ran for 2,400 steps, and an independent SDK client passed
arm, torso-yaw, both-gripper and CRC feedback checks on `rt/lowcmd`. The final
direct-lighting/stream-startup changes have not been rerun on the GPU because
that execution request was declined. The extended two-environment check also
needs rerunning after correcting its partial-reset argument to a tensor.
