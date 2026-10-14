# G1-D controller alignment — 2026-09-22

The simulator has two selectable controller modes. **Legacy is the default**
because it was working better for the current task. Aligned mode keeps the
packet-level changes available for controlled comparisons; it is not a
system-identification result or a claim that the physical robot is reproduced.

## Selecting a mode

Set this field under `robot` in `configs/scene.json`:

```json
"controller_alignment": false
```

Use the same setting in the simulation client:

```bash
--controller-alignment false
```

Both defaults are `false`, so the extra client option can be omitted. To restore
all alignment changes, set the JSON field to `true` and run the client with
`--controller-alignment true`. Restart both processes after changing modes.

The switch changes these behaviors:

| Behavior | Legacy (`false`, default) | Aligned (`true`) |
| --- | --- | --- |
| SDK motor fields used by simulation | `q` only | `q`, `dq`, `kp`, `kd`, `tau` |
| Shoulders/elbows Kp / Kd | 160 / 8 | 80 / 3 |
| Wrists Kp / Kd | 40 / 2 | 40 / 1.5 |
| Torso yaw Kp / Kd | 2000 / 80 | 300 / 3 |
| Arm velocity limit | 3 rad/s | 37 rad/s shoulders/elbows and wrist roll; 22 rad/s wrist pitch/yaw |
| Wrist effort limit | 5 Nm | 25 Nm roll; 5 Nm pitch/yaw |
| Torso-yaw velocity limit | 1 rad/s | 5 rad/s |
| Client target limiter in simulation | bypassed | same 20 rad/s host limiter as the real client |

The legacy values are the original Isaac simulation settings. The aligned gains
come from the checked-in `G1_29_ArmController`; its limits come from the pinned
G1-D URDF. These are software settings and manufacturer model limits, not
measurements of the physical robot.

## Temporarily disabling gravity feedforward

The simple OpenPI runner exposes `--gravity-feedforward true|false`, defaulting
to `true`. With `false`, it sends zero in each arm motor's `tau` field during
both homing and policy execution; position targets, gains, target limiting, and
gripper commands are unchanged. This is intended for an A/B diagnostic rather
than as a calibrated controller.

The OpenPI runners also expose `--gravity-model`. Its default is `g1d-dex1`, so
real and simulated runs calculate feedforward from the same G1-D + Dex1
inertial model used to build the simulated robot. This changes only Pinocchio's
gravity calculation; the existing hand14 model remains in use for inverse
kinematics. Gravity torque is evaluated at the latest measured joint positions,
while the policy output remains the commanded joint-position target. In other
words, the arm command follows
`tau_total = kp * (q_target - q_measured) - kd * dq_measured + g(q_measured)`.
To restore the previous feedforward model, use:

```bash
--gravity-model legacy-hand14
```

At the zero pose, the G1-D model requests about `-2.650 Nm` at each shoulder
pitch, `-2.431 Nm` at each elbow, and `-0.686 Nm` at each wrist pitch. The
legacy hand14 values are about `-3.643`, `-3.412`, and `-1.219 Nm`
respectively. The main arm inertials match; most of this difference comes from
the end effectors (`0.365 kg` Dex1 versus `0.697 kg` hand14 per side).

For an aligned-mode simulator comparison, add:

```bash
--sim true --controller-alignment true --gravity-feedforward false
```

Compare a fixed or homed arm pose first. If upward displacement disappears while
the desired position remains fixed, the effect comes from the feedforward/plant
combination. The checked-in real runner does request feedforward when the option
is left at its default: it calculates `tau = g(q_measured)` with Pinocchio at
30 Hz, and the arm publisher copies that value into `LowCmd.motor_cmd[].tau` at
250 Hz. Confirming
how the physical firmware realizes that requested effort requires recording raw
`LowCmd` and `LowState.motor_state[].tau_est` during a controlled comparison.

In legacy mode the simulator ignores `tau`, so changing this option does not
change simulated motion. It remains useful for aligned mode and real-robot A/B
tests.

## Scaling simulated gripper closure

Both OpenPI runners expose `--gripper-closure-scale`, with a default of `1.0`.
It multiplies the normalized distance from fully open before conversion to the
Dex1 range:

```text
adjusted = clip(1 - scale * (1 - policy_target), 0, 1)
```

Since Dex1 uses `0=closed` and `1=open`, values above `1.0` close farther while
preserving both endpoints. For example, with `--gripper-closure-scale 1.2`, a
policy target of `0.5` becomes `0.4`, while `1.0` remains fully open and `0.0`
remains fully closed. Start with `1.1` or `1.2` in simulation. Omit the option,
or set it back to `1.0`, to restore the original mapping. The option is local to
each client invocation, so it can be enabled for simulation without changing
the real-robot command.

## Policy action and DDS topic

The policy server returns absolute targets in the interleaved training layout
`[left arm 7, left gripper, right arm 7, right gripper]`. The client reorders
this to `[left arm 7, right arm 7, left gripper, right gripper]` before splitting
the arm and Dex1 commands. The OpenPI output transform converts the model's
internal arm deltas back to absolute joint positions, so the robot client must
not add the current joint state again. Arm targets are clamped to the selected
G1 model's joint limits before publication.

The simple runner exposes `--motion true|false`. Its default `false` publishes
`rt/lowcmd`, matching the existing G1-D invocation. Use `--motion true` to
publish `rt/arm_sdk` when the Unitree motion service owns the rest of the body;
the simulator bridge accepts either topic. This selection depends on the real
robot's operating mode and is independent of policy inference.

## Recording what the real robot receives

Run the read-only DDS recorder in a separate terminal before starting the real
application:

```bash
cd /home/humanoid/adam_unitree/unitree_lerobot
PYTHONNOUSERSITE=1 python unitree_lerobot/eval_robot/record_g1_arm_dds.py \
  --network-interface enx00e04c39c657 --duration 30 \
  --output logs/g1_real_arm.jsonl
```

It publishes nothing. It records both possible arm-command topics and
`rt/lowstate`, preserving the arm/waist `q`, `dq`, `kp`, `kd`, commanded `tau`,
reported `tau_est`, modes, timestamps, and CRC validity. Its one-second terminal
summary shows maximum commanded feedforward, reported torque estimate, and
position tracking error. A nonzero recorded `tau` proves that the host put
feedforward on the wire. Comparing otherwise identical stationary trials with
feedforward enabled and disabled shows whether the physical response changes;
`tau_est` is a firmware estimate, not an independent torque-sensor measurement.

## Aligned SDK command handling

When alignment is enabled, [dds/g1d_bridge.py](../../../dds/g1d_bridge.py) copies
these fields from active arm slots 15–28 and torso-yaw slot 12:

| Field | Simulation behavior |
| --- | --- |
| `q` | Absolute position target; existing articulation-limit clamping remains |
| `dq` | Joint velocity target in rad/s |
| `kp` | PhysX drive stiffness |
| `kd` | PhysX drive damping |
| `tau` | Additional feedforward joint effort in Nm |

The nominal relation is
`tau_total = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff`.
The existing **implicit PhysX drive** implements the PD part. No second explicit
PD torque is added. Gain writes also update Isaac Lab's actuator-model buffers,
so its approximate torque reports use the same gains as the physics drives.

Callbacks copy packets only. Gain/target writes occur on the simulation thread
before stepping. Gains are written to PhysX only when changed (and on reset).
Active command fields must be finite, and gains must be nonnegative. Invalid
packets leave the previous accepted command in place until its usual timeout.

On timeout, the existing watchdog latches measured position once, restores
default gains, and clears velocity/feedforward targets. Reset clears packets,
restores default gains, clears velocity/feedforward, and retains the post-reset
command hold. Inactive slots use fallback position holding/default gains with
zero velocity/feedforward; mode 0 is not an emulation of a powered-off motor.

`rt/arm_sdk` retains the existing approximate blend: `q` is interpolated between
measured position and the packet target, and `dq`/`tau` are multiplied by the
clamped blend weight. Packet gains apply for positive weight; zero weight uses
the watchdog hold. This blend is not a reproduction of Unitree's whole-body
controller. Weight 1 is the straightforward motor-command comparison case.

A publisher using aligned mode and sending only `q` while leaving `kp=kd=0`
requests zero PD effort. The workspace's G1-29 client already populates gains.
Legacy mode ignores packet gains and retains its configured simulation gains.
Dex1's SDK units and coupled-finger position adapter are unchanged.

## Physical model and remaining differences

The source URDF, prepared G1-D + Dex1 URDF, and composed USD were checked before
this change. All 31 link masses, centers of mass, and inertia tensors agreed
within numerical tolerance. No mass/inertia changes were warranted by that
check. Those values remain manufacturer model specifications, not measurements
of the particular robot and its attached equipment.

Remaining differences relevant to a trajectory comparison:

- Physics still runs at 120 Hz; the sorting task consumes commands at 30 Hz.
  A nominal 250 Hz SDK publisher can produce multiple packets between action
  samples. The bridge uses the latest packet, not a timestamped replay queue.
- In aligned mode, the host target limiter uses measured joint state, so different real/sim states
  can still produce different outgoing packets from the same high-level action.
  For identification, record commands after real-side limiting and replay those
  packets directly rather than passing them through the limiter again.
- The implicit drive is a continuous-time numerical approximation of PD, not
  the robot's sampled motor firmware. Delay, filtering, friction, armature, and
  torque-speed behavior remain uncalibrated.
- Manufacturer joint effort/speed limits do not establish effective motor
  torque-speed curves. Feedforward is an additional effort target; combined drive-plus-
  feedforward saturation is not established as equivalent to firmware limits.
- The chassis remains fixed; lift/torso pitch stay parked. Camera attachments,
  cable loads, gripper/contact behavior, and base motion require separate checks.
- Published simulated `tau_est` remains Isaac Lab's approximate actuator effort,
  not a calibrated physical torque-sensor measurement.

## Verification

CPU protocol checks (no DDS channels or robot commands):

```bash
source /home/humanoid/adam_unitree/setup_unitree_sim/activate.sh
cd /home/humanoid/adam_unitree/unitree_sim_isaaclab
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_g1d_*.py' -v
```

All 27 tests passed. These include SDK checks covering both the legacy
position-only path and aligned gain/velocity/feedforward propagation, zero gains,
invalid packets, gain-buffer synchronization, blend weight, inactive slots,
independent watchdogs, and clearing on timeout/reset.
`test_g1d_client.py` separately compares the actual outgoing packet from each
G1-29 client in real/sim modes, with small and large requested target changes,
without constructing DDS publishers or sending commands.

The bounded physics check opens no DDS channels and runs without cameras:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python tools/check_g1d_motor_control.py --headless --device cpu
```

The final CPU run passed all assertions and exited with status 0, printing
`G1D_MOTOR_CONTROL_CHECK_PASSED`. It checks the configured gains/effort/speed
limits and real PhysX response to position commands, velocity targets, and
nonzero feedforward with zero PD gains, plus timeout/reset.
It disables gravity/self-contact in its isolated scene to separate actuator
behavior from loads and collisions. It does not validate real-robot tracking,
camera rendering, or sorting success. The production task keeps its configured
gravity and collisions.

The test uses Kit's immediate shutdown after reporting its result, avoiding
renderer initialization during teardown of this CPU-only scene. GPU rendering
was not validated; a GPU-enabled attempt encountered unavailable GPU memory.

Restart the simulator and the G1 client to load these changes. No USD re-import
is needed: the runtime articulation configuration supplies defaults, and the
SDK bridge applies incoming motor fields.
