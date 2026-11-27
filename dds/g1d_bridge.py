"""G1-D SDK bridge using the existing G1/Dex1 wire protocol.

Callbacks only copy commands. All physics access happens on the simulation
thread; a separate 100 Hz publisher sends immutable state snapshots.

Command timing. Free-running (the default of Unitree's sim, kept for comparison):
each control step starts with the newest command that has arrived, so a command
sent in answer to a state acts one control step later than on the real robot.
Lockstep: the state is published as soon as it is computed, and the next step
waits until the client has answered it. The client (master-thesis deploy/
g1_robot.py, sim only) publishes the tick of the state it answered on
SIM_STEP_ACK_TOPIC after setting its targets; its controllers publish those
targets from their own threads (arms 250 Hz, grippers 200 Hz), so the bridge then
also waits for the second arm and gripper command received after the acknowledgement
(the first can come from a loop iteration that read the target before it changed).
With no client acknowledging (idle, between rollouts, old clients), the sim runs free.
"""

import math
import threading
import time
from typing import NamedTuple

import torch
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_, unitree_go_msg_dds__MotorState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.utils.crc import CRC

SIM_STEP_ACK_TOPIC = "rt/sim_step/ack"

from robots.g1d_joints import ARM_JOINTS, FINGER_JOINTS, REAL_GAP_PER_RAW_M, SDK_GRIPPER_OPEN, finger_to_grip, joint_state


class ArmCommand(NamedTuple):
    timestamp: float
    q: list[float]
    active: list[bool]
    weight: float
    dq: list[float]
    kp: list[float]
    kd: list[float]
    tau: list[float]


class G1DBridge:
    SDK_GRIPPER_OPEN = SDK_GRIPPER_OPEN

    def __init__(self, env, timeout=0.5, reset_hold=1.1, controller_alignment=False, lockstep=False,
                 lockstep_timeout=0.25, lockstep_idle=1.0, lockstep_command_wait=0.05):
        if env.num_envs != 1:
            raise ValueError("One DDS robot requires exactly one simulation environment")
        if not math.isfinite(reset_hold) or reset_hold < 0:
            raise ValueError("reset_hold must be finite and nonnegative")
        if not all(0 < value < float("inf") for value in (lockstep_timeout, lockstep_idle, lockstep_command_wait)):
            raise ValueError("Lockstep timeouts must be positive and finite")
        self.env = env
        self.robot = env.scene["robot"]
        self.timeout = timeout
        self.reset_hold = reset_hold
        self.controller_alignment = controller_alignment
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._write_lock = threading.Lock()
        self.lockstep = lockstep
        self.lockstep_timeout = lockstep_timeout
        self.lockstep_idle = lockstep_idle
        self.lockstep_command_wait = lockstep_command_wait
        self.lockstep_stats = {"answered": 0, "no_answer": 0, "no_command": 0, "free": 0}
        self._ack = None
        self._received = {}
        self._published_tick = None
        self._commands = {}
        self._reset_requested = None
        self._state = None
        self._error = None
        self._stop = threading.Event()
        self.rejected_commands = 0
        self.accepted_commands = 0
        self.crc = CRC()
        self._arm_ids = [self.robot.joint_names.index(n) for n in ARM_JOINTS]
        self._finger_ids = [self.robot.joint_names.index(n) for n in FINGER_JOINTS]
        self._yaw_id = self.robot.joint_names.index("torso_Joint")
        if self.controller_alignment:
            self._init_motor_targets()
        self._init_stiction()
        self._torso_id = self.robot.body_names.index("torso_link")
        self._held_action = joint_state(env).clone()
        self._held_yaw = self.robot.data.joint_pos[:, self._yaw_id].clone()
        self._fresh = set()
        self._ignore_commands_until = 0.0
        self.publishers = [ChannelPublisher("rt/lowstate", LowState_)]
        self.publishers += [ChannelPublisher(f"rt/dex1/{side}/state", MotorStates_) for side in ("left", "right")]
        for publisher in self.publishers:
            publisher.Init()
        self.subscribers = []
        for topic in ("rt/lowcmd", "rt/arm_sdk"):
            subscriber = ChannelSubscriber(topic, LowCmd_)
            subscriber.Init(lambda msg, topic=topic: self._arm_command(msg, topic), 1)
            self.subscribers.append(subscriber)
        for side in ("left", "right"):
            subscriber = ChannelSubscriber(f"rt/dex1/{side}/cmd", MotorCmds_)
            subscriber.Init(lambda msg, side=side: self._gripper_command(msg, side), 1)
            self.subscribers.append(subscriber)
        subscriber = ChannelSubscriber("rt/reset_pose/cmd", String_)
        subscriber.Init(self._reset_command, 1)
        self.subscribers.append(subscriber)
        subscriber = ChannelSubscriber(SIM_STEP_ACK_TOPIC, String_)
        subscriber.Init(self._step_ack, 1)
        self.subscribers.append(subscriber)
        self.update_state()
        self._thread = threading.Thread(target=self._publish_loop, name="g1d_state", daemon=True)
        self._thread.start()
        print(
            f"[G1-D SDK] Domain 1; arm slots 15–28, torso yaw 12; Dex1 0=closed, "
            f"5.4=open; controller={'aligned' if self.controller_alignment else 'legacy position-only'}; "
            f"post-reset command hold={self.reset_hold:.2f}s; command timing="
            f"{'lockstep' if self.lockstep else 'free-running (commands act one control step late)'}",
            flush=True,
        )

    def _arm_command(self, msg, topic):
        # Motor 29 is the arm_sdk blend weight; leg slots never command this robot.
        try:
            if msg.mode_pr != 0 or CRC().Crc(msg) != msg.crc:
                raise ValueError("Unsupported mode or invalid CRC")
            slots = [*range(15, 29), 12]
            q = [float(msg.motor_cmd[i].q) for i in slots]
            active = [msg.motor_cmd[i].mode == 1 for i in slots]
            if any(on and (not math.isfinite(value) or abs(value) > 100) for on, value in zip(active, q)):
                raise ValueError("Invalid position target")
            if self.controller_alignment:
                fields = {name: [float(getattr(msg.motor_cmd[i], name)) for i in slots]
                          for name in ("dq", "kp", "kd", "tau")}
                for name, values in fields.items():
                    if any(on and (not math.isfinite(value) or (name in ("kp", "kd") and value < 0))
                           for on, value in zip(active, values)):
                        raise ValueError(f"Invalid {name} command")
            else:
                fields = {name: [0.0] * len(slots) for name in ("dq", "kp", "kd", "tau")}
            weight = float(msg.motor_cmd[29].q) if topic == "rt/arm_sdk" else 1.0
            if not math.isfinite(weight):
                raise ValueError("Invalid arm_sdk weight")
            with self._lock:
                if time.monotonic() < self._ignore_commands_until:
                    return
                self._commands["arms"] = ArmCommand(
                    time.monotonic(), q, active, max(0.0, min(1.0, weight)), **fields,
                )
                self.accepted_commands += 1
                self._received["arms"] = self._received.get("arms", 0) + 1
                self._changed.notify_all()
        except (ValueError, IndexError, TypeError, OverflowError):
            self.rejected_commands += 1

    def _gripper_command(self, msg, side):
        try:
            q = float(msg.cmds[0].q)
            if not math.isfinite(q):
                raise ValueError("Invalid gripper target")
            with self._lock:
                if time.monotonic() < self._ignore_commands_until:
                    return
                self._commands[side] = (time.monotonic(), max(0.0, min(1.0, q / self.SDK_GRIPPER_OPEN)))
                self.accepted_commands += 1
                self._received[side] = self._received.get(side, 0) + 1
                self._changed.notify_all()
        except (ValueError, IndexError, TypeError):
            self.rejected_commands += 1

    def _reset_command(self, msg):
        if msg.data in ("1", "2"):
            with self._lock:
                self._reset_requested = msg.data

    def _step_ack(self, msg):
        try:
            tick = int(msg.data)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._ack = (tick, time.monotonic(), dict(self._received))
            self._changed.notify_all()

    @staticmethod
    def _tick_reached(tick, target):
        """tick >= target on the 32-bit millisecond clock of LowState.tick."""
        return (tick - target) % 2**32 < 2**31

    def wait_for_reply(self):
        """Lockstep: block until the client has answered the last published state (sim thread, before action()).

        Returns at once when lockstep is off, no client acknowledged a step within `lockstep_idle` seconds, or
        commands are ignored after a reset; otherwise after the acknowledgement and two newer commands on every
        live stream, or after a timeout (counted in lockstep_stats; the step then uses the newest command).
        """
        if not self.lockstep or self._published_tick is None:
            return
        target = self._published_tick
        with self._changed:
            now = time.monotonic()
            if self._ack is None or now - self._ack[1] > self.lockstep_idle or now < self._ignore_commands_until:
                self.lockstep_stats["free"] += 1
                return
            if not self._changed.wait_for(lambda: self._tick_reached(self._ack[0], target), self.lockstep_timeout):
                self.lockstep_stats["no_answer"] += 1
                return
            _, acked, counts = self._ack
            live = [key for key, value in self._commands.items() if acked - value[0] <= self.timeout]
            if self._changed.wait_for(
                lambda: all(self._received.get(key, 0) >= counts.get(key, 0) + 2 for key in live),
                self.lockstep_command_wait,
            ):
                self.lockstep_stats["answered"] += 1
            else:
                self.lockstep_stats["no_command"] += 1
        waits = sum(self.lockstep_stats.values())
        if waits % 1800 == 0:
            print(f"[G1-D SDK] Lockstep steps: {self.lockstep_stats}", flush=True)

    def take_reset_request(self):
        with self._lock:
            value, self._reset_requested = self._reset_requested, None
        return value

    def _init_motor_targets(self):
        """Cache named joint mappings and fallback gains; called on the sim thread."""
        self._motor_ids = [*self._arm_ids, self._yaw_id]
        self._default_kp = self.robot.data.default_joint_stiffness[:, self._motor_ids].clone()
        self._default_kd = self.robot.data.default_joint_damping[:, self._motor_ids].clone()
        self._motor_actuators = []
        for actuator in self.robot.actuators.values():
            ids = list(range(len(self.robot.joint_names)))[actuator.joint_indices] if isinstance(
                actuator.joint_indices, slice
            ) else list(actuator.joint_indices)
            pairs = [(local, self._motor_ids.index(joint)) for local, joint in enumerate(ids)
                     if joint in self._motor_ids]
            if pairs:
                if not actuator.is_implicit_model:
                    raise ValueError("G1-D SDK motor commands require implicit joint drives")
                local_ids, motor_ids = zip(*pairs)
                self._motor_actuators.append((actuator, list(local_ids), list(motor_ids)))
        self._last_kp = None
        self._last_kd = None

    def _init_stiction(self):
        """Friction of the arm joints as the position controller sees it (scene key robot.joint_stiction_nm:
        {joint suffix: breakaway torque in Nm}; aligned mode only).

        The real arm joints stop short of their target and then stand still while the target moves by up to
        0.02-0.03 rad (2026-11-27: the right shoulder pitch stops 0.019 rad short and stays at -0.849 rad for 3 s with
        the command 0.010-0.027 rad away, kp 80), then break away. PhysX joint friction (robot.joint_dynamics) lets
        the driven joint creep up to its target, and a policy whose targets lead the measured joints by the real
        arm's standing error then drifts in the sim. Model, per joint, with the breakaway torque F and the moving
        friction KINETIC_FRACTION * F:
          moving:   the drive pulls towards the target less the moving friction (target shifted by it / kp), so
                    the joint stops that far short, as with Coulomb friction;
          standing: a joint that has come to rest there sticks: the drive holds it where it stands until
                    kp * |target - that position| exceeds F."""
        robot = self.env.cfg.sorting["robot"] if hasattr(self.env.cfg, "sorting") else {}
        values = robot.get("joint_stiction_nm") or {}
        self._stiction = None
        if not values or not self.controller_alignment:
            return
        self._stiction = [float(values.get(name.split("_", 1)[1].rsplit("_joint", 1)[0], 0.0)) for name in ARM_JOINTS]
        self._stuck = [False] * len(ARM_JOINTS)
        self._stuck_q = [0.0] * len(ARM_JOINTS)
        self._still = [0] * len(ARM_JOINTS)
        # optional scene key robot.joint_stiction_kinetic_fraction: moving friction as a share of the breakaway torque
        self._kinetic_fraction = float(robot.get("joint_stiction_kinetic_fraction", self.KINETIC_FRACTION))
        print(f"[G1-D SDK] joint stiction [Nm]: {dict(zip(ARM_JOINTS, self._stiction))}, moving share {self._kinetic_fraction}", flush=True)

    STICK_SPEED = 0.02       # rad/s: slower than this for STICK_STEPS control steps counts as standing still
    STICK_STEPS = 2
    KINETIC_FRACTION = 0.5   # moving friction as a share of the breakaway torque (default)

    def _stick(self, arm, measured, arm_destinations):
        """The arm command with the joint friction model applied to its position targets (see _init_stiction)."""
        q, dq = list(arm.q), list(arm.dq)
        speed = self.robot.data.joint_vel[0, self._arm_ids].abs().cpu().tolist()
        position = measured[0, arm_destinations].cpu().tolist()
        for j, limit in enumerate(self._stiction):
            if limit <= 0 or not arm.active[j] or arm.kp[j] <= 0:
                self._stuck[j] = False
                continue
            kinetic = self._kinetic_fraction * limit
            error, shift = arm.q[j] - position[j], kinetic / arm.kp[j]
            pulled = arm.q[j] - math.copysign(shift, error)
            if self._stuck[j] and arm.kp[j] * abs(arm.q[j] - self._stuck_q[j]) > limit:
                self._stuck[j], self._still[j] = False, 0
            elif not self._stuck[j]:
                self._still[j] = self._still[j] + 1 if speed[j] < self.STICK_SPEED else 0
                # the controller's pull is below the moving friction (the joint cannot keep moving), or the joint is
                # at rest and what is left of the pull after the moving friction is below that friction
                if abs(error) <= shift or (self._still[j] >= self.STICK_STEPS and arm.kp[j] * abs(pulled - position[j]) < kinetic):
                    self._stuck[j], self._stuck_q[j] = True, position[j]
            q[j], dq[j] = (self._stuck_q[j], 0.0) if self._stuck[j] else (pulled, dq[j])
        return arm._replace(q=q, dq=dq)

    def _apply_motor_targets(self, command=None):
        """Use PhysX's single PD drive plus SDK feedforward, with no extra PD torque."""
        kp, kd = self._default_kp.clone(), self._default_kd.clone()
        dq, tau = torch.zeros_like(kp), torch.zeros_like(kp)
        if command is not None:
            for j, active in enumerate(command.active):
                if active:
                    kp[:, j], kd[:, j] = command.kp[j], command.kd[j]
                    dq[:, j] = command.weight * command.dq[j]
                    tau[:, j] = command.weight * command.tau[j]
        for name, values in (("kp", kp), ("kd", kd)):
            previous = getattr(self, f"_last_{name}")
            if previous is None or not torch.equal(previous, values):
                attribute = "stiffness" if name == "kp" else "damping"
                getattr(self.robot, f"write_joint_{attribute}_to_sim")(values, joint_ids=self._motor_ids)
                # Isaac Lab's write_*_to_sim does not update actuator-model buffers.
                # Keep these in sync so reported approximate torques use the same gains.
                for actuator, local_ids, motor_ids in self._motor_actuators:
                    getattr(actuator, attribute)[:, local_ids] = values[:, motor_ids]
                setattr(self, f"_last_{name}", values)
        self.robot.set_joint_velocity_target(dq, joint_ids=self._motor_ids)
        self.robot.set_joint_effort_target(tau, joint_ids=self._motor_ids)

    def reset(self):
        with self._lock:
            self._commands.clear()
            self._reset_requested = None
            self._ignore_commands_until = time.monotonic() + self.reset_hold
        self._held_action = joint_state(self.env).clone()
        self._held_yaw = self.robot.data.joint_pos[:, self._yaw_id].clone()
        self._fresh.clear()
        if getattr(self, "_stiction", None):
            self._stuck, self._still = [False] * len(ARM_JOINTS), [0] * len(ARM_JOINTS)
        if self.controller_alignment:
            self._last_kp = self._last_kd = None
            self._apply_motor_targets()
        self.env._g1d_torso_yaw_target = self._held_yaw.clone()
        self.update_state()

    def action(self):
        if self._error is not None:
            raise RuntimeError("G1-D state publisher failed") from self._error
        now = time.monotonic()
        with self._lock:
            commands = dict(self._commands)
        measured = joint_state(self.env)
        yaw_measured = self.robot.data.joint_pos[:, self._yaw_id]
        fresh = {key for key, value in commands.items() if now - value[0] <= self.timeout}
        if "arms" in fresh and commands["arms"][3] <= 0:
            fresh.remove("arms")
        arm_destinations = [*range(7), *range(8, 15)]
        # Latch once on timeout. Re-latching every step would let gravity cause drift.
        for key in self._fresh - fresh:
            if key == "arms":
                self._held_action[:, arm_destinations] = measured[:, arm_destinations]
                self._held_yaw = yaw_measured.clone()
            else:
                index = 7 if key == "left" else 15
                self._held_action[:, index] = measured[:, index]
        self._fresh = fresh
        action = self._held_action.clone()
        yaw = self._held_yaw.clone()
        arm = commands.get("arms")
        if "arms" in fresh and self._stiction:
            arm = self._stick(arm, measured, arm_destinations)
        if "arms" in fresh:
            q, active, weight = arm.q, arm.active, arm.weight
            for j, dest in enumerate(arm_destinations):
                if active[j]:
                    action[:, dest] = measured[:, dest] + weight * (q[j] - measured[:, dest])
            if active[14]:
                yaw = yaw_measured + weight * (q[14] - yaw_measured)
        for side, index in (("left", 7), ("right", 15)):
            command = commands.get(side)
            if side in fresh:
                action[:, index] = command[1]
        # Stale streams hold the measured position, independently for arms and each gripper.
        self.env._g1d_torso_yaw_target = yaw
        if self.controller_alignment:
            self._apply_motor_targets(arm if "arms" in fresh else None)
        return action

    def update_state(self):
        robot = self.robot
        q = robot.data.joint_pos[0].cpu().tolist()
        dq = robot.data.joint_vel[0].cpu().tolist()
        tau = robot.data.applied_torque[0].cpu().tolist()
        msg = unitree_hg_msg_dds__LowState_()
        msg.mode_machine = 0
        msg.tick = tick = int(self.env.common_step_counter * self.env.step_dt * 1000) % (2**32)
        for sdk_id, joint_id in zip([*range(15, 29), 12], [*self._arm_ids, self._yaw_id]):
            motor = msg.motor_state[sdk_id]
            motor.mode, motor.q, motor.dq, motor.tau_est = 1, q[joint_id], dq[joint_id], tau[joint_id]
        from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse
        quat = robot.data.body_quat_w[:, self._torso_id]
        msg.imu_state.quaternion = quat[0].cpu().tolist()  # SDK order: w, x, y, z
        msg.imu_state.rpy = [float(v[0]) for v in euler_xyz_from_quat(quat)]
        msg.imu_state.gyroscope = quat_apply_inverse(quat, robot.data.body_ang_vel_w[:, self._torso_id])[0].cpu().tolist()
        gravity = torch.tensor([[0., 0., 9.81]], device=self.env.device)
        msg.imu_state.accelerometer = quat_apply_inverse(quat, gravity)[0].cpu().tolist()
        msg.crc = self.crc.Crc(msg)
        messages = [msg]
        metres_per_sdk_unit = REAL_GAP_PER_RAW_M / 2  # finger travel per raw unit (both fingers move)
        for ids in (self._finger_ids[:2], self._finger_ids[2:]):
            motor = unitree_go_msg_dds__MotorState_()
            motor.q = float(finger_to_grip(torch.tensor(sum(q[i] for i in ids) / 2))) * self.SDK_GRIPPER_OPEN
            motor.dq = sum(dq[i] for i in ids) / (2 * metres_per_sdk_unit)
            motor.tau_est = sum(tau[i] for i in ids) * metres_per_sdk_unit
            message = MotorStates_()
            message.states.append(motor)
            messages.append(message)
        with self._lock:
            self._state = messages
            self._published_tick = tick
        if self.lockstep:
            self._write(messages)

    def _write(self, messages):
        with self._write_lock:
            for publisher, message in zip(self.publishers, messages):
                publisher.Write(message)

    def _publish_loop(self):
        try:
            while not self._stop.is_set():
                with self._lock:
                    messages = self._state
                self._write(messages)
                self._stop.wait(0.01)
        except Exception as exc:
            self._error = exc

    def close(self):
        if self.lockstep:
            print(f"[G1-D SDK] Lockstep steps: {self.lockstep_stats}", flush=True)
        self._stop.set()
        self._thread.join(timeout=2)
        for channel in self.subscribers + self.publishers:
            channel.Close()
