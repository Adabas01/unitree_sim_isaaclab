"""G1-D SDK bridge using the existing G1/Dex1 wire protocol.

Callbacks only copy commands. All physics access happens on the simulation
thread; a separate 100 Hz publisher sends immutable state snapshots.
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

from robots.g1d_joints import ARM_JOINTS, FINGER_JOINTS, GRIPPER_CLOSED, GRIPPER_OPEN, joint_state


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
    SDK_GRIPPER_OPEN = 5.4

    def __init__(self, env, timeout=0.5, reset_hold=1.1, controller_alignment=False):
        if env.num_envs != 1:
            raise ValueError("One DDS robot requires exactly one simulation environment")
        if not math.isfinite(reset_hold) or reset_hold < 0:
            raise ValueError("reset_hold must be finite and nonnegative")
        self.env = env
        self.robot = env.scene["robot"]
        self.timeout = timeout
        self.reset_hold = reset_hold
        self.controller_alignment = controller_alignment
        self._lock = threading.Lock()
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
        self.update_state()
        self._thread = threading.Thread(target=self._publish_loop, name="g1d_state", daemon=True)
        self._thread.start()
        print(
            f"[G1-D SDK] Domain 1; arm slots 15–28, torso yaw 12; Dex1 0=closed, "
            f"5.4=open; controller={'aligned' if self.controller_alignment else 'legacy position-only'}; "
            f"post-reset command hold={self.reset_hold:.2f}s",
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
        except (ValueError, IndexError, TypeError):
            self.rejected_commands += 1

    def _reset_command(self, msg):
        if msg.data in ("1", "2"):
            with self._lock:
                self._reset_requested = msg.data

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
        msg.tick = int(self.env.common_step_counter * self.env.step_dt * 1000) % (2**32)
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
        metres_per_sdk_unit = (GRIPPER_OPEN - GRIPPER_CLOSED) / self.SDK_GRIPPER_OPEN
        for ids in (self._finger_ids[:2], self._finger_ids[2:]):
            motor = unitree_go_msg_dds__MotorState_()
            motor.q = max(0.0, min(self.SDK_GRIPPER_OPEN, (sum(q[i] for i in ids) / 2 - GRIPPER_CLOSED) / metres_per_sdk_unit))
            motor.dq = sum(dq[i] for i in ids) / (2 * metres_per_sdk_unit)
            motor.tau_est = sum(tau[i] for i in ids) * metres_per_sdk_unit
            message = MotorStates_()
            message.states.append(motor)
            messages.append(message)
        with self._lock:
            self._state = messages

    def _publish_loop(self):
        try:
            while not self._stop.is_set():
                with self._lock:
                    messages = self._state
                for publisher, message in zip(self.publishers, messages):
                    publisher.Write(message)
                self._stop.wait(0.01)
        except Exception as exc:
            self._error = exc

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        for channel in self.subscribers + self.publishers:
            channel.Close()
