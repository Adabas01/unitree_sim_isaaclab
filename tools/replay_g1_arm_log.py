"""Replay recorded arm motor commands on simulation time; never opens DDS.

Uses the G1-D articulation with gravity and self-collision, a fixed chassis, and
locked non-arm joints. No scene objects, cameras, policy, or real robot channels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "unitree_lerobot/unitree_lerobot/eval_robot"))

import numpy as np
from arm_comparison import JOINTS, Trial, interpolate, load_trial, positive


def run(args, source):
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from robots.g1d import robot_config
    from robots.g1d_joints import FINGER_JOINTS

    dt = args.physics_dt
    scene = json.loads(args.scene.read_text())
    settings = scene["robot"]
    settings["controller_alignment"] = args.controller == "aligned"
    fixed = source.metadata.get("fixed_configuration", {})
    for name in ("torso_pitch", "torso_yaw", "lift_positions"):
        if name in fixed:
            settings[name] = fixed[name]
    q0 = interpolate(np.array([0.]), source.state_t, source.state_q)[0]
    dq0 = interpolate(np.array([0.]), source.state_t, source.state_dq)[0]
    settings["arm_positions"] = q0.tolist()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, dt=dt))
    cfg = robot_config(settings)
    cfg.prim_path = "/World/Robot"
    cfg.init_state.joint_pos.update({name: fixed.get("gripper_position", .0245) for name in FINGER_JOINTS})
    cfg.init_state.joint_vel = {f"{name}_joint": float(value) for name, value in zip(JOINTS, dq0)}
    robot = Articulation(cfg)
    sim.reset()
    robot.update(dt)
    arm_ids = [robot.joint_names.index(f"{name}_joint") for name in JOINTS]
    fixed_ids = [i for i in range(robot.num_joints) if i not in arm_ids]
    fixed_q = robot.data.default_joint_pos[:, fixed_ids].clone()
    # Physical constraints, not repeatedly overwriting the state after stepping.
    robot.write_joint_position_limit_to_sim(torch.stack((fixed_q, fixed_q), dim=-1), joint_ids=fixed_ids)
    initial = robot.data.default_joint_pos.clone()
    initial_dq = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(initial, initial_dq)
    robot.set_joint_position_target(initial)
    robot.update(dt)
    if args.armature is not None:
        robot.write_joint_armature_to_sim(torch.full((1, 14), args.armature, device=args.device), joint_ids=arm_ids)
    if args.friction is not None:
        robot.write_joint_friction_coefficient_to_sim(torch.full((1, 14), args.friction, device=args.device), joint_ids=arm_ids)
    actuator_mapping = []
    for actuator in robot.actuators.values():
        ids = list(range(robot.num_joints))[actuator.joint_indices] if isinstance(actuator.joint_indices, slice) else list(actuator.joint_indices)
        pairs = [(i, arm_ids.index(j)) for i, j in enumerate(ids) if j in arm_ids]
        if pairs:
            local, selected = zip(*pairs)
            actuator_mapping.append((actuator, list(local), list(selected)))

    applied = {name: [] for name in ("t", "q", "dq", "kp", "kd", "tau")}
    state = {name: [] for name in ("t", "q", "dq", "tau")}
    source_indices = []
    last_gains = None
    max_fixed_drift = 0.
    end = min(source.command_t[-1], source.state_t[-1])
    if args.max_duration is not None:
        end = min(end, args.max_duration)
    count = int(np.floor(end / dt))
    if count < 2:
        raise ValueError("Recording is shorter than two physics steps")
    def record(t):
        nonlocal max_fixed_drift
        state["t"].append(t)
        for name, values in (("q", robot.data.joint_pos), ("dq", robot.data.joint_vel), ("tau", robot.data.applied_torque)):
            a = values[0, arm_ids].cpu().numpy().copy()
            if not np.isfinite(a).all():
                raise RuntimeError("Nonfinite simulation state")
            state[name].append(a)
        max_fixed_drift = max(max_fixed_drift, float((robot.data.joint_pos[:, fixed_ids] - fixed_q).abs().max()))

    with torch.inference_mode():
        record(0.)
        index = 0
        previous_index = -1
        for step in range(count):
            t = step * dt
            while index + 1 < len(source.command_t) and source.command_t[index + 1] <= t:
                index += 1
            if source.command_t[index] > t:
                raise ValueError("First command must have timestamp zero")
            if index != previous_index:
                packet = {name: torch.tensor(getattr(source, f"command_{name}")[index:index + 1],
                                            dtype=torch.float32, device=args.device)
                          for name in ("q", "dq", "kp", "kd", "tau")}
                if args.controller == "legacy":
                    packet.update(kp=robot.data.default_joint_stiffness[:, arm_ids].clone(),
                                  kd=robot.data.default_joint_damping[:, arm_ids].clone(),
                                  dq=torch.zeros_like(packet["dq"]), tau=torch.zeros_like(packet["tau"]))
                if last_gains is None or not all(torch.equal(packet[n], last_gains[n]) for n in ("kp", "kd")):
                    robot.write_joint_stiffness_to_sim(packet["kp"], joint_ids=arm_ids)
                    robot.write_joint_damping_to_sim(packet["kd"], joint_ids=arm_ids)
                    for actuator, local, selected in actuator_mapping:
                        actuator.stiffness[:, local] = packet["kp"][:, selected]
                        actuator.damping[:, local] = packet["kd"][:, selected]
                    last_gains = {n: packet[n].clone() for n in ("kp", "kd")}
                robot.set_joint_position_target(packet["q"], joint_ids=arm_ids)
                robot.set_joint_velocity_target(packet["dq"], joint_ids=arm_ids)
                robot.set_joint_effort_target(packet["tau"], joint_ids=arm_ids)
                applied["t"].append(t)
                for name in ("q", "dq", "kp", "kd", "tau"):
                    applied[name].append(packet[name][0].cpu().numpy().copy())
                source_indices.append(index)
                previous_index = index
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            record((step + 1) * dt)
            if step % max(1, int(5 / dt)) == 0:
                print(f"Replay {t:.2f}/{end:.2f} simulated seconds", flush=True)
    if max_fixed_drift > .001:
        raise RuntimeError(f"Non-arm joint constraint drift {max_fixed_drift:.6f} exceeds tolerance")
    quantization = np.array(applied["t"]) - source.command_t[source_indices]
    metadata = dict(source.metadata, status="complete", backend="Isaac Lab implicit PhysX drives",
                    controller=args.controller, physics_dt=dt, robot_settings=settings,
                    source_file=str(args.input.resolve()),
                    source_command_count=len(source.command_t), applied_command_count=len(source_indices),
                    max_command_quantization_s=float(max(quantization)),
                    skipped_command_count=int(source_indices[-1] + 1 - len(source_indices)),
                    max_fixed_joint_drift=max_fixed_drift,
                    armature_override=args.armature, friction_override=args.friction,
                    replay_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    source_recording_sha256=source.metadata.get("source_sha256"))
    arrays = {f"command_{name}": np.array(value) for name, value in applied.items()}
    arrays.update({f"state_{name}": np.array(value) for name, value in state.items()})
    trial = Trial(**arrays, desired_t=source.desired_t.copy(), desired_q=source.desired_q.copy(), metadata=metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trial.save(args.output)
    print(f"Saved {args.output}; maximum timestamp quantization {max(quantization) * 1000:.3f} ms; "
          f"skipped {metadata['skipped_command_count']} commands", flush=True)


def main():
    # Parse/validate input before launching Kit or creating output.
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Real trial.npz or recorder JSONL")
    parser.add_argument("--output", type=Path, required=True, help="New simulated .npz file")
    parser.add_argument("--command-topic", choices=("rt/lowcmd", "rt/arm_sdk"))
    parser.add_argument("--scene", type=Path, default=ROOT / "tasks/g1d_tasks/sort_shapes/configs/scene.json")
    parser.add_argument("--physics-dt", type=float, default=.002)
    parser.add_argument("--max-gap", type=float, default=.1, help="Reject larger input/state gaps")
    parser.add_argument("--controller", choices=("aligned", "legacy"), default="aligned")
    parser.add_argument("--armature", type=float, help="Optional shared armature override for all arm joints")
    parser.add_argument("--friction", type=float, help="Optional shared static joint friction override")
    parser.add_argument("--max-duration", type=float, help="Limit replay duration, e.g. for a smoke test")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    for name in ("physics_dt", "max_gap", "max_duration"):
        if getattr(args, name) is not None:
            positive(getattr(args, name), name)
    for name in ("armature", "friction"):
        if getattr(args, name) is not None:
            positive(getattr(args, name), name, allow_zero=True)
    if args.output.exists():
        parser.error("Output exists; choose a new filename")
    source = load_trial(args.input, args.command_topic)
    if source.command_t[0] != 0 or not source.state_t[0] <= 0 <= source.state_t[-1]:
        parser.error("Input needs first command at t=0 and measured state bracketing t=0")
    if max(np.diff(source.command_t)) > args.max_gap or max(np.diff(source.state_t)) > args.max_gap:
        parser.error("Input has a command/state gap exceeding --max-gap")
    import pinocchio  # noqa: F401 -- load before Kit in this workspace
    launcher = AppLauncher(args)
    code = 0
    try:
        run(args, source)
    except Exception:
        code = 1
        traceback.print_exc()
    finally:
        import omni.kit.app
        omni.kit.app.get_app().post_quit(code)
        launcher.app.close(skip_cleanup=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
