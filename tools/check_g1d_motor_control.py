"""Bounded PhysX check of G1-D SDK motor fields; opens no DDS channels.

Run in unitree_sim_env: python tools/check_g1d_motor_control.py --headless --device cpu
Gravity and self-contact are disabled only in this isolated actuator check.
"""

import argparse
import json
from pathlib import Path
import sys
import threading
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
import pinocchio  # noqa: F401 -- this workspace requires import before Kit.

launcher = AppLauncher(args)
app = launcher.app
exit_code = 0
try:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    from robots.g1d import robot_config
    from robots.g1d_joints import ARM_JOINTS, joint_state
    from dds.g1d_bridge import G1DBridge

    dt = 1 / 120
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        device=args.device, dt=dt, gravity=(0.0, 0.0, 0.0),
    ))
    settings = json.loads((ROOT / "tasks/g1d_tasks/sort_shapes/configs/scene.json").read_text())["robot"]
    settings["controller_alignment"] = True
    settings["collisions"]["self_collisions"] = False
    cfg = robot_config(settings)
    cfg.prim_path = "/World/Robot"
    robot = Articulation(cfg)
    sim.reset()
    robot.update(dt)
    env = SimpleNamespace(scene={"robot": robot})

    # Exercise the production decoder and physics application without DDS transport.
    bridge = object.__new__(G1DBridge)
    bridge.env, bridge.robot = env, robot
    bridge.timeout, bridge.reset_hold = 0.5, 1.1
    bridge.controller_alignment = True
    bridge._lock = threading.Lock()
    bridge._commands, bridge._fresh = {}, set()
    bridge._error, bridge._ignore_commands_until = None, 0.0
    bridge.accepted_commands = bridge.rejected_commands = 0
    bridge._arm_ids = [robot.joint_names.index(n) for n in ARM_JOINTS]
    bridge._yaw_id = robot.joint_names.index("torso_Joint")
    bridge._held_action = joint_state(env).clone()
    bridge._held_yaw = robot.data.joint_pos[:, bridge._yaw_id].clone()
    bridge._init_motor_targets()
    bridge.update_state = lambda: None

    expected_kp = [80.0] * 4 + [40.0] * 3
    expected_kd = [3.0] * 4 + [1.5] * 3
    torch.testing.assert_close(bridge._default_kp.cpu(), torch.tensor([expected_kp * 2 + [300.0]]))
    torch.testing.assert_close(bridge._default_kd.cpu(), torch.tensor([expected_kd * 2 + [3.0]]))
    expected_effort = [25.0] * 5 + [5.0] * 2
    expected_velocity = [37.0] * 5 + [22.0] * 2
    torch.testing.assert_close(
        robot.root_physx_view.get_dof_max_forces()[:, bridge._motor_ids].cpu(),
        torch.tensor([expected_effort * 2 + [30.0]]),
    )
    torch.testing.assert_close(
        robot.root_physx_view.get_dof_max_velocities()[:, bridge._motor_ids].cpu(),
        torch.tensor([expected_velocity * 2 + [5.0]]),
    )

    msg = unitree_hg_msg_dds__LowCmd_()
    for slot, joint in zip([*range(15, 29), 12], bridge._motor_ids):
        motor = msg.motor_cmd[slot]
        motor.mode = 1
        motor.q = float(robot.data.default_joint_pos[0, joint])
        motor.kp = float(robot.data.default_joint_stiffness[0, joint])
        motor.kd = float(robot.data.default_joint_damping[0, joint])

    wrist = robot.joint_names.index("left_wrist_yaw_joint")

    def reset_pose():
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()
        robot.update(dt)

    def step(count):
        for _ in range(count):
            msg.crc = CRC().Crc(msg)
            bridge._arm_command(msg, "rt/lowcmd")
            action = bridge.action()
            targets = robot.data.default_joint_pos.clone()
            targets[:, bridge._arm_ids] = torch.cat((action[:, :7], action[:, 8:15]), dim=1)
            targets[:, bridge._yaw_id] = env._g1d_torso_yaw_target
            robot.set_joint_position_target(targets)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            assert torch.isfinite(robot.data.joint_pos).all()

    with torch.inference_mode():
        msg.motor_cmd[21].q += 0.15
        step(120)
        assert abs(float(robot.data.joint_pos[0, wrist]) - msg.motor_cmd[21].q) < 0.03
        torch.testing.assert_close(robot.root_physx_view.get_dof_stiffnesses()[:, bridge._motor_ids].cpu(),
                                   bridge._default_kp.cpu())
        print("POSITION_AND_GAINS_OK", flush=True)

        reset_pose()
        msg.motor_cmd[21].kp, msg.motor_cmd[21].kd = 0.0, 1.5
        msg.motor_cmd[21].dq = 0.2
        step(60)
        assert float(robot.data.joint_vel[0, wrist]) > 0.1
        assert float(robot.root_physx_view.get_dof_stiffnesses()[0, wrist]) == 0.0
        print("VELOCITY_TARGET_OK", flush=True)

        reset_pose()
        msg.motor_cmd[21].kd = msg.motor_cmd[21].dq = 0.0
        msg.motor_cmd[21].tau = 0.1
        step(12)
        assert float(robot.data.joint_vel[0, wrist]) > 0.05
        assert abs(float(robot.data.applied_torque[0, wrist]) - 0.1) < 1e-5
        print("FEEDFORWARD_WITH_ZERO_PD_OK", flush=True)

        bridge._commands["arms"] = bridge._commands["arms"]._replace(timestamp=time.monotonic() - 1.0)
        bridge.action()
        assert float(robot.data.joint_effort_target[0, wrist]) == 0.0
        assert float(robot.data.joint_vel_target[0, wrist]) == 0.0
        assert float(robot.root_physx_view.get_dof_stiffnesses()[0, wrist]) == 40.0
        bridge.reset()
        assert not bridge._commands
        print("TIMEOUT_AND_RESET_OK", flush=True)
    print("G1D_MOTOR_CONTROL_CHECK_PASSED", flush=True)
except Exception:
    exit_code = 1
    traceback.print_exc()
finally:
    # No render/Replicator jobs or DDS channels need draining. Immediate Kit
    # shutdown avoids initializing the renderer during CPU-only stage teardown.
    import omni.kit.app
    omni.kit.app.get_app().post_quit(exit_code)
    app.close(skip_cleanup=True)
raise SystemExit(exit_code)
