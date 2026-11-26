"""Run the G1-D sorting task, or a task spec (--task-spec), in preview or Unitree SDK control mode."""

import argparse
import json
import os
from pathlib import Path
import time
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--scene-config",
    type=Path,
    default=Path(__file__).parent / "tasks/g1d_tasks/sort_shapes/configs/scene.json",
)
parser.add_argument(
    "--evaluation-config",
    type=Path,
    default=Path(__file__).parent / "tasks/g1d_tasks/sort_shapes/configs/evaluation.json",
)
parser.add_argument(
    "--task-spec",
    help="Run a task spec instead of the sorting task: a name under tasks/g1d_tasks/spec_tasks/specs/ "
         "(e.g. stack_rgy_blocks) or a YAML path; same workspace, robot, cameras and DDS. Records go to "
         "logs/g1d_spec_tasks/<task>.jsonl unless --evaluation-output is given",
)
parser.add_argument(
    "--spec-placeholders", action="store_true",
    help="Replace a spec's registered container assets (not loadable yet) by open boxes of their bounding box",
)
parser.add_argument(
    "--perturbation", choices=("Default", "VB-POSE"), default="Default",
    help="Task specs only: object layout per reset. Default = the spec's poses every time; VB-POSE = the main objects "
         "re-placed at random in the reachable zone, as chiral's VB-POSE (tasks/g1d_tasks/spec_tasks/mdp.py)",
)
parser.add_argument("--perturb-seed", type=int, default=0,
                    help="Seed of the --perturbation draws (draw k of a run uses numpy default_rng([seed, k]))")
parser.add_argument(
    "--table-height", type=float,
    help="Table top height in metres (default: the scene's table.height). The torso origin is printed at start; "
         "the G1_Dex1 recordings had the table ~0.03 m above it, the g1d-sort recordings ~0.10 m below",
)
parser.add_argument(
    "--table-x", type=float,
    help="Table centre x in metres (default: the scene's table.center_xy[0]); move the table away when a raised "
         "table would touch the robot's body (the torso origin printed at start then changes)",
)
parser.add_argument(
    "--start-pose",
    help="Arm start pose after every reset: 14 comma-separated joint angles [rad], left arm then right arm, as the "
         "deploy/ clients' --home (write --start-pose=-0.25,...); default: the scene's robot.arm_positions",
)
parser.add_argument("--evaluation-output", type=Path, help="Override the JSONL evaluation output path")
parser.add_argument("--evaluation-run-label", help="Label this run, for example the policy checkpoint name")
parser.add_argument("--episode-timeout", type=float, help="Override episode timeout in simulated seconds")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=0, help="Stop after N control steps; 0 runs until the window closes")
parser.add_argument("--reset-every", type=float, default=0, help="Randomize the scene every N simulated seconds")
parser.add_argument("--capture-dir", type=Path, help="Save all four camera images after 60 control steps")
parser.add_argument("--capture-steps", help="With --capture-dir: save the images at these control steps instead "
                    "(comma-separated, e.g. 60,240,420), each in <capture-dir>/step_<n>/")
parser.add_argument("--sdk", action="store_true", help="Enable G1 arm/torso-yaw and Dex1 DDS control on domain 1")
parser.add_argument("--dds-interface", default=None, help="DDS network interface; default uses SDK autodetection")
parser.add_argument("--command-timeout", type=float, default=0.5, help="Hold measured joint positions after this many seconds without commands")
parser.add_argument("--reset-command-hold", type=float, default=1.1, help="Ignore SDK commands for this many seconds after an episode reset")
parser.add_argument(
    "--command-timing", choices=("free", "lockstep"), default="lockstep",
    help="free: each control step uses the newest command that has arrived, so commands act one control step "
         "later than on the real robot; lockstep: wait for the client's answer to each published state "
         "(clients from master-thesis deploy/ acknowledge each step). See dds/g1d_bridge.py",
)
parser.add_argument("--stream-cameras", action="store_true", help="Publish head/wrist images through the existing teleimager server")
parser.add_argument("--camera-profile", help="Camera set from the scene's cameras.profiles (e.g. manual, calibrated)")
parser.add_argument("--show-colliders", action="store_true", help="Show PhysX collision shapes in the interactive viewport")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (args.steps < 0 or args.reset_every < 0 or not 0 < args.command_timeout < float("inf")
        or not 0 <= args.reset_command_hold < float("inf")
        or (args.episode_timeout is not None and not 0 < args.episode_timeout < float("inf"))):
    parser.error("Steps/reset intervals must be nonnegative and command timings must be finite")
if not args.enable_cameras:
    parser.error("This task needs --enable_cameras")
os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parent)

# Match this workspace's required Pinocchio/Isaac library import order.
import pinocchio  # noqa: F401, E402

launcher = AppLauncher(args)
app = launcher.app
env = None
bridge = None
image_server = None
image_writer = None
exit_code = 0
try:
    import gymnasium as gym
    import torch
    from PIL import Image
    if args.show_colliders:
        import carb
        carb.settings.get_settings().set_int("/persistent/physics/visualizationDisplayColliders", 2)
    if args.sdk:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(1, args.dds_interface)
    import tasks  # noqa: F401
    from tasks.g1d_tasks.sort_shapes.layout import load_settings
    from tasks.g1d_tasks.sort_shapes.mdp import camera_rgb, joint_state
    from tasks.g1d_tasks.sort_shapes.evaluation import load_evaluation_settings

    evaluation = load_evaluation_settings(args.evaluation_config)
    if args.episode_timeout is not None:
        evaluation["episode_timeout_s"] = args.episode_timeout
    if args.evaluation_run_label is not None:
        evaluation["logging"]["run_label"] = args.evaluation_run_label
    settings = load_settings(args.scene_config, args.camera_profile)
    if args.table_height is not None:
        if not settings["table"]["thickness"] < args.table_height < 2.0:
            parser.error("--table-height must exceed the table thickness and be below 2 m")
        settings["table"]["height"] = args.table_height
    if args.table_x is not None:
        settings["table"]["center_xy"][0] = args.table_x
    if args.start_pose is not None:
        try:
            start_pose = [float(v) for v in args.start_pose.split(",")]
        except ValueError:
            parser.error("--start-pose takes 14 comma-separated numbers")
        if len(start_pose) != 14:
            parser.error(f"--start-pose needs 14 joint angles, got {len(start_pose)}")
        settings["robot"]["arm_positions"] = start_pose
    task_spec = None
    if args.perturbation != "Default" and not args.task_spec:
        parser.error("--perturbation needs --task-spec")
    if args.task_spec:
        from tasks.g1d_tasks.spec_tasks.spec import load_spec
        task_spec = load_spec(args.task_spec, placeholders=args.spec_placeholders)
        evaluation["logging"]["path"] = f"logs/g1d_spec_tasks/{task_spec['name']}.jsonl"
    if args.evaluation_output:
        evaluation["logging"]["path"] = str(args.evaluation_output.resolve())
    if task_spec:
        from tasks.g1d_tasks.spec_tasks.env_cfg import G1DSpecEnvCfg
        from tasks.g1d_tasks.spec_tasks.mdp import reset_objects
        task_id = "Isaac-Spec-Task-G1D-Dex1-Joint"
        cfg = G1DSpecEnvCfg(sorting=settings, evaluation=evaluation, task_spec=task_spec,
                            perturbation={"kind": args.perturbation, "seed": args.perturb_seed})
    else:
        from tasks.g1d_tasks.sort_shapes.env_cfg import G1DSortEnvCfg
        from tasks.g1d_tasks.sort_shapes.mdp import reset_objects
        task_id = "Isaac-Sort-Shapes-G1D-Dex1-Joint"
        cfg = G1DSortEnvCfg(sorting=settings, evaluation=evaluation)
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = gym.make(task_id, cfg=cfg).unwrapped
    env.reset(seed=args.seed)

    def reset_state_summary():
        position_values = []
        for name, item in env.scene.rigid_objects.items():
            position = item.data.root_pos_w[0] - env.scene.env_origins[0]
            xyz = position.cpu().tolist()
            position_values.append(f"{name}=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")
        robot = env.scene["robot"]
        pose_error = (robot.data.joint_pos - robot.data.default_joint_pos).abs().max().item()
        return ", ".join(position_values) if position_values else "no objects", pose_error

    hold = joint_state(env).clone()
    if args.sdk:
        from dds.g1d_bridge import G1DBridge
        bridge = G1DBridge(
            env,
            timeout=args.command_timeout,
            reset_hold=args.reset_command_hold,
            controller_alignment=cfg.sorting["robot"]["controller_alignment"],
            lockstep=args.command_timing == "lockstep",
        )
    # Recorded with every evaluation episode; None without SDK control.
    env.g1d_command_timing = args.command_timing if args.sdk else None
    if args.stream_cameras:
        from tools.shared_memory_utils import MultiImageWriter
        from teleimager.image_server import run_isaacsim_server
        image_writer = MultiImageWriter(enable_jpeg=True)
        image_writer.write_images({
            key: camera_rgb(env, name)[0].cpu().numpy()
            for key, name in (("head", "front_camera"), ("left", "left_wrist_camera"), ("right", "right_wrist_camera"))
        })
        image_server = run_isaacsim_server()
    reset_interval = max(1, round(args.reset_every / env.step_dt)) if args.reset_every else 0
    count = 0
    episode_number = 1
    if task_spec:
        print(f"G1-D task spec {task_spec['name']!r} ({task_spec['spec_file']}, hash {task_spec['spec_hash']}), "
              f"prompt: {task_spec['prompt']!r}; perturbation {args.perturbation}"
              + (f" seed {args.perturb_seed}" if args.perturbation != "Default" else ""), flush=True)
    print("G1-D scene ready. Cameras: front_camera, left_wrist_camera, right_wrist_camera, world_camera; "
          f"camera profile {cfg.sorting['cameras']['profile']!r}; controller_alignment "
          f"{cfg.sorting['robot']['controller_alignment']}; command timing {env.g1d_command_timing}.", flush=True)
    if args.show_colliders:
        print("Collider visualization enabled in the interactive viewport.", flush=True)
    if evaluation["logging"]["enabled"]:
        print(f"Evaluation output: {env.g1d_evaluation_recorder.output_path}", flush=True)
    print("SDK control enabled; waiting for commands." if bridge else "Preview holds the starting pose; add --sdk for SDK control.", flush=True)
    initial_positions, initial_pose_error = reset_state_summary()
    print(
        f"[G1-D evaluation] Episode 1 start: robot_pose_error={initial_pose_error:.6f}, "
        f"{initial_positions}",
        flush=True,
    )
    with torch.inference_mode():
        while app.is_running() and (args.steps == 0 or count < args.steps):
            start = time.monotonic()
            if env.sim.is_stopped():
                break
            if not env.sim.is_playing():
                env.sim.render()
                continue
            if bridge:
                bridge.wait_for_reply()
            request = bridge.take_reset_request() if bridge else None
            if request == "1":
                env.g1d_evaluation_recorder.write_pending("objects_reset")
                reset_objects(env, None)
                env.g1d_evaluation.reset()
                env.episode_length_buf.zero_()
            elif request == "2":
                env.reset()
                bridge.reset()
            _, _, terminated, truncated, _ = env.step(bridge.action() if bridge else hold)
            done = terminated | truncated
            if done.any():
                reasons = []
                if terminated.any():
                    reasons.append("success")
                if truncated.any():
                    reasons.append("timeout")
                counters_reset = not bool(env.episode_length_buf[done].any().item()) and not bool(
                    env.g1d_evaluation.steps[done].any().item()
                )
                positions, robot_pose_error = reset_state_summary()
                automatic_reset_ok = counters_reset and robot_pose_error <= 1e-5
                if not automatic_reset_ok:
                    env.reset()
                    positions, robot_pose_error = reset_state_summary()
                if bridge:
                    bridge.reset()
                episode_number += 1
                print(
                    f"[G1-D evaluation] Episode {episode_number - 1} ended "
                    f"({'+'.join(reasons)}); episode {episode_number} reset "
                    f"{'verified' if automatic_reset_ok else 'with explicit fallback'}, "
                    f"robot_pose_error={robot_pose_error:.6f}: {positions}",
                    flush=True,
                )
            # Images first: in lockstep the client answers as soon as the state is out.
            if image_writer:
                image_writer.write_images({
                    key: camera_rgb(env, name)[0].cpu().numpy()
                    for key, name in (("head", "front_camera"), ("left", "left_wrist_camera"), ("right", "right_wrist_camera"))
                })
            if bridge:
                bridge.update_state()
            count += 1
            if not torch.isfinite(env.scene["robot"].data.joint_pos).all():
                raise RuntimeError("Non-finite robot state")
            # --capture-dir: the views at step 60 (or the last step); with --reset-every, of every episode at its
            # step min(30, interval - 1), in <capture-dir>/episode_<n>/
            capture_to = None
            if args.capture_dir and args.capture_steps:
                if str(count) in args.capture_steps.split(","):
                    capture_to = args.capture_dir / f"step_{count:05d}"
            elif args.capture_dir and not reset_interval and count == min(60, args.steps or 60):
                capture_to = args.capture_dir
            elif args.capture_dir and reset_interval and count % reset_interval == min(30, reset_interval - 1):
                capture_to = args.capture_dir / f"episode_{count // reset_interval + 1:03d}"
            if capture_to:
                capture_to.mkdir(parents=True, exist_ok=True)
                for name in ("front_camera", "left_wrist_camera", "right_wrist_camera", "world_camera"):
                    rgb = camera_rgb(env, name)[0].cpu().numpy()
                    Image.fromarray(rgb).save(capture_to / f"{name}.png")
                manifest = dict(
                    task=task_id, task_spec=task_spec and task_spec["spec_file"], seed=args.seed, control_step=count,
                    step_dt=env.step_dt, settings=cfg.sorting, evaluation=cfg.evaluation,
                    object_poses={name: obj.data.root_state_w[:, :7].cpu().tolist()
                                  for name, obj in env.scene.rigid_objects.items()},
                    layout=(getattr(env, "spec_layout", None) or [None])[0],
                    hand_links_w={name: env.scene["robot"].data.body_pos_w[0, env.scene["robot"].body_names.index(name)].cpu().tolist()
                                  for name in env.scene["robot"].body_names if "dex1" in name},
                )
                (capture_to / "scene.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"Saved camera views to {capture_to.resolve()}", flush=True)
            if reset_interval and count % reset_interval == 0:
                env.reset()
                if bridge:
                    bridge.reset()
                print(f"Reset at step {count}", flush=True)
            if not args.headless or args.sdk:
                time.sleep(max(0.0, env.step_dt - (time.monotonic() - start)))
    print(f"Completed {count} G1-D control steps.", flush=True)
except KeyboardInterrupt:
    print("Stopping G1-D preview.", flush=True)
except Exception:
    exit_code = 1
    traceback.print_exc()
finally:
    if env is not None and hasattr(env, "g1d_evaluation_recorder"):
        env.g1d_evaluation_recorder.write_pending("closed")
    if bridge is not None:
        bridge.close()
    if image_server is not None:
        image_server.stop()
        image_server.wait()
    if image_writer is not None:
        image_writer.close()
    if env is not None:
        env.close()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(exit_code)
    app.close()
raise SystemExit(exit_code)
