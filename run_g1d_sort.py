"""Run the G1-D sorting task in preview or Unitree SDK control mode."""

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
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=0, help="Stop after N control steps; 0 runs until the window closes")
parser.add_argument("--reset-every", type=float, default=0, help="Randomize the scene every N simulated seconds")
parser.add_argument("--capture-dir", type=Path, help="Save all four camera images after 60 control steps")
parser.add_argument("--sdk", action="store_true", help="Enable G1 arm/torso-yaw and Dex1 DDS control on domain 1")
parser.add_argument("--dds-interface", default=None, help="DDS network interface; default uses SDK autodetection")
parser.add_argument("--command-timeout", type=float, default=0.5, help="Hold measured joint positions after this many seconds without commands")
parser.add_argument("--stream-cameras", action="store_true", help="Publish head/wrist images through the existing teleimager server")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 0 or args.reset_every < 0 or not 0 < args.command_timeout < float("inf"):
    parser.error("Steps/reset interval must be nonnegative and command timeout must be finite and positive")
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
    if args.sdk:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(1, args.dds_interface)
    import tasks  # noqa: F401
    from tasks.g1d_tasks.sort_shapes.layout import load_settings
    from tasks.g1d_tasks.sort_shapes.mdp import joint_state
    from tasks.g1d_tasks.sort_shapes.env_cfg import G1DSortEnvCfg

    cfg = G1DSortEnvCfg(sorting=load_settings(args.scene_config))
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = gym.make("Isaac-Sort-Shapes-G1D-Dex1-Joint", cfg=cfg).unwrapped
    env.reset(seed=args.seed)
    hold = joint_state(env).clone()
    if args.sdk:
        from dds.g1d_bridge import G1DBridge
        bridge = G1DBridge(env, timeout=args.command_timeout)
    if args.stream_cameras:
        from tools.shared_memory_utils import MultiImageWriter
        from teleimager.image_server import run_isaacsim_server
        image_writer = MultiImageWriter(enable_jpeg=True)
        image_writer.write_images({
            key: env.scene[name].data.output["rgb"][0, ..., :3].cpu().numpy()
            for key, name in (("head", "front_camera"), ("left", "left_wrist_camera"), ("right", "right_wrist_camera"))
        })
        image_server = run_isaacsim_server()
    reset_interval = max(1, round(args.reset_every / env.step_dt)) if args.reset_every else 0
    count = 0
    print("G1-D sorting scene ready. Cameras: front_camera, left_wrist_camera, right_wrist_camera, world_camera.", flush=True)
    print("SDK control enabled; waiting for commands." if bridge else "Preview holds the starting pose; add --sdk for SDK control.", flush=True)
    with torch.inference_mode():
        while app.is_running() and (args.steps == 0 or count < args.steps):
            start = time.monotonic()
            if env.sim.is_stopped():
                break
            if not env.sim.is_playing():
                env.sim.render()
                continue
            request = bridge.take_reset_request() if bridge else None
            if request == "1":
                from tasks.g1d_tasks.sort_shapes.mdp import reset_objects
                reset_objects(env, None)
            elif request == "2":
                env.reset()
                bridge.reset()
            _, _, terminated, truncated, _ = env.step(bridge.action() if bridge else hold)
            if bridge:
                if (terminated | truncated).any():
                    bridge.reset()
                bridge.update_state()
            if image_writer:
                image_writer.write_images({
                    key: env.scene[name].data.output["rgb"][0, ..., :3].cpu().numpy()
                    for key, name in (("head", "front_camera"), ("left", "left_wrist_camera"), ("right", "right_wrist_camera"))
                })
            count += 1
            if not torch.isfinite(env.scene["robot"].data.joint_pos).all():
                raise RuntimeError("Non-finite robot state")
            if args.capture_dir and count == min(60, args.steps or 60):
                args.capture_dir.mkdir(parents=True, exist_ok=True)
                for name, sensor in env.scene.sensors.items():
                    rgb = sensor.data.output["rgb"][0, ..., :3].cpu().numpy()
                    Image.fromarray(rgb).save(args.capture_dir / f"{name}.png")
                manifest = dict(
                    task="Isaac-Sort-Shapes-G1D-Dex1-Joint", seed=args.seed, control_step=count,
                    step_dt=env.step_dt, settings=cfg.sorting,
                    object_poses={name: obj.data.root_state_w[:, :7].cpu().tolist()
                                  for name, obj in env.scene.rigid_objects.items()},
                )
                (args.capture_dir / "scene.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"Saved camera views to {args.capture_dir.resolve()}", flush=True)
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
