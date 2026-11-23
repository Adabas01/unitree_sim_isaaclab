"""Author task-spec layouts from the Unitree G1_Dex1 recordings (the lab mid-train's data), so a spec can be in
distribution: the objects where the recordings had them, in the torso frame the specs use.

    # first / last head frames and start states of 8 episodes spread over the dataset (HTTP range reads, no download
    # of the 200 MB video files), plus a check image with the recorded fingertips projected into each first frame
    python tools/recording_layouts.py frames G1_Dex1_Find_Green_Blocks --count 8
    # correct the head camera for this dataset from gripper centres read off its first frames (keypoints JSON)
    python tools/recording_layouts.py fitcam ~/adam_unitree/datasets/unitree_reference/G1_Dex1_Find_Green_Blocks \
        --keypoints tasks/g1d_tasks/spec_tasks/specs/G1_DATA/recordings/G1_Dex1_Find_Green_Blocks.keypoints.json
    # torso-frame xy of pixels on the table (u,v) or of an object centre h metres above it (u,v,h)
    python tools/recording_layouts.py project 320,300 250,280,0.02 --frames <frames dir>
    # where the grippers closed (grasps) and opened (releases) in each fetched episode, by forward kinematics of the
    # recorded joints: object and container positions without the camera, and the table height from the lowest grasps
    python tools/recording_layouts.py events ~/adam_unitree/datasets/unitree_reference/G1_Dex1_Collect_Eggs
    # draw a spec's object boxes into the first frames of a dataset (compare the layout with the recordings)
    python tools/recording_layouts.py overlay tasks/g1d_tasks/spec_tasks/specs/G1_DATA/find_green_blocks.yaml \
        --frames ~/adam_unitree/datasets/unitree_reference/G1_Dex1_Find_Green_Blocks

Camera and table: ponimatkin's fit of the G1 recordings' head camera (chiral 554fb5e
assets/robots/g1/calib/g1_head_camera.json: torso_link frame, OpenCV axes, pinhole without distortion, fitted to
fingertip keypoints of the 21-dataset camera cluster fx 282.4, mean reprojection error 3.6 px) and his table fit
(g1_table.json: table top 0.03 m above the torso origin, from 1206 grasp onsets of 7 datasets). Datasets outside that
camera cluster may have had the head camera elsewhere: the `frames` check image shows it (the projected fingertips
must sit on the Dex1 fingertips). In the unibot datasets checked (Find_Green_Blocks, Collect_Eggs) they sit ~20 px
above the real ones (fx 263-296 cameras, not the fx 282 cluster); `fitcam` fits a small rotation of the camera to
gripper centres read off the frames, and `project` / `overlay` use it (`camera_fit.json` in the frames folder).
The recorded `state_d435` pose with the recorded intrinsics fits worse (checked on both datasets). Forward kinematics: chiral's URDF model of the G1 (`--chiral`, `--commit`; read with
git archive, nothing in that clone changes). Arm values are in SDK order although unibot labels the wrists
(yaw, roll, pitch) (ichrpi g1_data_ops/schema.py); gripper value = raw / 4.5.
"""

import argparse
import json
import math
import subprocess
import sys
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np

HEAD = {  # chiral 554fb5e assets/robots/g1/calib/g1_head_camera.json
    "position": (0.04151561425612346, 0.04287700423055978, 0.4280607119033177),
    "quat_xyzw": (-0.6451491893965147, 0.6625333986343136, -0.2750115824825839, 0.26306016158953216),
    "fx": 282.37, "fy": 282.75, "cx": 311.5, "cy": 256.6, "size": (640, 480),
}
TABLE_Z = 0.03  # chiral 554fb5e g1_table.json table_z_in_torso
GRIPPER_FULL_OPEN = 4.5
CAMERA_KEY = "observation.images.head_stereo_left"
DEFAULT_OUT = Path.home() / "adam_unitree" / "datasets" / "unitree_reference"


def rotation(quat_xyzw):
    x, y, z, w = quat_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


R_HEAD = rotation(HEAD["quat_xyzw"])  # camera (OpenCV) -> torso
P_CAM = np.array(HEAD["position"])
R_CAM = R_HEAD  # chiral's camera, or with a dataset's fitcam correction (use_camera_fit)


def rotvec(r):
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.eye(3)
    k = np.asarray(r, float) / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def use_camera_fit(frames_dir) -> dict | None:
    """Apply <frames_dir>/camera_fit.json (from fitcam), if there is one."""
    global R_CAM
    path = Path(frames_dir).expanduser() / "camera_fit.json" if frames_dir else None
    if path is None or not path.exists():
        R_CAM = R_HEAD
        return None
    fit = json.loads(path.read_text())
    R_CAM = R_HEAD @ rotvec(fit["rotvec_camera"])
    return fit


def to_pixels(points_torso, r_cam=None):
    """(N, 3) torso-frame points -> (N, 2) pixels and (N,) depth."""
    p = (np.asarray(points_torso, float).reshape(-1, 3) - P_CAM) @ (R_CAM if r_cam is None else r_cam)
    return np.stack((HEAD["fx"] * p[:, 0] / p[:, 2] + HEAD["cx"], HEAD["fy"] * p[:, 1] / p[:, 2] + HEAD["cy"]), 1), p[:, 2]


def to_table(u, v, height=0.0):
    """Torso-frame point where the pixel's ray meets the plane `height` above the table top."""
    ray = R_CAM @ np.array([(u - HEAD["cx"]) / HEAD["fx"], (v - HEAD["cy"]) / HEAD["fy"], 1.0])
    t = (TABLE_Z + height - P_CAM[2]) / ray[2]
    return P_CAM + t * ray


def chiral_spec(repo: Path, commit: str):
    """chiral's biman/robots/g1_spec.py and G1 URDF from `commit` (git archive into a temp dir), imported."""
    data = subprocess.run(["git", "-C", str(repo.expanduser()), "archive", commit, "biman/robots/g1_spec.py",
                           "assets/robots/g1/g1", "assets/robots/g1/g1d", "assets/robots/g1/calib"], check=True, capture_output=True).stdout
    tmp = Path(tempfile.mkdtemp(prefix="chiral_"))
    with tarfile.open(fileobj=BytesIO(data)) as tar:
        tar.extractall(tmp)
    (tmp / "biman" / "__init__.py").touch()
    (tmp / "biman" / "robots" / "__init__.py").touch()
    sys.path.insert(0, str(tmp))
    from biman.robots import g1_spec  # noqa: E402
    return g1_spec


def canonical(row) -> np.ndarray:
    """A recorded frame as the canonical 16-vector [left arm 7, left gripper, right arm 7, right gripper]."""
    grip = lambda key: float(np.clip(np.asarray(row[key]).ravel()[0] / GRIPPER_FULL_OPEN, 0.0, 1.0))
    return np.r_[np.asarray(row["observation.state.left_arm"], float), grip("observation.state.left_gripper"),
                 np.asarray(row["observation.state.right_arm"], float), grip("observation.state.right_gripper")]


def cmd_frames(args) -> None:
    import pandas as pd
    from huggingface_hub import HfFileSystem, hf_hub_download

    repo = f"unitreerobotics/{args.dataset}"
    out = (args.out or DEFAULT_OUT) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads(Path(hf_hub_download(repo, "meta/info.json", repo_type="dataset", local_dir=out)).read_text())
    fs = HfFileSystem()
    episode_files = sorted(fs.glob(f"datasets/{repo}/meta/episodes/*/*.parquet"))
    episodes = pd.concat([pd.read_parquet(fs.open(f, "rb")) for f in episode_files], ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    if args.episodes:
        chosen = [int(e) for e in args.episodes.split(",")]
    else:
        chosen = sorted({int(episodes.episode_index.iloc[round(k * (len(episodes) - 1) / max(args.count - 1, 1))])
                         for k in range(args.count)})
    tasks = pd.read_parquet(fs.open(f"datasets/{repo}/meta/tasks.parquet", "rb"))
    spec = chiral_spec(args.chiral, args.commit) if args.chiral else None
    kin = spec.kinematics("g1") if spec else None
    records = []
    for ep in chosen:
        e = episodes[episodes.episode_index == ep].iloc[0]
        chunk, file = int(e[f"videos/{CAMERA_KEY}/chunk_index"]), int(e[f"videos/{CAMERA_KEY}/file_index"])
        t0, t1 = float(e[f"videos/{CAMERA_KEY}/from_timestamp"]), float(e[f"videos/{CAMERA_KEY}/to_timestamp"])
        url = (f"https://huggingface.co/datasets/{repo}/resolve/main/"
               + info["video_path"].format(video_key=CAMERA_KEY, chunk_index=chunk, file_index=file))
        for name, t in (("first", t0), ("last", max(t0, t1 - 2.0 / info["fps"]))):
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.4f}", "-i", url,
                            "-frames:v", "1", "-q:v", "2", str(out / f"ep{ep:05d}_{name}.jpg")], check=True)
        data_file = info["data_path"].format(chunk_index=int(e["data/chunk_index"]), file_index=int(e["data/file_index"]))
        columns = ["episode_index", "frame_index", "observation.state.left_arm", "observation.state.right_arm",
                   "observation.state.left_gripper", "observation.state.right_gripper"]
        frame = pd.read_parquet(fs.open(f"datasets/{repo}/{data_file}", "rb"), columns=columns,
                                filters=[("episode_index", "==", ep), ("frame_index", "==", 0)]).iloc[0]
        q = canonical(frame)
        record = {"episode": ep, "length": int(e["length"]), "first_timestamp": t0, "video": url,
                  "task": str(e["tasks"][0]) if len(e["tasks"]) else "", "start_q": [round(float(v), 4) for v in q]}
        if kin is not None:
            tips = {side: spec.gripper_points_in_torso(kin, q, side, spec.fingertip_points(q[7 if side == "left" else 15]))
                    for side in ("left", "right")}
            record["fingertips_torso"] = {s: np.round(p, 4).tolist() for s, p in tips.items()}
            draw_points(out / f"ep{ep:05d}_first.jpg", out / f"ep{ep:05d}_check.jpg",
                        {s: to_pixels(p)[0] for s, p in tips.items()})
        records.append(record)
        print(f"episode {ep}: {e['length']} frames, start at {t0:.2f} s of {url.rsplit('/', 3)[-2:]}", flush=True)
    camera = info.get("camera_params", {}).get("head_stereo", {})
    meta = {"dataset": repo, "fps": info["fps"], "total_episodes": info["total_episodes"],
            "tasks": tasks.index.tolist() if "task_index" in tasks.columns else tasks.to_dict(orient="list"),
            "recorded_head_left_camera_matrix": camera.get("camera_matrix_left"),
            "projection": {"camera": HEAD, "table_z_in_torso": TABLE_Z, "source": "chiral 554fb5e g1_head_camera.json, g1_table.json"},
            "episodes": records}
    (out / "frames.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out}/frames.json")


def draw_points(src: Path, dst: Path, groups: dict) -> None:
    from PIL import Image, ImageDraw
    img = Image.open(src).convert("RGB")
    d = ImageDraw.Draw(img)
    colours = {"left": (255, 0, 255), "right": (0, 255, 255)}
    for name, pixels in groups.items():
        for u, v in pixels:
            d.ellipse((u - 5, v - 5, u + 5, v + 5), outline=colours.get(name, (255, 255, 0)), width=2)
    img.save(dst, quality=92)


def cmd_fitcam(args) -> None:
    from scipy.optimize import least_squares

    folder = Path(args.frames).expanduser()
    frames = json.loads((folder / "frames.json").read_text())
    marks = json.loads(Path(args.keypoints).read_text())["points"]
    points, observed, labels = [], [], []
    for e in frames["episodes"]:
        for side, uv in marks.get(str(e["episode"]), {}).items():
            points.append(np.mean(e["fingertips_torso"][side], axis=0))  # gripper centre: mean of the two tips
            observed.append(uv)
            labels.append(f"ep{e['episode']} {side}")
    points, observed = np.array(points), np.array(observed, float)

    def residual(r):
        return (to_pixels(points, R_HEAD @ rotvec(r))[0] - observed).ravel()

    before = residual(np.zeros(3)).reshape(-1, 2)
    fit = least_squares(residual, np.zeros(3))
    after = fit.fun.reshape(-1, 2)
    for label, b, a in zip(labels, before, after):
        print(f"{label:12s} model - marked before ({b[0]:+6.1f}, {b[1]:+6.1f}) px, after ({a[0]:+6.1f}, {a[1]:+6.1f}) px")
    norm = lambda r: np.linalg.norm(r, axis=1)
    result = {"rotvec_camera": fit.x.tolist(), "angle_deg": math.degrees(float(np.linalg.norm(fit.x))),
              "points": len(points), "keypoints": str(Path(args.keypoints).resolve()),
              "error_px_before": {"mean": float(norm(before).mean()), "max": float(norm(before).max())},
              "error_px_after": {"mean": float(norm(after).mean()), "max": float(norm(after).max())},
              "about": "R_camera->torso = chiral's G1 head rotation @ Rotation(rotvec_camera); position unchanged"}
    (folder / "camera_fit.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"{len(points)} gripper centres: error {result['error_px_before']['mean']:.1f} -> "
          f"{result['error_px_after']['mean']:.1f} px mean (max {result['error_px_after']['max']:.1f}); camera turned "
          f"{result['angle_deg']:.2f} deg; wrote {folder / 'camera_fit.json'}")


def cmd_events(args) -> None:
    import pandas as pd
    from huggingface_hub import HfFileSystem

    folder = Path(args.frames).expanduser()
    frames = json.loads((folder / "frames.json").read_text())
    repo = frames["dataset"]
    info = json.loads((folder / "meta" / "info.json").read_text())
    fs = HfFileSystem()
    episodes = pd.concat([pd.read_parquet(fs.open(f, "rb")) for f in sorted(fs.glob(f"datasets/{repo}/meta/episodes/*/*.parquet"))])
    spec = chiral_spec(args.chiral, args.commit)
    kin = spec.kinematics("g1")
    columns = ["episode_index", "frame_index", "observation.state.left_arm", "observation.state.right_arm",
               "observation.state.left_gripper", "observation.state.right_gripper"]
    rows = []
    for e in frames["episodes"]:
        ep = e["episode"]
        meta = episodes[episodes.episode_index == ep].iloc[0]
        data_file = info["data_path"].format(chunk_index=int(meta["data/chunk_index"]), file_index=int(meta["data/file_index"]))
        df = pd.read_parquet(fs.open(f"datasets/{repo}/{data_file}", "rb"), columns=columns,
                             filters=[("episode_index", "==", ep)]).sort_values("frame_index")
        q = np.stack([canonical(r) for _, r in df.iterrows()])
        for side, k in (("left", 7), ("right", 15)):
            g = q[:, k]
            closed = g < args.closed
            # an event where the gripper state crosses the threshold and stays on that side for 0.3 s
            hold = max(1, int(0.3 * info["fps"]))
            for t in range(1, len(g) - hold):
                for kind, now in (("grasp", closed), ("release", ~closed)):
                    if now[t] and not now[t - 1] and now[t:t + hold].all():
                        t_event = t if kind == "release" else min(len(g) - 1, t + hold)  # grasp: once closed
                        tips = spec.gripper_points_in_torso(kin, q[t_event], side, spec.fingertip_points(q[t_event, k]))
                        centre = tips.mean(0)
                        rows.append({"episode": ep, "side": side, "kind": kind, "frame": int(t_event),
                                     "time_s": round(t_event / info["fps"], 2),
                                     "centre_torso": np.round(centre, 4).tolist(),
                                     "lowest_tip_z": round(float(tips[:, 2].min()), 4)})
    out = folder / "events.json"
    out.write_text(json.dumps({"dataset": repo, "closed_below": args.closed, "events": rows}, indent=1) + "\n")
    if args.images:  # the head frame at each event, with the forward-kinematics gripper centre projected into it
        use_camera_fit(folder)
        videos = {e["episode"]: (e["video"], e["first_timestamp"]) for e in frames["episodes"]}
        wanted = {int(v) for v in args.images.split(",")}
        for r in rows:
            if r["episode"] not in wanted:
                continue
            url, t0 = videos[r["episode"]]
            name = folder / f"ep{r['episode']:05d}_{r['kind']}_{r['side']}_{r['frame']:04d}.jpg"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t0 + r['frame'] / info['fps']:.4f}",
                            "-i", url, "-frames:v", "1", "-q:v", "2", str(name)], check=True)
            draw_points(name, name, {r["side"]: to_pixels(np.array(r["centre_torso"]))[0]})
    for r in rows:
        x, y, z = r["centre_torso"]
        print(f"ep{r['episode']:5d} {r['side']:5s} {r['kind']:7s} t {r['time_s']:5.1f} s  centre ({x:+.3f}, {y:+.3f}, {z:+.3f})")
    grasps = np.array([r["lowest_tip_z"] for r in rows if r["kind"] == "grasp"])
    if len(grasps):
        print(f"{len(grasps)} grasps: lowest fingertip z in torso median {np.median(grasps):+.3f} (10th pct "
              f"{np.percentile(grasps, 10):+.3f}); chiral's table fit for the fx 282 datasets: {TABLE_Z:+.3f}")
    print(f"wrote {out}")


def cmd_project(args) -> None:
    global TABLE_Z
    TABLE_Z = args.table_z
    fit = use_camera_fit(args.frames)
    print(f"camera: {'chiral fit + ' + args.frames + '/camera_fit.json' if fit else 'chiral fit, uncorrected'}")
    for item in args.pixels:
        values = [float(t) for t in item.split(",")]
        u, v, h = (*values, 0.0)[:3]
        p = to_table(u, v, h)
        print(f"{item:>18s} -> torso x {p[0]:+.3f} y {p[1]:+.3f} (z {p[2]:+.3f}, table {TABLE_Z:+.3f})")


def cmd_overlay(args) -> None:
    import yaml
    from PIL import Image, ImageDraw

    global TABLE_Z
    TABLE_Z = args.table_z
    spec = yaml.safe_load(Path(args.spec).read_text())
    folder = Path(args.frames).expanduser()
    use_camera_fit(folder)
    firsts = sorted(folder.glob("ep*_first.jpg"))[: args.max]
    for src in firsts:
        img = Image.open(src).convert("RGB")
        d = ImageDraw.Draw(img)
        for o in spec["objects"]:
            bx, by, bz = (float(v) for v in o["bounding_box"])
            x, y = (float(v) for v in o["xy_torso"])
            qx, qy, qz, qw = (float(v) for v in o["orientation_torso_xyzw"])
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            c, s = math.cos(yaw), math.sin(yaw)
            zc = TABLE_Z + float(o["centre_z_above_table"])
            corners = [(x + c * dx - s * dy, y + s * dx + c * dy, zc + dz)
                       for dz in (-bz / 2, bz / 2) for dx, dy in ((-bx / 2, -by / 2), (bx / 2, -by / 2),
                                                                 (bx / 2, by / 2), (-bx / 2, by / 2))]
            px, _ = to_pixels(corners)
            colour = {"main": (255, 60, 255), "target": (0, 230, 255)}.get(o["role"], (255, 255, 0))
            for ring in (px[:4], px[4:]):
                d.line([tuple(p) for p in ring] + [tuple(ring[0])], fill=colour, width=2)
            for k in range(4):
                d.line([tuple(px[k]), tuple(px[k + 4])], fill=colour, width=1)
            d.text(tuple(px[4] + (2, -12)), o["name"], fill=colour)
        dst = folder / src.name.replace("_first.jpg", f"_overlay_{Path(args.spec).stem}.jpg")
        img.save(dst, quality=92)
        print(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("frames")
    p.add_argument("dataset", help="e.g. G1_Dex1_Find_Green_Blocks (under huggingface.co/datasets/unitreerobotics)")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--episodes", help="comma-separated episode indices instead of --count")
    p.add_argument("--out", type=Path)
    p.add_argument("--chiral", type=Path, default=Path.home() / "adam_unitree" / "isaaclab_biman_sim",
                   help="git clone of ponimatkin/chiral, for the G1 kinematics (empty: no fingertip check)")
    p.add_argument("--commit", default="554fb5e")
    p = sub.add_parser("fitcam")
    p.add_argument("frames", help="a frames folder (frames.json with fingertips_torso)")
    p.add_argument("--keypoints", required=True)
    p = sub.add_parser("events")
    p.add_argument("frames", help="a frames folder")
    p.add_argument("--closed", type=float, default=0.6, help="gripper value (1 = open) below which it counts as closed")
    p.add_argument("--images", help="comma-separated episodes: save their event frames with the gripper centre drawn")
    p.add_argument("--chiral", type=Path, default=Path.home() / "adam_unitree" / "isaaclab_biman_sim")
    p.add_argument("--commit", default="554fb5e")
    p = sub.add_parser("project")
    p.add_argument("pixels", nargs="+", help="u,v (on the table) or u,v,h (h metres above it)")
    p.add_argument("--frames", help="frames folder whose camera_fit.json to use")
    p.add_argument("--table-z", type=float, default=TABLE_Z, help="table top in the torso frame")
    p = sub.add_parser("overlay")
    p.add_argument("spec")
    p.add_argument("--frames", required=True)
    p.add_argument("--max", type=int, default=8)
    p.add_argument("--table-z", type=float, default=TABLE_Z, help="table top in the torso frame (the spec's)")
    args = parser.parse_args()
    {"frames": cmd_frames, "fitcam": cmd_fitcam, "events": cmd_events, "project": cmd_project,
     "overlay": cmd_overlay}[args.command](args)


if __name__ == "__main__":
    main()
