"""Replay a recorded bag through the planner's occupancy grid and dump the policy BEV.

Needs /slam/depth + /slam/odometry_visual + a camera_info in the bag (debug bags have them).
Usage:
  ROBOT_TYPE=go2 uv run python -m tinynav.rl.export_bev --bag <bag_dir> --out bev.npz --png bev.png
"""
import argparse

import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tinynav.core.math_utils import msg2np
from tinynav.core.planning_node import update_occupancy_grid
from tinynav.core.robot_specs import ROBOT_CONFIG
from tinynav.rl.bev import CHANNELS, GRID_RES, GRID_SHAPE, RAY_STEP, TRAIL_LEN, camera_yaw, make_bev, to_robot_frame, trail_channel

DEPTH, ODOM, GOAL, IMAGE = "/slam/depth", "/slam/odometry_visual", "/control/target_pose", "/camera/camera/infra1/image_rect_raw"
INFO_TOPICS = ("/camera/camera/infra2/camera_info", "/camera/camera/infra1/camera_info")


def read_bag(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=""), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    missing = {DEPTH, ODOM} - types.keys()
    if missing or not any(t in types for t in INFO_TOPICS):
        raise SystemExit(f"bag lacks {sorted(missing) or 'camera_info'}; topics: {sorted(types)}")
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in types:
            yield topic, deserialize_message(data, types[topic])


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def export(bag, every):
    bridge = CvBridge()
    grid, origin = np.zeros(GRID_SHAPE), np.array(GRID_SHAPE) * GRID_RES / -2.
    K, goal, image, last = None, None, None, None
    depths, odoms, hist = {}, {}, []
    out = {k: [] for k in ("bev", "goal", "vel", "stamp", "image")}
    frame = 0
    for topic, msg in read_bag(bag):
        if topic in INFO_TOPICS and K is None:
            K = np.array(msg.k).reshape(3, 3)
        elif topic == GOAL:
            goal = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        elif topic == IMAGE:
            image = msg
        elif topic in (DEPTH, ODOM):
            (depths if topic == DEPTH else odoms)[stamp_ns(msg)] = msg
        synced = depths.keys() & odoms.keys()
        if K is None or not synced:
            continue
        t = min(synced)
        depth = bridge.imgmsg_to_cv2(depths.pop(t), desired_encoding="32FC1")
        T, _ = msg2np(odoms.pop(t))
        depths = {k: v for k, v in depths.items() if k > t}
        odoms = {k: v for k, v in odoms.items() if k > t}
        grid, origin = update_occupancy_grid(grid, origin, depth, T, K, GRID_RES, RAY_STEP)

        center = (T[:3, 3] - T[:3, :3] @ ROBOT_CONFIG.cam_offset_3d)[:2]
        yaw = camera_yaw(T)
        vel = np.zeros(2, dtype=np.float32)  # (forward m/s, yaw rate rad/s) from consecutive poses
        if last is not None:
            dt = (t - last[0]) * 1e-9
            vel = np.array([to_robot_frame(center, last[1], last[2])[0] / dt,
                            np.angle(np.exp(1j * (yaw - last[2]))) / dt], dtype=np.float32)
        last = (t, center, yaw)
        hist = [center] + hist[:TRAIL_LEN - 1]
        frame += 1
        if frame % every:
            continue
        trail = trail_channel(np.array(hist), np.arange(len(hist)), center, yaw)
        out["bev"].append(np.concatenate([make_bev(grid, origin, GRID_RES, center, yaw, ref_z=T[2, 3]), trail[None]]))
        out["goal"].append(to_robot_frame(goal, center, yaw) if goal is not None else np.full(2, np.nan, np.float32))
        out["vel"].append(vel)
        out["stamp"].append(t)
        out["image"].append(bridge.imgmsg_to_cv2(image, desired_encoding="mono8")[::4, ::4] if image is not None else None)
    print(f"{frame} synced frames, kept {len(out['bev'])}")
    return out


def save_png(out, path, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.linspace(0, len(out["bev"]) - 1, n).astype(int)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    ext = (-3.2, 3.2, -3.2, 3.2)  # x: robot left..right, y: back..front (m)
    for col, i in enumerate(idx):
        bev, img = out["bev"][i], out["image"][i]
        if img is not None:
            axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"t={(out['stamp'][i] - out['stamp'][0]) * 1e-9:.1f}s")
        h = axes[1, col].imshow(bev[0], cmap="viridis", vmin=-0.5, vmax=1.0, extent=ext)
        axes[2, col].imshow(bev[1], cmap="gray", vmin=0, vmax=1, extent=ext)
        g = out["goal"][i]
        for ax in axes[1:, col]:
            ax.plot(0, 0, "r^", ms=8)
            if np.all(np.isfinite(g)) and np.abs(g).max() < 3.2:
                ax.plot(-g[1], g[0], "m*", ms=12)
            ax.set_xticks([]), ax.set_yticks([])
        axes[0, col].axis("off")
    axes[1, 0].set_ylabel(f"{CHANNELS[0]} (m above ground)")
    axes[2, 0].set_ylabel(CHANNELS[1])
    fig.colorbar(h, ax=axes[1, :].tolist(), shrink=0.8)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--png")
    ap.add_argument("--every", type=int, default=5, help="keep every N-th synced frame")
    args = ap.parse_args()
    out = export(args.bag, args.every)
    np.savez_compressed(args.out, **{k: np.stack(out[k]) for k in ("bev", "goal", "vel", "stamp")}, channels=CHANNELS)
    print(f"wrote {args.out}: bev {np.stack(out['bev']).shape}")
    if args.png:
        save_png(out, args.png)


if __name__ == "__main__":
    main()
