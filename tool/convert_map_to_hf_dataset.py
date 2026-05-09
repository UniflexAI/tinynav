import argparse
import io
import os
import shelve
import sys
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tool.video_db import VideoDB


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TinyNav map outputs to a HuggingFace dataset with interpolated feature trajectories."
    )
    parser.add_argument("--map-dir", required=True, help="TinyNav map directory (contains poses/db/video files).")
    parser.add_argument("--output-dir", required=True, help="Output HuggingFace dataset directory.")
    parser.add_argument("--pose-file", default="", help="Pose npy file. Default: <map-dir>/poses.npy")
    parser.add_argument("--intrinsics-file", default="", help="Intrinsics npy file. Default: <map-dir>/intrinsics.npy")
    parser.add_argument("--dt", type=float, default=0.1, help="Trajectory delta time in seconds.")
    parser.add_argument("--horizon", type=float, default=2.0, help="Trajectory horizon in seconds.")
    parser.add_argument("--max-features", type=int, default=256, help="Max keypoints used per sample.")
    parser.add_argument("--max-image-gap", type=float, default=0.2, help="Max allowed gap (seconds) from pose timestamp to nearest keyframe timestamp.")
    parser.add_argument("--pose-stride", type=int, default=1, help="Use every Nth pose timestamp.")
    return parser.parse_args()


def np_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


@dataclass
class PoseInterpolator:
    timestamps_ns: np.ndarray
    poses: list

    @classmethod
    def from_dict(cls, pose_dict):
        ts = np.array(sorted(int(k) for k in pose_dict.keys()), dtype=np.int64)
        poses = [np.asarray(pose_dict[int(t)], dtype=np.float64) for t in ts]
        return cls(timestamps_ns=ts, poses=poses)

    def query(self, target_ns: int):
        ts = self.timestamps_ns
        idx = int(np.searchsorted(ts, target_ns))
        if idx == 0:
            return self.poses[0].copy()
        if idx >= len(ts):
            return self.poses[-1].copy()
        t0 = int(ts[idx - 1])
        t1 = int(ts[idx])
        p0 = self.poses[idx - 1]
        p1 = self.poses[idx]
        if t1 == t0:
            return p0.copy()
        alpha = float(target_ns - t0) / float(t1 - t0)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        rot = R.from_matrix(np.stack([p0[:3, :3], p1[:3, :3]], axis=0))
        slerp = Slerp([0.0, 1.0], rot)
        r_interp = slerp(alpha).as_matrix()
        t_interp = (1.0 - alpha) * p0[:3, 3] + alpha * p1[:3, 3]
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = r_interp
        out[:3, 3] = t_interp
        return out


def depth_to_points_from_keypoints(depth: np.ndarray, kpts_xy: np.ndarray, K: np.ndarray):
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    h, w = depth.shape[:2]

    us = np.rint(kpts_xy[:, 0]).astype(np.int32)
    vs = np.rint(kpts_xy[:, 1]).astype(np.int32)
    in_bounds = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
    z = np.zeros((kpts_xy.shape[0],), dtype=np.float32)
    z[in_bounds] = depth[vs[in_bounds], us[in_bounds]]
    valid = in_bounds & np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    u = us[valid].astype(np.float32)
    v = vs[valid].astype(np.float32)
    z_valid = z[valid].astype(np.float32)
    x = (u - cx) * z_valid / fx
    y = (v - cy) * z_valid / fy
    pts = np.stack([x, y, z_valid], axis=1).astype(np.float32)
    return pts, kpts_xy[valid].astype(np.float32)


def project_world_to_image(points_world: np.ndarray, T_world_cam: np.ndarray, K: np.ndarray, h: int, w: int):
    if points_world.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=bool)
    T_cam_world = np.linalg.inv(T_world_cam)
    homo = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (T_cam_world @ homo.T).T[:, :3]
    z = cam[:, 2]
    valid_z = z > 1e-6
    uv = np.zeros((points_world.shape[0], 2), dtype=np.float32)
    valid = np.zeros((points_world.shape[0],), dtype=bool)
    if np.any(valid_z):
        x = cam[valid_z, 0]
        y = cam[valid_z, 1]
        z_ok = z[valid_z]
        u = K[0, 0] * (x / z_ok) + K[0, 2]
        v = K[1, 1] * (y / z_ok) + K[1, 2]
        uv_valid = np.stack([u, v], axis=1).astype(np.float32)
        idx = np.where(valid_z)[0]
        uv[idx] = uv_valid
        in_bounds = (uv_valid[:, 0] >= 0.0) & (uv_valid[:, 0] < w) & (uv_valid[:, 1] >= 0.0) & (uv_valid[:, 1] < h)
        valid[idx] = in_bounds
    return uv, valid


def main():
    args = parse_args()
    try:
        from datasets import Dataset
    except Exception as exc:
        raise RuntimeError("Please install huggingface datasets: `uv pip install datasets`") from exc

    map_dir = args.map_dir
    pose_file = args.pose_file or os.path.join(map_dir, "poses.npy")
    intrinsics_file = args.intrinsics_file or os.path.join(map_dir, "intrinsics.npy")
    dt_ns = int(round(args.dt * 1e9))
    horizon_ns = int(round(args.horizon * 1e9))
    num_steps = int(round(args.horizon / args.dt)) + 1
    max_gap_ns = int(round(args.max_image_gap * 1e9))

    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"pose file not found: {pose_file}")
    if not os.path.exists(intrinsics_file):
        raise FileNotFoundError(f"intrinsics file not found: {intrinsics_file}")

    pose_dict = np.load(pose_file, allow_pickle=True).item()
    K = np.asarray(np.load(intrinsics_file), dtype=np.float64)
    pose_interp = PoseInterpolator.from_dict(pose_dict)
    pose_timestamps = pose_interp.timestamps_ns[:: max(1, args.pose_stride)]

    infra_db = VideoDB(os.path.join(map_dir, "infra1_images_db"), mode="read")
    depths = shelve.open(os.path.join(map_dir, "depths"))
    features = shelve.open(os.path.join(map_dir, "features"))
    keyframe_ts = sorted(
        set(int(k) for k in depths.keys())
        & set(int(k) for k in features.keys())
        & set(int(k) for k in infra_db.ts_to_idx.keys())
    )
    if len(keyframe_ts) == 0:
        infra_db.close()
        depths.close()
        features.close()
        raise RuntimeError("No keyframes with infra1/depth/features found.")
    keyframe_ts_np = np.asarray(keyframe_ts, dtype=np.int64)

    rows = []
    skipped_missing_keyframe = 0
    skipped_missing_data = 0
    for pose_ts in pose_timestamps:
        idx = int(np.searchsorted(keyframe_ts_np, pose_ts))
        candidates = []
        if idx > 0:
            candidates.append(int(keyframe_ts_np[idx - 1]))
        if idx < len(keyframe_ts_np):
            candidates.append(int(keyframe_ts_np[idx]))
        if not candidates:
            skipped_missing_keyframe += 1
            continue
        nearest_ts = min(candidates, key=lambda t: abs(t - int(pose_ts)))
        if abs(nearest_ts - int(pose_ts)) > max_gap_ns:
            skipped_missing_keyframe += 1
            continue

        depth = np.asarray(depths[str(nearest_ts)], dtype=np.float32)
        feat = features[str(nearest_ts)]
        image = infra_db.read(nearest_ts)
        if image is None or depth.size == 0 or "kpts" not in feat:
            skipped_missing_data += 1
            continue

        kpts = np.asarray(feat["kpts"][0], dtype=np.float32)
        if kpts.shape[0] > args.max_features:
            kpts = kpts[: args.max_features]

        T0 = pose_interp.query(int(pose_ts))
        traj_ts = np.array([int(pose_ts) + i * dt_ns for i in range(num_steps)], dtype=np.int64)
        traj_pose = np.stack([pose_interp.query(int(t)) for t in traj_ts], axis=0).astype(np.float32)

        pts_cam0, kpts_anchor = depth_to_points_from_keypoints(depth, kpts, K)
        if pts_cam0.size == 0:
            skipped_missing_data += 1
            continue
        pts_h = np.concatenate([pts_cam0.astype(np.float64), np.ones((pts_cam0.shape[0], 1), dtype=np.float64)], axis=1)
        pts_world = (T0 @ pts_h.T).T[:, :3]

        h, w = depth.shape[:2]
        traj_uv = np.zeros((pts_world.shape[0], num_steps, 2), dtype=np.float32)
        traj_valid = np.zeros((pts_world.shape[0], num_steps), dtype=np.uint8)
        for step_i, Twc in enumerate(traj_pose):
            uv, valid = project_world_to_image(pts_world, Twc.astype(np.float64), K, h, w)
            traj_uv[:, step_i, :] = uv
            traj_valid[:, step_i] = valid.astype(np.uint8)

        rows.append(
            {
                "pose_timestamp_ns": int(pose_ts),
                "image_timestamp_ns": int(nearest_ts),
                "trajectory_timestamps_ns": traj_ts.tolist(),
                "infra1_image_npy": np_to_bytes(np.asarray(image, dtype=np.uint8)),
                "depth_npy": np_to_bytes(depth.astype(np.float32)),
                "pose_trajectory_npy": np_to_bytes(traj_pose),
                "feature_anchor_uv_npy": np_to_bytes(kpts_anchor.astype(np.float32)),
                "feature_trajectory_uv_npy": np_to_bytes(traj_uv.astype(np.float32)),
                "feature_trajectory_valid_npy": np_to_bytes(traj_valid.astype(np.uint8)),
            }
        )

    infra_db.close()
    depths.close()
    features.close()

    ds = Dataset.from_list(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    ds.save_to_disk(args.output_dir)
    print(
        f"Saved HF dataset to {args.output_dir}. "
        f"rows={len(rows)} skipped_missing_keyframe={skipped_missing_keyframe} skipped_missing_data={skipped_missing_data}"
    )


if __name__ == "__main__":
    main()
