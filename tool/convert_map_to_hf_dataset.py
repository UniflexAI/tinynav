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
        description="Convert TinyNav map outputs to a HuggingFace dataset for trajectory prediction."
    )
    parser.add_argument("--map-dir", required=True, help="TinyNav map directory (contains poses/db/video files).")
    parser.add_argument("--output-dir", required=True, help="Output HuggingFace dataset directory.")
    parser.add_argument("--pose-file", default="", help="Pose npy file. Default: <map-dir>/poses.npy")
    parser.add_argument("--intrinsics-file", default="", help="Intrinsics npy file. Default: <map-dir>/intrinsics.npy")
    parser.add_argument("--dt", type=float, default=0.1, help="Trajectory delta time in seconds.")
    parser.add_argument("--horizon", type=float, default=2.0, help="Trajectory horizon in seconds.")
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


def iter_rows(
    map_dir: str,
    pose_timestamps: np.ndarray,
    keyframe_ts_np: np.ndarray,
    pose_interp: PoseInterpolator,
    K: np.ndarray,
    dt_ns: int,
    num_steps: int,
    max_gap_ns: int,
):
    infra_db = VideoDB(os.path.join(map_dir, "infra1_images_db"), mode="read")
    depths = shelve.open(os.path.join(map_dir, "depths"))
    try:
        for pose_ts in pose_timestamps:
            idx = int(np.searchsorted(keyframe_ts_np, pose_ts))
            candidates = []
            if idx > 0:
                candidates.append(int(keyframe_ts_np[idx - 1]))
            if idx < len(keyframe_ts_np):
                candidates.append(int(keyframe_ts_np[idx]))
            if not candidates:
                continue
            nearest_ts = min(candidates, key=lambda t: abs(t - int(pose_ts)))
            if abs(nearest_ts - int(pose_ts)) > max_gap_ns:
                continue

            depth = np.asarray(depths[str(nearest_ts)], dtype=np.float32)
            image = infra_db.read(nearest_ts)
            if image is None or depth.size == 0:
                continue

            traj_ts = np.array([int(pose_ts) + i * dt_ns for i in range(num_steps)], dtype=np.int64)
            traj_pose = np.stack([pose_interp.query(int(t)) for t in traj_ts], axis=0).astype(np.float32)
            yield {
                "pose_timestamp_ns": int(pose_ts),
                "image_timestamp_ns": int(nearest_ts),
                "trajectory_timestamps_ns": traj_ts.tolist(),
                "infra1_image_npy": np_to_bytes(np.asarray(image, dtype=np.uint8)),
                "depth_npy": np_to_bytes(depth.astype(np.float32)),
                "intrinsics_npy": np_to_bytes(K.astype(np.float32)),
                "pose_trajectory_npy": np_to_bytes(traj_pose),
            }
    finally:
        infra_db.close()
        depths.close()


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
    num_steps = int(round(args.horizon / args.dt))
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
    keyframe_ts = sorted(
        set(int(k) for k in depths.keys())
        & set(int(k) for k in infra_db.ts_to_idx.keys())
    )
    if len(keyframe_ts) == 0:
        infra_db.close()
        depths.close()
        raise RuntimeError("No keyframes with infra1/depth found.")
    keyframe_ts_np = np.asarray(keyframe_ts, dtype=np.int64)

    infra_db.close()
    depths.close()

    ds = Dataset.from_generator(
        iter_rows,
        gen_kwargs={
            "map_dir": map_dir,
            "pose_timestamps": pose_timestamps,
            "keyframe_ts_np": keyframe_ts_np,
            "pose_interp": pose_interp,
            "K": K,
            "dt_ns": dt_ns,
            "num_steps": num_steps,
            "max_gap_ns": max_gap_ns,
        },
    )
    os.makedirs(args.output_dir, exist_ok=True)
    ds.save_to_disk(args.output_dir)
    print(f"Saved HF dataset to {args.output_dir}. rows={len(ds)}")


if __name__ == "__main__":
    main()
