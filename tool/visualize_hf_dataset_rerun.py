import argparse
import io

import cv2
import numpy as np
import rerun as rr
from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize TinyNav HF dataset with rerun-sdk.")
    parser.add_argument("--dataset-dir", required=True, help="HF dataset directory.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index.")
    parser.add_argument("--max-points", type=int, default=500, help="Max points to log.")
    parser.add_argument("--spawn", action="store_true", help="Spawn rerun viewer.")
    parser.add_argument("--recording-id", default="tinynav_hf", help="Rerun recording id.")
    return parser.parse_args()


def decode_npy_bytes(blob):
    return np.load(io.BytesIO(blob), allow_pickle=False)


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo = float(np.percentile(depth[valid], 5))
    hi = float(np.percentile(depth[valid], 95))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[:, :, ::-1]


def main():
    args = parse_args()
    ds = load_from_disk(args.dataset_dir)
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty.")
    if args.sample_idx < 0 or args.sample_idx >= len(ds):
        raise IndexError(f"sample-idx {args.sample_idx} out of range [0, {len(ds)-1}]")

    row = ds[args.sample_idx]
    image = decode_npy_bytes(row["infra1_image_npy"])
    depth = decode_npy_bytes(row["depth_npy"])
    anchor_uv = decode_npy_bytes(row["feature_anchor_uv_npy"]).astype(np.float32)
    traj_uv = decode_npy_bytes(row["feature_trajectory_uv_npy"]).astype(np.float32)
    traj_valid = decode_npy_bytes(row["feature_trajectory_valid_npy"]).astype(bool)
    traj_ts = np.asarray(row["trajectory_timestamps_ns"], dtype=np.int64)

    n_pts = traj_uv.shape[0]
    keep_n = min(max(1, args.max_points), n_pts)
    draw_idx = np.linspace(0, n_pts - 1, keep_n, dtype=np.int32) if n_pts > keep_n else np.arange(n_pts, dtype=np.int32)

    rr.init(args.recording_id, spawn=args.spawn)
    rr.log("hf/info", rr.TextDocument(
        f"sample_idx={args.sample_idx}\n"
        f"pose_timestamp_ns={row['pose_timestamp_ns']}\n"
        f"image_timestamp_ns={row['image_timestamp_ns']}\n"
        f"n_points={n_pts}\n"
        f"n_steps={traj_uv.shape[1]}"
    ))

    if image.ndim == 2:
        rr.log("hf/infra1", rr.Image(image))
    else:
        rr.log("hf/infra1", rr.Image(image))
    rr.log("hf/depth_colormap", rr.Image(depth_to_colormap(depth)))
    rr.log("hf/anchor_points", rr.Points2D(anchor_uv[draw_idx], colors=[0, 0, 255], radii=1.5))

    for step in range(traj_uv.shape[1]):
        rr.set_time_nanos("frame_time", int(traj_ts[step]))
        uv_step = traj_uv[draw_idx, step, :]
        valid_step = traj_valid[draw_idx, step]
        rr.log("hf/trajectory/all", rr.Points2D(uv_step, colors=[120, 120, 120], radii=1.0))
        rr.log("hf/trajectory/valid", rr.Points2D(uv_step[valid_step], colors=[0, 255, 0], radii=1.5))
        rr.log("hf/metrics/valid_count", rr.Scalars(np.array([int(np.count_nonzero(valid_step))], dtype=np.float32)))

    print(
        f"Logged sample {args.sample_idx} to rerun. "
        f"Use `rerun` viewer to inspect time sequence `frame_time`."
    )


if __name__ == "__main__":
    main()
