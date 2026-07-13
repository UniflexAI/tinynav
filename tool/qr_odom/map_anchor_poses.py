#!/usr/bin/env python3
"""
tool/qr_odom/map_anchor_poses.py

Average the pose of every visible AprilTag anchor board (see generate.py
--id-offset) into a built map's own coordinate frame, by re-detecting the
boards in that map's saved keyframe images and combining with the map's
final pose-graph-optimized keyframe poses. Fully offline — no ROS required —
since it only reads artifacts already written by build_map_node.py.

Reads (from --map_path):
  poses.npy          {timestamp_ns: 4x4 T_map_camera}, pose-graph-optimized
  intrinsics.npy      3x3 camera K (infra1)
  infra1_images_db/   keyframe images, keyed by the same timestamps

Writes:
  <map_path>/anchor_poses.json
    {label: {"tag_ids": [...], "T_map_board": 4x4, "n_samples": int,
              "mean_reproj_px": float}}

Usage:
  python tool/qr_odom/map_anchor_poses.py --map_path tinynav_db/map_a
  python tool/qr_odom/map_anchor_poses.py --map_path tinynav_db/map_b
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tool.qr_odom.board_detect import detect_board_poses, load_boards, make_detector
from tool.qr_odom.pose_utils import mean_T, rotation_angle_deg
from tool.video_db import VideoDB

QRCODE_DIR = Path("tinynav_db/qrcode")

MIN_TAGS = 2
MAX_REPROJ_PX = 3.0
ROTATION_OUTLIER_DEG = 5.0


def collect_anchor_poses(map_path: Path, qrcode_dir: Path = QRCODE_DIR) -> dict:
    boards = load_boards(qrcode_dir)
    if not boards:
        raise FileNotFoundError(f"No tag_grid_2x2_s76mm*.json board configs found in {qrcode_dir}")

    poses = np.load(map_path / "poses.npy", allow_pickle=True).item()
    K = np.load(map_path / "intrinsics.npy")
    video_db = VideoDB(str(map_path / "infra1_images_db"), mode="read")
    detector = make_detector()

    samples: dict[str, list[np.ndarray]] = {label: [] for label in boards}
    reproj_px: dict[str, list[float]] = {label: [] for label in boards}

    for timestamp, T_map_camera in poses.items():
        img = video_db.read(timestamp)
        if img is None:
            continue
        detections = detect_board_poses(
            img, K, boards, detector, min_tags=MIN_TAGS, max_reproj_px=MAX_REPROJ_PX,
        )
        for label, (T_camera_board, reproj_err, _n_tags) in detections.items():
            T_map_board = T_map_camera @ T_camera_board
            board_samples = samples[label]
            if board_samples:
                angle_deg = rotation_angle_deg(mean_T(board_samples)[:3, :3], T_map_board[:3, :3])
                if angle_deg > ROTATION_OUTLIER_DEG:
                    print(f"  [{label}] rotation outlier: {angle_deg:.1f}° from running mean, keeping anyway")
            board_samples.append(T_map_board)
            reproj_px[label].append(reproj_err)

    result = {}
    for label, board_samples in samples.items():
        if not board_samples:
            print(f"Board {label}: not visible in this map, skipping")
            continue
        T_map_board = mean_T(board_samples)
        result[label] = {
            "tag_ids": boards[label].getIds().flatten().tolist(),
            "T_map_board": T_map_board.tolist(),
            "n_samples": len(board_samples),
            "mean_reproj_px": float(np.mean(reproj_px[label])),
        }
        print(f"Board {label}: {len(board_samples)} samples, mean reproj {np.mean(reproj_px[label]):.2f}px")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Average AprilTag anchor board poses over a built map's keyframes."
    )
    parser.add_argument("--map_path", type=Path, required=True,
                         help="Map directory written by build_map_node.py")
    parser.add_argument("--qrcode_dir", type=Path, default=QRCODE_DIR,
                         help="Directory with tag_grid_2x2_s76mm*.json board configs")
    parser.add_argument("--out", type=Path, default=None,
                         help="Output JSON path (default: <map_path>/anchor_poses.json)")
    args = parser.parse_args()

    out_path = args.out or (args.map_path / "anchor_poses.json")
    result = collect_anchor_poses(args.map_path, args.qrcode_dir)
    if not result:
        raise SystemExit(f"No anchor boards detected in {args.map_path} — nothing written.")

    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {len(result)} anchor pose(s) -> {out_path}")


if __name__ == "__main__":
    main()
