#!/usr/bin/env python3
"""
Merge two TinyNav map directories into one map.

The tool finds a cross-map loop closure by comparing saved DINO embeddings,
verifies candidates with stored SuperPoint features and LightGlue, estimates the
query camera pose in the reference map via PnP, and applies the resulting world
transform to all poses from the second map.
"""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinynav.core.build_map_node import TinyNavDB, find_loop, save_map_occupancy
from tinynav.core.models_trt import LightGlueTRT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two TinyNav map directories")
    parser.add_argument("--map-a", required=True, type=Path, help="Reference map directory")
    parser.add_argument("--map-b", required=True, type=Path, help="Map directory transformed into map A world")
    parser.add_argument("--output", required=True, type=Path, help="Output merged map directory")
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-matches", type=int, default=50)
    parser.add_argument("--min-inliers", type=int, default=50)
    parser.add_argument("--max-loop-pairs", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it exists")
    return parser.parse_args()


def load_poses(map_dir: Path) -> dict[int, np.ndarray]:
    poses_path = map_dir / "poses.npy"
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing poses file: {poses_path}")
    poses = np.load(poses_path, allow_pickle=True).item()
    return {int(ts): np.asarray(pose, dtype=np.float64) for ts, pose in poses.items()}


def load_intrinsics(map_dir: Path) -> np.ndarray:
    intrinsics_path = map_dir / "intrinsics.npy"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Missing intrinsics file: {intrinsics_path}")
    return np.load(intrinsics_path)


def load_baseline(map_dir: Path) -> np.ndarray:
    baseline_path = map_dir / "baseline.npy"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline file: {baseline_path}")
    return np.load(baseline_path)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    if norm <= 0.0:
        return embedding
    return embedding / norm


def match_keypoints(
    matcher: LightGlueTRT,
    loop: asyncio.AbstractEventLoop,
    feats0: dict,
    feats1: dict,
    shape0: tuple[int, int],
    shape1: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    match_result = loop.run_until_complete(
        matcher.infer(
            feats0["kpts"],
            feats1["kpts"],
            feats0["descps"],
            feats1["descps"],
            feats0["mask"],
            feats1["mask"],
            shape0,
            shape1,
        )
    )
    match_indices = match_result["match_indices"][0]
    valid_mask = match_indices != -1
    keypoints0 = feats0["kpts"][0][valid_mask]
    keypoints1 = feats1["kpts"][0][match_indices[valid_mask]]
    matches = np.column_stack((np.flatnonzero(valid_mask), match_indices[valid_mask])).astype(np.int64)
    return keypoints0, keypoints1, matches


def keypoints_to_world_points(
    keypoints: np.ndarray,
    depth: np.ndarray,
    pose_camera_to_world: np.ndarray,
    K: np.ndarray,
    max_depth: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    h, w = depth.shape[:2]
    points_camera = []
    valid = []
    for kp in keypoints:
        u = int(round(float(kp[0])))
        v = int(round(float(kp[1])))
        if 0 <= u < w and 0 <= v < h:
            z = float(depth[v, u])
            is_valid = np.isfinite(z) and 0.0 < z < max_depth
        else:
            z = 0.0
            is_valid = False
        if is_valid:
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points_camera.append([x, y, z])
        else:
            points_camera.append([0.0, 0.0, 0.0])
        valid.append(is_valid)

    points_camera = np.asarray(points_camera, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    R = pose_camera_to_world[:3, :3]
    t = pose_camera_to_world[:3, 3]
    points_world = (R @ points_camera.T).T + t
    return points_world, valid


def estimate_cross_map_transform(
    map_a: Path,
    map_b: Path,
    poses_a: dict[int, np.ndarray],
    poses_b: dict[int, np.ndarray],
    K_a: np.ndarray,
    K_b: np.ndarray,
    similarity_threshold: float,
    top_k: int,
    min_matches: int,
    min_inliers: int,
    max_loop_pairs: int,
) -> tuple[np.ndarray, dict[str, object]]:
    db_a = TinyNavDB(str(map_a), is_scratch=False)
    db_b = TinyNavDB(str(map_b), is_scratch=False)
    matcher = LightGlueTRT()
    loop = asyncio.new_event_loop()
    try:
        timestamps_a = sorted(poses_a.keys())
        timestamps_b = sorted(poses_b.keys())
        raw_embeddings_a = [np.asarray(db_a.get_embedding(ts)).reshape(-1) for ts in timestamps_a]
        raw_embeddings_b = [np.asarray(db_b.get_embedding(ts)).reshape(-1) for ts in timestamps_b]
        zero_embeddings_a = sum(float(np.linalg.norm(embedding)) <= 0.0 for embedding in raw_embeddings_a)
        zero_embeddings_b = sum(float(np.linalg.norm(embedding)) <= 0.0 for embedding in raw_embeddings_b)
        if zero_embeddings_a == len(raw_embeddings_a):
            raise RuntimeError(f"All map A embeddings are zero: {map_a}")
        if zero_embeddings_b == len(raw_embeddings_b):
            raise RuntimeError(f"All map B embeddings are zero: {map_b}")

        embeddings_a = np.stack([normalize_embedding(embedding) for embedding in raw_embeddings_a])
        candidates = []
        stats = {
            "zero_embeddings_a": zero_embeddings_a,
            "zero_embeddings_b": zero_embeddings_b,
            "embedding_candidates": 0,
            "rejected_min_matches": 0,
            "rejected_valid_depth": 0,
            "rejected_pnp": 0,
        }

        for ts_b in tqdm(timestamps_b, desc="Searching cross-map loops", unit="frame"):
            embedding_b = normalize_embedding(db_b.get_embedding(ts_b))
            loop_list = find_loop(embedding_b, embeddings_a, similarity_threshold, top_k)
            stats["embedding_candidates"] += len(loop_list)
            for idx_a, similarity in loop_list:
                ts_a = timestamps_a[int(idx_a)]
                depth_a, _, features_a, _, _ = db_a.get_depth_embedding_features_images(ts_a)
                depth_b, _, features_b, _, _ = db_b.get_depth_embedding_features_images(ts_b)
                kpts_a, kpts_b, matches = match_keypoints(
                    matcher,
                    loop,
                    features_a,
                    features_b,
                    depth_a.shape[:2],
                    depth_b.shape[:2],
                )
                if len(matches) < min_matches:
                    stats["rejected_min_matches"] += 1
                    continue

                points_a_world, valid_depth = keypoints_to_world_points(kpts_a, depth_a, poses_a[ts_a], K_a)
                points_a_world = points_a_world[valid_depth]
                kpts_b_valid = kpts_b[valid_depth]
                if len(points_a_world) < min_inliers:
                    stats["rejected_valid_depth"] += 1
                    continue

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    points_a_world.astype(np.float32),
                    kpts_b_valid.astype(np.float32),
                    K_b,
                    None,
                )
                inlier_count = 0 if inliers is None else int(len(inliers))
                if not success or inlier_count < min_inliers:
                    stats["rejected_pnp"] += 1
                    continue

                R, _ = cv2.Rodrigues(rvec)
                T_b_camera_from_a_world = np.eye(4)
                T_b_camera_from_a_world[:3, :3] = R
                T_b_camera_from_a_world[:3, 3] = tvec.reshape(3)
                T_a_world_from_b_camera = np.linalg.inv(T_b_camera_from_a_world)
                T_a_world_from_b_world = T_a_world_from_b_camera @ np.linalg.inv(poses_b[ts_b])

                score = (inlier_count, float(similarity), len(matches))
                candidate = {
                    "score": score,
                    "T_map_b_to_map_a": T_a_world_from_b_world,
                    "timestamp_a": int(ts_a),
                    "timestamp_b": int(ts_b),
                    "similarity": float(similarity),
                    "matches": int(len(matches)),
                    "valid_depth_matches": int(len(points_a_world)),
                    "inliers": inlier_count,
                }
                candidates.append(candidate)

        if len(candidates) == 0:
            raise RuntimeError(
                "No valid cross-map loop found. "
                f"stats={stats}. Try lowering --similarity-threshold or --min-inliers "
                "if the maps overlap and embeddings are nonzero."
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        selected_candidates = candidates[:max_loop_pairs]
        optimized_transform = optimize_world_transform(selected_candidates, min_inliers)
        best = candidates[0]
        report = {
            "best_loop": {k: v for k, v in best.items() if k not in {"score", "T_map_b_to_map_a"}},
            "num_valid_loop_pairs": len(candidates),
            "num_optimized_loop_pairs": len(selected_candidates),
            "loop_pairs": [
                {k: v for k, v in candidate.items() if k not in {"score", "T_map_b_to_map_a"}}
                for candidate in selected_candidates
            ],
            "search_stats": stats,
        }
        return optimized_transform, report
    finally:
        loop.close()
        db_a.close()
        db_b.close()


def optimize_world_transform(candidates: list[dict[str, object]], min_inliers: int) -> np.ndarray:
    from tinynav.tinynav_cpp_bind import pose_graph_solve

    # Pose 0 is the unknown map-B-to-map-A transform; pose 1 is fixed identity.
    # A constraint (0, 1, T_obs) makes pose_graph_solve optimize pose 0 toward T_obs.
    initial_transform = np.asarray(candidates[0]["T_map_b_to_map_a"], dtype=np.float64)
    optimized_parameters = {
        0: initial_transform,
        1: np.eye(4, dtype=np.float64),
    }
    constant_pose_index = {1: True}
    constraints = []
    for candidate in candidates:
        inliers = int(candidate["inliers"])
        weight_scale = np.clip(inliers / max(1, min_inliers), 1.0, 10.0)
        weight = weight_scale * np.array([10.0, 10.0, 10.0], dtype=np.float64)
        constraints.append(
            (
                0,
                1,
                np.asarray(candidate["T_map_b_to_map_a"], dtype=np.float64),
                weight,
                weight,
            )
        )
    optimized = pose_graph_solve(optimized_parameters, constraints, constant_pose_index, 1000)
    return np.asarray(optimized[0], dtype=np.float64)


def ensure_output_dir(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)


def unique_timestamp(timestamp: int, used: set[int]) -> int:
    candidate = int(timestamp)
    while candidate in used:
        candidate += 1
    used.add(candidate)
    return candidate


def copy_entries(
    src_db: TinyNavDB,
    dst_db: TinyNavDB,
    timestamps: Iterable[int],
    timestamp_map: dict[int, int],
) -> None:
    for timestamp in tqdm(list(timestamps), desc="Copying DB entries", unit="frame"):
        dst_timestamp = timestamp_map[int(timestamp)]
        depth, embedding, features, rgb_loader, infra1_loader = src_db.get_depth_embedding_features_images(timestamp)
        rgb_image = rgb_loader()
        infra1_image = infra1_loader()
        dst_db.set_entry(
            dst_timestamp,
            depth=depth,
            embedding=embedding,
            features=features,
            infra1_image=infra1_image,
            rgb_image=rgb_image,
        )


def copy_optional_static_files(map_a: Path, output: Path) -> None:
    for name in ("T_rgb_to_infra1.npy", "rgb_camera_intrinsics.npy", "tf_messages.npy"):
        src = map_a / name
        if src.exists():
            shutil.copy2(src, output / name)


def write_merged_map(args: argparse.Namespace, T_map_b_to_map_a: np.ndarray, report: dict[str, object]) -> None:
    ensure_output_dir(args.output, args.overwrite)

    poses_a = load_poses(args.map_a)
    poses_b = load_poses(args.map_b)
    K_a = load_intrinsics(args.map_a)
    baseline_a = load_baseline(args.map_a)

    used_timestamps: set[int] = set()
    timestamp_map_a = {ts: unique_timestamp(ts, used_timestamps) for ts in sorted(poses_a.keys())}
    timestamp_map_b = {ts: unique_timestamp(ts, used_timestamps) for ts in sorted(poses_b.keys())}

    merged_poses = {}
    for src_ts, dst_ts in timestamp_map_a.items():
        merged_poses[dst_ts] = poses_a[src_ts]
    for src_ts, dst_ts in timestamp_map_b.items():
        merged_poses[dst_ts] = T_map_b_to_map_a @ poses_b[src_ts]
    merged_poses = dict(sorted(merged_poses.items()))

    np.save(args.output / "poses.npy", merged_poses, allow_pickle=True)
    np.save(args.output / "intrinsics.npy", K_a)
    np.save(args.output / "baseline.npy", baseline_a)
    np.save(args.output / "T_map_b_to_map_a.npy", T_map_b_to_map_a)
    copy_optional_static_files(args.map_a, args.output)

    src_a = TinyNavDB(str(args.map_a), is_scratch=False)
    src_b = TinyNavDB(str(args.map_b), is_scratch=False)
    dst = TinyNavDB(str(args.output), is_scratch=True)
    try:
        copy_entries(src_a, dst, sorted(poses_a.keys()), timestamp_map_a)
        copy_entries(src_b, dst, sorted(poses_b.keys()), timestamp_map_b)
    finally:
        src_a.close()
        src_b.close()
        dst.close()

    report = {
        **report,
        "map_a": str(args.map_a),
        "map_b": str(args.map_b),
        "output": str(args.output),
        "num_poses_a": len(poses_a),
        "num_poses_b": len(poses_b),
        "num_poses_merged": len(merged_poses),
        "timestamp_map_a": {str(k): int(v) for k, v in timestamp_map_a.items()},
        "timestamp_map_b": {str(k): int(v) for k, v in timestamp_map_b.items()},
    }
    (args.output / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    save_map_occupancy(str(args.output), merged_poses, K_a, baseline_a, resolution=0.1, step=10)


def main() -> int:
    args = parse_args()
    poses_a = load_poses(args.map_a)
    poses_b = load_poses(args.map_b)
    K_a = load_intrinsics(args.map_a)
    K_b = load_intrinsics(args.map_b)
    if not np.allclose(K_a, K_b, rtol=1e-3, atol=1e-3):
        print("Warning: map intrinsics differ; output map will use map A intrinsics.")

    T_map_b_to_map_a, report = estimate_cross_map_transform(
        map_a=args.map_a,
        map_b=args.map_b,
        poses_a=poses_a,
        poses_b=poses_b,
        K_a=K_a,
        K_b=K_b,
        similarity_threshold=args.similarity_threshold,
        top_k=args.top_k,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        max_loop_pairs=args.max_loop_pairs,
    )
    write_merged_map(args, T_map_b_to_map_a, report)
    print(f"Merged map written to: {args.output}")
    best_loop = report["best_loop"]
    print(f"Best loop: map_a={best_loop['timestamp_a']} map_b={best_loop['timestamp_b']} inliers={best_loop['inliers']}")
    print(f"Optimized loop pairs: {report['num_optimized_loop_pairs']}/{report['num_valid_loop_pairs']}")
    print(f"T_map_b_to_map_a saved to: {args.output / 'T_map_b_to_map_a.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
