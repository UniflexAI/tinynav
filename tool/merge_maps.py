"""Merge two tinynav maps into one.

Given map_a (base) and map_b (to be merged in), this:
  1. Relocalizes a sample of map_b's keyframes against map_a (embedding retrieval +
     LightGlue matching + PnP, the same pipeline MapNode.relocalize_with_depth uses
     online) to get each sampled map_b keyframe's pose in map_a's world frame.
  2. Fits a rigid transform T (map_b -> map_a) from those pose correspondences via
     RANSAC + Kabsch/SVD (same approach as
     tool/benchmark/benchmark_mapping.py:BenchmarkResults.compute_transformation).
     Exits with an error if too few keyframes relocalize or the fit has too few
     inliers.
  3. Fuses map_b's keyframe data (features/embeddings/depths/images), transformed by
     T, into map_a's frame, regenerates the occupancy grid / SDF map and (optionally)
     the VLAD retrieval index over the full merged keyframe set.
  4. Saves the result as a new map_c directory.

Runs offline (no ROS node/bag replay needed) inside the tinynav-dev container with
ROS + the project venv sourced, e.g.:
  /opt/venv/bin/python tool/merge_maps.py --map-a <dir> --map-b <dir> --map-c <dir>
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro

from tinynav.core.build_map_node import TinyNavDB, find_loop, generate_occupancy_map
from tinynav.core.math_utils import rerank_by_pnp_inliers
from tinynav.core.models_trt import Dinov2TRT, LightGlueTRT
from tinynav.core.vlad import compute_vlad_batch, train_vocabulary


@dataclass(frozen=True)
class Args:
    map_a: Path
    """Base map directory; the merged map is built in this map's world frame."""

    map_b: Path
    """Map to merge into map_a. Its keyframes are relocalized against map_a to fit
    the transform, then fused (transformed into map_a's frame) into the output."""

    map_c: Path
    """Output directory for the merged map (must not already exist)."""

    query_stride: int = 5
    """Use every Nth map_b keyframe (by pose timestamp order) as a relocalization query."""

    retrieval_top_k: int = 5
    """Number of map_a candidate keyframes to try per map_b query (by embedding similarity)."""

    retrieval_similarity_threshold: float = -1.0
    """Minimum cosine similarity for a map_a candidate to be considered (-1.0 = no filtering)."""

    min_match_count: int = 40
    """Minimum LightGlue matches between a query and a candidate keyframe to attempt PnP."""

    min_landmark_count: int = 60
    """Minimum valid (depth-backed) 3D landmarks required to attempt PnP for a candidate."""

    min_inlier_count: int = 50
    """Minimum PnP-RANSAC inliers required to accept a single-keyframe relocalization."""

    min_correspondences: int = 3
    """Minimum number of successfully relocalized map_b keyframes before attempting the fit."""

    fit_ransac_iterations: int = 1000
    """RANSAC iterations for the map_b -> map_a rigid transform fit."""

    fit_inlier_threshold_m: float = 0.2
    """RANSAC inlier distance threshold (meters) for the rigid transform fit."""

    min_fit_inliers: int = 5
    """Minimum RANSAC inliers required to accept the fitted transform; otherwise exit(1)."""

    occupancy_resolution: float = 0.1
    """Voxel resolution (meters) for the regenerated occupancy grid / SDF map."""

    occupancy_step: int = 10
    """Raycasting depth-pixel stride for the regenerated occupancy grid (see build_map_node.py)."""

    rebuild_vlad: bool = True
    """Recompute the VLAD vocabulary/descriptors over the full merged keyframe set."""

    vlad_vocab_size: int = 32
    vlad_iterations: int = 200
    vlad_batch_size: int = 1024
    vlad_seed: int = 42


def load_poses(map_path: Path) -> dict[int, np.ndarray]:
    return np.load(map_path / "poses.npy", allow_pickle=True).item()


def build_embedding_matrix(db: TinyNavDB, timestamps: list[int]) -> tuple[list[int], np.ndarray]:
    valid_ts = []
    embeddings = []
    for ts in timestamps:
        emb = db.get_embedding(ts)
        if emb is not None:
            valid_ts.append(ts)
            embeddings.append(emb)
    if not embeddings:
        return [], np.zeros((0, 0), dtype=np.float32)
    return valid_ts, np.stack(embeddings)


def match_keypoints(
    light_glue: LightGlueTRT,
    feats0: dict,
    feats1: dict,
    image_shape0: np.ndarray,
    image_shape1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-map SuperPoint/LightGlue matching.

    Same logic as BuildMapNode.match_keypoints / MapNode.match_keypoints
    (tinynav/core/build_map_node.py:857, tinynav/core/map_node.py:1903), generalized
    to accept per-image shapes since the two keyframes may come from different maps.
    """
    match_result = asyncio.run(light_glue.infer(
        feats0["kpts"], feats1["kpts"], feats0["descps"], feats1["descps"],
        feats0["mask"], feats1["mask"], image_shape0, image_shape1,
    ))
    match_indices = match_result["match_indices"][0]
    valid_mask = match_indices != -1
    keypoints0 = feats0["kpts"][0][valid_mask]
    keypoints1 = feats1["kpts"][0][match_indices[valid_mask]]
    return keypoints0, keypoints1


def keypoint_with_depth_to_3d(
    keypoints: np.ndarray, depth: np.ndarray, pose_from_camera_to_world: np.ndarray, K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of MapNode.keypoint_with_depth_to_3d (tinynav/core/map_node.py:2169)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    point_in_camera = []
    inliers = []
    for kp in keypoints:
        u, v = int(kp[0]), int(kp[1])
        Z = depth[v, u]
        if Z > 0 and Z < 50:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            inliers.append(True)
        else:
            X, Y = 0, 0
            inliers.append(False)
        point_in_camera.append(np.array([X, Y, Z]))
    point_in_camera = np.array(point_in_camera)
    inliers = np.array(inliers)
    rotation = pose_from_camera_to_world[:3, :3]
    translation = pose_from_camera_to_world[:3, 3]
    point_in_world = (rotation @ point_in_camera.T).T + translation
    return point_in_world, inliers


def estimate_rigid_transform(points_src: np.ndarray, points_dst: np.ndarray) -> np.ndarray | None:
    """Kabsch/SVD rigid (SE3, no scale) fit mapping points_src -> points_dst.

    Same algorithm as BenchmarkResults._estimate_rigid_transform
    (tool/benchmark/benchmark_mapping.py:496).
    """
    if len(points_src) != len(points_dst) or len(points_src) < 3:
        return None
    centroid_src = points_src.mean(axis=0)
    centroid_dst = points_dst.mean(axis=0)
    H = (points_src - centroid_src).T @ (points_dst - centroid_dst)
    U, _, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = Vt.T @ U.T
    t = centroid_dst - Rm @ centroid_src
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = t
    return T


def ransac_rigid_transform(
    points_src: np.ndarray, points_dst: np.ndarray, iterations: int, inlier_threshold_m: float,
) -> tuple[np.ndarray | None, int]:
    """RANSAC-wrapped rigid transform fit; same approach as
    BenchmarkResults.compute_transformation (tool/benchmark/benchmark_mapping.py:432)."""
    n = len(points_src)
    if n < 3:
        return None, 0
    best_T = None
    best_inliers = 0
    for _ in range(iterations):
        idx = np.random.choice(n, 3, replace=False)
        T = estimate_rigid_transform(points_src[idx], points_dst[idx])
        if T is None:
            continue
        transformed = (T[:3, :3] @ points_src.T).T + T[:3, 3]
        inlier_count = int(np.sum(np.linalg.norm(transformed - points_dst, axis=1) < inlier_threshold_m))
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_T = T
    if best_T is None:
        return None, 0
    transformed = (best_T[:3, :3] @ points_src.T).T + best_T[:3, 3]
    inlier_mask = np.linalg.norm(transformed - points_dst, axis=1) < inlier_threshold_m
    if int(np.sum(inlier_mask)) >= 3:
        refined_T = estimate_rigid_transform(points_src[inlier_mask], points_dst[inlier_mask])
        if refined_T is not None:
            best_T = refined_T
    return best_T, best_inliers


def relocalize_keyframe_in_map_a(
    ts_b: int,
    db_b: TinyNavDB,
    db_a: TinyNavDB,
    map_a_ts_list: list[int],
    map_a_embeddings: np.ndarray,
    poses_a: dict[int, np.ndarray],
    K_a: np.ndarray,
    light_glue: LightGlueTRT,
    args: Args,
) -> np.ndarray | None:
    """Relocalize one map_b keyframe against map_a.

    Returns the keyframe's camera-to-world pose in map_a's frame, or None if
    relocalization fails. Mirrors MapNode.relocalize_with_depth's DINO-embedding
    fallback path (tinynav/core/map_node.py:2073-2144).
    """
    depth_b, emb_b, feats_b, _, _ = db_b.get_depth_embedding_features_images(ts_b)
    if depth_b is None or emb_b is None or feats_b is None:
        return None
    candidates = find_loop(emb_b, map_a_embeddings, args.retrieval_similarity_threshold, args.retrieval_top_k)
    if not candidates:
        return None
    image_shape_b = np.array([depth_b.shape[1], depth_b.shape[0]], dtype=np.int64)

    pnp_candidates = []
    for idx_in_map, _similarity in candidates:
        ts_a = map_a_ts_list[int(idx_in_map)]
        depth_a, _, feats_a, _, _ = db_a.get_depth_embedding_features_images(ts_a)
        if depth_a is None or feats_a is None:
            continue
        image_shape_a = np.array([depth_a.shape[1], depth_a.shape[0]], dtype=np.int64)
        keypoints_a, keypoints_b = match_keypoints(light_glue, feats_a, feats_b, image_shape_a, image_shape_b)
        if len(keypoints_a) < args.min_match_count:
            continue
        points_3d_world, inliers = keypoint_with_depth_to_3d(keypoints_a, depth_a, poses_a[ts_a], K_a)
        points_3d_world = points_3d_world[inliers]
        points_2d_query = keypoints_b[inliers]
        if len(points_2d_query) <= args.min_landmark_count:
            continue
        pnp_candidates.append((points_3d_world, points_2d_query))

    if not pnp_candidates:
        return None
    success, pose_world_to_camera, _ratio, _idx, _inliers, _total = rerank_by_pnp_inliers(
        pnp_candidates, K_a, min_inlier_count=args.min_inlier_count,
    )
    if not success:
        return None
    return np.linalg.inv(pose_world_to_camera)


def fit_transform_b_to_a(
    map_a: Path, map_b: Path, poses_a: dict[int, np.ndarray], poses_b: dict[int, np.ndarray], args: Args,
) -> np.ndarray:
    K_a = np.load(map_a / "intrinsics.npy")
    K_b = np.load(map_b / "intrinsics.npy")
    if not np.allclose(np.diag(K_a)[:2], np.diag(K_b)[:2], rtol=0.05):
        print(
            f"warning: map_a/map_b intrinsics differ notably (K_a diag={np.diag(K_a)[:2]}, "
            f"K_b diag={np.diag(K_b)[:2]}); relocalization assumes the same camera, matching "
            "MapNode.relocalize_with_depth's existing convention of using the map's own K "
            "for both 3D lifting and PnP."
        )

    db_a = TinyNavDB(str(map_a), is_scratch=False)
    db_b = TinyNavDB(str(map_b), is_scratch=False)
    light_glue = LightGlueTRT()

    map_a_ts_list, map_a_embeddings = build_embedding_matrix(db_a, sorted(poses_a.keys()))
    if len(map_a_ts_list) == 0:
        db_a.close()
        db_b.close()
        print("error: map_a has no usable embeddings to query against.")
        sys.exit(1)

    query_ts_list = sorted(poses_b.keys())[:: args.query_stride]
    print(
        f"Querying {len(query_ts_list)}/{len(poses_b)} map_b keyframes against map_a "
        f"({len(map_a_ts_list)} keyframes)..."
    )

    points_src = []  # map_b's own-frame translations
    points_dst = []  # corresponding recovered translations in map_a's frame
    for i, ts_b in enumerate(query_ts_list):
        pose_in_a = relocalize_keyframe_in_map_a(
            ts_b, db_b, db_a, map_a_ts_list, map_a_embeddings, poses_a, K_a, light_glue, args,
        )
        if pose_in_a is not None:
            points_src.append(poses_b[ts_b][:3, 3])
            points_dst.append(pose_in_a[:3, 3])
        if (i + 1) % 10 == 0 or i + 1 == len(query_ts_list):
            print(f"  [{i + 1}/{len(query_ts_list)}] relocalized {len(points_src)} so far")

    db_a.close()
    db_b.close()

    if len(points_src) < args.min_correspondences:
        print(
            f"error: only {len(points_src)} map_b keyframes relocalized in map_a "
            f"(< min_correspondences={args.min_correspondences}); refusing to fit a transform."
        )
        sys.exit(1)

    points_src_arr = np.array(points_src)
    points_dst_arr = np.array(points_dst)
    T_b_to_a, inlier_count = ransac_rigid_transform(
        points_src_arr, points_dst_arr, args.fit_ransac_iterations, args.fit_inlier_threshold_m,
    )
    if T_b_to_a is None or inlier_count < args.min_fit_inliers:
        print(
            f"error: transform fit failed (inliers={inlier_count}, need >= {args.min_fit_inliers} "
            f"of {len(points_src)} correspondences)."
        )
        sys.exit(1)

    print(f"Fitted map_b -> map_a transform with {inlier_count}/{len(points_src)} inliers:\n{T_b_to_a}")
    return T_b_to_a


def _copy_keyframe(ts: int, src_db: TinyNavDB, dst_db: TinyNavDB) -> None:
    depth, embedding, features, rgb_loader, infra1_loader = src_db.get_depth_embedding_features_images(ts)
    semantic_embedding = src_db.semantic_embeddings.get(ts)
    dst_db.set_entry(
        ts,
        depth=depth,
        embedding=embedding,
        semantic_embedding=semantic_embedding,
        features=features,
        infra1_image=infra1_loader(),
        rgb_image=rgb_loader(),
    )


def _rebuild_vlad(db: TinyNavDB, poses: dict[int, np.ndarray], map_c: Path, args: Args) -> None:
    """Recompute VLAD vocabulary/descriptors over the merged keyframe set, same
    procedure as BuildMapNode._save_vlad (tinynav/core/build_map_node.py:948)."""
    dinov2 = Dinov2TRT()
    timestamps = sorted(poses.keys())
    patch_tokens_by_ts: dict[int, np.ndarray] = {}
    for i, ts in enumerate(timestamps):
        _, _, _, _, infra1_loader = db.get_depth_embedding_features_images(ts)
        image = infra1_loader()
        if image is None:
            continue
        patch_tokens_by_ts[ts] = asyncio.run(dinov2.infer_patch_tokens(image))
        if (i + 1) % 100 == 0:
            print(f"  VLAD patch tokens: {i + 1}/{len(timestamps)}")
    if not patch_tokens_by_ts:
        print("warning: no images available to recompute VLAD; skipping.")
        return
    kept_ts = sorted(patch_tokens_by_ts.keys())
    all_tokens = np.concatenate([patch_tokens_by_ts[ts] for ts in kept_ts], axis=0)
    centres = train_vocabulary(
        all_tokens,
        vocab_size=args.vlad_vocab_size,
        iterations=args.vlad_iterations,
        batch_size=args.vlad_batch_size,
        seed=args.vlad_seed,
    )
    descriptors = compute_vlad_batch([patch_tokens_by_ts[ts] for ts in kept_ts], centres)
    np.save(map_c / "vlad_vocab.npy", centres)
    np.save(map_c / "vlad_descriptors.npy", descriptors)
    np.save(map_c / "vlad_timestamps.npy", np.array(kept_ts, dtype=np.int64))
    print(f"VLAD saved: vocab={centres.shape}, descriptors={descriptors.shape}, keyframes={len(kept_ts)}")


def merge_maps(
    map_a: Path,
    map_b: Path,
    map_c: Path,
    poses_a: dict[int, np.ndarray],
    poses_b: dict[int, np.ndarray],
    T_b_to_a: np.ndarray,
    args: Args,
) -> None:
    os.makedirs(map_c)

    poses_b_in_a = {ts: T_b_to_a @ pose for ts, pose in poses_b.items()}
    collisions = set(poses_a) & set(poses_b_in_a)
    if collisions:
        print(
            f"error: map_a and map_b share {len(collisions)} keyframe timestamp(s); "
            "cannot merge (would silently overwrite data)."
        )
        sys.exit(1)
    merged_poses = {**poses_a, **poses_b_in_a}
    np.save(map_c / "poses.npy", merged_poses, allow_pickle=True)

    for name in ("intrinsics.npy", "baseline.npy", "T_rgb_to_infra1.npy", "rgb_camera_intrinsics.npy"):
        src = map_a / name
        if src.exists():
            shutil.copy2(src, map_c / name)
    if (map_a / "paths.json").exists():
        shutil.copy2(map_a / "paths.json", map_c / "paths.json")
        print(
            "warning: map_a has paths.json (POI path edits); the merged occupancy/SDF grid is "
            "regenerated from raw keyframe depth and does NOT re-apply those edits. Re-run "
            "map_editor.py's Build/Replace on map_c if you need them back."
        )
    if (map_a / "sdf_map.default.npy").exists():
        shutil.copy2(map_a / "sdf_map.default.npy", map_c / "sdf_map.default.npy")

    K_a = np.load(map_a / "intrinsics.npy")
    baseline_a = np.load(map_a / "baseline.npy")

    print("Fusing keyframe databases (features/embeddings/depths/images)...")
    db_a = TinyNavDB(str(map_a), is_scratch=False)
    db_b = TinyNavDB(str(map_b), is_scratch=False)
    db_c = TinyNavDB(str(map_c), is_scratch=True)
    for ts in sorted(poses_a.keys()):
        _copy_keyframe(ts, db_a, db_c)
    for ts in sorted(poses_b.keys()):
        _copy_keyframe(ts, db_b, db_c)
    db_a.close()
    db_b.close()
    db_c.close()

    db_c_ro = TinyNavDB(str(map_c), is_scratch=False)
    print("Regenerating occupancy grid / SDF map for the merged map...")
    occupancy_grid, occupancy_origin, occupancy_2d_image, sdf_map = generate_occupancy_map(
        merged_poses, db_c_ro, K_a, baseline_a, args.occupancy_resolution, args.occupancy_step,
    )
    occupancy_meta = np.array(
        [occupancy_origin[0], occupancy_origin[1], occupancy_origin[2], args.occupancy_resolution],
        dtype=np.float32,
    )
    np.save(map_c / "occupancy_grid.npy", occupancy_grid)
    np.save(map_c / "occupancy_meta.npy", occupancy_meta)
    np.save(map_c / "sdf_map.npy", sdf_map)
    cv2.imwrite(str(map_c / "occupancy_2d_image.png"), occupancy_2d_image)

    if args.rebuild_vlad:
        print("Recomputing VLAD vocabulary/descriptors over the merged keyframe set...")
        _rebuild_vlad(db_c_ro, merged_poses, map_c, args)

    db_c_ro.close()
    print(f"Merged map saved to {map_c} ({len(poses_a)} + {len(poses_b)} = {len(merged_poses)} keyframes).")


def main(args: Args) -> None:
    for p, name in ((args.map_a, "map_a"), (args.map_b, "map_b")):
        if not (p / "poses.npy").exists():
            print(f"error: {name} ({p}) does not look like a built tinynav map (missing poses.npy)")
            sys.exit(1)
    if args.map_c.exists():
        print(f"error: map_c already exists: {args.map_c}")
        sys.exit(1)
    if args.map_a.resolve() == args.map_b.resolve():
        print("error: map_a and map_b must be different maps")
        sys.exit(1)

    poses_a = load_poses(args.map_a)
    poses_b = load_poses(args.map_b)

    T_b_to_a = fit_transform_b_to_a(args.map_a, args.map_b, poses_a, poses_b, args)
    merge_maps(args.map_a, args.map_b, args.map_c, poses_a, poses_b, T_b_to_a, args)


if __name__ == "__main__":
    main(tyro.cli(Args))
