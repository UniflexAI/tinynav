#!/usr/bin/env python3
"""Calibrate the rigid transform between two independently-built tinynav maps
by cross-map relocalization.

Each of --source-map's keyframes is treated as a "query": it's retrieved
against --target-map's own DINOv2-patch-VLAD index (the same index
map_node.py's relocalize_with_depth() searches at runtime), matched against
the retrieved candidates with SuperPoint+LightGlue, and localized in the
target map's frame via PnP -- exactly map_node.py's live relocalization
pipeline, just run offline against a stored keyframe instead of a live camera
frame. Since every query keyframe already has a known pose in the source
map's own frame, each successful relocalization gives one (position in
source-map frame, position in target-map frame) correspondence for the same
physical point. The run repeats in both directions (source keyframes queried
against target, and target keyframes queried against source) to spread
samples across whatever part of each map the two sessions actually
overlapped, then fits T_source_to_target with RANSAC+Kabsch over all
collected correspondences.

This intentionally does NOT touch the app/backend/node_manager.py map-handoff
code's consumption side (map_handoff_from_<source>.json / "mapA_to_mapB")
-- it only produces the calibrated transform + quality metrics. Wire the
output into whatever schema you're actually shipping once this is verified.

Usage:
    uv run python tool/calibrate_map_transform.py \\
        --source-map /mnt/nas/share-all/tinynav/map/out_Newdoor08_19_12_28_10 \\
        --target-map /mnt/nas/share-all/tinynav/map/out_water_08_11_10_28_45 \\
        --output /tmp/newdoor_to_water.json
"""

import argparse
import asyncio
import json
import shelve
import time
from dataclasses import dataclass

import cv2
import numpy as np

from tinynav.core.models_trt import LightGlueTRT
from tinynav.core.vlad import compute_vlad


def find_loop(target_embedding: np.ndarray, embeddings: np.ndarray, loop_similarity_threshold: float, loop_top_k: int) -> list[tuple[int, float]]:
    """Same as tinynav.core.build_map_node.find_loop, inlined so this stays a
    read-only tool: build_map_node imports tool.video_db -> decord for its
    (write-mode) map-building path, which this script has no use for and
    doesn't want as a hard dependency just to read a prebuilt map's poses.
    """
    if len(embeddings) == 0:
        return []
    similarity = embeddings @ target_embedding
    order = np.argsort(similarity)
    return [(int(idx), float(similarity[idx])) for idx in order if similarity[idx] > loop_similarity_threshold][-loop_top_k:]


class ReadOnlyMapDB:
    """Read-only equivalent of tinynav.core.build_map_node.TinyNavDB's shelf
    access, for a NAS-mounted map directory that may be read-only. TinyNavDB's
    IntKeyShelf always does shelve.open(filename) with no flag (mode 'c',
    create/read-write), which fails outright against a read-only mount;
    this only opens what a calibration pass needs (features/depths/patch
    tokens), read-only, and never touches image data.
    """

    def __init__(self, map_path: str):
        self._features = shelve.open(f"{map_path}/features", flag="r")
        self._depths = shelve.open(f"{map_path}/depths", flag="r")
        self._patch_tokens = shelve.open(f"{map_path}/patch_tokens", flag="r")

    def features(self, key: int) -> dict:
        return self._features[str(key)]

    def depth(self, key: int) -> np.ndarray:
        return np.array(self._depths[str(key)])

    def patch_tokens(self, key: int) -> np.ndarray:
        return np.array(self._patch_tokens[str(key)])

    def close(self):
        self._features.close()
        self._depths.close()
        self._patch_tokens.close()


@dataclass
class LoadedMap:
    path: str
    poses: dict  # timestamp_ns -> 4x4 camera-to-map-world
    K: np.ndarray
    db: ReadOnlyMapDB
    vlad_centres: np.ndarray
    timestamps: list
    vlad_descriptors: np.ndarray  # (N, K*C), this map's own keyframes embedded in its own vocab
    image_hw: tuple  # (height, width), from a sample depth map


def load_map(map_path: str, max_keyframes: int | None, rng: np.random.Generator) -> LoadedMap:
    poses = np.load(f"{map_path}/poses.npy", allow_pickle=True).item()
    K = np.load(f"{map_path}/intrinsics.npy")
    db = ReadOnlyMapDB(map_path)
    with shelve.open(f"{map_path}/metadata", flag="r") as md:
        vlad_centres = np.array(md["vlad_centres"])

    timestamps = list(poses.keys())
    if max_keyframes is not None and len(timestamps) > max_keyframes:
        timestamps = list(rng.choice(timestamps, size=max_keyframes, replace=False))

    vlad_descriptors = np.stack([
        compute_vlad(db.patch_tokens(t), vlad_centres) for t in timestamps
    ])
    sample_depth = db.depth(timestamps[0])
    return LoadedMap(map_path, poses, K, db, vlad_centres, timestamps, vlad_descriptors, sample_depth.shape)


def match_keypoints(feats_ref: dict, feats_query: dict, light_glue: LightGlueTRT, image_shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    match_result = asyncio.run(light_glue.infer(
        feats_ref["kpts"], feats_query["kpts"], feats_ref["descps"], feats_query["descps"],
        feats_ref["mask"], feats_query["mask"], image_shape, image_shape,
    ))
    match_indices = match_result["match_indices"][0]
    valid_mask = match_indices != -1
    kp_ref = feats_ref["kpts"][0][valid_mask]
    kp_query = feats_query["kpts"][0][match_indices[valid_mask]]
    return kp_ref, kp_query


def keypoints_with_depth_to_world(keypoints: np.ndarray, depth: np.ndarray, pose_camera_to_world: np.ndarray, K: np.ndarray, max_depth_m: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    u = keypoints[:, 0].astype(np.int64)
    v = keypoints[:, 1].astype(np.int64)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    z = np.zeros(len(keypoints), dtype=np.float32)
    z[in_bounds] = depth[v[in_bounds], u[in_bounds]]
    valid = in_bounds & (z > 0) & (z < max_depth_m)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_camera = np.stack([x, y, z], axis=1)

    R = pose_camera_to_world[:3, :3]
    t = pose_camera_to_world[:3, 3]
    points_world = (R @ points_camera.T).T + t
    return points_world, valid


def relocalize_keyframe_in_map(
    query_features: dict, query_patch_tokens: np.ndarray, query_K: np.ndarray, target: LoadedMap,
    light_glue: LightGlueTRT, image_shape: np.ndarray, top_k: int, min_matches: int, min_landmarks: int,
) -> tuple[bool, np.ndarray, int, int]:
    """Returns (success, pose_camera_to_target_world, num_inliers, num_points)."""
    query_vlad = compute_vlad(query_patch_tokens, target.vlad_centres)
    candidates = find_loop(query_vlad, target.vlad_descriptors, -1.0, top_k)
    if not candidates:
        return False, np.eye(4), 0, 0

    pnp_candidates = []
    for idx, _similarity in candidates:
        t_ref = target.timestamps[idx]
        reference_features = target.db.features(t_ref)
        reference_depth = target.db.depth(t_ref)
        reference_kp, query_kp = match_keypoints(reference_features, query_features, light_glue, image_shape)
        if len(reference_kp) < min_matches:
            continue
        points_world, valid = keypoints_with_depth_to_world(reference_kp, reference_depth, target.poses[t_ref], target.K)
        points_world = points_world[valid]
        points_2d = query_kp[valid]
        if len(points_2d) <= min_landmarks:
            continue
        pnp_candidates.append((points_world.astype(np.float64), points_2d.astype(np.float64)))

    best = None
    for points_3d, points_2d in pnp_candidates:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, query_K, None)
        num_inliers = 0 if inliers is None else len(inliers)
        if not success or num_inliers < min_landmarks // 2:
            continue
        if best is None or num_inliers > best[2]:
            R, _ = cv2.Rodrigues(rvec)
            pose_world_to_camera = np.eye(4)
            pose_world_to_camera[:3, :3] = R
            pose_world_to_camera[:3, 3] = tvec.reshape(3)
            best = (np.linalg.inv(pose_world_to_camera), num_inliers, len(points_2d))

    if best is None:
        return False, np.eye(4), 0, 0
    pose_camera_to_world, num_inliers, num_points = best
    return True, pose_camera_to_world, num_inliers, num_points


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid R, t minimizing sum ||Q_i - (R @ P_i + t)||^2."""
    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - p_mean, Q - q_mean
    H = Pc.T @ Qc
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = q_mean - R @ p_mean
    return R, t


def ransac_rigid_fit(P: np.ndarray, Q: np.ndarray, inlier_thresh_m: float, num_iters: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(P)
    best_inliers = np.zeros(n, dtype=bool)
    for _ in range(num_iters):
        idx = rng.choice(n, size=3, replace=False)
        R, t = kabsch(P[idx], Q[idx])
        residuals = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
        inliers = residuals < inlier_thresh_m
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    R, t = kabsch(P[best_inliers], Q[best_inliers])
    residuals = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
    return R, t, residuals


def yaw_pitch_roll_deg(R: np.ndarray) -> tuple[float, float, float]:
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-map", required=True)
    parser.add_argument("--target-map", required=True)
    parser.add_argument("--output", default=None, help="Write the resulting T_source_to_target + quality metrics as JSON here")
    parser.add_argument("--max-keyframes-per-map", type=int, default=150, help="Cap on query keyframes sampled from EACH map (queried in both directions, so total relocalization attempts is roughly double this)")
    parser.add_argument("--top-k", type=int, default=5, help="VLAD retrieval candidates tried per query keyframe")
    parser.add_argument("--min-matches", type=int, default=50, help="Minimum LightGlue matches for a retrieval candidate to be tried in PnP")
    parser.add_argument("--min-landmarks", type=int, default=80, help="Minimum valid (matched + has depth) 3D-2D correspondences for a PnP candidate")
    parser.add_argument("--ransac-inlier-thresh-m", type=float, default=0.3, help="RANSAC inlier threshold (meters) on the final source/target position correspondences")
    parser.add_argument("--ransac-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading source map: {args.source_map}")
    source = load_map(args.source_map, args.max_keyframes_per_map, rng)
    print(f"  {len(source.timestamps)} sampled keyframes, K=\n{source.K}")
    print(f"Loading target map: {args.target_map}")
    target = load_map(args.target_map, args.max_keyframes_per_map, rng)
    print(f"  {len(target.timestamps)} sampled keyframes, K=\n{target.K}")

    light_glue = LightGlueTRT()
    source_image_shape = np.array([source.image_hw[1], source.image_hw[0]], dtype=np.int64)  # (width, height)
    target_image_shape = np.array([target.image_hw[1], target.image_hw[0]], dtype=np.int64)

    points_source, points_target = [], []
    t0 = time.time()
    for direction, (query_map, ref_map, query_image_shape) in enumerate([
        (source, target, source_image_shape),
        (target, source, target_image_shape),
    ]):
        label = "source->target" if direction == 0 else "target->source"
        num_success = 0
        for i, t_query in enumerate(query_map.timestamps):
            query_features = query_map.db.features(t_query)
            query_patch_tokens = query_map.db.patch_tokens(t_query)
            ok, pose_in_ref_world, num_inliers, num_points = relocalize_keyframe_in_map(
                query_features, query_patch_tokens, query_map.K, ref_map, light_glue, query_image_shape,
                args.top_k, args.min_matches, args.min_landmarks,
            )
            if ok:
                num_success += 1
                p_query_frame = query_map.poses[t_query][:3, 3]
                p_ref_frame = pose_in_ref_world[:3, 3]
                if direction == 0:
                    points_source.append(p_query_frame)
                    points_target.append(p_ref_frame)
                else:
                    points_source.append(p_ref_frame)
                    points_target.append(p_query_frame)
            if (i + 1) % 25 == 0:
                print(f"  [{label}] {i + 1}/{len(query_map.timestamps)} queries, {num_success} relocalized so far "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)
        print(f"[{label}] done: {num_success}/{len(query_map.timestamps)} keyframes relocalized")

    if len(points_source) < 10:
        print(f"FAIL: only {len(points_source)} cross-map correspondences found -- maps likely don't overlap, "
              f"or intrinsics/feature settings need adjusting. Cannot calibrate.")
        return 1

    P = np.array(points_source)
    Q = np.array(points_target)
    R, t, residuals = ransac_rigid_fit(P, Q, args.ransac_inlier_thresh_m, args.ransac_iters, rng)
    inliers = residuals < args.ransac_inlier_thresh_m
    yaw, pitch, roll = yaw_pitch_roll_deg(R)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    print(f"\n=== T_source_to_target ===\n{T}")
    print(f"yaw={yaw:.2f}deg pitch={pitch:.2f}deg roll={roll:.2f}deg")
    print(f"samples={len(P)} inliers={int(inliers.sum())} "
          f"rms_inlier_residual_m={float(np.sqrt(np.mean(residuals[inliers] ** 2))):.4f} "
          f"max_inlier_residual_m={float(residuals[inliers].max()):.4f}")
    print(f"overlap bbox in source frame: min={P[inliers].min(axis=0)} max={P[inliers].max(axis=0)}")

    if args.output:
        result = dict(
            source=args.source_map, target=args.target_map,
            T_source_to_target=T.tolist(),
            yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
            n_samples=len(P), n_inliers=int(inliers.sum()),
            rms_m=float(np.sqrt(np.mean(residuals[inliers] ** 2))),
            max_m=float(residuals[inliers].max()),
            overlap_bbox=[P[inliers].min(axis=0).tolist(), P[inliers].max(axis=0).tolist()],
        )
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
