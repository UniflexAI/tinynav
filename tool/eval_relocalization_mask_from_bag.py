from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

from tinynav.core.build_map_node import TinyNavDB, find_loop
from tinynav.core.math_utils import rerank_by_pnp_inliers
from tinynav.core.models_trt import Dinov2TRT, LightGlueTRT, SuperPointTRT


def load_mask(map_path: str) -> set[int]:
    mask_path = os.path.join(map_path, "relocalization_mask.json")
    if not os.path.exists(mask_path):
        return set()
    with open(mask_path, "r") as f:
        mask = json.load(f)
    return {int(timestamp) for timestamp in mask.get("excluded_timestamps", [])}


def load_map_embeddings(map_path: str, use_mask: bool) -> tuple[list[int], np.ndarray, TinyNavDB, set[int]]:
    db = TinyNavDB(map_path, is_scratch=False)
    poses = np.load(os.path.join(map_path, "poses.npy"), allow_pickle=True).item()
    excluded_timestamps = load_mask(map_path) if use_mask else set()
    timestamps = [
        int(timestamp)
        for timestamp in poses.keys()
        if int(timestamp) not in excluded_timestamps
    ]
    if len(timestamps) == 0:
        raise RuntimeError("No map keyframes remain after applying relocalization mask")

    embeddings = []
    for timestamp in timestamps:
        embedding = db.get_embedding(timestamp).astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        embeddings.append(embedding)
    return timestamps, np.stack(embeddings), db, excluded_timestamps


def iter_images_from_bag(bag_path: str, topic: str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topics = {topic_info.name: topic_info.type for topic_info in reader.get_all_topics_and_types()}
    if topic not in topics:
        raise ValueError(f"Topic not found in bag: {topic}. Available topics: {sorted(topics)}")
    msg_type = get_message(topics[topic])
    bridge = CvBridge()

    while reader.has_next():
        topic_name, raw, timestamp_ns = reader.read_next()
        if topic_name != topic:
            continue
        msg = deserialize_message(raw, msg_type)
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        yield int(timestamp_ns), image


def match_keypoints(
    lightglue: LightGlueTRT,
    feats0: dict,
    feats1: dict,
    image_shape: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    match_result = asyncio.run(
        lightglue.infer(
            feats0["kpts"],
            feats1["kpts"],
            feats0["descps"],
            feats1["descps"],
            feats0["mask"],
            feats1["mask"],
            image_shape,
            image_shape,
        )
    )
    match_indices = match_result["match_indices"][0]
    valid_mask = match_indices != -1
    keypoints0 = feats0["kpts"][0][valid_mask]
    keypoints1 = feats1["kpts"][0][match_indices[valid_mask]]
    matches = [[i, int(index)] for i, index in enumerate(match_indices) if index != -1]
    return keypoints0, keypoints1, np.asarray(matches, dtype=np.int64)


def keypoint_with_depth_to_3d(
    keypoints: np.ndarray,
    depth: np.ndarray,
    pose_from_camera_to_world: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = []
    valid = []
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = depth.shape[:2]
    for kp in keypoints:
        u = int(kp[0])
        v = int(kp[1])
        if u < 0 or u >= w or v < 0 or v >= h:
            points.append(np.zeros(3, dtype=np.float32))
            valid.append(False)
            continue
        z = float(depth[v, u])
        if z > 0.0 and z < 50.0:
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            valid.append(True)
        else:
            x = 0.0
            y = 0.0
            valid.append(False)
        points.append(np.asarray([x, y, z], dtype=np.float32))
    points = np.asarray(points, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    rotation = pose_from_camera_to_world[:3, :3]
    translation = pose_from_camera_to_world[:3, 3]
    points_world = (rotation @ points.T).T + translation
    return points_world, valid


def run(args: argparse.Namespace) -> dict:
    map_timestamps, map_embeddings, db, excluded_timestamps = load_map_embeddings(args.map_path, args.use_mask)
    map_poses = np.load(os.path.join(args.map_path, "poses.npy"), allow_pickle=True).item()
    map_K = np.load(os.path.join(args.map_path, "intrinsics.npy"))

    dino = Dinov2TRT()
    superpoint = SuperPointTRT()
    lightglue = LightGlueTRT()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    query_count = 0
    saved_count = 0
    retrieval_hit_excluded_count = 0
    pnp_success_count = 0
    relocalization_success_count = 0
    top1_timestamps: list[int] = []

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for frame_idx, (query_timestamp, image) in enumerate(iter_images_from_bag(args.bag_path, args.topic)):
            if frame_idx % max(1, args.every_n) != 0:
                continue
            if args.max_frames > 0 and saved_count >= args.max_frames:
                break

            query_count += 1
            query_embedding = asyncio.run(dino.infer(image)).astype(np.float32)
            norm = np.linalg.norm(query_embedding)
            if norm <= 1e-8:
                continue
            query_embedding = query_embedding / norm
            hits = find_loop(query_embedding, map_embeddings, args.threshold, args.topk)
            retrieved = [
                {
                    "rank": rank,
                    "map_index": int(idx),
                    "timestamp_ns": int(map_timestamps[idx]),
                    "similarity": float(similarity),
                    "excluded": int(map_timestamps[idx]) in excluded_timestamps,
                }
                for rank, (idx, similarity) in enumerate(reversed(hits))
            ]
            if retrieved:
                top1_timestamps.append(int(retrieved[0]["timestamp_ns"]))
            if any(item["excluded"] for item in retrieved):
                retrieval_hit_excluded_count += 1

            pnp_candidates = []
            query_features = None
            image_shape = np.array([image.shape[1], image.shape[0]], dtype=np.int64)
            for item in retrieved:
                map_timestamp = int(item["timestamp_ns"])
                reference_pose = map_poses[map_timestamp]
                reference_depth, _, reference_features, _, _ = db.get_depth_embedding_features_images(map_timestamp)
                if query_features is None:
                    query_features = asyncio.run(superpoint.infer(image))
                ref_matched, query_matched, matches = match_keypoints(lightglue, reference_features, query_features, image_shape)
                if len(matches) < args.pnp_min_matches:
                    continue
                points_3d, valid = keypoint_with_depth_to_3d(ref_matched, reference_depth, reference_pose, map_K)
                points_3d = points_3d[valid]
                points_2d = query_matched[valid]
                if len(points_2d) <= args.pnp_min_points:
                    continue
                pnp_candidates.append((points_3d, points_2d))

            pnp_success, _, inlier_ratio, _, inlier_count, point_count = rerank_by_pnp_inliers(
                pnp_candidates,
                map_K,
                min_point_count=args.pnp_min_points,
                min_inlier_count=args.pnp_min_inliers,
            )
            pnp_success_count += int(pnp_success)
            relocalization_success = bool(pnp_success and inlier_count >= args.pnp_min_inliers)
            relocalization_success_count += int(relocalization_success)

            row = {
                "query_timestamp_ns": int(query_timestamp),
                "retrieved": retrieved,
                "retrieved_excluded_count": sum(1 for item in retrieved if item["excluded"]),
                "pnp_success": bool(pnp_success),
                "inlier_count": int(inlier_count),
                "point_count": int(point_count),
                "inlier_ratio": float(inlier_ratio),
                "relocalization_success": relocalization_success,
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            saved_count += 1
            if saved_count % 20 == 0:
                print(f"processed={saved_count}")

    summary = {
        "bag_path": args.bag_path,
        "map_path": args.map_path,
        "topic": args.topic,
        "use_mask": bool(args.use_mask),
        "excluded_keyframes": len(excluded_timestamps),
        "retrieval_keyframes": len(map_timestamps),
        "queries": saved_count,
        "retrieval_hit_excluded_count": retrieval_hit_excluded_count,
        "pnp_success_count": pnp_success_count,
        "relocalization_success_count": relocalization_success_count,
        "top1_unique_count": len(set(top1_timestamps)),
    }
    summary["pnp_success_rate"] = pnp_success_count / max(1, saved_count)
    summary["relocalization_success_rate"] = relocalization_success_count / max(1, saved_count)
    summary["mask_violation_rate"] = retrieval_hit_excluded_count / max(1, saved_count)

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)
            f.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-path", required=True)
    parser.add_argument("--map-path", required=True)
    parser.add_argument("--topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--every-n", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--use-mask", action="store_true")
    parser.add_argument("--pnp-min-matches", type=int, default=50)
    parser.add_argument("--pnp-min-points", type=int, default=80)
    parser.add_argument("--pnp-min-inliers", type=int, default=50)
    parser.add_argument("--out-jsonl", default="/tinynav/tinynav_temp/relocalization_mask_eval.jsonl")
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
