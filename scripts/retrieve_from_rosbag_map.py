#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

from tinynav.core.build_map_node import TinyNavDB, find_loop
from tinynav.core.models_trt import Dinov2TRT, LightGlueTRT, SuperPointTRT


def build_map_embeddings(map_path: str) -> tuple[list[int], np.ndarray, TinyNavDB]:
    db = TinyNavDB(map_path, is_scratch=False)
    poses_path = os.path.join(map_path, "poses.npy")
    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"Missing map poses file: {poses_path}")
    poses = np.load(poses_path, allow_pickle=True).item()
    timestamps = sorted(int(t) for t in poses.keys())
    embs = []
    for ts in timestamps:
        e = db.get_embedding(ts).astype(np.float32)
        n = np.linalg.norm(e)
        if n > 1e-8:
            e = e / n
        embs.append(e)
    if not embs:
        raise RuntimeError("No map embeddings found.")
    return timestamps, np.stack(embs, axis=0), db


def match_keypoints(lightglue: LightGlueTRT, feats0: dict, feats1: dict, image_shape: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    match_result = asyncio.run(
        lightglue.infer(
            feats0["kpts"], feats1["kpts"], feats0["descps"], feats1["descps"], feats0["mask"], feats1["mask"], image_shape, image_shape
        )
    )
    match_indices = match_result["match_indices"][0]
    valid_mask = match_indices != -1
    keypoints0 = feats0["kpts"][0][valid_mask]
    keypoints1 = feats1["kpts"][0][match_indices[valid_mask]]
    matches = []
    for i, index in enumerate(match_indices):
        if index != -1:
            matches.append([i, index])
    return keypoints0, keypoints1, np.array(matches, dtype=np.int64)


def keypoint_with_depth_to_3d(keypoints: np.ndarray, depth: np.ndarray, pose_from_camera_to_world: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point_in_camera = []
    inliers = []
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    h, w = depth.shape[:2]
    for kp in keypoints:
        u = int(kp[0])
        v = int(kp[1])
        if u < 0 or u >= w or v < 0 or v >= h:
            point_in_camera.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            inliers.append(False)
            continue
        z = float(depth[v, u])
        if z > 0.0 and z < 50.0:
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            inliers.append(True)
        else:
            x = 0.0
            y = 0.0
            inliers.append(False)
        point_in_camera.append(np.array([x, y, z], dtype=np.float32))
    point_in_camera = np.array(point_in_camera, dtype=np.float32)
    inliers = np.array(inliers, dtype=bool)
    rotation = pose_from_camera_to_world[:3, :3]
    translation = pose_from_camera_to_world[:3, 3]
    point_in_world = (rotation @ point_in_camera.T).T + translation
    return point_in_world, inliers


def draw_match_image(ref_image: np.ndarray, query_image: np.ndarray, ref_kpts: np.ndarray, query_kpts: np.ndarray, inlier_idx: np.ndarray | None) -> np.ndarray:
    if ref_image.ndim == 2:
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2BGR)
    if query_image.ndim == 2:
        query_image = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
    if inlier_idx is not None and len(inlier_idx) > 0:
        sel = inlier_idx.reshape(-1)
        if np.max(sel) < len(ref_kpts) and np.max(sel) < len(query_kpts):
            ref_kpts = ref_kpts[sel]
            query_kpts = query_kpts[sel]
    cv_ref = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=3) for p in ref_kpts]
    cv_query = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=3) for p in query_kpts]
    cv_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0.0) for i in range(len(cv_ref))]
    return cv2.drawMatches(ref_image, cv_ref, query_image, cv_query, cv_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def iter_infra1_images(bag_path: str, topic: str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )

    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in topics:
        raise ValueError(f"Topic not found in bag: {topic}")
    msg_type = get_message(topics[topic])
    bridge = CvBridge()

    while reader.has_next():
        tpc, raw, ts_ns = reader.read_next()
        if tpc != topic:
            continue
        msg = deserialize_message(raw, msg_type)
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        yield int(ts_ns), img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_path", required=True)
    parser.add_argument("--map_path", required=True)
    parser.add_argument("--topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--every_n", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=0, help="0 means no cap")
    parser.add_argument("--out_jsonl", default="tinynav_temp/retrieval_from_bag.jsonl")
    parser.add_argument("--save_debug_dir", default="", help="optional debug image dir")
    parser.add_argument("--review_root", default="tinynav_temp/retrieval_from_bag_review", help="output root for review session_*")
    parser.add_argument("--pnp_min_matches", type=int, default=50)
    parser.add_argument("--pnp_min_inliers", type=int, default=50)
    args = parser.parse_args()

    map_ts, map_embs, map_db = build_map_embeddings(args.map_path)
    map_poses = np.load(os.path.join(args.map_path, "poses.npy"), allow_pickle=True).item()
    map_K = np.load(os.path.join(args.map_path, "intrinsics.npy"))
    embed_model = Dinov2TRT()
    superpoint = SuperPointTRT()
    lightglue = LightGlueTRT()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    if args.save_debug_dir:
        os.makedirs(args.save_debug_dir, exist_ok=True)
    session_dir = os.path.join(args.review_root, datetime.now().strftime("session_%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    index_path = os.path.join(session_dir, "index.jsonl")

    frame_idx = 0
    saved = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f, open(index_path, "w", encoding="utf-8") as index_f:
        for ts_ns, infra1 in iter_infra1_images(args.bag_path, args.topic):
            if frame_idx % max(1, args.every_n) != 0:
                frame_idx += 1
                continue
            frame_idx += 1
            if args.max_frames > 0 and saved >= args.max_frames:
                break

            q = asyncio.run(embed_model.infer(infra1)).astype(np.float32)
            qn = np.linalg.norm(q)
            if qn <= 1e-8:
                continue
            q = q / qn
            query_features = asyncio.run(superpoint.infer(infra1))
            image_shape = np.array([infra1.shape[1], infra1.shape[0]], dtype=np.int64)

            hits = find_loop(q, map_embs, args.threshold, args.topk)
            hits = [(int(i), float(s)) for i, s in hits]
            retrieved = []
            pnp_success = False
            inlier_count = 0
            inlier_ratio = 0.0
            pnp_match_path = ""
            for idx, sim in hits:
                map_timestamp = int(map_ts[idx])
                retrieved.append(
                    {
                        "map_index": idx,
                        "timestamp_ns": map_timestamp,
                        "similarity": sim,
                    }
                )
            point_3d_in_world_list = []
            point_2d_in_query_list = []
            best_match_vis = None
            for rank, hit in enumerate(retrieved):
                map_timestamp = int(hit["timestamp_ns"])
                ref_pose = map_poses[map_timestamp]
                ref_depth, _, ref_features, _, infra1_loader = map_db.get_depth_embedding_features_images(map_timestamp)
                ref_img = infra1_loader() if infra1_loader is not None else None
                if ref_features is None:
                    continue
                ref_matched_kpts, query_matched_kpts, matches = match_keypoints(lightglue, ref_features, query_features, image_shape)
                if len(matches) < args.pnp_min_matches:
                    continue
                point_3d_in_world, valid = keypoint_with_depth_to_3d(ref_matched_kpts, ref_depth, ref_pose, map_K)
                point_3d_in_world = point_3d_in_world[valid]
                point_2d_in_query = query_matched_kpts[valid]
                if len(point_3d_in_world) == 0:
                    continue
                point_3d_in_world_list.append(point_3d_in_world)
                point_2d_in_query_list.append(point_2d_in_query)
                if best_match_vis is None and ref_img is not None:
                    best_match_vis = (ref_img, ref_matched_kpts[valid], query_matched_kpts[valid])

            if len(point_3d_in_world_list) > 0:
                pts3d = np.concatenate(point_3d_in_world_list, axis=0)
                pts2d = np.concatenate(point_2d_in_query_list, axis=0)
                if len(pts3d) >= args.pnp_min_matches:
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d, pts2d, map_K, None)
                    pnp_success = bool(success)
                    inlier_count = int(len(inliers)) if inliers is not None else 0
                    inlier_ratio = float(inlier_count / max(1, len(pts2d)))
                    if best_match_vis is not None:
                        ref_img, ref_pts, query_pts = best_match_vis
                        match_img = draw_match_image(ref_img, infra1, ref_pts, query_pts, inliers)
                        sample_name = f"sample_{saved:06d}_{ts_ns}"
                        sample_dir = os.path.join(session_dir, sample_name)
                        os.makedirs(sample_dir, exist_ok=True)
                        pnp_match_path = os.path.join(sample_dir, "pnp_match.png")
                        cv2.imwrite(pnp_match_path, match_img)

            row = {
                "query_timestamp_ns": int(ts_ns),
                "topic": args.topic,
                "retrieved": retrieved,
                "pnp_success": bool(pnp_success),
                "inlier_count": int(inlier_count),
                "inlier_ratio": float(inlier_ratio),
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

            sample_name = f"sample_{saved:06d}_{ts_ns}"
            sample_dir = os.path.join(session_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)
            query_path = os.path.join(sample_dir, "query.png")
            cv2.imwrite(query_path, infra1)
            retrieved_ts = []
            similarities = []
            if args.save_debug_dir:
                q_path = os.path.join(args.save_debug_dir, f"query_{saved:06d}_{ts_ns}.png")
                cv2.imwrite(q_path, infra1)
                for rank, hit in enumerate(retrieved):
                    _, _, _, _, infra1_loader = map_db.get_depth_embedding_features_images(hit["timestamp_ns"])
                    ref = infra1_loader() if infra1_loader is not None else None
                    if ref is not None:
                        cv2.imwrite(os.path.join(sample_dir, f"retrieved_{rank:03d}.png"), ref)
                        r_path = os.path.join(
                            args.save_debug_dir,
                            f"query_{saved:06d}_rank{rank}_mapts_{hit['timestamp_ns']}.png",
                        )
                        cv2.imwrite(r_path, ref)
                    retrieved_ts.append(int(hit["timestamp_ns"]))
                    similarities.append(float(hit["similarity"]))
            else:
                for rank, hit in enumerate(retrieved):
                    _, _, _, _, infra1_loader = map_db.get_depth_embedding_features_images(hit["timestamp_ns"])
                    ref = infra1_loader() if infra1_loader is not None else None
                    if ref is not None:
                        cv2.imwrite(os.path.join(sample_dir, f"retrieved_{rank:03d}.png"), ref)
                    retrieved_ts.append(int(hit["timestamp_ns"]))
                    similarities.append(float(hit["similarity"]))

            sample_meta = {
                "query_timestamp_ns": int(ts_ns),
                "retrieved_timestamps_ns": retrieved_ts,
                "similarities": similarities,
                "selected_idx": -1,
                "pnp_success": bool(pnp_success),
                "inlier_count": int(inlier_count),
                "inlier_ratio": float(inlier_ratio),
                "pnp_match_path": pnp_match_path,
                "relocalization_success": bool(pnp_success and inlier_count >= args.pnp_min_inliers),
                "false_case": False,
                "false_case_reason": "offline_retrieval",
                "review_label": "unreviewed",
                "review_note": "",
            }
            sample_json_path = os.path.join(sample_dir, "sample.json")
            with open(sample_json_path, "w", encoding="utf-8") as sf:
                json.dump(sample_meta, sf, ensure_ascii=True, indent=2)
            index_f.write(
                json.dumps(
                    {"sample_id": sample_name, "query_timestamp_ns": int(ts_ns), "sample_json": sample_json_path},
                    ensure_ascii=True,
                )
                + "\n"
            )

            saved += 1
            if saved % 20 == 0:
                print(f"processed={saved}")

    print(f"done: wrote {saved} rows to {args.out_jsonl}")
    print(f"review session: {session_dir}")


if __name__ == "__main__":
    main()
