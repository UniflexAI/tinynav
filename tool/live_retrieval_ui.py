#!/usr/bin/env python3
"""Live top-1 VLAD image-retrieval viewer.

Replays a query rosbag frame by frame, embeds each color frame with the same
DINOv2 + VLAD pipeline used at map-build time, and serves a small web page
showing the live query frame next to the current top-1 matching keyframe
from a pre-built TinyNav map. When a match is confident (z-score above
threshold), it also runs matcher+PnP relocalization -- the same step
tinynav.core.map_node.MapNode.relocalize_with_depth performs. --matcher
selects between EfficientLoFTR (the default in map_node.py; tolerates the
day/night appearance gap between the map and a live query far better) and
SuperPoint+LightGlue (measured on this branch's own calibration data:
SuperPoint+LightGlue relocalized ~9% of VLAD-confident cross-session frames
vs. ~39% for EfficientLoFTR; see tool/count_loftr_vs_sp_lg.py).

Usage:
    uv run python tool/live_retrieval_ui.py \\
        --tinynav_map_path /tinynav/output/map_day_20260716 \\
        --query_bag /tinynav/dataset/202601718/rosbags/bag_2026_07_17_08_38_17 \\
        --matcher sp_lg
Then open http://127.0.0.1:8642 in a browser.
"""
import argparse
import asyncio
import base64
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from cv_bridge import CvBridge
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from rosidl_runtime_py.utilities import get_message

from tinynav.core.build_map_node import TinyNavDB, find_loop_by_zscore
from tinynav.core.depth_reprojection import reproject_depth_to_camera
from tinynav.core.math_utils import rerank_by_pnp_inliers
from tinynav.core.models_trt import Dinov2TRT, EfficientLoFTRTRT, LightGlueTRT, SuperPointTRT
from tinynav.core.vlad_retrieval import compute_vlad

MATCHER_LABELS = {"efficientloftr": "EfficientLoFTR", "sp_lg": "SuperPoint+LightGlue"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live top-1 VLAD retrieval viewer")
    parser.add_argument("--tinynav_map_path", required=True, help="Pre-built map directory (from build_map_node.py)")
    parser.add_argument("--query_bag", required=True, help="Rosbag2 directory to replay as the live query stream")
    parser.add_argument("--image_topic", default="/camera/camera/color/image_rect_raw/compressed", help="Color topic used for VLAD retrieval + relocalization matching")
    parser.add_argument("--matcher", choices=sorted(MATCHER_LABELS), default="efficientloftr", help="Matcher used for the relocalization step, run on confident VLAD candidates")
    parser.add_argument("--play_rate", type=float, default=1.0, help="Bag playback speed multiplier")
    parser.add_argument("--query_hz", type=float, default=2.0, help="Max rate at which frames are embedded and matched")
    parser.add_argument("--z_threshold", type=float, default=4.0, help="Standard deviations above the mean of all candidates for a match to be shown as confident")
    parser.add_argument("--relocalization_top_k", type=int, default=3, help="Number of top VLAD candidates to attempt matching+PnP relocalization against")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True, help="Replay the bag on completion")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    args = parser.parse_args()
    if args.play_rate <= 0.0:
        raise ValueError(f"--play_rate must be > 0, got {args.play_rate}")
    return args


def load_map(map_path: str):
    map_path = Path(map_path)
    poses = np.load(map_path / "poses.npy", allow_pickle=True).item()
    codebook = np.load(map_path / "vlad_codebook.npy")
    map_K = np.load(map_path / "intrinsics.npy")
    map_rgb_K = np.load(map_path / "rgb_camera_intrinsics.npy", allow_pickle=True)
    map_T_rgb_to_infra1 = np.load(map_path / "T_rgb_to_infra1.npy", allow_pickle=True)
    assert map_T_rgb_to_infra1 is not None and map_T_rgb_to_infra1.shape == (4, 4), \
        f"Map at {map_path} has no valid T_rgb_to_infra1.npy; rebuild it to use color EfficientLoFTR+PnP relocalization."
    map_rgb_image_shape = np.load(map_path / "rgb_image_shape.npy")
    db = TinyNavDB(str(map_path), is_scratch=False)
    idx_to_timestamp = {idx: ts for idx, ts in enumerate(poses.keys())}
    embeddings = np.stack([db.get_embedding(ts) for ts in idx_to_timestamp.values()])
    return db, codebook, embeddings, idx_to_timestamp, poses, map_K, map_rgb_K, map_T_rgb_to_infra1, map_rgb_image_shape


def keypoint_with_depth_to_3d(keypoints: np.ndarray, depth: np.ndarray, pose_from_camera_to_world: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    point_in_camera = []
    inliers = []
    for kp in keypoints:
        u, v = int(kp[0]), int(kp[1])
        z = depth[v, u] if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1] else 0.0
        if 0 < z < 50:
            point_in_camera.append([(u - cx) * z / fx, (v - cy) * z / fy, z])
            inliers.append(True)
        else:
            point_in_camera.append([0.0, 0.0, 0.0])
            inliers.append(False)
    point_in_camera = np.array(point_in_camera)
    inliers = np.array(inliers)
    rotation = pose_from_camera_to_world[:3, :3]
    translation = pose_from_camera_to_world[:3, 3]
    point_in_world = (rotation @ point_in_camera.T).T + translation
    return point_in_world, inliers


def match_sp_lg(superpoint: SuperPointTRT, light_glue: LightGlueTRT, image0_bgr: np.ndarray, image1_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SuperPoint+LightGlue equivalent of EfficientLoFTRTRT.infer's (mkpts0, mkpts1, mconf)
    return shape, so callers can swap matchers without branching downstream (PnP, visualization).

    The LightGlue TRT engine only exposes match_indices, not per-match scores, so mconf here is
    a constant placeholder (all matches equally "confident") -- it exists only so the UI's mconf
    field has something to display, not as a real quality signal like EfficientLoFTR's mconf.
    """
    gray0 = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)
    feats0 = asyncio.run(superpoint.infer(gray0))
    feats1 = asyncio.run(superpoint.infer(gray1))
    image_shape0 = np.array([gray0.shape[1], gray0.shape[0]], dtype=np.int64)
    image_shape1 = np.array([gray1.shape[1], gray1.shape[0]], dtype=np.int64)
    match_result = asyncio.run(light_glue.infer(
        feats0["kpts"], feats1["kpts"], feats0["descps"], feats1["descps"],
        feats0["mask"], feats1["mask"], image_shape0, image_shape1,
    ))
    match_indices = match_result["match_indices"][0]
    valid_mask = match_indices != -1
    mkpts0 = feats0["kpts"][0][valid_mask]
    mkpts1 = feats1["kpts"][0][match_indices[valid_mask]]
    mconf = np.ones(len(mkpts0), dtype=np.float32)
    return mkpts0, mkpts1, mconf


def draw_matches_image(image0: np.ndarray, image1: np.ndarray, keypoints0: np.ndarray, keypoints1: np.ndarray, inlier_mask: np.ndarray = None) -> np.ndarray:
    """keypoints0[i] and keypoints1[i] must already be paired (as returned by EfficientLoFTRTRT.infer).

    inlier_mask, if given, colors PnP-inlier matches green and every other match (rejected by
    depth validity or by PnP RANSAC) red, side by side on one canvas.
    """
    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = image0
    canvas[:h1, w0:w0 + w1] = image1
    if inlier_mask is None:
        inlier_mask = np.ones(len(keypoints0), dtype=bool)
    for (x0, y0), (x1, y1), is_inlier in zip(keypoints0, keypoints1, inlier_mask):
        p0 = (int(x0), int(y0))
        p1 = (int(x1) + w0, int(y1))
        color = (0, 255, 0) if is_inlier else (0, 0, 255)  # BGR: green=inlier, red=everything else
        cv2.circle(canvas, p0, 3, color, -1)
        cv2.circle(canvas, p1, 3, color, -1)
        cv2.line(canvas, p0, p1, color, 1)
    return canvas


def encode_jpeg_b64(image: np.ndarray, max_edge_px: int = 640, quality: int = 80) -> str:
    h, w = image.shape[:2]
    scale = min(1.0, max_edge_px / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class RetrievalState:
    """Holds the latest retrieval result; readers only ever see the newest frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._seq = 0
        self._payload: dict = {"status": "starting"}

    def publish(self, payload: dict) -> None:
        with self._lock:
            self._seq += 1
            self._payload = payload

    def snapshot(self) -> tuple[int, dict]:
        with self._lock:
            return self._seq, self._payload


def bag_message_iterator(bag_path: str, topic: str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in topic_types:
        raise ValueError(f"Topic {topic!r} not found in bag {bag_path}; available: {sorted(topic_types)}")
    try:
        reader.set_filter(StorageFilter(topics=[topic]))
    except Exception:
        pass
    msg_type = get_message(topic_types[topic])
    while reader.has_next():
        read_topic, data, timestamp_ns = reader.read_next()
        if read_topic != topic:
            continue
        yield deserialize_message(data, msg_type), int(timestamp_ns)


def retrieval_loop(args: argparse.Namespace, state: RetrievalState) -> None:
    db, codebook, map_embeddings, idx_to_timestamp, map_poses, map_K, map_rgb_K, map_T_rgb_to_infra1, map_rgb_image_shape = load_map(args.tinynav_map_path)
    dinov2_model = Dinov2TRT()
    if args.matcher == "efficientloftr":
        efficientloftr_matcher = EfficientLoFTRTRT()

        def match_images(image0, image1):
            match_result = asyncio.run(efficientloftr_matcher.infer(image0, image1))
            return match_result["mkpts0"], match_result["mkpts1"], match_result["mconf"]
    else:
        superpoint_extractor = SuperPointTRT()
        light_glue_matcher = LightGlueTRT()

        def match_images(image0, image1):
            return match_sp_lg(superpoint_extractor, light_glue_matcher, image0, image1)
    matcher_label = MATCHER_LABELS[args.matcher]
    bridge = CvBridge()
    min_query_interval_ns = int(1e9 / args.query_hz) if args.query_hz > 0 else 0
    frame_index = 0

    while True:
        timestamps = [ts for _, ts in bag_message_iterator(args.query_bag, args.image_topic)]
        if not timestamps:
            state.publish({"status": "error", "message": f"No messages on {args.image_topic} in {args.query_bag}"})
            return
        bag_start_ts_ns, bag_end_ts_ns = timestamps[0], timestamps[-1]

        playback_start_wall_s = None
        playback_start_ts_ns = None
        last_query_ts_ns = None

        for msg, timestamp_ns in bag_message_iterator(args.query_bag, args.image_topic):
            if playback_start_wall_s is None:
                playback_start_wall_s = time.monotonic()
                playback_start_ts_ns = timestamp_ns
            elapsed_bag_s = (timestamp_ns - playback_start_ts_ns) * 1e-9
            sleep_s = playback_start_wall_s + elapsed_bag_s / args.play_rate - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

            if last_query_ts_ns is not None and (timestamp_ns - last_query_ts_ns) < min_query_interval_ns:
                continue
            last_query_ts_ns = timestamp_ns

            query_image = bridge.compressed_imgmsg_to_cv2(msg)
            patch_tokens = asyncio.run(dinov2_model.infer_patch_tokens(query_image))
            query_embedding = compute_vlad(patch_tokens, codebook)

            similarities = map_embeddings @ query_embedding
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            std = similarities.std()
            z_score = float((best_score - similarities.mean()) / std) if std > 0.0 else 0.0
            match_timestamp = idx_to_timestamp[best_idx]
            _, _, _, rgb_loader, infra1_loader = db.get_depth_embedding_features_images(match_timestamp)
            match_image = rgb_loader()
            if match_image is None:
                match_image = infra1_loader()
                if match_image is not None and match_image.ndim == 2:
                    match_image = cv2.cvtColor(match_image, cv2.COLOR_GRAY2BGR)

            is_confident = z_score >= args.z_threshold
            reloc_success = False
            reloc_reason = "z-score below threshold, relocalization not attempted"
            match_count = 0
            mean_conf = 0.0
            pnp_inlier_count = 0
            pnp_point_count = 0
            pose_translation = None
            match_vis_b64 = None

            if is_confident:
                idx_and_similarity = find_loop_by_zscore(query_embedding, map_embeddings, args.z_threshold, args.relocalization_top_k)
                pnp_candidates = []
                candidate_match_counts = []
                candidate_mean_confs = []
                candidate_ref_kpts = []
                candidate_query_kpts = []
                candidate_depth_inliers = []
                candidate_ref_bgr = []
                for idx_in_map, _ in idx_and_similarity:
                    ts_in_map = idx_to_timestamp[idx_in_map]
                    reference_rgb_pose_in_world = map_poses[ts_in_map] @ map_T_rgb_to_infra1
                    _, _, _, cand_rgb_loader, _ = db.get_depth_embedding_features_images(ts_in_map)
                    cand_rgb = cand_rgb_loader()
                    if cand_rgb is None:
                        continue
                    reference_color_depth = reproject_depth_to_camera(
                        db.get_depth(ts_in_map), map_K, np.linalg.inv(map_T_rgb_to_infra1), map_rgb_K, map_rgb_image_shape,
                    )
                    ref_kpts, query_kpts, mconf = match_images(cand_rgb, query_image)
                    if len(ref_kpts) < 50:
                        continue
                    point_3d_in_world, depth_inliers = keypoint_with_depth_to_3d(ref_kpts, reference_color_depth, reference_rgb_pose_in_world, map_rgb_K)
                    point_3d_list = point_3d_in_world[depth_inliers]
                    point_2d_list = query_kpts[depth_inliers]
                    if len(point_2d_list) <= 80:
                        continue
                    pnp_candidates.append((point_3d_list, point_2d_list))
                    candidate_match_counts.append(len(ref_kpts))
                    candidate_mean_confs.append(float(mconf.mean()) if len(mconf) else 0.0)
                    candidate_ref_kpts.append(ref_kpts)
                    candidate_query_kpts.append(query_kpts)
                    candidate_depth_inliers.append(depth_inliers)
                    candidate_ref_bgr.append(cand_rgb)

                def build_vis(i, inlier_mask=None):
                    return draw_matches_image(
                        candidate_ref_bgr[i], query_image, candidate_ref_kpts[i], candidate_query_kpts[i], inlier_mask,
                    )

                if not pnp_candidates:
                    reloc_reason = f"no VLAD candidate had enough {matcher_label} matches + valid depth landmarks"
                else:
                    # Report the best-matching candidate's raw stats even if PnP itself later
                    # fails geometrically, so "matches" always reflects what the matcher actually found.
                    best_by_matches = int(np.argmax(candidate_match_counts))
                    match_count = candidate_match_counts[best_by_matches]
                    mean_conf = candidate_mean_confs[best_by_matches]
                    match_vis_b64 = encode_jpeg_b64(build_vis(best_by_matches), max_edge_px=960)

                    success, best_pose, _, best_candidate_index, inlier_count, point_count = rerank_by_pnp_inliers(pnp_candidates, map_rgb_K)
                    pnp_inlier_count = inlier_count
                    pnp_point_count = point_count
                    if best_candidate_index >= 0:
                        match_count = candidate_match_counts[best_candidate_index]
                        mean_conf = candidate_mean_confs[best_candidate_index]
                        # rerank_by_pnp_inliers only returns counts; re-run RANSAC on just the
                        # winning candidate to recover which specific matches were inliers, for
                        # the green/red visualization below.
                        points_3d, points_2d = pnp_candidates[best_candidate_index]
                        ransac_success, _, _, ransac_inliers = cv2.solvePnPRansac(points_3d, points_2d, map_rgb_K, None)
                        depth_inliers = candidate_depth_inliers[best_candidate_index]
                        full_inlier_mask = np.zeros(len(depth_inliers), dtype=bool)
                        if ransac_success and ransac_inliers is not None:
                            depth_valid_indices = np.where(depth_inliers)[0]
                            full_inlier_mask[depth_valid_indices[ransac_inliers.flatten()]] = True
                        match_vis_b64 = encode_jpeg_b64(build_vis(best_candidate_index, full_inlier_mask), max_edge_px=960)
                    if success:
                        reloc_success = True
                        reloc_reason = None
                        pose_translation = np.linalg.inv(best_pose)[:3, 3].tolist()
                    else:
                        reloc_reason = "PnP RANSAC failed to find a valid pose"

            frame_index += 1
            state.publish({
                "status": "ok",
                "frame_index": frame_index,
                "progress_pct": 100.0 * (timestamp_ns - bag_start_ts_ns) / max(1, bag_end_ts_ns - bag_start_ts_ns),
                "query_timestamp_ns": timestamp_ns,
                "match_timestamp_ns": int(match_timestamp),
                "similarity": best_score,
                "z_score": z_score,
                "is_confident": is_confident,
                "query_image_b64": encode_jpeg_b64(query_image),
                "match_image_b64": encode_jpeg_b64(match_image) if match_image is not None else None,
                "relocalization_attempted": is_confident,
                "relocalization_success": reloc_success,
                "relocalization_reason": reloc_reason,
                "match_count": match_count,
                "mean_conf": mean_conf,
                "pnp_inlier_count": pnp_inlier_count,
                "pnp_point_count": pnp_point_count,
                "pose_translation": pose_translation,
                "match_vis_b64": match_vis_b64,
                "matcher_label": matcher_label,
            })

        if not args.loop:
            return


HTML_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TinyNav Live Retrieval</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 12px; color: #ccc; }
  #status { font-size: 13px; color: #999; margin-bottom: 12px; }
  #status.live { color: #4caf50; }
  #status.dead { color: #e05a4e; }
  .row { display: flex; gap: 16px; flex-wrap: wrap; }
  .panel { flex: 1 1 380px; min-width: 300px; background: #1a1a1a; border-radius: 8px; padding: 12px; box-sizing: border-box; }
  .panel h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; color: #888; margin: 0 0 8px; }
  .panel img { width: 100%; height: auto; border-radius: 4px; background: #000; display: block; }
  .meta { font-size: 12px; color: #aaa; margin-top: 8px; display: flex; justify-content: space-between; }
  #score { font-size: 28px; font-weight: 700; text-align: center; margin: 12px 0; }
  #score.confident { color: #4caf50; }
  #score.weak { color: #e0a34e; }
  #bar-track { height: 4px; background: #333; border-radius: 2px; overflow: hidden; margin-top: 12px; }
  #bar-fill { height: 100%; background: #4caf50; width: 0%; transition: width 0.2s linear; }
  #reloc-panel { background: #1a1a1a; border-radius: 8px; padding: 12px; margin-top: 16px; box-sizing: border-box; }
  #reloc-panel h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; color: #888; margin: 0 0 8px; }
  #reloc-status { font-size: 15px; font-weight: 600; margin-bottom: 8px; }
  #reloc-status.success { color: #4caf50; }
  #reloc-status.fail { color: #e05a4e; }
  #reloc-status.skipped { color: #666; }
  #reloc-stats { font-size: 12px; color: #aaa; display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }
  #reloc-vis { width: 100%; height: auto; border-radius: 4px; background: #000; display: none; }
</style>
</head>
<body>
  <h1>TinyNav Live Retrieval &mdash; top-1 VLAD match + <span id="matcher-name">relocalization</span></h1>
  <div id="status">connecting&hellip;</div>
  <div id="score"></div>
  <div class="row">
    <div class="panel">
      <h2>Query (live)</h2>
      <img id="query-img">
      <div class="meta"><span id="query-ts"></span><span id="frame-idx"></span></div>
    </div>
    <div class="panel">
      <h2>Top-1 match (map keyframe)</h2>
      <img id="match-img">
      <div class="meta"><span id="match-ts"></span></div>
    </div>
  </div>
  <div id="bar-track"><div id="bar-fill"></div></div>

  <div id="reloc-panel">
    <h2>Relocalization &mdash; <span id="reloc-matcher-name">matcher</span> + PnP <span style="font-weight:400; text-transform:none; letter-spacing:normal; color:#666;">(green = PnP inlier, red = rejected match)</span></h2>
    <div id="reloc-status"></div>
    <div id="reloc-stats">
      <span id="reloc-matches"></span>
      <span id="reloc-conf"></span>
      <span id="reloc-inliers"></span>
      <span id="reloc-pose"></span>
    </div>
    <img id="reloc-vis">
  </div>

<script>
function fmtNs(ns) {
  if (!ns) return "";
  return (ns / 1e9).toFixed(3) + " s";
}

function connect() {
  const statusEl = document.getElementById("status");
  const ws = new WebSocket(`ws://${location.host}/ws/live`);

  ws.onopen = () => { statusEl.textContent = "live"; statusEl.className = "live"; };
  ws.onclose = () => {
    statusEl.textContent = "disconnected, retrying…";
    statusEl.className = "dead";
    setTimeout(connect, 1000);
  };
  ws.onerror = () => ws.close();

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.status === "error") {
      statusEl.textContent = "error: " + data.message;
      statusEl.className = "dead";
      return;
    }
    document.getElementById("query-img").src = "data:image/jpeg;base64," + data.query_image_b64;
    if (data.match_image_b64) {
      document.getElementById("match-img").src = "data:image/jpeg;base64," + data.match_image_b64;
    }
    document.getElementById("query-ts").textContent = fmtNs(data.query_timestamp_ns);
    document.getElementById("match-ts").textContent = fmtNs(data.match_timestamp_ns);
    document.getElementById("frame-idx").textContent = "frame " + data.frame_index;
    document.getElementById("bar-fill").style.width = data.progress_pct + "%";

    const scoreEl = document.getElementById("score");
    scoreEl.textContent = "similarity " + data.similarity.toFixed(3) + "  (z=" + data.z_score.toFixed(2) + ")";
    scoreEl.className = data.is_confident ? "confident" : "weak";

    const relocStatusEl = document.getElementById("reloc-status");
    const relocVisEl = document.getElementById("reloc-vis");
    if (!data.relocalization_attempted) {
      relocStatusEl.textContent = "skipped — " + data.relocalization_reason;
      relocStatusEl.className = "skipped";
      relocVisEl.style.display = "none";
    } else if (data.relocalization_success) {
      relocStatusEl.textContent = "SUCCESS";
      relocStatusEl.className = "success";
    } else {
      relocStatusEl.textContent = "FAILED — " + data.relocalization_reason;
      relocStatusEl.className = "fail";
    }
    document.getElementById("matcher-name").textContent = data.matcher_label + " relocalization";
    document.getElementById("reloc-matcher-name").textContent = data.matcher_label;
    document.getElementById("reloc-matches").textContent = data.matcher_label + " matches: " + data.match_count;
    document.getElementById("reloc-conf").textContent = "mean conf: " + data.mean_conf.toFixed(2);
    document.getElementById("reloc-inliers").textContent = "PnP inliers: " + data.pnp_inlier_count + " / " + data.pnp_point_count;
    document.getElementById("reloc-pose").textContent = data.pose_translation
      ? "pose (x,y,z): " + data.pose_translation.map(v => v.toFixed(2)).join(", ")
      : "pose: n/a";
    if (data.match_vis_b64) {
      relocVisEl.src = "data:image/jpeg;base64," + data.match_vis_b64;
      relocVisEl.style.display = "block";
    } else {
      relocVisEl.style.display = "none";
    }
  };
}
connect();
</script>
</body>
</html>
"""


def build_app(state: RetrievalState) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        last_seq_sent = -1
        try:
            while True:
                seq, payload = state.snapshot()
                if seq != last_seq_sent:
                    last_seq_sent = seq
                    await ws.send_json(payload)
                await asyncio.sleep(0.03)
        except WebSocketDisconnect:
            pass

    return app


def main() -> None:
    args = parse_args()
    state = RetrievalState()
    threading.Thread(target=retrieval_loop, args=(args, state), daemon=True).start()
    uvicorn.run(build_app(state), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
