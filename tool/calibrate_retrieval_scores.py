#!/usr/bin/env python3
"""Calibrate what a "good" VLAD top-1 similarity score looks like across sessions.

Replays a query rosbag once (as fast as possible, no real-time pacing), computes
the top-1 VLAD match for every sampled frame against a pre-built map, and saves
the score distribution plus representative query/match image pairs at several
score percentiles so they can be inspected visually.

Usage:
    uv run python tool/calibrate_retrieval_scores.py \\
        --tinynav_map_path /tinynav/output/map_day_20260716 \\
        --query_bag /tinynav/dataset/202601718/rosbags/bag_2026_07_17_08_38_17 \\
        --out_dir /tinynav/output/calibration_20260716
"""
import argparse
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
from cv_bridge import CvBridge

from tinynav.core.models_trt import Dinov2TRT
from tinynav.core.vlad_retrieval import compute_vlad
from tool.live_retrieval_ui import bag_message_iterator, load_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate cross-session VLAD retrieval scores")
    parser.add_argument("--tinynav_map_path", required=True)
    parser.add_argument("--query_bag", required=True)
    parser.add_argument("--image_topic", default="/camera/camera/color/image_rect_raw/compressed")
    parser.add_argument("--sample_hz", type=float, default=2.0, help="Max rate at which frames are embedded and matched")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_examples", type=int, default=7, help="Representative score percentiles to save image pairs for")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db, codebook, map_embeddings, idx_to_timestamp = load_map(args.tinynav_map_path)
    dinov2_model = Dinov2TRT()
    bridge = CvBridge()
    min_query_interval_ns = int(1e9 / args.sample_hz) if args.sample_hz > 0 else 0

    results = []
    last_query_ts_ns = None
    for msg, timestamp_ns in bag_message_iterator(args.query_bag, args.image_topic):
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
        match_timestamp = int(idx_to_timestamp[best_idx])
        results.append({"query_timestamp_ns": timestamp_ns, "match_timestamp_ns": match_timestamp, "similarity": best_score, "z_score": z_score})
        print(f"t={timestamp_ns} similarity={best_score:.4f} z_score={z_score:.2f} match={match_timestamp}")

    scores = np.array([r["similarity"] for r in results])
    zscores = np.array([r["z_score"] for r in results])
    stats = {
        "count": len(scores),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "p10": float(np.percentile(scores, 10)),
        "p25": float(np.percentile(scores, 25)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
        "z_min": float(zscores.min()),
        "z_max": float(zscores.max()),
        "z_mean": float(zscores.mean()),
        "z_median": float(np.median(zscores)),
    }
    print("=== score distribution ===")
    print(json.dumps(stats, indent=2))

    with open(out_dir / "scores.json", "w") as f:
        json.dump({"stats": stats, "results": results}, f, indent=2)

    percentiles = np.linspace(0, 100, args.num_examples)
    example_indices = sorted(set(int(np.argmin(np.abs(scores - np.percentile(scores, p)))) for p in percentiles))

    query_reader = bag_message_iterator(args.query_bag, args.image_topic)
    query_by_timestamp = {}
    target_timestamps = {results[i]["query_timestamp_ns"] for i in example_indices}
    for msg, timestamp_ns in query_reader:
        if timestamp_ns in target_timestamps:
            query_by_timestamp[timestamp_ns] = bridge.compressed_imgmsg_to_cv2(msg)
        if len(query_by_timestamp) == len(target_timestamps):
            break

    for rank, i in enumerate(example_indices):
        r = results[i]
        query_image = query_by_timestamp[r["query_timestamp_ns"]]
        _, _, _, rgb_loader, infra1_loader = db.get_depth_embedding_features_images(r["match_timestamp_ns"])
        match_image = rgb_loader()
        if match_image is None:
            match_image = infra1_loader()
        tag = f"{rank:02d}_score_{r['similarity']:.3f}"
        cv2.imwrite(str(out_dir / f"{tag}_query.jpg"), query_image)
        if match_image is not None:
            cv2.imwrite(str(out_dir / f"{tag}_match.jpg"), match_image)
        print(f"saved example {tag}")


if __name__ == "__main__":
    main()
