#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from tool.benchmark.map_retrieval_self_consistency_common import (
    _apply_se2,
    _load_descriptor_embeddings,
    _load_poses,
    _parse_float_list,
    _parse_int_list,
    _ransac_fit_se2,
    _retrieve_rows,
    _write_jsonl,
)


def run_fit(args: argparse.Namespace) -> dict:
    start_time = time.time()
    map_a = Path(args.map_a)
    map_b = Path(args.map_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    map_a_poses = _load_poses(map_a)
    map_b_poses = _load_poses(map_b)
    map_a_timestamps = sorted(map_a_poses)
    map_b_timestamps = sorted(map_b_poses)
    if args.max_queries > 0:
        map_b_timestamps = map_b_timestamps[: args.max_queries]
    if args.every_n > 1:
        map_b_timestamps = map_b_timestamps[:: args.every_n]

    topk_values = _parse_int_list(args.topk)
    thresholds = _parse_float_list(args.distance_thresholds)
    max_topk = max(topk_values)

    map_a_embeddings, map_b_embeddings = _load_descriptor_embeddings(
        map_a,
        map_b,
        map_a_timestamps,
        map_b_timestamps,
        args,
    )
    similarities = map_a_embeddings @ map_b_embeddings.T
    query_rows = _retrieve_rows(similarities, map_a_timestamps, map_a_poses, map_b_timestamps, max_topk)

    src_xy = np.stack([map_b_poses[int(row["query_timestamp_ns"])][:2, 3] for row in query_rows], axis=0)
    dst_xy = np.stack([map_a_poses[int(row["retrieved"][0]["timestamp_ns"])][:2, 3] for row in query_rows], axis=0)
    rotation, translation, top1_residuals = _ransac_fit_se2(
        src_xy,
        dst_xy,
        args.ransac_threshold_m,
        args.ransac_iterations,
        args.seed,
    )
    query_gt_xy = _apply_se2(src_xy, rotation, translation)

    for row, gt_xy, residual in zip(query_rows, query_gt_xy, top1_residuals):
        row["pose_a_gt_xy_self_consistency"] = gt_xy.tolist()
        row["top1_residual_m"] = float(residual)

    yaw_deg = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    transform = np.eye(4, dtype=np.float64)
    transform[:2, :2] = rotation
    transform[:2, 3] = translation

    summary = {
        "type": "cross_map_self_consistency_transform",
        "note": "T is fitted from this descriptor backend's Top1 keyframe matches. It is a self-supervised alignment signal, not external GT.",
        "descriptor_backend": args.descriptor_backend,
        "map_a": str(map_a),
        "map_b": str(map_b),
        "map_a_keyframes": len(map_a_timestamps),
        "map_b_queries": len(query_rows),
        "topk": topk_values,
        "distance_thresholds_m": thresholds,
        "ransac_threshold_m": args.ransac_threshold_m,
        "T_map_a_map_b_self_consistency_se2": {
            "T": transform.tolist(),
            "R_xy": rotation.tolist(),
            "t_xy": translation.tolist(),
            "yaw_deg": yaw_deg,
        },
        "top1_residual_m": {
            "mean": float(np.mean(top1_residuals)),
            "median": float(np.median(top1_residuals)),
            "p90": float(np.percentile(top1_residuals, 90)),
            "max": float(np.max(top1_residuals)),
        },
        "top1_inlier_ratio": {
            f"{threshold}m": float(np.mean(top1_residuals <= threshold)) for threshold in thresholds
        },
        "elapsed_s": time.time() - start_time,
    }
    if args.descriptor_backend == "superpoint-bow":
        summary["bow"] = {
            "vocab_size": args.bow_vocab_size,
            "sample_limit": args.bow_sample_limit,
            "kmeans_iterations": args.bow_kmeans_iterations,
        }
    if args.descriptor_backend == "anyloc-vlad":
        summary["anyloc_vlad"] = {
            "model": args.anyloc_model,
            "image_size": args.anyloc_image_size,
            "device": args.anyloc_device,
            "vocab_size": args.vlad_vocab_size,
            "sample_limit": args.vlad_sample_limit,
            "kmeans_iterations": args.vlad_kmeans_iterations,
            "embedding_cache_dir": args.embedding_cache_dir,
        }

    _write_jsonl(output_dir / "per_query_results.jsonl", query_rows)
    with (output_dir / "self_transform.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a self-supervised SE(2) map transform from cross-map Top1 retrieval matches."
    )
    parser.add_argument("--map-a", required=True, help="Reference map directory")
    parser.add_argument("--map-b", required=True, help="Query/eval map directory")
    parser.add_argument(
        "--descriptor-backend",
        choices=["stored", "superpoint-bow", "anyloc-vlad"],
        default="stored",
        help=(
            "'stored' uses embeddings.db, 'superpoint-bow' builds a map-A SuperPoint BoW vocabulary, "
            "'anyloc-vlad' builds a map-A DINOv2 patch-token VLAD vocabulary."
        ),
    )
    parser.add_argument("--output-dir", default="/tinynav/output/map_retrieval_self_consistency_fit")
    parser.add_argument("--topk", default="1,3,5,10")
    parser.add_argument("--distance-thresholds", default="0.5,1.0")
    parser.add_argument("--ransac-threshold-m", type=float, default=0.5)
    parser.add_argument("--ransac-iterations", type=int, default=3000)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bow-vocab-size", type=int, default=512)
    parser.add_argument("--bow-sample-limit", type=int, default=120_000)
    parser.add_argument("--bow-kmeans-iterations", type=int, default=80)
    parser.add_argument("--bow-kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--embedding-cache-dir", default="/tinynav/tinynav_temp/map_retrieval_descriptor_cache")
    parser.add_argument("--anyloc-model", default="", help="DINOv2 torch hub model name for AnyLoc VLAD")
    parser.add_argument("--anyloc-image-size", type=int, default=224)
    parser.add_argument("--anyloc-device", default="", help="Device for AnyLoc VLAD, for example cuda or cpu")
    parser.add_argument("--vlad-vocab-size", type=int, default=32)
    parser.add_argument("--vlad-sample-limit", type=int, default=120_000)
    parser.add_argument("--vlad-kmeans-iterations", type=int, default=80)
    parser.add_argument("--vlad-kmeans-batch-size", type=int, default=4096)
    args = parser.parse_args()

    summary = run_fit(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
