#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from tool.benchmark.map_retrieval_self_consistency import (
    _apply_se2,
    _load_poses,
    _parse_float_list,
    _parse_int_list,
    _positive_set,
    _write_csv,
    _write_jsonl,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_self_transform(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    transform_data = data["T_map_a_map_b_self_consistency_se2"]
    if "R_xy" in transform_data and "t_xy" in transform_data:
        rotation = np.asarray(transform_data["R_xy"], dtype=np.float64)
        translation = np.asarray(transform_data["t_xy"], dtype=np.float64)
    else:
        transform = np.asarray(transform_data["T"], dtype=np.float64)
        rotation = transform[:2, :2]
        translation = transform[:2, 3]
    return rotation, translation, data


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.time()
    map_a = Path(args.map_a)
    map_b = Path(args.map_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    map_a_poses = _load_poses(map_a)
    map_b_poses = _load_poses(map_b)
    map_a_timestamps = sorted(map_a_poses)
    map_a_positions = np.stack([map_a_poses[timestamp][:3, 3] for timestamp in map_a_timestamps], axis=0)

    query_rows = _load_jsonl(Path(args.retrieval_json))
    rotation, translation, transform_summary = _load_self_transform(Path(args.transform_json))
    topk_values = _parse_int_list(args.topk)
    thresholds = _parse_float_list(args.distance_thresholds)

    src_xy = np.stack([map_b_poses[int(row["query_timestamp_ns"])][:2, 3] for row in query_rows], axis=0)
    query_gt_xy = _apply_se2(src_xy, rotation, translation)
    top1_residuals = []
    for row, gt_xy in zip(query_rows, query_gt_xy):
        top1_timestamp = int(row["retrieved"][0]["timestamp_ns"])
        residual = float(np.linalg.norm(map_a_poses[top1_timestamp][:2, 3] - gt_xy))
        row["pose_a_gt_xy_self_consistency"] = gt_xy.tolist()
        row["top1_residual_m"] = residual
        top1_residuals.append(residual)
    top1_residuals_np = np.asarray(top1_residuals, dtype=np.float64)

    metrics: list[dict[str, Any]] = []
    for threshold in thresholds:
        for topk in topk_values:
            hit_count = 0
            precision_values = []
            recall_values = []
            iou_values = []
            for row, gt_xy in zip(query_rows, query_gt_xy):
                gt_set = _positive_set(map_a_timestamps, map_a_positions, gt_xy, threshold)
                predicted = {int(hit["timestamp_ns"]) for hit in row["retrieved"][:topk]}
                intersection = predicted & gt_set
                union = predicted | gt_set
                if intersection:
                    hit_count += 1
                precision_values.append(len(intersection) / max(1, len(predicted)))
                recall_values.append(len(intersection) / len(gt_set) if gt_set else 0.0)
                iou_values.append(len(intersection) / len(union) if union else 0.0)
            metrics.append(
                {
                    "threshold_m": threshold,
                    "topk": topk,
                    "query_count": len(query_rows),
                    "hit_count": hit_count,
                    "recall_at_k": hit_count / max(1, len(query_rows)),
                    "mean_precision": float(np.mean(precision_values)),
                    "mean_set_recall": float(np.mean(recall_values)),
                    "mean_iou": float(np.mean(iou_values)),
                }
            )

    summary = {
        "type": "cross_map_self_consistency_eval",
        "note": "Metrics are evaluated with a self-supervised transform fitted from retrieval matches, not external GT.",
        "descriptor_backend": transform_summary.get("descriptor_backend"),
        "map_a": str(map_a),
        "map_b": str(map_b),
        "transform_json": str(args.transform_json),
        "retrieval_json": str(args.retrieval_json),
        "map_a_keyframes": len(map_a_timestamps),
        "map_b_queries": len(query_rows),
        "topk": topk_values,
        "distance_thresholds_m": thresholds,
        "T_map_a_map_b_self_consistency_se2": transform_summary["T_map_a_map_b_self_consistency_se2"],
        "top1_residual_m": {
            "mean": float(np.mean(top1_residuals_np)),
            "median": float(np.median(top1_residuals_np)),
            "p90": float(np.percentile(top1_residuals_np, 90)),
            "max": float(np.max(top1_residuals_np)),
        },
        "top1_inlier_ratio": {
            f"{threshold}m": float(np.mean(top1_residuals_np <= threshold)) for threshold in thresholds
        },
        "metrics": metrics,
        "elapsed_s": time.time() - start_time,
    }

    _write_jsonl(output_dir / "per_query_results.jsonl", query_rows)
    _write_csv(output_dir / "metrics.csv", metrics)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Top-K retrieval self-consistency with a previously fitted self-supervised transform."
    )
    parser.add_argument("--map-a", required=True, help="Reference map directory")
    parser.add_argument("--map-b", required=True, help="Query/eval map directory")
    parser.add_argument("--transform-json", required=True, help="self_transform.json from map_retrieval_fit_self_t.py")
    parser.add_argument("--retrieval-json", required=True, help="per_query_results.jsonl from map_retrieval_fit_self_t.py")
    parser.add_argument("--output-dir", default="/tinynav/output/map_retrieval_self_consistency_eval")
    parser.add_argument("--topk", default="1,3,5,10")
    parser.add_argument("--distance-thresholds", default="0.5,1.0")
    args = parser.parse_args()

    summary = run_eval(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
