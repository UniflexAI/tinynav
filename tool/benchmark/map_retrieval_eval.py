#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tinynav.core.build_map_node import TinyNavDB


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    timestamp_ns: int
    similarity: float
    distance_m: float


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_poses(map_path: Path) -> dict[int, np.ndarray]:
    poses_path = map_path / "poses.npy"
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing poses file: {poses_path}")
    raw_poses = np.load(poses_path, allow_pickle=True).item()
    return {int(timestamp): np.asarray(pose, dtype=np.float64) for timestamp, pose in raw_poses.items()}


def _load_tba(tba_json: Path | None) -> np.ndarray:
    if tba_json is None:
        return np.eye(4, dtype=np.float64)
    with tba_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "Tba" not in data:
        raise KeyError(f"{tba_json} must contain a 'Tba' 4x4 matrix")
    tba = np.asarray(data["Tba"], dtype=np.float64)
    if tba.shape != (4, 4):
        raise ValueError(f"Tba must be a 4x4 matrix, got shape {tba.shape}")
    return tba


def _load_normalized_embeddings(map_path: Path, timestamps: list[int]) -> np.ndarray:
    db = TinyNavDB(str(map_path), is_scratch=False)
    embeddings: list[np.ndarray] = []
    try:
        for timestamp in timestamps:
            embedding = np.asarray(db.get_embedding(timestamp), dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(embedding))
            if norm <= 1e-8:
                raise ValueError(f"Embedding for timestamp {timestamp} in {map_path} has near-zero norm")
            embeddings.append(embedding / norm)
    finally:
        db.close()
    if not embeddings:
        raise RuntimeError(f"No embeddings found in {map_path}")
    return np.stack(embeddings, axis=0)


def _pose_distance(pose_a: np.ndarray, pose_b: np.ndarray, mode: str) -> float:
    delta = pose_a[:3, 3] - pose_b[:3, 3]
    if mode == "xy":
        return float(np.linalg.norm(delta[:2]))
    if mode == "xyz":
        return float(np.linalg.norm(delta))
    raise ValueError(f"Unsupported distance mode: {mode}")


def _retrieve_topk(
    query_embedding: np.ndarray,
    map_a_embeddings: np.ndarray,
    map_a_timestamps: list[int],
    map_a_poses: dict[int, np.ndarray],
    pose_a_gt: np.ndarray,
    topk: int,
    distance_mode: str,
) -> list[RetrievalHit]:
    similarities = map_a_embeddings @ query_embedding
    if topk >= len(similarities):
        top_indices = np.argsort(-similarities)
    else:
        unsorted = np.argpartition(-similarities, topk - 1)[:topk]
        top_indices = unsorted[np.argsort(-similarities[unsorted])]

    hits: list[RetrievalHit] = []
    for rank, index in enumerate(top_indices, start=1):
        timestamp = int(map_a_timestamps[int(index)])
        distance = _pose_distance(map_a_poses[timestamp], pose_a_gt, distance_mode)
        hits.append(
            RetrievalHit(
                rank=rank,
                timestamp_ns=timestamp,
                similarity=float(similarities[int(index)]),
                distance_m=distance,
            )
        )
    return hits


def _positive_set(
    map_a_timestamps: list[int],
    map_a_positions: np.ndarray,
    pose_a_gt: np.ndarray,
    threshold_m: float,
    distance_mode: str,
) -> set[int]:
    gt_position = pose_a_gt[:3, 3]
    deltas = map_a_positions - gt_position[None, :]
    if distance_mode == "xy":
        distances = np.linalg.norm(deltas[:, :2], axis=1)
    elif distance_mode == "xyz":
        distances = np.linalg.norm(deltas, axis=1)
    else:
        raise ValueError(f"Unsupported distance mode: {distance_mode}")
    return {int(map_a_timestamps[i]) for i in np.flatnonzero(distances <= threshold_m)}


def _metrics_for_threshold(
    query_rows: list[dict[str, Any]],
    map_a_timestamps: list[int],
    map_a_positions: np.ndarray,
    threshold_m: float,
    topk_values: list[int],
    distance_mode: str,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for topk in topk_values:
        hit_count = 0
        precision_values: list[float] = []
        recall_values: list[float] = []
        iou_values: list[float] = []

        for row in query_rows:
            pose_a_gt = np.asarray(row["pose_a_gt"], dtype=np.float64)
            gt_set = _positive_set(map_a_timestamps, map_a_positions, pose_a_gt, threshold_m, distance_mode)
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
                "threshold_m": threshold_m,
                "topk": topk,
                "query_count": len(query_rows),
                "hit_count": hit_count,
                "recall_at_k": hit_count / max(1, len(query_rows)),
                "mean_precision": float(np.mean(precision_values)) if precision_values else 0.0,
                "mean_set_recall": float(np.mean(recall_values)) if recall_values else 0.0,
                "mean_iou": float(np.mean(iou_values)) if iou_values else 0.0,
            }
        )
    return metrics


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
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

    map_a_embeddings = _load_normalized_embeddings(map_a, map_a_timestamps)
    map_b_embeddings_all = _load_normalized_embeddings(map_b, map_b_timestamps)
    tba = _load_tba(Path(args.tba_json) if args.tba_json else None)

    topk_values = _parse_int_list(args.topk)
    thresholds = _parse_float_list(args.distance_thresholds)
    max_topk = max(topk_values)
    if max_topk <= 0:
        raise ValueError("--topk values must be positive")
    if max_topk > len(map_a_timestamps):
        max_topk = len(map_a_timestamps)

    map_a_positions = np.stack([map_a_poses[timestamp][:3, 3] for timestamp in map_a_timestamps], axis=0)

    query_rows: list[dict[str, Any]] = []
    for timestamp_b, embedding_b in zip(map_b_timestamps, map_b_embeddings_all):
        pose_b = map_b_poses[timestamp_b]
        pose_a_gt = tba @ pose_b
        hits = _retrieve_topk(
            embedding_b,
            map_a_embeddings,
            map_a_timestamps,
            map_a_poses,
            pose_a_gt,
            max_topk,
            args.distance_mode,
        )
        distances = [hit.distance_m for hit in hits]
        row = {
            "query_timestamp_ns": int(timestamp_b),
            "pose_a_gt": pose_a_gt.tolist(),
            "retrieved": [
                {
                    "rank": hit.rank,
                    "timestamp_ns": hit.timestamp_ns,
                    "similarity": hit.similarity,
                    "distance_m": hit.distance_m,
                }
                for hit in hits
            ],
            "top1_error_m": distances[0] if distances else None,
            "topk_min_error_m": min(distances) if distances else None,
        }
        for topk in topk_values:
            top_distances = distances[: min(topk, len(distances))]
            row[f"top{topk}_min_error_m"] = min(top_distances) if top_distances else None
        query_rows.append(row)

    metrics: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics.extend(
            _metrics_for_threshold(
                query_rows,
                map_a_timestamps,
                map_a_positions,
                threshold,
                topk_values,
                args.distance_mode,
            )
        )

    top1_errors = [row["top1_error_m"] for row in query_rows if row["top1_error_m"] is not None]
    topk_errors = [row["topk_min_error_m"] for row in query_rows if row["topk_min_error_m"] is not None]
    summary = {
        "map_a": str(map_a),
        "map_b": str(map_b),
        "tba_json": str(args.tba_json) if args.tba_json else None,
        "distance_mode": args.distance_mode,
        "map_a_keyframes": len(map_a_timestamps),
        "map_b_queries": len(query_rows),
        "topk": topk_values,
        "distance_thresholds_m": thresholds,
        "top1_error_m": {
            "mean": float(np.mean(top1_errors)) if top1_errors else None,
            "median": float(np.median(top1_errors)) if top1_errors else None,
            "max": float(np.max(top1_errors)) if top1_errors else None,
        },
        "topk_min_error_m": {
            "mean": float(np.mean(topk_errors)) if topk_errors else None,
            "median": float(np.median(topk_errors)) if topk_errors else None,
            "max": float(np.max(topk_errors)) if topk_errors else None,
        },
        "metrics": metrics,
    }

    _write_jsonl(output_dir / "per_query_results.jsonl", query_rows)
    _write_metrics_csv(output_dir / "metrics.csv", metrics)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cross-map image retrieval with a fixed transform Tba. "
            "Map B keyframes are used as queries, Map A keyframes are the retrieval database."
        )
    )
    parser.add_argument("--map-a", required=True, help="Reference map directory")
    parser.add_argument("--map-b", required=True, help="Query map directory")
    parser.add_argument("--tba-json", default="", help="JSON file containing a 4x4 Tba matrix. Defaults to identity.")
    parser.add_argument("--output-dir", default="/tinynav/output/map_retrieval_eval", help="Output directory")
    parser.add_argument("--topk", default="1,3,5,10", help="Comma-separated topK values")
    parser.add_argument(
        "--distance-thresholds",
        default="0.5,1.0,2.0,3.0,5.0",
        help="Comma-separated distance thresholds in meters",
    )
    parser.add_argument("--distance-mode", choices=["xy", "xyz"], default="xy", help="Position distance mode")
    parser.add_argument("--every-n", type=int, default=1, help="Evaluate every Nth Map B keyframe")
    parser.add_argument("--max-queries", type=int, default=0, help="Maximum number of Map B queries. 0 means no cap.")
    args = parser.parse_args()

    summary = run_eval(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
