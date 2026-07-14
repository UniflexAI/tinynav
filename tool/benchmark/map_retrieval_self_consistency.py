#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from tinynav.core.build_map_node import TinyNavDB


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


def _normalize_rows(rows: list[np.ndarray], map_path: Path) -> np.ndarray:
    if not rows:
        raise RuntimeError(f"No descriptors loaded from {map_path}")
    x = np.stack([np.asarray(row, dtype=np.float32).reshape(-1) for row in rows], axis=0)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError(f"{map_path} contains near-zero descriptor")
    return x / np.maximum(norms, 1e-8)


def _load_stored_embeddings(map_path: Path, timestamps: list[int]) -> np.ndarray:
    db = TinyNavDB(str(map_path), is_scratch=False)
    rows: list[np.ndarray] = []
    try:
        for timestamp in timestamps:
            rows.append(np.asarray(db.get_embedding(timestamp), dtype=np.float32))
    finally:
        db.close()
    return _normalize_rows(rows, map_path)


def _valid_superpoint_desc(feature: dict[str, Any]) -> np.ndarray:
    desc = np.asarray(feature["descps"])
    mask = np.asarray(feature.get("mask", np.ones(desc.shape[:2], dtype=bool))).reshape(-1)
    desc = desc.reshape(-1, desc.shape[-1])[mask]
    if desc.size == 0:
        return np.zeros((0, 256), dtype=np.float32)
    desc = desc.astype(np.float32, copy=False)
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    return desc / np.maximum(norms, 1e-8)


def _load_superpoint_features(map_path: Path, timestamps: list[int]) -> list[np.ndarray]:
    db = TinyNavDB(str(map_path), is_scratch=False)
    rows: list[np.ndarray] = []
    try:
        for index, timestamp in enumerate(timestamps, start=1):
            rows.append(_valid_superpoint_desc(db.features[int(timestamp)]))
            if index % 300 == 0:
                print(f"Loaded SuperPoint features {index}/{len(timestamps)} for {map_path}", flush=True)
    finally:
        db.close()
    return rows


def _sample_descriptors(
    descs: list[np.ndarray],
    sample_limit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total = sum(len(desc) for desc in descs)
    if total == 0:
        raise RuntimeError("No SuperPoint descriptors available for BoW vocabulary")
    if total <= sample_limit:
        return np.concatenate(descs, axis=0).astype(np.float32, copy=False)

    pieces: list[np.ndarray] = []
    for desc in descs:
        if len(desc) == 0:
            continue
        count = min(len(desc), max(1, int(round(sample_limit * len(desc) / total))))
        indices = rng.choice(len(desc), size=count, replace=False)
        pieces.append(desc[indices])
    sample = np.concatenate(pieces, axis=0)
    if len(sample) > sample_limit:
        indices = rng.choice(len(sample), size=sample_limit, replace=False)
        sample = sample[indices]
    return sample.astype(np.float32, copy=False)


def _minibatch_kmeans(
    sample: np.ndarray,
    vocab_size: int,
    rng: np.random.Generator,
    iterations: int,
    batch_size: int,
) -> np.ndarray:
    if len(sample) < vocab_size:
        raise ValueError(f"Need at least {vocab_size} descriptors, got {len(sample)}")
    centers = sample[rng.choice(len(sample), size=vocab_size, replace=False)].copy()
    counts = np.zeros(vocab_size, dtype=np.int64)
    for iteration in range(iterations):
        batch = sample[rng.choice(len(sample), size=min(batch_size, len(sample)), replace=False)]
        labels = cKDTree(centers).query(batch, k=1, workers=-1)[1]
        for label, vector in zip(labels, batch):
            counts[label] += 1
            eta = 1.0 / counts[label]
            centers[label] = (1.0 - eta) * centers[label] + eta * vector
        if (iteration + 1) % 10 == 0:
            print(f"BoW k-means iteration {iteration + 1}/{iterations}", flush=True)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    return (centers / np.maximum(norms, 1e-8)).astype(np.float32)


def _bow_histograms(descs: list[np.ndarray], centers: np.ndarray) -> np.ndarray:
    tree = cKDTree(centers)
    histograms = np.zeros((len(descs), len(centers)), dtype=np.float32)
    for index, desc in enumerate(descs, start=1):
        if len(desc):
            labels = tree.query(desc, k=1, workers=-1)[1]
            histograms[index - 1] = np.bincount(labels, minlength=len(centers)).astype(np.float32)
        if index % 300 == 0:
            print(f"BoW quantized {index}/{len(descs)}", flush=True)
    return histograms


def _tfidf_normalize(map_a_hist: np.ndarray, map_b_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    document_frequency = np.count_nonzero(map_a_hist > 0, axis=0)
    idf = np.log((1.0 + map_a_hist.shape[0]) / (1.0 + document_frequency)) + 1.0

    def transform(histograms: np.ndarray) -> np.ndarray:
        x = np.log1p(histograms) * idf[None, :]
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return (x / np.maximum(norms, 1e-8)).astype(np.float32)

    return transform(map_a_hist), transform(map_b_hist)


def _load_superpoint_bow_embeddings(
    map_a: Path,
    map_b: Path,
    map_a_timestamps: list[int],
    map_b_timestamps: list[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    map_a_desc = _load_superpoint_features(map_a, map_a_timestamps)
    map_b_desc = _load_superpoint_features(map_b, map_b_timestamps)
    sample = _sample_descriptors(map_a_desc, args.bow_sample_limit, rng)
    print(
        f"Training BoW vocabulary: sample={sample.shape}, vocab_size={args.bow_vocab_size}",
        flush=True,
    )
    centers = _minibatch_kmeans(
        sample,
        args.bow_vocab_size,
        rng,
        args.bow_kmeans_iterations,
        args.bow_kmeans_batch_size,
    )
    map_a_hist = _bow_histograms(map_a_desc, centers)
    map_b_hist = _bow_histograms(map_b_desc, centers)
    return _tfidf_normalize(map_a_hist, map_b_hist)


def _load_descriptor_embeddings(
    map_a: Path,
    map_b: Path,
    map_a_timestamps: list[int],
    map_b_timestamps: list[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    if args.descriptor_backend == "stored":
        return (
            _load_stored_embeddings(map_a, map_a_timestamps),
            _load_stored_embeddings(map_b, map_b_timestamps),
        )
    if args.descriptor_backend == "superpoint-bow":
        return _load_superpoint_bow_embeddings(map_a, map_b, map_a_timestamps, map_b_timestamps, args)
    raise ValueError(f"Unsupported descriptor backend: {args.descriptor_backend}")


def _fit_se2(src_xy: np.ndarray, dst_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = src_xy.mean(axis=0)
    dst_mean = dst_xy.mean(axis=0)
    src_centered = src_xy - src_mean
    dst_centered = dst_xy - dst_mean
    h = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = dst_mean - rotation @ src_mean
    return rotation, translation


def _apply_se2(xy: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return xy @ rotation.T + translation[None, :]


def _ransac_fit_se2(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    threshold_m: float,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(len(src_xy), dtype=bool)
    best_residuals = np.full(len(src_xy), np.inf, dtype=np.float64)
    for _ in range(iterations):
        indices = rng.choice(len(src_xy), size=2, replace=False)
        if np.linalg.norm(src_xy[indices[0]] - src_xy[indices[1]]) < 1e-6:
            continue
        rotation, translation = _fit_se2(src_xy[indices], dst_xy[indices])
        residuals = np.linalg.norm(_apply_se2(src_xy, rotation, translation) - dst_xy, axis=1)
        inliers = residuals <= threshold_m
        if inliers.sum() > best_inliers.sum() or (
            inliers.sum() == best_inliers.sum()
            and np.median(residuals[inliers]) < np.median(best_residuals[best_inliers])
        ):
            best_inliers = inliers
            best_residuals = residuals
    if best_inliers.sum() >= 2:
        rotation, translation = _fit_se2(src_xy[best_inliers], dst_xy[best_inliers])
    else:
        rotation, translation = _fit_se2(src_xy, dst_xy)
    residuals = np.linalg.norm(_apply_se2(src_xy, rotation, translation) - dst_xy, axis=1)
    return rotation, translation, residuals


def _retrieve_rows(
    similarities: np.ndarray,
    map_a_timestamps: list[int],
    map_a_poses: dict[int, np.ndarray],
    map_b_timestamps: list[int],
    topk: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_index, timestamp_b in enumerate(map_b_timestamps):
        query_similarities = similarities[:, query_index]
        if topk >= len(query_similarities):
            top_indices = np.argsort(-query_similarities)
        else:
            unsorted = np.argpartition(-query_similarities, topk - 1)[:topk]
            top_indices = unsorted[np.argsort(-query_similarities[unsorted])]
        retrieved = []
        for rank, map_a_index in enumerate(top_indices, start=1):
            timestamp_a = int(map_a_timestamps[int(map_a_index)])
            retrieved.append(
                {
                    "rank": rank,
                    "timestamp_ns": timestamp_a,
                    "similarity": float(query_similarities[int(map_a_index)]),
                    "pose_xy": map_a_poses[timestamp_a][:2, 3].tolist(),
                }
            )
        rows.append({"query_timestamp_ns": int(timestamp_b), "retrieved": retrieved})
    return rows


def _positive_set(
    map_a_timestamps: list[int],
    map_a_positions: np.ndarray,
    gt_xy: np.ndarray,
    threshold_m: float,
) -> set[int]:
    distances = np.linalg.norm(map_a_positions[:, :2] - gt_xy[None, :], axis=1)
    return {int(map_a_timestamps[index]) for index in np.flatnonzero(distances <= threshold_m)}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
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
    map_a_positions = np.stack([map_a_poses[timestamp][:3, 3] for timestamp in map_a_timestamps], axis=0)

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

    yaw_deg = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    t4 = np.eye(4, dtype=np.float64)
    t4[:2, :2] = rotation
    t4[:2, 3] = translation

    for row, gt_xy, residual in zip(query_rows, query_gt_xy, top1_residuals):
        row["pose_a_gt_xy_self_consistency"] = gt_xy.tolist()
        row["top1_residual_m"] = float(residual)

    summary = {
        "type": "cross_map_self_consistency",
        "note": (
            "T is fitted from this descriptor backend's Top1 keyframe matches. "
            "Metrics are self-consistency signals, not external GT accuracy."
        ),
        "descriptor_backend": args.descriptor_backend,
        "map_a": str(map_a),
        "map_b": str(map_b),
        "map_a_keyframes": len(map_a_timestamps),
        "map_b_queries": len(query_rows),
        "topk": topk_values,
        "distance_thresholds_m": thresholds,
        "ransac_threshold_m": args.ransac_threshold_m,
        "T_map_a_map_b_self_consistency_se2": {
            "T": t4.tolist(),
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
        "metrics": metrics,
        "elapsed_s": time.time() - start_time,
    }
    if args.descriptor_backend == "superpoint-bow":
        summary["bow"] = {
            "vocab_size": args.bow_vocab_size,
            "sample_limit": args.bow_sample_limit,
            "kmeans_iterations": args.bow_kmeans_iterations,
        }

    _write_jsonl(output_dir / "per_query_results.jsonl", query_rows)
    _write_csv(output_dir / "metrics.csv", metrics)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Self-supervised cross-map retrieval consistency evaluation. "
            "Map B keyframes query Map A; a SE(2) transform is fitted from Top1 matches."
        )
    )
    parser.add_argument("--map-a", required=True, help="Reference map directory")
    parser.add_argument("--map-b", required=True, help="Query/eval map directory")
    parser.add_argument(
        "--descriptor-backend",
        choices=["stored", "superpoint-bow"],
        default="stored",
        help="'stored' uses embeddings.db, 'superpoint-bow' builds a map-A SuperPoint BoW vocabulary.",
    )
    parser.add_argument("--output-dir", default="/tinynav/output/map_retrieval_self_consistency")
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
    args = parser.parse_args()

    summary = run_eval(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
