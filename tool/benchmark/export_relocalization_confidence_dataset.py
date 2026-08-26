#!/usr/bin/env python3
"""
Export a relocalization confidence dataset from two existing TinyNav maps.

The script treats `map_gt` as the retrieval/reference database and `map_eval`
as the query map. It first estimates T_map_eval_to_gt from Top1 retrieval
matches, then exports TopK SuperPoint+LightGlue+PnP attempts with automatic
candidate/keypoint labels:

    good candidate:
        PnP succeeds and pose error to T_map_eval_to_gt * eval_pose is small.

    bad candidate:
        PnP succeeds but pose error is large.

    keypoint label:
        1  reliable PnP inlier from a good candidate
        0  unreliable PnP inlier from a bad candidate
       -1  ignore / not enough supervision

This is intended as raw training data for a learned relocalization keypoint
confidence model. It does not modify runtime relocalization behavior.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from map_fusion_benchmark import (
    _depth_distribution,
    _draw_match_image,
    _image_shape_wh,
    _inlier_distribution,
    _keypoints_to_world,
    _landmark_geometry,
    _load_pose_dict,
    _match_keypoints,
    _pnp_pose,
    _ransac_transform,
    _sample_timestamps_from_map,
    _summary,
)
from tinynav.core.build_map_node import TinyNavDB, find_loop
from tinynav.core.models_trt import LightGlueTRT


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in Path(value).name)


def _json_array(array: np.ndarray, *, precision: int | None = None) -> list:
    arr = np.asarray(array)
    if precision is not None and np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr.astype(float), precision)
    return arr.tolist()


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def _image_loader_image(loader) -> np.ndarray:
    image = loader()
    if image is None:
        raise RuntimeError("Map image loader returned None. Is this an existing non-scratch map?")
    return image


def _retrieve_candidates(
    eval_desc: np.ndarray,
    gt_descriptors: np.ndarray,
    *,
    top_k: int,
) -> list[tuple[int, float]]:
    # find_loop returns the best candidates at the end of an ascending list.
    return list(reversed(find_loop(eval_desc, gt_descriptors, -1.0, top_k)))


def _fit_transform_from_top1_retrieval(
    *,
    map_gt_dir: Path,
    map_eval_dir: Path,
    gt_db: TinyNavDB,
    eval_db: TinyNavDB,
    gt_timestamps: list[int],
    gt_descriptors: np.ndarray,
    ransac_threshold_m: float,
    ransac_iterations: int,
    seed: int,
    alignment_mode: str,
) -> tuple[np.ndarray, dict, dict[int, dict]]:
    gt_poses = _load_pose_dict(map_gt_dir / "poses.npy")
    eval_poses = _load_pose_dict(map_eval_dir / "poses.npy")

    source_poses: dict[int, np.ndarray] = {}
    target_poses: dict[int, np.ndarray] = {}
    top1_rows: dict[int, dict] = {}
    missing_vlad = 0

    for eval_ts in sorted(eval_poses):
        if eval_ts not in eval_db.vlad_descriptors:
            missing_vlad += 1
            continue
        candidates = _retrieve_candidates(
            eval_db.vlad_descriptors[eval_ts],
            gt_descriptors,
            top_k=1,
        )
        if not candidates:
            continue
        gt_idx, similarity = candidates[0]
        gt_ts = int(gt_timestamps[int(gt_idx)])
        source_poses[int(eval_ts)] = eval_poses[int(eval_ts)]
        target_poses[int(eval_ts)] = gt_poses[gt_ts]
        top1_rows[int(eval_ts)] = {
            "gt_timestamp_ns": gt_ts,
            "similarity": float(similarity),
        }

    if len(source_poses) < 3:
        raise RuntimeError(
            f"Need at least 3 Top1 retrieval pairs to fit transform, got {len(source_poses)}"
        )

    transform, inlier_timestamps, info = _ransac_transform(
        source_poses=source_poses,
        target_poses=target_poses,
        inlier_threshold_m=ransac_threshold_m,
        iterations=ransac_iterations,
        seed=seed,
        alignment_mode=alignment_mode,
    )
    info = {
        **info,
        "fit_source": "top1_retrieval_self_consistency",
        "top1_pairs": len(source_poses),
        "missing_eval_vlad_descriptors": missing_vlad,
    }
    inlier_set = set(map(int, inlier_timestamps))
    for eval_ts, row in top1_rows.items():
        row["used_as_transform_inlier"] = eval_ts in inlier_set
    return transform, info, top1_rows


def _candidate_label(
    *,
    pnp_success: bool,
    pose_error_m: float | None,
    good_error_threshold_m: float,
    bad_error_threshold_m: float,
) -> str:
    if not pnp_success or pose_error_m is None:
        return "pnp_fail"
    if pose_error_m <= good_error_threshold_m:
        return "good"
    if pose_error_m >= bad_error_threshold_m:
        return "bad"
    return "ignore"


def _keypoint_labels(
    *,
    match_count: int,
    pnp_inlier_indices: np.ndarray,
    candidate_label: str,
) -> list[int]:
    labels = np.full(match_count, -1, dtype=np.int32)
    if candidate_label == "good":
        labels[pnp_inlier_indices.astype(np.int32)] = 1
    elif candidate_label == "bad":
        labels[pnp_inlier_indices.astype(np.int32)] = 0
    return labels.tolist()


def _export_sample_candidate(
    *,
    sample_dir: Path,
    eval_ts: int,
    gt_ts: int,
    rank: int,
    similarity: float,
    next_similarity: float | None,
    expected_pose_in_gt: np.ndarray,
    gt_pose: np.ndarray,
    gt_depth: np.ndarray,
    gt_features: dict,
    gt_image: np.ndarray,
    eval_features: dict,
    eval_image: np.ndarray,
    gt_K: np.ndarray,
    eval_K: np.ndarray,
    matcher: LightGlueTRT,
    loop: asyncio.AbstractEventLoop,
    min_inliers: int,
    good_error_threshold_m: float,
    bad_error_threshold_m: float,
    max_match_lines: int,
    include_arrays: bool,
) -> dict:
    gt_shape = _image_shape_wh(gt_image)
    eval_shape = _image_shape_wh(eval_image)
    ref_kpts_all, query_kpts_all = _match_keypoints(
        matcher,
        gt_features,
        eval_features,
        gt_shape,
        eval_shape,
        loop=loop,
    )
    points_world, depth_valid = _keypoints_to_world(ref_kpts_all, gt_depth, gt_pose, gt_K)
    original_valid_indices = np.where(depth_valid)[0]
    points_world_valid = points_world[depth_valid].astype(np.float32)
    query_valid = query_kpts_all[depth_valid].astype(np.float32)

    success, pose_camera_to_world, pnp_inliers = _pnp_pose(
        points_world_valid,
        query_valid,
        eval_K,
        min_inliers,
    )
    original_inliers = (
        original_valid_indices[pnp_inliers]
        if success
        else np.empty((0,), dtype=np.int32)
    )

    pose_error_m = None
    pose_delta_xyz_m = None
    if success:
        pose_delta = pose_camera_to_world[:3, 3] - expected_pose_in_gt[:3, 3]
        pose_delta_xyz_m = pose_delta.tolist()
        pose_error_m = float(np.linalg.norm(pose_delta))

    label = _candidate_label(
        pnp_success=success,
        pose_error_m=pose_error_m,
        good_error_threshold_m=good_error_threshold_m,
        bad_error_threshold_m=bad_error_threshold_m,
    )
    labels = _keypoint_labels(
        match_count=len(query_kpts_all),
        pnp_inlier_indices=original_inliers,
        candidate_label=label,
    )

    depth_values = gt_depth[
        np.clip(np.round(ref_kpts_all[:, 1]).astype(int), 0, gt_depth.shape[0] - 1),
        np.clip(np.round(ref_kpts_all[:, 0]).astype(int), 0, gt_depth.shape[1] - 1),
    ].astype(np.float32)
    inlier_depths = depth_values[original_inliers] if success else np.empty((0,), dtype=np.float32)
    inlier_points = points_world[original_inliers] if success else np.empty((0, 3), dtype=np.float32)

    sample_dir.mkdir(parents=True, exist_ok=True)
    query_path = sample_dir / "query.png"
    reference_path = sample_dir / "reference.png"
    match_path = sample_dir / "match_vis.jpg"
    cv2.imwrite(str(query_path), eval_image)
    cv2.imwrite(str(reference_path), gt_image)
    cv2.imwrite(
        str(match_path),
        _draw_match_image(
            gt_image,
            eval_image,
            ref_kpts_all,
            query_kpts_all,
            original_inliers,
            max_match_lines,
        ),
    )

    row = {
        "query_timestamp_ns": int(eval_ts),
        "reference_timestamp_ns": int(gt_ts),
        "candidate_rank": int(rank),
        "retrieval_similarity": float(similarity),
        "retrieval_margin_to_next": (
            float(similarity - next_similarity) if next_similarity is not None else None
        ),
        "pnp_success": bool(success),
        "pnp_inlier_count": int(len(original_inliers)),
        "pnp_inlier_ratio": float(len(original_inliers) / max(len(query_valid), 1)),
        "match_count": int(len(query_kpts_all)),
        "valid_landmark_count": int(len(query_valid)),
        "pose_error_m": pose_error_m,
        "pose_delta_xyz_m": pose_delta_xyz_m,
        "candidate_label": label,
        "image_shape_query_hw": list(map(int, eval_image.shape[:2])),
        "image_shape_reference_hw": list(map(int, gt_image.shape[:2])),
        "files": {
            "query": "query.png",
            "reference": "reference.png",
            "match_vis": "match_vis.jpg",
        },
        "label_meaning": {
            "1": "reliable PnP inlier from good relocalization",
            "0": "unreliable PnP inlier from bad relocalization",
            "-1": "ignore / unused",
        },
        "inlier_distribution": _inlier_distribution(query_kpts_all, original_inliers, eval_image.shape[:2]),
        "depth_distribution": _depth_distribution(inlier_depths),
        "landmark_geometry": _landmark_geometry(inlier_points),
        "expected_pose_in_gt": expected_pose_in_gt.tolist(),
        "pnp_pose_in_gt": pose_camera_to_world.tolist() if success else None,
    }

    if include_arrays:
        row.update(
            {
                "query_keypoints": _json_array(query_kpts_all, precision=3),
                "reference_keypoints": _json_array(ref_kpts_all, precision=3),
                "matches": [[i, i] for i in range(len(query_kpts_all))],
                "depth_valid_match_indices": original_valid_indices.astype(int).tolist(),
                "pnp_inlier_match_indices": original_inliers.astype(int).tolist(),
                "keypoint_labels": labels,
            }
        )
    else:
        _write_json(
            sample_dir / "keypoints.json",
            {
                "query_keypoints": _json_array(query_kpts_all, precision=3),
                "reference_keypoints": _json_array(ref_kpts_all, precision=3),
                "matches": [[i, i] for i in range(len(query_kpts_all))],
                "depth_valid_match_indices": original_valid_indices.astype(int).tolist(),
                "pnp_inlier_match_indices": original_inliers.astype(int).tolist(),
                "keypoint_labels": labels,
            },
        )
        row["files"]["keypoints"] = "keypoints.json"

    _write_json(sample_dir / "sample.json", row)
    return row


def _write_html(output_dir: Path, rows: list[dict], summary: dict) -> None:
    cards = []
    for row in rows:
        rel_dir = Path(row["sample_dir"]).relative_to(output_dir)
        match_uri = html.escape(str(rel_dir / "match_vis.jpg"))
        label = html.escape(row["candidate_label"])
        cards.append(
            f"""
            <section class="card {label}">
              <h2>sample {row['sample_index']:04d} · top {row['candidate_rank']} · {label}</h2>
              <p>
                query={row['query_timestamp_ns']} · ref={row['reference_timestamp_ns']} ·
                sim={row['retrieval_similarity']:.4f} ·
                matches={row['match_count']} · landmarks={row['valid_landmark_count']} ·
                inliers={row['pnp_inlier_count']} · error={row['pose_error_m'] if row['pose_error_m'] is not None else 'n/a'} m
              </p>
              <p>
                lower-half={row['inlier_distribution']['lower_half_ratio']:.3f} ·
                bottom-third={row['inlier_distribution']['bottom_third_ratio']:.3f} ·
                x/y-span={row['inlier_distribution']['x_span_norm']:.3f}/{row['inlier_distribution']['y_span_norm']:.3f}
              </p>
              <img src="{match_uri}" />
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>TinyNav Relocalization Confidence Dataset</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    pre {{ background: #0f172a; color: #e2e8f0; padding: 16px; border-radius: 12px; overflow: auto; }}
    .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 18px; margin: 18px 0; box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06); }}
    .good {{ border-left: 8px solid #22c55e; }}
    .bad {{ border-left: 8px solid #ef4444; }}
    .ignore, .pnp_fail {{ border-left: 8px solid #94a3b8; }}
    img {{ max-width: 100%; border-radius: 12px; border: 1px solid #e2e8f0; }}
  </style>
</head>
<body><main>
  <h1>TinyNav Relocalization Confidence Dataset</h1>
  <p>Green lines in match images are PnP inliers. Labels are generated from PnP pose error against the fitted map_eval→map_gt transform.</p>
  <h2>summary</h2>
  <pre>{html.escape(json.dumps(summary, indent=2))}</pre>
  {''.join(cards)}
</main></body></html>
"""
    (output_dir / "index.html").write_text(html_text)


def run(args: argparse.Namespace) -> Path:
    map_gt_dir = Path(args.map_gt).resolve()
    map_eval_dir = Path(args.map_eval).resolve()
    if not map_gt_dir.exists():
        raise FileNotFoundError(map_gt_dir)
    if not map_eval_dir.exists():
        raise FileNotFoundError(map_eval_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path(args.output_root).resolve()
            / f"{timestamp}_{_safe_name(str(map_gt_dir))}_vs_{_safe_name(str(map_eval_dir))}_relocalization_confidence_dataset"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    gt_poses = _load_pose_dict(map_gt_dir / "poses.npy")
    eval_poses = _load_pose_dict(map_eval_dir / "poses.npy")
    gt_K = np.load(map_gt_dir / "intrinsics.npy")
    eval_K = np.load(map_eval_dir / "intrinsics.npy")
    gt_timestamps = sorted(gt_poses)

    gt_db = TinyNavDB(str(map_gt_dir), is_scratch=False)
    eval_db = TinyNavDB(str(map_eval_dir), is_scratch=False)
    loop = asyncio.new_event_loop()
    matcher = LightGlueTRT()
    rows: list[dict] = []
    top1_rows: dict[int, dict] = {}

    try:
        gt_descriptors = np.stack([gt_db.vlad_descriptors[t] for t in gt_timestamps])
        transform, transform_info, top1_rows = _fit_transform_from_top1_retrieval(
            map_gt_dir=map_gt_dir,
            map_eval_dir=map_eval_dir,
            gt_db=gt_db,
            eval_db=eval_db,
            gt_timestamps=gt_timestamps,
            gt_descriptors=gt_descriptors,
            ransac_threshold_m=args.ransac_threshold_m,
            ransac_iterations=args.ransac_iterations,
            seed=args.seed,
            alignment_mode=args.alignment_mode,
        )

        if args.timestamps_file:
            sampled_timestamps = np.loadtxt(args.timestamps_file, dtype=np.int64)
        else:
            sampled_timestamps = _sample_timestamps_from_map(
                map_eval_dir,
                args.num_samples,
                args.trim_ratio,
            )

        for sample_index, eval_ts_raw in enumerate(sampled_timestamps):
            eval_ts = int(eval_ts_raw)
            if eval_ts not in eval_poses or eval_ts not in eval_db.vlad_descriptors:
                continue

            eval_depth, _, eval_features, _, eval_image_loader = eval_db.get_depth_embedding_features_images(eval_ts)
            del eval_depth
            eval_image = _image_loader_image(eval_image_loader)
            expected_pose_in_gt = transform @ eval_poses[eval_ts]
            candidates = _retrieve_candidates(
                eval_db.vlad_descriptors[eval_ts],
                gt_descriptors,
                top_k=args.top_k,
            )

            for rank, (gt_idx, similarity) in enumerate(candidates, start=1):
                next_similarity = candidates[rank][1] if rank < len(candidates) else None
                gt_ts = int(gt_timestamps[int(gt_idx)])
                gt_depth, _, gt_features, _, gt_image_loader = gt_db.get_depth_embedding_features_images(gt_ts)
                gt_image = _image_loader_image(gt_image_loader)
                sample_dir = samples_dir / f"sample_{sample_index:06d}_rank_{rank:02d}_{eval_ts}_{gt_ts}"
                row = _export_sample_candidate(
                    sample_dir=sample_dir,
                    eval_ts=eval_ts,
                    gt_ts=gt_ts,
                    rank=rank,
                    similarity=float(similarity),
                    next_similarity=float(next_similarity) if next_similarity is not None else None,
                    expected_pose_in_gt=expected_pose_in_gt,
                    gt_pose=gt_poses[gt_ts],
                    gt_depth=gt_depth,
                    gt_features=gt_features,
                    gt_image=gt_image,
                    eval_features=eval_features,
                    eval_image=eval_image,
                    gt_K=gt_K,
                    eval_K=eval_K,
                    matcher=matcher,
                    loop=loop,
                    min_inliers=args.min_inliers,
                    good_error_threshold_m=args.good_error_threshold_m,
                    bad_error_threshold_m=args.bad_error_threshold_m,
                    max_match_lines=args.max_match_lines,
                    include_arrays=args.include_arrays_in_sample_json,
                )
                row["sample_index"] = int(sample_index)
                row["sample_dir"] = str(sample_dir)
                rows.append(row)
    finally:
        loop.close()
        gt_db.close()
        eval_db.close()

    label_counts: dict[str, int] = {}
    for row in rows:
        label_counts[row["candidate_label"]] = label_counts.get(row["candidate_label"], 0) + 1
    pose_errors = [row["pose_error_m"] for row in rows if row["pose_error_m"] is not None]
    summary = {
        "inputs": {
            "map_gt": str(map_gt_dir),
            "map_eval": str(map_eval_dir),
            "output_dir": str(output_dir),
        },
        "parameters": {
            "num_samples": args.num_samples,
            "trim_ratio": args.trim_ratio,
            "top_k": args.top_k,
            "min_inliers": args.min_inliers,
            "good_error_threshold_m": args.good_error_threshold_m,
            "bad_error_threshold_m": args.bad_error_threshold_m,
            "ransac_threshold_m": args.ransac_threshold_m,
            "alignment_mode": args.alignment_mode,
        },
        "transform_map_eval_to_gt": transform.tolist(),
        "transform_fit": transform_info,
        "sample_candidates": len(rows),
        "label_counts": label_counts,
        "pose_error_m": _summary(pose_errors),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "top1_transform_pairs.json", top1_rows)
    with (output_dir / "samples_index.jsonl").open("w") as f:
        for row in rows:
            slim = {
                key: row[key]
                for key in [
                    "sample_index",
                    "query_timestamp_ns",
                    "reference_timestamp_ns",
                    "candidate_rank",
                    "retrieval_similarity",
                    "pnp_success",
                    "pnp_inlier_count",
                    "pnp_inlier_ratio",
                    "match_count",
                    "valid_landmark_count",
                    "pose_error_m",
                    "candidate_label",
                    "sample_dir",
                ]
            }
            f.write(json.dumps(slim) + "\n")
    np.save(output_dir / "T_map_eval_to_gt.npy", transform)
    _write_html(output_dir, rows, summary)

    print(f"Dataset export complete: {output_dir}")
    print(f"HTML report: {output_dir / 'index.html'}")
    print(json.dumps({"sample_candidates": len(rows), "label_counts": label_counts}, indent=2))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a keypoint-level relocalization confidence dataset from two TinyNav maps."
    )
    parser.add_argument("--map-gt", required=True, help="Reference/retrieval TinyNav map directory")
    parser.add_argument("--map-eval", required=True, help="Query/eval TinyNav map directory")
    parser.add_argument("--output-root", default="output", help="Parent directory for timestamped output")
    parser.add_argument("--output-dir", help="Exact output directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of eval keyframes to sample; <=0 means all")
    parser.add_argument("--trim-ratio", type=float, default=0.05, help="Trim this fraction from map_eval start/end")
    parser.add_argument("--timestamps-file", help="Optional text file of eval timestamps in ns")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieval candidates per query")
    parser.add_argument("--min-inliers", type=int, default=50, help="Minimum PnP inliers for PnP success")
    parser.add_argument("--good-error-threshold-m", type=float, default=0.50, help="Pose error threshold for good label")
    parser.add_argument("--bad-error-threshold-m", type=float, default=1.50, help="Pose error threshold for bad label")
    parser.add_argument("--ransac-threshold-m", type=float, default=0.20, help="Transform fitting inlier threshold")
    parser.add_argument("--ransac-iterations", type=int, default=1000, help="Transform fitting RANSAC iterations")
    parser.add_argument("--alignment-mode", choices=["se2_z", "se3"], default="se2_z")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-match-lines", type=int, default=120, help="Maximum PnP inlier lines drawn in match_vis")
    parser.add_argument(
        "--include-arrays-in-sample-json",
        action="store_true",
        help="Embed keypoint arrays in sample.json instead of separate keypoints.json",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
