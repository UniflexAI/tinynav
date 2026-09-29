#!/usr/bin/env python3
"""Update a TinyNav map with a newly recorded bag.

The source bag is first built into a temporary TinyNav map. Its keyframes are then
retrieved against the destination map using the destination map's VLAD vocabulary
so descriptors are comparable without training a new vocabulary. A robust
SE(2)+z transform is fitted from the retrieved keyframe pose pairs. The original
destination map is never modified: a new update directory is written only when
the fitted transform passes conservative quality gates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shelve
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SHELVE_STORES = [
    "features",
    "embeddings",
    "semantic_embeddings",
    "patch_tokens",
    "depths",
    "vlad_descriptors",
]
VIO_IMAGE_TOPIC = "/camera/camera/vio_image"


def _load_poses(map_path: Path) -> dict[int, np.ndarray]:
    path = map_path / "poses.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing poses file: {path}")
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in np.load(path, allow_pickle=True).item().items()}


def _require_map(path: Path) -> None:
    required = ["poses.npy", "intrinsics.npy", "baseline.npy", "occupancy_grid.npy", "occupancy_meta.npy"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is not a complete TinyNav map, missing: {missing}")


def _default_output_path(dst: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dst.parent / f"{dst.name}_update_{stamp}"


def _bag_topics(bag_path: Path) -> set[str]:
    import rosbag2_py

    info = rosbag2_py.Info()
    metadata = info.read_metadata(str(bag_path), "")
    topics: set[str] = set()
    for topic in metadata.topics_with_message_count:
        if hasattr(topic, "name"):
            topics.add(topic.name)
        else:
            topics.add(topic.topic_metadata.name)
    return topics


def _source_node_for_bag(bag_path: Path, repo_root: Path, verbose_timer: bool, log_dir: Path) -> tuple[str, list[str]]:
    topics = _bag_topics(bag_path)
    if VIO_IMAGE_TOPIC in topics:
        return "looper_bridge", [sys.executable, str(repo_root / "tool/looper_bridge_node.py")]

    cmd = [
        sys.executable,
        str(repo_root / "tinynav/core/perception_node.py"),
        "--log_file",
        str(log_dir / "perception.log"),
    ]
    if verbose_timer:
        cmd.append("--verbose_timer")
    return "perception", cmd


def _run_build_map(
    src_bag: Path,
    src_map: Path,
    play_rate: float,
    global_frames_ratio: float,
    verbose_timer: bool,
) -> None:
    from launch import LaunchDescription, LaunchService
    from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown

    src_map.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    source_name, source_cmd = _source_node_for_bag(src_bag, repo_root, verbose_timer, src_map)
    build_cmd = [
        sys.executable,
        str(repo_root / "tinynav/core/build_map_node.py"),
        "--bag_file",
        str(src_bag),
        "--map_save_path",
        str(src_map),
        "--play_rate",
        str(play_rate),
        "--global_frames_ratio",
        str(global_frames_ratio),
    ]
    if not verbose_timer:
        build_cmd.append("--no_verbose_timer")

    print(f"building source map from bag using {source_name}...")
    source = ExecuteProcess(
        cmd=source_cmd,
        name=f"update_map_{source_name}",
        output="screen",
        cwd=str(repo_root),
        additional_env=env,
    )
    mapping = ExecuteProcess(
        cmd=build_cmd,
        name="update_map_build_map",
        output="screen",
        cwd=str(repo_root),
        additional_env=env,
    )
    on_mapping_exit = RegisterEventHandler(
        OnProcessExit(target_action=mapping, on_exit=[EmitEvent(event=Shutdown())])
    )
    service = LaunchService()
    service.include_launch_description(LaunchDescription([source, mapping, on_mapping_exit]))
    rc = service.run()
    if rc not in (0, None):
        raise RuntimeError(f"source map build launch failed with exit code {rc}")
    _require_map(src_map)


def _topk_from_chunk(similarities: np.ndarray, timestamps: list[int], top_k: int) -> list[tuple[int, float]]:
    if top_k >= len(similarities):
        indices = np.argsort(-similarities)
    else:
        rough = np.argpartition(-similarities, top_k - 1)[:top_k]
        indices = rough[np.argsort(-similarities[rough])]
    return [(int(timestamps[int(i)]), float(similarities[int(i)])) for i in indices]


def _retrieve_src_against_dst(
    *,
    dst_map: Path,
    src_map: Path,
    top_k: int,
    every_n: int,
    max_queries: int,
    dst_chunk_size: int,
) -> list[dict[str, Any]]:
    from tinynav.core.build_map_node import TinyNavDB
    from tinynav.core.vlad import compute_vlad

    dst_poses = _load_poses(dst_map)
    src_poses = _load_poses(src_map)
    dst_timestamps = sorted(dst_poses)
    src_timestamps = sorted(src_poses)
    if every_n > 1:
        src_timestamps = src_timestamps[::every_n]
    if max_queries > 0:
        src_timestamps = src_timestamps[:max_queries]

    dst_db = TinyNavDB(str(dst_map), is_scratch=False)
    src_db = TinyNavDB(str(src_map), is_scratch=False)
    try:
        dst_centres = np.asarray(dst_db.metadata["vlad_centres"], dtype=np.float32)
        rows: list[dict[str, Any]] = []
        for query_index, src_ts in enumerate(src_timestamps, start=1):
            query_vlad = compute_vlad(src_db.patch_tokens[int(src_ts)], dst_centres)
            best: list[tuple[int, float]] = []
            for start in range(0, len(dst_timestamps), dst_chunk_size):
                chunk_ts = dst_timestamps[start : start + dst_chunk_size]
                desc = np.stack([dst_db.vlad_descriptors[int(t)] for t in chunk_ts]).astype(np.float32)
                chunk_best = _topk_from_chunk(desc @ query_vlad, chunk_ts, min(top_k, len(chunk_ts)))
                best.extend(chunk_best)
                best.sort(key=lambda item: item[1], reverse=True)
                best = best[:top_k]
            rows.append(
                {
                    "src_timestamp_ns": int(src_ts),
                    "src_xyz": src_poses[int(src_ts)][:3, 3].tolist(),
                    "retrieved": [
                        {
                            "dst_timestamp_ns": int(dst_ts),
                            "similarity": float(sim),
                            "dst_xyz": dst_poses[int(dst_ts)][:3, 3].tolist(),
                        }
                        for dst_ts, sim in best
                    ],
                }
            )
            if query_index % 25 == 0:
                print(f"retrieved {query_index}/{len(src_timestamps)} source keyframes")
        return rows
    finally:
        dst_db.close()
        src_db.close()


def _estimate_se2_z(points_src: np.ndarray, points_dst: np.ndarray) -> np.ndarray:
    if len(points_src) < 2:
        raise ValueError("need at least two point pairs")
    src_xy = points_src[:, :2]
    dst_xy = points_dst[:, :2]
    src_mean = src_xy.mean(axis=0)
    dst_mean = dst_xy.mean(axis=0)
    h = (src_xy - src_mean).T @ (dst_xy - dst_mean)
    u, _, vt = np.linalg.svd(h)
    rot_xy = vt.T @ u.T
    if np.linalg.det(rot_xy) < 0:
        vt[-1, :] *= -1
        rot_xy = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:2, :2] = rot_xy
    transform[:2, 3] = dst_mean - rot_xy @ src_mean
    transform[2, 3] = float(np.median(points_dst[:, 2] - points_src[:, 2]))
    return transform


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (transform @ np.c_[points, np.ones(len(points))].T).T[:, :3]


def _ransac_fit_transform(
    rows: list[dict[str, Any]],
    *,
    min_similarity: float,
    inlier_threshold_m: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    pairs = [
        row
        for row in rows
        if row["retrieved"] and float(row["retrieved"][0]["similarity"]) >= min_similarity
    ]
    if len(pairs) < 2:
        raise RuntimeError(f"not enough retrieval pairs above similarity threshold: {len(pairs)}")

    src = np.asarray([row["src_xyz"] for row in pairs], dtype=np.float64)
    dst = np.asarray([row["retrieved"][0]["dst_xyz"] for row in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(len(pairs), dtype=bool)
    best_transform = np.eye(4, dtype=np.float64)

    for _ in range(max(1, iterations)):
        sample = rng.choice(len(pairs), size=2, replace=False)
        if np.linalg.norm(src[sample[0], :2] - src[sample[1], :2]) < 1e-3:
            continue
        candidate = _estimate_se2_z(src[sample], dst[sample])
        residuals = np.linalg.norm(_transform_points(candidate, src) - dst, axis=1)
        mask = residuals <= inlier_threshold_m
        if int(mask.sum()) > int(best_mask.sum()):
            best_mask = mask
            best_transform = candidate

    if int(best_mask.sum()) >= 2:
        best_transform = _estimate_se2_z(src[best_mask], dst[best_mask])
    residuals = np.linalg.norm(_transform_points(best_transform, src) - dst, axis=1)
    inliers = residuals <= inlier_threshold_m
    inlier_residuals = residuals[inliers]
    yaw_deg = math.degrees(math.atan2(best_transform[1, 0], best_transform[0, 0]))

    return {
        "T_dst_src": best_transform.tolist(),
        "yaw_deg": yaw_deg,
        "candidate_pairs": len(pairs),
        "inlier_count": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()) if len(inliers) else 0.0,
        "median_residual_m": float(np.median(inlier_residuals)) if len(inlier_residuals) else None,
        "p90_residual_m": float(np.percentile(inlier_residuals, 90)) if len(inlier_residuals) else None,
        "max_residual_m": float(np.max(inlier_residuals)) if len(inlier_residuals) else None,
        "all_residual_m": residuals.tolist(),
        "inlier_src_timestamps": [
            int(row["src_timestamp_ns"]) for row, ok in zip(pairs, inliers) if bool(ok)
        ],
        "inlier_dst_timestamps": [
            int(row["retrieved"][0]["dst_timestamp_ns"]) for row, ok in zip(pairs, inliers) if bool(ok)
        ],
    }


def _trajectory_span_m(poses: dict[int, np.ndarray], timestamps: list[int]) -> float:
    if len(timestamps) < 2:
        return 0.0
    pts = np.asarray([poses[int(t)][:3, 3] for t in timestamps], dtype=np.float64)
    return float(np.linalg.norm(np.max(pts[:, :2], axis=0) - np.min(pts[:, :2], axis=0)))


def _passes_quality_gate(args: argparse.Namespace, src_map: Path, fit: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    src_poses = _load_poses(src_map)
    inlier_src = [int(t) for t in fit["inlier_src_timestamps"]]
    span_m = _trajectory_span_m(src_poses, inlier_src)
    fit["inlier_src_xy_span_m"] = span_m
    median = fit["median_residual_m"]
    if fit["inlier_count"] < args.min_inliers:
        reasons.append(f"inlier_count {fit['inlier_count']} < {args.min_inliers}")
    if fit["candidate_pairs"] < args.min_pairs:
        reasons.append(f"candidate_pairs {fit['candidate_pairs']} < {args.min_pairs}")
    if fit["inlier_ratio"] < args.min_inlier_ratio:
        reasons.append(f"inlier_ratio {fit['inlier_ratio']:.3f} < {args.min_inlier_ratio}")
    if median is None or median > args.max_median_residual_m:
        reasons.append(f"median_residual_m {median} > {args.max_median_residual_m}")
    if span_m < args.min_inlier_span_m:
        reasons.append(f"inlier source xy span {span_m:.3f}m < {args.min_inlier_span_m}m")
    return len(reasons) == 0, reasons


def _copy_shelve_entries(src_map: Path, output_map: Path, timestamp_map: dict[int, int], dst_centres: np.ndarray) -> None:
    from tinynav.core.vlad import compute_vlad

    for store in SHELVE_STORES:
        src_path = src_map / f"{store}.db"
        if not src_path.exists():
            continue
        src_db = shelve.open(str(src_map / store), flag="r")
        out_db = shelve.open(str(output_map / store))
        try:
            for old_ts, new_ts in timestamp_map.items():
                key = str(int(old_ts))
                if key not in src_db:
                    continue
                if store == "vlad_descriptors":
                    continue
                out_db[str(int(new_ts))] = src_db[key]
        finally:
            src_db.close()
            out_db.close()

    src_patch_db = shelve.open(str(src_map / "patch_tokens"), flag="r")
    out_vlad_db = shelve.open(str(output_map / "vlad_descriptors"))
    try:
        for old_ts, new_ts in timestamp_map.items():
            key = str(int(old_ts))
            if key in src_patch_db:
                out_vlad_db[str(int(new_ts))] = compute_vlad(src_patch_db[key], dst_centres)
    finally:
        src_patch_db.close()
        out_vlad_db.close()


def _merge_maps(args: argparse.Namespace, dst_map: Path, src_map: Path, output_map: Path, fit: dict[str, Any]) -> dict[str, Any]:
    import cv2

    from tinynav.core.build_map_node import TinyNavDB, generate_occupancy_map

    if output_map.exists():
        raise FileExistsError(f"output map already exists: {output_map}")
    print(f"copying destination map to {output_map}...")
    shutil.copytree(dst_map, output_map)

    dst_poses = _load_poses(dst_map)
    src_poses = _load_poses(src_map)
    transform = np.asarray(fit["T_dst_src"], dtype=np.float64)
    timestamp_offset = max(dst_poses) - min(src_poses) + 1_000_000_000
    timestamp_map = {int(ts): int(ts + timestamp_offset) for ts in src_poses}
    merged_poses = dict(dst_poses)
    for old_ts, pose in src_poses.items():
        merged_poses[timestamp_map[int(old_ts)]] = transform @ pose
    np.save(output_map / "poses.npy", merged_poses, allow_pickle=True)

    dst_db = TinyNavDB(str(dst_map), is_scratch=False)
    try:
        dst_centres = np.asarray(dst_db.metadata["vlad_centres"], dtype=np.float32)
    finally:
        dst_db.close()
    _copy_shelve_entries(src_map, output_map, timestamp_map, dst_centres)

    print("regenerating occupancy and sdf for the updated map...")
    K = np.load(output_map / "intrinsics.npy")
    baseline = np.load(output_map / "baseline.npy")
    db = TinyNavDB(str(output_map), is_scratch=False)
    try:
        occupancy_grid, occupancy_origin, occupancy_2d_image, sdf_map = generate_occupancy_map(
            merged_poses,
            db,
            K,
            baseline,
            resolution=args.occupancy_resolution,
            step=args.occupancy_step,
        )
    finally:
        db.close()
    occupancy_meta = np.array(
        [occupancy_origin[0], occupancy_origin[1], occupancy_origin[2], args.occupancy_resolution],
        dtype=np.float32,
    )
    np.save(output_map / "occupancy_grid.npy", occupancy_grid)
    np.save(output_map / "occupancy_meta.npy", occupancy_meta)
    np.save(output_map / "sdf_map.npy", sdf_map)
    cv2.imwrite(str(output_map / "occupancy_2d_image.png"), occupancy_2d_image)

    return {
        "output_map": str(output_map),
        "dst_keyframes": len(dst_poses),
        "src_keyframes_added": len(src_poses),
        "merged_keyframes": len(merged_poses),
        "timestamp_offset_ns": int(timestamp_offset),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    print(f"wrote report: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dst_map = Path(args.dst).resolve()
    src_bag = Path(args.src).resolve()
    output_map = Path(args.output).resolve() if args.output else _default_output_path(dst_map)
    _require_map(dst_map)
    if not src_bag.exists():
        raise FileNotFoundError(src_bag)

    temp_ctx: tempfile.TemporaryDirectory | None = None
    if args.src_map:
        src_map = Path(args.src_map).resolve()
        _require_map(src_map)
    else:
        if args.work_dir:
            src_map = Path(args.work_dir).resolve() / "src_map"
            if src_map.exists() and not args.reuse_src_map:
                shutil.rmtree(src_map)
        else:
            if args.keep_tmp:
                src_map = Path(tempfile.mkdtemp(prefix="tinynav_update_map_")) / "src_map"
            else:
                temp_ctx = tempfile.TemporaryDirectory(prefix="tinynav_update_map_")
                src_map = Path(temp_ctx.name) / "src_map"
        if not args.reuse_src_map or not (src_map / "poses.npy").exists():
            _run_build_map(
                src_bag,
                src_map,
                args.play_rate,
                args.global_frames_ratio,
                args.verbose_timer,
            )
        else:
            _require_map(src_map)

    rows = _retrieve_src_against_dst(
        dst_map=dst_map,
        src_map=src_map,
        top_k=args.top_k,
        every_n=args.every_n,
        max_queries=args.max_queries,
        dst_chunk_size=args.dst_chunk_size,
    )
    fit = _ransac_fit_transform(
        rows,
        min_similarity=args.min_similarity,
        inlier_threshold_m=args.ransac_threshold_m,
        iterations=args.ransac_iterations,
        seed=args.seed,
    )
    ok, reject_reasons = _passes_quality_gate(args, src_map, fit)
    report = {
        "type": "tinynav_map_update",
        "dst": str(dst_map),
        "src": str(src_bag),
        "src_map": str(src_map),
        "output": str(output_map),
        "dry_run": bool(args.dry_run),
        "quality_ok": bool(ok),
        "reject_reasons": reject_reasons,
        "fit": fit,
        "retrieval": {
            "query_count": len(rows),
            "top_k": args.top_k,
            "min_similarity": args.min_similarity,
            "every_n": args.every_n,
            "max_queries": args.max_queries,
            "dst_chunk_size": args.dst_chunk_size,
        },
        "rows": rows if args.save_rows else rows[: min(len(rows), 200)],
    }

    if ok and not args.dry_run:
        report["merge"] = _merge_maps(args, dst_map, src_map, output_map, fit)
    elif not ok:
        print("refusing to merge: " + "; ".join(reject_reasons))
    else:
        print("dry-run requested; not writing updated map")

    report_path = Path(args.report).resolve() if args.report else output_map.parent / f"{output_map.name}_update_report.json"
    _write_report(report_path, report)
    if temp_ctx is None and args.keep_tmp and not args.src_map and not args.work_dir:
        print(f"kept temporary source map at {src_map}")
    if temp_ctx is not None:
        temp_ctx.cleanup()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a source bag map and conservatively merge it into an existing TinyNav map.")
    parser.add_argument("--dst", required=True, help="Destination map directory to update. It is never modified.")
    parser.add_argument("--src", required=True, help="Source rosbag directory used to build the update map.")
    parser.add_argument("--output", help="Updated map output directory. Defaults to <dst>_update_<timestamp>.")
    parser.add_argument("--report", help="JSON report path. Defaults next to the output map.")
    parser.add_argument("--src-map", help="Use an already-built source map instead of building --src.")
    parser.add_argument("--work-dir", help="Directory for the temporary built source map.")
    parser.add_argument("--reuse-src-map", action="store_true", help="Reuse work-dir/src_map if it already exists.")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep the auto-created temporary source map.")
    parser.add_argument("--dry-run", action="store_true", help="Only build/retrieve/fit/report; do not merge.")
    parser.add_argument("--save-rows", action="store_true", help="Write all per-query retrieval rows to the report.")
    parser.add_argument("--play-rate", type=float, default=1.0)
    parser.add_argument("--global-frames-ratio", type=float, default=1.1)
    parser.add_argument("--no-verbose-timer", dest="verbose_timer", action="store_false", default=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--dst-chunk-size", type=int, default=512)
    parser.add_argument("--min-similarity", type=float, default=0.20)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.50)
    parser.add_argument("--ransac-iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.50)
    parser.add_argument("--max-median-residual-m", type=float, default=0.30)
    parser.add_argument("--min-inlier-span-m", type=float, default=3.0)
    parser.add_argument("--occupancy-resolution", type=float, default=0.1)
    parser.add_argument("--occupancy-step", type=int, default=10)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({k: summary[k] for k in ["quality_ok", "reject_reasons", "fit"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
