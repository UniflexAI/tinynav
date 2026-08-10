#!/usr/bin/env python3
"""
Cross-map keyframe relocalization benchmark.

This tool evaluates keyframe relocalization poses against an independently
built eval map trajectory aligned into the GT map frame:

    relocalized_pose_in_map_gt  vs  T_map_eval_to_gt * map_eval_keyframe_pose

Inputs:
  - GT source: `--map-gt` (existing map) or `--bag-gt` (build from bag).
  - Eval source: `--map-eval` (existing map) or `--bag-eval` (build from bag
    and replay against map_gt for localization).

When `--map-eval` is used, map build and localization are skipped; the tool
expects existing localization results in the work directory.

The original benchmark_mapping.py is intentionally left untouched. This script
shares only small utility concepts and produces a standalone HTML report.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import rosbag2_py
from launch import LaunchDescription, LaunchService
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown

from benchmark_mapping import find_closest_pose


PoseDict = Dict[int, np.ndarray]
VIO_IMAGE_TOPIC = "/camera/camera/vio_image"


def _load_pose_dict(path: Path) -> PoseDict:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True).item()
    return {int(k): np.asarray(v, dtype=float) for k, v in data.items()}


def _bag_topics(bag_path: str) -> set[str]:
    info = rosbag2_py.Info()
    metadata = info.read_metadata(bag_path, "")
    topics = set()
    for topic in metadata.topics_with_message_count:
        # Humble exposes TopicInformation.topic_metadata.name; some newer
        # rosbag2_py builds expose .name directly.
        if hasattr(topic, "name"):
            topics.add(topic.name)
        else:
            topics.add(topic.topic_metadata.name)
    return topics


def _source_node_for_bag(bag_path: str) -> tuple[str, list[str]]:
    topics = _bag_topics(bag_path)
    if VIO_IMAGE_TOPIC in topics:
        return "looper_bridge", ["python3", "/tinynav/tool/looper_bridge_node.py"]
    return "perception", ["python3", "/tinynav/tinynav/core/perception_node.py"]


def _generate_mapping_launch(
    *,
    bag_path: str,
    map_dir: Path,
    rate: float,
    verbose_timer: bool,
) -> LaunchDescription:
    source_name, source_cmd = _source_node_for_bag(bag_path)
    if source_name == "perception":
        source_cmd += ["--log_file", str(map_dir / "perception.log")]
        if verbose_timer:
            source_cmd.append("--verbose_timer")

    build_cmd = [
        "python3",
        "/tinynav/tinynav/core/build_map_node.py",
        "--map_save_path",
        str(map_dir),
        "--bag_file",
        str(bag_path),
        "--play_rate",
        str(rate),
    ]
    if not verbose_timer:
        build_cmd.append("--no_verbose_timer")

    source = ExecuteProcess(cmd=source_cmd, name=f"benchmark_{source_name}", output="screen")
    mapping = ExecuteProcess(cmd=build_cmd, name="benchmark_build_map", output="screen")
    on_mapping_exit = RegisterEventHandler(
        OnProcessExit(target_action=mapping, on_exit=[EmitEvent(event=Shutdown())])
    )
    return LaunchDescription([source, mapping, on_mapping_exit])


def _generate_localization_launch(
    *,
    bag_path: str,
    map_gt_dir: Path,
    localization_dir: Path,
    rate: float,
    timeout: float,
    verbose_timer: bool,
) -> LaunchDescription:
    source_name, source_cmd = _source_node_for_bag(bag_path)
    if source_name == "perception":
        source_cmd += ["--log_file", str(localization_dir / "perception.log")]
        if verbose_timer:
            source_cmd.append("--verbose_timer")

    localization_cmd = [
        "python3",
        "/tinynav/tinynav/core/map_node.py",
        "--tinynav_db_path",
        str(localization_dir),
        "--tinynav_map_path",
        str(map_gt_dir),
    ]
    if not verbose_timer:
        localization_cmd.append("--no_verbose_timer")

    bag_play = ExecuteProcess(
        cmd=["ros2", "bag", "play", str(bag_path), "--rate", str(rate), "--clock"],
        name="benchmark_bag_eval_play",
        output="screen",
    )
    source = ExecuteProcess(cmd=source_cmd, name=f"benchmark_{source_name}", output="screen")
    localization = ExecuteProcess(
        cmd=localization_cmd,
        name="benchmark_map_gt_localization",
        output="screen",
    )
    coordinator = ExecuteProcess(
        cmd=[
            "python3",
            "/tinynav/tool/benchmark/data_saving_coordinator.py",
            str(timeout),
        ],
        name="benchmark_localization_coordinator",
        output="screen",
    )
    on_bag_exit = RegisterEventHandler(
        OnProcessExit(target_action=bag_play, on_exit=[coordinator])
    )
    on_coordinator_exit = RegisterEventHandler(
        OnProcessExit(target_action=coordinator, on_exit=[EmitEvent(event=Shutdown())])
    )
    return LaunchDescription([source, localization, bag_play, on_bag_exit, on_coordinator_exit])


def _run_launch(ld: LaunchDescription):
    service = LaunchService()
    service.include_launch_description(ld)
    service.run()


def _build_map_from_bag(
    *,
    bag_path: str,
    map_dir: Path,
    rate: float,
    verbose_timer: bool,
):
    map_dir.mkdir(parents=True, exist_ok=True)
    source_name, _ = _source_node_for_bag(bag_path)
    print(f"Building {map_dir} from {bag_path} using {source_name}")
    _run_launch(
        _generate_mapping_launch(
            bag_path=bag_path,
            map_dir=map_dir,
            rate=rate,
            verbose_timer=verbose_timer,
        )
    )
    _require_file(map_dir / "poses.npy", "map build")


def _localize_eval_bag_in_gt_map(
    *,
    bag_eval: str,
    map_gt_dir: Path,
    localization_dir: Path,
    rate: float,
    timeout: float,
    verbose_timer: bool,
):
    localization_dir.mkdir(parents=True, exist_ok=True)
    source_name, _ = _source_node_for_bag(bag_eval)
    print(f"Replaying eval bag against GT map using {source_name}")
    _run_launch(
        _generate_localization_launch(
            bag_path=bag_eval,
            map_gt_dir=map_gt_dir,
            localization_dir=localization_dir,
            rate=rate,
            timeout=timeout,
            verbose_timer=verbose_timer,
        )
    )
    _require_file(localization_dir / "relocalization_poses.npy", "localization")


def _require_file(path: Path, step_name: str):
    if not path.exists():
        raise RuntimeError(f"{step_name} did not produce required file: {path}")


def _sample_timestamps_from_map(
    map_dir: Path,
    num_samples: int,
    trim_ratio: float,
) -> np.ndarray:
    poses = _load_pose_dict(map_dir / "poses.npy")
    timestamps = np.asarray(sorted(poses), dtype=np.int64)
    if len(timestamps) == 0:
        raise RuntimeError(f"No poses found in map: {map_dir}")

    trim_count = int(len(timestamps) * trim_ratio)
    if trim_count * 2 < len(timestamps):
        timestamps = timestamps[trim_count : len(timestamps) - trim_count]
    if num_samples <= 0 or num_samples >= len(timestamps):
        return timestamps

    indices = np.linspace(0, len(timestamps) - 1, num_samples)
    return timestamps[np.round(indices).astype(np.int64)]


def _query_eval_reference_and_fusion(
    *,
    timestamps: np.ndarray,
    map_eval_dir: Path,
    localization_dir: Path,
    max_anchor_dt_ns: int,
) -> Tuple[PoseDict, PoseDict, dict]:
    map_eval_keyframe_poses = _load_pose_dict(map_eval_dir / "poses.npy")
    fusion_anchor_poses = _load_pose_dict(localization_dir / "relocalization_poses.npy")

    map_eval_reference_poses: PoseDict = {}
    fusion_poses: PoseDict = {}
    skipped = {
        "map_eval_reference_missing": 0,
        "fusion_missing": 0,
        "map_eval_pose_source": "keyframe_pose",
        "fusion_pose_source": "relocalization_pose",
    }
    for timestamp in timestamps:
        ts = int(timestamp)
        anchor_ts, reference_pose = find_closest_pose(ts, map_eval_keyframe_poses)
        if anchor_ts is None or abs(ts - int(anchor_ts)) > max_anchor_dt_ns:
            reference_pose = None

        anchor_ts, fusion_pose = find_closest_pose(ts, fusion_anchor_poses)
        if anchor_ts is None or abs(ts - int(anchor_ts)) > max_anchor_dt_ns:
            fusion_pose = None
        if reference_pose is None:
            skipped["map_eval_reference_missing"] += 1
        else:
            map_eval_reference_poses[ts] = reference_pose
        if fusion_pose is None:
            skipped["fusion_missing"] += 1
        else:
            fusion_poses[ts] = fusion_pose

    return map_eval_reference_poses, fusion_poses, skipped


def _estimate_rigid_transform(points_src: np.ndarray, points_dst: np.ndarray) -> np.ndarray:
    if len(points_src) != len(points_dst) or len(points_src) < 3:
        raise ValueError("Need at least 3 paired points")
    centroid_src = np.mean(points_src, axis=0)
    centroid_dst = np.mean(points_dst, axis=0)
    src_centered = points_src - centroid_src
    dst_centered = points_dst - centroid_dst
    u, _, vt = np.linalg.svd(src_centered.T @ dst_centered)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = centroid_dst - rot @ centroid_src
    return transform


def _estimate_se2_z_transform(points_src: np.ndarray, points_dst: np.ndarray) -> np.ndarray:
    if len(points_src) != len(points_dst) or len(points_src) < 2:
        raise ValueError("Need at least 2 paired points")

    src_xy = points_src[:, :2]
    dst_xy = points_dst[:, :2]
    centroid_src = np.mean(src_xy, axis=0)
    centroid_dst = np.mean(dst_xy, axis=0)
    src_centered = src_xy - centroid_src
    dst_centered = dst_xy - centroid_dst
    u, _, vt = np.linalg.svd(src_centered.T @ dst_centered)
    rot_2d = vt.T @ u.T
    if np.linalg.det(rot_2d) < 0:
        vt[-1, :] *= -1
        rot_2d = vt.T @ u.T

    transform = np.eye(4)
    transform[:2, :2] = rot_2d
    transform[:2, 3] = centroid_dst - rot_2d @ centroid_src
    transformed_z = points_src[:, 2]
    transform[2, 3] = float(np.median(points_dst[:, 2] - transformed_z))
    return transform


def _ransac_transform(
    *,
    source_poses: PoseDict,
    target_poses: PoseDict,
    inlier_threshold_m: float,
    iterations: int,
    seed: int,
    alignment_mode: str,
) -> Tuple[np.ndarray, list[int], dict]:
    timestamps = sorted(set(source_poses) & set(target_poses))
    if len(timestamps) < 3:
        raise RuntimeError("Need at least 3 common timestamps to estimate transform")

    src = np.array([source_poses[t][:3, 3] for t in timestamps], dtype=float)
    dst = np.array([target_poses[t][:3, 3] for t in timestamps], dtype=float)
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(len(timestamps), dtype=bool)
    best_transform = np.eye(4)
    estimator = _estimate_se2_z_transform if alignment_mode == "se2_z" else _estimate_rigid_transform
    sample_size = 2 if alignment_mode == "se2_z" else 3

    for _ in range(max(iterations, 1)):
        sample_idx = rng.choice(len(timestamps), size=sample_size, replace=False)
        candidate = estimator(src[sample_idx], dst[sample_idx])
        transformed = (candidate @ np.c_[src, np.ones(len(src))].T).T[:, :3]
        distances = np.linalg.norm(transformed - dst, axis=1)
        mask = distances <= inlier_threshold_m
        if int(mask.sum()) > int(best_mask.sum()):
            best_mask = mask
            best_transform = candidate

    if best_mask.sum() >= sample_size:
        best_transform = estimator(src[best_mask], dst[best_mask])
        transformed = (best_transform @ np.c_[src, np.ones(len(src))].T).T[:, :3]
        distances = np.linalg.norm(transformed - dst, axis=1)
        best_mask = distances <= inlier_threshold_m

    inlier_timestamps = [timestamps[i] for i, ok in enumerate(best_mask) if ok]
    return best_transform, inlier_timestamps, {
        "candidate_pairs": len(timestamps),
        "inlier_count": len(inlier_timestamps),
        "inlier_ratio": len(inlier_timestamps) / max(len(timestamps), 1),
        "inlier_threshold_m": inlier_threshold_m,
        "ransac_iterations": iterations,
        "alignment_mode": alignment_mode,
    }


def _rotation_error_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rot_err = rot_a.T @ rot_b
    value = np.clip((np.trace(rot_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def _compute_errors(
    *,
    map_eval_poses: PoseDict,
    fusion_poses: PoseDict,
    transform_map_eval_to_gt: np.ndarray,
) -> list[dict]:
    rows = []
    for timestamp in sorted(set(map_eval_poses) & set(fusion_poses)):
        map_eval_in_gt = transform_map_eval_to_gt @ map_eval_poses[timestamp]
        fusion = fusion_poses[timestamp]
        delta_xyz = map_eval_in_gt[:3, 3] - fusion[:3, 3]
        rows.append({
            "timestamp_ns": int(timestamp),
            "time_s": float(timestamp) / 1e9,
            "translation_error_m": float(np.linalg.norm(delta_xyz)),
            "rotation_error_deg": _rotation_error_deg(map_eval_in_gt[:3, :3], fusion[:3, :3]),
            "map_eval_in_gt_xyz": map_eval_in_gt[:3, 3].tolist(),
            "fusion_xyz": fusion[:3, 3].tolist(),
            "eval_minus_gt_xyz_m": delta_xyz.tolist(),
            "abs_eval_minus_gt_xyz_m": np.abs(delta_xyz).tolist(),
        })
    return rows


def _summary(values: Iterable[float]) -> dict:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None, "rmse": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
    }


def _threshold_stats(errors: list[dict], thresholds_m: list[float]) -> dict:
    values = np.array([row["translation_error_m"] for row in errors], dtype=float)
    total = int(values.size)
    result = {}
    for threshold in thresholds_m:
        count = int(np.sum(values <= threshold)) if total else 0
        result[f"{threshold:.2f}m"] = {"count": count, "ratio": count / total if total else 0.0}
    return result


def _xyz_summary(errors: list[dict], field: str) -> dict:
    values = np.array([row[field] for row in errors], dtype=float)
    if values.size == 0:
        return {"dx": _summary([]), "dy": _summary([]), "dz": _summary([])}
    return {
        "dx": _summary(values[:, 0]),
        "dy": _summary(values[:, 1]),
        "dz": _summary(values[:, 2]),
    }


def _plot_trajectory(errors: list[dict], output_path: Path):
    ref_xyz = np.array([row["map_eval_in_gt_xyz"] for row in errors], dtype=float)
    fusion_xyz = np.array([row["fusion_xyz"] for row in errors], dtype=float)
    plt.figure(figsize=(12, 10))
    if len(ref_xyz):
        plt.plot(ref_xyz[:, 0], ref_xyz[:, 1], "-", label="map_eval * T reference", linewidth=2)
        plt.plot(fusion_xyz[:, 0], fusion_xyz[:, 1], "-", label="relocalization pose in map_gt", linewidth=2)
        plt.scatter(ref_xyz[:, 0], ref_xyz[:, 1], s=10, alpha=0.45)
        plt.scatter(fusion_xyz[:, 0], fusion_xyz[:, 1], s=10, alpha=0.45)
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectory comparison in map_gt frame (XY projection)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _plot_trajectory_projection(
    errors: list[dict],
    output_path: Path,
    *,
    axes: tuple[int, int],
    labels: tuple[str, str],
    title: str,
):
    ref_xyz = np.array([row["map_eval_in_gt_xyz"] for row in errors], dtype=float)
    fusion_xyz = np.array([row["fusion_xyz"] for row in errors], dtype=float)
    i, j = axes
    fig, ax = plt.subplots(figsize=(13, 8))
    if len(ref_xyz):
        ax.plot(ref_xyz[:, i], ref_xyz[:, j], "-", label="map_eval * T reference", linewidth=2.2, color="#2563eb")
        ax.plot(fusion_xyz[:, i], fusion_xyz[:, j], "-", label="relocalization pose in map_gt", linewidth=2.2, color="#f97316")
        ax.scatter(ref_xyz[:, i], ref_xyz[:, j], s=18, alpha=0.4, color="#2563eb")
        ax.scatter(fusion_xyz[:, i], fusion_xyz[:, j], s=18, alpha=0.4, color="#f97316")
        ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_error_curve(errors: list[dict], output_path: Path):
    t0 = errors[0]["time_s"] if errors else 0.0
    times = np.array([row["time_s"] - t0 for row in errors], dtype=float)
    trans_errors = np.array([row["translation_error_m"] for row in errors], dtype=float)
    rot_errors = np.array([row["rotation_error_deg"] for row in errors], dtype=float)
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(times, trans_errors, color="#2563eb", label="translation error [m]")
    ax1.set_xlabel("time since first sample [s]")
    ax1.set_ylabel("translation error [m]", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(times, rot_errors, color="#f97316", alpha=0.75, label="rotation error [deg]")
    ax2.set_ylabel("rotation error [deg]", color="#f97316")
    ax2.tick_params(axis="y", labelcolor="#f97316")
    plt.title("Relocalization pose vs map_eval*T error over time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_xyz_error_curve(errors: list[dict], output_path: Path):
    t0 = errors[0]["time_s"] if errors else 0.0
    times = np.array([row["time_s"] - t0 for row in errors], dtype=float)
    deltas = np.array([row["eval_minus_gt_xyz_m"] for row in errors], dtype=float)
    trans_errors = np.array([row["translation_error_m"] for row in errors], dtype=float)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    if len(deltas):
        ax.plot(times, deltas[:, 0], label="dx = eval - gt [m]", color="#2563eb", linewidth=1.8)
        ax.plot(times, deltas[:, 1], label="dy = eval - gt [m]", color="#16a34a", linewidth=1.8)
        ax.plot(times, deltas[:, 2], label="dz = eval - gt [m]", color="#f97316", linewidth=1.8)
        ax.plot(times, trans_errors, label="translation norm [m]", color="#64748b", linewidth=1.4, alpha=0.75)
        ax.axhline(0.0, color="#111827", linewidth=1, alpha=0.35)
    ax.set_xlabel("time since first sample [s]")
    ax.set_ylabel("error [m]")
    ax.set_title("Per-axis position residual: map_eval*T - relocalization pose in map_gt")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _img_data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _fmt(value: object, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _to_bgr(image: np.ndarray | None) -> np.ndarray:
    if image is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _image_shape_wh(image: np.ndarray) -> np.ndarray:
    """(width, height), the axis order LightGlueTRT expects for keypoint normalization."""
    height, width = image.shape[:2]
    return np.array([width, height], dtype=np.int64)


def _match_keypoints(
    matcher,
    feats0: dict,
    feats1: dict,
    image_shape0: np.ndarray,
    image_shape1: np.ndarray,
    *,
    loop: asyncio.AbstractEventLoop | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    own_loop = loop is None
    if own_loop:
        loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            matcher.infer(
                feats0["kpts"],
                feats1["kpts"],
                feats0["descps"],
                feats1["descps"],
                feats0["mask"],
                feats1["mask"],
                image_shape0,
                image_shape1,
            )
        )
    finally:
        if own_loop:
            loop.close()
    match_indices = result["match_indices"][0]
    valid_mask = match_indices != -1
    keypoints0 = feats0["kpts"][0][valid_mask]
    keypoints1 = feats1["kpts"][0][match_indices[valid_mask]]
    return np.asarray(keypoints0), np.asarray(keypoints1)


def _keypoints_to_world(
    keypoints: np.ndarray,
    depth: np.ndarray,
    pose_camera_to_world: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    height, width = depth.shape[:2]

    us = np.round(keypoints[:, 0]).astype(int)
    vs = np.round(keypoints[:, 1]).astype(int)
    us = np.clip(us, 0, width - 1)
    vs = np.clip(vs, 0, height - 1)

    zs = depth[vs, us].astype(np.float32)
    valid = (zs > 0.0) & (zs < 50.0)

    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy
    points_camera = np.stack([xs, ys, zs], axis=-1)
    points_camera[~valid] = 0.0

    points_world = points_camera @ pose_camera_to_world[:3, :3].T + pose_camera_to_world[:3, 3]
    return points_world.astype(np.float32), valid


def _depth_values_for_keypoints(keypoints: np.ndarray, depth: np.ndarray) -> np.ndarray:
    height, width = depth.shape[:2]
    us = np.round(keypoints[:, 0]).astype(int)
    vs = np.round(keypoints[:, 1]).astype(int)
    us = np.clip(us, 0, width - 1)
    vs = np.clip(vs, 0, height - 1)
    values = depth[vs, us].astype(np.float32)
    values = np.where((values > 0.0) & (values < 50.0), values, np.nan)
    return values


def _pnp_pose(
    points_world: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    min_inliers: int,
) -> tuple[bool, np.ndarray, np.ndarray]:
    if len(points_2d) <= 4:
        return False, np.eye(4), np.empty((0,), dtype=np.int32)
    # Match production's tinynav.core.math_utils.estimate_pose so diagnostic inlier counts are
    # comparable to what relocalization would actually see, not an unrelated OpenCV default.
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_world, points_2d, K, None,
        reprojectionError=2.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < min_inliers:
        return False, np.eye(4), np.empty((0,), dtype=np.int32)
    T_world_to_camera = np.eye(4)
    rot, _ = cv2.Rodrigues(rvec)
    T_world_to_camera[:3, :3] = rot
    T_world_to_camera[:3, 3] = tvec.reshape(3)
    return True, np.linalg.inv(T_world_to_camera), inliers.reshape(-1).astype(np.int32)


def _inlier_distribution(
    query_kpts: np.ndarray,
    inlier_indices: np.ndarray,
    image_shape: tuple[int, int],
) -> dict:
    if len(inlier_indices) == 0:
        return {
            "median_y_norm": None,
            "lower_half_ratio": 0.0,
            "bottom_third_ratio": 0.0,
            "x_span_norm": 0.0,
            "y_span_norm": 0.0,
            "bbox_area_norm": 0.0,
            "grid_coverage_4x4": 0,
        }
    h, w = image_shape[:2]
    pts = query_kpts[inlier_indices]
    x_norm = np.clip(pts[:, 0] / max(w, 1), 0.0, 1.0)
    y_norm = np.clip(pts[:, 1] / max(h, 1), 0.0, 1.0)
    xs = np.clip((x_norm * 4).astype(np.int32), 0, 3)
    ys = np.clip((y_norm * 4).astype(np.int32), 0, 3)
    return {
        "median_y_norm": float(np.median(y_norm)),
        "lower_half_ratio": float(np.mean(y_norm >= 0.5)),
        "bottom_third_ratio": float(np.mean(y_norm >= 2.0 / 3.0)),
        "x_span_norm": float(np.max(x_norm) - np.min(x_norm)),
        "y_span_norm": float(np.max(y_norm) - np.min(y_norm)),
        "bbox_area_norm": float((np.max(x_norm) - np.min(x_norm)) * (np.max(y_norm) - np.min(y_norm))),
        "grid_coverage_4x4": int(len(set(zip(xs.tolist(), ys.tolist())))),
    }


def _depth_distribution(depth_values: np.ndarray) -> dict:
    values = depth_values[np.isfinite(depth_values)]
    if len(values) == 0:
        return {
            "depth_median_m": None,
            "depth_p10_m": None,
            "depth_p90_m": None,
            "depth_iqr_m": None,
            "depth_rel_iqr": None,
        }
    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
    return {
        "depth_median_m": float(p50),
        "depth_p10_m": float(p10),
        "depth_p90_m": float(p90),
        "depth_iqr_m": float(p75 - p25),
        "depth_rel_iqr": float((p75 - p25) / max(p50, 1e-6)),
    }


def _landmark_geometry(points_world: np.ndarray) -> dict:
    if len(points_world) < 4:
        return {
            "landmark_span_m": None,
            "landmark_z_span_m": None,
            "landmark_planarity": None,
            "landmark_linearity": None,
            "landmark_spatiality": None,
        }
    centered = points_world - np.mean(points_world, axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    l0 = float(max(eigvals[0], 1e-9))
    l1 = float(eigvals[1])
    l2 = float(eigvals[2])
    mins = np.min(points_world, axis=0)
    maxs = np.max(points_world, axis=0)
    return {
        "landmark_span_m": float(np.linalg.norm(maxs - mins)),
        "landmark_z_span_m": float(maxs[2] - mins[2]),
        "landmark_planarity": float((l1 - l2) / l0),
        "landmark_linearity": float((l0 - l1) / l0),
        "landmark_spatiality": float(l2 / l0),
    }


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _trial_loop_quality(candidate: dict) -> float:
    if not candidate["pnp_success"]:
        return 0.0
    similarity_score = _clamp01((candidate["dino_vlad_similarity"] - 0.20) / 0.18)
    inlier_score = _clamp01(candidate["pnp_inlier_count"] / 150.0) * _clamp01(candidate["pnp_inlier_ratio"])
    image_score = 0.5 * _clamp01(candidate["grid_coverage_4x4"] / 12.0) + 0.5 * _clamp01(candidate["bbox_area_norm"] / 0.35)
    depth_score = _clamp01((candidate["depth_rel_iqr"] or 0.0) / 0.35)
    spatiality_score = _clamp01(((candidate["landmark_spatiality"] or 0.0) / 0.015) ** 0.5)
    return float(
        0.15 * similarity_score
        + 0.30 * inlier_score
        + 0.25 * image_score
        + 0.15 * depth_score
        + 0.15 * spatiality_score
    )


def _draw_match_image(
    ref_image: np.ndarray,
    query_image: np.ndarray,
    ref_kpts: np.ndarray,
    query_kpts: np.ndarray,
    inlier_indices: np.ndarray,
    max_lines: int,
) -> np.ndarray:
    left = _to_bgr(ref_image)
    right = _to_bgr(query_image)
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    canvas = np.concatenate([left, right], axis=1)
    offset = left.shape[1]
    indices = list(map(int, inlier_indices))
    if len(indices) > max_lines:
        pick = np.round(np.linspace(0, len(indices) - 1, max_lines)).astype(int)
        indices = [indices[i] for i in pick]
    for idx in indices:
        p0 = tuple(np.round(ref_kpts[idx]).astype(int))
        p1 = tuple(np.round(query_kpts[idx]).astype(int) + np.array([offset, 0]))
        cv2.circle(canvas, p0, 3, (37, 99, 235), -1)
        cv2.circle(canvas, p1, 3, (249, 115, 22), -1)
        cv2.line(canvas, p0, p1, (34, 197, 94), 1, cv2.LINE_AA)
    return canvas


def _compute_retrieval_diagnostics(
    *,
    map_gt_dir: Path,
    map_eval_dir: Path,
    output_dir: Path,
    errors: list[dict],
    transform_map_eval_to_gt: np.ndarray,
    sample_count: int,
    top_k: int,
    min_inliers: int,
    max_lines: int,
) -> list[dict]:
    from tinynav.core.build_map_node import TinyNavDB, find_loop
    from tinynav.core.models_trt import LightGlueTRT

    selected = sorted(errors, key=lambda row: row["translation_error_m"], reverse=True)[:sample_count]
    if not selected:
        return []

    image_dir = output_dir / "retrieval_diagnostics"
    image_dir.mkdir(parents=True, exist_ok=True)

    gt_poses = _load_pose_dict(map_gt_dir / "poses.npy")
    eval_poses = _load_pose_dict(map_eval_dir / "poses.npy")
    gt_timestamps = sorted(gt_poses)
    gt_db = TinyNavDB(str(map_gt_dir), is_scratch=False)
    eval_db = TinyNavDB(str(map_eval_dir), is_scratch=False)
    matcher = LightGlueTRT()
    gt_K = np.load(map_gt_dir / "intrinsics.npy")
    gt_descriptors = np.stack([gt_db.vlad_descriptors[t] for t in gt_timestamps])

    rows = []
    loop = asyncio.new_event_loop()
    try:
        for sample_index, error_row in enumerate(selected):
            eval_ts = int(error_row["timestamp_ns"])
            if eval_ts not in eval_poses or eval_ts not in eval_db.vlad_descriptors:
                continue
            eval_desc = eval_db.vlad_descriptors[eval_ts]
            candidates = list(reversed(find_loop(eval_desc, gt_descriptors, -1.0, top_k)))
            eval_depth, _, eval_features, _, eval_image_loader = eval_db.get_depth_embedding_features_images(eval_ts)
            del eval_depth
            eval_image = eval_image_loader()
            eval_image_shape = _image_shape_wh(eval_image)
            expected_pose = transform_map_eval_to_gt @ eval_poses[eval_ts]
            fusion_pose = np.eye(4)
            fusion_pose[:3, 3] = np.asarray(error_row["fusion_xyz"], dtype=float)

            candidate_rows = []
            for rank, (gt_idx, sim) in enumerate(candidates, start=1):
                next_sim = candidates[rank][1] if rank < len(candidates) else None
                gt_ts = gt_timestamps[int(gt_idx)]
                gt_depth, _, gt_features, _, gt_image_loader = gt_db.get_depth_embedding_features_images(gt_ts)
                gt_image = gt_image_loader()
                gt_image_shape = _image_shape_wh(gt_image)
                ref_kpts_all, query_kpts_all = _match_keypoints(
                    matcher, gt_features, eval_features, gt_image_shape, eval_image_shape, loop=loop
                )
                points_world, depth_valid = _keypoints_to_world(ref_kpts_all, gt_depth, gt_poses[gt_ts], gt_K)
                depth_values_all = _depth_values_for_keypoints(ref_kpts_all, gt_depth)
                points_world_valid = points_world[depth_valid].astype(np.float32)
                depth_values_valid = depth_values_all[depth_valid]
                query_valid = query_kpts_all[depth_valid].astype(np.float32)
                success, pose_camera_to_world, pnp_inliers = _pnp_pose(
                    points_world_valid,
                    query_valid,
                    gt_K,
                    min_inliers,
                )
                original_valid_indices = np.where(depth_valid)[0]
                original_inliers = (
                    original_valid_indices[pnp_inliers]
                    if success
                    else np.empty((0,), dtype=np.int32)
                )
                expected_delta = None
                fusion_delta = None
                if success:
                    expected_delta = pose_camera_to_world[:3, 3] - expected_pose[:3, 3]
                    fusion_delta = pose_camera_to_world[:3, 3] - fusion_pose[:3, 3]
                dist = _inlier_distribution(query_kpts_all, original_inliers, eval_image.shape[:2])
                inlier_depths = depth_values_valid[pnp_inliers] if success else np.empty((0,), dtype=np.float32)
                inlier_points = points_world_valid[pnp_inliers] if success else np.empty((0, 3), dtype=np.float32)
                depth_stats = _depth_distribution(inlier_depths)
                geometry_stats = _landmark_geometry(inlier_points)
                image_name = f"sample_{sample_index:03d}_rank_{rank}_{eval_ts}_{gt_ts}.jpg"
                cv2.imwrite(
                    str(image_dir / image_name),
                    _draw_match_image(
                        gt_image,
                        eval_image,
                        ref_kpts_all,
                        query_kpts_all,
                        original_inliers,
                        max_lines,
                    ),
                )
                candidate_row = {
                    "rank": rank,
                    "gt_timestamp_ns": int(gt_ts),
                    "dino_vlad_similarity": float(sim),
                    "dino_vlad_margin_to_next": float(sim - next_sim) if next_sim is not None else None,
                    "match_count": int(len(query_kpts_all)),
                    "landmark_count": int(len(query_valid)),
                    "pnp_success": bool(success),
                    "pnp_inlier_count": int(len(original_inliers)),
                    "pnp_inlier_ratio": float(len(original_inliers) / max(len(query_valid), 1)),
                    "pnp_error_to_eval_pose_m": (
                        float(np.linalg.norm(expected_delta)) if expected_delta is not None else None
                    ),
                    "pnp_dz_to_eval_pose_m": (
                        float(expected_delta[2]) if expected_delta is not None else None
                    ),
                    "pnp_error_to_relocalization_m": (
                        float(np.linalg.norm(fusion_delta)) if fusion_delta is not None else None
                    ),
                    "pnp_dz_to_relocalization_m": (
                        float(fusion_delta[2]) if fusion_delta is not None else None
                    ),
                    "image": str(Path("retrieval_diagnostics") / image_name),
                    **dist,
                    **depth_stats,
                    **geometry_stats,
                }
                candidate_row["trial_loop_quality"] = _trial_loop_quality(candidate_row)
                candidate_rows.append(candidate_row)

            rows.append(
                {
                    "sample_index": sample_index,
                    "timestamp_ns": eval_ts,
                    "benchmark_translation_error_m": error_row["translation_error_m"],
                    "benchmark_rotation_error_deg": error_row["rotation_error_deg"],
                    "eval_minus_gt_xyz_m": error_row["eval_minus_gt_xyz_m"],
                    "map_eval_in_gt_xyz": error_row["map_eval_in_gt_xyz"],
                    "relocalization_xyz": error_row["fusion_xyz"],
                    "candidates": candidate_rows,
                }
            )
    finally:
        loop.close()
        gt_db.close()
        eval_db.close()

    (output_dir / "retrieval_diagnostics.json").write_text(json.dumps(rows, indent=2))
    return rows


def _write_html_report(
    output_dir: Path,
    metrics: dict,
    errors: list[dict],
    retrieval_diagnostics: Optional[list[dict]] = None,
):
    trajectory_uri = _img_data_uri(output_dir / "trajectory_xy.png")
    trajectory_xz_uri = _img_data_uri(output_dir / "trajectory_xz.png")
    trajectory_yz_uri = _img_data_uri(output_dir / "trajectory_yz.png")
    error_uri = _img_data_uri(output_dir / "translation_rotation_error.png")
    xyz_error_uri = _img_data_uri(output_dir / "xyz_error.png")
    trans = metrics["translation_error_m"]
    rot = metrics["rotation_error_deg"]
    xyz = metrics["eval_minus_gt_xyz_m"]
    abs_xyz = metrics["abs_eval_minus_gt_xyz_m"]
    threshold_rows = "\n".join(
        f"<tr><td>{html.escape(k)}</td><td>{v['count']}</td><td>{v['ratio'] * 100:.1f}%</td></tr>"
        for k, v in metrics["thresholds"].items()
    )
    top_errors = sorted(errors, key=lambda row: row["translation_error_m"], reverse=True)[:20]
    error_rows = "\n".join(
        "<tr>"
        f"<td>{row['timestamp_ns']}</td>"
        f"<td>{row['translation_error_m']:.4f}</td>"
        f"<td>{row['rotation_error_deg']:.3f}</td>"
        f"<td>{', '.join(f'{x:.3f}' for x in row['eval_minus_gt_xyz_m'])}</td>"
        f"<td>{', '.join(f'{x:.2f}' for x in row['map_eval_in_gt_xyz'])}</td>"
        f"<td>{', '.join(f'{x:.2f}' for x in row['fusion_xyz'])}</td>"
        "</tr>"
        for row in top_errors
    )
    transform_json = json.dumps(metrics["transform_map_eval_to_gt"], indent=2)
    retrieval_sections = ""
    if retrieval_diagnostics:
        cards = []
        for row in retrieval_diagnostics:
            candidate_cards = []
            for c in row["candidates"]:
                image_uri = _img_data_uri(output_dir / c["image"])
                candidate_cards.append(
                    f"""
                    <div class="candidate">
                      <h3>top {c['rank']} · DINO-VLAD sim={c['dino_vlad_similarity']:.4f} · DINO-VLAD margin→next={_fmt(c['dino_vlad_margin_to_next'], 4)} · PnP={'ok' if c['pnp_success'] else 'fail'}</h3>
                      <div class="table-scroll"><table class="diagnostic-table">
                          <tr><th>DINO-VLAD sim</th><th>DINO-VLAD margin</th><th>quality</th><th>matches</th><th>landmarks</th><th>inliers</th><th>ratio</th><th>PnP→eval [m]</th><th>dz→eval [m]</th><th>PnP→relocal [m]</th><th>lower half</th><th>bottom 1/3</th><th>x/y span</th><th>depth med/IQR [m]</th><th>3D spatiality</th><th>3D planarity</th><th>grid</th></tr>
                          <tr>
                            <td>{_fmt(c['dino_vlad_similarity'], 4)}</td><td>{_fmt(c['dino_vlad_margin_to_next'], 4)}</td><td>{_fmt(c['trial_loop_quality'], 3)}</td><td>{c['match_count']}</td><td>{c['landmark_count']}</td><td>{c['pnp_inlier_count']}</td><td>{_fmt(c['pnp_inlier_ratio'], 3)}</td>
                            <td>{_fmt(c['pnp_error_to_eval_pose_m'], 3)}</td><td>{_fmt(c['pnp_dz_to_eval_pose_m'], 3)}</td><td>{_fmt(c['pnp_error_to_relocalization_m'], 3)}</td>
                            <td>{_fmt(c['lower_half_ratio'], 3)}</td><td>{_fmt(c['bottom_third_ratio'], 3)}</td><td>{_fmt(c['x_span_norm'], 2)} / {_fmt(c['y_span_norm'], 2)}</td>
                            <td>{_fmt(c['depth_median_m'], 2)} / {_fmt(c['depth_iqr_m'], 2)}</td><td>{_fmt(c['landmark_spatiality'], 4)}</td><td>{_fmt(c['landmark_planarity'], 3)}</td><td>{c['grid_coverage_4x4']}</td>
                          </tr>
                        </table></div>
                      <img src="{image_uri}" alt="top {c['rank']} retrieval candidate" />
                    </div>
                    """
                )
            cards.append(
                f"""
                <section>
                  <h2>retrieval diagnostic sample {row['sample_index']} · ts={row['timestamp_ns']}</h2>
                  <p>Benchmark error: trans={row['benchmark_translation_error_m']:.3f} m, rot={row['benchmark_rotation_error_deg']:.3f} deg,
                  eval-gt residual={', '.join(f'{x:.3f}' for x in row['eval_minus_gt_xyz_m'])} m.</p>
                  {''.join(candidate_cards)}
                </section>
                """
            )
        retrieval_sections = f"""
          <section><h2>Largest Error Retrieval/PnP Diagnostics</h2>
            <p>For the largest benchmark errors, each sample queries map_gt with the eval keyframe descriptor and shows top candidates. Match image: left=GT candidate, right=eval query, green lines=PnP inliers.</p>
          </section>
          {''.join(cards)}
        """

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TinyNav Keyframe Relocalization Benchmark</title>
  <style>
    :root {{ --bg:#0b1020; --panel:rgba(255,255,255,.075); --line:rgba(255,255,255,.14); --text:#f4f7fb; --muted:#aeb9cc; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--text); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      background:radial-gradient(circle at 10% 0%,rgba(96,165,250,.25),transparent 32%),radial-gradient(circle at 90% 8%,rgba(52,211,153,.17),transparent 28%),var(--bg); }}
    main {{ max-width:1180px; margin:0 auto; padding:44px 24px 80px; }}
    .hero,section {{ border:1px solid var(--line); border-radius:26px; background:rgba(255,255,255,.055); box-shadow:0 24px 70px rgba(0,0,0,.22); }}
    .hero {{ padding:34px; background:linear-gradient(145deg,rgba(255,255,255,.12),rgba(255,255,255,.045)); }}
    section {{ margin-top:24px; padding:28px; }}
    h1 {{ margin:0 0 12px; font-size:46px; letter-spacing:-1.6px; }}
    h2 {{ margin:0 0 18px; font-size:28px; }}
    p,td {{ color:var(--muted); line-height:1.65; }}
    code,pre {{ background:rgba(0,0,0,.35); border:1px solid var(--line); border-radius:14px; }}
    pre {{ padding:16px; overflow:auto; color:#dbeafe; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:22px; }}
    .metric {{ padding:18px; border-radius:20px; background:var(--panel); border:1px solid var(--line); }}
    .metric strong {{ display:block; font-size:30px; margin-bottom:6px; }}
    .metric span {{ color:var(--muted); font-size:13px; }}
    .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    img {{ max-width:100%; border-radius:18px; border:1px solid var(--line); background:white; }}
    table {{ width:100%; border-collapse:collapse; overflow:hidden; border-radius:16px; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; }}
    th {{ color:#dbeafe; background:rgba(96,165,250,.12); }}
    .table-scroll {{ max-width:100%; overflow-x:auto; border:1px solid var(--line); border-radius:16px; }}
    .table-scroll table {{ border-radius:0; }}
    .diagnostic-table {{ min-width:1320px; font-size:12px; }}
    .diagnostic-table th,.diagnostic-table td {{ padding:8px 10px; white-space:nowrap; }}
    h3 {{ margin:14px 0 10px; font-size:18px; }}
    .candidate {{ margin-top:18px; padding-top:14px; border-top:1px solid var(--line); }}
    .candidate img {{ margin-top:12px; }}
    .flow {{ display:flex; flex-wrap:wrap; gap:10px; }}
    .step {{ padding:11px 13px; border-radius:999px; background:rgba(96,165,250,.12); border:1px solid rgba(96,165,250,.24); }}
    @media (max-width:900px) {{ .grid,.cols {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <div class="hero">
    <h1>TinyNav Keyframe Relocalization Benchmark</h1>
    <p>Evaluate keyframe relocalization poses against the eval map keyframe trajectory aligned into GT map frame:
      <code>relocalization_pose_map_gt vs T_map_eval_to_gt * map_eval_keyframe_pose</code>.</p>
    <div class="flow">
      <div class="step">GT source → map_gt</div>
      <div class="step">eval bag → map_eval</div>
      <div class="step">eval bag + map_gt → relocalization poses</div>
      <div class="step">fit T(map_eval→gt)</div>
      <div class="step">compare relocalization vs map_eval*T</div>
    </div>
    <div class="grid">
      <div class="metric"><strong>{metrics['sampled_timestamps']}</strong><span>sampled timestamps</span></div>
      <div class="metric"><strong>{metrics['paired_poses']}</strong><span>paired poses</span></div>
      <div class="metric"><strong>{_fmt(trans['median'], 3)} m</strong><span>median translation error</span></div>
      <div class="metric"><strong>{_fmt(trans['p90'], 3)} m</strong><span>p90 translation error</span></div>
    </div>
  </div>

  <section><h2>Inputs</h2><table>
    <tr><th>Item</th><th>Value</th></tr>
    <tr><td>GT source</td><td>{html.escape(metrics['inputs']['gt_source'])}</td></tr>
    <tr><td>Eval source</td><td>{html.escape(metrics['inputs']['eval_source'])}</td></tr>
    <tr><td>map_gt dir</td><td>{html.escape(metrics['inputs']['map_gt_dir'])}</td></tr>
    <tr><td>map_eval dir</td><td>{html.escape(metrics['inputs']['map_eval_dir'])}</td></tr>
    <tr><td>localization dir</td><td>{html.escape(metrics['inputs']['localization_dir'])}</td></tr>
  </table></section>

  <section><h2>Error Summary</h2><div class="cols">
    <table><tr><th>Translation metric</th><th>Value [m]</th></tr>
      <tr><td>mean</td><td>{_fmt(trans['mean'])}</td></tr><tr><td>median</td><td>{_fmt(trans['median'])}</td></tr>
      <tr><td>p90</td><td>{_fmt(trans['p90'])}</td></tr><tr><td>p95</td><td>{_fmt(trans['p95'])}</td></tr>
      <tr><td>max</td><td>{_fmt(trans['max'])}</td></tr><tr><td>rmse</td><td>{_fmt(trans['rmse'])}</td></tr></table>
    <table><tr><th>Rotation metric</th><th>Value [deg]</th></tr>
      <tr><td>mean</td><td>{_fmt(rot['mean'])}</td></tr><tr><td>median</td><td>{_fmt(rot['median'])}</td></tr>
      <tr><td>p90</td><td>{_fmt(rot['p90'])}</td></tr><tr><td>p95</td><td>{_fmt(rot['p95'])}</td></tr>
      <tr><td>max</td><td>{_fmt(rot['max'])}</td></tr><tr><td>rmse</td><td>{_fmt(rot['rmse'])}</td></tr></table>
  </div></section>

  <section><h2>XYZ Residual Summary</h2>
    <p>Signed residual is <code>eval_minus_gt_xyz_m = T_map_eval_to_gt * map_eval_pose - relocalization_pose_in_map_gt</code>.
    Positive <code>dz</code> means the aligned eval map is higher than the GT/relocalization pose.</p>
    <div class="cols">
      <table><tr><th>Axis</th><th>mean [m]</th><th>median [m]</th><th>p90 [m]</th><th>max [m]</th></tr>
        <tr><td>dx</td><td>{_fmt(xyz['dx']['mean'])}</td><td>{_fmt(xyz['dx']['median'])}</td><td>{_fmt(xyz['dx']['p90'])}</td><td>{_fmt(xyz['dx']['max'])}</td></tr>
        <tr><td>dy</td><td>{_fmt(xyz['dy']['mean'])}</td><td>{_fmt(xyz['dy']['median'])}</td><td>{_fmt(xyz['dy']['p90'])}</td><td>{_fmt(xyz['dy']['max'])}</td></tr>
        <tr><td>dz</td><td>{_fmt(xyz['dz']['mean'])}</td><td>{_fmt(xyz['dz']['median'])}</td><td>{_fmt(xyz['dz']['p90'])}</td><td>{_fmt(xyz['dz']['max'])}</td></tr>
      </table>
      <table><tr><th>Axis</th><th>abs mean [m]</th><th>abs median [m]</th><th>abs p90 [m]</th><th>abs max [m]</th></tr>
        <tr><td>|dx|</td><td>{_fmt(abs_xyz['dx']['mean'])}</td><td>{_fmt(abs_xyz['dx']['median'])}</td><td>{_fmt(abs_xyz['dx']['p90'])}</td><td>{_fmt(abs_xyz['dx']['max'])}</td></tr>
        <tr><td>|dy|</td><td>{_fmt(abs_xyz['dy']['mean'])}</td><td>{_fmt(abs_xyz['dy']['median'])}</td><td>{_fmt(abs_xyz['dy']['p90'])}</td><td>{_fmt(abs_xyz['dy']['max'])}</td></tr>
        <tr><td>|dz|</td><td>{_fmt(abs_xyz['dz']['mean'])}</td><td>{_fmt(abs_xyz['dz']['median'])}</td><td>{_fmt(abs_xyz['dz']['p90'])}</td><td>{_fmt(abs_xyz['dz']['max'])}</td></tr>
      </table>
    </div>
  </section>

  <section><h2>Acceptance by Translation Threshold</h2><table>
    <tr><th>Threshold</th><th>Count</th><th>Ratio</th></tr>{threshold_rows}
  </table></section>

  <section><h2>Trajectory and Error Curves</h2>
  <div><img src="{error_uri}" alt="Error curve" /></div>
  <div><img src="{xyz_error_uri}" alt="XYZ residual curve" /></div>
  <p>For stairs and multi-floor motion, inspect XZ / YZ projections as separate large plots. They expose vertical floor drift better than the top-down XY view.</p>
  <div class="cols">
    <div><img src="{trajectory_xz_uri}" alt="XZ trajectory comparison" /></div>
    <div><img src="{trajectory_yz_uri}" alt="YZ trajectory comparison" /></div>
  </div>
  <p>XY projection is kept below for quick top-down inspection.</p>
  <div><img src="{trajectory_uri}" alt="XY trajectory comparison" /></div></section>

  <section><h2>Estimated Transform: T_map_eval_to_gt</h2>
    <p>RANSAC inliers: {metrics['transform_fit']['inlier_count']} / {metrics['transform_fit']['candidate_pairs']}
    ({metrics['transform_fit']['inlier_ratio'] * 100:.1f}%), threshold: {metrics['transform_fit']['inlier_threshold_m']:.3f} m.</p>
    <pre>{html.escape(transform_json)}</pre>
  </section>

  <section><h2>Largest Translation Errors</h2><table>
    <tr><th>timestamp ns</th><th>trans err [m]</th><th>rot err [deg]</th><th>eval-gt dx,dy,dz [m]</th><th>map_eval*T xyz</th><th>relocalization xyz</th></tr>{error_rows}
  </table></section>
  {retrieval_sections}
</main></body></html>
"""
    (output_dir / "index.html").write_text(html_text)


def _safe_name(path_or_name: str) -> str:
    name = Path(path_or_name).name or "map"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _make_run_output_dir(args: argparse.Namespace) -> Path:
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.output_dir:
        out = Path(args.output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out
    map_name_source = args.map_gt or args.bag_gt or "map_gt"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / f"{timestamp}_{_safe_name(map_name_source)}_benchmark"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _make_run_work_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.work_dir:
        work = Path(args.work_dir).resolve()
    else:
        root = Path(args.work_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        work = root / f"{output_dir.name}_work"
    work.mkdir(parents=True, exist_ok=True)
    return work


def run(args: argparse.Namespace) -> Path:
    output_dir = _make_run_output_dir(args)
    work_dir = _make_run_work_dir(args, output_dir)
    map_gt_dir = Path(args.map_gt).resolve() if args.map_gt else work_dir / "map_gt"
    map_eval_dir = Path(args.map_eval).resolve() if args.map_eval else work_dir / "map_eval"
    localization_dir = work_dir / "eval_localized_in_map_gt"

    if args.bag_gt:
        print("\nStep 1/6: building map_gt from bag_gt")
        _build_map_from_bag(
            bag_path=args.bag_gt,
            map_dir=map_gt_dir,
            rate=args.rate,
            verbose_timer=args.verbose_timer,
        )
    else:
        print(f"\nStep 1/6: using existing map_gt: {map_gt_dir}")

    if args.bag_eval:
        print("\nStep 2/6: building map_eval from bag_eval")
        _build_map_from_bag(
            bag_path=args.bag_eval,
            map_dir=map_eval_dir,
            rate=args.rate,
            verbose_timer=args.verbose_timer,
        )
        print("\nStep 3/6: replaying bag_eval against map_gt")
        _localize_eval_bag_in_gt_map(
            bag_eval=args.bag_eval,
            map_gt_dir=map_gt_dir,
            localization_dir=localization_dir,
            rate=args.rate,
            timeout=args.timeout,
            verbose_timer=args.verbose_timer,
        )
    else:
        print(f"\nStep 2/6: using existing map_eval: {map_eval_dir}")
        print("Step 3/6: skipping localization (no bag_eval)")

    if args.timestamps_file:
        timestamps = np.loadtxt(args.timestamps_file, dtype=np.int64)
    else:
        # Sample from map_eval's own keyframe timestamps rather than the bag's recording-time
        # metadata: message header stamps aren't always epoch time (e.g. Looper bags use a
        # relative/device clock), so bag-metadata-based sampling can't be reliably matched back
        # against map_eval/localization poses, which are keyed by header stamp.
        timestamps = _sample_timestamps_from_map(map_eval_dir, args.num_samples, args.trim_ratio)

    print("\nStep 4/6: querying map_eval reference poses and fusion poses")
    relocalization_path = localization_dir / "relocalization_poses.npy"
    if not relocalization_path.exists():
        raise FileNotFoundError(
            f"Localization results not found: {relocalization_path}\n"
            f"--map-eval skips localization. Either use --bag-eval to run localization, "
            f"or ensure localization results exist in the work directory."
        )
    map_eval_reference_poses, fusion_poses, skipped = _query_eval_reference_and_fusion(
        timestamps=timestamps,
        map_eval_dir=map_eval_dir,
        localization_dir=localization_dir,
        max_anchor_dt_ns=int(args.max_anchor_dt_s * 1e9),
    )
    paired_timestamps = sorted(set(map_eval_reference_poses) & set(fusion_poses))
    if len(paired_timestamps) < 3:
        raise RuntimeError(f"Only {len(paired_timestamps)} paired poses found; need at least 3")
    map_eval_reference_poses = {ts: map_eval_reference_poses[ts] for ts in paired_timestamps}
    fusion_poses = {ts: fusion_poses[ts] for ts in paired_timestamps}

    print("\nStep 5/6: fitting T_map_eval_to_gt")
    transform_map_eval_to_gt, inlier_timestamps, transform_info = _ransac_transform(
        source_poses=map_eval_reference_poses,
        target_poses=fusion_poses,
        inlier_threshold_m=args.ransac_threshold_m,
        iterations=args.ransac_iterations,
        seed=args.seed,
        alignment_mode=args.alignment_mode,
    )

    fit_source = "ransac_all_pairs"
    eval_map_poses = map_eval_reference_poses
    eval_fusion_poses = fusion_poses
    if args.evaluate_inliers_only:
        fit_source = "ransac_inliers_only"
        eval_map_poses = {ts: map_eval_reference_poses[ts] for ts in inlier_timestamps}
        eval_fusion_poses = {ts: fusion_poses[ts] for ts in inlier_timestamps}

    print("\nStep 6/6: computing errors and writing HTML report")
    errors = _compute_errors(
        map_eval_poses=eval_map_poses,
        fusion_poses=eval_fusion_poses,
        transform_map_eval_to_gt=transform_map_eval_to_gt,
    )
    if not errors:
        raise RuntimeError("No errors computed")

    metrics = {
        "inputs": {
            "gt_source": args.map_gt or args.bag_gt,
            "bag_gt": args.bag_gt,
            "eval_source": args.map_eval or args.bag_eval,
            "bag_eval": args.bag_eval,
            "map_gt_dir": str(map_gt_dir),
            "map_eval_dir": str(map_eval_dir),
            "localization_dir": str(localization_dir),
            "output_dir": str(output_dir),
            "work_dir": str(work_dir),
        },
        "sampled_timestamps": int(len(timestamps)),
        "paired_poses": int(len(paired_timestamps)),
        "evaluated_poses": int(len(errors)),
        "skipped": skipped,
        "fit_source": fit_source,
        "transform_fit": transform_info,
        "transform_map_eval_to_gt": transform_map_eval_to_gt.tolist(),
        "translation_error_m": _summary(row["translation_error_m"] for row in errors),
        "rotation_error_deg": _summary(row["rotation_error_deg"] for row in errors),
        "eval_minus_gt_xyz_m": _xyz_summary(errors, "eval_minus_gt_xyz_m"),
        "abs_eval_minus_gt_xyz_m": _xyz_summary(errors, "abs_eval_minus_gt_xyz_m"),
        "thresholds": _threshold_stats(errors, args.thresholds_m),
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "per_sample_errors.json").write_text(json.dumps(errors, indent=2))
    np.save(output_dir / "T_map_eval_to_gt.npy", transform_map_eval_to_gt)
    np.savetxt(output_dir / "sampled_timestamps_ns.txt", np.array(timestamps, dtype=np.int64), fmt="%d")

    _plot_trajectory(errors, output_dir / "trajectory_xy.png")
    _plot_trajectory_projection(
        errors,
        output_dir / "trajectory_xz.png",
        axes=(0, 2),
        labels=("x [m]", "z [m]"),
        title="Trajectory comparison in map_gt frame (XZ projection)",
    )
    _plot_trajectory_projection(
        errors,
        output_dir / "trajectory_yz.png",
        axes=(1, 2),
        labels=("y [m]", "z [m]"),
        title="Trajectory comparison in map_gt frame (YZ projection)",
    )
    _plot_error_curve(errors, output_dir / "translation_rotation_error.png")
    _plot_xyz_error_curve(errors, output_dir / "xyz_error.png")
    retrieval_diagnostics = None
    if not args.disable_retrieval_diagnostics:
        print("\nExtra: computing retrieval/PnP diagnostics for largest errors")
        retrieval_diagnostics = _compute_retrieval_diagnostics(
            map_gt_dir=map_gt_dir,
            map_eval_dir=map_eval_dir,
            output_dir=output_dir,
            errors=errors,
            transform_map_eval_to_gt=transform_map_eval_to_gt,
            sample_count=args.retrieval_diagnostic_samples,
            top_k=args.retrieval_diagnostic_top_k,
            min_inliers=args.retrieval_diagnostic_min_inliers,
            max_lines=args.retrieval_diagnostic_max_lines,
        )
    _write_html_report(output_dir, metrics, errors, retrieval_diagnostics)

    print(f"\nBenchmark complete: {output_dir / 'index.html'}")
    print(json.dumps(metrics["translation_error_m"], indent=2))
    return output_dir / "index.html"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark keyframe relocalization against map_eval trajectory transformed into map_gt."
    )
    gt = parser.add_mutually_exclusive_group(required=True)
    gt.add_argument("--bag-gt", help="ROS2 bag used to build map_gt")
    gt.add_argument("--map-gt", help="Existing GT/reference map directory")
    ev = parser.add_mutually_exclusive_group(required=True)
    ev.add_argument("--bag-eval", help="ROS2 bag used to build map_eval and replay against map_gt")
    ev.add_argument("--map-eval", help="Existing eval map directory; skips map build and localization")
    parser.add_argument("--output-root", default="output", help="Parent directory for timestamped benchmark folder")
    parser.add_argument("--output-dir", help="Exact output directory; overrides timestamped folder creation")
    parser.add_argument("--work-root", default="output/benchmark_work", help="Parent directory for generated maps and localization scratch data")
    parser.add_argument("--work-dir", help="Exact work directory for generated maps and localization scratch data")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of sampled timestamps")
    parser.add_argument("--trim-ratio", type=float, default=0.05, help="Trim this fraction of map_eval keyframes from the start/end before sampling")
    parser.add_argument("--timestamps-file", help="Optional text file containing timestamps in ns")
    parser.add_argument("--rate", type=float, default=1.0, help="Replay/build rate")
    parser.add_argument("--timeout", type=float, default=60.0, help="Data save timeout in seconds")
    parser.add_argument("--verbose-timer", action="store_true", help="Enable verbose node timer logs")
    parser.add_argument("--max-anchor-dt-s", type=float, default=1.0, help="Max timestamp distance to anchor pose")
    parser.add_argument("--ransac-threshold-m", type=float, default=0.20, help="RANSAC inlier threshold")
    parser.add_argument("--ransac-iterations", type=int, default=1000, help="RANSAC iterations")
    parser.add_argument(
        "--alignment-mode",
        choices=["se2_z", "se3"],
        default="se2_z",
        help="Map alignment model. se2_z estimates planar yaw/xy plus z offset; se3 estimates full 3D rigid transform.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--evaluate-inliers-only", action="store_true", help="Only evaluate RANSAC inliers")
    parser.add_argument(
        "--thresholds-m",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30, 0.50, 1.00],
        help="Translation thresholds reported in HTML",
    )
    parser.add_argument(
        "--disable-retrieval-diagnostics",
        action="store_true",
        help="Skip top-k retrieval/PnP diagnostics for the largest translation errors",
    )
    parser.add_argument(
        "--retrieval-diagnostic-samples",
        type=int,
        default=12,
        help="Number of largest-error samples to inspect with retrieval/PnP",
    )
    parser.add_argument(
        "--retrieval-diagnostic-top-k",
        type=int,
        default=3,
        help="Number of GT retrieval candidates shown for each diagnostic sample",
    )
    parser.add_argument(
        "--retrieval-diagnostic-min-inliers",
        type=int,
        default=50,
        help="Minimum PnP inliers required to mark a diagnostic candidate as successful",
    )
    parser.add_argument(
        "--retrieval-diagnostic-max-lines",
        type=int,
        default=80,
        help="Maximum PnP inlier match lines drawn per diagnostic image",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
