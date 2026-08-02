#!/usr/bin/env python3
"""
Cross-map keyframe relocalization benchmark.

This tool evaluates keyframe relocalization poses against an independently
built eval map trajectory aligned into the GT map frame:

    relocalized_pose_in_map_gt  vs  T_map_eval_to_gt * map_eval_keyframe_pose

Inputs:
  - GT source: either --map-gt or --bag-gt. If a bag is given, it is built into
    map_gt first.
  - Eval source: --bag-eval is used to build map_eval unless --map-eval is
    provided, and is replayed against map_gt to produce relocalization poses.

The original benchmark_mapping.py is intentionally left untouched. This script
shares only small utility concepts and produces a standalone HTML report.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from launch import LaunchDescription, LaunchService
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown

from benchmark_mapping import BagMetadataExtractor, find_closest_pose


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
    timeout: float,
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
    timeout: float,
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
            timeout=timeout,
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


def _sample_timestamps_from_bag(
    bag_path: str,
    num_samples: int,
    trim_ratio: float,
) -> np.ndarray:
    start_ns, end_ns = BagMetadataExtractor.get_bag_time_range(bag_path)
    if start_ns is None or end_ns is None:
        raise RuntimeError(f"Failed to read bag time range: {bag_path}")
    duration = end_ns - start_ns
    trim_ns = int(duration * trim_ratio)
    sample_start = start_ns + trim_ns
    sample_end = end_ns - trim_ns
    if sample_start >= sample_end:
        sample_start, sample_end = start_ns, end_ns
    return BagMetadataExtractor.sample_timestamps_evenly(
        sample_start,
        sample_end,
        num_samples,
    )


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


def _ransac_transform(
    *,
    source_poses: PoseDict,
    target_poses: PoseDict,
    inlier_threshold_m: float,
    iterations: int,
    seed: int,
) -> Tuple[np.ndarray, list[int], dict]:
    timestamps = sorted(set(source_poses) & set(target_poses))
    if len(timestamps) < 3:
        raise RuntimeError("Need at least 3 common timestamps to estimate transform")

    src = np.array([source_poses[t][:3, 3] for t in timestamps], dtype=float)
    dst = np.array([target_poses[t][:3, 3] for t in timestamps], dtype=float)
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(len(timestamps), dtype=bool)
    best_transform = np.eye(4)

    for _ in range(max(iterations, 1)):
        sample_idx = rng.choice(len(timestamps), size=3, replace=False)
        candidate = _estimate_rigid_transform(src[sample_idx], dst[sample_idx])
        transformed = (candidate @ np.c_[src, np.ones(len(src))].T).T[:, :3]
        distances = np.linalg.norm(transformed - dst, axis=1)
        mask = distances <= inlier_threshold_m
        if int(mask.sum()) > int(best_mask.sum()):
            best_mask = mask
            best_transform = candidate

    if best_mask.sum() >= 3:
        best_transform = _estimate_rigid_transform(src[best_mask], dst[best_mask])
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
        rows.append({
            "timestamp_ns": int(timestamp),
            "time_s": float(timestamp) / 1e9,
            "translation_error_m": float(np.linalg.norm(map_eval_in_gt[:3, 3] - fusion[:3, 3])),
            "rotation_error_deg": _rotation_error_deg(map_eval_in_gt[:3, :3], fusion[:3, :3]),
            "map_eval_in_gt_xyz": map_eval_in_gt[:3, 3].tolist(),
            "fusion_xyz": fusion[:3, 3].tolist(),
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


def _plot_trajectory(errors: list[dict], output_path: Path):
    ref_xy = np.array([row["map_eval_in_gt_xyz"][:2] for row in errors], dtype=float)
    fusion_xy = np.array([row["fusion_xyz"][:2] for row in errors], dtype=float)
    plt.figure(figsize=(8, 7))
    if len(ref_xy):
        plt.plot(ref_xy[:, 0], ref_xy[:, 1], "-", label="map_eval * T reference", linewidth=2)
        plt.plot(fusion_xy[:, 0], fusion_xy[:, 1], "-", label="relocalization pose in map_gt", linewidth=2)
        plt.scatter(ref_xy[:, 0], ref_xy[:, 1], s=10, alpha=0.45)
        plt.scatter(fusion_xy[:, 0], fusion_xy[:, 1], s=10, alpha=0.45)
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectory comparison in map_gt frame")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


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


def _img_data_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _fmt(value: object, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _write_html_report(output_dir: Path, metrics: dict, errors: list[dict]):
    trajectory_uri = _img_data_uri(output_dir / "trajectory_xy.png")
    error_uri = _img_data_uri(output_dir / "translation_rotation_error.png")
    trans = metrics["translation_error_m"]
    rot = metrics["rotation_error_deg"]
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
        f"<td>{', '.join(f'{x:.2f}' for x in row['map_eval_in_gt_xyz'])}</td>"
        f"<td>{', '.join(f'{x:.2f}' for x in row['fusion_xyz'])}</td>"
        "</tr>"
        for row in top_errors
    )
    transform_json = json.dumps(metrics["transform_map_eval_to_gt"], indent=2)

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
    <tr><td>Eval bag</td><td>{html.escape(metrics['inputs']['bag_eval'])}</td></tr>
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

  <section><h2>Acceptance by Translation Threshold</h2><table>
    <tr><th>Threshold</th><th>Count</th><th>Ratio</th></tr>{threshold_rows}
  </table></section>

  <section><h2>Trajectory and Error Curves</h2><div class="cols">
    <div><img src="{trajectory_uri}" alt="Trajectory comparison" /></div>
    <div><img src="{error_uri}" alt="Error curve" /></div>
  </div></section>

  <section><h2>Estimated Transform: T_map_eval_to_gt</h2>
    <p>RANSAC inliers: {metrics['transform_fit']['inlier_count']} / {metrics['transform_fit']['candidate_pairs']}
    ({metrics['transform_fit']['inlier_ratio'] * 100:.1f}%), threshold: {metrics['transform_fit']['inlier_threshold_m']:.3f} m.</p>
    <pre>{html.escape(transform_json)}</pre>
  </section>

  <section><h2>Largest Translation Errors</h2><table>
    <tr><th>timestamp ns</th><th>trans err [m]</th><th>rot err [deg]</th><th>map_eval*T xyz</th><th>fusion xyz</th></tr>{error_rows}
  </table></section>
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


def run(args: argparse.Namespace) -> Path:
    if not args.map_gt and not args.bag_gt:
        raise ValueError("Provide either --map-gt or --bag-gt")
    if not args.bag_eval:
        raise ValueError("--bag-eval is required")

    output_dir = _make_run_output_dir(args)
    map_gt_dir = Path(args.map_gt).resolve() if args.map_gt else output_dir / "map_gt"
    map_eval_dir = Path(args.map_eval).resolve() if args.map_eval else output_dir / "map_eval"
    localization_dir = output_dir / "eval_localized_in_map_gt"

    if not args.skip_runs:
        if args.bag_gt and not args.map_gt and not args.skip_map_gt:
            print("\nStep 1/6: building map_gt from bag_gt")
            _build_map_from_bag(
                bag_path=args.bag_gt,
                map_dir=map_gt_dir,
                rate=args.rate,
                timeout=args.timeout,
                verbose_timer=args.verbose_timer,
            )
        else:
            print(f"\nStep 1/6: using existing map_gt: {map_gt_dir}")

        if not args.map_eval and not args.skip_map_eval:
            print("\nStep 2/6: building map_eval from bag_eval")
            _build_map_from_bag(
                bag_path=args.bag_eval,
                map_dir=map_eval_dir,
                rate=args.rate,
                timeout=args.timeout,
                verbose_timer=args.verbose_timer,
            )
        else:
            print(f"\nStep 2/6: using existing map_eval: {map_eval_dir}")

        if not args.skip_localization:
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
        print("\nSkipping ROS runs and using existing directories")

    if args.timestamps_file:
        timestamps = np.loadtxt(args.timestamps_file, dtype=np.int64)
    else:
        timestamps = _sample_timestamps_from_bag(args.bag_eval, args.num_samples, args.trim_ratio)

    print("\nStep 4/6: querying map_eval reference poses and fusion poses")
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
            "bag_eval": args.bag_eval,
            "map_gt_dir": str(map_gt_dir),
            "map_eval_dir": str(map_eval_dir),
            "localization_dir": str(localization_dir),
            "output_dir": str(output_dir),
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
        "thresholds": _threshold_stats(errors, args.thresholds_m),
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "per_sample_errors.json").write_text(json.dumps(errors, indent=2))
    np.save(output_dir / "T_map_eval_to_gt.npy", transform_map_eval_to_gt)
    np.savetxt(output_dir / "sampled_timestamps_ns.txt", np.array(timestamps, dtype=np.int64), fmt="%d")

    _plot_trajectory(errors, output_dir / "trajectory_xy.png")
    _plot_error_curve(errors, output_dir / "translation_rotation_error.png")
    _write_html_report(output_dir, metrics, errors)

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
    parser.add_argument("--bag-eval", required=True, help="ROS2 bag used for eval map and replay")
    parser.add_argument("--map-eval", help="Existing eval map directory; if omitted, built from --bag-eval")
    parser.add_argument("--output-root", default="output", help="Parent directory for timestamped benchmark folder")
    parser.add_argument("--output-dir", help="Exact output directory; overrides timestamped folder creation")
    parser.add_argument("--skip-runs", action="store_true", help="Skip map build/localization and only evaluate existing dirs")
    parser.add_argument("--skip-map-gt", action="store_true", help="Do not build map_gt")
    parser.add_argument("--skip-map-eval", action="store_true", help="Do not build map_eval")
    parser.add_argument("--skip-localization", action="store_true", help="Do not replay bag_eval against map_gt")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of sampled timestamps")
    parser.add_argument("--trim-ratio", type=float, default=0.05, help="Trim this fraction from bag start/end")
    parser.add_argument("--timestamps-file", help="Optional text file containing timestamps in ns")
    parser.add_argument("--rate", type=float, default=1.0, help="Replay/build rate")
    parser.add_argument("--timeout", type=float, default=60.0, help="Data save timeout in seconds")
    parser.add_argument("--verbose-timer", action="store_true", help="Enable verbose node timer logs")
    parser.add_argument("--max-anchor-dt-s", type=float, default=1.0, help="Max timestamp distance to anchor pose")
    parser.add_argument("--ransac-threshold-m", type=float, default=0.20, help="RANSAC inlier threshold")
    parser.add_argument("--ransac-iterations", type=int, default=1000, help="RANSAC iterations")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--evaluate-inliers-only", action="store_true", help="Only evaluate RANSAC inliers")
    parser.add_argument(
        "--thresholds-m",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.30, 0.50, 1.00],
        help="Translation thresholds reported in HTML",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
