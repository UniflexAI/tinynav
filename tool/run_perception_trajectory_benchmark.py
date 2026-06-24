#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

LOGGER = logging.getLogger("perception_trajectory_benchmark")
DEFAULT_GT_TOPIC = "/ground_truth/odometry"
DEFAULT_ESTIMATE_TOPICS = ["/slam/odometry_visual", "/slam/odometry"]
DEFAULT_PERCEPTION_CMD = "uv run python /tinynav/tinynav/core/perception_node.py"
DEFAULT_DEPTH_TOPIC = "/slam/depth"


@dataclass(frozen=True)
class Trajectory:
    topic: str
    timestamps_ns: np.ndarray
    poses: np.ndarray

    def __len__(self) -> int:
        return int(len(self.timestamps_ns))


@dataclass(frozen=True)
class EvalResult:
    topic: str
    slug: str
    timestamps_ns: np.ndarray
    gt_poses: np.ndarray
    est_aligned_poses: np.ndarray
    translation_errors_m: np.ndarray
    rotation_errors_deg: np.ndarray
    metrics: dict[str, Any]


@dataclass
class ManagedProcess:
    name: str
    proc: subprocess.Popen[Any]
    log_file: Any
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TinyNav perception on a ROS 2 bag and compare estimated VIO trajectory against GT.",
    )
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run perception + rosbag play + rosbag record, then evaluate.")
    run_parser.add_argument("--input_bag", required=True, help="Input ROS 2 bag containing stereo images, IMU, and GT odometry.")
    run_parser.add_argument("--output_dir", required=True, help="Output directory for logs, recorded bag, TUM files, metrics, and plots.")
    run_parser.add_argument("--record_bag", default="", help="Default: <output_dir>/recorded_bag")
    run_parser.add_argument("--perception_cmd", default=DEFAULT_PERCEPTION_CMD)
    run_parser.add_argument("--play_rate", type=float, default=1.0)
    run_parser.add_argument("--play_args", default="", help="Extra args appended to `ros2 bag play`.")
    run_parser.add_argument(
        "--extra_record_topic",
        action="append",
        default=None,
        help="Additional topic to include in the result bag. Repeat for multiple topics.",
    )
    run_parser.add_argument("--startup_sec", type=float, default=5.0, help="Wait after starting perception before recording.")
    run_parser.add_argument("--record_startup_sec", type=float, default=2.0, help="Wait after starting rosbag record before playback.")
    run_parser.add_argument("--shutdown_timeout_sec", type=float, default=10.0)
    run_parser.add_argument("--player_timeout_sec", type=float, default=0.0, help="0 means no timeout.")
    run_parser.add_argument("--skip_eval", action="store_true")
    run_parser.add_argument("--overwrite", action="store_true")
    add_eval_args(run_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluate an existing recorded ROS 2 bag.")
    eval_parser.add_argument("--recorded_bag", required=True, help="ROS 2 bag containing GT and estimate odometry topics.")
    eval_parser.add_argument("--output_dir", required=True, help="Output directory for TUM files, metrics, and plots.")
    eval_parser.add_argument("--overwrite", action="store_true")
    add_eval_args(eval_parser)

    offline_parser = subparsers.add_parser("offline", help="Read a ROS 2 bag directly and feed perception sequentially.")
    offline_parser.add_argument("--input_bag", required=True, help="Input ROS 2 bag containing stereo images, IMU, and GT odometry.")
    offline_parser.add_argument("--output_dir", required=True, help="Output directory for logs, recorded bag, TUM files, metrics, and plots.")
    offline_parser.add_argument("--record_bag", default="", help="Default: <output_dir>/recorded_bag")
    offline_parser.add_argument("--left_topic", default="/camera/camera/infra1/image_rect_raw")
    offline_parser.add_argument("--right_topic", default="/camera/camera/infra2/image_rect_raw")
    offline_parser.add_argument("--right_info_topic", default="/camera/camera/infra2/camera_info")
    offline_parser.add_argument("--imu_topic", default="/camera/camera/imu")
    offline_parser.add_argument("--stereo_slop_sec", type=float, default=0.02)
    offline_parser.add_argument(
        "--process_period_sec",
        type=float,
        default=0.1333,
        help="Minimum time between processed stereo frames. Use 0 to process every stereo pair.",
    )
    offline_parser.add_argument("--skip_eval", action="store_true")
    offline_parser.add_argument("--overwrite", action="store_true")
    add_eval_args(offline_parser)
    return parser.parse_args()


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gt_topic", default=DEFAULT_GT_TOPIC)
    parser.add_argument(
        "--estimate_topic",
        action="append",
        default=None,
        help="Estimated trajectory topic. Repeat for multiple topics. Default: /slam/odometry_visual and /slam/odometry.",
    )
    parser.add_argument(
        "--timestamp_source",
        choices=["auto", "header", "bag"],
        default="auto",
        help="Use message header stamp, bag timestamp, or header if non-zero else bag timestamp.",
    )
    parser.add_argument("--drop_start_sec", type=float, default=0.0, help="Drop initial matched samples before scoring.")
    parser.add_argument("--rpe_delta_sec", type=float, default=1.0)
    parser.add_argument("--rpe_tolerance_sec", type=float, default=0.2)
    parser.add_argument(
        "--alignment",
        choices=["se3", "se2", "yaw", "none"],
        default="se3",
        help="Trajectory alignment before scoring. `se2` fits yaw plus translation; `yaw` only rotates around world Z at the origin.",
    )
    parser.add_argument("--plot_max_points", type=int, default=5000)
    parser.add_argument("--depth_topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--disable_depth_summary", action="store_true")


def estimate_topics(args: argparse.Namespace) -> list[str]:
    return list(args.estimate_topic) if args.estimate_topic else list(DEFAULT_ESTIMATE_TOPICS)


def slugify_topic(topic: str) -> str:
    slug = topic.strip("/").replace("/", "_") or "root"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", slug)


def prepare_output_dir(path: Path, overwrite: bool, *, clean: bool = False) -> None:
    if path.exists() and clean:
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def guard_eval_output_does_not_contain_bag(recorded_bag: Path, output_dir: Path, overwrite: bool) -> None:
    if not overwrite:
        return
    recorded_bag = recorded_bag.resolve()
    output_dir = output_dir.resolve()
    if recorded_bag == output_dir or recorded_bag.is_relative_to(output_dir):
        raise ValueError(
            f"Refusing to --overwrite output_dir={output_dir} because it contains recorded_bag={recorded_bag}"
        )


def start_process(name: str, command: str | list[str], log_path: Path) -> ManagedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    if isinstance(command, str):
        popen_args: str | list[str] = ["bash", "-lc", command]
        printable = command
    else:
        popen_args = command
        printable = " ".join(shlex.quote(part) for part in command)
    LOGGER.info("starting %s: %s", name, printable)
    proc = subprocess.Popen(
        popen_args,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(Path.cwd()),
        preexec_fn=os.setsid,
    )
    return ManagedProcess(name=name, proc=proc, log_file=log_file, log_path=log_path)


def stop_process(process: ManagedProcess, timeout_sec: float) -> None:
    proc = process.proc
    try:
        if proc.poll() is None:
            LOGGER.info("stopping %s", process.name)
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            try:
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                LOGGER.warning("%s did not stop after SIGINT; sending SIGTERM", process.name)
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    LOGGER.warning("%s did not stop after SIGTERM; sending SIGKILL", process.name)
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=3.0)
    finally:
        process.log_file.close()


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    prepare_output_dir(output_dir, bool(args.overwrite), clean=True)
    logs_dir = output_dir / "logs"
    record_bag = Path(args.record_bag).resolve() if args.record_bag else output_dir / "recorded_bag"
    if record_bag.exists():
        if not args.overwrite:
            raise FileExistsError(f"Recorded bag exists: {record_bag}. Use --overwrite to replace it.")
        shutil.rmtree(record_bag)

    extra_record_topics = list(args.extra_record_topic) if getattr(args, "extra_record_topic", None) else []
    topics_to_record = list(dict.fromkeys([args.gt_topic, *estimate_topics(args), *extra_record_topics]))
    recorder_cmd = ["ros2", "bag", "record", "-o", str(record_bag), *topics_to_record]
    play_cmd = ["ros2", "bag", "play", str(Path(args.input_bag).resolve()), "--rate", str(float(args.play_rate))]
    if args.play_args:
        play_cmd.extend(shlex.split(args.play_args))

    perception = recorder = player = None
    try:
        perception = start_process("perception", args.perception_cmd, logs_dir / "perception.log")
        time.sleep(float(args.startup_sec))
        recorder = start_process("recorder", recorder_cmd, logs_dir / "recorder.log")
        time.sleep(float(args.record_startup_sec))
        player = start_process("rosbag_play", play_cmd, logs_dir / "rosbag_play.log")
        timeout = None if float(args.player_timeout_sec) <= 0.0 else float(args.player_timeout_sec)
        player_return = player.proc.wait(timeout=timeout)
        LOGGER.info("rosbag_play exited with code %s", player_return)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"rosbag play timed out after {args.player_timeout_sec}s") from exc
    finally:
        if player is not None:
            stop_process(player, float(args.shutdown_timeout_sec))
        if recorder is not None:
            stop_process(recorder, float(args.shutdown_timeout_sec))
        if perception is not None:
            stop_process(perception, float(args.shutdown_timeout_sec))
        time.sleep(1.0)

    run_report = {
        "input_bag": str(Path(args.input_bag).resolve()),
        "recorded_bag": str(record_bag),
        "output_dir": str(output_dir),
        "recorded_topics": topics_to_record,
        "logs_dir": str(logs_dir),
    }
    with (output_dir / "run_report.json").open("w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)

    if args.skip_eval:
        return run_report
    eval_report = evaluate_recorded_bag(record_bag, output_dir, args)
    run_report["eval"] = eval_report
    with (output_dir / "run_report.json").open("w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)
    return run_report


def bag_db_paths(bag_path: Path) -> list[Path]:
    if bag_path.is_file() and bag_path.suffix == ".db3":
        return [bag_path]
    db_paths = sorted(bag_path.glob("*.db3"))
    if not db_paths:
        raise FileNotFoundError(f"No .db3 files found in ROS 2 bag: {bag_path}")
    return db_paths


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def message_timestamp_ns(msg: Any, bag_timestamp_ns: int, source: str) -> int:
    if source == "bag":
        return int(bag_timestamp_ns)
    header_stamp = 0
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        header_stamp = stamp_to_ns(msg.header.stamp)
    if source == "header":
        return int(header_stamp)
    return int(header_stamp) if header_stamp > 0 else int(bag_timestamp_ns)


def message_to_pose(msg: Any) -> np.ndarray:
    if hasattr(msg, "pose") and hasattr(msg.pose, "pose"):
        pose_msg = msg.pose.pose
    elif hasattr(msg, "pose"):
        pose_msg = msg.pose
    elif hasattr(msg, "transform"):
        pose_msg = msg.transform
    else:
        raise TypeError(f"Unsupported pose message type: {type(msg)!r}")

    if hasattr(pose_msg, "position"):
        position = pose_msg.position
        orientation = pose_msg.orientation
    else:
        position = pose_msg.translation
        orientation = pose_msg.rotation

    quat = [orientation.x, orientation.y, orientation.z, orientation.w]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()
    T[:3, 3] = [position.x, position.y, position.z]
    return T


def load_topic_metadata(conn: sqlite3.Connection) -> dict[str, tuple[int, str]]:
    rows = conn.execute("SELECT id, name, type FROM topics").fetchall()
    return {str(name): (int(topic_id), str(type_name)) for topic_id, name, type_name in rows}


def load_rosbag_trajectories(bag_path: Path, topics: list[str], timestamp_source: str) -> dict[str, Trajectory]:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    samples: dict[str, list[tuple[int, np.ndarray]]] = {topic: [] for topic in topics}
    for db_path in bag_db_paths(bag_path):
        conn = sqlite3.connect(str(db_path))
        try:
            metadata = load_topic_metadata(conn)
            for topic in topics:
                if topic not in metadata:
                    continue
                topic_id, type_name = metadata[topic]
                msg_type = get_message(type_name)
                rows = conn.execute(
                    "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                    (topic_id,),
                )
                for bag_timestamp_ns, data in rows:
                    msg = deserialize_message(bytes(data), msg_type)
                    timestamp_ns = message_timestamp_ns(msg, int(bag_timestamp_ns), timestamp_source)
                    if timestamp_ns <= 0:
                        continue
                    samples[topic].append((timestamp_ns, message_to_pose(msg)))
        finally:
            conn.close()

    trajectories: dict[str, Trajectory] = {}
    for topic, topic_samples in samples.items():
        if not topic_samples:
            continue
        topic_samples.sort(key=lambda item: item[0])
        timestamps = np.asarray([item[0] for item in topic_samples], dtype=np.int64)
        poses = np.stack([item[1] for item in topic_samples], axis=0)
        timestamps, unique_indices = np.unique(timestamps, return_index=True)
        trajectories[topic] = Trajectory(topic=topic, timestamps_ns=timestamps, poses=poses[unique_indices])
    return trajectories


class CapturedPublisher:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.messages: list[tuple[int, Any]] = []

    def publish(self, msg: Any) -> None:
        timestamp_ns = 0
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            timestamp_ns = stamp_to_ns(msg.header.stamp)
        self.messages.append((int(timestamp_ns), msg))


def read_selected_bag_messages(bag_path: Path, topic_names: set[str]) -> list[tuple[int, str, Any]]:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    selected: list[tuple[int, str, Any]] = []
    for db_path in bag_db_paths(bag_path):
        conn = sqlite3.connect(str(db_path))
        try:
            metadata = load_topic_metadata(conn)
            topic_ids = {topic_id: (topic, get_message(type_name)) for topic, (topic_id, type_name) in metadata.items() if topic in topic_names}
            if not topic_ids:
                continue
            placeholders = ",".join("?" for _ in topic_ids)
            rows = conn.execute(
                f"SELECT timestamp, topic_id, data FROM messages WHERE topic_id IN ({placeholders}) ORDER BY timestamp",
                tuple(topic_ids),
            )
            for bag_timestamp_ns, topic_id, data in rows:
                topic, msg_type = topic_ids[int(topic_id)]
                selected.append((int(bag_timestamp_ns), topic, deserialize_message(bytes(data), msg_type)))
        finally:
            conn.close()
    selected.sort(key=lambda item: item[0])
    return selected


def pair_stereo_messages(left_messages: list[tuple[int, Any]], right_messages: list[tuple[int, Any]], slop_sec: float) -> list[tuple[int, Any, Any]]:
    pairs: list[tuple[int, Any, Any]] = []
    right_index = 0
    used_right: set[int] = set()
    slop_ns = int(round(float(slop_sec) * 1e9))
    for left_bag_ns, left_msg in left_messages:
        left_ns = stamp_to_ns(left_msg.header.stamp)
        best_index = -1
        best_delta = slop_ns + 1
        while right_index < len(right_messages) and stamp_to_ns(right_messages[right_index][1].header.stamp) < left_ns - slop_ns:
            right_index += 1
        scan_index = right_index
        while scan_index < len(right_messages):
            right_ns = stamp_to_ns(right_messages[scan_index][1].header.stamp)
            if right_ns > left_ns + slop_ns:
                break
            if scan_index not in used_right:
                delta = abs(right_ns - left_ns)
                if delta < best_delta:
                    best_delta = delta
                    best_index = scan_index
            scan_index += 1
        if best_index >= 0:
            used_right.add(best_index)
            pairs.append((left_bag_ns, left_msg, right_messages[best_index][1]))
    return pairs


def write_offline_recorded_bag(
    record_bag: Path,
    gt_topic: str,
    gt_messages: list[tuple[int, Any]],
    publishers: dict[str, CapturedPublisher],
) -> None:
    from rclpy.serialization import serialize_message
    from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata

    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=str(record_bag), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    writer.create_topic(TopicMetadata(name=gt_topic, type="nav_msgs/msg/Odometry", serialization_format="cdr"))
    writer.create_topic(TopicMetadata(name="/slam/odometry_visual", type="nav_msgs/msg/Odometry", serialization_format="cdr"))
    writer.create_topic(TopicMetadata(name="/slam/depth", type="sensor_msgs/msg/Image", serialization_format="cdr"))

    records: list[tuple[int, str, Any]] = []
    records.extend((stamp_to_ns(msg.header.stamp), gt_topic, msg) for _, msg in gt_messages)
    records.extend((timestamp_ns, "/slam/odometry_visual", msg) for timestamp_ns, msg in publishers["odom"].messages)
    records.extend((timestamp_ns, "/slam/depth", msg) for timestamp_ns, msg in publishers["depth"].messages)
    records.sort(key=lambda item: item[0])
    for timestamp_ns, topic, msg in records:
        writer.write(topic, serialize_message(msg), int(timestamp_ns))


def run_offline_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    prepare_output_dir(output_dir, bool(args.overwrite), clean=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    record_bag = Path(args.record_bag).resolve() if args.record_bag else output_dir / "recorded_bag"
    if record_bag.exists():
        if not args.overwrite:
            raise FileExistsError(f"Recorded bag exists: {record_bag}. Use --overwrite to replace it.")
        shutil.rmtree(record_bag)

    import rclpy
    from tinynav.core.perception_node import PerceptionNode

    topics = {args.left_topic, args.right_topic, args.right_info_topic, args.imu_topic, args.gt_topic}
    messages = read_selected_bag_messages(Path(args.input_bag).resolve(), topics)
    left_messages: list[tuple[int, Any]] = []
    right_messages: list[tuple[int, Any]] = []
    imu_messages: list[tuple[int, Any]] = []
    info_messages: list[tuple[int, Any]] = []
    gt_messages: list[tuple[int, Any]] = []
    for bag_timestamp_ns, topic, msg in messages:
        if topic == args.left_topic:
            left_messages.append((bag_timestamp_ns, msg))
        elif topic == args.right_topic:
            right_messages.append((bag_timestamp_ns, msg))
        elif topic == args.imu_topic:
            imu_messages.append((bag_timestamp_ns, msg))
        elif topic == args.right_info_topic:
            info_messages.append((bag_timestamp_ns, msg))
        elif topic == args.gt_topic:
            gt_messages.append((bag_timestamp_ns, msg))

    stereo_pairs = pair_stereo_messages(left_messages, right_messages, float(args.stereo_slop_sec))
    if not stereo_pairs:
        raise RuntimeError(f"No stereo pairs found for {args.left_topic} and {args.right_topic}")
    if not info_messages:
        raise RuntimeError(f"No camera info messages found for {args.right_info_topic}")
    if not imu_messages:
        raise RuntimeError(f"No IMU messages found for {args.imu_topic}")
    if not gt_messages:
        raise RuntimeError(f"No GT odometry messages found for {args.gt_topic}")

    rclpy.init(args=None)
    perception = PerceptionNode()
    publishers = {
        "odom": CapturedPublisher("/slam/odometry_visual"),
        "depth": CapturedPublisher("/slam/depth"),
        "slam_camera_info": CapturedPublisher("/slam/camera_info"),
        "disparity": CapturedPublisher("/slam/disparity_vis"),
        "keyframe_pose": CapturedPublisher("/slam/keyframe_odom"),
        "keyframe_image": CapturedPublisher("/slam/keyframe_image"),
        "keyframe_depth": CapturedPublisher("/slam/keyframe_depth"),
        "stats": CapturedPublisher("/slam/data"),
    }
    perception.odom_pub = publishers["odom"]
    perception.depth_pub = publishers["depth"]
    perception.slam_camera_info_pub = publishers["slam_camera_info"]
    perception.disparity_pub_vis = publishers["disparity"]
    perception.keyframe_pose_pub = publishers["keyframe_pose"]
    perception.keyframe_image_pub = publishers["keyframe_image"]
    perception.keyframe_depth_pub = publishers["keyframe_depth"]
    perception.stats_pub = publishers["stats"]
    perception.tf_broadcaster.sendTransform = lambda *_args, **_kwargs: None

    imu_index = 0
    processed = 0
    skipped_by_period = 0
    last_processed_stereo_ns: int | None = None
    process_period_ns = max(0, int(round(float(args.process_period_sec) * 1e9)))
    try:
        perception.info_callback(info_messages[0][1])
        for _, left_msg, right_msg in stereo_pairs:
            stereo_ns = stamp_to_ns(left_msg.header.stamp)
            while imu_index < len(imu_messages) and stamp_to_ns(imu_messages[imu_index][1].header.stamp) <= stereo_ns:
                perception._process_imu_msg(imu_messages[imu_index][1])
                imu_index += 1
            if perception.T_body_last is None:
                continue
            if (
                process_period_ns > 0
                and last_processed_stereo_ns is not None
                and stereo_ns - last_processed_stereo_ns < process_period_ns
            ):
                skipped_by_period += 1
                continue
            perception._async_loop.run_until_complete(perception.process(left_msg, right_msg))
            last_processed_stereo_ns = stereo_ns
            processed += 1
    finally:
        perception.destroy_node()
        rclpy.shutdown()

    write_offline_recorded_bag(record_bag, args.gt_topic, gt_messages, publishers)
    run_report = {
        "mode": "offline",
        "input_bag": str(Path(args.input_bag).resolve()),
        "recorded_bag": str(record_bag),
        "output_dir": str(output_dir),
        "logs_dir": str(logs_dir),
        "input_counts": {
            "left_images": len(left_messages),
            "right_images": len(right_messages),
            "stereo_pairs": len(stereo_pairs),
            "imu": len(imu_messages),
            "ground_truth": len(gt_messages),
        },
        "processed_stereo_pairs": processed,
        "skipped_by_process_period": skipped_by_period,
        "process_period_sec": float(args.process_period_sec),
        "captured_counts": {key: len(pub.messages) for key, pub in publishers.items()},
    }
    with (output_dir / "run_report.json").open("w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)

    if args.skip_eval:
        return run_report
    eval_report = evaluate_recorded_bag(record_bag, output_dir, args)
    run_report["eval"] = eval_report
    with (output_dir / "run_report.json").open("w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)
    return run_report


def matrix_to_tum_row(timestamp_ns: int, pose: np.ndarray) -> list[float]:
    quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
    xyz = pose[:3, 3]
    return [timestamp_ns * 1e-9, xyz[0], xyz[1], xyz[2], quat[0], quat[1], quat[2], quat[3]]


def write_tum(path: Path, trajectory: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for timestamp_ns, pose in zip(trajectory.timestamps_ns, trajectory.poses):
            row = matrix_to_tum_row(int(timestamp_ns), pose)
            f.write("{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(*row))


def write_error_csv(path: Path, result: EvalResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "timestamp_sec", "translation_error_m", "rotation_error_deg"])
        for timestamp_ns, trans, rot in zip(result.timestamps_ns, result.translation_errors_m, result.rotation_errors_deg):
            writer.writerow([int(timestamp_ns), f"{timestamp_ns * 1e-9:.9f}", f"{trans:.9f}", f"{rot:.9f}"])


def image_msg_to_depth_array(msg: Any) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    height = int(msg.height)
    width = int(msg.width)
    if encoding in {"32fc1", "32fc"}:
        dtype = np.dtype(">f4" if msg.is_bigendian else "<f4")
        image = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(height, int(msg.step) // 4)[:, :width]
        return image.astype(np.float32, copy=False)
    if encoding in {"16uc1", "mono16"}:
        dtype = np.dtype(">u2" if msg.is_bigendian else "<u2")
        image = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(height, int(msg.step) // 2)[:, :width]
        return image.astype(np.float32) * 0.001
    raise ValueError(f"Unsupported depth image encoding: {msg.encoding}")


def summarize_depth_bag(recorded_bag: Path, topic: str, output_dir: Path, timestamp_source: str) -> dict[str, Any] | None:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    rows_out: list[dict[str, Any]] = []
    for db_path in bag_db_paths(recorded_bag):
        conn = sqlite3.connect(str(db_path))
        try:
            metadata = load_topic_metadata(conn)
            if topic not in metadata:
                continue
            topic_id, type_name = metadata[topic]
            msg_type = get_message(type_name)
            rows = conn.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                (topic_id,),
            )
            for bag_timestamp_ns, data in rows:
                msg = deserialize_message(bytes(data), msg_type)
                timestamp_ns = message_timestamp_ns(msg, int(bag_timestamp_ns), timestamp_source)
                depth = image_msg_to_depth_array(msg)
                finite = np.isfinite(depth)
                positive = finite & (depth > 0.0)
                finite_depth = depth[finite]
                positive_depth = depth[positive]
                rows_out.append(
                    {
                        "timestamp_ns": int(timestamp_ns),
                        "timestamp_sec": float(timestamp_ns * 1e-9),
                        "width": int(depth.shape[1]),
                        "height": int(depth.shape[0]),
                        "finite_ratio": float(np.count_nonzero(finite) / depth.size),
                        "positive_ratio": float(np.count_nonzero(positive) / depth.size),
                        "finite_mean_m": float(np.mean(finite_depth)) if len(finite_depth) else 0.0,
                        "positive_mean_m": float(np.mean(positive_depth)) if len(positive_depth) else 0.0,
                        "positive_median_m": float(np.median(positive_depth)) if len(positive_depth) else 0.0,
                        "positive_p90_m": float(np.percentile(positive_depth, 90)) if len(positive_depth) else 0.0,
                        "positive_max_m": float(np.max(positive_depth)) if len(positive_depth) else 0.0,
                    }
                )
        finally:
            conn.close()

    if not rows_out:
        return None

    rows_out.sort(key=lambda row: int(row["timestamp_ns"]))
    csv_path = output_dir / f"{slugify_topic(topic)}_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    times = np.asarray([row["timestamp_sec"] for row in rows_out], dtype=np.float64)
    times -= times[0]
    positive_ratio = np.asarray([row["positive_ratio"] for row in rows_out], dtype=np.float64)
    positive_median = np.asarray([row["positive_median_m"] for row in rows_out], dtype=np.float64)
    positive_p90 = np.asarray([row["positive_p90_m"] for row in rows_out], dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(times, positive_ratio, linewidth=1.2)
    axes[0].set_ylabel("positive ratio")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(times, positive_median, linewidth=1.2, label="median")
    axes[1].plot(times, positive_p90, linewidth=1.2, label="p90")
    axes[1].set_xlabel("time from first depth [s]")
    axes[1].set_ylabel("depth [m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(f"Depth summary: {topic}")
    fig.tight_layout()
    png_path = output_dir / f"{slugify_topic(topic)}_summary.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    positive_ratios = np.asarray([row["positive_ratio"] for row in rows_out], dtype=np.float64)
    positive_means = np.asarray([row["positive_mean_m"] for row in rows_out], dtype=np.float64)
    positive_medians = np.asarray([row["positive_median_m"] for row in rows_out], dtype=np.float64)
    positive_p90s = np.asarray([row["positive_p90_m"] for row in rows_out], dtype=np.float64)
    summary = {
        "topic": topic,
        "frames": int(len(rows_out)),
        "csv": str(csv_path),
        "plot": str(png_path),
        "positive_ratio_mean": float(np.mean(positive_ratios)),
        "positive_ratio_min": float(np.min(positive_ratios)),
        "positive_mean_m_mean": float(np.mean(positive_means)),
        "positive_median_m_mean": float(np.mean(positive_medians)),
        "positive_p90_m_mean": float(np.mean(positive_p90s)),
        "first_timestamp_ns": int(rows_out[0]["timestamp_ns"]),
        "last_timestamp_ns": int(rows_out[-1]["timestamp_ns"]),
    }
    return summary


def transform_inverse(T: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def interpolate_trajectory(reference: Trajectory, target_timestamps_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(reference) < 2:
        raise ValueError(f"Need at least two GT poses for interpolation, got {len(reference)}")
    ref_t = reference.timestamps_ns.astype(np.float64) * 1e-9
    target_t = target_timestamps_ns.astype(np.float64) * 1e-9
    valid = (target_t >= ref_t[0]) & (target_t <= ref_t[-1])
    if not np.any(valid):
        return np.empty((0, 4, 4), dtype=np.float64), valid

    valid_t = target_t[valid]
    positions = np.column_stack(
        [np.interp(valid_t, ref_t, reference.poses[:, dim, 3]) for dim in range(3)]
    )
    rotations = Slerp(ref_t, Rotation.from_matrix(reference.poses[:, :3, :3]))(valid_t).as_matrix()
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(valid_t), axis=0)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = positions
    return poses, valid


def align_se3_no_scale(source_positions: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
    if len(source_positions) < 3:
        raise ValueError(f"Need at least 3 pose pairs for SE3 alignment, got {len(source_positions)}")
    source_centroid = np.mean(source_positions, axis=0)
    target_centroid = np.mean(target_positions, axis=0)
    source_centered = source_positions - source_centroid
    target_centered = target_positions - target_centroid
    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = target_centroid - R @ source_centroid
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def align_yaw_about_origin(source_positions: np.ndarray, target_positions: np.ndarray) -> tuple[np.ndarray, float]:
    if len(source_positions) < 2:
        raise ValueError(f"Need at least 2 pose pairs for yaw alignment, got {len(source_positions)}")
    source_xy = source_positions[:, :2]
    target_xy = target_positions[:, :2]
    a = float(np.sum(target_xy[:, 0] * source_xy[:, 0] + target_xy[:, 1] * source_xy[:, 1]))
    b = float(np.sum(target_xy[:, 1] * source_xy[:, 0] - target_xy[:, 0] * source_xy[:, 1]))
    yaw = float(np.arctan2(b, a))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T, yaw


def align_yaw_with_translation(source_positions: np.ndarray, target_positions: np.ndarray) -> tuple[np.ndarray, float]:
    if len(source_positions) < 3:
        raise ValueError(f"Need at least 3 pose pairs for SE2 alignment, got {len(source_positions)}")
    source_centroid = np.mean(source_positions, axis=0)
    target_centroid = np.mean(target_positions, axis=0)
    source_centered = source_positions - source_centroid
    target_centered = target_positions - target_centroid
    T, yaw = align_yaw_about_origin(source_centered, target_centered)
    T[:3, 3] = target_centroid - T[:3, :3] @ source_centroid
    return T, yaw


def trajectory_alignment_transform(
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "se3":
        T = align_se3_no_scale(source_positions, target_positions)
        return T, {"mode": mode, "transform": T.tolist()}
    if mode == "se2":
        T, yaw = align_yaw_with_translation(source_positions, target_positions)
        return T, {"mode": mode, "transform": T.tolist(), "yaw_rad": yaw, "yaw_deg": float(np.rad2deg(yaw))}
    if mode == "yaw":
        T, yaw = align_yaw_about_origin(source_positions, target_positions)
        return T, {"mode": mode, "transform": T.tolist(), "yaw_rad": yaw, "yaw_deg": float(np.rad2deg(yaw))}
    if mode == "none":
        T = np.eye(4, dtype=np.float64)
        return T, {"mode": mode, "transform": T.tolist()}
    raise ValueError(f"Unsupported alignment mode: {mode}")


def apply_left_transform(transform: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njk->nik", transform, poses)


def rotation_errors_deg(gt_poses: np.ndarray, est_poses: np.ndarray) -> np.ndarray:
    R_delta = np.einsum("nji,njk->nik", gt_poses[:, :3, :3], est_poses[:, :3, :3])
    return np.rad2deg(Rotation.from_matrix(R_delta).magnitude())


def basic_stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {key: 0.0 for key in ["rmse", "mean", "median", "p90", "p95", "max", "final"]}
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "final": float(values[-1]),
    }


def path_length(poses: np.ndarray) -> float:
    if len(poses) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)))


def compute_rpe(
    gt_poses: np.ndarray,
    est_poses: np.ndarray,
    timestamps_ns: np.ndarray,
    delta_sec: float,
    tolerance_sec: float,
) -> dict[str, Any]:
    times = timestamps_ns.astype(np.float64) * 1e-9
    trans_errors: list[float] = []
    rot_errors: list[float] = []
    for i, timestamp in enumerate(times):
        target = timestamp + delta_sec
        idx = int(np.searchsorted(times, target))
        candidates = []
        if idx < len(times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            continue
        j = min(candidates, key=lambda candidate: abs(times[candidate] - target))
        if j <= i or abs(times[j] - target) > tolerance_sec:
            continue
        gt_rel = transform_inverse(gt_poses[i]) @ gt_poses[j]
        est_rel = transform_inverse(est_poses[i]) @ est_poses[j]
        error = transform_inverse(gt_rel) @ est_rel
        trans_errors.append(float(np.linalg.norm(error[:3, 3])))
        rot_errors.append(float(np.rad2deg(Rotation.from_matrix(error[:3, :3]).magnitude())))
    trans = np.asarray(trans_errors, dtype=np.float64)
    rot = np.asarray(rot_errors, dtype=np.float64)
    return {
        "delta_sec": float(delta_sec),
        "tolerance_sec": float(tolerance_sec),
        "pairs": int(len(trans)),
        "translation_m": basic_stats(trans),
        "rotation_deg": basic_stats(rot),
    }


def evaluate_pair(
    gt: Trajectory,
    estimate: Trajectory,
    *,
    drop_start_sec: float,
    rpe_delta_sec: float,
    rpe_tolerance_sec: float,
    alignment: str,
) -> EvalResult:
    gt_interp, valid_mask = interpolate_trajectory(gt, estimate.timestamps_ns)
    est_timestamps = estimate.timestamps_ns[valid_mask]
    est_poses = estimate.poses[valid_mask]
    if len(est_poses) < 3:
        raise ValueError(f"{estimate.topic}: only {len(est_poses)} estimate poses overlap GT")

    if drop_start_sec > 0.0:
        start_time = est_timestamps[0] + int(round(drop_start_sec * 1e9))
        keep = est_timestamps >= start_time
        est_timestamps = est_timestamps[keep]
        est_poses = est_poses[keep]
        gt_interp = gt_interp[keep]
    if len(est_poses) < 3:
        raise ValueError(f"{estimate.topic}: only {len(est_poses)} poses remain after --drop_start_sec")

    T_gt_from_est, alignment_info = trajectory_alignment_transform(est_poses[:, :3, 3], gt_interp[:, :3, 3], alignment)
    est_aligned = apply_left_transform(T_gt_from_est, est_poses)
    trans_errors = np.linalg.norm(gt_interp[:, :3, 3] - est_aligned[:, :3, 3], axis=1)
    rot_errors = rotation_errors_deg(gt_interp, est_aligned)
    gt_distance = path_length(gt_interp)
    est_distance = path_length(est_aligned)
    rpe = compute_rpe(gt_interp, est_aligned, est_timestamps, rpe_delta_sec, rpe_tolerance_sec)
    metrics = {
        "topic": estimate.topic,
        "samples": int(len(est_timestamps)),
        "timestamp_start_ns": int(est_timestamps[0]),
        "timestamp_end_ns": int(est_timestamps[-1]),
        "duration_sec": float((est_timestamps[-1] - est_timestamps[0]) * 1e-9),
        "gt_path_length_m": gt_distance,
        "estimate_path_length_m": est_distance,
        "alignment": alignment_info,
        "se3_alignment": T_gt_from_est.tolist() if alignment == "se3" else None,
        "ape_translation_m": basic_stats(trans_errors),
        "ape_rotation_deg": basic_stats(rot_errors),
        "rpe": rpe,
        "final_drift_percent_of_gt_path": float(100.0 * trans_errors[-1] / gt_distance) if gt_distance > 0.0 else 0.0,
        "rmse_percent_of_gt_path": float(100.0 * np.sqrt(np.mean(trans_errors**2)) / gt_distance) if gt_distance > 0.0 else 0.0,
    }
    return EvalResult(
        topic=estimate.topic,
        slug=slugify_topic(estimate.topic),
        timestamps_ns=est_timestamps,
        gt_poses=gt_interp,
        est_aligned_poses=est_aligned,
        translation_errors_m=trans_errors,
        rotation_errors_deg=rot_errors,
        metrics=metrics,
    )


def downsample_indices(count: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or count <= max_points:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_points, dtype=np.int64)


def plot_trajectory_xy(output_path: Path, gt: Trajectory, results: list[EvalResult], max_points: int, alignment: str) -> None:
    plt.figure(figsize=(10, 8))
    gt_idx = downsample_indices(len(gt), max_points)
    plt.plot(gt.poses[gt_idx, 0, 3], gt.poses[gt_idx, 1, 3], color="black", linewidth=1.5, label="GT")
    for result in results:
        idx = downsample_indices(len(result.timestamps_ns), max_points)
        plt.plot(
            result.est_aligned_poses[idx, 0, 3],
            result.est_aligned_poses[idx, 1, 3],
            linewidth=1.2,
            label=result.topic,
        )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"TinyNav VIO trajectory after {alignment} alignment")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_trajectory_3d(output_path: Path, gt: Trajectory, results: list[EvalResult], max_points: int, alignment: str) -> None:
    fig = plt.figure(figsize=(10, 8))
    try:
        ax = fig.add_subplot(111, projection="3d")
    except ValueError:
        plt.close(fig)
        plot_trajectory_xz_fallback(output_path, gt, results, max_points, alignment)
        return
    gt_idx = downsample_indices(len(gt), max_points)
    ax.plot(gt.poses[gt_idx, 0, 3], gt.poses[gt_idx, 1, 3], gt.poses[gt_idx, 2, 3], color="black", linewidth=1.5, label="GT")
    for result in results:
        idx = downsample_indices(len(result.timestamps_ns), max_points)
        ax.plot(
            result.est_aligned_poses[idx, 0, 3],
            result.est_aligned_poses[idx, 1, 3],
            result.est_aligned_poses[idx, 2, 3],
            linewidth=1.2,
            label=result.topic,
        )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"TinyNav VIO trajectory after {alignment} alignment")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trajectory_xz_fallback(output_path: Path, gt: Trajectory, results: list[EvalResult], max_points: int, alignment: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    gt_idx = downsample_indices(len(gt), max_points)
    axes[0].plot(gt.poses[gt_idx, 0, 3], gt.poses[gt_idx, 1, 3], color="black", linewidth=1.5, label="GT")
    axes[1].plot(gt.poses[gt_idx, 0, 3], gt.poses[gt_idx, 2, 3], color="black", linewidth=1.5, label="GT")
    for result in results:
        idx = downsample_indices(len(result.timestamps_ns), max_points)
        axes[0].plot(result.est_aligned_poses[idx, 0, 3], result.est_aligned_poses[idx, 1, 3], linewidth=1.2, label=result.topic)
        axes[1].plot(result.est_aligned_poses[idx, 0, 3], result.est_aligned_poses[idx, 2, 3], linewidth=1.2, label=result.topic)
    axes[0].set_title("XY trajectory")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[1].set_title("XZ trajectory")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("z [m]")
    for ax in axes:
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"3D projection unavailable; showing 2D trajectory projections after {alignment} alignment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_series(output_path: Path, results: list[EvalResult], *, rotation: bool) -> None:
    plt.figure(figsize=(11, 5))
    for result in results:
        t = (result.timestamps_ns - result.timestamps_ns[0]).astype(np.float64) * 1e-9
        y = result.rotation_errors_deg if rotation else result.translation_errors_m
        plt.plot(t, y, linewidth=1.1, label=result.topic)
    plt.xlabel("time from first matched estimate [s]")
    plt.ylabel("rotation APE [deg]" if rotation else "translation APE [m]")
    plt.title("Rotation APE" if rotation else "Translation APE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_result_outputs(
    output_dir: Path,
    gt: Trajectory,
    estimates: dict[str, Trajectory],
    results: list[EvalResult],
    plot_max_points: int,
    alignment: str,
) -> None:
    write_tum(output_dir / "ground_truth.tum", gt)
    for topic, estimate in estimates.items():
        write_tum(output_dir / f"{slugify_topic(topic)}.tum", estimate)
    for result in results:
        write_tum(output_dir / f"{result.slug}_aligned.tum", Trajectory(result.topic, result.timestamps_ns, result.est_aligned_poses))
        write_tum(output_dir / f"{result.slug}_matched_gt.tum", Trajectory("matched_gt", result.timestamps_ns, result.gt_poses))
        write_error_csv(output_dir / f"{result.slug}_errors.csv", result)
    if results:
        plot_trajectory_xy(output_dir / "trajectory_xy.png", gt, results, plot_max_points, alignment)
        plot_trajectory_3d(output_dir / "trajectory_3d.png", gt, results, plot_max_points, alignment)
        plot_error_series(output_dir / "ape_translation_over_time.png", results, rotation=False)
        plot_error_series(output_dir / "ape_rotation_over_time.png", results, rotation=True)


def evaluate_recorded_bag(recorded_bag: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    prepare_output_dir(output_dir, bool(getattr(args, "overwrite", False)), clean=False)
    topics = [args.gt_topic, *estimate_topics(args)]
    trajectories = load_rosbag_trajectories(recorded_bag, topics, args.timestamp_source)
    if args.gt_topic not in trajectories:
        present = sorted(trajectories)
        raise RuntimeError(f"Missing GT topic {args.gt_topic} in {recorded_bag}. Loaded topics: {present}")
    gt = trajectories[args.gt_topic]
    estimate_trajectories = {topic: trajectories[topic] for topic in estimate_topics(args) if topic in trajectories}
    missing_estimates = [topic for topic in estimate_topics(args) if topic not in trajectories]
    if not estimate_trajectories:
        raise RuntimeError(f"No estimate topics found in {recorded_bag}; requested {estimate_topics(args)}")

    results: list[EvalResult] = []
    failures: dict[str, str] = {}
    for topic, estimate in estimate_trajectories.items():
        try:
            result = evaluate_pair(
                gt,
                estimate,
                drop_start_sec=float(args.drop_start_sec),
                rpe_delta_sec=float(args.rpe_delta_sec),
                rpe_tolerance_sec=float(args.rpe_tolerance_sec),
                alignment=str(args.alignment),
            )
            results.append(result)
        except Exception as exc:  # keep other estimate topics evaluable
            failures[topic] = str(exc)
            LOGGER.warning("failed to evaluate %s: %s", topic, exc)

    if not results:
        raise RuntimeError(f"No estimate trajectory could be evaluated. Failures: {failures}")

    write_result_outputs(output_dir, gt, estimate_trajectories, results, int(args.plot_max_points), str(args.alignment))
    depth_summary = None
    if not getattr(args, "disable_depth_summary", False):
        depth_summary = summarize_depth_bag(recorded_bag, args.depth_topic, output_dir, args.timestamp_source)
    report = {
        "recorded_bag": str(recorded_bag.resolve()),
        "output_dir": str(output_dir.resolve()),
        "timestamp_source": args.timestamp_source,
        "gt_topic": args.gt_topic,
        "estimate_topics": estimate_topics(args),
        "alignment": args.alignment,
        "missing_estimate_topics": missing_estimates,
        "failed_estimate_topics": failures,
        "trajectory_counts": {topic: len(traj) for topic, traj in trajectories.items()},
        "depth_summary": depth_summary,
        "metrics": {result.topic: result.metrics for result in results},
        "outputs": {
            "ground_truth_tum": str(output_dir / "ground_truth.tum"),
            "trajectory_xy_png": str(output_dir / "trajectory_xy.png"),
            "trajectory_3d_png": str(output_dir / "trajectory_3d.png"),
            "ape_translation_png": str(output_dir / "ape_translation_over_time.png"),
            "ape_rotation_png": str(output_dir / "ape_rotation_over_time.png"),
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print_summary(report)
    return report


def print_summary(report: dict[str, Any]) -> None:
    print(json.dumps({"output_dir": report["output_dir"], "counts": report["trajectory_counts"]}, indent=2))
    for topic, metrics in report["metrics"].items():
        ape_t = metrics["ape_translation_m"]
        ape_r = metrics["ape_rotation_deg"]
        rpe_t = metrics["rpe"]["translation_m"]
        print(
            f"{topic}: samples={metrics['samples']} "
            f"APE trans rmse/median/p90={ape_t['rmse']:.4f}/{ape_t['median']:.4f}/{ape_t['p90']:.4f} m, "
            f"APE rot rmse/median/p90={ape_r['rmse']:.3f}/{ape_r['median']:.3f}/{ape_r['p90']:.3f} deg, "
            f"RPE trans rmse={rpe_t['rmse']:.4f} m"
        )
    if report.get("missing_estimate_topics"):
        print(f"missing estimate topics: {report['missing_estimate_topics']}")
    if report.get("failed_estimate_topics"):
        print(f"failed estimate topics: {report['failed_estimate_topics']}")
    if report.get("depth_summary"):
        depth = report["depth_summary"]
        print(
            f"{depth['topic']}: frames={depth['frames']} "
            f"positive_ratio mean/min={depth['positive_ratio_mean']:.3f}/{depth['positive_ratio_min']:.3f}, "
            f"depth mean/median/p90={depth['positive_mean_m_mean']:.3f}/{depth['positive_median_m_mean']:.3f}/{depth['positive_p90_m_mean']:.3f} m"
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")
    if args.command == "run":
        run_pipeline(args)
    elif args.command == "offline":
        run_offline_pipeline(args)
    elif args.command == "eval":
        output_dir = Path(args.output_dir).resolve()
        guard_eval_output_does_not_contain_bag(Path(args.recorded_bag), output_dir, bool(args.overwrite))
        prepare_output_dir(output_dir, bool(args.overwrite), clean=bool(args.overwrite))
        evaluate_recorded_bag(Path(args.recorded_bag).resolve(), output_dir, args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
