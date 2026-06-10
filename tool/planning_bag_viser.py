from __future__ import annotations

import argparse
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import viser
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.ndimage import distance_transform_edt
from tinynav.core.planning_node import (
    GO2_CONFIG,
    ObstacleConfig,
    build_obstacle_map,
    generate_predefined_trajectory_vocabularies,
    generate_trajectory_library_3d,
    roll_occupancy_grid,
    run_raycasting_loopy,
    score_trajectories_by_ESDF,
)


PLANNING_ODOM_TOPIC = "/insight/vio_20hz"
LEGACY_PLANNING_ODOM_TOPIC = "/slam/odometry_visual"
PLANNING_DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
LEGACY_PLANNING_DEPTH_TOPIC = "/slam/depth"
PLANNING_PATH_TOPIC = "/planning/trajectory_path"

PLANNING_INPUT_TOPICS = {
    PLANNING_DEPTH_TOPIC,
    PLANNING_ODOM_TOPIC,
    PLANNING_PATH_TOPIC,
    "/camera/camera/infra2/camera_info",
    "/control/target_pose",
    "/mapping/poi_change",
}
READABLE_PLANNING_TOPICS = PLANNING_INPUT_TOPICS | {LEGACY_PLANNING_ODOM_TOPIC, LEGACY_PLANNING_DEPTH_TOPIC}


@dataclass
class PoseFrame:
    timestamp_ns: int
    header_timestamp_ns: int | None
    position: np.ndarray
    wxyz: np.ndarray
    matrix: np.ndarray


@dataclass
class TargetFrame:
    timestamp_ns: int
    position: np.ndarray
    wxyz: np.ndarray


@dataclass
class TargetEvent:
    timestamp_ns: int
    target: TargetFrame | None


@dataclass
class DepthFrame:
    timestamp_ns: int
    points: np.ndarray
    colors: np.ndarray


@dataclass
class RaycastInputFrame:
    timestamp_ns: int
    depth: np.ndarray
    pose: PoseFrame
    k: np.ndarray


@dataclass
class OccupancyFrame:
    timestamp_ns: int
    points: np.ndarray
    colors: np.ndarray
    obstacle_points: np.ndarray
    obstacle_mask: np.ndarray
    front_clearance: float
    origin: np.ndarray
    grid: np.ndarray
    pose: PoseFrame


@dataclass
class TrajectoryFrame:
    timestamp_ns: int
    candidates: list[np.ndarray]
    selected: np.ndarray | None
    selected_param: np.ndarray | None
    status: str


@dataclass
class RecordedPathFrame:
    timestamp_ns: int
    header_timestamp_ns: int | None
    points: np.ndarray
    first_stamp_ns: int | None


@dataclass
class BagData:
    poses: list[PoseFrame]
    targets: list[TargetFrame]
    target_events: list[TargetEvent]
    depth_frames: list[DepthFrame]
    raycast_inputs: list[RaycastInputFrame]
    occupancy_frames: list[OccupancyFrame]
    trajectory_frames: list[TrajectoryFrame]
    recorded_path_frames: list[RecordedPathFrame]
    topics: dict[str, str]
    start_ns: int | None
    end_ns: int | None


def _quat_xyzw_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _header_stamp_ns(msg) -> int | None:
    if not hasattr(msg, "header"):
        return None
    stamp_ns = _stamp_to_ns(msg.header.stamp)
    return stamp_ns if stamp_ns > 0 else None


def _format_stamp(stamp_ns: int | None) -> str:
    if stamp_ns is None:
        return "none"
    return f"{stamp_ns / 1e9:.6f}"


def _odom_to_pose_frame(timestamp_ns: int, msg) -> PoseFrame:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    R = _quat_xyzw_to_matrix(q.x, q.y, q.z, q.w)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([p.x, p.y, p.z], dtype=np.float64)
    return PoseFrame(
        timestamp_ns=timestamp_ns,
        header_timestamp_ns=_header_stamp_ns(msg),
        position=T[:3, 3].copy(),
        wxyz=np.array([q.w, q.x, q.y, q.z], dtype=np.float64),
        matrix=T,
    )


def _pose_stamped_to_pose_frame(timestamp_ns: int, msg) -> PoseFrame:
    p = msg.pose.position
    q = msg.pose.orientation
    R = _quat_xyzw_to_matrix(q.x, q.y, q.z, q.w)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([p.x, p.y, p.z], dtype=np.float64)
    return PoseFrame(
        timestamp_ns=timestamp_ns,
        header_timestamp_ns=_header_stamp_ns(msg),
        position=T[:3, 3].copy(),
        wxyz=np.array([q.w, q.x, q.y, q.z], dtype=np.float64),
        matrix=T,
    )


def _path_to_frame(timestamp_ns: int, msg) -> RecordedPathFrame:
    points = []
    first_stamp_ns = None
    for idx, pose_stamped in enumerate(msg.poses):
        p = pose_stamped.pose.position
        points.append([p.x, p.y, p.z])
        if idx == 0:
            first_stamp_ns = _stamp_to_ns(pose_stamped.header.stamp)
    if points:
        points_arr = np.asarray(points, dtype=np.float32)
    else:
        points_arr = np.zeros((0, 3), dtype=np.float32)
    return RecordedPathFrame(
        timestamp_ns=timestamp_ns,
        header_timestamp_ns=_header_stamp_ns(msg),
        points=points_arr,
        first_stamp_ns=first_stamp_ns,
    )


def _target_to_frame(timestamp_ns: int, msg) -> TargetFrame:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return TargetFrame(
        timestamp_ns=timestamp_ns,
        position=np.array([p.x, p.y, p.z], dtype=np.float64),
        wxyz=np.array([q.w, q.x, q.y, q.z], dtype=np.float64),
    )


def _decode_depth(msg) -> np.ndarray | None:
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    if msg.encoding in ("16UC1", "mono16"):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
    return None


def _latest_before(frames, timestamp_ns: int):
    if not frames:
        return None
    timestamps = [f.timestamp_ns for f in frames]
    idx = bisect_right(timestamps, timestamp_ns) - 1
    if idx < 0:
        return None
    return frames[idx]


def _target_at(events: list[TargetEvent], timestamp_ns: int) -> TargetFrame | None:
    event = _latest_before(events, timestamp_ns)
    if event is None:
        return None
    return event.target


def _camera_to_robot_center(T: np.ndarray) -> np.ndarray:
    return T[:3, 3] - T[:3, :3] @ GO2_CONFIG.cam_offset_3d


def _quat_angle_rad(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> float:
    q1 = np.asarray(q1_xyzw, dtype=np.float64)
    q2 = np.asarray(q2_xyzw, dtype=np.float64)
    q1_norm = np.linalg.norm(q1)
    q2_norm = np.linalg.norm(q2)
    if q1_norm <= 0.0 or q2_norm <= 0.0:
        return float("inf")
    q1 = q1 / q1_norm
    q2 = q2 / q2_norm
    dot = abs(float(np.dot(q1, q2)))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def _seed_from_last_trajectory(
    last_traj: np.ndarray | None,
    last_base_stamp: float | None,
    query_stamp: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    if last_traj is None or last_base_stamp is None or len(last_traj) < 1:
        return None
    rel_t = query_stamp - last_base_stamp
    if rel_t < 0.0:
        return None
    traj_end_stamp = last_base_stamp + float(len(last_traj) - 1) * dt
    if query_stamp > traj_end_stamp:
        return None
    idx = int(round(rel_t / dt))
    idx = max(0, min(idx, len(last_traj) - 1))
    seed_stamp = last_base_stamp + float(idx) * dt
    p = last_traj[idx, :3].copy()
    q = last_traj[idx, 3:7].copy()
    if len(last_traj) == 1:
        v = np.zeros(3, dtype=np.float64)
    elif idx < len(last_traj) - 1:
        v = (last_traj[idx + 1, :3] - last_traj[idx, :3]) / dt
    else:
        v = (last_traj[idx, :3] - last_traj[idx - 1, :3]) / dt
    return p, v.astype(np.float64), q, seed_stamp


def _front_obstacle_dist(
    T: np.ndarray,
    obstacle_mask: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    max_dist: float = 0.5,
) -> float:
    center = _camera_to_robot_center(T)
    fwd = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    n = (fwd[0] ** 2 + fwd[1] ** 2) ** 0.5
    fx, fy = (fwd[0] / n, fwd[1] / n) if n > 1e-6 else (1.0, 0.0)
    lx, ly = -fy, fx
    fl, _, hw = GO2_CONFIG.footprint_from_control()
    rows, cols = obstacle_mask.shape
    steps = int(max_dist / resolution) + 1
    for step in range(steps):
        d_from_face = step * resolution
        d_from_center = fl + d_from_face
        for w in (-hw, 0.0, hw):
            xi = int((center[0] + fx * d_from_center + lx * w - origin[0]) / resolution)
            yi = int((center[1] + fy * d_from_center + ly * w - origin[1]) / resolution)
            if 0 <= xi < rows and 0 <= yi < cols and obstacle_mask[xi, yi]:
                return d_from_face
    return max_dist + 1.0


def _depth_to_points(
    timestamp_ns: int,
    depth: np.ndarray,
    camera_info,
    pose: PoseFrame | None,
    pixel_stride: int,
    max_points: int,
) -> DepthFrame | None:
    if depth is None or camera_info is None:
        return None

    K = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    if fx <= 0.0 or fy <= 0.0:
        return None

    v, u = np.mgrid[0 : depth.shape[0] : pixel_stride, 0 : depth.shape[1] : pixel_stride]
    z = depth[v, u]
    valid = np.isfinite(z) & (z > 0.05) & (z < 10.0)
    if not np.any(valid):
        return None

    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=1)

    if len(points) > max_points:
        step = int(np.ceil(len(points) / max_points))
        points = points[::step]
        z = z[::step]

    if pose is not None:
        homog = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
        points = (pose.matrix @ homog.T).T[:, :3]

    depth_norm = np.clip(z / 5.0, 0.0, 1.0)
    colors_bgr = cv2.applyColorMap(np.uint8((1.0 - depth_norm) * 255).reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    colors = colors_bgr[:, ::-1].astype(np.float32) / 255.0
    return DepthFrame(timestamp_ns=timestamp_ns, points=points.astype(np.float32), colors=colors)


def _occupancy_to_points(grid: np.ndarray, origin: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray] | None:
    occupied = np.argwhere(grid > 0.1)
    if len(occupied) == 0:
        return None

    points = origin + occupied.astype(np.float64) * resolution
    values = np.clip(grid[occupied[:, 0], occupied[:, 1], occupied[:, 2]] / 0.2, 0.0, 1.0)
    colors_bgr = cv2.applyColorMap(np.uint8(values * 255).reshape(-1, 1), cv2.COLORMAP_INFERNO).reshape(-1, 3)
    colors = colors_bgr[:, ::-1].astype(np.float32) / 255.0
    return points.astype(np.float32), colors


def _obstacle_mask_to_points(
    obstacle_mask: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    z: float,
) -> np.ndarray:
    cells = np.argwhere(obstacle_mask)
    if len(cells) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    points = np.empty((len(cells), 3), dtype=np.float64)
    points[:, 0] = origin[0] + (cells[:, 0].astype(np.float64) + 0.5) * resolution
    points[:, 1] = origin[1] + (cells[:, 1].astype(np.float64) + 0.5) * resolution
    points[:, 2] = z
    return points.astype(np.float32)


def build_occupancy_frames(
    raycast_inputs: list[RaycastInputFrame],
    max_frames: int,
    every: int,
    grid_shape: tuple[int, int, int] = (100, 100, 10),
    resolution: float = 0.1,
    step: int = 10,
) -> list[OccupancyFrame]:
    if max_frames <= 0 or not raycast_inputs:
        return []

    origin = np.asarray(grid_shape, dtype=np.float64) * resolution / -2.0
    grid = np.zeros(grid_shape, dtype=np.float64)
    frames: list[OccupancyFrame] = []
    every = max(1, every)

    for input_idx, frame in enumerate(raycast_inputs):
        if input_idx % every != 0:
            continue

        center = origin + np.asarray(grid_shape, dtype=np.float64) * resolution / 2.0
        robot_pos = frame.pose.matrix[:3, 3]
        if np.linalg.norm(robot_pos - center) > 0.1:
            new_origin = robot_pos - np.asarray(grid_shape, dtype=np.float64) * resolution / 2.0
            grid, origin = roll_occupancy_grid(grid, origin, new_origin, resolution)

        fx, fy = frame.k[0, 0], frame.k[1, 1]
        cx, cy = frame.k[0, 2], frame.k[1, 2]
        new_occ = run_raycasting_loopy(frame.depth, frame.pose.matrix, grid_shape, fx, fy, cx, cy, origin, step, resolution)
        grid *= 0.99
        grid += new_occ
        grid = np.clip(grid, -0.2, 0.2)

        cloud = _occupancy_to_points(grid, origin, resolution)
        if cloud is not None:
            points, colors = cloud
            obstacle_mask = build_obstacle_map(
                grid,
                origin,
                resolution,
                robot_z=frame.pose.matrix[2, 3],
                config=ObstacleConfig(),
            )
            obstacle_points = _obstacle_mask_to_points(
                obstacle_mask,
                origin,
                resolution,
                frame.pose.matrix[2, 3],
            )
            front_clearance = _front_obstacle_dist(frame.pose.matrix, obstacle_mask, origin, resolution)
            frames.append(
                OccupancyFrame(
                    timestamp_ns=frame.timestamp_ns,
                    points=points,
                    colors=colors,
                    obstacle_points=obstacle_points,
                    obstacle_mask=obstacle_mask.copy(),
                    front_clearance=front_clearance,
                    origin=origin.copy(),
                    grid=grid.copy(),
                    pose=frame.pose,
                )
            )
        if len(frames) >= max_frames:
            break

    return frames


def _normalize_trajectories(trajectories: np.ndarray) -> np.ndarray:
    if trajectories.ndim != 3:
        return np.zeros((0, 0, 7), dtype=np.float64)
    if trajectories.shape[2] >= 7:
        return trajectories[:, :, :7].astype(np.float64)
    return np.zeros((0, 0, 7), dtype=np.float64)


def _build_trajectory_frame(
    occ: OccupancyFrame,
    target: TargetFrame | None,
    smoothed_velocity: float,
    last_param: np.ndarray,
    last_traj: np.ndarray | None,
    last_traj_base_stamp: float | None,
    max_candidates: int,
    dt: float = 0.1,
    planning_latency_s: float = 0.2,
    seed_fallback_distance_m: float = 2.0,
    seed_fallback_rotation_rad: float = np.deg2rad(15.0),
    resolution: float = 0.1,
) -> tuple[TrajectoryFrame, np.ndarray, np.ndarray | None, float | None]:
    obstacle_mask = build_obstacle_map(
        occ.grid,
        occ.origin,
        resolution,
        robot_z=occ.pose.matrix[2, 3],
        config=ObstacleConfig(),
    )
    esdf_map = distance_transform_edt(~obstacle_mask).astype(np.float32) * resolution

    stamp = occ.timestamp_ns / 1e9
    query_stamp = stamp + planning_latency_s
    planning_base_stamp = stamp
    seed = _seed_from_last_trajectory(last_traj, last_traj_base_stamp, query_stamp, dt)
    if seed is not None:
        init_p_seed, _, init_q_seed, seed_stamp = seed
        current_center = _camera_to_robot_center(occ.pose.matrix)
        odom_q_xyzw = np.array(
            [occ.pose.wxyz[1], occ.pose.wxyz[2], occ.pose.wxyz[3], occ.pose.wxyz[0]],
            dtype=np.float64,
        )
        seed_position_ok = np.linalg.norm(init_p_seed - current_center) <= seed_fallback_distance_m
        seed_rotation_ok = _quat_angle_rad(init_q_seed, odom_q_xyzw) <= seed_fallback_rotation_rad
        if seed_position_ok and seed_rotation_ok:
            init_p, init_q = init_p_seed, init_q_seed
            planning_base_stamp = seed_stamp
        else:
            seed = None
    if seed is None:
        init_p = _camera_to_robot_center(occ.pose.matrix)
        init_q = np.array(
            [occ.pose.wxyz[1], occ.pose.wxyz[2], occ.pose.wxyz[3], occ.pose.wxyz[0]],
            dtype=np.float64,
        )

    trajectories, params = generate_trajectory_library_3d(init_p=init_p, init_q=init_q, dt=dt)
    trajectories = _normalize_trajectories(trajectories)
    vocab_trajs, vocab_params = generate_predefined_trajectory_vocabularies(init_p=init_p, init_q=init_q, dt=dt)
    vocab_trajs = _normalize_trajectories(vocab_trajs)
    if len(vocab_trajs) > 0:
        trajectories = np.concatenate([trajectories, vocab_trajs], axis=0)
        params = np.concatenate([params, vocab_params], axis=0)

    front_len, rear_len, half_w = GO2_CONFIG.footprint_from_control()
    scores, _ = score_trajectories_by_ESDF(
        trajectories,
        esdf_map,
        occ.origin,
        resolution,
        GO2_CONFIG.safety_radius,
        front_len,
        rear_len,
        half_w,
    )
    scores = np.asarray(scores, dtype=np.float64)
    esdf_top = np.argsort(scores, kind="stable")[: max(0, max_candidates)]

    selected = None
    selected_param = None
    status = "no_target" if target is None else "all_collision" if np.all(np.isinf(scores)) else "selected"
    next_last_traj = last_traj
    next_last_base_stamp = last_traj_base_stamp
    next_last_param = last_param
    if target is not None and not np.all(np.isinf(scores)):
        front_clearance = _front_obstacle_dist(occ.pose.matrix, obstacle_mask, occ.origin, resolution)
        should_reverse = front_clearance <= 0.30
        target_pose = target.position
        costs = []
        for traj, param, score in zip(trajectories, params, scores):
            is_backward_traj = param[0] < 0.0
            reverse_gate_penalty = 0.0
            if should_reverse and not is_backward_traj:
                reverse_gate_penalty = 1e9
            elif not should_reverse and is_backward_traj:
                reverse_gate_penalty = 1e9
            dist = np.linalg.norm(np.asarray(traj[-1, :3]) - target_pose)
            costs.append(
                score * 100000
                + 100 * dist
                + 40 * abs(last_param[0] - param[0])
                + 10 * abs(last_param[1] - param[1])
                + reverse_gate_penalty
            )
        best_idx = int(np.argsort(np.asarray(costs), kind="stable")[0])
        selected = trajectories[best_idx].copy()
        selected_param = params[best_idx].copy()
        next_last_traj = selected
        next_last_base_stamp = planning_base_stamp
        next_last_param = selected_param

    return (
        TrajectoryFrame(
            timestamp_ns=occ.timestamp_ns,
            candidates=[trajectories[int(i)].copy() for i in esdf_top],
            selected=selected,
            selected_param=selected_param,
            status=status,
        ),
        next_last_param,
        next_last_traj,
        next_last_base_stamp,
    )


def build_trajectory_frames(
    occupancy_frames: list[OccupancyFrame],
    target_events: list[TargetEvent],
    max_candidates: int,
    seed_fallback_rotation_rad: float,
) -> list[TrajectoryFrame]:
    frames: list[TrajectoryFrame] = []
    if not occupancy_frames:
        return frames

    last_param = np.array([0.0, 0.0], dtype=np.float64)
    last_traj = None
    last_traj_base_stamp = None
    last_pose = None
    last_stamp = None
    smoothed_velocity = 0.0

    for occ in occupancy_frames:
        stamp = occ.timestamp_ns / 1e9
        if last_pose is not None and last_stamp is not None:
            dt = max(1e-3, stamp - last_stamp)
            velocity_estimated = np.linalg.norm(occ.pose.matrix[:3, 3] - last_pose[:3, 3]) / dt
            smoothed_velocity = 0.9 * smoothed_velocity + 0.1 * velocity_estimated
        last_pose = occ.pose.matrix.copy()
        last_stamp = stamp

        frame, last_param, last_traj, last_traj_base_stamp = _build_trajectory_frame(
            occ,
            _target_at(target_events, occ.timestamp_ns),
            smoothed_velocity,
            last_param,
            last_traj,
            last_traj_base_stamp,
            max_candidates,
            seed_fallback_rotation_rad=seed_fallback_rotation_rad,
        )
        if frame.selected_param is not None:
            print(
                "[offline planning] "
                f"t={occ.timestamp_ns / 1e9:.3f} "
                f"vx={frame.selected_param[0]:.3f} "
                f"omega_y={frame.selected_param[1]:.3f}"
            )
        frames.append(frame)
    return frames


def read_bag(
    bag_path: Path,
    depth_stride: int,
    max_depth_points: int,
    max_depth_frames: int,
    max_occupancy_frames: int,
    occupancy_every: int,
    max_candidate_trajectories: int,
    seed_fallback_rotation_rad: float,
) -> BagData:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_types = {topic: get_message(type_name) for topic, type_name in topics.items()}

    poses: list[PoseFrame] = []
    targets: list[TargetFrame] = []
    target_events: list[TargetEvent] = []
    depth_frames: list[DepthFrame] = []
    raycast_inputs: list[RaycastInputFrame] = []
    recorded_path_frames: list[RecordedPathFrame] = []
    camera_info = None
    start_ns = None
    end_ns = None

    while reader.has_next():
        topic, serialized_msg, timestamp_ns = reader.read_next()
        timestamp_ns = int(timestamp_ns)
        start_ns = timestamp_ns if start_ns is None else min(start_ns, timestamp_ns)
        end_ns = timestamp_ns if end_ns is None else max(end_ns, timestamp_ns)

        if topic not in READABLE_PLANNING_TOPICS:
            continue

        msg = deserialize_message(serialized_msg, msg_types[topic])
        if topic == PLANNING_ODOM_TOPIC:
            poses.append(_pose_stamped_to_pose_frame(timestamp_ns, msg))
        elif topic == LEGACY_PLANNING_ODOM_TOPIC:
            poses.append(_odom_to_pose_frame(timestamp_ns, msg))
        elif topic == PLANNING_PATH_TOPIC:
            recorded_path_frames.append(_path_to_frame(timestamp_ns, msg))
        elif topic == "/control/target_pose":
            target = _target_to_frame(timestamp_ns, msg)
            targets.append(target)
            target_events.append(TargetEvent(timestamp_ns=timestamp_ns, target=target))
        elif topic == "/mapping/poi_change":
            target_events.append(TargetEvent(timestamp_ns=timestamp_ns, target=None))
        elif topic == "/camera/camera/infra2/camera_info":
            camera_info = msg
        elif topic in (PLANNING_DEPTH_TOPIC, LEGACY_PLANNING_DEPTH_TOPIC):
            pose = _latest_before(poses, timestamp_ns)
            depth = _decode_depth(msg)
            if depth is None or camera_info is None or pose is None:
                continue
            if len(depth_frames) < max_depth_frames:
                frame = _depth_to_points(timestamp_ns, depth, camera_info, pose, depth_stride, max_depth_points)
                if frame is not None:
                    depth_frames.append(frame)
            if len(raycast_inputs) < max_occupancy_frames * max(1, occupancy_every):
                raycast_inputs.append(
                    RaycastInputFrame(
                        timestamp_ns=timestamp_ns,
                        depth=depth.copy(),
                        pose=pose,
                        k=np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3).copy(),
                    )
                )

    poses.sort(key=lambda f: f.timestamp_ns)
    targets.sort(key=lambda f: f.timestamp_ns)
    target_events.sort(key=lambda f: f.timestamp_ns)
    depth_frames.sort(key=lambda f: f.timestamp_ns)
    raycast_inputs.sort(key=lambda f: f.timestamp_ns)
    recorded_path_frames.sort(key=lambda f: f.timestamp_ns)
    occupancy_frames = build_occupancy_frames(raycast_inputs, max_occupancy_frames, occupancy_every)
    trajectory_frames = build_trajectory_frames(
        occupancy_frames,
        target_events,
        max_candidate_trajectories,
        seed_fallback_rotation_rad,
    )
    return BagData(
        poses=poses,
        targets=targets,
        target_events=target_events,
        depth_frames=depth_frames,
        raycast_inputs=raycast_inputs,
        occupancy_frames=occupancy_frames,
        trajectory_frames=trajectory_frames,
        recorded_path_frames=recorded_path_frames,
        topics=topics,
        start_ns=start_ns,
        end_ns=end_ns,
    )


def _line_segments(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)


def _trajectory_segments(trajectories: list[np.ndarray]) -> np.ndarray:
    segments = [_line_segments(traj[:, :3]) for traj in trajectories if len(traj) > 1]
    if not segments:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.concatenate(segments, axis=0)


def _footprint_segments(T: np.ndarray) -> np.ndarray:
    forward = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    left = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
    center = _camera_to_robot_center(T)
    front_len, rear_len, half_w = GO2_CONFIG.footprint_from_control()
    corners = np.asarray(
        [
            center + forward * front_len + left * half_w,
            center + forward * front_len - left * half_w,
            center - forward * rear_len - left * half_w,
            center - forward * rear_len + left * half_w,
        ],
        dtype=np.float32,
    )
    return np.stack([corners, np.roll(corners, -1, axis=0)], axis=1)


def _front_clearance_segments(T: np.ndarray, clearance: float, max_dist: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    center = _camera_to_robot_center(T)
    forward = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    left = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
    front_len, _, half_w = GO2_CONFIG.footprint_from_control()
    start_center = center + forward * front_len
    line_len = min(clearance, max_dist)
    end_center = start_center + forward * line_len
    segments = np.asarray(
        [
            [start_center - left * half_w, start_center + left * half_w],
            [start_center, end_center],
            [end_center - left * half_w, end_center + left * half_w],
        ],
        dtype=np.float32,
    )
    color = np.array([1.0, 0.05, 0.0], dtype=np.float32) if clearance <= 0.30 else np.array([0.0, 0.9, 0.25], dtype=np.float32)
    colors = np.tile(color.reshape(1, 1, 3), (len(segments), 2, 1))
    return segments, colors


def run_viewer(
    data: BagData,
    port: int,
    point_size: float,
    occupancy_point_size: float,
    candidate_line_width: float,
    selected_line_width: float,
    rate: float,
) -> None:
    server = viser.ViserServer(port=port)
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+z")

    if len(data.poses) > 1:
        all_pose_points = np.stack([p.position for p in data.poses]).astype(np.float32)
        server.scene.add_line_segments(
            "/odom/full_trail",
            points=_line_segments(all_pose_points),
            colors=np.tile(np.array([[[0.2, 0.6, 1.0], [0.2, 0.6, 1.0]]], dtype=np.float32), (max(0, len(all_pose_points) - 1), 1, 1)),
            line_width=2,
        )

    state = {"idx": 0, "play": False, "last_update": time.monotonic(), "handles": {}}
    max_idx = max(0, len(data.poses) - 1)
    with server.gui.add_folder("Replay") as _:
        frame_slider = server.gui.add_slider("Frame", min=0, max=max_idx, step=1, initial_value=0)
        prev_button = server.gui.add_button("Prev Frame")
        next_button = server.gui.add_button("Next Frame")
        frame_text = server.gui.add_markdown("**Frame**\n\nNo frame loaded.")
        play_toggle = server.gui.add_checkbox("Play", initial_value=False)
        initial_rate = max(0.1, float(rate))
        rate_slider = server.gui.add_slider("Rate", min=0.1, max=max(5.0, initial_rate), step=0.1, initial_value=initial_rate)
        show_depth = server.gui.add_checkbox("Show Depth Points", initial_value=False)
        show_occupancy = server.gui.add_checkbox("Show Occupancy Grid", initial_value=True)
        show_obstacles = server.gui.add_checkbox("Show Planner Obstacles", initial_value=True)
        show_candidates = server.gui.add_checkbox("Show Candidate Trajectories", initial_value=True)
        show_selected = server.gui.add_checkbox("Show Selected Trajectory", initial_value=True)
        show_recorded_path = server.gui.add_checkbox("Show Recorded Path", initial_value=True)
        show_target = server.gui.add_checkbox("Show Target", initial_value=True)
        show_footprint = server.gui.add_checkbox("Show Footprint", initial_value=True)
        obstacle_text = server.gui.add_markdown(
            "**Obstacle Judgement**\n\n"
            "`occupied`: grid value `> 0.1`  \n"
            "`z band`: robot z `[-0.4, 0.4] m`  \n"
            "`wall span`: `>= 0.2 m`  \n"
            "`dilation`: `1` cell"
        )
        selected_param_text = server.gui.add_markdown("**Selected Trajectory**\n\nNo selected trajectory.")

    def render(idx: int) -> None:
        if not data.poses:
            return
        handles = state["handles"]
        for name in (
            "current",
            "trail",
            "target",
            "depth",
            "occupancy",
            "obstacles",
            "front_clearance",
            "candidates",
            "selected",
            "recorded_path",
            "footprint",
        ):
            handle = handles.pop(name, None)
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    pass

        idx = int(np.clip(idx, 0, len(data.poses) - 1))
        pose = data.poses[idx]
        state["idx"] = idx
        rel_s = 0.0
        if data.start_ns is not None:
            rel_s = (pose.timestamp_ns - data.start_ns) / 1e9
        traj_frame = _latest_before(data.trajectory_frames, pose.timestamp_ns)
        recorded_path = _latest_before(data.recorded_path_frames, pose.timestamp_ns)
        odom_header_ns = pose.header_timestamp_ns
        if recorded_path is not None and recorded_path.first_stamp_ns is not None:
            path_msg_stamp_s = recorded_path.timestamp_ns / 1e9
            path_first_stamp_s = recorded_path.first_stamp_ns / 1e9
            path_header_s = _format_stamp(recorded_path.header_timestamp_ns)
            if odom_header_ns is not None:
                path_lag_text = f"`odom header - path first`: `{(odom_header_ns - recorded_path.first_stamp_ns) / 1e9:.3f}` s"
            else:
                path_lag_text = "`odom header - path first`: `none`"
            path_timestamp_text = (
                f"`path record stamp`: `{path_msg_stamp_s:.6f}` s  \n"
                f"`path header stamp`: `{path_header_s}` s  \n"
                f"`path first stamp`: `{path_first_stamp_s:.6f}` s  \n"
                f"{path_lag_text}"
            )
        elif recorded_path is not None:
            path_timestamp_text = (
                f"`path record stamp`: `{recorded_path.timestamp_ns / 1e9:.6f}` s  \n"
                f"`path header stamp`: `{_format_stamp(recorded_path.header_timestamp_ns)}` s  \n"
                "`path first stamp`: `none`"
            )
        else:
            path_timestamp_text = "`path first stamp`: `none`"
        frame_text.content = (
            "**Frame**\n\n"
            f"`id`: `{idx}` / `{max_idx}`  \n"
            f"`odom record stamp`: `{pose.timestamp_ns / 1e9:.6f}` s  \n"
            f"`odom header stamp`: `{_format_stamp(pose.header_timestamp_ns)}` s  \n"
            f"`relative`: `{rel_s:.3f}` s  \n"
            f"{path_timestamp_text}"
        )
        handles["current"] = server.scene.add_transform_controls("/odom/current", position=pose.position, wxyz=pose.wxyz)
        if show_footprint.value:
            footprint = _footprint_segments(pose.matrix)
            handles["footprint"] = server.scene.add_line_segments(
                "/robot/footprint",
                points=footprint,
                colors=np.tile(
                    np.array([[[1.0, 0.62, 0.0], [1.0, 0.62, 0.0]]], dtype=np.float32),
                    (len(footprint), 1, 1),
                ),
                line_width=3,
            )

        trail = np.stack([p.position for p in data.poses[: idx + 1]]).astype(np.float32)
        if len(trail) > 1:
            colors = np.tile(np.array([[[0.0, 1.0, 0.3], [0.0, 1.0, 0.3]]], dtype=np.float32), (len(trail) - 1, 1, 1))
            handles["trail"] = server.scene.add_line_segments(
                "/odom/replayed_trail",
                points=_line_segments(trail),
                colors=colors,
                line_width=4,
            )

        target = _target_at(data.target_events, pose.timestamp_ns)
        if show_target.value and target is not None:
            handles["target"] = server.scene.add_icosphere(
                "/target_pose",
                position=target.position,
                radius=0.12,
                color=(255, 180, 0),
            )

        depth = _latest_before(data.depth_frames, pose.timestamp_ns)
        if show_depth.value and depth is not None:
            handles["depth"] = server.scene.add_point_cloud(
                "/depth/current",
                points=depth.points,
                colors=depth.colors,
                point_size=point_size,
                point_shape="rounded",
            )

        occupancy = _latest_before(data.occupancy_frames, pose.timestamp_ns)
        if show_occupancy.value and occupancy is not None:
            handles["occupancy"] = server.scene.add_point_cloud(
                "/occupancy/current",
                points=occupancy.points,
                colors=occupancy.colors,
                point_size=occupancy_point_size,
                point_shape="square",
            )
        if occupancy is not None:
            reverse_gate = occupancy.front_clearance <= 0.30
            obstacle_text.content = (
                "**Obstacle Judgement**\n\n"
                "`occupied`: grid value `> 0.1`  \n"
                "`z band`: robot z `[-0.4, 0.4] m`  \n"
                "`wall span`: `>= 0.2 m`  \n"
                "`dilation`: `1` cell  \n"
                f"`obstacle cells`: `{len(occupancy.obstacle_points)}`  \n"
                f"`front clearance`: `{occupancy.front_clearance:.2f} m`  \n"
                f"`reverse gate`: `{'ON' if reverse_gate else 'OFF'}`"
            )
        else:
            obstacle_text.content = (
                "**Obstacle Judgement**\n\n"
                "No occupancy frame available for this pose."
            )
        if show_obstacles.value and occupancy is not None:
            if len(occupancy.obstacle_points) > 0:
                handles["obstacles"] = server.scene.add_point_cloud(
                    "/planning/obstacle_mask",
                    points=occupancy.obstacle_points,
                    colors=np.tile(np.array([[1.0, 0.05, 0.0]], dtype=np.float32), (len(occupancy.obstacle_points), 1)),
                    point_size=max(occupancy_point_size * 1.6, 0.055),
                    point_shape="square",
                )
            clearance_segments, clearance_colors = _front_clearance_segments(pose.matrix, occupancy.front_clearance)
            handles["front_clearance"] = server.scene.add_line_segments(
                "/planning/front_clearance",
                points=clearance_segments,
                colors=clearance_colors,
                line_width=5,
            )

        if traj_frame is not None and traj_frame.selected_param is not None:
            selected_param_text.content = (
                "**Selected Trajectory**\n\n"
                f"`t`: `{traj_frame.timestamp_ns / 1e9:.3f}`  \n"
                f"`vx`: `{traj_frame.selected_param[0]:.3f}`  \n"
                f"`omega_y`: `{traj_frame.selected_param[1]:.3f}`"
            )
        else:
            if traj_frame is not None and traj_frame.status == "all_collision":
                selected_param_text.content = (
                    "**Selected Trajectory**\n\n"
                    f"`t`: `{traj_frame.timestamp_ns / 1e9:.3f}`  \n"
                    "**All candidate trajectories are in collision.**"
                )
            elif traj_frame is not None and traj_frame.status == "no_target":
                selected_param_text.content = (
                    "**Selected Trajectory**\n\n"
                    f"`t`: `{traj_frame.timestamp_ns / 1e9:.3f}`  \n"
                    "No active target pose."
                )
            else:
                selected_param_text.content = "**Selected Trajectory**\n\nNo selected trajectory."
        if traj_frame is not None and show_candidates.value:
            candidate_segments = _trajectory_segments(traj_frame.candidates)
            if len(candidate_segments) > 0:
                candidate_color = np.array([[[0.35, 0.45, 0.55], [0.35, 0.45, 0.55]]], dtype=np.float32)
                handles["candidates"] = server.scene.add_line_segments(
                    "/trajectories/candidates",
                    points=candidate_segments,
                    colors=np.tile(candidate_color, (len(candidate_segments), 1, 1)),
                    line_width=candidate_line_width,
                )
        if traj_frame is not None and show_selected.value and traj_frame.selected is not None:
            selected_segments = _line_segments(traj_frame.selected[:, :3])
            if len(selected_segments) > 0:
                selected_color = np.array([[[0.0, 1.0, 0.15], [0.0, 1.0, 0.15]]], dtype=np.float32)
                handles["selected"] = server.scene.add_line_segments(
                    "/trajectories/selected",
                    points=selected_segments,
                    colors=np.tile(selected_color, (len(selected_segments), 1, 1)),
                    line_width=selected_line_width,
                )
        if recorded_path is not None and show_recorded_path.value and len(recorded_path.points) > 1:
            recorded_segments = _line_segments(recorded_path.points)
            if len(recorded_segments) > 0:
                recorded_color = np.array([[[1.0, 0.82, 0.0], [1.0, 0.82, 0.0]]], dtype=np.float32)
                handles["recorded_path"] = server.scene.add_line_segments(
                    "/planning/recorded_trajectory_path",
                    points=recorded_segments,
                    colors=np.tile(recorded_color, (len(recorded_segments), 1, 1)),
                    line_width=max(2.0, selected_line_width * 0.75),
                )

    @frame_slider.on_update
    def _(_) -> None:
        render(int(frame_slider.value))

    @prev_button.on_click
    def _(_) -> None:
        frame_slider.value = max(0, state["idx"] - 1)

    @next_button.on_click
    def _(_) -> None:
        frame_slider.value = min(max_idx, state["idx"] + 1)

    @play_toggle.on_update
    def _(_) -> None:
        state["play"] = bool(play_toggle.value)
        state["last_update"] = time.monotonic()

    @show_depth.on_update
    def _(_) -> None:
        render(state["idx"])

    @show_occupancy.on_update
    def _(_) -> None:
        render(state["idx"])

    @show_obstacles.on_update
    def _(_) -> None:
        render(state["idx"])

    @show_candidates.on_update
    def _(_) -> None:
        render(state["idx"])

    @show_selected.on_update
    def _(_) -> None:
        render(state["idx"])

    @show_recorded_path.on_update
    def _(_) -> None:
        render(state["idx"])

    @show_target.on_update
    def _(_) -> None:
        render(state["idx"])

    @show_footprint.on_update
    def _(_) -> None:
        render(state["idx"])

    print(f"Viser server running on port {port}")
    print(
        f"Loaded poses={len(data.poses)} targets={len(data.targets)} "
        f"depth_frames={len(data.depth_frames)} occupancy_frames={len(data.occupancy_frames)} "
        f"trajectory_frames={len(data.trajectory_frames)} "
        f"recorded_paths={len(data.recorded_path_frames)}"
    )
    if data.start_ns is not None and data.end_ns is not None:
        print(f"Bag duration: {(data.end_ns - data.start_ns) / 1e9:.2f}s")
    expected_topics = set(PLANNING_INPUT_TOPICS)
    if PLANNING_ODOM_TOPIC not in data.topics and LEGACY_PLANNING_ODOM_TOPIC in data.topics:
        expected_topics.remove(PLANNING_ODOM_TOPIC)
    if PLANNING_DEPTH_TOPIC not in data.topics and LEGACY_PLANNING_DEPTH_TOPIC in data.topics:
        expected_topics.remove(PLANNING_DEPTH_TOPIC)
    missing = sorted(expected_topics - set(data.topics.keys()))
    if missing:
        print("Missing expected topics:")
        for topic in missing:
            print(f"  {topic}")

    render(0)
    try:
        while True:
            if state["play"] and data.poses:
                now = time.monotonic()
                step_dt = max(0.02, 1.0 / max(0.1, float(rate_slider.value)))
                if now - state["last_update"] >= step_dt:
                    next_idx = (state["idx"] + 1) % len(data.poses)
                    frame_slider.value = next_idx
                    state["last_update"] = now
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a planning debug rosbag directly with Viser.")
    parser.add_argument("--bag", required=True, type=Path, help="Path to a rosbag directory.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument("--depth-stride", type=int, default=20, help="Pixel stride for sampled depth point cloud.")
    parser.add_argument("--max-depth-points", type=int, default=50_000, help="Maximum sampled points per depth frame.")
    parser.add_argument("--max-depth-frames", type=int, default=200, help="Maximum depth frames to keep in memory.")
    parser.add_argument("--max-occupancy-frames", type=int, default=200, help="Maximum raycast occupancy frames to keep in memory.")
    parser.add_argument("--occupancy-every", type=int, default=1, help="Raycast every Nth depth frame.")
    parser.add_argument("--max-candidate-trajectories", type=int, default=100, help="Maximum candidate trajectories to visualize per frame.")
    parser.add_argument("--seed-fallback-rotation-deg", type=float, default=15.0, help="Use current odom when the previous selected trajectory seed rotates farther than this from odom.")
    parser.add_argument("--point-size", type=float, default=0.015, help="Viser depth point size.")
    parser.add_argument("--occupancy-point-size", type=float, default=0.035, help="Viser occupancy voxel point size.")
    parser.add_argument("--candidate-line-width", type=float, default=1.0, help="Viser line width for candidate trajectories.")
    parser.add_argument("--selected-line-width", type=float, default=5.0, help="Viser line width for the selected trajectory.")
    parser.add_argument("--rate", type=float, default=10.0, help="Pose frames advanced per second when playing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bag.exists():
        raise FileNotFoundError(args.bag)
    data = read_bag(
        args.bag,
        depth_stride=max(1, args.depth_stride),
        max_depth_points=max(1, args.max_depth_points),
        max_depth_frames=max(0, args.max_depth_frames),
        max_occupancy_frames=max(0, args.max_occupancy_frames),
        occupancy_every=max(1, args.occupancy_every),
        max_candidate_trajectories=max(0, args.max_candidate_trajectories),
        seed_fallback_rotation_rad=np.deg2rad(max(0.0, args.seed_fallback_rotation_deg)),
    )
    run_viewer(
        data,
        port=args.port,
        point_size=args.point_size,
        occupancy_point_size=args.occupancy_point_size,
        candidate_line_width=args.candidate_line_width,
        selected_line_width=args.selected_line_width,
        rate=args.rate,
    )


if __name__ == "__main__":
    main()
