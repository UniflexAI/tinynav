from __future__ import annotations

import argparse
import csv
import math
import time
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tool.planning_bag_viser import (
    LEGACY_PLANNING_DEPTH_TOPIC,
    LEGACY_PLANNING_ODOM_TOPIC,
    PLANNING_DEPTH_TOPIC,
    PLANNING_ODOM_TOPIC,
    RaycastInputFrame,
    _decode_depth,
    _latest_before,
    _odom_to_pose_frame,
    _pose_stamped_to_pose_frame,
    build_occupancy_frames,
)


DEBUG_TOPICS = {
    "/insight/vio_100hz",
    "/planning/trajectory_path",
    "/cmd_vel",
}


@dataclass
class BagEvent:
    timestamp_ns: int
    order: int
    topic: str
    msg: object


@dataclass
class CmdFrame:
    timestamp_ns: int
    vx: float
    vyaw: float


@dataclass
class DebugFrame:
    timestamp_ns: int
    odom_stamp: float
    odom_x: float
    odom_y: float
    odom_z: float
    odom_yaw: float
    vx_expected: float
    vyaw_expected: float
    measured_vx: float
    measured_vyaw: float
    zero_reason: str
    path_available: bool
    path_len: int
    path_is_static: bool
    path_start_t: float
    path_end_t: float
    query_t: float
    trajectory_expired: bool
    target_idx: int
    target_x: float
    target_y: float
    target_yaw: float
    tx: float
    ty: float
    heading_err: float
    v_ref: float
    w_ref: float
    vx_recorded: float | None = None
    vyaw_recorded: float | None = None
    cmd_match_dt: float | None = None
    path_points: np.ndarray | None = None
    target_point: np.ndarray | None = None


@dataclass
class OdomFrame:
    timestamp_ns: int
    stamp_sec: float
    camera_position: np.ndarray
    rotation: np.ndarray
    robot_position: np.ndarray
    robot_yaw: float


@dataclass
class TrajectoryRollout:
    index: int
    path_index: int
    start_sample: str
    start_source: str
    path_timestamp_ns: int
    path_start_t: float
    path_end_t: float
    start_odom_timestamp_ns: int
    start_odom_stamp: float
    planned_xy: np.ndarray
    simulated_xy: np.ndarray
    target_xy: np.ndarray
    cmd_vx: np.ndarray
    cmd_vyaw: np.ndarray
    lateral_error: np.ndarray
    heading_error: np.ndarray
    zero_reasons: list[str]
    final_error: float
    mean_error: float
    max_error: float
    occupancy_points: np.ndarray | None = None
    occupancy_colors: np.ndarray | None = None
    obstacle_points: np.ndarray | None = None


def stamp_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(pose) -> np.ndarray:
    p = pose.position
    q = pose.orientation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_matrix((q.x, q.y, q.z, q.w))
    T[:3, 3] = np.array([p.x, p.y, p.z], dtype=np.float64)
    return T


def pose_stamped_to_matrix(msg) -> np.ndarray:
    return pose_to_matrix(msg.pose)


def yaw_from_rotation(rotation: np.ndarray) -> float:
    forward = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if np.linalg.norm(forward[:2]) <= 1e-6:
        return 0.0
    return math.atan2(float(forward[1]), float(forward[0]))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def line_segments(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)


def robot_pose_from_pose_stamped(msg, timestamp_ns: int) -> OdomFrame:
    T = pose_stamped_to_matrix(msg)
    rotation = T[:3, :3]
    camera_position = T[:3, 3].copy()
    camera_offset = np.array([0.0, 0.0, 0.35], dtype=np.float64)
    robot_position = camera_position - rotation @ camera_offset
    robot_yaw = yaw_from_rotation(rotation)
    return OdomFrame(
        timestamp_ns=timestamp_ns,
        stamp_sec=stamp_sec(msg.header.stamp),
        camera_position=camera_position,
        rotation=rotation.copy(),
        robot_position=robot_position,
        robot_yaw=robot_yaw,
    )


def path_msg_to_ref(path_msg):
    return OfflineCmdVelController().rebuild_path(path_msg)


def odom_from_path_index(path_ref: np.ndarray, points: np.ndarray, path_timestamp_ns: int, index: int) -> OdomFrame:
    index = int(np.clip(index, 0, len(path_ref) - 1))
    yaw = float(path_ref[index, 2])
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotation = np.array(
        [
            [0.0, -s, c],
            [0.0, c, s],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    robot_position = points[index].astype(np.float64).copy()
    camera_offset = np.array([0.0, 0.0, 0.35], dtype=np.float64)
    camera_position = robot_position + rotation @ camera_offset
    return OdomFrame(
        timestamp_ns=path_timestamp_ns,
        stamp_sec=float(path_ref[index, 5]),
        camera_position=camera_position,
        rotation=rotation,
        robot_position=robot_position,
        robot_yaw=yaw,
    )


def trajectory_expired(path_ref: np.ndarray, now_sec: float, lookahead_s: float, grace_s: float) -> bool:
    if path_ref is None or len(path_ref) == 0:
        return True
    return now_sec + lookahead_s > float(path_ref[-1, 5]) + grace_s


def find_tracking_target_in_ref(path_ref: np.ndarray, track_idx: int, now_sec: float, lookahead_s: float) -> tuple[np.ndarray | None, int]:
    if path_ref is None or len(path_ref) == 0:
        return None, -1
    start_idx = int(np.clip(track_idx, 0, len(path_ref) - 1))
    t_vec = path_ref[start_idx:, 5]
    if len(t_vec) > 1 and np.min(np.diff(t_vec)) > 0.0:
        query_t = now_sec + lookahead_s
        rel_idx = int(np.searchsorted(t_vec, query_t, side="left"))
        target_idx = min(start_idx + rel_idx, len(path_ref) - 1)
        return path_ref[target_idx], target_idx
    return path_ref[start_idx], start_idx


def controller_command(robot_xy: np.ndarray, robot_yaw: float, path_ref: np.ndarray, track_idx: int, now_sec: float):
    lookahead_s = 0.15
    expire_grace_s = 0.05
    if path_ref is None or len(path_ref) == 0:
        return 0.0, 0.0, track_idx, None, np.nan, np.nan, "no /planning/trajectory_path arrived yet"
    if trajectory_expired(path_ref, now_sec, lookahead_s, expire_grace_s):
        return 0.0, 0.0, len(path_ref) - 1, None, np.nan, np.nan, "trajectory expired"

    target, target_idx = find_tracking_target_in_ref(path_ref, track_idx, now_sec, lookahead_s)
    if target is None:
        return 0.0, 0.0, track_idx, None, np.nan, np.nan, "no valid tracking target"

    dx = float(target[0] - robot_xy[0])
    dy = float(target[1] - robot_xy[1])
    cy = math.cos(robot_yaw)
    sy = math.sin(robot_yaw)
    tx = cy * dx + sy * dy
    ty = -sy * dx + cy * dy
    heading_err = wrap_angle(float(target[2]) - robot_yaw)
    v_ref = float(target[3])
    w_ref = float(target[4])

    b = 1.2
    zeta = 0.7
    k = 2.0 * zeta * math.sqrt(w_ref * w_ref + b * v_ref * v_ref)
    v = v_ref * math.cos(heading_err) + k * tx
    wz = w_ref + k * heading_err + b * v_ref * (1.0 if abs(heading_err) < 1e-6 else math.sin(heading_err) / heading_err) * ty
    v *= 1.2
    v = float(np.clip(v, -0.2, 0.6))
    wz = float(np.clip(wz, -0.8, 0.8))

    heading_to_goal = wrap_angle(float(path_ref[-1, 2]) - robot_yaw)
    if np.linalg.norm(robot_xy[:2] - path_ref[-1, :2]) < 0.1 and abs(heading_to_goal) < 0.1:
        return 0.0, 0.0, target_idx, target, tx, ty, "local trajectory endpoint reached"
    return v, wz, target_idx, target, tx, ty, ""


def nearest_path_error(path_xy: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) == 0 or len(points_xy) == 0:
        return np.zeros(0, dtype=np.float64)
    errors = []
    for p in points_xy:
        errors.append(float(np.min(np.linalg.norm(path_xy - p[None, :], axis=1))))
    return np.asarray(errors, dtype=np.float64)


class OfflineCmdVelController:
    def __init__(self):
        self.position = np.zeros(3, dtype=np.float64)
        self.rotation = np.eye(3, dtype=np.float64)
        self.odom_pose_initialized = False
        self.odom_stamp_sec = None
        self.path_ref = None
        self.path_points = None
        self.track_idx = 0
        self.last_traj_update_sec = None
        self.latest_path_start_t = float("nan")
        self.latest_path_end_t = float("nan")
        self.time_lookahead_s = 0.15
        self.trajectory_expire_grace_s = 0.05
        self.vx_gain_comp = 1.2
        self._last_robot_pos_raw = None
        self._last_robot_yaw_raw = None
        self._last_odom_stamp_sec = None

    def now_sec(self, fallback_timestamp_ns: int) -> float:
        if self.odom_stamp_sec is not None:
            return float(self.odom_stamp_sec)
        return float(fallback_timestamp_ns) * 1e-9

    def odom_callback(self, msg, timestamp_ns: int) -> DebugFrame:
        measured_pose = pose_stamped_to_matrix(msg)
        measured_position = measured_pose[:3, 3]
        measured_rotation = measured_pose[:3, :3]
        camera_offset = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        robot_pos_raw = measured_position - measured_rotation @ camera_offset
        robot_yaw_raw = yaw_from_rotation(measured_rotation)
        odom_stamp_sec = stamp_sec(msg.header.stamp)

        measured_vx = 0.0
        measured_vyaw = 0.0
        if self._last_robot_pos_raw is not None and self._last_odom_stamp_sec is not None:
            dt = max(1e-3, odom_stamp_sec - self._last_odom_stamp_sec)
            delta = robot_pos_raw - self._last_robot_pos_raw
            forward = measured_rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if np.linalg.norm(forward[:2]) > 1e-6:
                forward_xy = forward[:2] / np.linalg.norm(forward[:2])
                measured_vx = float(np.dot(delta[:2], forward_xy) / dt)
            measured_vyaw = float(wrap_angle(robot_yaw_raw - self._last_robot_yaw_raw) / dt)
        self._last_robot_pos_raw = robot_pos_raw.copy()
        self._last_robot_yaw_raw = robot_yaw_raw
        self._last_odom_stamp_sec = odom_stamp_sec

        if not self.odom_pose_initialized:
            self.position = measured_position
            self.rotation = measured_rotation
            self.odom_pose_initialized = True
        else:
            alpha = 0.35
            self.position = (1.0 - alpha) * self.position + alpha * measured_position
            self.rotation = measured_rotation

        self.odom_stamp_sec = odom_stamp_sec
        return self.control_loop(timestamp_ns, measured_vx, measured_vyaw)

    def path_callback(self, msg, timestamp_ns: int) -> None:
        now = self.now_sec(timestamp_ns)
        if self.last_traj_update_sec is not None and now - self.last_traj_update_sec < 0.2:
            return

        new_ref, new_points = self.rebuild_path(msg)
        if new_ref is None:
            self.path_ref = None
            self.path_points = None
            self.track_idx = 0
            self.latest_path_start_t = float("nan")
            self.latest_path_end_t = float("nan")
        elif self.path_ref is None or len(self.path_ref) == 0:
            self.path_ref = new_ref
            self.path_points = new_points
            self.track_idx = 0
            self.latest_path_start_t = float(new_ref[0, 5])
            self.latest_path_end_t = float(new_ref[-1, 5])
        else:
            new_start_t = float(new_ref[0, 5])
            keep_mask = self.path_ref[:, 5] < new_start_t
            kept = self.path_ref[keep_mask]
            kept_points = self.path_points[keep_mask] if self.path_points is not None else np.zeros((0, 3), dtype=np.float64)
            if len(kept) == 0:
                self.path_ref = new_ref
                self.path_points = new_points
            else:
                self.path_ref = np.vstack((kept, new_ref))
                self.path_points = np.vstack((kept_points, new_points))
            self.track_idx = int(np.clip(np.searchsorted(self.path_ref[:, 5], now, side="left"), 0, len(self.path_ref) - 1))
            self.latest_path_start_t = float(new_ref[0, 5])
            self.latest_path_end_t = float(new_ref[-1, 5])
        self.last_traj_update_sec = now

    def rebuild_path(self, path_msg):
        n = len(path_msg.poses)
        if n == 0:
            return None, None

        xy_yaw = np.zeros((n, 3), dtype=np.float64)
        points = np.zeros((n, 3), dtype=np.float64)
        t = np.zeros(n, dtype=np.float64)
        last_yaw = 0.0
        for i, pose_stamped in enumerate(path_msg.poses):
            pose_np = pose_to_matrix(pose_stamped.pose)
            xy_yaw[i, :2] = pose_np[:2, 3]
            points[i] = pose_np[:3, 3]
            t[i] = stamp_sec(pose_stamped.header.stamp)

            forward = pose_np[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if np.linalg.norm(forward[:2]) > 1e-6:
                last_yaw = math.atan2(float(forward[1]), float(forward[0]))
            xy_yaw[i, 2] = last_yaw

        path_ref = np.zeros((n, 6), dtype=np.float64)
        path_ref[:, :3] = xy_yaw
        path_ref[:, 5] = t
        if n > 1:
            dt_arr = np.diff(t)
            if np.min(dt_arr) <= 0.0:
                t = t[0] + np.arange(n, dtype=np.float64) * 0.2

            yaw_u = np.unwrap(xy_yaw[:, 2])
            for i in range(n - 1):
                dt = max(1e-3, float(t[i + 1] - t[i]))
                ds = float(np.linalg.norm(xy_yaw[i + 1, :2] - xy_yaw[i, :2]))
                path_ref[i, 3] = ds / dt
                path_ref[i, 4] = (yaw_u[i + 1] - yaw_u[i]) / dt
            path_ref[-1, 3] = path_ref[-2, 3]
            path_ref[-1, 4] = path_ref[-2, 4]

        return path_ref, points

    def control_loop(self, timestamp_ns: int, measured_vx: float, measured_vyaw: float) -> DebugFrame:
        odom_yaw = yaw_from_rotation(self.rotation)
        now_sec = self.now_sec(timestamp_ns)
        path_start_t = self.latest_path_start_t
        path_end_t = self.latest_path_end_t
        query_t = now_sec + self.time_lookahead_s
        expired = False if self.path_ref is None or len(self.path_ref) == 0 else query_t > path_end_t + self.trajectory_expire_grace_s
        base = dict(
            timestamp_ns=timestamp_ns,
            odom_stamp=float(self.odom_stamp_sec if self.odom_stamp_sec is not None else timestamp_ns * 1e-9),
            odom_x=float(self.position[0]),
            odom_y=float(self.position[1]),
            odom_z=float(self.position[2]),
            odom_yaw=float(odom_yaw),
            vx_expected=0.0,
            vyaw_expected=0.0,
            measured_vx=float(measured_vx),
            measured_vyaw=float(measured_vyaw),
            path_available=self.path_ref is not None,
            path_len=0 if self.path_ref is None else int(len(self.path_ref)),
            path_is_static=self.path_is_static(),
            path_start_t=path_start_t,
            path_end_t=path_end_t,
            query_t=float(query_t),
            trajectory_expired=bool(expired),
            target_idx=-1,
            target_x=float("nan"),
            target_y=float("nan"),
            target_yaw=float("nan"),
            tx=float("nan"),
            ty=float("nan"),
            heading_err=float("nan"),
            v_ref=float("nan"),
            w_ref=float("nan"),
            path_points=None if self.path_points is None else self.path_points.copy(),
            target_point=None,
        )

        if self.path_ref is None:
            return DebugFrame(**base, zero_reason="no /planning/trajectory_path arrived yet")

        if expired:
            self.track_idx = len(self.path_ref) - 1
            return DebugFrame(**base, zero_reason="trajectory expired")

        camera_offset = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        robot_pos = self.position - self.rotation @ camera_offset
        robot_yaw = odom_yaw

        target, target_idx = self.find_tracking_target(robot_pos, robot_yaw, self.odom_stamp_sec)
        if target is None:
            return DebugFrame(**base, zero_reason="no valid tracking target")

        tx, ty, heading_err = self.target_error(robot_pos, robot_yaw, target)
        v_ref = float(target[3])
        w_ref = float(target[4])

        b = 1.2
        zeta = 0.7
        k = 2.0 * zeta * math.sqrt(w_ref * w_ref + b * v_ref * v_ref)
        v = v_ref * math.cos(heading_err) + k * tx
        wz = w_ref + k * heading_err + b * v_ref * self.sinc(heading_err) * ty
        v *= self.vx_gain_comp
        v = float(np.clip(v, -0.2, 0.6))
        wz = float(np.clip(wz, -0.8, 0.8))

        heading_to_goal = wrap_angle(float(self.path_ref[-1, 2]) - robot_yaw)
        zero_reason = ""
        if np.linalg.norm(robot_pos[:2] - self.path_ref[-1, :2]) < 0.1 and abs(heading_to_goal) < 0.1:
            v = 0.0
            wz = 0.0
            zero_reason = "local trajectory endpoint reached"
        elif abs(v) < 1e-9 and abs(wz) < 1e-9 and self.path_is_static():
            zero_reason = "static trajectory_path"
        elif abs(v) < 1e-9 and abs(wz) < 1e-9:
            zero_reason = "computed zero command"

        target_point = np.array([target[0], target[1], self.position[2]], dtype=np.float64)
        base.update(
            vx_expected=v,
            vyaw_expected=wz,
            zero_reason=zero_reason,
            target_idx=target_idx,
            target_x=float(target[0]),
            target_y=float(target[1]),
            target_yaw=float(target[2]),
            tx=float(tx),
            ty=float(ty),
            heading_err=float(heading_err),
            v_ref=v_ref,
            w_ref=w_ref,
            target_point=target_point,
        )
        return DebugFrame(**base)

    def find_tracking_target(self, robot_pos: np.ndarray, robot_yaw: float, now_sec: float | None):
        if self.path_ref is None or len(self.path_ref) == 0:
            return None, -1

        start_idx = int(np.clip(self.track_idx, 0, len(self.path_ref) - 1))
        t_vec = self.path_ref[start_idx:, 5]
        if now_sec is not None and len(t_vec) > 1 and np.min(np.diff(t_vec)) > 0.0:
            query_t = now_sec + self.time_lookahead_s
            rel_idx = int(np.searchsorted(t_vec, query_t, side="left"))
            target_idx = min(start_idx + rel_idx, len(self.path_ref) - 1)
            self.track_idx = target_idx
            return self.path_ref[target_idx], target_idx

        delta = self.path_ref[start_idx:, :2] - robot_pos[:2]
        dist = np.linalg.norm(delta, axis=1)
        if float(np.max(dist)) < 0.05:
            yaw_err = np.abs([wrap_angle(float(yaw) - robot_yaw) for yaw in self.path_ref[start_idx:, 2]])
            nearest_idx = start_idx + int(np.argmin(yaw_err))
        else:
            nearest_idx = start_idx + int(np.argmin(dist))
        target_idx = min(nearest_idx + 1, len(self.path_ref) - 1)
        self.track_idx = nearest_idx
        return self.path_ref[target_idx], target_idx

    @staticmethod
    def target_error(robot_pos: np.ndarray, robot_yaw: float, target: np.ndarray):
        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]
        cy = math.cos(robot_yaw)
        sy = math.sin(robot_yaw)
        tx = cy * dx + sy * dy
        ty = -sy * dx + cy * dy
        heading_err = wrap_angle(float(target[2]) - robot_yaw)
        return tx, ty, heading_err

    @staticmethod
    def sinc(a: float) -> float:
        if abs(a) < 1e-6:
            return 1.0
        return math.sin(a) / a

    def path_is_static(self) -> bool:
        if self.path_ref is None or len(self.path_ref) < 2:
            return False
        pos_delta = np.linalg.norm(self.path_ref[:, :2] - self.path_ref[0, :2], axis=1)
        yaw_delta = np.abs([wrap_angle(float(yaw) - float(self.path_ref[0, 2])) for yaw in self.path_ref[:, 2]])
        return bool(np.max(pos_delta) < 1e-5 and np.max(yaw_delta) < 1e-5)


def normalize_bag_path(path: Path) -> Path:
    if path.is_file() and path.suffix == ".db3":
        return path.parent
    return path


def read_debug_events(bag_path: Path):
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from rosidl_runtime_py.utilities import get_message

    bag_path = normalize_bag_path(bag_path)
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_types = {topic: get_message(type_name) for topic, type_name in topics.items() if topic in DEBUG_TOPICS}

    events: list[BagEvent] = []
    cmd_frames: list[CmdFrame] = []
    order = 0
    while reader.has_next():
        topic, serialized_msg, timestamp_ns = reader.read_next()
        timestamp_ns = int(timestamp_ns)
        if topic not in DEBUG_TOPICS:
            order += 1
            continue
        msg = deserialize_message(serialized_msg, msg_types[topic])
        if topic == "/cmd_vel":
            cmd_frames.append(CmdFrame(timestamp_ns, float(msg.linear.x), float(msg.angular.z)))
        else:
            events.append(BagEvent(timestamp_ns, order, topic, msg))
        order += 1

    events.sort(key=lambda event: (event.timestamp_ns, event.order))
    cmd_frames.sort(key=lambda frame: frame.timestamp_ns)
    return topics, events, cmd_frames


def read_occupancy_frames_for_debug(bag_path: Path, max_frames: int, every: int):
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from rosidl_runtime_py.utilities import get_message

    bag_path = normalize_bag_path(bag_path)
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    readable = {
        PLANNING_ODOM_TOPIC,
        LEGACY_PLANNING_ODOM_TOPIC,
        PLANNING_DEPTH_TOPIC,
        LEGACY_PLANNING_DEPTH_TOPIC,
        "/camera/camera/infra2/camera_info",
    }
    msg_types = {topic: get_message(type_name) for topic, type_name in topics.items() if topic in readable}

    poses = []
    raycast_inputs: list[RaycastInputFrame] = []
    camera_info = None
    max_inputs = max(0, max_frames) * max(1, every)

    while reader.has_next():
        topic, serialized_msg, timestamp_ns = reader.read_next()
        timestamp_ns = int(timestamp_ns)
        if topic not in readable:
            continue
        msg = deserialize_message(serialized_msg, msg_types[topic])
        if topic == PLANNING_ODOM_TOPIC:
            poses.append(_pose_stamped_to_pose_frame(timestamp_ns, msg))
        elif topic == LEGACY_PLANNING_ODOM_TOPIC:
            poses.append(_odom_to_pose_frame(timestamp_ns, msg))
        elif topic == "/camera/camera/infra2/camera_info":
            camera_info = msg
        elif topic in (PLANNING_DEPTH_TOPIC, LEGACY_PLANNING_DEPTH_TOPIC):
            if camera_info is None or (max_inputs > 0 and len(raycast_inputs) >= max_inputs):
                continue
            pose = _latest_before(poses, timestamp_ns)
            depth = _decode_depth(msg)
            if pose is None or depth is None:
                continue
            raycast_inputs.append(
                RaycastInputFrame(
                    timestamp_ns=timestamp_ns,
                    depth=depth.copy(),
                    pose=pose,
                    k=np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3).copy(),
                )
            )

    poses.sort(key=lambda f: f.timestamp_ns)
    raycast_inputs.sort(key=lambda f: f.timestamp_ns)
    return build_occupancy_frames(
        raycast_inputs,
        max_frames=max(0, max_frames),
        every=max(1, every),
    )


def attach_recorded_cmds(frames: list[DebugFrame], cmd_frames: list[CmdFrame], match_window_s: float) -> None:
    if not frames or not cmd_frames:
        return
    cmd_times = [frame.timestamp_ns for frame in cmd_frames]
    window_ns = int(match_window_s * 1e9)
    for frame in frames:
        idx = bisect_left(cmd_times, frame.timestamp_ns)
        candidates = []
        if idx < len(cmd_frames):
            candidates.append(cmd_frames[idx])
        if idx > 0:
            candidates.append(cmd_frames[idx - 1])
        if not candidates:
            continue
        best = min(candidates, key=lambda cmd: abs(cmd.timestamp_ns - frame.timestamp_ns))
        dt_ns = best.timestamp_ns - frame.timestamp_ns
        if abs(dt_ns) <= window_ns:
            frame.vx_recorded = best.vx
            frame.vyaw_recorded = best.vyaw
            frame.cmd_match_dt = dt_ns * 1e-9


def simulate(events: list[BagEvent], cmd_frames: list[CmdFrame], match_window_s: float, max_odom_frames: int) -> list[DebugFrame]:
    controller = OfflineCmdVelController()
    frames: list[DebugFrame] = []
    for event in events:
        if event.topic == "/planning/trajectory_path":
            controller.path_callback(event.msg, event.timestamp_ns)
        elif event.topic == "/insight/vio_100hz":
            frames.append(controller.odom_callback(event.msg, event.timestamp_ns))
            if max_odom_frames > 0 and len(frames) >= max_odom_frames:
                break
    attach_recorded_cmds(frames, cmd_frames, match_window_s)
    return frames


def simulate_trajectory_rollout(
    path_ref: np.ndarray,
    start_odom: OdomFrame,
    dt: float,
    vx_tau_s: float,
    vx_delay_s: float,
    yaw_gain: float,
    yaw_delay_s: float,
) -> TrajectoryRollout | None:
    if path_ref is None or len(path_ref) < 2:
        return None
    start_t = float(start_odom.stamp_sec)
    end_t = float(path_ref[-1, 5])
    if start_t > end_t + 0.05:
        return None

    t = max(start_t, float(path_ref[0, 5]))
    robot_xy = start_odom.robot_position[:2].astype(np.float64).copy()
    robot_yaw = float(start_odom.robot_yaw)
    track_idx = int(np.clip(np.searchsorted(path_ref[:, 5], t, side="left"), 0, len(path_ref) - 1))

    simulated = []
    targets = []
    cmd_vx = []
    cmd_vyaw = []
    lateral_error = []
    heading_error = []
    zero_reasons = []
    delayed_vx = [0.0] * max(0, int(round(max(0.0, vx_delay_s) / dt)))
    delayed_vyaw = [0.0] * max(0, int(round(max(0.0, yaw_delay_s) / dt)))
    applied_vx = 0.0
    vx_alpha = 1.0 if vx_tau_s <= 1e-6 else min(1.0, dt / vx_tau_s)

    max_steps = max(2, int(math.ceil(max(0.0, end_t - t) / dt)) + 1)
    for _ in range(max_steps):
        v, wz, track_idx, target, _tx, ty, zero_reason = controller_command(robot_xy, robot_yaw, path_ref, track_idx, t)
        delayed_vx.append(v)
        delayed_vx_cmd = delayed_vx.pop(0)
        applied_vx += vx_alpha * (delayed_vx_cmd - applied_vx)
        delayed_vyaw.append(wz)
        applied_wz = float(yaw_gain) * delayed_vyaw.pop(0)
        simulated.append([robot_xy[0], robot_xy[1], start_odom.robot_position[2]])
        cmd_vx.append(v)
        cmd_vyaw.append(wz)
        lateral_error.append(0.0 if math.isnan(ty) else float(ty))
        zero_reasons.append(zero_reason)
        if target is not None:
            targets.append([target[0], target[1], start_odom.robot_position[2]])
            heading_error.append(wrap_angle(float(target[2]) - robot_yaw))
        else:
            targets.append([robot_xy[0], robot_xy[1], start_odom.robot_position[2]])
            heading_error.append(float("nan"))

        robot_xy = robot_xy + np.array([math.cos(robot_yaw), math.sin(robot_yaw)], dtype=np.float64) * applied_vx * dt
        robot_yaw = wrap_angle(robot_yaw + applied_wz * dt)
        t += dt
        if t > end_t + 1e-9:
            break

    simulated_arr = np.asarray(simulated, dtype=np.float64)
    planned_xy = path_ref[:, :2].astype(np.float64)
    errors = nearest_path_error(planned_xy, simulated_arr[:, :2])
    return TrajectoryRollout(
        index=-1,
        path_index=-1,
        start_sample="",
        start_source="",
        path_timestamp_ns=0,
        path_start_t=float(path_ref[0, 5]),
        path_end_t=float(path_ref[-1, 5]),
        start_odom_timestamp_ns=start_odom.timestamp_ns,
        start_odom_stamp=start_odom.stamp_sec,
        planned_xy=np.column_stack((planned_xy, np.full(len(planned_xy), start_odom.robot_position[2]))),
        simulated_xy=simulated_arr,
        target_xy=np.asarray(targets, dtype=np.float64),
        cmd_vx=np.asarray(cmd_vx, dtype=np.float64),
        cmd_vyaw=np.asarray(cmd_vyaw, dtype=np.float64),
        lateral_error=np.asarray(lateral_error, dtype=np.float64),
        heading_error=np.asarray(heading_error, dtype=np.float64),
        zero_reasons=zero_reasons,
        final_error=float(errors[-1]) if len(errors) else float("nan"),
        mean_error=float(np.mean(errors)) if len(errors) else float("nan"),
        max_error=float(np.max(errors)) if len(errors) else float("nan"),
    )


def build_trajectory_rollouts(
    events: list[BagEvent],
    dt: float,
    max_rollouts: int,
    occupancy_frames=None,
    start_mode: str = "vio",
    vx_tau_s: float = 0.07,
    vx_delay_s: float = 0.13,
    yaw_gain: float = 0.85,
    yaw_delay_s: float = 0.06,
) -> list[TrajectoryRollout]:
    odoms: list[OdomFrame] = []
    rollouts: list[TrajectoryRollout] = []
    path_index = 0
    for event in events:
        if event.topic == "/insight/vio_100hz":
            odoms.append(robot_pose_from_pose_stamped(event.msg, event.timestamp_ns))
        elif event.topic == "/planning/trajectory_path":
            if start_mode != "path-first" and not odoms:
                continue
            path_ref, points = path_msg_to_ref(event.msg)
            if path_ref is None:
                continue
            if start_mode == "path-first":
                start_specs = [("start", 0), ("middle", len(path_ref) // 2)]
            else:
                start_specs = [("vio", -1)]
            for start_sample, sample_idx in start_specs:
                if start_mode == "path-first":
                    start_odom = odom_from_path_index(path_ref, points, event.timestamp_ns, sample_idx)
                else:
                    odom_times = [odom.timestamp_ns for odom in odoms]
                    idx = bisect_left(odom_times, event.timestamp_ns)
                    start_odom = odoms[max(0, idx - 1)]
                rollout = simulate_trajectory_rollout(
                    path_ref,
                    start_odom,
                    dt,
                    vx_tau_s=vx_tau_s,
                    vx_delay_s=vx_delay_s,
                    yaw_gain=yaw_gain,
                    yaw_delay_s=yaw_delay_s,
                )
                if rollout is None:
                    continue
                rollout.index = len(rollouts)
                rollout.path_index = path_index
                rollout.start_sample = start_sample
                rollout.start_source = start_mode
                rollout.path_timestamp_ns = event.timestamp_ns
                if occupancy_frames:
                    occ = _latest_before(occupancy_frames, event.timestamp_ns)
                    if occ is not None:
                        rollout.occupancy_points = occ.points
                        rollout.occupancy_colors = occ.colors
                        rollout.obstacle_points = occ.obstacle_points
                rollouts.append(rollout)
                if max_rollouts > 0 and len(rollouts) >= max_rollouts:
                    return rollouts
            path_index += 1
    return rollouts


def write_csv(frames: list[DebugFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp_ns",
        "odom_stamp",
        "odom_x",
        "odom_y",
        "odom_z",
        "odom_yaw",
        "path_available",
        "path_len",
        "path_is_static",
        "target_idx",
        "target_x",
        "target_y",
        "target_yaw",
        "tx",
        "ty",
        "heading_err",
        "v_ref",
        "w_ref",
        "vx_expected",
        "vyaw_expected",
        "measured_vx",
        "measured_vyaw",
        "vx_tracking_error",
        "vyaw_tracking_error",
        "path_start_t",
        "path_end_t",
        "query_t",
        "path_age_s",
        "path_time_remaining_s",
        "trajectory_expired",
        "vx_recorded",
        "vyaw_recorded",
        "cmd_match_dt",
        "vx_error",
        "vyaw_error",
        "zero_reason",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            vx_error = "" if frame.vx_recorded is None else frame.vx_expected - frame.vx_recorded
            vyaw_error = "" if frame.vyaw_recorded is None else frame.vyaw_expected - frame.vyaw_recorded
            path_age_s = "" if math.isnan(frame.path_start_t) else frame.odom_stamp - frame.path_start_t
            path_time_remaining_s = "" if math.isnan(frame.path_end_t) else frame.path_end_t - frame.query_t
            writer.writerow(
                {
                    "timestamp_ns": frame.timestamp_ns,
                    "odom_stamp": frame.odom_stamp,
                    "odom_x": frame.odom_x,
                    "odom_y": frame.odom_y,
                    "odom_z": frame.odom_z,
                    "odom_yaw": frame.odom_yaw,
                    "path_available": frame.path_available,
                    "path_len": frame.path_len,
                    "path_is_static": frame.path_is_static,
                    "target_idx": frame.target_idx,
                    "target_x": frame.target_x,
                    "target_y": frame.target_y,
                    "target_yaw": frame.target_yaw,
                    "tx": frame.tx,
                    "ty": frame.ty,
                    "heading_err": frame.heading_err,
                    "v_ref": frame.v_ref,
                    "w_ref": frame.w_ref,
                    "vx_expected": frame.vx_expected,
                    "vyaw_expected": frame.vyaw_expected,
                    "measured_vx": frame.measured_vx,
                    "measured_vyaw": frame.measured_vyaw,
                    "vx_tracking_error": frame.vx_expected - frame.measured_vx,
                    "vyaw_tracking_error": frame.vyaw_expected - frame.measured_vyaw,
                    "path_start_t": frame.path_start_t,
                    "path_end_t": frame.path_end_t,
                    "query_t": frame.query_t,
                    "path_age_s": path_age_s,
                    "path_time_remaining_s": path_time_remaining_s,
                    "trajectory_expired": frame.trajectory_expired,
                    "vx_recorded": "" if frame.vx_recorded is None else frame.vx_recorded,
                    "vyaw_recorded": "" if frame.vyaw_recorded is None else frame.vyaw_recorded,
                    "cmd_match_dt": "" if frame.cmd_match_dt is None else frame.cmd_match_dt,
                    "vx_error": vx_error,
                    "vyaw_error": vyaw_error,
                    "zero_reason": frame.zero_reason,
                }
            )


def print_summary(topics: dict[str, str], events: list[BagEvent], cmd_frames: list[CmdFrame], frames: list[DebugFrame], print_rows: int) -> None:
    topic_counts = Counter(event.topic for event in events)
    print("Topics:")
    for topic in sorted(DEBUG_TOPICS):
        state = "present" if topic in topics else "missing"
        count = len(cmd_frames) if topic == "/cmd_vel" else topic_counts.get(topic, 0)
        print(f"  {topic}: {state}, messages={count}")

    print("\nOffline controller:")
    print(f"  odom frames evaluated: {len(frames)}")
    if frames:
        print(f"  odom time range: {frames[0].timestamp_ns / 1e9:.3f} -> {frames[-1].timestamp_ns / 1e9:.3f}")
    if cmd_frames:
        print(f"  cmd_vel time range: {cmd_frames[0].timestamp_ns / 1e9:.3f} -> {cmd_frames[-1].timestamp_ns / 1e9:.3f}")
    zero_counts = Counter(frame.zero_reason for frame in frames if frame.zero_reason)
    if zero_counts:
        print("  zero reasons:")
        for reason, count in zero_counts.most_common():
            print(f"    {reason}: {count}")
    else:
        print("  zero reasons: none")

    compared = [f for f in frames if f.vx_recorded is not None and f.vyaw_recorded is not None]
    print(f"  frames matched to recorded /cmd_vel: {len(compared)}")
    if compared:
        vx_err = np.array([f.vx_expected - f.vx_recorded for f in compared], dtype=np.float64)
        vyaw_err = np.array([f.vyaw_expected - f.vyaw_recorded for f in compared], dtype=np.float64)
        print(f"  max |vx error|: {np.max(np.abs(vx_err)):.6f}")
        print(f"  max |vyaw error|: {np.max(np.abs(vyaw_err)):.6f}")
        print(f"  mean |vx error|: {np.mean(np.abs(vx_err)):.6f}")
        print(f"  mean |vyaw error|: {np.mean(np.abs(vyaw_err)):.6f}")

    moving = [f for f in frames if not f.zero_reason]
    print(f"  nonzero control frames: {len(moving)}")
    if moving:
        vx_track = np.array([f.vx_expected - f.measured_vx for f in moving], dtype=np.float64)
        vyaw_track = np.array([f.vyaw_expected - f.measured_vyaw for f in moving], dtype=np.float64)
        lateral = np.array([abs(f.ty) for f in moving if not math.isnan(f.ty)], dtype=np.float64)
        heading = np.array([abs(f.heading_err) for f in moving if not math.isnan(f.heading_err)], dtype=np.float64)
        print(f"  mean |cmd vx - measured vx|: {np.mean(np.abs(vx_track)):.3f} m/s")
        print(f"  p95 |cmd vx - measured vx|: {np.percentile(np.abs(vx_track), 95):.3f} m/s")
        print(f"  mean |cmd vyaw - measured vyaw|: {np.mean(np.abs(vyaw_track)):.3f} rad/s")
        print(f"  p95 |cmd vyaw - measured vyaw|: {np.percentile(np.abs(vyaw_track), 95):.3f} rad/s")
        if len(lateral) > 0:
            print(f"  mean lateral error |ty|: {np.mean(lateral):.3f} m")
            print(f"  p95 lateral error |ty|: {np.percentile(lateral, 95):.3f} m")
        if len(heading) > 0:
            print(f"  mean heading error: {np.mean(heading):.3f} rad")
            print(f"  p95 heading error: {np.percentile(heading, 95):.3f} rad")

    expired_count = sum(1 for f in frames if f.trajectory_expired)
    print(f"  trajectory expired frames: {expired_count}")
    valid_path_age = np.array(
        [f.odom_stamp - f.path_start_t for f in frames if not math.isnan(f.path_start_t)],
        dtype=np.float64,
    )
    if len(valid_path_age) > 0:
        print(f"  mean path age: {np.mean(valid_path_age):.3f}s")
        print(f"  p95 path age: {np.percentile(valid_path_age, 95):.3f}s")

    if print_rows <= 0 or not frames:
        return
    print("\nRows:")
    for idx, frame in enumerate(frames[-print_rows:], start=max(0, len(frames) - print_rows)):
        recorded = "none"
        if frame.vx_recorded is not None and frame.vyaw_recorded is not None:
            recorded = f"vx={frame.vx_recorded:.3f} vyaw={frame.vyaw_recorded:.3f} dt={frame.cmd_match_dt:.3f}s"
        reason = frame.zero_reason or "nonzero"
        print(
            f"  [{idx}] t={frame.timestamp_ns / 1e9:.3f} "
            f"expected vx={frame.vx_expected:.3f} vyaw={frame.vyaw_expected:.3f} "
            f"measured vx={frame.measured_vx:.3f} vyaw={frame.measured_vyaw:.3f} "
            f"recorded={recorded} reason={reason} "
            f"target_idx={frame.target_idx} tx={frame.tx:.3f} ty={frame.ty:.3f} heading={frame.heading_err:.3f}"
        )


def run_viser(
    frames: list[DebugFrame],
    rollouts: list[TrajectoryRollout],
    port: int,
    rate: float,
    sim_vx_tau_s: float,
    sim_vx_delay_s: float,
    sim_yaw_gain: float,
    sim_yaw_delay_s: float,
) -> None:
    import viser

    if not frames:
        raise ValueError("No debug frames to visualize.")

    server = viser.ViserServer(port=port)
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+z")
    odom_points = np.array([[f.odom_x, f.odom_y, f.odom_z] for f in frames], dtype=np.float32)
    if len(odom_points) > 1:
        server.scene.add_line_segments(
            "/cmd_vel_debug/odom_trail",
            points=line_segments(odom_points),
            colors=np.tile(np.array([[[0.0, 0.8, 1.0], [0.0, 0.8, 1.0]]], dtype=np.float32), (len(odom_points) - 1, 1, 1)),
            line_width=2,
        )

    rollout_state = {"idx": 0, "handles": {}}
    if rollouts:
        rollout_by_key = {(rollout.path_index, rollout.start_sample): rollout for rollout in rollouts}
        path_indices = sorted({rollout.path_index for rollout in rollouts})
        sample_names = [name for name in ("start", "middle", "vio") if any(rollout.start_sample == name for rollout in rollouts)]
        with server.gui.add_folder("Planning Trajectory Path") as _:
            rollout_slider = server.gui.add_slider("/planning/trajectory_path", min=0, max=len(path_indices) - 1, step=1, initial_value=0)
            start_sample_slider = server.gui.add_slider("Start Sample", min=0, max=len(sample_names) - 1, step=1, initial_value=0)
            show_rollout_occupancy = server.gui.add_checkbox("Show Occupancy Grid", initial_value=True)
            show_rollout_obstacles = server.gui.add_checkbox("Show Obstacles", initial_value=True)
            rollout_info = server.gui.add_markdown("")
    else:
        rollout_by_key = {}
        path_indices = []
        sample_names = []
        rollout_slider = None
        start_sample_slider = None
        show_rollout_occupancy = None
        show_rollout_obstacles = None
        rollout_info = None

    def clear_rollout() -> None:
        handles = rollout_state["handles"]
        for name in ("planned", "simulated", "start", "targets", "occupancy", "obstacles"):
            handle = handles.pop(name, None)
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    pass

    def render_rollout(rollout: TrajectoryRollout) -> None:
        if not rollouts:
            return
        clear_rollout()
        rollout_state["idx"] = rollout.index

        planned_segments = line_segments(rollout.planned_xy.astype(np.float32))
        if len(planned_segments) > 0:
            rollout_state["handles"]["planned"] = server.scene.add_line_segments(
                "/cmd_vel_debug/rollout/planned",
                points=planned_segments,
                colors=np.tile(np.array([[[0.1, 1.0, 0.25], [0.1, 1.0, 0.25]]], dtype=np.float32), (len(planned_segments), 1, 1)),
                line_width=5,
            )
        simulated_segments = line_segments(rollout.simulated_xy.astype(np.float32))
        if len(simulated_segments) > 0:
            rollout_state["handles"]["simulated"] = server.scene.add_line_segments(
                "/cmd_vel_debug/rollout/simulated",
                points=simulated_segments,
                colors=np.tile(np.array([[[1.0, 0.1, 0.85], [1.0, 0.1, 0.85]]], dtype=np.float32), (len(simulated_segments), 1, 1)),
                line_width=4,
            )
        rollout_state["handles"]["start"] = server.scene.add_icosphere(
            "/cmd_vel_debug/rollout/start_odom",
            position=rollout.simulated_xy[0].astype(np.float32),
            radius=0.08,
            color=(0, 220, 255),
        )
        if len(rollout.target_xy) > 0:
            target_step = max(1, len(rollout.target_xy) // 40)
            target_points = rollout.target_xy[::target_step].astype(np.float32)
            rollout_state["handles"]["targets"] = server.scene.add_point_cloud(
                "/cmd_vel_debug/rollout/controller_targets",
                points=target_points,
                colors=np.tile(np.array([[1.0, 0.72, 0.0]], dtype=np.float32), (len(target_points), 1)),
                point_size=0.045,
                point_shape="rounded",
            )
        if (
            show_rollout_occupancy is not None
            and show_rollout_occupancy.value
            and rollout.occupancy_points is not None
            and len(rollout.occupancy_points) > 0
        ):
            rollout_state["handles"]["occupancy"] = server.scene.add_point_cloud(
                "/cmd_vel_debug/rollout/occupancy_grid",
                points=rollout.occupancy_points,
                colors=rollout.occupancy_colors,
                point_size=0.035,
                point_shape="square",
            )
        if (
            show_rollout_obstacles is not None
            and show_rollout_obstacles.value
            and rollout.obstacle_points is not None
            and len(rollout.obstacle_points) > 0
        ):
            rollout_state["handles"]["obstacles"] = server.scene.add_point_cloud(
                "/cmd_vel_debug/rollout/obstacles",
                points=rollout.obstacle_points,
                colors=np.tile(np.array([[1.0, 0.05, 0.0]], dtype=np.float32), (len(rollout.obstacle_points), 1)),
                point_size=0.055,
                point_shape="square",
            )

        nonzero = int(np.count_nonzero(np.abs(rollout.cmd_vx) + np.abs(rollout.cmd_vyaw) > 1e-6))
        zero_count = len(rollout.cmd_vx) - nonzero
        if rollout_info is not None:
            rollout_info.content = (
                "**/planning/trajectory_path Controller Simulation**\n\n"
                f"`trajectory_path index`: `{rollout.path_index}` / `{path_indices[-1]}`  \n"
                f"`start sample`: `{rollout.start_sample}`  \n"
                f"`start source`: `{rollout.start_source}`  \n"
                f"`vx plant`: `e^-0.13s / (0.07s + 1)` (`tau={sim_vx_tau_s:.3f}s`, `delay={sim_vx_delay_s:.3f}s`)  \n"
                f"`yaw plant`: `0.85e^-0.06s` (`gain={sim_yaw_gain:.3f}`, `delay={sim_yaw_delay_s:.3f}s`)  \n"
                f"`path record stamp`: `{rollout.path_timestamp_ns / 1e9:.6f}` s  \n"
                f"`start odom record stamp`: `{rollout.start_odom_timestamp_ns / 1e9:.6f}` s  \n"
                f"`start odom header stamp`: `{rollout.start_odom_stamp:.6f}` s  \n"
                f"`path start/end`: `{rollout.path_start_t:.6f}` -> `{rollout.path_end_t:.6f}` s  \n"
                f"`final error`: `{rollout.final_error:.3f} m`  \n"
                f"`mean error`: `{rollout.mean_error:.3f} m`  \n"
                f"`max error`: `{rollout.max_error:.3f} m`  \n"
                f"`p95 |lateral|`: `{np.percentile(np.abs(rollout.lateral_error), 95):.3f} m`  \n"
                f"`p95 |heading|`: `{np.nanpercentile(np.abs(rollout.heading_error), 95):.3f} rad`  \n"
                f"`generated cmd vx range`: `{np.min(rollout.cmd_vx):.3f}` .. `{np.max(rollout.cmd_vx):.3f}`  \n"
                f"`generated cmd vyaw range`: `{np.min(rollout.cmd_vyaw):.3f}` .. `{np.max(rollout.cmd_vyaw):.3f}`  \n"
                f"`nonzero/zero steps`: `{nonzero}` / `{zero_count}`  \n"
                f"`occupancy points`: `{0 if rollout.occupancy_points is None else len(rollout.occupancy_points)}`  \n"
                f"`obstacle points`: `{0 if rollout.obstacle_points is None else len(rollout.obstacle_points)}`"
            )

    def selected_rollout() -> TrajectoryRollout | None:
        if rollout_slider is None or start_sample_slider is None or not path_indices or not sample_names:
            return None
        path_idx = path_indices[int(np.clip(int(rollout_slider.value), 0, len(path_indices) - 1))]
        sample = sample_names[int(np.clip(int(start_sample_slider.value), 0, len(sample_names) - 1))]
        rollout = rollout_by_key.get((path_idx, sample))
        if rollout is not None:
            return rollout
        for fallback_sample in sample_names:
            rollout = rollout_by_key.get((path_idx, fallback_sample))
            if rollout is not None:
                return rollout
        return None

    def render_selected_rollout() -> None:
        rollout = selected_rollout()
        if rollout is not None:
            render_rollout(rollout)

    if rollout_slider is not None:
        @rollout_slider.on_update
        def _(_) -> None:
            render_selected_rollout()
        @start_sample_slider.on_update
        def _(_) -> None:
            render_selected_rollout()
        @show_rollout_occupancy.on_update
        def _(_) -> None:
            render_selected_rollout()
        @show_rollout_obstacles.on_update
        def _(_) -> None:
            render_selected_rollout()

    render_selected_rollout()
    print(f"Viser server running on port {port}")
    try:
        while True:
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline debug cmd_vel_control.py from a ROS 2 bag without ros2 bag play.")
    parser.add_argument("--bag", required=True, type=Path, help="Path to a rosbag directory or .db3 file.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--print-rows", type=int, default=20, help="Print the last N evaluated odom rows.")
    parser.add_argument("--cmd-match-window-s", type=float, default=0.2, help="Nearest recorded /cmd_vel match window.")
    parser.add_argument("--max-odom-frames", type=int, default=0, help="Limit evaluated odom frames; 0 means no limit.")
    parser.add_argument("--viser", action="store_true", help="Launch an offline Viser view.")
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    parser.add_argument("--rate", type=float, default=10.0, help="Viser playback frames per second.")
    parser.add_argument("--rollout-dt", type=float, default=0.02, help="Kinematic simulation dt for each trajectory rollout.")
    parser.add_argument(
        "--rollout-start",
        choices=("vio", "path-first"),
        default="path-first",
        help="Start each trajectory rollout from latest VIO odom or from the first path pose.",
    )
    parser.add_argument("--max-rollouts", type=int, default=0, help="Limit trajectory rollouts; 0 means no limit.")
    parser.add_argument("--max-occupancy-frames", type=int, default=1000, help="Maximum occupancy frames for rollout obstacle display.")
    parser.add_argument("--occupancy-every", type=int, default=1, help="Raycast every Nth depth frame for rollout occupancy display.")
    parser.add_argument("--sim-vx-tau-s", type=float, default=0.07, help="Forward velocity first-order plant time constant.")
    parser.add_argument("--sim-vx-delay-s", type=float, default=0.13, help="Forward velocity plant delay.")
    parser.add_argument("--sim-yaw-gain", type=float, default=0.85, help="Yaw plant gain for simulated rollout.")
    parser.add_argument("--sim-yaw-delay-s", type=float, default=0.06, help="Yaw plant delay for simulated rollout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bag_path = normalize_bag_path(args.bag)
    if not bag_path.exists():
        raise FileNotFoundError(bag_path)

    topics, events, cmd_frames = read_debug_events(bag_path)
    frames = simulate(
        events,
        cmd_frames,
        match_window_s=max(0.0, args.cmd_match_window_s),
        max_odom_frames=max(0, args.max_odom_frames),
    )
    print_summary(topics, events, cmd_frames, frames, max(0, args.print_rows))
    rollouts = []
    if args.viser:
        occupancy_frames = read_occupancy_frames_for_debug(
            bag_path,
            max_frames=max(0, args.max_occupancy_frames),
            every=max(1, args.occupancy_every),
        )
        rollouts = build_trajectory_rollouts(
            events,
            dt=max(1e-3, args.rollout_dt),
            max_rollouts=max(0, args.max_rollouts),
            occupancy_frames=occupancy_frames,
            start_mode=args.rollout_start,
            vx_tau_s=max(0.0, float(args.sim_vx_tau_s)),
            vx_delay_s=max(0.0, float(args.sim_vx_delay_s)),
            yaw_gain=float(args.sim_yaw_gain),
            yaw_delay_s=max(0.0, float(args.sim_yaw_delay_s)),
        )
        if rollouts:
            final_errors = np.array([r.final_error for r in rollouts], dtype=np.float64)
            mean_errors = np.array([r.mean_error for r in rollouts], dtype=np.float64)
            print("\nKinematic trajectory rollouts:")
            print(f"  rollouts: {len(rollouts)}")
            print(f"  rollout start: {args.rollout_start}")
            print(f"  vx plant: tau={args.sim_vx_tau_s:.3f}s, delay={args.sim_vx_delay_s:.3f}s")
            print(f"  yaw plant: gain={args.sim_yaw_gain:.3f}, delay={args.sim_yaw_delay_s:.3f}s")
            print(f"  occupancy frames: {len(occupancy_frames)}")
            print(f"  mean final error: {np.nanmean(final_errors):.3f} m")
            print(f"  p95 final error: {np.nanpercentile(final_errors, 95):.3f} m")
            print(f"  mean mean-error: {np.nanmean(mean_errors):.3f} m")
        else:
            print("\nKinematic trajectory rollouts: none")
    if args.csv is not None:
        write_csv(frames, args.csv)
        print(f"\nWrote CSV: {args.csv}")
    if args.viser:
        run_viser(
            frames,
            rollouts,
            port=args.port,
            rate=max(0.1, args.rate),
            sim_vx_tau_s=max(0.0, float(args.sim_vx_tau_s)),
            sim_vx_delay_s=max(0.0, float(args.sim_vx_delay_s)),
            sim_yaw_gain=float(args.sim_yaw_gain),
            sim_yaw_delay_s=max(0.0, float(args.sim_yaw_delay_s)),
        )


if __name__ == "__main__":
    main()
