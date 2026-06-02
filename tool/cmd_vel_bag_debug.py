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


DEBUG_TOPICS = {
    "/slam/odometry",
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
    zero_reason: str
    path_available: bool
    path_len: int
    path_is_static: bool
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


def odom_to_matrix(msg) -> np.ndarray:
    return pose_to_matrix(msg.pose.pose)


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
        self.time_lookahead_s = 0.1

    def now_sec(self, fallback_timestamp_ns: int) -> float:
        if self.odom_stamp_sec is not None:
            return float(self.odom_stamp_sec)
        return float(fallback_timestamp_ns) * 1e-9

    def odom_callback(self, msg, timestamp_ns: int) -> DebugFrame:
        measured_pose = odom_to_matrix(msg)
        measured_position = measured_pose[:3, 3]
        measured_rotation = measured_pose[:3, :3]

        if not self.odom_pose_initialized:
            self.position = measured_position
            self.rotation = measured_rotation
            self.odom_pose_initialized = True
        else:
            alpha = 0.35
            self.position = (1.0 - alpha) * self.position + alpha * measured_position
            self.rotation = measured_rotation

        self.odom_stamp_sec = stamp_sec(msg.header.stamp)
        return self.control_loop(timestamp_ns)

    def path_callback(self, msg, timestamp_ns: int) -> None:
        now = self.now_sec(timestamp_ns)
        if self.last_traj_update_sec is not None and now - self.last_traj_update_sec < 0.2:
            return

        new_ref, new_points = self.rebuild_path(msg)
        if new_ref is None:
            self.path_ref = None
            self.path_points = None
            self.track_idx = 0
        elif self.path_ref is None or len(self.path_ref) == 0:
            self.path_ref = new_ref
            self.path_points = new_points
            self.track_idx = 0
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

    def control_loop(self, timestamp_ns: int) -> DebugFrame:
        odom_yaw = yaw_from_rotation(self.rotation)
        base = dict(
            timestamp_ns=timestamp_ns,
            odom_stamp=float(self.odom_stamp_sec if self.odom_stamp_sec is not None else timestamp_ns * 1e-9),
            odom_x=float(self.position[0]),
            odom_y=float(self.position[1]),
            odom_z=float(self.position[2]),
            odom_yaw=float(odom_yaw),
            vx_expected=0.0,
            vyaw_expected=0.0,
            path_available=self.path_ref is not None,
            path_len=0 if self.path_ref is None else int(len(self.path_ref)),
            path_is_static=self.path_is_static(),
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

        camera_offset = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        robot_pos = self.position - self.rotation @ camera_offset
        robot_yaw = odom_yaw

        target, target_idx = self.find_tracking_target(robot_pos, robot_yaw, self.odom_stamp_sec)
        if target is None:
            return DebugFrame(**base, zero_reason="no valid tracking target")

        tx, ty, heading_err = self.target_error(robot_pos, robot_yaw, target)
        v_ref = float(target[3])
        w_ref = float(target[4])

        b = 2.0
        zeta = 0.7
        k = 2.0 * zeta * math.sqrt(w_ref * w_ref + b * v_ref * v_ref)
        v = v_ref * math.cos(heading_err) + k * tx
        wz = w_ref + k * heading_err + b * v_ref * self.sinc(heading_err) * ty
        v = float(np.clip(v, -0.2, 0.6))
        wz = float(np.clip(wz, -0.8, 0.8))

        heading_to_goal = wrap_angle(float(self.path_ref[-1, 2]) - robot_yaw)
        zero_reason = ""
        if np.linalg.norm(robot_pos[:2] - self.path_ref[-1, :2]) < 0.1 and abs(heading_to_goal) < 0.1:
            v = 0.0
            wz = 0.0
            zero_reason = "target reached"
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
        elif event.topic == "/slam/odometry":
            frames.append(controller.odom_callback(event.msg, event.timestamp_ns))
            if max_odom_frames > 0 and len(frames) >= max_odom_frames:
                break
    attach_recorded_cmds(frames, cmd_frames, match_window_s)
    return frames


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
            f"recorded={recorded} reason={reason} "
            f"target_idx={frame.target_idx} tx={frame.tx:.3f} ty={frame.ty:.3f} heading={frame.heading_err:.3f}"
        )


def run_viser(frames: list[DebugFrame], port: int, rate: float) -> None:
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

    state = {"idx": 0, "play": False, "last_update": time.monotonic(), "handles": {}}
    with server.gui.add_folder("Cmd Vel Debug") as _:
        slider = server.gui.add_slider("Frame", min=0, max=len(frames) - 1, step=1, initial_value=0)
        play = server.gui.add_checkbox("Play", initial_value=False)
        rate_slider = server.gui.add_slider("Rate", min=0.1, max=max(5.0, rate), step=0.1, initial_value=rate)
        info = server.gui.add_markdown("")

    def clear() -> None:
        handles = state["handles"]
        for name in ("odom", "path", "target"):
            handle = handles.pop(name, None)
            if handle is not None:
                try:
                    handle.remove()
                except Exception:
                    pass

    def render(idx: int) -> None:
        clear()
        idx = int(np.clip(idx, 0, len(frames) - 1))
        frame = frames[idx]
        state["idx"] = idx
        state["handles"]["odom"] = server.scene.add_icosphere(
            "/cmd_vel_debug/odom",
            position=np.array([frame.odom_x, frame.odom_y, frame.odom_z], dtype=np.float32),
            radius=0.06,
            color=(0, 220, 255),
        )
        if frame.path_points is not None and len(frame.path_points) > 1:
            segments = line_segments(frame.path_points.astype(np.float32))
            state["handles"]["path"] = server.scene.add_line_segments(
                "/cmd_vel_debug/path",
                points=segments,
                colors=np.tile(np.array([[[0.2, 1.0, 0.25], [0.2, 1.0, 0.25]]], dtype=np.float32), (len(segments), 1, 1)),
                line_width=4,
            )
        if frame.target_point is not None:
            state["handles"]["target"] = server.scene.add_icosphere(
                "/cmd_vel_debug/tracking_target",
                position=frame.target_point.astype(np.float32),
                radius=0.08,
                color=(255, 180, 0),
            )
        recorded = "none"
        if frame.vx_recorded is not None and frame.vyaw_recorded is not None:
            recorded = f"`vx={frame.vx_recorded:.3f}`, `vyaw={frame.vyaw_recorded:.3f}`, `dt={frame.cmd_match_dt:.3f}s`"
        info.content = (
            "**Cmd Vel Debug**\n\n"
            f"`frame`: `{idx}`  \n"
            f"`time`: `{frame.timestamp_ns / 1e9:.3f}`  \n"
            f"`expected`: `vx={frame.vx_expected:.3f}`, `vyaw={frame.vyaw_expected:.3f}`  \n"
            f"`recorded`: {recorded}  \n"
            f"`zero_reason`: `{frame.zero_reason or 'nonzero'}`  \n"
            f"`target_idx`: `{frame.target_idx}`  \n"
            f"`tx`: `{frame.tx:.3f}`, `ty`: `{frame.ty:.3f}`, `heading_err`: `{frame.heading_err:.3f}`  \n"
            f"`v_ref`: `{frame.v_ref:.3f}`, `w_ref`: `{frame.w_ref:.3f}`"
        )

    @slider.on_update
    def _(_) -> None:
        render(int(slider.value))

    @play.on_update
    def _(_) -> None:
        state["play"] = bool(play.value)
        state["last_update"] = time.monotonic()

    render(0)
    print(f"Viser server running on port {port}")
    try:
        while True:
            if state["play"]:
                now = time.monotonic()
                step_dt = max(0.02, 1.0 / max(0.1, float(rate_slider.value)))
                if now - state["last_update"] >= step_dt:
                    next_idx = (state["idx"] + 1) % len(frames)
                    slider.value = next_idx
                    render(next_idx)
                    state["last_update"] = now
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
    if args.csv is not None:
        write_csv(frames, args.csv)
        print(f"\nWrote CSV: {args.csv}")
    if args.viser:
        run_viser(frames, port=args.port, rate=max(0.1, args.rate))


if __name__ == "__main__":
    main()
