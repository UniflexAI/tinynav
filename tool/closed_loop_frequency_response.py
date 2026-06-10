from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from tinynav.core.math_utils import pose_msg2np


DEFAULT_A_VALUES = "0.5,1.0,1.5"
DEFAULT_B_VALUES = "0.25,0.5,0.75"
DEFAULT_FREQUENCIES = "0.05,0.10,0.20,0.30"
CAMERA_OFFSET = np.array([0.0, 0.0, 0.35], dtype=np.float64)


@dataclass
class PoseSample:
    recv_mono_s: float
    ros_stamp_s: float
    camera_position: np.ndarray
    rotation: np.ndarray
    robot_position: np.ndarray
    yaw_rad: float
    unwrapped_yaw_rad: float


@dataclass
class CmdSample:
    recv_mono_s: float
    vx: float
    vyaw: float


@dataclass
class RawSample:
    a_m: float
    b_m: float
    frequency_hz: float
    t_segment_s: float
    ros_stamp_s: float
    ref_x: float
    ref_y: float
    ref_yaw: float
    ref_vx: float
    ref_vyaw: float
    vio_x: float
    vio_y: float
    vio_z: float
    vio_yaw: float
    nearest_ref_idx: int
    nearest_ref_t: float
    along_error_m: float
    lateral_error_m: float
    heading_error_rad: float
    cmd_vx: float
    cmd_vyaw: float
    measured_speed_mps: float = float("nan")
    measured_vyaw_radps: float = float("nan")


@dataclass
class SummaryRow:
    a_m: float
    b_m: float
    frequency_hz: float
    samples: int
    duration_s: float
    rms_lateral_error_m: float
    p95_lateral_error_m: float
    max_lateral_error_m: float
    rms_heading_error_rad: float
    p95_heading_error_rad: float
    max_heading_error_rad: float
    completion_ratio: float
    mean_cmd_vx: float
    max_abs_cmd_vx: float
    mean_abs_cmd_vyaw: float
    max_abs_cmd_vyaw: float
    vx_saturation_ratio: float
    vyaw_saturation_ratio: float
    mean_measured_speed_mps: float
    max_measured_speed_mps: float
    mean_abs_measured_vyaw_radps: float
    max_abs_measured_vyaw_radps: float
    progress_lag_s: float
    ok: bool


def parse_float_list(value: str) -> list[float]:
    values = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    if any(x <= 0.0 for x in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def path_token(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def values_token(values: list[float]) -> str:
    return "-".join(path_token(v) for v in values)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_rotation_matrix(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array(
        [
            [0.0, -s, c],
            [0.0, c, s],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def yaw_from_rotation(rotation: np.ndarray) -> float:
    forward = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if np.linalg.norm(forward[:2]) <= 1e-6:
        return 0.0
    return math.atan2(float(forward[1]), float(forward[0]))


def make_pose_stamped(x: float, y: float, z: float, yaw: float, stamp, frame_id: str) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = float(z)
    quat = R.from_matrix(yaw_rotation_matrix(yaw)).as_quat()
    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])
    return msg


def lissajous_local(a_m: float, b_m: float, frequency_hz: float, t: np.ndarray):
    omega = 2.0 * math.pi * frequency_hz
    theta = omega * t
    x = a_m * np.sin(theta)
    y = b_m * np.sin(2.0 * theta)
    dx = a_m * omega * np.cos(theta)
    dy = 2.0 * b_m * omega * np.cos(2.0 * theta)
    ddx = -a_m * omega * omega * np.sin(theta)
    ddy = -4.0 * b_m * omega * omega * np.sin(2.0 * theta)
    yaw = np.unwrap(np.arctan2(dy, dx))
    speed = np.sqrt(dx * dx + dy * dy)
    denom = np.maximum(speed * speed, 1e-9)
    vyaw = (dx * ddy - dy * ddx) / denom
    return x, y, yaw, speed, vyaw


def rotate_translate_xy(x: np.ndarray, y: np.ndarray, anchor_xy: np.ndarray, rotation_yaw: float):
    c = math.cos(rotation_yaw)
    s = math.sin(rotation_yaw)
    world_x = anchor_xy[0] + c * x - s * y
    world_y = anchor_xy[1] + s * x + c * y
    return world_x, world_y


def lissajous_reference(
    a_m: float,
    b_m: float,
    frequency_hz: float,
    t: np.ndarray,
    anchor_xy: np.ndarray,
    anchor_z: float,
    anchor_yaw: float,
):
    x, y, yaw_local, speed, vyaw = lissajous_local(a_m, b_m, frequency_hz, t)
    tangent0 = math.atan2(2.0 * b_m, a_m)
    rotation_yaw = anchor_yaw - tangent0
    world_x, world_y = rotate_translate_xy(x, y, anchor_xy, rotation_yaw)
    world_yaw = np.array([wrap_angle(v + rotation_yaw) for v in yaw_local], dtype=np.float64)
    z = np.full_like(world_x, anchor_z, dtype=np.float64)
    return np.column_stack((world_x, world_y, z, world_yaw, speed, vyaw, t))


def nearest_reference_error(sample: PoseSample, ref_path: np.ndarray):
    if len(ref_path) == 0:
        return -1, 0.0, 0.0, 0.0, 0.0
    delta = sample.robot_position[:2][None, :] - ref_path[:, :2]
    dist = np.linalg.norm(delta, axis=1)
    idx = int(np.argmin(dist))
    ref = ref_path[idx]
    yaw = float(ref[3])
    dx = sample.robot_position[0] - ref[0]
    dy = sample.robot_position[1] - ref[1]
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    along = cy * dx + sy * dy
    lateral = -sy * dx + cy * dy
    heading = wrap_angle(sample.yaw_rad - yaw)
    return idx, float(ref[6]), float(along), float(lateral), float(heading)


def finite_or_nan(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr


def percentile_abs(values, q: float) -> float:
    arr = finite_or_nan(values)
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(np.abs(arr), q))


def rms(values) -> float:
    arr = finite_or_nan(values)
    if len(arr) == 0:
        return float("nan")
    return float(math.sqrt(np.mean(arr * arr)))


def estimate_progress_lag(samples: list[RawSample], frequency_hz: float) -> float:
    if len(samples) < 4:
        return float("nan")
    t = np.array([s.t_segment_s for s in samples], dtype=np.float64)
    ref = np.array([s.nearest_ref_t for s in samples], dtype=np.float64)
    lag = t - ref
    period = 1.0 / frequency_hz
    lag = ((lag + 0.5 * period) % period) - 0.5 * period
    return float(np.median(lag))


def summarize_case(
    a_m: float,
    b_m: float,
    frequency_hz: float,
    expected_duration_s: float,
    samples: list[RawSample],
    max_vx: float,
    max_vyaw: float,
) -> SummaryRow:
    if not samples:
        return SummaryRow(a_m, b_m, frequency_hz, 0, 0.0, *(float("nan"),) * 18, ok=False)
    lateral = np.array([s.lateral_error_m for s in samples], dtype=np.float64)
    heading = np.array([s.heading_error_rad for s in samples], dtype=np.float64)
    cmd_vx = np.array([s.cmd_vx for s in samples], dtype=np.float64)
    cmd_vyaw = np.array([s.cmd_vyaw for s in samples], dtype=np.float64)
    speed = finite_or_nan([s.measured_speed_mps for s in samples])
    vyaw = finite_or_nan([s.measured_vyaw_radps for s in samples])
    duration = float(samples[-1].t_segment_s - samples[0].t_segment_s) if len(samples) > 1 else 0.0
    completion = min(1.0, max(0.0, samples[-1].nearest_ref_t / max(1e-6, expected_duration_s)))
    return SummaryRow(
        a_m=a_m,
        b_m=b_m,
        frequency_hz=frequency_hz,
        samples=len(samples),
        duration_s=duration,
        rms_lateral_error_m=rms(lateral),
        p95_lateral_error_m=percentile_abs(lateral, 95),
        max_lateral_error_m=float(np.max(np.abs(lateral))),
        rms_heading_error_rad=rms(heading),
        p95_heading_error_rad=percentile_abs(heading, 95),
        max_heading_error_rad=float(np.max(np.abs(heading))),
        completion_ratio=completion,
        mean_cmd_vx=float(np.mean(cmd_vx)),
        max_abs_cmd_vx=float(np.max(np.abs(cmd_vx))),
        mean_abs_cmd_vyaw=float(np.mean(np.abs(cmd_vyaw))),
        max_abs_cmd_vyaw=float(np.max(np.abs(cmd_vyaw))),
        vx_saturation_ratio=float(np.mean(np.abs(cmd_vx) >= max_vx - 1e-3)),
        vyaw_saturation_ratio=float(np.mean(np.abs(cmd_vyaw) >= max_vyaw - 1e-3)),
        mean_measured_speed_mps=float(np.mean(speed)) if len(speed) else float("nan"),
        max_measured_speed_mps=float(np.max(speed)) if len(speed) else float("nan"),
        mean_abs_measured_vyaw_radps=float(np.mean(np.abs(vyaw))) if len(vyaw) else float("nan"),
        max_abs_measured_vyaw_radps=float(np.max(np.abs(vyaw))) if len(vyaw) else float("nan"),
        progress_lag_s=estimate_progress_lag(samples, frequency_hz),
        ok=True,
    )


class ClosedLoopFrequencyResponse(Node):
    def __init__(self, args):
        super().__init__("closed_loop_frequency_response")
        self.args = args
        self.latest_pose: PoseSample | None = None
        self.latest_cmd = CmdSample(time.monotonic(), 0.0, 0.0)
        self._last_pose_stamp_s: float | None = None
        self._last_raw_yaw: float | None = None
        self._unwrapped_yaw = 0.0
        self._last_robot_pos: np.ndarray | None = None
        self._last_robot_yaw: float | None = None
        self._last_velocity_stamp_s: float | None = None
        self._measured_speed = float("nan")
        self._measured_vyaw = float("nan")

        self.create_subscription(PoseStamped, args.vio_topic, self._pose_cb, 100)
        self.create_subscription(Twist, args.cmd_topic, self._cmd_cb, 100)
        self.path_pub = None
        if not args.dry_run:
            self.path_pub = self.create_publisher(RosPath, args.path_topic, 10)

    def _pose_cb(self, msg: PoseStamped):
        pose = pose_msg2np(msg)
        rotation = pose[:3, :3]
        camera_position = pose[:3, 3]
        robot_position = camera_position - rotation @ CAMERA_OFFSET
        yaw = yaw_from_rotation(rotation)
        stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_raw_yaw is None:
            self._last_raw_yaw = yaw
            self._unwrapped_yaw = yaw
        else:
            self._unwrapped_yaw += wrap_angle(yaw - self._last_raw_yaw)
            self._last_raw_yaw = yaw
        if self._last_robot_pos is not None and self._last_velocity_stamp_s is not None:
            dt = stamp_s - self._last_velocity_stamp_s
            if dt > 1e-4:
                self._measured_speed = float(np.linalg.norm(robot_position[:2] - self._last_robot_pos[:2]) / dt)
                self._measured_vyaw = float(wrap_angle(yaw - self._last_robot_yaw) / dt)
        self._last_robot_pos = robot_position.copy()
        self._last_robot_yaw = yaw
        self._last_velocity_stamp_s = stamp_s
        self._last_pose_stamp_s = stamp_s
        self.latest_pose = PoseSample(
            recv_mono_s=time.monotonic(),
            ros_stamp_s=stamp_s,
            camera_position=camera_position.copy(),
            rotation=rotation.copy(),
            robot_position=robot_position.copy(),
            yaw_rad=yaw,
            unwrapped_yaw_rad=self._unwrapped_yaw,
        )

    def _cmd_cb(self, msg: Twist):
        self.latest_cmd = CmdSample(
            recv_mono_s=time.monotonic(),
            vx=float(msg.linear.x),
            vyaw=float(msg.angular.z),
        )

    def wait_for_pose(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_pose is not None:
                return True
        return False

    def warn_other_path_publishers(self):
        if self.args.dry_run:
            return
        deadline = time.monotonic() + 1.0
        infos = []
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            infos = self.get_publishers_info_by_topic(self.args.path_topic)
            if infos:
                break
        others = [info for info in infos if info.node_name != self.get_name()]
        if others:
            names = ", ".join(sorted({f"/{info.node_name}" for info in others}))
            msg = f"other {self.args.path_topic} publishers detected: {names}; stop planning_node before this test."
            if self.args.allow_other_path_publishers:
                self.get_logger().warning(msg)
            else:
                raise RuntimeError(msg)

    def publish_path_window(self, full_ref: np.ndarray, segment_start_mono_s: float):
        if self.args.dry_run or self.path_pub is None or self.latest_pose is None:
            return
        elapsed = max(0.0, time.monotonic() - segment_start_mono_s)
        horizon_t = elapsed + np.arange(0.0, self.args.path_duration_s + 1e-9, self.args.path_dt)
        msg = RosPath()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        max_t = float(full_ref[-1, 6])
        for rel_t in horizon_t:
            t = min(rel_t, max_t)
            idx = int(np.searchsorted(full_ref[:, 6], t, side="left"))
            idx = min(max(0, idx), len(full_ref) - 1)
            row = full_ref[idx]
            stamp_s = self.latest_pose.ros_stamp_s + max(0.0, rel_t - elapsed)
            sec = int(math.floor(stamp_s))
            nanosec = int(round((stamp_s - sec) * 1e9))
            if nanosec >= 1000000000:
                sec += 1
                nanosec -= 1000000000
            stamp = Time(nanoseconds=sec * 1000000000 + nanosec).to_msg()
            msg.poses.append(make_pose_stamped(row[0], row[1], row[2], row[3], stamp, self.args.frame_id))
        self.path_pub.publish(msg)

    def publish_static_path_for(self, duration_s: float):
        if self.args.dry_run or self.path_pub is None:
            return
        period_s = 1.0 / max(0.1, float(self.args.path_hz))
        end = time.monotonic() + duration_s
        next_pub = time.monotonic()
        while time.monotonic() < end and rclpy.ok():
            now = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.0)
            pose = self.latest_pose
            if pose is not None and now >= next_pub:
                ref = lissajous_reference(0.01, 0.005, 0.05, np.array([0.0, 0.2]), pose.robot_position[:2], pose.robot_position[2], pose.yaw_rad)
                self.publish_path_window(ref, now)
                next_pub += period_s
            time.sleep(min(0.01, max(0.0, next_pub - now)))

    def run_case(self, a_m: float, b_m: float, frequency_hz: float) -> tuple[list[RawSample], SummaryRow]:
        if self.latest_pose is None:
            raise RuntimeError(f"no {self.args.vio_topic} pose available")
        self.publish_static_path_for(self.args.zero_duration_s)
        rclpy.spin_once(self, timeout_sec=0.05)
        start_pose = self.latest_pose
        if start_pose is None:
            raise RuntimeError(f"no {self.args.vio_topic} pose available after zero hold")

        duration_s = self.args.cycles / frequency_hz
        ref_t = np.arange(0.0, duration_s + self.args.path_dt * 0.5, self.args.path_dt)
        full_ref = lissajous_reference(
            a_m,
            b_m,
            frequency_hz,
            ref_t,
            start_pose.robot_position[:2],
            start_pose.robot_position[2],
            start_pose.yaw_rad,
        )
        self.save_path(a_m, b_m, frequency_hz, full_ref)

        segment_start = time.monotonic()
        next_path_pub = segment_start
        samples: list[RawSample] = []
        self.get_logger().info(
            f"lissajous A={a_m:.3f}m B={b_m:.3f}m f={frequency_hz:.3f}Hz duration={duration_s:.2f}s"
        )
        while rclpy.ok():
            now = time.monotonic()
            elapsed = now - segment_start
            if elapsed >= duration_s:
                break
            pose = self.latest_pose
            if pose is not None and now - pose.recv_mono_s > self.args.vio_timeout_s:
                self.get_logger().error(
                    f"{self.args.vio_topic} timeout: last update {now - pose.recv_mono_s:.3f}s ago"
                )
                break
            if now >= next_path_pub:
                self.publish_path_window(full_ref, segment_start)
                next_path_pub += 1.0 / float(self.args.path_hz)

            rclpy.spin_once(self, timeout_sec=min(0.005, max(0.0, next_path_pub - now)))
            pose = self.latest_pose
            if pose is None:
                continue
            idx, nearest_t, along, lateral, heading = nearest_reference_error(pose, full_ref)
            ref_idx = min(max(idx, 0), len(full_ref) - 1)
            ref = full_ref[ref_idx]
            cmd = self.latest_cmd
            sample = RawSample(
                a_m=a_m,
                b_m=b_m,
                frequency_hz=frequency_hz,
                t_segment_s=elapsed,
                ros_stamp_s=pose.ros_stamp_s,
                ref_x=float(ref[0]),
                ref_y=float(ref[1]),
                ref_yaw=float(ref[3]),
                ref_vx=float(ref[4]),
                ref_vyaw=float(ref[5]),
                vio_x=float(pose.robot_position[0]),
                vio_y=float(pose.robot_position[1]),
                vio_z=float(pose.robot_position[2]),
                vio_yaw=float(pose.yaw_rad),
                nearest_ref_idx=idx,
                nearest_ref_t=nearest_t,
                along_error_m=along,
                lateral_error_m=lateral,
                heading_error_rad=heading,
                cmd_vx=cmd.vx,
                cmd_vyaw=cmd.vyaw,
                measured_speed_mps=self._measured_speed,
                measured_vyaw_radps=self._measured_vyaw,
            )
            samples.append(sample)
            if np.linalg.norm(pose.robot_position[:2] - start_pose.robot_position[:2]) > self.args.max_abs_displacement_m:
                self.get_logger().error("displacement limit reached; stopping case")
                break
            if abs(pose.unwrapped_yaw_rad - start_pose.unwrapped_yaw_rad) > self.args.max_abs_yaw_rad:
                self.get_logger().error("yaw limit reached; stopping case")
                break

        self.publish_static_path_for(self.args.zero_duration_s)
        return samples, summarize_case(
            a_m,
            b_m,
            frequency_hz,
            duration_s,
            samples,
            self.args.max_cmd_vx,
            self.args.max_cmd_vyaw,
        )

    def save_path(self, a_m: float, b_m: float, frequency_hz: float, full_ref: np.ndarray):
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        name = f"path_A{path_token(a_m)}_B{path_token(b_m)}_f{path_token(frequency_hz)}.npy"
        np.save(self.args.output_dir / name, full_ref)


def write_raw_csv(path: Path, rows: list[RawSample]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RawSample.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def write_summary_csv(path: Path, rows: list[SummaryRow]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(SummaryRow.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def resolve_output_paths(args):
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    param = (
        f"lissajous_A{values_token(args.a_values)}"
        f"_B{values_token(args.b_values)}"
        f"_f{values_token(args.frequencies)}"
    )
    run_dir = args.output_dir / f"{param}_{run_stamp}"
    args.output_dir = run_dir
    if args.raw_output is None:
        args.raw_output = run_dir / f"{param}_raw.csv"
    if args.summary_output is None:
        args.summary_output = run_dir / f"{param}_summary.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Closed-loop Lissajous trajectory tracking test via /planning/trajectory_path.")
    parser.add_argument("--mode", choices=("lissajous",), default="lissajous")
    parser.add_argument("--a-values", type=parse_float_list, default=parse_float_list(DEFAULT_A_VALUES))
    parser.add_argument("--b-values", type=parse_float_list, default=parse_float_list(DEFAULT_B_VALUES))
    parser.add_argument("--frequencies", type=parse_float_list, default=parse_float_list(DEFAULT_FREQUENCIES))
    parser.add_argument("--cycles", type=float, default=2.0)
    parser.add_argument("--path-hz", type=float, default=5.0)
    parser.add_argument("--path-duration-s", type=float, default=2.0)
    parser.add_argument("--path-dt", type=float, default=0.1)
    parser.add_argument("--vio-topic", default="/insight/vio_100hz")
    parser.add_argument("--cmd-topic", default="/cmd_vel")
    parser.add_argument("--path-topic", default="/planning/trajectory_path")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--vio-timeout-s", type=float, default=0.5)
    parser.add_argument("--zero-duration-s", type=float, default=1.0)
    parser.add_argument("--max-abs-displacement-m", type=float, default=1.5)
    parser.add_argument("--max-abs-yaw-rad", type=float, default=2.0)
    parser.add_argument("--max-cmd-vx", type=float, default=0.6)
    parser.add_argument("--max-cmd-vyaw", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/closed_loop_frequency_response_logs"))
    parser.add_argument("--raw-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--allow-other-path-publishers", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Do not publish /planning/trajectory_path.")
    return parser


def validate_args(args):
    if args.cycles <= 0.0:
        raise ValueError("--cycles must be positive")
    if args.path_hz <= 0.0 or args.path_hz > 10.0:
        raise ValueError("--path-hz must be in (0, 10]")
    if args.path_duration_s <= 0.0:
        raise ValueError("--path-duration-s must be positive")
    if args.path_dt <= 0.0:
        raise ValueError("--path-dt must be positive")
    if args.zero_duration_s < 0.0:
        raise ValueError("--zero-duration-s must be non-negative")


def main(args=None):
    parsed = build_arg_parser().parse_args(args=args)
    validate_args(parsed)
    resolve_output_paths(parsed)

    rclpy.init()
    node = ClosedLoopFrequencyResponse(parsed)
    all_samples: list[RawSample] = []
    summaries: list[SummaryRow] = []
    try:
        if not node.wait_for_pose(timeout_s=5.0):
            raise RuntimeError(f"timed out waiting for {parsed.vio_topic}")
        node.warn_other_path_publishers()
        for a_m in parsed.a_values:
            for b_m in parsed.b_values:
                for frequency_hz in parsed.frequencies:
                    samples, summary = node.run_case(a_m, b_m, frequency_hz)
                    all_samples.extend(samples)
                    summaries.append(summary)
                    node.get_logger().info(
                        f"summary A={a_m:.3f} B={b_m:.3f} f={frequency_hz:.3f}: "
                        f"rms_lat={summary.rms_lateral_error_m:.3f}m "
                        f"p95_lat={summary.p95_lateral_error_m:.3f}m "
                        f"rms_heading={summary.rms_heading_error_rad:.3f}rad "
                        f"max_cmd_vx={summary.max_abs_cmd_vx:.3f} "
                        f"max_cmd_vyaw={summary.max_abs_cmd_vyaw:.3f}"
                    )
    except KeyboardInterrupt:
        node.get_logger().warning("interrupted")
    finally:
        node.publish_static_path_for(parsed.zero_duration_s)
        write_raw_csv(parsed.raw_output, all_samples)
        write_summary_csv(parsed.summary_output, summaries)
        node.get_logger().info(f"wrote raw CSV: {parsed.raw_output}")
        node.get_logger().info(f"wrote summary CSV: {parsed.summary_output}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
