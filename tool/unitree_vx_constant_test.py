from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node

from tinynav.core.math_utils import pose_msg2np


@dataclass
class PoseSample:
    recv_mono_s: float
    ros_stamp_s: float
    position: np.ndarray
    rotation: np.ndarray


@dataclass
class RawSample:
    t_segment_s: float
    ros_stamp_s: float
    cmd_vx: float
    x: float
    y: float
    z: float
    projected_s_m: float


class UnitreeVxConstantTest(Node):
    def __init__(self, args):
        super().__init__("unitree_vx_constant_test")
        self.args = args
        self.latest_pose: PoseSample | None = None
        self._last_recorded_stamp_s: float | None = None

        self.create_subscription(PoseStamped, "/insight/vio_100hz", self._pose_cb, 100)
        self.cmd_pub = None
        if not args.dry_run:
            self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def _pose_cb(self, msg: PoseStamped):
        pose = pose_msg2np(msg)
        self.latest_pose = PoseSample(
            recv_mono_s=time.monotonic(),
            ros_stamp_s=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            position=pose[:3, 3].copy(),
            rotation=pose[:3, :3].copy(),
        )

    def wait_for_pose(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_pose is not None:
                return True
        return False

    def warn_other_cmd_vel_publishers(self):
        if self.args.dry_run:
            return
        deadline = time.monotonic() + 1.0
        infos = []
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            infos = self.get_publishers_info_by_topic("/cmd_vel")
            if infos:
                break
        others = [info for info in infos if info.node_name != self.get_name()]
        if others:
            names = ", ".join(sorted({f"/{info.node_name}" for info in others}))
            self.get_logger().warning(
                f"other /cmd_vel publishers detected: {names}; stop planner/teleop before live constant test."
            )

    def publish_cmd(self, vx: float):
        if self.args.dry_run or self.cmd_pub is None:
            return
        msg = Twist()
        msg.linear.x = float(vx)
        self.cmd_pub.publish(msg)

    def publish_zero_for(self, duration_s: float):
        period_s = 1.0 / float(self.args.cmd_hz)
        end = time.monotonic() + duration_s
        next_pub = time.monotonic()
        while time.monotonic() < end and rclpy.ok():
            now = time.monotonic()
            if now >= next_pub:
                self.publish_cmd(0.0)
                next_pub += period_s
            rclpy.spin_once(self, timeout_sec=min(0.01, max(0.0, next_pub - now)))

    @staticmethod
    def forward_axis_from_pose(pose: PoseSample) -> np.ndarray:
        axis = pose.rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis[2] = 0.0
        norm = float(np.linalg.norm(axis[:2]))
        if norm <= 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return axis / norm

    def record_new_pose(
        self,
        segment_start_mono_s: float,
        start_position: np.ndarray,
        forward_axis: np.ndarray,
    ) -> RawSample | None:
        pose = self.latest_pose
        if pose is None:
            return None
        if self._last_recorded_stamp_s is not None and pose.ros_stamp_s <= self._last_recorded_stamp_s:
            return None
        self._last_recorded_stamp_s = pose.ros_stamp_s

        t_segment = max(0.0, pose.recv_mono_s - segment_start_mono_s)
        projected = float(np.dot(pose.position - start_position, forward_axis))
        return RawSample(
            t_segment_s=t_segment,
            ros_stamp_s=pose.ros_stamp_s,
            cmd_vx=self.args.velocity,
            x=float(pose.position[0]),
            y=float(pose.position[1]),
            z=float(pose.position[2]),
            projected_s_m=projected,
        )

    def run_test(self) -> list[RawSample]:
        if self.latest_pose is None:
            raise RuntimeError("no /insight/vio_100hz pose available")

        self.publish_zero_for(self.args.zero_duration_s)
        start_pose = self.latest_pose
        if start_pose is None:
            raise RuntimeError("no /insight/vio_100hz pose available after zero hold")

        start_position = start_pose.position.copy()
        forward_axis = self.forward_axis_from_pose(start_pose)
        expected_translation = self.args.velocity * self.args.duration_s
        period_s = 1.0 / float(self.args.cmd_hz)
        segment_start = time.monotonic()
        next_pub = segment_start
        self._last_recorded_stamp_s = None
        samples: list[RawSample] = []

        self.get_logger().info(
            f"constant vx={self.args.velocity:.3f}m/s duration={self.args.duration_s:.2f}s "
            f"expected_translation={expected_translation:.3f}m"
        )
        while rclpy.ok():
            now = time.monotonic()
            t_segment = now - segment_start
            if t_segment >= self.args.duration_s:
                break

            pose = self.latest_pose
            if pose is not None and now - pose.recv_mono_s > self.args.vio_timeout_s:
                self.get_logger().error(
                    f"/insight/vio_100hz timeout: last update {now - pose.recv_mono_s:.3f}s ago"
                )
                break

            if now >= next_pub:
                self.publish_cmd(self.args.velocity)
                next_pub += period_s

            rclpy.spin_once(self, timeout_sec=min(0.005, max(0.0, next_pub - now)))
            sample = self.record_new_pose(segment_start, start_position, forward_axis)
            if sample is not None:
                samples.append(sample)
                if abs(sample.projected_s_m) >= self.args.max_abs_displacement_m:
                    self.get_logger().error(
                        f"displacement limit reached at {sample.projected_s_m:.3f}m"
                    )
                    break

        self.publish_zero_for(self.args.zero_duration_s)
        return samples


def analyze(samples: list[RawSample], expected_translation: float):
    if len(samples) < 2:
        return None

    times = np.array([s.t_segment_s for s in samples], dtype=np.float64)
    projected = np.array([s.projected_s_m for s in samples], dtype=np.float64)
    measured_translation = float(projected[-1])
    elapsed_s = float(times[-1] - times[0])

    x = np.column_stack([times, np.ones_like(times)])
    slope, intercept = np.linalg.lstsq(x, projected, rcond=None)[0]
    fitted_translation = float((slope * times[-1] + intercept) - (slope * times[0] + intercept))

    return {
        "samples": len(samples),
        "elapsed_s": elapsed_s,
        "measured_translation_m": measured_translation,
        "fitted_velocity_mps": float(slope),
        "fitted_translation_m": fitted_translation,
        "expected_translation_m": float(expected_translation),
        "translation_error_m": float(measured_translation - expected_translation),
    }


def write_raw_csv(path: Path, rows: list[RawSample]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "t_segment_s",
        "ros_stamp_s",
        "cmd_vx",
        "x",
        "y",
        "z",
        "projected_s_m",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Command a constant Unitree forward velocity and measure VIO translation."
    )
    parser.add_argument("--velocity", type=float, default=0.20, help="Constant vx command in m/s.")
    parser.add_argument("--duration-s", type=float, default=5.0, help="Command duration in seconds.")
    parser.add_argument("--cmd-hz", type=float, default=50.0)
    parser.add_argument("--zero-duration-s", type=float, default=1.0)
    parser.add_argument("--vio-timeout-s", type=float, default=0.5)
    parser.add_argument("--max-abs-displacement-m", type=float, default=1.5)
    parser.add_argument("--tolerance-m", type=float, default=0.15)
    parser.add_argument("--raw-output", type=Path, default=Path("/tmp/unitree_vx_constant_test_raw.csv"))
    parser.add_argument("--dry-run", action="store_true", help="Do not publish /cmd_vel.")
    return parser


def validate_args(args):
    if args.cmd_hz <= 0.0 or args.cmd_hz > 50.0:
        raise ValueError("--cmd-hz must be in (0, 50]")
    if args.duration_s <= 0.0:
        raise ValueError("--duration-s must be positive")
    if args.zero_duration_s < 0.0:
        raise ValueError("--zero-duration-s must be non-negative")
    if args.vio_timeout_s <= 0.0:
        raise ValueError("--vio-timeout-s must be positive")
    if args.max_abs_displacement_m <= 0.0:
        raise ValueError("--max-abs-displacement-m must be positive")
    if args.tolerance_m < 0.0:
        raise ValueError("--tolerance-m must be non-negative")


def main(args=None):
    parsed = build_arg_parser().parse_args(args=args)
    validate_args(parsed)

    rclpy.init()
    node = UnitreeVxConstantTest(parsed)
    samples: list[RawSample] = []
    expected_translation = parsed.velocity * parsed.duration_s
    try:
        if not node.wait_for_pose(timeout_s=5.0):
            raise RuntimeError("timed out waiting for /insight/vio_100hz")
        node.warn_other_cmd_vel_publishers()
        samples = node.run_test()
        result = analyze(samples, expected_translation)
        if result is None:
            node.get_logger().warning("not enough samples to measure translation")
        else:
            ok = abs(result["translation_error_m"]) <= parsed.tolerance_m
            level = node.get_logger().info if ok else node.get_logger().warning
            level(
                f"result expected={result['expected_translation_m']:.3f}m "
                f"measured={result['measured_translation_m']:.3f}m "
                f"error={result['translation_error_m']:.3f}m "
                f"fit_vx={result['fitted_velocity_mps']:.3f}m/s "
                f"fit_translation={result['fitted_translation_m']:.3f}m "
                f"elapsed={result['elapsed_s']:.3f}s samples={result['samples']} "
                f"status={'PASS' if ok else 'WARN'}"
            )
    except KeyboardInterrupt:
        node.get_logger().warning("interrupted; publishing zero command")
    finally:
        node.publish_zero_for(parsed.zero_duration_s)
        write_raw_csv(parsed.raw_output, samples)
        node.get_logger().info(f"wrote raw CSV: {parsed.raw_output}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
