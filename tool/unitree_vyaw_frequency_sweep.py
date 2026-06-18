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
from rclpy.node import Node

from tinynav.core.math_utils import pose_msg2np


DEFAULT_FREQUENCIES = "0.05,0.10,0.20,0.30,0.50,1.0"


@dataclass
class PoseSample:
    recv_mono_s: float
    ros_stamp_s: float
    yaw_rad: float


@dataclass
class RawSample:
    frequency_hz: float
    t_segment_s: float
    ros_stamp_s: float
    cmd_vyaw: float
    yaw_rad: float
    unwrapped_yaw_rad: float
    measured_vyaw_radps: float = float("nan")


@dataclass
class FrequencyResult:
    frequency_hz: float
    cmd_amp: float
    measured_yaw_rate_offset: float
    yaw_amp: float
    measured_yaw_rate_amp: float
    gain: float
    phase_rad: float
    lag_rad: float
    delay_s: float
    samples_used: int


def parse_float_list(value: str) -> list[float]:
    values = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    if any(x <= 0.0 for x in values):
        raise argparse.ArgumentTypeError("frequencies must be positive")
    return values


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def path_token(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def frequencies_token(frequencies: list[float]) -> str:
    return "-".join(path_token(f) for f in frequencies)


def resolve_output_paths(args, prefix: str):
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    param_name = (
        f"{prefix}_amp{path_token(args.amplitude)}"
        f"_freq{frequencies_token(args.frequencies)}"
    )
    run_dir = args.output_dir / f"{param_name}_{run_stamp}"
    if args.raw_output is None:
        args.raw_output = run_dir / f"{param_name}_raw.csv"
    if args.summary_output is None:
        args.summary_output = run_dir / f"{param_name}_summary.csv"


def yaw_from_pose_msg(msg: PoseStamped) -> float:
    pose = pose_msg2np(msg)
    forward = pose[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if np.linalg.norm(forward[:2]) <= 1e-6:
        return 0.0
    return math.atan2(float(forward[1]), float(forward[0]))


def fit_sinusoid(times: np.ndarray, values: np.ndarray, frequency_hz: float):
    omega_t = 2.0 * math.pi * frequency_hz * times
    x = np.column_stack(
        [
            np.ones_like(times),
            np.sin(omega_t),
            np.cos(omega_t),
        ]
    )
    coeff, *_ = np.linalg.lstsq(x, values, rcond=None)
    offset, sin_coeff, cos_coeff = [float(v) for v in coeff]
    amplitude = math.hypot(sin_coeff, cos_coeff)
    phase = math.atan2(cos_coeff, sin_coeff)
    return offset, amplitude, phase


def fit_yaw_with_linear_trend(times: np.ndarray, yaw: np.ndarray, frequency_hz: float):
    omega_t = 2.0 * math.pi * frequency_hz * times
    x = np.column_stack(
        [
            times,
            np.ones_like(times),
            np.sin(omega_t),
            np.cos(omega_t),
        ]
    )
    coeff, *_ = np.linalg.lstsq(x, yaw, rcond=None)
    slope, offset, sin_coeff, cos_coeff = [float(v) for v in coeff]
    return slope, offset, sin_coeff, cos_coeff


def analyze_frequency(
    frequency_hz: float,
    samples: list[RawSample],
    warmup_cycles: float,
) -> FrequencyResult | None:
    if len(samples) < 4:
        return None

    times = np.array([s.t_segment_s for s in samples], dtype=np.float64)
    cmd_vyaw = np.array([s.cmd_vyaw for s in samples], dtype=np.float64)
    unwrapped_yaw = np.array([s.unwrapped_yaw_rad for s in samples], dtype=np.float64)

    analysis_start = warmup_cycles / frequency_hz
    mask = times >= analysis_start
    if int(np.count_nonzero(mask)) < 4:
        mask = np.ones_like(times, dtype=bool)

    fit_t = times[mask]
    fit_cmd = cmd_vyaw[mask]
    fit_yaw = unwrapped_yaw[mask]
    if len(fit_t) < 4:
        return None

    _, cmd_amp, cmd_phase = fit_sinusoid(fit_t, fit_cmd, frequency_hz)
    measured_yaw_rate_offset, _, yaw_sin_coeff, yaw_cos_coeff = fit_yaw_with_linear_trend(
        fit_t, fit_yaw, frequency_hz
    )
    if cmd_amp <= 1e-9:
        return None

    omega = 2.0 * math.pi * frequency_hz
    yaw_amp = math.hypot(yaw_sin_coeff, yaw_cos_coeff)
    yaw_rate_sin_coeff = -omega * yaw_cos_coeff
    yaw_rate_cos_coeff = omega * yaw_sin_coeff
    measured_yaw_rate_amp = omega * yaw_amp
    measured_phase = math.atan2(yaw_rate_cos_coeff, yaw_rate_sin_coeff)

    for sample in samples:
        omega_t = omega * sample.t_segment_s
        sample.measured_vyaw_radps = (
            measured_yaw_rate_offset
            + yaw_rate_sin_coeff * math.sin(omega_t)
            + yaw_rate_cos_coeff * math.cos(omega_t)
        )

    phase_rad = wrap_angle(measured_phase - cmd_phase)
    lag_rad = -phase_rad
    delay_s = lag_rad / (2.0 * math.pi * frequency_hz)
    return FrequencyResult(
        frequency_hz=frequency_hz,
        cmd_amp=cmd_amp,
        measured_yaw_rate_offset=measured_yaw_rate_offset,
        yaw_amp=yaw_amp,
        measured_yaw_rate_amp=measured_yaw_rate_amp,
        gain=measured_yaw_rate_amp / cmd_amp,
        phase_rad=phase_rad,
        lag_rad=lag_rad,
        delay_s=delay_s,
        samples_used=int(np.count_nonzero(mask)),
    )


class UnitreeVyawFrequencySweep(Node):
    def __init__(self, args):
        super().__init__("unitree_vyaw_frequency_sweep")
        self.args = args
        self.latest_pose: PoseSample | None = None
        self._last_recorded_stamp_s: float | None = None
        self._last_raw_yaw: float | None = None
        self._unwrapped_yaw = 0.0

        self.create_subscription(PoseStamped, "/insight/vio_100hz", self._pose_cb, 100)
        self.cmd_pub = None
        if not args.dry_run:
            self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def _pose_cb(self, msg: PoseStamped):
        self.latest_pose = PoseSample(
            recv_mono_s=time.monotonic(),
            ros_stamp_s=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            yaw_rad=yaw_from_pose_msg(msg),
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
                f"other /cmd_vel publishers detected: {names}; stop planner/teleop before live sweep."
            )

    def publish_cmd(self, vyaw: float):
        if self.args.dry_run or self.cmd_pub is None:
            return
        msg = Twist()
        msg.angular.z = float(vyaw)
        self.cmd_pub.publish(msg)

    def command_vyaw(self, frequency_hz: float, t_segment_s: float) -> float:
        return self.args.amplitude * math.sin(2.0 * math.pi * frequency_hz * t_segment_s)

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

    def reset_unwrap(self, start_yaw_rad: float):
        self._last_raw_yaw = start_yaw_rad
        self._unwrapped_yaw = 0.0

    def record_new_pose(self, frequency_hz: float, segment_start_mono_s: float) -> RawSample | None:
        pose = self.latest_pose
        if pose is None:
            return None
        if self._last_recorded_stamp_s is not None and pose.ros_stamp_s <= self._last_recorded_stamp_s:
            return None
        self._last_recorded_stamp_s = pose.ros_stamp_s

        if self._last_raw_yaw is None:
            self.reset_unwrap(pose.yaw_rad)
        else:
            self._unwrapped_yaw += wrap_angle(pose.yaw_rad - self._last_raw_yaw)
            self._last_raw_yaw = pose.yaw_rad

        t_segment = max(0.0, pose.recv_mono_s - segment_start_mono_s)
        return RawSample(
            frequency_hz=frequency_hz,
            t_segment_s=t_segment,
            ros_stamp_s=pose.ros_stamp_s,
            cmd_vyaw=self.command_vyaw(frequency_hz, t_segment),
            yaw_rad=pose.yaw_rad,
            unwrapped_yaw_rad=self._unwrapped_yaw,
        )

    def run_frequency(self, frequency_hz: float) -> tuple[list[RawSample], bool]:
        if self.latest_pose is None:
            raise RuntimeError("no /insight/vio_100hz pose available")

        self.publish_zero_for(self.args.zero_duration_s)
        start_pose = self.latest_pose
        if start_pose is None:
            raise RuntimeError("no /insight/vio_100hz pose available after zero hold")

        duration_s = self.args.cycles_per_freq / frequency_hz
        expected_yaw_amp = self.args.amplitude / (2.0 * math.pi * frequency_hz)
        period_s = 1.0 / float(self.args.cmd_hz)
        segment_start = time.monotonic()
        next_pub = segment_start
        self._last_recorded_stamp_s = None
        self.reset_unwrap(start_pose.yaw_rad)
        samples: list[RawSample] = []

        self.get_logger().info(
            f"frequency={frequency_hz:.3f}Hz duration={duration_s:.2f}s "
            f"amplitude={self.args.amplitude:.3f}rad/s expected_yaw_amp={expected_yaw_amp:.3f}rad"
        )
        while rclpy.ok():
            now = time.monotonic()
            t_segment = now - segment_start
            if t_segment >= duration_s:
                break

            pose = self.latest_pose
            if pose is not None and now - pose.recv_mono_s > self.args.vio_timeout_s:
                self.get_logger().error(
                    f"/insight/vio_100hz timeout: last update {now - pose.recv_mono_s:.3f}s ago"
                )
                return samples, False

            if now >= next_pub:
                self.publish_cmd(self.command_vyaw(frequency_hz, t_segment))
                next_pub += period_s

            rclpy.spin_once(self, timeout_sec=min(0.005, max(0.0, next_pub - now)))
            sample = self.record_new_pose(frequency_hz, segment_start)
            if sample is not None:
                samples.append(sample)
                if abs(sample.unwrapped_yaw_rad) >= self.args.max_abs_yaw_rad:
                    self.get_logger().error(
                        f"yaw limit reached at {sample.unwrapped_yaw_rad:.3f}rad "
                        f"for frequency={frequency_hz:.3f}Hz"
                    )
                    return samples, False

        self.publish_zero_for(self.args.zero_duration_s)
        return samples, True


def write_raw_csv(path: Path, rows: list[RawSample]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frequency_hz",
        "t_segment_s",
        "ros_stamp_s",
        "cmd_vyaw",
        "yaw_rad",
        "unwrapped_yaw_rad",
        "measured_vyaw_radps",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def write_summary_csv(path: Path, rows: list[FrequencyResult]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frequency_hz",
        "cmd_amp",
        "measured_yaw_rate_offset",
        "yaw_amp",
        "measured_yaw_rate_amp",
        "gain",
        "phase_rad",
        "lag_rad",
        "delay_s",
        "samples_used",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep Unitree yaw rate with sine commands and estimate frequency response."
    )
    parser.add_argument("--amplitude", type=float, default=0.15, help="Sine yaw-rate amplitude in rad/s.")
    parser.add_argument("--frequencies", type=parse_float_list, default=parse_float_list(DEFAULT_FREQUENCIES))
    parser.add_argument("--cycles", "--cycles-per-freq", dest="cycles_per_freq", type=float, default=4.0)
    parser.add_argument("--warmup-cycles", type=float, default=1.0)
    parser.add_argument("--cmd-hz", type=float, default=50.0)
    parser.add_argument("--max-abs-yaw-rad", type=float, default=1.57)
    parser.add_argument("--vio-timeout-s", type=float, default=0.5)
    parser.add_argument("--zero-duration-s", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/unitree_frequency_sweep_logs"))
    parser.add_argument("--raw-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Do not publish /cmd_vel.")
    return parser


def validate_args(args):
    if args.amplitude <= 0.0:
        raise ValueError("--amplitude must be positive")
    if args.cmd_hz <= 0.0 or args.cmd_hz > 50.0:
        raise ValueError("--cmd-hz must be in (0, 50]")
    if args.cycles_per_freq <= 0.0:
        raise ValueError("--cycles must be positive")
    if args.warmup_cycles < 0.0:
        raise ValueError("--warmup-cycles must be non-negative")
    if args.warmup_cycles >= args.cycles_per_freq:
        raise ValueError("--warmup-cycles must be smaller than --cycles")
    if args.max_abs_yaw_rad <= 0.0:
        raise ValueError("--max-abs-yaw-rad must be positive")


def main(args=None):
    parsed = build_arg_parser().parse_args(args=args)
    validate_args(parsed)
    resolve_output_paths(parsed, "vyaw")

    rclpy.init()
    node = UnitreeVyawFrequencySweep(parsed)
    all_samples: list[RawSample] = []
    results: list[FrequencyResult] = []
    try:
        if not node.wait_for_pose(timeout_s=5.0):
            raise RuntimeError("timed out waiting for /insight/vio_100hz")
        node.warn_other_cmd_vel_publishers()

        for frequency_hz in parsed.frequencies:
            samples, ok = node.run_frequency(frequency_hz)
            result = analyze_frequency(frequency_hz, samples, parsed.warmup_cycles)
            if result is not None:
                results.append(result)
                node.get_logger().info(
                    f"result f={frequency_hz:.3f}Hz "
                    f"yaw_offset_rate={result.measured_yaw_rate_offset:.4f}rad/s "
                    f"yaw_amp={result.yaw_amp:.4f}rad "
                    f"vyaw_amp={result.measured_yaw_rate_amp:.4f}rad/s gain={result.gain:.3f} "
                    f"phase={result.phase_rad:.3f}rad lag={result.lag_rad:.3f}rad delay={result.delay_s:.3f}s "
                    f"samples={result.samples_used}"
                )
            else:
                node.get_logger().warning(f"not enough samples to analyze frequency={frequency_hz:.3f}Hz")
            all_samples.extend(samples)
            if not ok:
                break
    except KeyboardInterrupt:
        node.get_logger().warning("interrupted; publishing zero command")
    finally:
        node.publish_zero_for(parsed.zero_duration_s)
        write_raw_csv(parsed.raw_output, all_samples)
        write_summary_csv(parsed.summary_output, results)
        node.get_logger().info(f"wrote raw CSV: {parsed.raw_output}")
        node.get_logger().info(f"wrote summary CSV: {parsed.summary_output}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
