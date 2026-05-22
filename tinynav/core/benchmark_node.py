import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import String

from tinynav.core.math_utils import np2msg

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Runtime parameters for the figure-eight local-planning benchmark."""

    odom_topic: str = "/slam/odometry_visual"
    publish_rate_hz: float = 10.0
    path_points: int = 500
    waypoint_count: int = 20
    length_m: float = 4.0
    width_m: float = 2.0
    height_m: float = 0.0
    lookahead_m: float = 1.5
    finish_radius_m: float = 0.35
    score_sigma_m: float = 0.5
    auto_start: bool = True
    mode: str = "siso_vx_sine"
    sine_amplitude_mps: float = 0.3
    sine_frequency_hz: float = 1.0
    sine_duration_s: float = 20.0
    sine_bias_mps: float = 0.0


def _yaw_from_quat(q) -> float:
    """Return yaw from a ROS quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _figure_eight_local_path(length_m: float, width_m: float, height_m: float, points: int) -> np.ndarray:
    """Generate a planar figure-eight path starting at the origin and tangent to +x.

    Local frame convention: +x forward, +y left. The curve is a Gerono lemniscate:
    x = length/2 * sin(t), y = width/2 * sin(2t). Starting at t=0 makes the
    first target naturally forward from the robot's initial pose.
    """
    n = max(8, int(points))
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=True)
    x = 0.5 * float(length_m) * np.sin(t)
    y = 0.5 * float(width_m) * np.sin(2.0 * t)
    z = np.full_like(x, float(height_m))
    return np.stack([x, y, z], axis=1)


def _transform_path(local_path: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return local_path @ rot.T + origin.reshape(1, 3)


def _path_distances(path: np.ndarray) -> np.ndarray:
    if len(path) == 0:
        return np.zeros((0,), dtype=np.float64)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _sample_by_arclength(path: np.ndarray, count: int, include_start: bool = False) -> np.ndarray:
    """Sample a path at equal arclength intervals.

    For the waypoint benchmark we skip the starting pose and sample 20 targets at
    1/20, 2/20, ... 20/20 of the full loop. That avoids wasting the first
    command on the robot's current pose while still closing the 8-shape at the
    origin.
    """
    n = max(1, int(count))
    if len(path) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(path) == 1:
        return np.repeat(path, n, axis=0)
    s = _path_distances(path)
    total = float(s[-1])
    if total <= 1e-6:
        return np.repeat(path[:1], n, axis=0)
    start_frac = 0.0 if include_start else 1.0 / n
    targets = np.linspace(start_frac * total, total, n, endpoint=True)
    out = np.zeros((n, 3), dtype=np.float64)
    for dim in range(3):
        out[:, dim] = np.interp(targets, s, path[:, dim])
    return out


def _nearest_path_distances(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Return each point's nearest 2D distance to a polyline's line segments."""
    if len(points) == 0 or len(path) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(path) == 1:
        delta = points[:, :2] - path[0, :2]
        return np.linalg.norm(delta, axis=1)

    pts = points[:, :2]
    a = path[:-1, :2]
    b = path[1:, :2]
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    denom = np.where(denom < 1e-9, 1.0, denom)
    ap = pts[:, None, :] - a[None, :, :]
    t = np.sum(ap * ab[None, :, :], axis=2) / denom[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.min(np.linalg.norm(pts[:, None, :] - closest, axis=2), axis=1)


def _score_tracking(trajectory: np.ndarray, path: np.ndarray, path_s: np.ndarray, sigma_m: float) -> dict:
    """Compare executed odom trajectory against the reference path.

    Score is 0-100. It combines tracking error and completion progress:
      - error_score uses exp(-RMSE / sigma), so smaller path deviation is better;
      - completion_score is the fraction of reference arclength reached.
    """
    n_samples = int(len(trajectory))
    if n_samples == 0 or len(path) == 0 or len(path_s) == 0:
        return {
            "score": 0.0,
            "rmse_m": None,
            "mean_error_m": None,
            "max_error_m": None,
            "completion_percent": 0.0,
            "samples": n_samples,
        }

    # Downsample trajectory before the O(n×m) distance matrix to bound scoring time.
    # Original last point is preserved separately for completion calculation.
    _MAX_SCORING_SAMPLES = 2000
    if n_samples > _MAX_SCORING_SAMPLES:
        idx = np.round(np.linspace(0, n_samples - 1, _MAX_SCORING_SAMPLES)).astype(int)
        traj_ds = trajectory[idx]
    else:
        traj_ds = trajectory

    dists = _nearest_path_distances(traj_ds, path)
    rmse = float(np.sqrt(np.mean(np.square(dists))))
    mean = float(np.mean(dists))
    max_err = float(np.max(dists))

    final_pos = trajectory[-1]
    final_idx = int(np.argmin(np.linalg.norm(path[:, :2] - final_pos[:2], axis=1)))
    total = float(path_s[-1]) if len(path_s) else 0.0
    completion = float(path_s[final_idx] / total) if total > 1e-6 else 0.0
    completion = max(0.0, min(1.0, completion))

    sigma = max(1e-3, float(sigma_m))
    error_score = math.exp(-rmse / sigma)
    score = 100.0 * (0.75 * error_score + 0.25 * completion)

    return {
        "score": round(score, 1),
        "rmse_m": round(rmse, 3),
        "mean_error_m": round(mean, 3),
        "max_error_m": round(max_err, 3),
        "completion_percent": round(completion * 100.0, 1),
        "samples": n_samples,
    }


class BenchmarkNode(Node):
    """Publishes a synthetic figure-eight global path and rolling local target.

    This node intentionally bypasses map_node/relocalization. It anchors a
    figure-eight path in the odom/world frame from the first odometry pose, then
    periodically:
      - publishes the full path on /mapping/global_plan for visualization;
      - publishes a lookahead point on /control/target_pose for planning_node.

    Later benchmark stages can subscribe to the same odometry and path state to
    compute tracking metrics.
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__("benchmark_node")
        self.config = config

        self.odom_sub = self.create_subscription(Odometry, config.odom_topic, self._odom_callback, 10)
        self.cmd_sub = self.create_subscription(String, "/benchmark/cmd", self._cmd_callback, 10)
        self.global_plan_pub = self.create_publisher(Path, "/mapping/global_plan", 10)
        self.target_pose_pub = self.create_publisher(Odometry, "/control/target_pose", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/benchmark/status", 10)
        self.result_pub = self.create_publisher(String, "/benchmark/result", 10)

        period = 1.0 / max(1e-3, float(config.publish_rate_hz))
        self.timer = self.create_timer(period, self._timer_callback)

        self.latest_odom: Odometry | None = None
        self.path_world: np.ndarray | None = None
        self.path_s: np.ndarray | None = None
        self.dense_path_world: np.ndarray | None = None
        self.current_waypoint_idx = 0
        self.progress_idx = 0
        self.started = False
        self.completed = False
        self.started_at: float | None = None
        self.trajectory: list[list[float]] = []
        self.result: dict | None = None
        self.siso_samples: list[list[float]] = []  # [t_rel, cmd_vx, odom_vx, x_rel, y_rel, z_rel, yaw_rel, x, y]
        self.siso_origin: np.ndarray | None = None
        self.siso_yaw: float = 0.0

        self.get_logger().info(
            f"benchmark_node ready: odom_topic={config.odom_topic} "
            f"auto_start={config.auto_start} "
            f"mode={config.mode} "
            f"figure8=({config.length_m:.2f}m x {config.width_m:.2f}m) "
            f"siso_vx=A{config.sine_amplitude_mps:.2f} f{config.sine_frequency_hz:.2f}Hz "
            f"lookahead={config.lookahead_m:.2f}m"
        )

    def _cmd_callback(self, msg: String):
        try:
            data = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            data = {"action": msg.data.strip()}
        action = str(data.get("action", "start")).lower()
        if action in {"start", "restart", "reset"}:
            self._reset_path()
            self.started = False
            self.completed = False
            self.started_at = None
            self.trajectory = []
            self.siso_samples = []
            self.result = None
            if self.latest_odom is not None:
                self._start_from_odom(self.latest_odom)
        elif action in {"stop", "cancel"}:
            if self.started and not self.completed:
                self._finish("stopped")
        else:
            self.get_logger().warning(f"Unknown benchmark action: {action}")

    def _odom_callback(self, msg: Odometry):
        self.latest_odom = msg
        if self.config.auto_start and not self.started and not self.completed:
            self._start_from_odom(msg)
        if self.started and not self.completed:
            self._record_odom(msg)

    def _reset_path(self):
        self.path_world = None
        self.path_s = None
        self.dense_path_world = None
        self.current_waypoint_idx = 0
        self.progress_idx = 0
        self.trajectory = []
        self.siso_samples = []
        self.siso_origin = None
        self.result = None

    def _record_odom(self, odom: Odometry):
        p = odom.pose.pose.position
        t = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9
        self.trajectory.append([t, float(p.x), float(p.y), float(p.z)])

    def _start_from_odom(self, odom: Odometry):
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        origin = np.array([p.x, p.y, p.z], dtype=np.float64)
        yaw = _yaw_from_quat(q)
        local_path = _figure_eight_local_path(
            self.config.length_m,
            self.config.width_m,
            self.config.height_m,
            self.config.path_points,
        )
        self.dense_path_world = _transform_path(local_path, origin, yaw)
        self.path_world = _sample_by_arclength(
            self.dense_path_world, self.config.waypoint_count, include_start=False
        )
        self.path_s = _path_distances(self.path_world)
        self.current_waypoint_idx = 0
        self.progress_idx = 0
        self.started = True
        self.completed = False
        self.started_at = time.time()
        self.trajectory = []
        self.result = None
        self.siso_samples = []
        self.siso_origin = origin.copy()
        self.siso_yaw = yaw
        self._record_odom(odom)
        if self.config.mode == "siso_vx_sine":
            self.get_logger().info(
                f"Started SISO vx sine benchmark: A={self.config.sine_amplitude_mps:.2f}m/s "
                f"f={self.config.sine_frequency_hz:.2f}Hz duration={self.config.sine_duration_s:.1f}s"
            )
        else:
            self.get_logger().info(
                f"Started figure-eight benchmark with {len(self.path_world)} waypoints "
                f"sampled from {len(self.dense_path_world) if self.dense_path_world is not None else 0} path points"
            )
        self._publish_status("running")

    def _timer_callback(self):
        if self.latest_odom is None:
            self._publish_status("waiting_for_odom")
            return
        if not self.started or self.completed:
            return
        if self.path_world is None or self.path_s is None:
            self._start_from_odom(self.latest_odom)
            return

        if self.config.mode == "siso_vx_sine":
            self._timer_siso_vx_sine(self.latest_odom)
            return

        self._update_progress(self.latest_odom)
        self._publish_global_plan()
        if self.path_world is not None and not self.completed:
            self._publish_target_pose()
        if not self.completed:
            self._publish_status("running")


    def _timer_siso_vx_sine(self, odom: Odometry):
        if self.started_at is None:
            return
        elapsed = max(0.0, time.time() - self.started_at)
        duration = max(0.1, float(self.config.sine_duration_s))
        if elapsed >= duration:
            self.cmd_vel_pub.publish(Twist())
            self._finish("completed")
            self.get_logger().info("SISO vx sine benchmark completed")
            return

        amp = float(self.config.sine_amplitude_mps)
        freq = float(self.config.sine_frequency_hz)
        bias = float(self.config.sine_bias_mps)
        cmd_vx = bias + amp * math.sin(2.0 * math.pi * freq * elapsed)
        cmd = Twist()
        cmd.linear.x = float(cmd_vx)
        self.cmd_vel_pub.publish(cmd)

        q = odom.pose.pose.orientation
        yaw = _yaw_from_quat(q)
        vx_world = float(odom.twist.twist.linear.x)
        vy_world = float(odom.twist.twist.linear.y)
        odom_vx = math.cos(yaw) * vx_world + math.sin(yaw) * vy_world
        pos = odom.pose.pose.position
        if self.siso_origin is None:
            self.siso_origin = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
            self.siso_yaw = yaw
        # Store raw world positions. The reported SISO x/y trace is computed
        # later by projecting onto the measured dominant motion axis, so the
        # plot is independent of whether the run happens along world X, Y, or a
        # diagonal direction.
        z_rel = float(pos.z) - float(self.siso_origin[2])
        yaw_rel = float(yaw - self.siso_yaw)
        self.siso_samples.append([elapsed, float(cmd_vx), float(odom_vx), 0.0, 0.0, z_rel, yaw_rel, float(pos.x), float(pos.y)])
        self._publish_status("running")

    def _siso_projected_samples(self) -> np.ndarray:
        samples = np.asarray(self.siso_samples, dtype=np.float64)
        if len(samples) == 0:
            return samples
        out = samples.copy()
        if out.shape[1] < 9 or len(out) < 2:
            return out

        xy = out[:, 7:9]
        origin = xy[0]
        centered = xy - origin

        # Use the dominant measured motion axis, not world X/Y and not the
        # final displacement. A full sine run can end near the start, making
        # endpoint-based axis/sign selection unstable exactly when the run
        # completes.
        cov = centered.T @ centered
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.array([1.0, 0.0], dtype=np.float64)

        t = out[:, 0]
        cmd = out[:, 1]
        expected_x = np.cumsum(cmd * np.diff(t, prepend=t[0]))
        proj = centered @ axis
        rmse_pos = float(np.sqrt(np.mean(np.square(proj - expected_x))))
        rmse_neg = float(np.sqrt(np.mean(np.square((-proj) - expected_x))))
        if rmse_neg < rmse_pos:
            axis = -axis
            proj = -proj

        lateral_axis = np.array([-axis[1], axis[0]], dtype=np.float64)
        out[:, 3] = proj
        out[:, 4] = centered @ lateral_axis
        return out

    def _score_siso_vx_sine(self) -> dict:
        samples = self._siso_projected_samples()
        n = int(len(samples))
        if n < 2:
            return {"score": 0.0, "rmse_m": None, "mean_error_m": None, "max_error_m": None, "completion_percent": 0.0, "samples": n}
        t = samples[:, 0]
        cmd = samples[:, 1]
        odom_vx = samples[:, 2]
        actual_x = samples[:, 3]
        dt = np.diff(t, prepend=t[0])
        expected_x = np.cumsum(cmd * dt)
        vel_err = odom_vx - cmd
        pos_err = actual_x - expected_x
        vel_rmse = float(np.sqrt(np.mean(np.square(vel_err))))
        pos_rmse = float(np.sqrt(np.mean(np.square(pos_err))))
        max_pos_err = float(np.max(np.abs(pos_err)))
        if np.std(cmd) > 1e-6 and np.std(odom_vx) > 1e-6:
            corr = float(np.corrcoef(cmd, odom_vx)[0, 1])
        else:
            corr = 0.0
        denom = max(0.03, abs(float(self.config.sine_amplitude_mps)))
        score = 100.0 * math.exp(-vel_rmse / denom) * max(0.0, corr)
        return {
            "score": round(max(0.0, min(100.0, score)), 1),
            "rmse_m": round(pos_rmse, 3),
            "mean_error_m": round(float(np.mean(np.abs(pos_err))), 3),
            "max_error_m": round(max_pos_err, 3),
            "velocity_rmse_mps": round(vel_rmse, 3),
            "correlation": round(corr, 3),
            "completion_percent": 100.0,
            "samples": n,
        }

    def _update_progress(self, odom: Odometry):
        assert self.path_world is not None
        pos = np.array(
            [odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z],
            dtype=np.float64,
        )
        if len(self.path_world) == 0:
            self._finish("completed")
            return

        target = self.path_world[self.current_waypoint_idx]
        dist = float(np.linalg.norm(target[:2] - pos[:2]))
        while dist <= self.config.finish_radius_m:
            if self.current_waypoint_idx >= len(self.path_world) - 1:
                self.progress_idx = len(self.path_world) - 1
                self._finish("completed")
                self.get_logger().info("Figure-eight waypoint benchmark completed")
                return
            self.current_waypoint_idx += 1
            self.progress_idx = self.current_waypoint_idx
            target = self.path_world[self.current_waypoint_idx]
            dist = float(np.linalg.norm(target[:2] - pos[:2]))

    def _finish(self, state: str):
        if self.completed and self.result is not None:
            return
        self.completed = True
        if self.config.mode == "siso_vx_sine":
            self.cmd_vel_pub.publish(Twist())
            self.result = self._score_siso_vx_sine()
            self.result["mode"] = self.config.mode
            self.result["amplitude_mps"] = round(float(self.config.sine_amplitude_mps), 3)
            self.result["frequency_hz"] = round(float(self.config.sine_frequency_hz), 3)
        elif self.path_world is None or self.path_s is None:
            self.result = None
        else:
            raw = np.asarray(self.trajectory, dtype=np.float64)
            # trajectory is stored as [t, x, y, z]; drop timestamp for scoring
            traj_xyz = raw[:, 1:] if raw.ndim == 2 and raw.shape[1] == 4 else raw
            self.result = _score_tracking(
                traj_xyz,
                self.path_world,
                self.path_s,
                self.config.score_sigma_m,
            )
            self.result["waypoints"] = int(len(self.path_world))
        if self.result is not None and self.started_at is not None:
            self.result["duration_s"] = round(time.time() - self.started_at, 2)
        self._publish_status(state)
        self._publish_result(state)

    def _target_index(self) -> int:
        return self.current_waypoint_idx

    def _publish_global_plan(self):
        assert self.path_world is not None
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        for x, y, z in self.path_world:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.global_plan_pub.publish(msg)

    def _publish_target_pose(self):
        assert self.path_world is not None
        target = self.path_world[self._target_index()]
        T = np.eye(4)
        T[:3, 3] = target
        self.target_pose_pub.publish(np2msg(T, self.get_clock().now().to_msg(), "world", "benchmark_target"))

    def _publish_status(self, state: str):
        total = float(self.path_s[-1]) if self.path_s is not None and len(self.path_s) else 0.0
        progress = float(self.path_s[self.progress_idx]) if self.path_s is not None and len(self.path_s) else 0.0
        if self.config.mode == "siso_vx_sine":
            duration = max(0.1, float(self.config.sine_duration_s))
            elapsed = max(0.0, time.time() - self.started_at) if self.started_at is not None else 0.0
            progress = min(duration, elapsed)
            total = duration
        payload = {
            "state": state,
            "mode": self.config.mode,
            "progress_m": round(progress, 3),
            "total_m": round(total, 3),
            "percent": round((progress / total * 100.0) if total > 0 else 0.0, 1),
            "progress_idx": self.progress_idx,
            "current_waypoint_idx": self.current_waypoint_idx,
            "path_points": int(len(self.path_world)) if self.path_world is not None else 0,
            "waypoints": int(len(self.path_world)) if self.path_world is not None else 0,
            "trajectory_samples": len(self.trajectory),
            "siso_samples": len(self.siso_samples),
        }
        if self.config.mode == "siso_vx_sine" and self.siso_samples:
            max_points = 400
            step = max(1, len(self.siso_samples) // max_points)
            projected = self._siso_projected_samples()
            payload["siso_trace"] = [
                {
                    "t": round(float(v[0]), 4),
                    "cmd_vx": round(float(v[1]), 4),
                    "odom_vx": round(float(v[2]), 4),
                    "x_rel": round(float(v[3]), 4),
                    "y_rel": round(float(v[4]), 4),
                    "z_rel": round(float(v[5]), 4),
                    "yaw_rel": round(float(v[6]), 4),
                    "x": round(float(v[7]), 4) if len(v) > 7 else None,
                    "y": round(float(v[8]), 4) if len(v) > 8 else None,
                }
                for v in projected[::step]
            ]
        if self.result is not None:
            payload["result"] = self.result
        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)

    def _publish_result(self, state: str):
        if self.result is None:
            return
        payload = {
            "state": state,
            **self.result,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.result_pub.publish(msg)


def main(args=None):
    logging.basicConfig(level=logging.INFO)
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--odom_topic", type=str, default=BenchmarkConfig.odom_topic)
    parser.add_argument("--publish_rate_hz", type=float, default=BenchmarkConfig.publish_rate_hz)
    parser.add_argument("--path_points", type=int, default=BenchmarkConfig.path_points)
    parser.add_argument("--waypoint_count", type=int, default=BenchmarkConfig.waypoint_count)
    parser.add_argument("--length_m", type=float, default=BenchmarkConfig.length_m)
    parser.add_argument("--width_m", type=float, default=BenchmarkConfig.width_m)
    parser.add_argument("--height_m", type=float, default=BenchmarkConfig.height_m)
    parser.add_argument("--lookahead_m", type=float, default=BenchmarkConfig.lookahead_m)
    parser.add_argument("--finish_radius_m", type=float, default=BenchmarkConfig.finish_radius_m)
    parser.add_argument("--score_sigma_m", type=float, default=BenchmarkConfig.score_sigma_m)
    parser.add_argument("--auto_start", action="store_true", default=BenchmarkConfig.auto_start)
    parser.add_argument("--mode", type=str, default=BenchmarkConfig.mode)
    parser.add_argument("--sine_amplitude_mps", type=float, default=BenchmarkConfig.sine_amplitude_mps)
    parser.add_argument("--sine_frequency_hz", type=float, default=BenchmarkConfig.sine_frequency_hz)
    parser.add_argument("--sine_duration_s", type=float, default=BenchmarkConfig.sine_duration_s)
    parser.add_argument("--sine_bias_mps", type=float, default=BenchmarkConfig.sine_bias_mps)
    parser.add_argument("--no_auto_start", dest="auto_start", action="store_false")
    parsed_args, _ = parser.parse_known_args(sys.argv[1:])

    node = BenchmarkNode(BenchmarkConfig(**vars(parsed_args)))
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
