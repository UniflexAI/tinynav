import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
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
    length_m: float = 4.0
    width_m: float = 2.0
    height_m: float = 0.0
    lookahead_m: float = 1.5
    finish_radius_m: float = 0.35
    auto_start: bool = True


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

        self.odom_sub = self.create_subscription(Odometry, config.odom_topic, self._odom_callback, 20)
        self.cmd_sub = self.create_subscription(String, "/benchmark/cmd", self._cmd_callback, 10)
        self.global_plan_pub = self.create_publisher(Path, "/mapping/global_plan", 10)
        self.target_pose_pub = self.create_publisher(Odometry, "/control/target_pose", 10)
        self.status_pub = self.create_publisher(String, "/benchmark/status", 10)

        period = 1.0 / max(1e-3, float(config.publish_rate_hz))
        self.timer = self.create_timer(period, self._timer_callback)

        self.latest_odom: Odometry | None = None
        self.path_world: np.ndarray | None = None
        self.path_s: np.ndarray | None = None
        self.progress_idx = 0
        self.started = False
        self.completed = False
        self.started_at: float | None = None

        self.get_logger().info(
            f"benchmark_node ready: odom_topic={config.odom_topic} "
            f"auto_start={config.auto_start} "
            f"figure8=({config.length_m:.2f}m x {config.width_m:.2f}m) "
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
            if self.latest_odom is not None:
                self._start_from_odom(self.latest_odom)
        elif action in {"stop", "cancel"}:
            self.started = False
            self.completed = True
            self._publish_status("stopped")
        else:
            self.get_logger().warning("Unknown benchmark action: %s", action)

    def _odom_callback(self, msg: Odometry):
        self.latest_odom = msg
        if self.config.auto_start and not self.started and not self.completed:
            self._start_from_odom(msg)

    def _reset_path(self):
        self.path_world = None
        self.path_s = None
        self.progress_idx = 0

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
        self.path_world = _transform_path(local_path, origin, yaw)
        self.path_s = _path_distances(self.path_world)
        self.progress_idx = 0
        self.started = True
        self.completed = False
        self.started_at = time.time()
        self.get_logger().info("Started figure-eight benchmark with %d path points", len(self.path_world))
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

        self._update_progress(self.latest_odom)
        self._publish_global_plan()
        self._publish_target_pose()
        self._publish_status("running")

    def _update_progress(self, odom: Odometry):
        assert self.path_world is not None
        pos = np.array(
            [odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z],
            dtype=np.float64,
        )
        # Monotonic nearest-point search in a bounded forward window. This keeps
        # figure-eight self-crossings from jumping backwards while staying cheap.
        start = self.progress_idx
        end = min(len(self.path_world), start + 80)
        window = self.path_world[start:end]
        if len(window) > 0:
            local_idx = int(np.argmin(np.linalg.norm(window[:, :2] - pos[:2], axis=1)))
            self.progress_idx = max(self.progress_idx, start + local_idx)

        is_near_end = self.progress_idx >= len(self.path_world) - 3
        end_dist = float(np.linalg.norm(self.path_world[-1, :2] - pos[:2]))
        if is_near_end and end_dist <= self.config.finish_radius_m:
            self.completed = True
            self._publish_status("completed")
            self.get_logger().info("Figure-eight benchmark completed")

    def _target_index(self) -> int:
        assert self.path_s is not None
        target_s = self.path_s[self.progress_idx] + float(self.config.lookahead_m)
        idx = int(np.searchsorted(self.path_s, target_s, side="left"))
        return min(idx, len(self.path_s) - 1)

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
        payload = {
            "state": state,
            "progress_m": round(progress, 3),
            "total_m": round(total, 3),
            "percent": round((progress / total * 100.0) if total > 0 else 0.0, 1),
            "progress_idx": self.progress_idx,
            "path_points": int(len(self.path_world)) if self.path_world is not None else 0,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)


def main(args=None):
    logging.basicConfig(level=logging.INFO)
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--odom_topic", type=str, default=BenchmarkConfig.odom_topic)
    parser.add_argument("--publish_rate_hz", type=float, default=BenchmarkConfig.publish_rate_hz)
    parser.add_argument("--path_points", type=int, default=BenchmarkConfig.path_points)
    parser.add_argument("--length_m", type=float, default=BenchmarkConfig.length_m)
    parser.add_argument("--width_m", type=float, default=BenchmarkConfig.width_m)
    parser.add_argument("--height_m", type=float, default=BenchmarkConfig.height_m)
    parser.add_argument("--lookahead_m", type=float, default=BenchmarkConfig.lookahead_m)
    parser.add_argument("--finish_radius_m", type=float, default=BenchmarkConfig.finish_radius_m)
    parser.add_argument("--auto_start", action="store_true", default=BenchmarkConfig.auto_start)
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
