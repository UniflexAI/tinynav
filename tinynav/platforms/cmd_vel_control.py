import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
import numpy as np

from tinynav.core.math_utils import msg2np


def pick_lookahead_point(path_world: list, robot_xy: np.ndarray, lookahead_dist: float = 1.5):
    if not path_world:
        return None
    d_best = float("inf")
    i_best = 0
    for i, p in enumerate(path_world):
        d = np.linalg.norm(p[:2] - robot_xy)
        if d < d_best:
            d_best = d
            i_best = i
    for i in range(i_best, len(path_world)):
        if np.linalg.norm(path_world[i][:2] - robot_xy) >= lookahead_dist:
            return path_world[i]
    return path_world[-1]


def signed_angle_between(v_from: np.ndarray, v_to: np.ndarray) -> float:
    cross = v_from[0] * v_to[1] - v_from[1] * v_to[0]
    dot = v_from[0] * v_to[0] + v_from[1] * v_to[1]
    return float(np.arctan2(cross, dot))


class CmdVelControlLooperNode(Node):
    def __init__(self):
        super().__init__("cmd_vel_control_looper")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Path, "/planning/trajectory_path", self._path_cb, 10)
        self.create_subscription(Odometry, "/slam/odometry", self._odom_cb, 10)

        self.path_world: list = []
        self.last_path_update_time = None
        self.latest_T = None

        self.cmd_rate_hz       = 30.0
        self.lookahead_dist    = 1.5
        self.max_linear_speed  = 0.8
        self.max_reverse_speed = 0.1
        self.max_angular_speed = 0.5
        self.max_linear_acc    = 2.0
        self.max_angular_acc   = 2.5
        self.path_stale_slow_s = 0.3
        self.path_stale_stop_s = 0.6

        self.latest_cmd = Twist()
        self.prev_cmd   = Twist()
        self.last_cmd_pub_time = time.monotonic()
        self.cmd_timer = self.create_timer(1.0 / self.cmd_rate_hz, self._cmd_timer_cb)

    def _path_cb(self, msg: Path):
        self.path_world = [
            np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z], dtype=np.float32)
            for p in msg.poses
        ]
        self.last_path_update_time = time.monotonic()
        if self.latest_T is not None:
            self._update_cmd(self.latest_T)

    def _odom_cb(self, msg: Odometry):
        T, _ = msg2np(msg)
        self.latest_T = T
        self._update_cmd(T)

    def _update_cmd(self, T: np.ndarray):
        cmd = Twist()
        if len(self.path_world) < 2:
            self.latest_cmd = cmd
            return

        robot_xy   = T[:2, 3]
        forward_xy = (T[:3, :3] @ np.array([0.0, 0.0, 1.0]))[:2]

        lookahead = pick_lookahead_point(self.path_world, robot_xy, self.lookahead_dist)
        to_wp  = lookahead[:2] - robot_xy
        norm_f = np.linalg.norm(forward_xy)
        norm_t = np.linalg.norm(to_wp)
        if norm_f > 1e-6 and norm_t > 1e-6:
            heading_err = signed_angle_between(forward_xy / norm_f, to_wp / norm_t)
            cmd.angular.z = float(np.clip(1.8 * heading_err, -self.max_angular_speed, self.max_angular_speed))
            heading_scale = max(0.0, float(np.cos(heading_err)))
            cmd.linear.x  = float(np.clip(
                self.max_linear_speed * heading_scale, 0.0, self.max_linear_speed
            ))
            if abs(heading_err) > 1.0:
                cmd.linear.x *= 0.40

        cmd.linear.x  = float(np.clip(cmd.linear.x,  -self.max_reverse_speed, self.max_linear_speed))
        cmd.angular.z = float(np.clip(cmd.angular.z, -self.max_angular_speed,  self.max_angular_speed))
        self.latest_cmd = cmd

    def _clamp_step(self, target: float, current: float, max_delta: float) -> float:
        return float(np.clip(target - current, -max_delta, max_delta) + current)

    def _cmd_timer_cb(self):
        now = time.monotonic()
        dt  = max(1e-3, now - self.last_cmd_pub_time)
        self.last_cmd_pub_time = now

        age = float("inf") if self.last_path_update_time is None else (now - self.last_path_update_time)
        target = Twist()
        target.linear.x  = self.latest_cmd.linear.x
        target.angular.z = self.latest_cmd.angular.z
        if age > self.path_stale_stop_s:
            target.linear.x  = 0.0
            target.angular.z = 0.0
        elif age > self.path_stale_slow_s:
            target.linear.x  *= 0.3
            target.angular.z *= 0.5

        max_dv = self.max_linear_acc  * dt
        max_dw = self.max_angular_acc * dt
        out = Twist()
        out.linear.x  = self._clamp_step(target.linear.x,  self.prev_cmd.linear.x,  max_dv)
        out.angular.z = self._clamp_step(target.angular.z, self.prev_cmd.angular.z, max_dw)

        self.cmd_pub.publish(out)
        self.prev_cmd = out


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelControlLooperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
