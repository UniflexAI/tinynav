import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import numpy as np
import logging
import time

from tinynav.core.math_utils import msg2np as _msg2np

# Module-level logger for cases where self.get_logger() is not available
logger = logging.getLogger(__name__)


def _nearest_tangent(path_world: list, robot_xy: np.ndarray) -> np.ndarray:
    """Direction of the path segment closest to the robot."""
    d_best = float('inf')
    i_best = 0
    for i in range(len(path_world) - 1):
        d = np.linalg.norm(path_world[i][:2] - robot_xy)
        if d < d_best:
            d_best = d
            i_best = i
    return path_world[i_best + 1][:2] - path_world[i_best][:2]


class CmdVelControlNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_control_node')
        self.logger = self.get_logger()  # Use ROS2 logger
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Odometry, '/slam/odometry', self.pose_callback, 10)
        self.create_subscription(Path, '/planning/trajectory_path', self.path_callback, 10)

        self.last_path_time = 0.0
        self.pose = None
        self.path_world = []

        # === Control loop (ported from planning_node_compare style) ===
        self.cmd_rate_hz = 30.0
        self.path_stale_slow_s = 0.3
        self.path_stale_stop_s = 0.6
        self.max_linear_speed = 0.8
        self.max_linear_acc = 2.0   # m/s^2
        self.max_angular_acc = 2.5  # rad/s^2
        self.max_angular_speed = 0.5

        self.latest_cmd = Twist()
        self.prev_cmd = Twist()
        self.last_cmd_pub_time = time.monotonic()
        self.last_path_update_time = None
        self.cmd_timer = self.create_timer(1.0 / self.cmd_rate_hz, self.cmd_timer_callback)

    def _update_cmd(self):
        if not self.path_world or self.pose is None:
            return
        T, _ = _msg2np(self.pose)
        robot_xy = T[:2, 3]
        forward_xy = (T[:3, :3] @ np.array([0.0, 0.0, 1.0]))[:2]

        tangent = _nearest_tangent(self.path_world, robot_xy)
        norm_f = np.linalg.norm(forward_xy)
        norm_t = np.linalg.norm(tangent)
        if norm_f < 1e-6 or norm_t < 1e-6:
            return

        cross = forward_xy[0] / norm_f * tangent[1] / norm_t - forward_xy[1] / norm_f * tangent[0] / norm_t
        dot   = forward_xy[0] / norm_f * tangent[0] / norm_t + forward_xy[1] / norm_f * tangent[1] / norm_t
        heading_err = float(np.arctan2(cross, dot))

        dist_to_end = float(np.linalg.norm(self.path_world[-1][:2] - robot_xy))
        dist_scale  = float(np.clip(dist_to_end, 0.2, 1.0))
        heading_scale = max(0.0, float(np.cos(heading_err)))

        vx = float(np.clip(self.max_linear_speed * heading_scale * dist_scale, 0.0, self.max_linear_speed))
        if abs(heading_err) > 1.0:
            vx *= 0.40
        vyaw = float(np.clip(1.8 * heading_err, -self.max_angular_speed, self.max_angular_speed))

        self.latest_cmd.linear.x = vx
        self.latest_cmd.angular.z = vyaw

    def pose_callback(self, msg):
        self.pose = msg
        self._update_cmd()

    def _clamp_step(self, target: float, current: float, max_delta: float) -> float:
        return float(np.clip(target - current, -max_delta, max_delta) + current)

    def cmd_timer_callback(self):
        now = time.monotonic()
        dt = max(1e-3, now - self.last_cmd_pub_time)
        self.last_cmd_pub_time = now

        # Stale-path protection: slow down, then stop if planner has not refreshed.
        age = float('inf') if self.last_path_update_time is None else (now - self.last_path_update_time)
        target_cmd = Twist()
        target_cmd.linear.x = self.latest_cmd.linear.x
        target_cmd.angular.z = self.latest_cmd.angular.z
        if age > self.path_stale_stop_s:
            target_cmd.linear.x = 0.0
            target_cmd.angular.z = 0.0
        elif age > self.path_stale_slow_s:
            target_cmd.linear.x *= 0.3
            target_cmd.angular.z *= 0.5

        # Acceleration limiting for smoother control.
        max_dv = self.max_linear_acc * dt
        max_dw = self.max_angular_acc * dt
        out = Twist()
        out.linear.x = self._clamp_step(target_cmd.linear.x, self.prev_cmd.linear.x, max_dv)
        out.angular.z = self._clamp_step(target_cmd.angular.z, self.prev_cmd.angular.z, max_dw)
        out.linear.y = 0.0

        self.cmd_pub.publish(out)
        self.prev_cmd = out

    def path_callback(self, msg):
        if msg is None or len(msg.poses) < 2:
            return
        self.path_world = [
            np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z], dtype=np.float32)
            for p in msg.poses
        ]
        self.last_path_update_time = time.monotonic()
        self._update_cmd()

    def destroy_node(self):
        self.logger.info("Destroying cmd_vel_control connection.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    node = CmdVelControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
