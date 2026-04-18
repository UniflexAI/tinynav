import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
from scipy.spatial.transform import Rotation as R
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class CmdVelControlNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_control_node')
        self.logger = self.get_logger()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/slam/odometry', self.pose_callback, 10)
        self.create_subscription(Path, '/planning/trajectory_path', self.path_callback, 10)

        # Camera +z = robot +x (forward); used only for heading extraction
        self.T_robot_to_camera = np.array([
            [0, -1, 0, 0],
            [0,  0,-1, 0],
            [1,  0, 0, 0],
            [0,  0, 0, 1]], dtype=np.float32)

        self.pose = None
        self.path = None
        self.last_path_update_time = None

        # Pure Pursuit parameters
        self.lookahead_distance = 0.4   # m
        self.max_linear_speed  = 0.8    # m/s
        self.max_angular_speed = 0.8    # rad/s

        # 20 Hz publish loop with stale-path protection and acc limiting
        self.cmd_rate_hz       = 20.0
        self.path_stale_slow_s = 0.3
        self.path_stale_stop_s = 0.6
        self.max_linear_acc    = 0.4    # m/s²
        self.max_angular_acc   = 0.8    # rad/s²

        self.latest_cmd = Twist()
        self.prev_cmd   = Twist()
        self.last_cmd_pub_time = time.monotonic()
        self.cmd_timer = self.create_timer(1.0 / self.cmd_rate_hz, self.cmd_timer_callback)

    # ── odom callback (50 Hz) ──────────────────────────────────────────────
    def pose_callback(self, msg: Odometry):
        self.pose = msg
        self._update_pure_pursuit()

    # ── path callback (planner rate ~5-10 Hz) ─────────────────────────────
    def path_callback(self, msg: Path):
        if msg is None or len(msg.poses) < 2:
            return
        self.path = msg
        self.last_path_update_time = time.monotonic()
        self._update_pure_pursuit()

    # ── Pure Pursuit state update ──────────────────────────────────────────
    def _update_pure_pursuit(self):
        if self.path is None or self.pose is None:
            return

        robot_xy, robot_heading = self._robot_pose_2d()

        # Path waypoint positions in world frame (x, y)
        waypoints = np.array([
            [p.pose.position.x, p.pose.position.y]
            for p in self.path.poses
        ], dtype=np.float64)

        # Find nearest waypoint, then search for lookahead point ahead of it
        dists = np.linalg.norm(waypoints - robot_xy, axis=1)
        nearest_idx = int(np.argmin(dists))
        lookahead_pt = self._find_lookahead(robot_xy, waypoints, nearest_idx)

        # Heading error (robot frame)
        dx = lookahead_pt[0] - robot_xy[0]
        dy = lookahead_pt[1] - robot_xy[1]
        target_heading = np.arctan2(dy, dx)
        alpha = target_heading - robot_heading
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi   # wrap to [-π, π]

        # Linear speed: reduce when heading error is large
        abs_alpha = abs(alpha)
        v = float(self.max_linear_speed * max(0.0, 1.0 - abs_alpha / (np.pi / 2)))

        # Pure Pursuit: ω = 2·v·sin(α) / L  (rotate in place if v≈0)
        if v > 1e-3:
            wz = float(np.clip(
                2.0 * v * np.sin(alpha) / self.lookahead_distance,
                -self.max_angular_speed, self.max_angular_speed))
        else:
            wz = float(np.clip(
                alpha * 1.5,
                -self.max_angular_speed, self.max_angular_speed))

        self.latest_cmd.linear.x  = v
        self.latest_cmd.angular.z = wz
        self.logger.debug(f"pure_pursuit v={v:.3f} wz={wz:.3f} alpha={np.degrees(alpha):.1f}°")

    def _robot_pose_2d(self):
        """Return (xy, heading) in world frame from odom (camera pose)."""
        p = self.pose.pose.pose
        R_cam = R.from_quat([p.orientation.x, p.orientation.y,
                              p.orientation.z, p.orientation.w]).as_matrix()
        # Camera +z = robot forward (+x)
        fwd = R_cam[:, 2]
        heading = np.arctan2(fwd[1], fwd[0])
        return np.array([p.position.x, p.position.y]), heading

    def _find_lookahead(self, robot_xy: np.ndarray,
                        waypoints: np.ndarray, start_idx: int) -> np.ndarray:
        L = self.lookahead_distance
        for i in range(start_idx, len(waypoints) - 1):
            seg_start = waypoints[i]
            seg_end   = waypoints[i + 1]
            d = seg_end - seg_start
            f = seg_start - robot_xy
            a = float(d @ d)
            if a < 1e-9:
                continue
            b = 2.0 * float(f @ d)
            c = float(f @ f) - L * L
            disc = b * b - 4 * a * c
            if disc >= 0:
                t = (-b + np.sqrt(disc)) / (2 * a)
                if 0.0 <= t <= 1.0:
                    return seg_start + t * d
        return waypoints[-1]

    # ── 20 Hz publish loop ─────────────────────────────────────────────────
    def _clamp_step(self, target: float, current: float, max_delta: float) -> float:
        return float(np.clip(target - current, -max_delta, max_delta) + current)

    def cmd_timer_callback(self):
        now = time.monotonic()
        dt  = max(1e-3, now - self.last_cmd_pub_time)
        self.last_cmd_pub_time = now

        age = float('inf') if self.last_path_update_time is None \
              else (now - self.last_path_update_time)

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
