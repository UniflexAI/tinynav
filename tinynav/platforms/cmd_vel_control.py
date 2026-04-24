import math
import logging

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class CmdVelControlNode(Node):
    TIMER_HZ = 50.0

    # Acceleration limits
    MAX_LINEAR_ACC = 0.3   # m/s^2 per second
    MAX_ANGULAR_ACC = 2.5  # rad/s^2 per second

    LOOKAHEAD_DIST = 0.2

    # Proportional gain on heading error -> angular velocity
    HEADING_GAIN = 1.5

    # Path interpolation resolution
    PATH_RESOLUTION = 0.05  # meters
    HEADING_FALLBACK_POS_EPS = 0.08  # meters

    def __init__(self):
        super().__init__("cmd_vel_control_node")
        self.logger = self.get_logger()

        # Camera-to-robot base transform (left-multiply camera pose to get robot pose)
        self.T_robot_to_camera = np.array([
            [0, -1, 0, 0],
            [0,  0, -1, 0],
            [1,  0,  0, 0],
            [0,  0,  0, 1],
        ], dtype=np.float64)

        # Previous published command for acceleration limiting
        self._prev_vx = 0.0
        self._prev_vyaw = 0.0

        # Propagated camera pose from perception /slam/odom_100hz
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.rotation = np.eye(3)
        self._odom_100hz_valid = False

        # Trajectory
        self.trajectory: Path | None = None

        # Subscribers
        self.create_subscription(Odometry, "/slam/odom_100hz", self._odom_100hz_cb, 10)
        self.create_subscription(Path, "/planning/trajectory_path", self._traj_cb, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.debug_pub = self.create_publisher(Float32MultiArray, "/control/debug", 10)

        # 50 Hz timer
        self.create_timer(1.0 / self.TIMER_HZ, self._control_loop)

    def _odom_100hz_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        self.position = np.array([p.x, p.y, p.z])
        self.velocity = np.array([v.x, v.y, v.z])
        self.rotation = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        self._odom_100hz_valid = True

    def _traj_cb(self, msg: Path):
        if msg.poses:
            self.trajectory = msg

    def _control_loop(self):
        if not self._odom_100hz_valid or self.trajectory is None:
            self._publish_zero()
            return

        T_cam = np.eye(4)
        T_cam[:3, :3] = self.rotation
        T_cam[:3, 3] = self.position
        T_robot = T_cam @ self.T_robot_to_camera

        robot_pos = T_robot[:3, 3]
        robot_yaw = math.atan2(T_robot[1, 0], T_robot[0, 0])

        forward_err, heading_err = self._compute_path_error(robot_pos, robot_yaw)
        if forward_err is None:
            self._publish_zero()
            return

        vx = float(np.clip(forward_err, 0.0, 0.5))
        vyaw = float(np.clip(self.HEADING_GAIN * heading_err, -0.5, 0.5))
        self._prev_vx = vx
        self._prev_vyaw = vyaw

        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = vyaw
        self.cmd_pub.publish(cmd)

        dbg = Float32MultiArray()
        dbg.data = [float(heading_err), float(vx), float(vyaw)]
        self.debug_pub.publish(dbg)

    def _interpolate_path(self, poses):
        if len(poses) == 0:
            return []
        if len(poses) == 1:
            p = poses[0].pose.position
            return [(float(p.x), float(p.y))]

        new_poses = []
        for i in range(len(poses) - 1):
            p1 = poses[i].pose.position
            p2 = poses[i + 1].pose.position

            x1, y1 = p1.x, p1.y
            x2, y2 = p2.x, p2.y

            dist = math.hypot(x2 - x1, y2 - y1)
            steps = max(1, int(dist / self.PATH_RESOLUTION))

            for s in range(steps):
                t = s / steps
                x = x1 * (1 - t) + x2 * t
                y = y1 * (1 - t) + y2 * t

                new_poses.append((x, y))

        last = poses[-1].pose.position
        new_poses.append((last.x, last.y))

        return new_poses

    def _compute_path_error(self, robot_pos: np.ndarray, robot_yaw: float):
        if self.trajectory is None or not self.trajectory.poses:
            return None, None

        path_xy = self._interpolate_path(self.trajectory.poses)

        rx, ry = robot_pos[0], robot_pos[1]

        best_idx = 0
        best_dist = float("inf")
        for i, (x, y) in enumerate(path_xy):
            d = (x - rx) ** 2 + (y - ry) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i

        target_xy = None
        for i in range(best_idx, len(path_xy)):
            x, y = path_xy[i]
            if math.hypot(x - rx, y - ry) >= self.LOOKAHEAD_DIST:
                target_xy = (x, y)
                break
        if target_xy is None:
            target_xy = path_xy[-1]

        dx = target_xy[0] - rx
        dy = target_xy[1] - ry
        forward_err = math.hypot(dx, dy)

        # Simplified heading rule:
        # 1) if position error is large enough, head to target XY;
        # 2) otherwise, use path orientation yaw as heading target.
        if forward_err > self.HEADING_FALLBACK_POS_EPS:
            target_yaw = math.atan2(dy, dx)
        else:
            q = self.trajectory.poses[-1].pose.orientation
            target_yaw = float(R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2])
        heading_err = self._wrap_angle(target_yaw - robot_yaw)
        return forward_err, heading_err

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _clamp_step(target: float, current: float, max_delta: float) -> float:
        return float(np.clip(target - current, -max_delta, max_delta) + current)

    def _publish_zero(self):
        dt = 1.0 / self.TIMER_HZ
        self._prev_vx = self._clamp_step(0.0, self._prev_vx, self.MAX_LINEAR_ACC * dt)
        self._prev_vyaw = self._clamp_step(0.0, self._prev_vyaw, self.MAX_ANGULAR_ACC * dt)
        cmd = Twist()
        cmd.linear.x = self._prev_vx
        cmd.angular.z = self._prev_vyaw
        self.cmd_pub.publish(cmd)

    def destroy_node(self):
        self.logger.info("Destroying cmd_vel_control node.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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


if __name__ == "__main__":
    main()
