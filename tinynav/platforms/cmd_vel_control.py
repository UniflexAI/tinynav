import argparse
import math
import logging
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
from scipy.spatial.transform import Rotation as R


class CmdVelControlNode(Node):
    def __init__(self):
        super().__init__("cmd_vel_control_node")
        self.logger = logging.getLogger(__name__)

        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self._odom_pose_initialized = False

        self._odom_stamp_sec = None

        # columns: x, y, yaw, v_ref, w_ref
        self._path_ref = None
        self._track_idx = 0
        self._last_traj_update_sec = None

        # === ROS Interfaces ===
        self.create_subscription(Odometry, "/slam/odometry_100hz", self._odom_cb, 10)
        self.create_subscription(Path, "/planning/trajectory_path", self._traj_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    # =========================
    # ODOM
    # =========================
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        measured_position = np.array([p.x, p.y, p.z], dtype=np.float64)
        measured_rotation = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        if not self._odom_pose_initialized:
            self.position = measured_position
            self.rotation = measured_rotation
            self._odom_pose_initialized = True
        else:
            alpha = 0.35  # First-order odom low-pass filter; smaller is smoother but laggier.
            self.position = (1.0 - alpha) * self.position + alpha * measured_position
            self.rotation = measured_rotation

        self._odom_stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._control_loop()

    # =========================
    # PATH
    # =========================
    def _traj_cb(self, msg: Path):
        now = self._now_sec()
        if (
            self._last_traj_update_sec is not None
            and now - self._last_traj_update_sec < 0.2  # Drop path updates faster than 5 Hz.
        ):
            return

        self._rebuild_path(msg)
        self._last_traj_update_sec = now

    def _rebuild_path(self, path_msg: Path):
        n = len(path_msg.poses)
        if n == 0:
            self._path_ref = None
            self._track_idx = 0
            return

        xy_yaw = np.zeros((n, 3), dtype=np.float64)
        t = np.zeros(n, dtype=np.float64)
        last_yaw = 0.0
        for i, pose in enumerate(path_msg.poses):
            xy_yaw[i, 0] = pose.pose.position.x
            xy_yaw[i, 1] = pose.pose.position.y
            t[i] = pose.header.stamp.sec + pose.header.stamp.nanosec * 1e-9

            q = pose.pose.orientation
            rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            forward = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if np.linalg.norm(forward[:2]) > 1e-6:
                last_yaw = math.atan2(float(forward[1]), float(forward[0]))
            xy_yaw[i, 2] = last_yaw

        path_ref = np.zeros((n, 5), dtype=np.float64)
        path_ref[:, :3] = xy_yaw
        if n > 1:
            t = t - t[0]
            if np.min(np.diff(t)) <= 0.0:
                t = np.arange(n, dtype=np.float64) * 0.2  # Path generation publishes every 0.2 seconds.

            yaw_u = np.unwrap(xy_yaw[:, 2])
            for i in range(n - 1):
                dt = max(1e-3, float(t[i + 1] - t[i]))
                ds = float(np.linalg.norm(xy_yaw[i + 1, :2] - xy_yaw[i, :2]))
                path_ref[i, 3] = ds / dt
                path_ref[i, 4] = (yaw_u[i + 1] - yaw_u[i]) / dt
            path_ref[-1, 3] = path_ref[-2, 3]
            path_ref[-1, 4] = path_ref[-2, 4]

        self._path_ref = path_ref
        self._track_idx = 0

    # =========================
    # CONTROL LOOP
    # =========================
    def _control_loop(self):
        if self._path_ref is None:
            self._publish_zero()
            return

        # Match planning_node.camera_to_robot_center() and its +Z forward convention.
        camera_offset = np.array([0.0, 0.0, 0.35], dtype=np.float64)  # GO2 control center to camera.
        robot_pos = self.position - self.rotation @ camera_offset
        forward = self.rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        robot_yaw = math.atan2(forward[1], forward[0])

        # Select the reference point to track.
        target = self._find_tracking_target(robot_pos, robot_yaw)
        if target is None:
            self._publish_zero()
            return

        # Convert tracking error into the robot frame.
        tx, ty, heading_err = self._target_error(robot_pos, robot_yaw, target)
        v_ref = float(target[3])
        w_ref = float(target[4])

        b = 2.0
        zeta = 0.7
        k = 2.0 * zeta * math.sqrt(w_ref * w_ref + b * v_ref * v_ref)
        v = v_ref * math.cos(heading_err) + k * tx
        wz = w_ref + k * heading_err + b * v_ref * self._sinc(heading_err) * ty
        v = float(np.clip(v, -0.2, 0.6))
        wz = float(np.clip(wz, -0.8, 0.8))

        heading_to_goal = self._wrap_angle(float(self._path_ref[-1, 2]) - robot_yaw)
        if (
            np.linalg.norm(robot_pos[:2] - self._path_ref[-1, :2]) < 0.1
            and abs(heading_to_goal) < 0.1
        ):
            self._publish_zero()
            return

        # Publish command.
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)

        self.logger.info("cmd v=%.3f wz=%.3f", v, wz)

    # =========================
    # Tracking core
    # =========================
    def _find_tracking_target(self, robot_pos, robot_yaw):
        if self._path_ref is None or len(self._path_ref) == 0:
            return None

        start_idx = int(np.clip(self._track_idx, 0, len(self._path_ref) - 1))
        delta = self._path_ref[start_idx:, :2] - robot_pos[:2]
        dist = np.linalg.norm(delta, axis=1)

        if float(np.max(dist)) < 0.05:
            yaw_err = np.abs([self._wrap_angle(float(yaw) - robot_yaw) for yaw in self._path_ref[start_idx:, 2]])
            nearest_idx = start_idx + int(np.argmin(yaw_err))
        else:
            nearest_idx = start_idx + int(np.argmin(dist))

        target_idx = min(nearest_idx + 1, len(self._path_ref) - 1)  # Track one point ahead for stability.
        self._track_idx = nearest_idx
        return self._path_ref[target_idx]

    def _target_error(self, robot_pos, robot_yaw, target):
        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]

        cy = math.cos(robot_yaw)
        sy = math.sin(robot_yaw)

        tx = cy * dx + sy * dy
        ty = -sy * dx + cy * dy
        heading_err = self._wrap_angle(float(target[2]) - robot_yaw)

        return tx, ty, heading_err

    @staticmethod
    def _wrap_angle(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _sinc(a):
        if abs(a) < 1e-6:
            return 1.0
        return math.sin(a) / a

    # =========================
    # Utils
    # =========================
    def _publish_zero(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def _now_sec(self):
        if self._odom_stamp_sec is not None:
            return self._odom_stamp_sec
        return self.get_clock().now().nanoseconds * 1e-9

    def destroy_node(self):
        self.logger.info("Destroying node")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="cmd_vel_control.log")
    parsed_args, _ = parser.parse_known_args(sys.argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(parsed_args.log_file),
        ],
    )

    node = CmdVelControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()