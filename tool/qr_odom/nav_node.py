#!/usr/bin/env python3
"""
tool/qr_nav_node.py

Navigate to a fixed target pose defined relative to an AprilTag board.
Does NOT use camera — pure odometry-based control after goal is set.
Holonomic PI controller (same as tool/qr_odom/target_node.py).

Frame chain
-----------
  Goal (fixed in map frame):
    T_map_goal = T_map_qrworld @ T_qrworld_robot   (both predefined)

  Current pose in map frame:
    T_world_camera  ← /slam/odometry_fused
    T_world_map     ← TF world→map  (broadcast by map_node)
    T_map_camera    = inv(T_world_map) @ T_world_camera
    T_map_robot     = T_map_camera @ T_CAMERA_ROBOT

  Control error (robot frame):
    T_robot_goal = inv(T_map_robot) @ T_map_goal

Calibration files
-----------------
  tinynav_db/qrcode/tag_mappose.json   T_map_qrworld
  tinynav_db/qrcode/tag_target.json    T_qrworld_robot  (existing)

Topics
------
  Subscribed:  /slam/odometry_fused    nav_msgs/Odometry
  TF lookup:   world → map             (broadcast by map_node)
  Published:   /control/cmd_vel        geometry_msgs/Twist
               /qr_world/nav_done      std_msgs/Bool        (once, on reaching goal)
"""

import json
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener

from tinynav.core.math_utils import msg2np, tf2np

DB_DIR           = Path("tinynav_db/qrcode")
TAG_MAPPOSE_PATH = DB_DIR / "tag_mappose.json"
TARGET_PATH      = DB_DIR / "tag_target.json"

ODOM_TOPIC     = "/slam/odometry_fused"
CMD_VEL_TOPIC  = "/control/cmd_vel"
NAV_DONE_TOPIC = "/qr_world/nav_done"

# PI controller gains, limits, and drivetrain deadband compensation
# (same as tool/qr_odom/target_node.py).
K_LINEAR        = 0.5   # (m/s) / m
K_LINEAR_I      = 0.05  # (m/s) / (m*s)
K_ANGULAR       = 1.0   # (rad/s) / rad
K_ANGULAR_I     = 0.10  # (rad/s) / (rad*s)
MAX_LINEAR      = 0.3   # m/s
MAX_LINEAR_I    = 0.05  # m/s — max integral contribution
MAX_ANGULAR     = 0.5   # rad/s
MAX_ANGULAR_I   = 0.10  # rad/s — max integral contribution
MIN_LINEAR      = 0.15  # m/s — below this the base may not move
MIN_ANGULAR     = 0.15  # rad/s — below this the base may not rotate
DIST_THRESH     = 0.06  # m  — switch from approach to heading-align
HEADING_THRESH  = 0.06  # rad — stop when aligned
CMD_DEADBAND    = 1e-3
MAX_CONTROL_DT  = 0.1   # s — cap integral step after odom stalls

# camera → robot frame transform (same convention as qr_target_node.py)
R_CAMERA_ROBOT = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=np.float64)
T_CAMERA_ROBOT = np.eye(4, dtype=np.float64)
T_CAMERA_ROBOT[:3, :3] = R_CAMERA_ROBOT


def _clip_with_min(value: float, min_abs: float, max_abs: float) -> float:
    if abs(value) < CMD_DEADBAND:
        return 0.0
    clipped = float(np.clip(value, -max_abs, max_abs))
    if abs(clipped) < min_abs:
        return float(np.sign(clipped) * min_abs)
    return clipped


class QRNavNode(Node):
    def __init__(self):
        super().__init__("qr_nav_node")

        d_map    = json.loads(TAG_MAPPOSE_PATH.read_text())
        d_target = json.loads(TARGET_PATH.read_text())

        T_map_qrworld   = np.array(d_map["T_map_qrworld"])
        T_qrworld_robot = np.array(d_target["T_qrworld_robot"])

        # Fixed goal in map frame — does not change with SLAM drift
        self._T_map_goal: np.ndarray = T_map_qrworld @ T_qrworld_robot

        self._T_world_camera: np.ndarray | None = None

        self._linear_error_i = np.zeros(2, dtype=np.float64)
        self._heading_error_i = 0.0
        self._last_control_time: float | None = None
        self._reached = False

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(Odometry, ODOM_TOPIC, self._odom_cb, 100)
        self._cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self._nav_done_pub = self.create_publisher(Bool, NAV_DONE_TOPIC, 10)

        self.get_logger().info(
            f"qr_nav_node: goal fixed in map frame, subscribing {ODOM_TOPIC}"
        )

    def _odom_cb(self, msg: Odometry) -> None:
        T_world_camera, _ = msg2np(msg)
        self._T_world_camera = T_world_camera
        self._control(msg.header.stamp)

    def _publish_reached_once(self) -> None:
        if self._reached:
            return
        self._reached = True
        self._nav_done_pub.publish(Bool(data=True))
        self.get_logger().info("qr target reached.")

    def _control_dt(self) -> float:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_control_time is None:
            self._last_control_time = now
            return 0.0
        dt = min(max(now - self._last_control_time, 0.0), MAX_CONTROL_DT)
        self._last_control_time = now
        return dt

    def _linear_pi_cmd(self, dx: float, dy: float, dt: float) -> tuple[float, float]:
        self._linear_error_i += np.array([dx, dy], dtype=np.float64) * dt
        linear_i_limit = MAX_LINEAR_I / K_LINEAR_I
        self._linear_error_i = np.clip(self._linear_error_i, -linear_i_limit, linear_i_limit)
        return (
            _clip_with_min(
                K_LINEAR * dx + K_LINEAR_I * self._linear_error_i[0],
                MIN_LINEAR, MAX_LINEAR,
            ),
            _clip_with_min(
                K_LINEAR * dy + K_LINEAR_I * self._linear_error_i[1],
                MIN_LINEAR, MAX_LINEAR,
            ),
        )

    def _heading_pi_cmd(self, heading_error: float, dt: float) -> float:
        self._heading_error_i += heading_error * dt
        angular_i_limit = MAX_ANGULAR_I / K_ANGULAR_I
        self._heading_error_i = float(np.clip(self._heading_error_i, -angular_i_limit, angular_i_limit))
        return _clip_with_min(
            K_ANGULAR * heading_error + K_ANGULAR_I * self._heading_error_i,
            MIN_ANGULAR, MAX_ANGULAR,
        )

    def _control(self, stamp) -> None:
        if self._T_world_camera is None:
            return

        # Get T_world_map from TF
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                "world", "map", rclpy.time.Time()
            )
            _, _, T_world_map = tf2np(tf_msg)
        except Exception:
            self.get_logger().warn(
                "TF world→map not available", throttle_duration_sec=2.0)
            return

        # Current robot pose in map frame
        T_map_world  = np.linalg.inv(T_world_map)
        T_map_camera = T_map_world @ self._T_world_camera
        T_map_robot  = T_map_camera @ T_CAMERA_ROBOT

        # Error in robot frame
        T_robot_goal  = np.linalg.inv(T_map_robot) @ self._T_map_goal
        dx            = T_robot_goal[0, 3]
        dy            = T_robot_goal[1, 3]
        dist          = np.hypot(dx, dy)
        heading_error = np.arctan2(T_robot_goal[1, 0], T_robot_goal[0, 0])
        dt            = self._control_dt()

        cmd = Twist()
        if dist > DIST_THRESH:
            self._heading_error_i = 0.0
            self._reached = False
            cmd.linear.x, cmd.linear.y = self._linear_pi_cmd(dx, dy, dt)
        elif abs(heading_error) > HEADING_THRESH:
            self._linear_error_i[:] = 0.0
            self._reached = False
            cmd.angular.z = self._heading_pi_cmd(heading_error, dt)
        else:
            self._linear_error_i[:] = 0.0
            self._heading_error_i = 0.0
            self._publish_reached_once()

        self._cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = QRNavNode()
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
