#!/usr/bin/env python3
"""
tool/qr_nav_node.py

Navigate to a fixed target pose defined relative to an AprilTag board.
Does NOT use camera — pure odometry-based control after goal is set.

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
"""

import json
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from tinynav.core.math_utils import msg2np, tf2np

DB_DIR           = Path("tinynav_db/qrcode")
TAG_MAPPOSE_PATH = DB_DIR / "tag_mappose.json"
TARGET_PATH      = DB_DIR / "tag_target.json"

ODOM_TOPIC    = "/slam/odometry_fused"
CMD_VEL_TOPIC = "/control/cmd_vel"

# Proportional controller gains (same as qr_target_node.py)
K_LINEAR      = 0.5
K_ANGULAR     = 1.0
MAX_LINEAR    = 0.3   # m/s
MAX_ANGULAR   = 0.5   # rad/s
DIST_THRESH   = 0.05  # m
HEADING_THRESH = 0.05 # rad

# camera → robot frame transform (same convention as qr_target_node.py)
R_CAMERA_ROBOT = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=np.float64)
T_CAMERA_ROBOT = np.eye(4, dtype=np.float64)
T_CAMERA_ROBOT[:3, :3] = R_CAMERA_ROBOT


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

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(Odometry, ODOM_TOPIC, self._odom_cb, 100)
        self._cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)

        self.get_logger().info(
            f"qr_nav_node: goal fixed in map frame, subscribing {ODOM_TOPIC}"
        )

    def _odom_cb(self, msg: Odometry) -> None:
        T_world_camera, _ = msg2np(msg)
        self._T_world_camera = T_world_camera
        self._control(msg.header.stamp)

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
        T_robot_goal = np.linalg.inv(T_map_robot) @ self._T_map_goal
        dx      = T_robot_goal[0, 3]
        dy      = T_robot_goal[1, 3]
        dist    = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx)
        dtheta  = np.arctan2(T_robot_goal[1, 0], T_robot_goal[0, 0])

        cmd = Twist()
        if dist > DIST_THRESH:
            cmd.linear.x  = float(np.clip(K_LINEAR  * dx,      -MAX_LINEAR,  MAX_LINEAR))
            cmd.angular.z = float(np.clip(K_ANGULAR * bearing, -MAX_ANGULAR, MAX_ANGULAR))
        elif abs(dtheta) > HEADING_THRESH:
            cmd.angular.z = float(np.clip(K_ANGULAR * dtheta, -MAX_ANGULAR, MAX_ANGULAR))
        else:
            self.get_logger().info("Goal reached.", throttle_duration_sec=5.0)

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
