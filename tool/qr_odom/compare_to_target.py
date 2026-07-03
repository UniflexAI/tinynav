#!/usr/bin/env python3
"""
tool/qr_odom/compare_to_target.py

Diagnostic: how far is the current pose from the calibrated target_pose,
comparing raw QR odometry against the EKF-fused odometry.

Frame chain (same as nav_node.py)
----------------------------------
  T_map_goal  = T_map_qrworld @ T_qrworld_robot          (fixed, from calibration)
  T_map_robot = inv(T_world_map) @ T_world_camera @ T_CAMERA_ROBOT

Topics
------
  Subscribed:  /qr/odom               nav_msgs/Odometry   (raw QR pose)
               /slam/odometry_fused   nav_msgs/Odometry   (EKF-fused pose)
  TF lookup:   world → map            (broadcast by map_node)

Usage
-----
  python tool/qr_odom/compare_to_target.py
"""

import json
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from tinynav.core.math_utils import msg2np, tf2np
from tool.qr_odom.robot_frame import T_CAMERA_ROBOT


def _yaw_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Yaw-only angle (deg) of the relative rotation between two 3×3 rotation
    matrices (same convention as nav_node.py's atan2(R[1,0], R[0,0]))."""
    relative = R_a.T @ R_b
    return float(np.degrees(np.arctan2(relative[1, 0], relative[0, 0])))

DB_DIR           = Path("tinynav_db/qrcode")
TAG_MAPPOSE_PATH = DB_DIR / "tag_mappose.json"
TARGET_PATH      = DB_DIR / "tag_target.json"

QR_ODOM_TOPIC  = "/qr/odom"
EKF_ODOM_TOPIC = "/slam/odometry_fused"



class CompareToTargetNode(Node):
    def __init__(self):
        super().__init__("qr_compare_to_target_node")

        d_map    = json.loads(TAG_MAPPOSE_PATH.read_text())
        d_target = json.loads(TARGET_PATH.read_text())
        T_map_qrworld   = np.array(d_map["T_map_qrworld"])
        T_qrworld_robot = np.array(d_target["T_qrworld_robot"])
        self._T_map_goal = T_map_qrworld @ T_qrworld_robot
        self._goal_pos   = self._T_map_goal[:3, 3]
        self._goal_R     = self._T_map_goal[:3, :3]

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(Odometry, QR_ODOM_TOPIC,  self._qr_cb,  10)
        self.create_subscription(Odometry, EKF_ODOM_TOPIC, self._ekf_cb, 100)

        self.get_logger().info(
            f"Comparing {QR_ODOM_TOPIC} and {EKF_ODOM_TOPIC} against target_pose "
            f"(map frame goal position = {self._goal_pos.tolist()})"
        )

    def _lookup_T_world_map(self) -> np.ndarray | None:
        try:
            tf_msg = self._tf_buffer.lookup_transform("world", "map", rclpy.time.Time())
        except Exception:
            self.get_logger().warn("TF world→map not available", throttle_duration_sec=2.0)
            return None
        _, _, T_world_map = tf2np(tf_msg)
        return T_world_map

    def _error_to_goal(self, T_world_camera: np.ndarray) -> tuple[float, float] | None:
        T_world_map = self._lookup_T_world_map()
        if T_world_map is None:
            return None
        T_map_robot = np.linalg.inv(T_world_map) @ T_world_camera @ T_CAMERA_ROBOT
        dist  = float(np.linalg.norm(T_map_robot[:3, 3] - self._goal_pos))
        angle = _yaw_angle_deg(self._goal_R, T_map_robot[:3, :3])
        return dist, angle

    def _qr_cb(self, msg: Odometry) -> None:
        T_world_camera, _ = msg2np(msg)
        err = self._error_to_goal(T_world_camera)
        if err is not None:
            dist, angle = err
            self.get_logger().info(
                f"[qr_odom]  dist_to_target = {dist:.3f} m  yaw_to_target = {angle:.2f} deg",
                throttle_duration_sec=0.5)

    def _ekf_cb(self, msg: Odometry) -> None:
        T_world_camera, _ = msg2np(msg)
        err = self._error_to_goal(T_world_camera)
        if err is not None:
            dist, angle = err
            self.get_logger().info(
                f"[ekf_fused] dist_to_target = {dist:.3f} m  yaw_to_target = {angle:.2f} deg",
                throttle_duration_sec=0.5)


def main(args=None):
    rclpy.init(args=args)
    node = CompareToTargetNode()
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
