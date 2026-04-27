#!/usr/bin/env python3
"""
lookat — Rotate the robot to face a map-frame target coordinate.

Usage:
  ros2 run tinynav lookat --target x,y,z [--tolerance 5] [--timeout 10]

Blocks until the robot's heading is within <tolerance> degrees of the target
direction, or until <timeout> seconds elapse.

Example:
  ros2 run tinynav lookat --target 5.0,2.0,0.0
  ros2 run tinynav lookat --target 5,2,0 --tolerance 3 --timeout 15
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Odometry
from rclpy.node import Node

from tinynav.core.math_utils import np2msg


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


class LookAtSkill(Node):
    def __init__(self, target: list, tolerance_deg: float, timeout_s: float):
        super().__init__("skill_lookat")
        self.target = target
        self.tolerance_rad = math.radians(tolerance_deg)
        self.timeout = timeout_s
        self.aligned = False
        self.desired_yaw: float | None = None
        self.current_pose: np.ndarray | None = None

        # Subscribe to pose in map frame
        self.create_subscription(Odometry, "/mapping/current_pose_in_map", self._pose_cb, 10)
        # Publish target pose for controller
        self.target_pose_pub = self.create_publisher(Odometry, "/control/target_pose", 10)

        self.get_logger().info(f"lookat: target={target}, tolerance={tolerance_deg}°, timeout={timeout_s}s")

    def _pose_cb(self, msg: Odometry) -> None:
        # Convert Odometry msg to 4x4 matrix
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        from scipy.spatial.transform import Rotation as Rot
        T = np.eye(4)
        T[:3, :3] = Rot.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
        T[:3, 3] = [pos.x, pos.y, pos.z]
        self.current_pose = T

        # Compute desired yaw on first pose update
        if self.desired_yaw is None:
            dx = self.target[0] - T[0, 3]
            dy = self.target[1] - T[1, 3]
            self.desired_yaw = math.atan2(dy, dx)
            # Publish target pose: same position, desired heading
            target_mat = np.eye(4)
            target_mat[:3, 3] = T[:3, 3]
            c, s = math.cos(self.desired_yaw), math.sin(self.desired_yaw)
            target_mat[0, 0], target_mat[0, 1] = c, -s
            target_mat[1, 0], target_mat[1, 1] = s, c
            stamp = self.get_clock().now().to_msg()
            self.target_pose_pub.publish(np2msg(target_mat, stamp, "world", "map"))
            self.get_logger().info(f"lookat: target yaw={math.degrees(self.desired_yaw):.1f}°")

        # Check alignment
        if self.desired_yaw is not None:
            current_yaw = math.atan2(T[1, 0], T[0, 0])
            if abs(_angle_diff(current_yaw, self.desired_yaw)) < self.tolerance_rad:
                self.get_logger().info("lookat: ✓ aligned")
                self.aligned = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Rotate to face a target (blocking)")
    parser.add_argument("--target", required=True, help="Target x,y,z in map frame")
    parser.add_argument("--tolerance", type=float, default=5.0, help="Heading tolerance in degrees")
    parser.add_argument("--timeout", type=float, default=10.0, help="Max wait in seconds")
    args = parser.parse_args()

    target = [float(v) for v in args.target.split(",")]
    if len(target) != 3:
        print("Error: --target must be x,y,z", file=sys.stderr)
        return 1

    rclpy.init()
    node = LookAtSkill(target, args.tolerance, args.timeout)
    deadline = time.time() + args.timeout
    while rclpy.ok() and not node.aligned and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    if not node.aligned:
        node.get_logger().warn("lookat: timeout, exiting anyway")
    else:
        node.get_logger().info("lookat: done")

    node.destroy_node()
    rclpy.shutdown()
    return 0 if node.aligned else 2


if __name__ == "__main__":
    raise SystemExit(main())
