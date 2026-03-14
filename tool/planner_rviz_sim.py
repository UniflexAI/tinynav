#!/usr/bin/env python3
"""
RViz-interactive planner simulation harness.

What it does:
- Simulates a simple robot state in 2D (x, y, heading)
- Publishes:
  - /slam/odometry (Odometry)
  - /slam/depth (Image, 32FC1)
  - /camera/camera/infra2/camera_info (CameraInfo)
- Subscribes:
  - /cmd_vel (Twist) from planner
  - /goal_pose (PoseStamped) from RViz "2D Goal Pose"
  - /initialpose (PoseWithCovarianceStamped) from RViz "2D Pose Estimate"
- Re-publishes RViz goal as /control/target_pose (Odometry) for planner.

Usage (example):
  # terminal A
  source /opt/ros/humble/setup.bash
  export PYTHONPATH=/home/xiaolefang/workspace/tinynav:/home/xiaolefang/workspace/tinynav/.venv/lib/python3.10/site-packages:$PYTHONPATH
  python3 /home/xiaolefang/workspace/tinynav/tool/planner_rviz_sim.py

  # terminal B
  source /opt/ros/humble/setup.bash
  export PYTHONPATH=/home/xiaolefang/workspace/tinynav:/home/xiaolefang/workspace/tinynav/.venv/lib/python3.10/site-packages:$PYTHONPATH
  python3 -m tinynav.core.planning_node --sensor_source realsense

Then in RViz:
- Fixed Frame: world
- Add displays: Path(/planning/trajectory_path), Odometry(/slam/odometry), PointCloud2(/planning/occupied_voxels)
- Use "2D Goal Pose" to click goal; robot will navigate toward it.
"""

import math
import argparse
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


def yaw_to_quat(yaw: float):
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


class PlannerRvizSim(Node):
    def __init__(
        self,
        hz: float = 20.0,
        depth_m: float = 2.5,
        scene: str = "flat",
        target_mode: str = "once",
    ):
        super().__init__("planner_rviz_sim")

        self.bridge = CvBridge()

        self.odom_pub = self.create_publisher(Odometry, "/slam/odometry", 10)
        self.depth_pub = self.create_publisher(Image, "/slam/depth", 10)
        self.cam_info_pub = self.create_publisher(CameraInfo, "/camera/camera/infra2/camera_info", 10)
        self.target_pub = self.create_publisher(Odometry, "/control/target_pose", 10)
        self.poi_change_pub = self.create_publisher(Odometry, "/mapping/poi_change", 10)

        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.initialpose_callback, 10)

        self.hz = float(hz)
        self.dt = 1.0 / self.hz
        self.depth_m = float(depth_m)
        self.scene = scene
        self.target_mode = target_mode

        self.goal_x = 3.0
        self.goal_y = 0.0
        self.have_goal = False
        self.goal_publish_latch = False

        self.width = 424
        self.height = 240
        self.fx = 300.0
        self.fy = 300.0
        self.cx = self.width * 0.5
        self.cy = self.height * 0.5
        self.baseline = 0.05

        # Sim robot state
        self.x = 0.0
        self.y = 0.0
        self.z = 0.6
        self.yaw = 0.0

        self.v = 0.0
        self.w = 0.0

        self.last_stamp_ns = self.get_clock().now().nanoseconds

        self.timer = self.create_timer(self.dt, self.tick)
        self.get_logger().info("planner_rviz_sim started. Use RViz 2D Goal Pose (/goal_pose).")

    def cmd_callback(self, msg: Twist):
        # Planner controls x forward and z yaw-rate.
        self.v = float(msg.linear.x)
        self.w = float(msg.angular.z)

    def goal_callback(self, msg: PoseStamped):
        self.goal_x = float(msg.pose.position.x)
        self.goal_y = float(msg.pose.position.y)
        self.have_goal = True
        self.goal_publish_latch = False

        # Trigger planner reset path branch if needed.
        dummy = Odometry()
        dummy.header.stamp = self.get_clock().now().to_msg()
        dummy.header.frame_id = "world"
        dummy.child_frame_id = "map"
        self.poi_change_pub.publish(dummy)

        self.get_logger().info(
            f"New goal: x={self.goal_x:.2f}, y={self.goal_y:.2f}"
        )

    def initialpose_callback(self, msg: PoseWithCovarianceStamped):
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.z = float(msg.pose.pose.position.z if abs(msg.pose.pose.position.z) > 1e-6 else 0.6)
        q = msg.pose.pose.orientation
        # standard yaw extraction for world-z rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        self.get_logger().info(
            f"Reset pose to x={self.x:.2f}, y={self.y:.2f}, yaw={self.yaw:.2f}"
        )

    def make_camera_info(self, stamp_msg):
        msg = CameraInfo()
        msg.header.stamp = stamp_msg
        msg.header.frame_id = "camera"
        msg.width = self.width
        msg.height = self.height

        msg.k = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0,
        ]

        # right camera projection matrix: P[0,3] = -fx * baseline
        msg.p = [
            self.fx, 0.0, self.cx, -self.fx * self.baseline,
            0.0, self.fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return msg

    def build_depth_image(self):
        depth = np.full((self.height, self.width), self.depth_m, dtype=np.float32)

        if self.scene == "flat":
            return depth

        # Add synthetic obstacles in camera depth image (forward is image center area).
        if self.scene in ("corridor", "stairs"):
            # left/right walls with a forward gap
            depth[:, :35] = np.minimum(depth[:, :35], 0.7)
            depth[:, -35:] = np.minimum(depth[:, -35:], 0.7)

        if self.scene in ("obstacles", "stairs"):
            # center obstacle block
            r0, r1 = int(self.height * 0.45), int(self.height * 0.75)
            c0, c1 = int(self.width * 0.44), int(self.width * 0.56)
            depth[r0:r1, c0:c1] = np.minimum(depth[r0:r1, c0:c1], 0.45)

            # second obstacle slightly right
            r0, r1 = int(self.height * 0.35), int(self.height * 0.62)
            c0, c1 = int(self.width * 0.66), int(self.width * 0.78)
            depth[r0:r1, c0:c1] = np.minimum(depth[r0:r1, c0:c1], 0.55)

        if self.scene == "stairs":
            # stair-like bands: closer depth on lower image rows
            for k in range(6):
                rs = int(self.height * (0.55 + 0.06 * k))
                re = min(self.height, rs + int(self.height * 0.045))
                z = max(0.35, 1.3 - 0.15 * k)
                depth[rs:re, :] = np.minimum(depth[rs:re, :], z)

        return depth

    def publish_depth(self, stamp_msg):
        depth = self.build_depth_image()
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header.stamp = stamp_msg
        depth_msg.header.frame_id = "camera"
        self.depth_pub.publish(depth_msg)

    def publish_odom(self, stamp_msg, vx: float):
        # Build rotation so that local +z is heading direction in world XY.
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)

        # x_local, y_local(up), z_local(forward)
        x_local = np.array([-s, c, 0.0], dtype=np.float64)
        y_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_local = np.array([c, s, 0.0], dtype=np.float64)

        R = np.column_stack([x_local, y_local, z_local])

        # Convert matrix to quaternion (robust for this orthonormal R)
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0.0:
            t = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * t
            qx = (R[2, 1] - R[1, 2]) / t
            qy = (R[0, 2] - R[2, 0]) / t
            qz = (R[1, 0] - R[0, 1]) / t
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            t = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / t
            qx = 0.25 * t
            qy = (R[0, 1] + R[1, 0]) / t
            qz = (R[0, 2] + R[2, 0]) / t
        elif R[1, 1] > R[2, 2]:
            t = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / t
            qx = (R[0, 1] + R[1, 0]) / t
            qy = 0.25 * t
            qz = (R[1, 2] + R[2, 1]) / t
        else:
            t = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / t
            qx = (R[0, 2] + R[2, 0]) / t
            qy = (R[1, 2] + R[2, 1]) / t
            qz = 0.25 * t

        odom = Odometry()
        odom.header.stamp = stamp_msg
        odom.header.frame_id = "world"
        odom.child_frame_id = "camera"
        odom.pose.pose.position.x = float(self.x)
        odom.pose.pose.position.y = float(self.y)
        odom.pose.pose.position.z = float(self.z)
        odom.pose.pose.orientation.x = float(qx)
        odom.pose.pose.orientation.y = float(qy)
        odom.pose.pose.orientation.z = float(qz)
        odom.pose.pose.orientation.w = float(qw)

        # planner code interprets twist.linear as world-frame velocity then projects.
        odom.twist.twist.linear.x = float(vx * math.cos(self.yaw))
        odom.twist.twist.linear.y = float(vx * math.sin(self.yaw))
        odom.twist.twist.linear.z = 0.0

        self.odom_pub.publish(odom)

    def publish_target(self, stamp_msg):
        if not self.have_goal:
            return
        if self.target_mode == "once" and self.goal_publish_latch:
            return

        odom = Odometry()
        odom.header.stamp = stamp_msg
        odom.header.frame_id = "world"
        odom.child_frame_id = "world"
        odom.pose.pose.position.x = float(self.goal_x)
        odom.pose.pose.position.y = float(self.goal_y)
        odom.pose.pose.position.z = float(self.z)
        odom.pose.pose.orientation.w = 1.0
        self.target_pub.publish(odom)
        self.goal_publish_latch = True

    def tick(self):
        now_ns = self.get_clock().now().nanoseconds
        dt = max(1e-3, (now_ns - self.last_stamp_ns) * 1e-9)
        self.last_stamp_ns = now_ns

        # Very simple unicycle dynamics in world XY
        v = float(np.clip(self.v, -0.5, 0.5))
        w = float(np.clip(self.w, -1.5, 1.5))

        self.yaw += w * dt
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt

        stamp_msg = self.get_clock().now().to_msg()
        self.publish_target(stamp_msg)
        self.publish_odom(stamp_msg, v)
        self.publish_depth(stamp_msg)
        self.cam_info_pub.publish(self.make_camera_info(stamp_msg))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=float, default=20.0, help="Publish/control rate")
    parser.add_argument("--depth", type=float, default=2.5, help="Synthetic base depth in meters")
    parser.add_argument(
        "--scene",
        type=str,
        choices=["flat", "corridor", "obstacles", "stairs"],
        default="flat",
        help="Synthetic depth scene profile",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        choices=["once", "continuous"],
        default="once",
        help="How /control/target_pose is published after RViz goal",
    )
    args = parser.parse_args()

    rclpy.init()
    node = PlannerRvizSim(
        hz=args.hz,
        depth_m=args.depth,
        scene=args.scene,
        target_mode=args.target_mode,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
