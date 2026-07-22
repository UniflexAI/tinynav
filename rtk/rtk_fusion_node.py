#!/usr/bin/env python3
import copy
import json
import math
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def odom_position(msg: Odometry) -> np.ndarray:
    p = msg.pose.pose.position
    return np.array([p.x, p.y, p.z], dtype=np.float64)


def yaw_matrix(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotate_quat_z(quat_msg, yaw: float):
    quat = [quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w]
    out = R.from_euler("z", yaw) * R.from_quat(quat)
    x, y, z, w = out.as_quat()
    quat_msg.x = float(x)
    quat_msg.y = float(y)
    quat_msg.z = float(z)
    quat_msg.w = float(w)


def parse_stage_filter(value: str) -> set[str]:
    return {part.strip().upper() for part in str(value or "").split(",") if part.strip()}


class RtkFusionNode(Node):
    def __init__(self):
        super().__init__("rtk_fusion_node")
        self._declare_params()
        self.slam_topic = self.get_parameter("slam_odom_topic").value
        self.rtk_topic = self.get_parameter("rtk_odom_topic").value
        self.fused_topic = self.get_parameter("fused_odom_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.fused_frame_id = self.get_parameter("fused_frame_id").value
        self.correction_alpha = float(self.get_parameter("correction_alpha").value)
        self.min_alignment_baseline_m = float(self.get_parameter("min_alignment_baseline_m").value)
        self.max_rtk_age_s = float(self.get_parameter("max_rtk_age_s").value)
        self.max_slam_age_s = float(self.get_parameter("max_slam_age_s").value)
        self.max_rtk_position_std_m = float(self.get_parameter("max_rtk_position_std_m").value)
        self.max_rtk_jump_m = float(self.get_parameter("max_rtk_jump_m").value)
        self.max_offset_step_m = float(self.get_parameter("max_offset_step_m").value)
        self.rotate_slam_orientation = bool(self.get_parameter("rotate_slam_orientation").value)
        self.require_fix_status = bool(self.get_parameter("require_fix_status").value)
        self.min_navsat_status = int(self.get_parameter("min_navsat_status").value)
        self.max_fix_age_s = float(self.get_parameter("max_fix_age_s").value)
        self.age_from_arrival = bool(self.get_parameter("age_from_arrival").value)
        self.receiver_status_topic = self.get_parameter("receiver_status_topic").value
        self.use_receiver_status_gate = bool(self.get_parameter("use_receiver_status_gate").value)
        self.require_receiver_status = bool(self.get_parameter("require_receiver_status").value)
        self.max_receiver_status_age_s = float(self.get_parameter("max_receiver_status_age_s").value)
        self.allowed_receiver_stages = parse_stage_filter(self.get_parameter("allowed_receiver_stages").value)

        self.latest_slam: Odometry | None = None
        self.latest_rtk: Odometry | None = None
        self.latest_fix: NavSatFix | None = None
        self.latest_receiver_status = None
        self.last_receiver_status_time = None
        self.last_fix_arrival = None
        self.last_slam_arrival = None
        self.last_rtk_arrival = None
        self.last_used_rtk_position: np.ndarray | None = None
        self.anchor_slam: np.ndarray | None = None
        self.anchor_rtk: np.ndarray | None = None
        self.yaw_offset = 0.0
        self.alignment_ready = False
        self.offset = np.zeros(3, dtype=np.float64)
        self.offset_ready = False
        self.last_reject_reason = "waiting_for_data"
        self.last_rtk_used_time = None

        self.fused_pub = self.create_publisher(Odometry, self.fused_topic, 50)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.create_subscription(Odometry, self.slam_topic, self.slam_callback, 50)
        self.create_subscription(Odometry, self.rtk_topic, self.rtk_callback, 10)
        self.create_subscription(NavSatFix, self.get_parameter("fix_topic").value, self.fix_callback, 10)
        self.create_subscription(String, self.receiver_status_topic, self.receiver_status_callback, 10)
        self.create_timer(0.5, self.publish_status)
        self.get_logger().info(
            f"RTK fusion ready. slam={self.slam_topic}, rtk={self.rtk_topic}, fused={self.fused_topic}"
        )

    def _declare_params(self):
        self.declare_parameter("slam_odom_topic", "/slam/odometry")
        self.declare_parameter("rtk_odom_topic", "/rtk/odom")
        self.declare_parameter("fused_odom_topic", "/slam/odometry_fused")
        self.declare_parameter("status_topic", "/rtk/fusion_status")
        self.declare_parameter("fused_frame_id", "world")
        self.declare_parameter("correction_alpha", 0.05)
        self.declare_parameter("min_alignment_baseline_m", 2.0)
        self.declare_parameter("max_rtk_age_s", 1.5)
        self.declare_parameter("max_slam_age_s", 0.5)
        self.declare_parameter("max_rtk_position_std_m", 1.0)
        self.declare_parameter("max_rtk_jump_m", 3.0)
        self.declare_parameter("max_offset_step_m", 0.25)
        self.declare_parameter("rotate_slam_orientation", True)
        self.declare_parameter("fix_topic", "/fix")
        self.declare_parameter("require_fix_status", True)
        self.declare_parameter("min_navsat_status", int(NavSatStatus.STATUS_GBAS_FIX))
        self.declare_parameter("max_fix_age_s", 1.5)
        # Gate staleness on local arrival time (monotonic) rather than the GNSS
        # UTC timestamp inside /fix and /rtk/odom. The receiver clock and the
        # host clock need not be NTP-synced; comparing a GPS-UTC stamp against
        # the node clock can make every message look stale (or falsely fresh).
        self.declare_parameter("age_from_arrival", True)
        self.declare_parameter("receiver_status_topic", "/rtk/receiver_status")
        self.declare_parameter("use_receiver_status_gate", True)
        self.declare_parameter("require_receiver_status", False)
        self.declare_parameter("max_receiver_status_age_s", 1.5)
        self.declare_parameter("allowed_receiver_stages", "RTK_FLOAT,RTK_FIXED")

    def slam_callback(self, msg: Odometry):
        self.latest_slam = msg
        self.last_slam_arrival = time.monotonic()
        fused = self.make_fused_odom(msg)
        self.fused_pub.publish(fused)

    def fix_callback(self, msg: NavSatFix):
        self.latest_fix = msg
        self.last_fix_arrival = time.monotonic()

    def receiver_status_callback(self, msg: String):
        try:
            self.latest_receiver_status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.last_reject_reason = "bad_receiver_status_json"
            return
        self.last_receiver_status_time = time.monotonic()

    def rtk_callback(self, msg: Odometry):
        self.latest_rtk = msg
        self.last_rtk_arrival = time.monotonic()
        if self.latest_slam is None:
            self.last_reject_reason = "waiting_for_slam"
            return
        if not self.rtk_quality_ok(msg):
            return
        slam_pos = odom_position(self.latest_slam)
        rtk_pos = odom_position(msg)
        if self.last_used_rtk_position is not None:
            jump = float(np.linalg.norm(rtk_pos[:2] - self.last_used_rtk_position[:2]))
            if jump > self.max_rtk_jump_m:
                self.last_reject_reason = f"rtk_jump_{jump:.2f}m"
                return

        self.update_alignment(slam_pos, rtk_pos)
        aligned_slam = yaw_matrix(self.yaw_offset) @ slam_pos
        target_offset = rtk_pos - aligned_slam
        if self.offset_ready:
            step = target_offset - self.offset
            step_norm = float(np.linalg.norm(step[:2]))
            if step_norm > self.max_offset_step_m:
                step *= self.max_offset_step_m / max(step_norm, 1e-6)
            self.offset += self.correction_alpha * step
        else:
            self.offset = target_offset
            self.offset_ready = True

        self.last_used_rtk_position = rtk_pos
        self.last_rtk_used_time = time.monotonic()
        self.last_reject_reason = "using_rtk"

    def update_alignment(self, slam_pos: np.ndarray, rtk_pos: np.ndarray):
        if self.anchor_slam is None:
            self.anchor_slam = slam_pos
            self.anchor_rtk = rtk_pos
            return
        if self.alignment_ready:
            return
        slam_delta = slam_pos - self.anchor_slam
        rtk_delta = rtk_pos - self.anchor_rtk
        if np.linalg.norm(slam_delta[:2]) < self.min_alignment_baseline_m:
            self.last_reject_reason = "waiting_for_alignment_baseline"
            return
        if np.linalg.norm(rtk_delta[:2]) < self.min_alignment_baseline_m:
            self.last_reject_reason = "waiting_for_rtk_baseline"
            return
        slam_yaw = math.atan2(float(slam_delta[1]), float(slam_delta[0]))
        rtk_yaw = math.atan2(float(rtk_delta[1]), float(rtk_delta[0]))
        self.yaw_offset = self.wrap_angle(rtk_yaw - slam_yaw)
        self.alignment_ready = True
        self.get_logger().info(f"RTK/SLAM yaw alignment ready: {math.degrees(self.yaw_offset):.2f} deg")

    def rtk_quality_ok(self, msg: Odometry) -> bool:
        mono = time.monotonic()
        clock_now = stamp_to_sec(self.get_clock().now().to_msg())

        def age(arrival, stamp):
            if self.age_from_arrival:
                return None if arrival is None else mono - arrival
            return clock_now - stamp_to_sec(stamp)

        if self.require_fix_status:
            if self.latest_fix is None:
                self.last_reject_reason = "waiting_for_fix"
                return False
            fix_age = age(self.last_fix_arrival, self.latest_fix.header.stamp)
            if fix_age is not None and fix_age > self.max_fix_age_s:
                self.last_reject_reason = f"fix_stale_{fix_age:.2f}s"
                return False
            if int(self.latest_fix.status.status) < self.min_navsat_status:
                self.last_reject_reason = f"fix_status_{int(self.latest_fix.status.status)}"
                return False
        if not self.receiver_status_ok():
            return False
        rtk_age = age(self.last_rtk_arrival, msg.header.stamp)
        slam_age = age(self.last_slam_arrival, self.latest_slam.header.stamp)
        if rtk_age is not None and rtk_age > self.max_rtk_age_s:
            self.last_reject_reason = f"rtk_stale_{rtk_age:.2f}s"
            return False
        if slam_age is not None and slam_age > self.max_slam_age_s:
            self.last_reject_reason = f"slam_stale_{slam_age:.2f}s"
            return False
        cov = msg.pose.covariance
        std_xy = math.sqrt(max(float(cov[0]), float(cov[7]), 0.0))
        if std_xy > self.max_rtk_position_std_m:
            self.last_reject_reason = f"rtk_cov_{std_xy:.2f}m"
            return False
        return True

    def receiver_status_ok(self) -> bool:
        if not self.use_receiver_status_gate:
            return True
        if self.latest_receiver_status is None or self.last_receiver_status_time is None:
            if self.require_receiver_status:
                self.last_reject_reason = "waiting_for_receiver_status"
                return False
            return True
        age = time.monotonic() - self.last_receiver_status_time
        if age > self.max_receiver_status_age_s:
            if self.require_receiver_status:
                self.last_reject_reason = f"receiver_status_stale_{age:.2f}s"
                return False
            return True
        stage = str(self.latest_receiver_status.get("receiver_stage") or "UNKNOWN").upper()
        if self.allowed_receiver_stages and stage not in self.allowed_receiver_stages:
            self.last_reject_reason = f"receiver_stage_{stage}"
            return False
        return True

    def make_fused_odom(self, slam_msg: Odometry) -> Odometry:
        fused = copy.deepcopy(slam_msg)
        fused.header.frame_id = self.fused_frame_id
        if not self.offset_ready:
            return fused
        slam_pos = odom_position(slam_msg)
        fused_pos = yaw_matrix(self.yaw_offset) @ slam_pos + self.offset
        fused.pose.pose.position.x = float(fused_pos[0])
        fused.pose.pose.position.y = float(fused_pos[1])
        fused.pose.pose.position.z = float(fused_pos[2])
        if self.rotate_slam_orientation and self.alignment_ready:
            rotate_quat_z(fused.pose.pose.orientation, self.yaw_offset)
        return fused

    def publish_status(self):
        payload = {
            "offset_ready": self.offset_ready,
            "alignment_ready": self.alignment_ready,
            "yaw_offset_deg": math.degrees(self.yaw_offset),
            "offset": [float(v) for v in self.offset],
            "last_reject_reason": self.last_reject_reason,
            "rtk_topic": self.rtk_topic,
            "slam_topic": self.slam_topic,
            "fused_topic": self.fused_topic,
            "fix_status": None if self.latest_fix is None else int(self.latest_fix.status.status),
            "min_navsat_status": self.min_navsat_status,
            "receiver_status_topic": self.receiver_status_topic,
            "receiver_stage": None if self.latest_receiver_status is None else self.latest_receiver_status.get("receiver_stage"),
            "receiver_position_type": None
            if self.latest_receiver_status is None
            else self.latest_receiver_status.get("receiver_position_type"),
            "gga_quality": None if self.latest_receiver_status is None else self.latest_receiver_status.get("gga_quality"),
            "gga_quality_name": None
            if self.latest_receiver_status is None
            else self.latest_receiver_status.get("gga_quality_name"),
            "allowed_receiver_stages": sorted(self.allowed_receiver_stages),
            "use_receiver_status_gate": self.use_receiver_status_gate,
            "receiver_status_age_s": None
            if self.last_receiver_status_time is None
            else time.monotonic() - self.last_receiver_status_time,
            "rtk_used_age_s": None if self.last_rtk_used_time is None else time.monotonic() - self.last_rtk_used_time,
        }
        self.status_pub.publish(String(data=json.dumps(payload)))

    @staticmethod
    def wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)
    node = RtkFusionNode()
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
