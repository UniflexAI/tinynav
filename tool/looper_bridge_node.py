import argparse
import copy
import os
import sys
import time
from collections import deque

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage

from tinynav.core.math_utils import np2msg, pose_msg2np


# Matches planning_node._make_T_front_cam_to_rear_cam: camera1 is butt-mounted,
# optical Ry(pi) relative to the front camera (x right, y down, z forward).
def make_T_front_from_rear(baseline_m: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    T[:3, 3] = [0.0, 0.0, -float(baseline_m)]
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def is_finite_pose(T: np.ndarray) -> bool:
    return T.shape == (4, 4) and np.isfinite(T).all()


def rot_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)))


class StreamHealth:
    """Numerical VIO health from pose samples. Status topics are ignored."""

    def __init__(self, dropout_s: float, freeze_s: float = 0.8, freeze_m: float = 0.005):
        self.dropout_s = float(dropout_s)
        self.freeze_s = float(freeze_s)
        self.freeze_m = float(freeze_m)
        self.T: np.ndarray | None = None
        self.stamp_sec: float | None = None
        self.recv_mono: float | None = None
        self.reason = "no_data"
        self._xyz = deque(maxlen=256)
        self._jump = False
        self._samples = 0
        self._jump_streak = 0
        self._jump_bad_until_mono = 0.0
        self._poses = deque(maxlen=256)

    def update(self, T: np.ndarray, stamp_sec: float, now_mono: float) -> None:
        self.reason = "ok"
        if not is_finite_pose(T):
            self.reason = "nan"
            self.recv_mono = now_mono
            return

        self._samples += 1
        hard_jump = False
        rate_bad = False
        jump_reason = ""
        if self.T is not None and self.stamp_sec is not None and self._samples > 5:
            dt = stamp_sec - self.stamp_sec
            if 1e-4 < dt <= self.dropout_s * 2.0:
                dp = T[:3, 3] - self.T[:3, 3]
                dist = float(np.linalg.norm(dp))
                speed = dist / max(dt, 1e-6)
                dR = self.T[:3, :3].T @ T[:3, :3]
                angle = rot_angle(dR)
                angular_rate = angle / max(dt, 1e-6)
                hard_jump = dist > 0.3 or angle > np.deg2rad(25.0)
                rate_bad = dt >= 0.005 and (speed > 2.5 or angular_rate > 4.0)
                jump_reason = (
                    f"jump dist={dist:.3f}m angle={np.rad2deg(angle):.1f}deg "
                    f"dt={dt:.4f}s speed={speed:.2f}m/s "
                    f"angular_rate={angular_rate:.2f}rad/s"
                )
        if hard_jump:
            self._jump_streak = 0
            self._jump_bad_until_mono = max(self._jump_bad_until_mono, now_mono + 1.0)
        elif rate_bad:
            self._jump_streak += 1
            if self._jump_streak >= 3:
                self._jump_bad_until_mono = max(self._jump_bad_until_mono, now_mono + 1.0)
        else:
            self._jump_streak = 0
        self._jump = now_mono < self._jump_bad_until_mono
        if hard_jump or rate_bad or self._jump:
            remaining = max(0.0, self._jump_bad_until_mono - now_mono)
            self.reason = f"{jump_reason or 'jump latched'} hold={remaining:.2f}s"

        self.T = T.copy()
        self.stamp_sec = stamp_sec
        self.recv_mono = now_mono
        self._xyz.append((now_mono, T[:3, 3].copy()))
        self._poses.append((now_mono, T.copy()))
        cutoff = now_mono - self.freeze_s
        while self._xyz and self._xyz[0][0] < cutoff:
            self._xyz.popleft()

    def has_data(self) -> bool:
        return self.recv_mono is not None

    def dropout(self, now_mono: float) -> bool:
        if self.recv_mono is None:
            return False
        return (now_mono - self.recv_mono) > self.dropout_s

    def frozen(self, now_mono: float) -> bool:
        if self.dropout(now_mono) or len(self._xyz) < 8:
            return False
        pts = np.stack([xyz for _, xyz in self._xyz], axis=0)
        return float(np.linalg.norm(pts[-1] - pts[0])) < self.freeze_m

    def moving(self, now_mono: float, min_m: float = 0.05) -> bool:
        if self.dropout(now_mono) or len(self._xyz) < 8:
            return False
        pts = np.stack([xyz for _, xyz in self._xyz], axis=0)
        return float(np.linalg.norm(pts[-1] - pts[0])) >= min_m

    def alive(self, now_mono: float) -> bool:
        return (
            self.T is not None
            and not self.dropout(now_mono)
            and not self._jump
            and self.reason != "nan"
        )

    def good(self, now_mono: float) -> bool:
        return self.alive(now_mono) and not self.frozen(now_mono)

    def motion_delta(self, now_mono: float, window_s: float = 0.5) -> np.ndarray | None:
        if self.T is None or len(self._poses) < 2:
            return None
        target = now_mono - window_s
        anchor = None
        for sample_mono, sample_T in self._poses:
            anchor = sample_T
            if sample_mono >= target:
                break
        if anchor is None:
            return None
        return inv_T(anchor) @ self.T


class FrontVisualHealth:
    """Detect front-camera cover from direct image/depth taps."""

    def __init__(self, enabled: bool, bad_after_s: float, recover_after_s: float, sample_timeout_s: float = 0.7):
        self.enabled = bool(enabled)
        self.bad_after_s = float(bad_after_s)
        self.recover_after_s = float(recover_after_s)
        self.sample_timeout_s = float(sample_timeout_s)
        self._image_bad = False
        self._image_reason = "image:no_data"
        self._image_mono: float | None = None
        self._depth_bad = False
        self._depth_reason = "depth:no_data"
        self._depth_mono: float | None = None
        self._bad_since: float | None = None
        self._good_since: float | None = None
        self._covered = False
        self.reason = "disabled" if not self.enabled else "no_data"

    def update_image(self, bad: bool, reason: str, now_mono: float) -> None:
        self._image_bad = bool(bad)
        self._image_reason = reason
        self._image_mono = now_mono
        self._update(now_mono)

    def update_depth(self, bad: bool, reason: str, now_mono: float) -> None:
        self._depth_bad = bool(bad)
        self._depth_reason = reason
        self._depth_mono = now_mono
        self._update(now_mono)

    def covered(self, now_mono: float) -> bool:
        self._update(now_mono)
        return self._covered

    def _recent(self, stamp: float | None, now_mono: float) -> bool:
        return stamp is not None and (now_mono - stamp) <= self.sample_timeout_s

    def _update(self, now_mono: float) -> None:
        if not self.enabled:
            self._covered = False
            self.reason = "disabled"
            return

        image_recent = self._recent(self._image_mono, now_mono)
        depth_recent = self._recent(self._depth_mono, now_mono)
        if not image_recent and not depth_recent:
            self.reason = "no_recent_visual_sample"
            self._bad_since = None
            self._good_since = None
            return

        # Image handles dark/washed/low-texture covers; depth handles a hand or
        # object held very close to the camera even when the infra image has texture.
        bad = (image_recent and self._image_bad) or (depth_recent and self._depth_bad)
        if image_recent and depth_recent:
            signal_reason = f"{self._image_reason};{self._depth_reason}"
        else:
            signal_reason = self._image_reason if image_recent else self._depth_reason

        if bad:
            self._good_since = None
            if self._bad_since is None:
                self._bad_since = now_mono
            if (now_mono - self._bad_since) >= self.bad_after_s:
                self._covered = True
            self.reason = f"{signal_reason} bad_for={now_mono - self._bad_since:.2f}s"
            return

        self._bad_since = None
        if self._good_since is None:
            self._good_since = now_mono
        if not self._covered or (now_mono - self._good_since) >= self.recover_after_s:
            self._covered = False
        self.reason = f"{signal_reason} good_for={now_mono - self._good_since:.2f}s"


class LooperBridgeNode(Node):
    def __init__(self, args):
        super().__init__("looper_bridge_node")
        self.args = args
        self.bridge = CvBridge()

        self.cached_camera_info = None
        self.last_keyframe_pose = None
        self.last_keyframe_time = None
        self.last_pose = None
        self.last_pose_time = None
        self._missing_input_counter = 0
        self._started_at_mono = time.monotonic()
        self._last_sync_at_mono: float | None = None
        self._exit_code = 0
        # Disabled: covering the front camera stops TimeSynchronizer, and this
        # watchdog previously killed the whole bridge (including rear VIO fallback).
        self._sync_watchdog_s = 0.0
        self._sync_watchdog_grace_s = float(args.sync_watchdog_grace_s)
        self.get_logger().info("Sync watchdog disabled")

        # The Looper stamps vio_image/depth with its own free-running boot-relative
        # clock, not wall time (confirmed: neither an NTP-style device time sync nor a
        # full device reboot changes that -- it's not a transient offset, it's what the
        # firmware stamps with). planning_node's lidar_sync_callback needs /slam/depth
        # and /slam/odometry_visual to line up with /lidar/points' real wall-clock stamps
        # (from the Hesai driver on a separate host, the A2 board), so estimate the
        # device-clock -> wall-clock offset once at startup (median of the first samples,
        # to average out per-message reception jitter) and add it to every device stamp,
        # rather than re-stamping at reception time -- this keeps the device's own
        # frame-to-frame spacing (precise capture timing) and only shifts the epoch.
        self._device_clock_offset_ns = None
        self._device_clock_offset_samples = []
        self._device_clock_offset_warmup_count = 20

        image_reliability = (
            ReliabilityPolicy.BEST_EFFORT
            if args.image_reliability == "best_effort"
            else ReliabilityPolicy.RELIABLE
        )
        # Looper (insight_full) currently publishes images as RELIABLE; a BEST_EFFORT
        # subscriber will not match until the device publisher QoS is changed too.
        self.image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, int(args.image_qos_depth)),
            reliability=image_reliability,
        )
        self.tf_static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._rear_fallback_enabled = bool(args.rear_fallback)
        self._rear_baseline_m = float(args.rear_baseline_m)
        self._T_front_from_rear = make_T_front_from_rear(self._rear_baseline_m)
        self._assert_rear_extrinsic()

        self._front_100 = StreamHealth(dropout_s=0.40)
        self._rear_100 = StreamHealth(dropout_s=0.40)
        self._front_img = StreamHealth(dropout_s=0.40)
        self._rear_img = StreamHealth(dropout_s=0.40)
        self._front_visual = FrontVisualHealth(
            enabled=bool(args.front_cover_fallback),
            bad_after_s=float(args.front_cover_bad_s),
            recover_after_s=float(args.front_cover_recover_s),
        )
        self._vio_source = "front"
        self._failed_since_mono: float | None = None
        self._T_front_lock: np.ndarray | None = None
        self._T_rear_lock: np.ndarray | None = None
        self._rear_alignment_history = deque(maxlen=400)
        self._rear_alignment_guard_s = 0.8
        self._front_recovery_good_since: float | None = None
        self._front_recovery_confirm_s = 1.0
        self._recover_T_pub_lock: np.ndarray | None = None
        self._recover_T_front_100_raw_lock: np.ndarray | None = None
        self._recover_T_front_img_raw_lock: np.ndarray | None = None
        self._last_pub_T: np.ndarray | None = None
        self._last_odom_stamp = None
        self._last_odom_mono: float | None = None
        self._latest_depth: Image | None = None
        self._latest_image: Image | None = None

        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera/camera/infra1/camera_info", self.camera_info_callback, 10
        )
        self.tf_static_sub = self.create_subscription(TFMessage, "/tf_static", self.tf_callback, self.tf_static_qos)

        self.vio_100hz_sub = self.create_subscription(
            PoseStamped, "/camera/camera/vio_100hz", self.vio_100hz_callback, 50
        )
        if self._rear_fallback_enabled:
            self.rear_vio_100hz_sub = self.create_subscription(
                PoseStamped, "/camera1/camera/vio_100hz", self.rear_vio_100hz_callback, 50
            )
            self.rear_vio_image_sub = self.create_subscription(
                PoseStamped, "/camera1/camera/vio_image", self.rear_vio_image_callback, 20
            )

        sync_queue = max(2, int(args.sync_queue_size))
        self._replay_sync_enabled = bool(args.replay_sync)
        self._replay_sync_queue_size = sync_queue
        self._replay_depth = {}
        self._replay_pose = {}
        self._replay_image = {}
        if self._replay_sync_enabled:
            # Bag playback may deliver topics in bursts that defeat the online
            # message_filters queues. Match recorded messages by their original
            # header timestamp instead; live operation keeps the existing path.
            self.depth_latest_sub = self.create_subscription(
                Image,
                "/camera/camera/depth/image_rect_raw",
                self._on_replay_depth,
                self.image_qos,
            )
            self.pose_sub = self.create_subscription(
                PoseStamped,
                "/camera/camera/vio_image",
                self._on_replay_pose,
                10,
            )
            self.image_latest_sub = self.create_subscription(
                Image,
                "/camera/camera/infra1/image_rect_raw",
                self._on_replay_image,
                self.image_qos,
            )
            self.sync = None
        else:
            self.depth_sub = message_filters.Subscriber(
                self, Image, "/camera/camera/depth/image_rect_raw", qos_profile=self.image_qos
            )
            self.pose_sub = message_filters.Subscriber(self, PoseStamped, "/camera/camera/vio_image")
            self.image_sub = message_filters.Subscriber(
                self, Image, "/camera/camera/infra1/image_rect_raw", qos_profile=self.image_qos
            )
            if args.sync_slop > 0.0:
                self.sync = message_filters.ApproximateTimeSynchronizer(
                    [self.depth_sub, self.pose_sub, self.image_sub],
                    queue_size=sync_queue,
                    slop=float(args.sync_slop),
                )
            else:
                self.sync = message_filters.TimeSynchronizer(
                    [self.depth_sub, self.pose_sub, self.image_sub], queue_size=sync_queue
                )
            self.sync.registerCallback(self.sync_callback)

            # Direct depth/image taps so rear fallback can keep /slam/depth flowing
            # after front vio_image (and therefore TimeSynchronizer) goes silent.
            self.depth_latest_sub = self.create_subscription(
                Image, "/camera/camera/depth/image_rect_raw", self._on_front_depth, self.image_qos
            )
            self.image_latest_sub = self.create_subscription(
                Image, "/camera/camera/infra1/image_rect_raw", self._on_front_image, self.image_qos
            )

        self.odom_pub = self.create_publisher(Odometry, "/slam/odometry", 10)
        self.odom_visual_pub = self.create_publisher(
            Odometry, "/slam/odometry_visual", 10
        )
        self.depth_pub = self.create_publisher(Image, "/slam/depth", 10)
        self.disparity_pub_vis = self.create_publisher(Image, "/slam/disparity_vis", 10)
        self.slam_camera_info_pub = self.create_publisher(CameraInfo, "/slam/camera_info", 10)
        self.camera_info_alias_pub = self.create_publisher(
            CameraInfo, "/camera/camera/infra2/camera_info", 10
        )
        self.keyframe_pose_visual_pub = self.create_publisher(
            Odometry, "/slam/keyframe_odom", 10
        )
        self.keyframe_image_pub = self.create_publisher(Image, "/slam/keyframe_image", 10)
        self.keyframe_depth_pub = self.create_publisher(Image, "/slam/keyframe_depth", 10)
        self.vio_source_pub = self.create_publisher(String, "/slam/vio_source", self._latched_qos)
        self.nav_paused_pub = self.create_publisher(Bool, "/nav/paused", self._latched_qos)
        self._health_timer = self.create_timer(0.05, self._health_tick)

        self.get_logger().info(
            "Bridging /camera/camera/vio_image + /camera/camera/depth/image_rect_raw + /camera/camera/infra1/image_rect_raw into TinyNav /slam topics."
        )
        self.get_logger().info(
            "Bridging /camera/camera/vio_100hz into /slam/odometry."
        )
        if self._rear_fallback_enabled:
            self.get_logger().info(
                "Rear VIO fallback enabled: /camera1/camera/vio_100hz relative motion "
                f"conjugated through 180deg optical extrinsic, baseline={self._rear_baseline_m:.3f}m. "
                "Downstream still consumes /slam/odometry in the front world."
            )
        sync_mode = "replay timestamp" if self._replay_sync_enabled else (
            f"approximate slop={args.sync_slop:.3f}s"
            if args.sync_slop > 0.0
            else "exact"
        )
        self.get_logger().info(
            f"Image QoS: reliability={args.image_reliability}, depth={self.image_qos.depth}; "
            f"sync={sync_mode}, queue={sync_queue}"
        )
        self._publish_vio_source(self._vio_source)
        self._publish_nav_paused(self._vio_source)

    def _assert_rear_extrinsic(self) -> None:
        # Robot moves +1m in body forward: rear optical sees -Z, front must see +Z.
        delta_r = np.eye(4, dtype=np.float64)
        delta_r[:3, 3] = [0.0, 0.0, -1.0]
        delta_f = self._T_front_from_rear @ delta_r @ inv_T(self._T_front_from_rear)
        t = delta_f[:3, 3]
        if abs(t[0]) > 1e-6 or abs(t[1]) > 1e-6 or abs(t[2] - 1.0) > 1e-6:
            raise RuntimeError(f"rear 180deg extrinsic self-check failed, got t={t}")

    def _publish_vio_source(self, source: str) -> None:
        msg = String()
        msg.data = source
        self.vio_source_pub.publish(msg)

    def _publish_nav_paused(self, source: str) -> None:
        self.nav_paused_pub.publish(Bool(data=(source != "front")))

    def _set_source(self, source: str, now_mono: float) -> None:
        prev = self._vio_source
        if source == "failed":
            if self._failed_since_mono is None:
                self._failed_since_mono = now_mono
            if (now_mono - self._failed_since_mono) < 1.0:
                return
        else:
            self._failed_since_mono = None
        if source == prev:
            return

        if source == "rear":
            self._activate_rear_alignment(now_mono)
            self._recover_T_pub_lock = None
            self._recover_T_front_100_raw_lock = None
            self._recover_T_front_img_raw_lock = None
            self._front_recovery_good_since = None
        elif prev in ("rear", "failed") and source == "front":
            if self._last_pub_T is not None and self._front_100.T is not None:
                self._recover_T_pub_lock = self._last_pub_T.copy()
                self._recover_T_front_100_raw_lock = self._front_100.T.copy()
                self._recover_T_front_img_raw_lock = (
                    None if self._front_img.T is None else self._front_img.T.copy()
                )
            self._T_front_lock = None
            self._T_rear_lock = None
            self._front_recovery_good_since = None

        self._vio_source = source
        self._publish_vio_source(source)
        self._publish_nav_paused(source)
        self.get_logger().warn(
            f"VIO source {prev} -> {source} "
            f"front({self._front_100.reason}) rear({self._rear_100.reason}) "
            f"visual({self._front_visual.reason})"
        )

    def _record_rear_alignment(self, now_mono: float) -> None:
        if (
            self._vio_source != "front"
            or self._front_100.T is None
            or self._rear_100.T is None
            or not self._front_100.alive(now_mono)
            or not self._rear_100.alive(now_mono)
        ):
            return
        front_common = self._compose_from_front_raw(self._front_100.T, "100hz")
        self._rear_alignment_history.append(
            (now_mono, front_common.copy(), self._rear_100.T.copy())
        )

    def _activate_rear_alignment(self, now_mono: float) -> None:
        cutoff = now_mono - self._rear_alignment_guard_s
        anchor = None
        for candidate in reversed(self._rear_alignment_history):
            if candidate[0] <= cutoff:
                anchor = candidate
                break
        if anchor is None and self._rear_alignment_history:
            anchor = self._rear_alignment_history[0]
        if anchor is not None:
            anchor_mono, front_common, rear_raw = anchor
            self._T_front_lock = front_common.copy()
            self._T_rear_lock = rear_raw.copy()
            self.get_logger().warn(
                "Rear takeover using trusted dual-VIO alignment "
                f"from {now_mono - anchor_mono:.2f}s before switch"
            )
            return
        self._T_front_lock = None if self._last_pub_T is None else self._last_pub_T.copy()
        self._T_rear_lock = None if self._rear_100.T is None else self._rear_100.T.copy()
        if self._T_rear_lock is None and self._rear_img.T is not None:
            self._T_rear_lock = self._rear_img.T.copy()
        self.get_logger().warn("Rear takeover has no dual-VIO alignment history; using current pose")

    def _front_rear_motion_agree(self, now_mono: float) -> bool:
        delta_front = self._front_100.motion_delta(now_mono)
        delta_rear = self._rear_100.motion_delta(now_mono)
        if delta_front is None or delta_rear is None:
            return False
        delta_rear_front = (
            self._T_front_from_rear @ delta_rear @ inv_T(self._T_front_from_rear)
        )
        translation_error = float(
            np.linalg.norm(delta_front[:3, 3] - delta_rear_front[:3, 3])
        )
        rotation_error = rot_angle(
            delta_front[:3, :3].T @ delta_rear_front[:3, :3]
        )
        return translation_error <= 0.15 and rotation_error <= np.deg2rad(10.0)

    def _evaluate_source(self, now_mono: float) -> str:
        if not self._rear_fallback_enabled:
            if not self._front_100.has_data() and not self._front_img.has_data():
                return self._vio_source
            return "front" if self._front_100.good(now_mono) or self._front_img.good(now_mono) else "failed"

        if not any(s.has_data() for s in (self._front_100, self._rear_100, self._front_img, self._rear_img)):
            return self._vio_source

        front_100_unavailable = (
            not self._front_100.has_data() or self._front_100.dropout(now_mono)
        )
        rear_100_unavailable = (
            not self._rear_100.has_data() or self._rear_100.dropout(now_mono)
        )
        front_good = self._front_100.good(now_mono) or (
            front_100_unavailable and self._front_img.good(now_mono)
        )
        front_alive = self._front_100.alive(now_mono) or (
            front_100_unavailable and self._front_img.alive(now_mono)
        )
        rear_alive = self._rear_100.alive(now_mono) or (
            rear_100_unavailable and self._rear_img.alive(now_mono)
        )
        rear_moving = self._rear_100.moving(now_mono) or self._rear_img.moving(now_mono)

        # While rear owns odometry, keep the robot paused until front has been
        # numerically healthy and agrees with rear relative motion for a full
        # confirmation interval. Image/depth appearance never changes VIO source.
        if self._vio_source == "rear":
            front_recovered = self._front_100.alive(now_mono) and self._front_rear_motion_agree(now_mono)
            if front_recovered:
                if self._front_recovery_good_since is None:
                    self._front_recovery_good_since = now_mono
                recovered_for = now_mono - self._front_recovery_good_since
                desired = "front" if recovered_for >= self._front_recovery_confirm_s else "rear"
            else:
                self._front_recovery_good_since = None
                desired = "rear" if rear_alive else "failed"
            self._set_source(desired, now_mono)
            return self._vio_source

        # Stay on front while it is still producing poses. Otherwise switch to
        # rear when front is dead/jumping, or frozen while rear is clearly moving.
        # A standing robot must not count as fallback or failed.
        if front_good:
            desired = "front"
        elif front_alive:
            desired = "rear" if (rear_alive and rear_moving) else "front"
        elif rear_alive:
            desired = "rear"
        else:
            desired = "failed"
        self._set_source(desired, now_mono)
        return self._vio_source

    def _compose_from_rear(self) -> np.ndarray | None:
        T_rear = self._rear_100.T if self._rear_100.T is not None else self._rear_img.T
        if T_rear is None or self._T_front_lock is None:
            return self._last_pub_T
        if self._T_rear_lock is None:
            self._T_rear_lock = T_rear.copy()
        delta_r = inv_T(self._T_rear_lock) @ T_rear
        delta_f = self._T_front_from_rear @ delta_r @ inv_T(self._T_front_from_rear)
        return self._T_front_lock @ delta_f

    def _compose_from_front_raw(self, T_front_raw: np.ndarray, front_stream: str) -> np.ndarray:
        raw_lock = (
            self._recover_T_front_img_raw_lock
            if front_stream == "image"
            else self._recover_T_front_100_raw_lock
        )
        if self._recover_T_pub_lock is None or raw_lock is None:
            return T_front_raw
        return self._recover_T_pub_lock @ inv_T(raw_lock) @ T_front_raw

    def _resolved_T(
        self,
        T_front_raw: np.ndarray | None,
        now_mono: float,
        front_stream: str = "100hz",
    ) -> np.ndarray | None:
        source = self._evaluate_source(now_mono)
        if source == "front":
            if T_front_raw is None:
                T_front_raw = self._front_100.T
            if T_front_raw is None:
                return self._last_pub_T
            return self._compose_from_front_raw(T_front_raw, front_stream)
        if source == "rear":
            return self._compose_from_rear()
        return None

    def _next_fallback_stamp(self, now_mono: float):
        if self._last_odom_stamp is None or self._last_odom_mono is None:
            return self.get_clock().now().to_msg()
        dt_ns = int(max(0.001, min(0.05, now_mono - self._last_odom_mono)) * 1e9)
        last_ns = int(self._last_odom_stamp.sec) * 1_000_000_000 + int(self._last_odom_stamp.nanosec)
        return Time(nanoseconds=last_ns + dt_ns).to_msg()

    def _publish_100hz(self, T: np.ndarray, stamp) -> None:
        odom_msg = np2msg(T, stamp, "world", "camera")
        self.odom_pub.publish(odom_msg)
        self._last_pub_T = T.copy()
        self._last_odom_stamp = stamp
        self._last_odom_mono = time.monotonic()

    def vio_100hz_callback(self, pose_msg: PoseStamped):
        now_mono = time.monotonic()
        T_raw = pose_msg2np(pose_msg)
        self._front_100.update(T_raw, self.stamp_to_sec(pose_msg.header.stamp), now_mono)
        T_out = self._resolved_T(T_raw, now_mono)
        if T_out is None or self._vio_source != "front":
            return
        self._publish_100hz(T_out, pose_msg.header.stamp)
        self.get_logger().info(
            f"Bridged first /camera/camera/vio_100hz message at "
            f"{pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec:09d} to /slam/odometry.",
            once=True,
        )

    def rear_vio_100hz_callback(self, pose_msg: PoseStamped):
        now_mono = time.monotonic()
        T_raw = pose_msg2np(pose_msg)
        self._rear_100.update(T_raw, self.stamp_to_sec(pose_msg.header.stamp), now_mono)
        self._record_rear_alignment(now_mono)
        T_out = self._resolved_T(None, now_mono)
        if T_out is None or self._vio_source != "rear":
            return
        self._publish_100hz(T_out, self._next_fallback_stamp(now_mono))
        self.get_logger().info(
            "Rear VIO is publishing /slam/odometry (front-frame, 180deg conjugated).",
            once=True,
        )

    def rear_vio_image_callback(self, pose_msg: PoseStamped):
        now_mono = time.monotonic()
        T_raw = pose_msg2np(pose_msg)
        self._rear_img.update(T_raw, self.stamp_to_sec(pose_msg.header.stamp), now_mono)
        if self._rear_100.alive(now_mono):
            return
        T_out = self._resolved_T(None, now_mono)
        if T_out is None or self._vio_source != "rear":
            return
        self._publish_100hz(T_out, self._next_fallback_stamp(now_mono))

    def _health_tick(self):
        now_mono = time.monotonic()
        source = self._evaluate_source(now_mono)
        if int(now_mono * 2) != int((now_mono - 0.05) * 2):
            self._publish_vio_source(source)
        if source != "rear" or self._latest_depth is None:
            return
        if self._last_sync_at_mono is not None and (now_mono - self._last_sync_at_mono) < 0.20:
            return
        self._publish_fallback_visual(self._latest_depth, self._latest_image)

    def _sample_front_image_health(self, msg: Image, now_mono: float) -> None:
        if not self._front_visual.enabled:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            arr = np.asarray(img)
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            arr = arr[::4, ::4]
            if arr.size == 0:
                self._front_visual.update_image(True, "image:empty", now_mono)
                return
            if np.issubdtype(arr.dtype, np.integer) and np.iinfo(arr.dtype).max > 255:
                arr = arr.astype(np.float32) * (255.0 / float(np.iinfo(arr.dtype).max))
            else:
                arr = arr.astype(np.float32)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            bad = (
                mean <= float(self.args.front_cover_image_mean_low)
                or mean >= float(self.args.front_cover_image_mean_high)
                or std <= float(self.args.front_cover_image_std)
            )
            reason = f"image:mean={mean:.1f},std={std:.1f}"
            self._front_visual.update_image(bad, reason, now_mono)
        except Exception as exc:
            self._front_visual.update_image(True, f"image:error:{type(exc).__name__}", now_mono)

    def _sample_front_depth_health(self, msg: Image, now_mono: float) -> None:
        if not self._front_visual.enabled:
            return
        try:
            depth = self.decode_depth_meters(msg)
            depth = depth[::4, ::4]
            valid = np.isfinite(depth) & (depth >= 0.05) & (depth <= 8.0)
            valid_count = int(np.count_nonzero(valid))
            valid_ratio = float(valid_count) / max(1, int(valid.size))
            near = valid & (depth <= float(self.args.front_cover_depth_near_m))
            near_ratio = float(np.count_nonzero(near)) / max(1, valid_count)
            bad = (
                valid_ratio <= float(self.args.front_cover_depth_min_valid_ratio)
                or (
                    valid_count >= int(self.args.front_cover_depth_min_valid_pixels)
                    and near_ratio >= float(self.args.front_cover_depth_near_ratio)
                )
            )
            reason = f"depth:valid={valid_ratio:.3f},near={near_ratio:.3f}"
            self._front_visual.update_depth(bad, reason, now_mono)
        except Exception as exc:
            self._front_visual.update_depth(True, f"depth:error:{type(exc).__name__}", now_mono)

    def _on_front_depth(self, msg: Image):
        self._latest_depth = msg
        now_mono = time.monotonic()
        if self._evaluate_source(now_mono) != "rear":
            return
        if self._last_sync_at_mono is not None and (now_mono - self._last_sync_at_mono) < 0.20:
            return
        self._publish_fallback_visual(msg, self._latest_image)

    def _on_front_image(self, msg: Image):
        self._latest_image = msg

    @staticmethod
    def _stamp_ns(msg) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _trim_replay_cache(self, cache: dict) -> None:
        while len(cache) > self._replay_sync_queue_size:
            cache.pop(next(iter(cache)))

    def _try_replay_sync(self) -> None:
        common = self._replay_depth.keys() & self._replay_pose.keys() & self._replay_image.keys()
        for stamp_ns in sorted(common):
            self.get_logger().info("Replay synchronizer matched its first triplet.", once=True)
            depth_msg = self._replay_depth.pop(stamp_ns)
            pose_msg = self._replay_pose.pop(stamp_ns)
            image_msg = self._replay_image.pop(stamp_ns)
            self.sync_callback(depth_msg, pose_msg, image_msg)

    def _on_replay_depth(self, msg: Image) -> None:
        self.get_logger().info("Replay synchronizer received depth.", once=True)
        self._on_front_depth(msg)
        self._replay_depth[self._stamp_ns(msg)] = msg
        self._trim_replay_cache(self._replay_depth)
        self._try_replay_sync()

    def _on_replay_pose(self, msg: PoseStamped) -> None:
        self.get_logger().info("Replay synchronizer received vio_image.", once=True)
        self._replay_pose[self._stamp_ns(msg)] = msg
        self._trim_replay_cache(self._replay_pose)
        self._try_replay_sync()

    def _on_replay_image(self, msg: Image) -> None:
        self.get_logger().info("Replay synchronizer received infra1 image.", once=True)
        self._on_front_image(msg)
        self._replay_image[self._stamp_ns(msg)] = msg
        self._trim_replay_cache(self._replay_image)
        self._try_replay_sync()

    def _publish_fallback_visual(self, depth_msg: Image, image_msg: Image | None):
        if self.cached_camera_info is None:
            return
        T = self._compose_from_rear()
        if T is None:
            return
        stamp = self.correct_device_stamp(depth_msg.header.stamp)
        odom_msg = self.build_odom(T, stamp)
        self.odom_visual_pub.publish(odom_msg)
        depth_m = self.decode_depth_meters(depth_msg)
        depth_out = self.build_depth_msg(depth_m, stamp)
        self.depth_pub.publish(depth_out)

        camera_info_out = copy.deepcopy(self.cached_camera_info)
        camera_info_out.header.stamp = stamp
        camera_info_out.header.frame_id = "camera"
        self.slam_camera_info_pub.publish(camera_info_out)
        self.camera_info_alias_pub.publish(camera_info_out)

        image_out = None
        if image_msg is not None:
            image_out = copy.deepcopy(image_msg)
            image_out.header.stamp = stamp
            image_out.header.frame_id = "camera"

        if self.should_add_keyframe(T, stamp):
            self.keyframe_pose_visual_pub.publish(odom_msg)
            if image_out is not None:
                self.keyframe_image_pub.publish(image_out)
            self.keyframe_depth_pub.publish(depth_out)
            self.last_keyframe_pose = T.copy()
            self.last_keyframe_time = self.stamp_to_sec(stamp)

        self._last_sync_at_mono = time.monotonic()
        self._last_pub_T = T.copy()

    def camera_info_callback(self, msg: CameraInfo):
        self.cached_camera_info = msg
        self.get_logger().info(
            f"Received camera info from /camera/camera/infra1/camera_info with frame {msg.header.frame_id}.",
            once=True,
        )

    def tf_callback(self, msg: TFMessage):
        self.get_logger().info("Received TF_STATIC for Looper bridge.", once=True)

    def log_missing_inputs(self):
        self._missing_input_counter += 1
        if self._missing_input_counter % 30 != 1:
            return
        if self.cached_camera_info is None:
            self.get_logger().info("Waiting for Looper bridge inputs: /camera/camera/infra1/camera_info")

    @staticmethod
    def stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def correct_device_stamp(self, device_stamp):
        """Map a Looper device timestamp (free-running, boot-relative) onto this
        host's wall clock, using an offset calibrated from the first samples.

        The offset is (wall_clock_now - device_stamp_now), sampled once per callback
        during warm-up; each sample also carries this callback's own reception latency
        as noise, so the median of several samples is used instead of a single reading.
        """
        # RGB is decompressed inside build_map_node and keeps the recorded device
        # timestamp. Preserve that timestamp during bag replay so its four-way
        # synchronizer can join RGB with the bridged image, odom, and depth.
        if self._replay_sync_enabled:
            return copy.deepcopy(device_stamp)

        device_ns = device_stamp.sec * 1_000_000_000 + device_stamp.nanosec
        if self._device_clock_offset_ns is None:
            wall_ns = self.get_clock().now().nanoseconds
            self._device_clock_offset_samples.append(wall_ns - device_ns)
            offset_ns = int(np.median(self._device_clock_offset_samples))
            if len(self._device_clock_offset_samples) >= self._device_clock_offset_warmup_count:
                self._device_clock_offset_ns = offset_ns
                self.get_logger().info(
                    f"Looper device clock offset calibrated: {offset_ns / 1e9:.3f}s "
                    f"(median of {len(self._device_clock_offset_samples)} samples)"
                )
        else:
            offset_ns = self._device_clock_offset_ns
        return Time(nanoseconds=device_ns + offset_ns).to_msg()

    def should_add_keyframe(self, T_world_camera: np.ndarray, stamp) -> bool:
        if self.last_keyframe_pose is None or self.last_keyframe_time is None:
            return True
        current_time = self.stamp_to_sec(stamp)
        translation = np.linalg.norm(
            T_world_camera[:3, 3] - self.last_keyframe_pose[:3, 3]
        )
        relative_rotation = self.last_keyframe_pose[:3, :3].T @ T_world_camera[:3, :3]
        rotation_angle = np.arccos(
            np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        )
        return (
            translation >= self.args.keyframe_translation
            or rotation_angle >= np.deg2rad(self.args.keyframe_rotation_deg)
            or current_time - self.last_keyframe_time > 3.0
        )

    def build_odom(self, T_world_camera: np.ndarray, stamp) -> Odometry:
        velocity = None
        current_time = stamp.sec + stamp.nanosec * 1e-9
        if self.last_pose is not None and self.last_pose_time is not None:
            dt = current_time - self.last_pose_time
            if dt > 1e-3:
                velocity = (T_world_camera[:3, 3] - self.last_pose[:3, 3]) / dt

        odom_msg = np2msg(
            T_world_camera,
            stamp,
            "world",
            "camera",
            velocity=velocity,
        )
        self.last_pose = T_world_camera.copy()
        self.last_pose_time = current_time
        return odom_msg

    def decode_depth_meters(self, depth_msg: Image) -> np.ndarray:
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = np.asarray(depth)
        if depth_msg.encoding in ("mono16", "16UC1"):
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)
        return depth

    def build_depth_msg(self, depth_m: np.ndarray, stamp) -> Image:
        depth_out = self.bridge.cv2_to_imgmsg(depth_m, encoding="32FC1")
        depth_out.header.stamp = stamp
        depth_out.header.frame_id = "camera"
        return depth_out

    def build_disparity_vis(self, depth_m: np.ndarray, stamp) -> Image:
        depth = np.asarray(depth_m, dtype=np.float32)

        valid = np.isfinite(depth) & (depth > 1e-3)
        disparity_u8 = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(valid):
            inv_depth = np.zeros(depth.shape, dtype=np.float32)
            inv_depth[valid] = 1.0 / depth[valid]
            disp_min = float(np.min(inv_depth[valid]))
            disp_max = float(np.max(inv_depth[valid]))
            if disp_max > disp_min:
                disparity_u8[valid] = np.clip(
                    255.0 * (inv_depth[valid] - disp_min) / (disp_max - disp_min),
                    0.0,
                    255.0,
                ).astype(np.uint8)
            else:
                disparity_u8[valid] = 255

        disp_color = cv2.applyColorMap(disparity_u8, cv2.COLORMAP_PLASMA)
        disp_color[~valid] = 0
        disp_color_msg = self.bridge.cv2_to_imgmsg(disp_color, encoding="bgr8")
        disp_color_msg.header.stamp = stamp
        disp_color_msg.header.frame_id = "camera"
        return disp_color_msg

    def sync_callback(self, depth_msg: Image, pose_msg: PoseStamped, image_msg: Image):
        if self.cached_camera_info is None:
            self.log_missing_inputs()
            return

        now_mono = time.monotonic()
        T_raw = pose_msg2np(pose_msg)
        self._front_img.update(T_raw, self.stamp_to_sec(pose_msg.header.stamp), now_mono)
        T_world_camera = self._resolved_T(T_raw, now_mono, front_stream="image")
        if T_world_camera is None:
            return

        stamp = self.correct_device_stamp(pose_msg.header.stamp)

        odom_msg = self.build_odom(T_world_camera, stamp)
        self.odom_visual_pub.publish(odom_msg)
        depth_m = self.decode_depth_meters(depth_msg)
        depth_out = self.build_depth_msg(depth_m, stamp)
        disparity_vis_msg = None

        image_out = copy.deepcopy(image_msg)
        image_out.header.stamp = stamp
        image_out.header.frame_id = "camera"

        camera_info_out = copy.deepcopy(self.cached_camera_info)
        camera_info_out.header.stamp = stamp
        camera_info_out.header.frame_id = "camera"

        self.depth_pub.publish(depth_out)
        if disparity_vis_msg is not None:
            self.disparity_pub_vis.publish(disparity_vis_msg)
        self.slam_camera_info_pub.publish(camera_info_out)
        self.camera_info_alias_pub.publish(camera_info_out)

        if self.should_add_keyframe(T_world_camera, stamp):
            self.keyframe_pose_visual_pub.publish(odom_msg)
            self.keyframe_image_pub.publish(image_out)
            self.keyframe_depth_pub.publish(depth_out)
            self.last_keyframe_pose = T_world_camera.copy()
            self.last_keyframe_time = self.stamp_to_sec(stamp)

        self._last_sync_at_mono = time.monotonic()
        self._last_pub_T = T_world_camera.copy()

    def _sync_watchdog_tick(self):
        if self._sync_watchdog_s <= 0.0:
            return
        now = time.monotonic()
        if self._last_sync_at_mono is None:
            if now - self._started_at_mono <= self._sync_watchdog_grace_s:
                return
            self._request_exit(
                2,
                f"sync watchdog: no successful sync within startup grace "
                f"{self._sync_watchdog_grace_s:.1f}s",
            )
            return
        gap = now - self._last_sync_at_mono
        if gap <= self._sync_watchdog_s:
            return
        self._request_exit(
            1,
            f"sync watchdog: no sync_callback for {gap:.1f}s "
            f"(limit {self._sync_watchdog_s:.1f}s)",
        )

    def _request_exit(self, code: int, reason: str):
        if self._exit_code != 0:
            return
        self._exit_code = int(code)
        self.get_logger().error(f"{reason}; exiting for supervisor restart")
        rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframe-translation", type=float, default=0.03)
    parser.add_argument("--keyframe-rotation-deg", type=float, default=1.0)
    parser.add_argument(
        "--sync-watchdog-s",
        type=float,
        default=0.0,
        help="Exit if sync_callback is silent for this many seconds (0 disables).",
    )
    parser.add_argument(
        "--sync-watchdog-grace-s",
        type=float,
        default=45.0,
        help="Allow this long after startup before the first sync_callback.",
    )
    parser.add_argument(
        "--image-reliability",
        choices=("reliable", "best_effort"),
        default="reliable",
        help="Image/depth subscriber reliability (must match Looper publisher).",
    )
    parser.add_argument(
        "--image-qos-depth",
        type=int,
        default=5,
        help="Image/depth subscriber history depth.",
    )
    parser.add_argument(
        "--sync-queue-size",
        type=int,
        default=10,
        help="message_filters sync queue size.",
    )
    parser.add_argument(
        "--sync-slop",
        type=float,
        default=0.0,
        help="If >0, use ApproximateTimeSynchronizer with this slop (seconds).",
    )
    parser.add_argument(
        "--replay-sync",
        action="store_true",
        help="Synchronize recorded depth, pose, and image messages by exact header timestamp.",
    )
    parser.add_argument(
        "--rear-baseline-m",
        type=float,
        default=float(os.environ.get("TINYNAV_REAR_CAM_BASELINE_M", "0.6")),
        help="Front-to-rear camera baseline along front optical -Z (meters).",
    )
    parser.add_argument(
        "--rear-fallback",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("TINYNAV_REAR_VIO_FALLBACK", "1") not in ("0", "false", "False"),
        help="Use camera1 VIO relative motion as front-world fallback.",
    )
    parser.add_argument(
        "--front-cover-fallback",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("TINYNAV_FRONT_COVER_FALLBACK", "1") not in ("0", "false", "False"),
        help="Allow rear VIO fallback when the front camera image/depth looks covered.",
    )
    parser.add_argument(
        "--front-cover-bad-s",
        type=float,
        default=float(os.environ.get("TINYNAV_FRONT_COVER_BAD_S", "0.4")),
        help="Seconds of covered-looking front visual input before switching to rear.",
    )
    parser.add_argument(
        "--front-cover-recover-s",
        type=float,
        default=float(os.environ.get("TINYNAV_FRONT_COVER_RECOVER_S", "1.2")),
        help="Seconds of healthy front visual input before allowing switch back to front.",
    )
    parser.add_argument(
        "--front-cover-image-std",
        type=float,
        default=float(os.environ.get("TINYNAV_FRONT_COVER_IMAGE_STD", "3.0")),
        help="Front infra stddev threshold below which the image is treated as covered.",
    )
    parser.add_argument(
        "--front-cover-image-mean-low",
        type=float,
        default=float(os.environ.get("TINYNAV_FRONT_COVER_IMAGE_MEAN_LOW", "8.0")),
        help="Front infra mean threshold below which the image is treated as covered.",
    )
    parser.add_argument(
        "--front-cover-image-mean-high",
        type=float,
        default=float(os.environ.get("TINYNAV_FRONT_COVER_IMAGE_MEAN_HIGH", "248.0")),
        help="Front infra mean threshold above which the image is treated as covered.",
    )
    parser.add_argument(
        "--front-cover-depth-min-valid-ratio",
        type=float,
        default=float(os.environ.get("TINYNAV_FRONT_COVER_DEPTH_MIN_VALID_RATIO", "0.01")),
        help="Depth-only fallback cover threshold when front infra images are not recent.",
    )
    parser.add_argument(
        "--front-cover-depth-near-m",
        type=float,
        default=float(os.environ.get("TINYNAV_FRONT_COVER_DEPTH_NEAR_M", "0.35")),
        help="Depth points no farther than this are counted as close-cover candidates.",
    )
    parser.add_argument(
        "--front-cover-depth-near-ratio",
        type=float,
        default=float(os.environ.get("TINYNAV_FRONT_COVER_DEPTH_NEAR_RATIO", "0.35")),
        help="Covered if this fraction of valid front depth points is very close.",
    )
    parser.add_argument(
        "--front-cover-depth-min-valid-pixels",
        type=int,
        default=int(os.environ.get("TINYNAV_FRONT_COVER_DEPTH_MIN_VALID_PIXELS", "50")),
        help="Minimum sampled valid depth pixels before close-cover ratio is trusted.",
    )
    return parser.parse_args()


def main(args=None):
    rclpy.init(args=args)
    node = LooperBridgeNode(parse_args())
    exit_code = 0
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        exit_code = int(getattr(node, "_exit_code", 0) or 0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
