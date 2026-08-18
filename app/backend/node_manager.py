"""
BackendNode — extends Ros2NodeManager with extra subscriptions for pose and
mapping progress, plus a NodeRunner that spins it in a background thread.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import base64

import rclpy
import rclpy.time
import tf2_ros
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point32, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import CompressedImage, Image, PointCloud, PointCloud2
from std_msgs.msg import Bool, Float32, String

from tool.ros2_node_manager import Ros2NodeManager

_REALSENSE_SCRIPT = '/tinynav/scripts/run_realsense_sensor.sh'
_VENV_SITE = '/tinynav/.venv/lib/python3.10/site-packages'
_MAP_BUILD_DOMAIN_LOOPER = '231'  # isolated domain to avoid live looper topic collision during map build

# build_map_node.py emits "MAPPING_PERCENT:<float>" lines on stdout so the
# parent process can track progress without a separate bridge subprocess.
_MAPPING_PERCENT_PREFIX = 'MAPPING_PERCENT:'

_COLOR_TOPIC_REALSENSE = '/camera/camera/color/image_raw'
_COLOR_TOPIC_LOOPER = '/camera/camera/color/image_rect_raw/compressed'
_DEPTH_TOPIC_LOOPER = '/camera/camera/depth/image_rect_raw'

_IMAGE_TOPICS_REALSENSE = [
    _COLOR_TOPIC_REALSENSE,
    '/camera/camera/infra1/image_rect_raw',
    '/camera/camera/infra2/image_rect_raw',
    '/slam/depth',
]
# Preview topics are listed separately. Backend subscribes to at most one
# selected topic, and only while a /ws/preview client is connected.
_IMAGE_TOPICS_LOOPER = [
    _COLOR_TOPIC_LOOPER,
    '/camera/camera/infra1/image_rect_raw',
    '/camera/camera/infra2/image_rect_raw',
    _DEPTH_TOPIC_LOOPER,
]
_IMAGE_TOPICS_LOOPER_CAM1 = [
    '/camera1/camera/color/image_rect_raw/compressed',
    '/camera1/camera/infra1/image_rect_raw',
    '/camera1/camera/infra2/image_rect_raw',
    '/camera1/camera/depth/image_rect_raw',
]
_IMAGE_TOPICS_ALL = _IMAGE_TOPICS_REALSENSE  # fallback
_LOOPER_NODE_NAMES = {'/insight_full', '/insight_full1'}


def _paired_camera1_topic(topic: str) -> str | None:
    """Map a /camera/camera/... topic to its /camera1/camera/... twin."""
    prefix = '/camera/camera/'
    if topic.startswith(prefix):
        return '/camera1/camera/' + topic[len(prefix):]
    return None


def _to_bgr_preview(arr: np.ndarray) -> np.ndarray:
    if arr is None or arr.size == 0:
        return arr
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    return arr


def _stitch_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = _to_bgr_preview(left)
    right = _to_bgr_preview(right)
    h = min(left.shape[0], right.shape[0])
    if left.shape[0] != h:
        left = cv2.resize(
            left,
            (max(1, int(left.shape[1] * h / left.shape[0])), h),
            interpolation=cv2.INTER_AREA,
        )
    if right.shape[0] != h:
        right = cv2.resize(
            right,
            (max(1, int(right.shape[1] * h / right.shape[0])), h),
            interpolation=cv2.INTER_AREA,
        )
    return np.hstack([left, right])
_PREVIEW_MIN_INTERVAL = 0.2  # 5 fps
# Planning WS is 5 fps; processing height/voxels faster than that is wasted CPU.
_PLANNING_VIZ_MIN_INTERVAL = 0.2
_SENSOR_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
_PREVIEW_MAX_EDGE_PX = int(os.environ.get('TINYNAV_PREVIEW_MAX_EDGE_PX', '320'))
_PREVIEW_JPEG_QUALITY = int(os.environ.get('TINYNAV_PREVIEW_JPEG_QUALITY', '50'))
_MAP_HANDOFF_LOCALIZATION_TIMEOUT_S = float(
    os.environ.get('TINYNAV_MAP_HANDOFF_LOCALIZATION_TIMEOUT_S', '0')
)
_RTK_YAW_INIT_SPEED_MPS = float(os.environ.get('TINYNAV_RTK_YAW_INIT_SPEED_MPS', '0.3'))
_RTK_YAW_INIT_DURATION_S = float(os.environ.get('TINYNAV_RTK_YAW_INIT_DURATION_S', '5.0'))
_RTK_YAW_INIT_RATE_HZ = float(os.environ.get('TINYNAV_RTK_YAW_INIT_RATE_HZ', '10.0'))
_RTK_YAW_INIT_CLEARANCE_M = float(os.environ.get('TINYNAV_RTK_YAW_INIT_CLEARANCE_M', '1.8'))
_RTK_YAW_INIT_CLEARANCE_MAX_AGE_S = float(
    os.environ.get('TINYNAV_RTK_YAW_INIT_CLEARANCE_MAX_AGE_S', '1.0')
)
_VIO_STATUS_NORMAL = {'TRACKING', 'TRACKING_STATIC'}


def _resize_preview_frame(arr: np.ndarray, max_edge_px: int = _PREVIEW_MAX_EDGE_PX) -> np.ndarray:
    """Downscale preview frame so the longest side is <= max_edge_px."""
    if max_edge_px <= 0 or arr is None or arr.size == 0:
        return arr
    height, width = arr.shape[:2]
    longest = max(height, width)
    if longest <= max_edge_px:
        return arr
    scale = max_edge_px / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(arr, new_size, interpolation=cv2.INTER_AREA)


def _encode_preview_jpeg(arr: np.ndarray) -> bytes:
    arr = _resize_preview_frame(arr)
    ok, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, _PREVIEW_JPEG_QUALITY])
    if not ok:
        raise RuntimeError('failed to encode preview jpeg')
    return buf.tobytes()


class BackendNode(Ros2NodeManager):
    """Ros2NodeManager + subscriptions needed by the HTTP/WS layer."""

    def __init__(self, tinynav_db_path: str = '/tinynav/tinynav_db'):
        super().__init__(tinynav_db_path=tinynav_db_path)

        self._lock = threading.Lock()
        self.mapping_percent: float = 0.0
        self.current_pose: dict | None = None   # latest pose from SLAM or map

        # Callbacks invoked (in the rclpy spin thread) on new data.
        # Keep them cheap — just put data on a queue or set an event.
        self.pose_callbacks: list = []
        self.state_callbacks: list = []
        self.preview_callbacks: dict[str, list] = {}  # topic -> [callbacks]

        # Planning / localization state (read via get_planning_snapshot)
        self._odom_pose: dict | None = None
        self._odom_pose_received_at: float | None = None
        self._ekf_odom_pose: dict | None = None
        self._odom_pose_at_kf: dict | None = None  # odom pose snapshotted at last mapPose update
        self._map_pose: dict | None = None
        self._localized: bool = False
        self._esdf_bytes: bytes = b''
        self._obstacle_bytes: bytes = b''
        self._trajectory: list = []
        self._global_path: list = []
        self._final_global_path: list = []
        self._footprint: list = []   # 4 corner points [{x,y},...] in world frame
        self._voxel_points: list = []
        self._grid_info: dict | None = None
        self._nav_target_pose: dict | None = None

        # Insight VIO guard is only enabled for looper mode. When VIO loses
        # tracking, stop nav nodes and later resume the remaining POIs after
        # VIO recovers and relocalization succeeds.
        self._vio_status: str | None = None
        self._vio_status_sub = None
        self._vio_guard_stopped: bool = False
        self._vio_guard_recovering: bool = False
        self._vio_resume_poi_ids: list[int | str] = []
        self._active_nav_poi_ids: list[int | str] = []
        self._active_nav_pois: list[dict] = []

        # Debug recording (independent of main state machine)
        self._debug_record_proc: subprocess.Popen | None = None
        self._debug_record_path: str | None = None

        self.create_subscription(Float32, '/mapping/percent', self._on_mapping_percent, 10)
        self.create_subscription(Odometry, '/slam/odometry_visual', self._on_slam_odom, 10)
        self.create_subscription(Odometry, '/slam/odometry_fused', self._on_ekf_odom, 10)
        self.create_subscription(
            Odometry, '/mapping/current_pose_in_map', self._on_pose_in_map, 10
        )
        # Mark localized as soon as any relocalization succeeds (published unconditionally
        # by map_node, unlike current_pose_in_map which requires POIs to be set).
        self.create_subscription(
            Odometry, '/map/relocalization', self._on_relocalization, 10
        )
        self.create_subscription(
            Image, '/planning/height_map', self._on_height_map, _SENSOR_QOS
        )
        self.create_subscription(
            OccupancyGrid, '/planning/obstacle_mask', self._on_obstacle_mask, 1
        )
        self.create_subscription(Path, '/planning/trajectory_path', self._on_trajectory_path, 1)
        self.create_subscription(Path, '/mapping/global_plan', self._on_global_plan, 1)
        self.create_subscription(Path, '/mapping/final_global_plan', self._on_final_global_plan, 1)
        self.create_subscription(
            Odometry, '/control/target_pose', self._on_nav_target_pose, 1
        )
        self.create_subscription(
            PointCloud, '/planning/footprint', self._on_footprint, 1
        )
        self.create_subscription(
            PointCloud2, '/planning/occupied_voxels', self._on_occupied_voxels, _SENSOR_QOS
        )
        self._last_height_map_time = 0.0
        self._last_obstacle_mask_time = 0.0
        self._last_voxel_time = 0.0
        self._last_footprint_time = 0.0
        self._planning_viz_wanted_until = 0.0
        self._voxels_wanted_until = 0.0
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Publisher for POI nav target consumed by map_node via /mapping/cmd_pois
        self._cmd_pois_pub = self.create_publisher(String, '/mapping/cmd_pois', 10)

        # Manual local target for planning_node, used by the operate tab long-press tool.
        self._target_pose_pub = self.create_publisher(Odometry, '/control/target_pose', 10)

        # Latched publisher — new subscribers (cmd_vel_control) get current state immediately on connect
        _latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pause_pub = self.create_publisher(Bool, '/nav/paused', _latched_qos)
        self._nav_active_pub = self.create_publisher(Bool, '/nav/active', _latched_qos)
        self._current_map_pub = self.create_publisher(String, '/map/current_map', _latched_qos)
        self._planning_config_pub = self.create_publisher(String, '/planning/config', _latched_qos)
        self._localization_config_pub = self.create_publisher(String, '/localization/config', _latched_qos)
        self._nav_paused = False
        self._nav_active = False
        self._rtk_bridge_status: dict | None = None
        self._rtk_map_status: dict | None = None
        self._rtk_log_state: dict[str, tuple[tuple, float]] = {}
        self._front_clearance_m: float | None = None
        self._front_clearance_received_at: float | None = None
        self._rtk_yaw_init_thread: threading.Thread | None = None
        self._rtk_yaw_init_stop_event = threading.Event()
        self._rtk_yaw_init_active: bool = False
        # Decided once when nav nodes start (matches MapNode.rtk_mode, which is
        # also fixed at its own startup) -- not re-evaluated while nav is running.
        self._nav_rtk_mode: str = 'off'

        # Publisher for robot action commands (sit / stand)
        self._action_pub = self.create_publisher(String, '/service/command', 10)

        # Publisher for teleop velocity commands
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Sensor mode detection and image subscriptions
        self._sensor_mode: str = 'unknown'  # 'looper' | 'realsense' | 'unknown'
        self._image_subs: dict = {}
        self._last_frame: dict[str, bytes] = {}   # topic -> latest JPEG bytes
        self._last_frame_time: dict[str, float] = {}
        self._looper_bridge_proc: subprocess.Popen | None = None
        self._looper_bridge_want_running = False
        self._looper_bridge_last_restart_mono = 0.0
        self._looper_bridge_restart_cooldown_s = float(
            os.environ.get('TINYNAV_LOOPER_RESTART_COOLDOWN_S', '10')
        )
        self._realsense_proc: subprocess.Popen | None = None
        self._perception_proc: subprocess.Popen | None = None
        self._imu_propagation_proc: subprocess.Popen | None = None
        self._planning_proc: subprocess.Popen | None = None
        self._planning_occupancy_source: str = 'depth'
        self._odom_source: str = 'vio'
        self._planning_log_thread: threading.Thread | None = None
        self._planning_log_stop_event = threading.Event()
        self._unitree_proc: subprocess.Popen | None = None

        # Battery level from /battery topic (published by unitree_control)
        self._battery: float | None = None

        # Path to the last successfully verified bag (after stop + ros2 bag info check)
        self._last_verified_bag: str | None = None

        # Nav nodes (map_node + cmd_vel_control) managed independently of _stop_all
        self._nav_nodes_running: bool = False
        self._map_node_proc: subprocess.Popen | None = None
        self._cmd_vel_proc: subprocess.Popen | None = None

        # Auto-localization assist: sweep yaw while waiting for localization
        self._loc_assist_enabled: bool = False
        self._loc_assist_thread: threading.Thread | None = None
        self._loc_assist_stop_event = threading.Event()

        self._nav_progress: dict | None = None
        self.nav_progress_callbacks: list = []
        self._map_handoff_active: bool = False
        self._handled_map_handoffs: set[tuple[str, int | str]] = set()
        self._nav_done_seq: int = 0

        self._nav_active_pub.publish(Bool(data=False))

        self.create_subscription(Float32, '/battery', self._on_battery, 10)
        self.create_subscription(Float32, '/planning/front_clearance', self._on_front_clearance, 10)
        self.create_subscription(Bool, '/mapping/nav_done', self._on_nav_done, 10)
        self.create_subscription(String, '/mapping/nav_progress', self._on_nav_progress, 10)
        self.create_subscription(String, '/rtk/status', self._on_rtk_bridge_status, 10)
        self.create_subscription(String, '/rtk/init_status', self._on_rtk_map_status, 10)
        self.create_subscription(String, '/planning/config', self._on_planning_config, 10)
        self.create_subscription(String, '/localization/config', self._on_localization_config, 10)
        self._detect_and_init_sensor()
        self._start_unitree_if_configured()
        self.create_timer(2.0, self._supervise_looper_bridge)

    # ------------------------------------------------------------------ #
    # ROS callbacks                                                        #
    # ------------------------------------------------------------------ #

    def _on_battery(self, msg: Float32):
        with self._lock:
            self._battery = float(msg.data)

    def _on_front_clearance(self, msg: Float32):
        with self._lock:
            self._front_clearance_m = float(msg.data)
            self._front_clearance_received_at = time.monotonic()

    def _decode_json_status(self, raw: str, topic: str) -> dict | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Invalid {topic} JSON: {exc}')
            return None
        if not isinstance(data, dict):
            self.get_logger().warn(f'Invalid {topic} payload: {data!r}')
            return None
        return data

    def _log_rtk_status(self, key: str, signature: tuple, message: str):
        now = time.time()
        last_signature, last_time = self._rtk_log_state.get(key, (None, 0.0))
        if signature != last_signature or now - last_time >= 10.0:
            self.get_logger().info(message)
            self._rtk_log_state[key] = (signature, now)

    def _on_planning_config(self, msg: String):
        data = self._decode_json_status(msg.data, '/planning/config')
        if data is None:
            return
        source = data.get('occupancy_source')
        if source not in ('depth', 'lidar'):
            return
        with self._lock:
            self._planning_occupancy_source = source

    def _on_localization_config(self, msg: String):
        data = self._decode_json_status(msg.data, '/localization/config')
        if data is None:
            return
        source = data.get('odom_source')
        if source not in ('vio', 'ekf'):
            return
        with self._lock:
            self._odom_source = source

    def _on_rtk_bridge_status(self, msg: String):
        data = self._decode_json_status(msg.data, '/rtk/status')
        if data is None:
            return
        with self._lock:
            self._rtk_bridge_status = data
        signature = (
            data.get('accepted'),
            data.get('navsat_status'),
            data.get('navsat_status_name'),
            data.get('rtk_position_type'),
            data.get('rtk_calculate_status_name'),
        )
        self._log_rtk_status(
            'bridge',
            signature,
            'RTK bridge: '
            f"accepted={data.get('accepted')} "
            f"navsat_status={data.get('navsat_status')} "
            f"navsat_status_name={data.get('navsat_status_name')} "
            f"position_type={data.get('rtk_position_type')} "
            f"calculate_status={data.get('rtk_calculate_status_name')}",
        )

    def _on_rtk_map_status(self, msg: String):
        data = self._decode_json_status(msg.data, '/rtk/init_status')
        if data is None:
            return
        with self._lock:
            self._rtk_map_status = data
        signature = (
            data.get('state'),
            data.get('have_map'),
            data.get('fix_ok'),
            data.get('yaw_ready'),
            data.get('map'),
            data.get('navsat_status'),
        )
        self._log_rtk_status(
            'map',
            signature,
            'RTK map: '
            f"state={data.get('state')} "
            f"have_map={data.get('have_map')} "
            f"fix_ok={data.get('fix_ok')} "
            f"yaw_ready={data.get('yaw_ready')} "
            f"map={data.get('map')} "
            f"navsat_status={data.get('navsat_status')}",
        )
        self._maybe_start_rtk_yaw_init(data)

    def _latest_front_clearance(self) -> float | None:
        with self._lock:
            clearance = self._front_clearance_m
            received_at = self._front_clearance_received_at
        if clearance is None or received_at is None:
            return None
        if time.monotonic() - received_at > _RTK_YAW_INIT_CLEARANCE_MAX_AGE_S:
            return None
        return clearance

    def _front_is_clear_for_rtk_yaw_init(self) -> bool:
        clearance = self._latest_front_clearance()
        if clearance is None:
            self._log_rtk_status(
                'yaw_init_clearance',
                ('missing',),
                'RTK yaw-init waiting: no fresh /planning/front_clearance',
            )
            return False
        if clearance < _RTK_YAW_INIT_CLEARANCE_M:
            self._log_rtk_status(
                'yaw_init_clearance',
                ('blocked', round(clearance, 1)),
                f'RTK yaw-init blocked: front_clearance={clearance:.2f}m '
                f'< required={_RTK_YAW_INIT_CLEARANCE_M:.2f}m',
            )
            return False
        return True

    def _rtk_map_needs_yaw_init(self) -> bool:
        with self._lock:
            status = dict(self._rtk_map_status) if self._rtk_map_status else None
        if not status:
            return False
        return bool(status.get('need_forward_init')) or status.get('state') == 'NEED_YAW_INIT'

    def _maybe_start_rtk_yaw_init(self, status: dict):
        needs_yaw_init = bool(status.get('need_forward_init')) or status.get('state') == 'NEED_YAW_INIT'
        if not needs_yaw_init:
            self._rtk_yaw_init_stop_event.set()
            return
        if self._nav_rtk_mode != 'replace':
            return
        with self._lock:
            already_running = (
                self._rtk_yaw_init_thread is not None
                and self._rtk_yaw_init_thread.is_alive()
            )
            nav_ready = self._nav_nodes_running
        if already_running:
            return
        if not nav_ready:
            self._log_rtk_status(
                'yaw_init_nav_ready',
                ('not_running',),
                'RTK yaw-init waiting: nav nodes are not running',
            )
            return
        if not self._front_is_clear_for_rtk_yaw_init():
            return

        self._rtk_yaw_init_stop_event.clear()
        self._rtk_yaw_init_thread = threading.Thread(
            target=self._rtk_yaw_init_loop,
            daemon=True,
        )
        self._rtk_yaw_init_thread.start()

    def _stop_rtk_yaw_init(self):
        self._rtk_yaw_init_stop_event.set()
        if (
            self._rtk_yaw_init_thread is not None
            and self._rtk_yaw_init_thread is not threading.current_thread()
        ):
            self._rtk_yaw_init_thread.join(timeout=2.0)
            self._rtk_yaw_init_thread = None
        self._publish_cmd_vel(0.0, 0.0)

    def _rtk_yaw_init_should_stop(self, stop: threading.Event) -> bool:
        if stop.is_set():
            return True
        if not self._rtk_map_needs_yaw_init():
            self.get_logger().info('RTK yaw-init complete: RTK no longer requests heading init')
            return True
        with self._lock:
            nav_ready = self._nav_nodes_running
        if not nav_ready:
            self.get_logger().info('RTK yaw-init stopped: nav nodes are no longer running')
            return True
        return not self._front_is_clear_for_rtk_yaw_init()

    def _rtk_yaw_init_loop(self):
        interval = 1.0 / max(_RTK_YAW_INIT_RATE_HZ, 1.0)
        deadline = time.monotonic() + max(_RTK_YAW_INIT_DURATION_S, 0.0)
        stop = self._rtk_yaw_init_stop_event
        loc_assist_running = (
            self._loc_assist_thread is not None
            and self._loc_assist_thread.is_alive()
        )
        if loc_assist_running:
            self._stop_loc_assist()
            self.get_logger().info('RTK yaw-init stopped localization assist for direct /cmd_vel ownership')
        with self._lock:
            cmd_vel_proc = self._cmd_vel_proc
            self._cmd_vel_proc = None
            self._rtk_yaw_init_active = True
            was_nav_active = self._nav_active
        if was_nav_active:
            self._set_nav_active(False)
            self.get_logger().info('RTK yaw-init paused navigation before taking /cmd_vel ownership')
        had_cmd_vel_proc = cmd_vel_proc is not None and cmd_vel_proc.poll() is None
        if had_cmd_vel_proc:
            self._kill_proc(cmd_vel_proc)
            self.get_logger().info('RTK yaw-init stopped cmd_vel_control for direct /cmd_vel ownership')
        self._action_pub.publish(String(data='play stand'))
        self.get_logger().info('RTK yaw-init sent balance stand command')
        self.get_logger().info(
            f'RTK yaw-init started: linear.x={_RTK_YAW_INIT_SPEED_MPS:.2f}m/s '
            f'duration={_RTK_YAW_INIT_DURATION_S:.1f}s '
            f'required_clearance={_RTK_YAW_INIT_CLEARANCE_M:.2f}m'
        )
        try:
            while time.monotonic() < deadline:
                if self._rtk_yaw_init_should_stop(stop):
                    break
                self._publish_cmd_vel(_RTK_YAW_INIT_SPEED_MPS, 0.0)
                time.sleep(interval)
        finally:
            self._publish_cmd_vel(0.0, 0.0)
            with self._lock:
                self._rtk_yaw_init_active = False
                restart_cmd_vel = (
                    had_cmd_vel_proc
                    and self._nav_nodes_running
                    and self._cmd_vel_proc is None
                )
                resume_nav = was_nav_active and self._nav_nodes_running
                self._rtk_yaw_init_thread = None
            if restart_cmd_vel:
                env = os.environ.copy()
                env['PYTHONPATH'] = _VENV_SITE + ':' + env.get('PYTHONPATH', '')
                self._cmd_vel_proc = self._launch_proc(
                    'cmd_vel_control',
                    ['uv', 'run', 'python', '/tinynav/tinynav/platforms/cmd_vel_control.py'],
                    env=env,
                )
                self.get_logger().info('RTK yaw-init restarted cmd_vel_control')
            if resume_nav:
                self._set_nav_active(True)
                self.get_logger().info('RTK yaw-init resumed navigation')
            self.get_logger().info('RTK yaw-init stopped')

    def _set_nav_active(self, active: bool):
        with self._lock:
            self._nav_active = bool(active)
        self._nav_active_pub.publish(Bool(data=bool(active)))

    def _on_nav_done(self, msg: Bool):
        if not msg.data or self.state != 'navigation':
            return

        # map_node publishes the final 100% nav_progress and nav_done back-to-back,
        # and ROS does not guarantee cross-topic callback ordering.  If the last
        # POI is also a nav_flow handoff point, nav_done can arrive first.  Give
        # the progress callback a short grace window to start the handoff before
        # marking navigation idle.
        with self._lock:
            self._nav_done_seq += 1
            seq = self._nav_done_seq
            if self._map_handoff_active:
                return

        def finalize_if_no_handoff():
            latest_progress = None
            with self._lock:
                if seq != self._nav_done_seq or self._map_handoff_active or self.state != 'navigation':
                    return
                latest_progress = dict(self._nav_progress) if self._nav_progress else None

            if latest_progress:
                self._maybe_start_map_handoff(latest_progress)

            should_publish_nav_inactive = False
            with self._lock:
                if seq != self._nav_done_seq or self._map_handoff_active or self.state != 'navigation':
                    return
                self._nav_active = False
                self._active_nav_poi_ids = []
                self._active_nav_pois = []
                self.state = 'idle'
                should_publish_nav_inactive = True
            if should_publish_nav_inactive:
                self._nav_active_pub.publish(Bool(data=False))
            self._pub_state()

        threading.Timer(0.3, finalize_if_no_handoff).start()

    def _on_nav_progress(self, msg: String):
        try:
            data = json.loads(msg.data)
            with self._lock:
                self._nav_progress = data
            for cb in self.nav_progress_callbacks:
                cb(data)
            self._maybe_start_map_handoff(data)
        except json.JSONDecodeError:
            pass

    def _maybe_start_map_handoff(self, progress: dict):
        """Demo map-collaboration hook.

        If the active map folder contains map_handoff.json and the current
        route index has a rule, reaching that route index switches to the
        target map, waits for relocalization, then sends the next POI list.

        Schema, in the currently active map folder:
          {
            "0": {"target_map": "map_...", "poi_list": [1, 2]},
            "2": {"target_map": "map_other", "poi_list": [0]}
          }

        Keys are matched against POI name first, then POI id, with the old
        current-route index behavior kept only as a legacy fallback. poi_list
        values may be POI IDs or POI names in the target map's pois.json.
        """
        try:
            poi_index = int(progress.get('poi_index'))
            percent = float(progress.get('percent', 0.0))
        except (TypeError, ValueError):
            return
        poi_id = progress.get('poi_id')
        try:
            poi_id = int(poi_id) if poi_id is not None else None
        except (TypeError, ValueError):
            poi_id = None
        poi_name = progress.get('poi_name') if isinstance(progress.get('poi_name'), str) else None
        if percent < 100.0:
            return

        active_map = self._active_map_name()
        if not active_map:
            return
        key = (active_map, poi_name or poi_id or poi_index)
        with self._lock:
            if self._map_handoff_active or key in self._handled_map_handoffs:
                return

        rule = self._load_map_handoff_rule(poi_index, poi_id=poi_id, poi_name=poi_name)
        if rule is None:
            return

        with self._lock:
            self._map_handoff_active = True
            self._handled_map_handoffs.add(key)
        threading.Thread(
            target=self._run_map_handoff,
            args=(active_map, poi_index, rule),
            daemon=True,
        ).start()

    def _active_map_name(self) -> str | None:
        try:
            if os.path.islink(self.map_path):
                return os.path.basename(os.path.realpath(self.map_path))
            if os.path.isdir(self.map_path):
                return os.path.basename(self.map_path)
        except OSError:
            return None
        return None

    @staticmethod
    def _rtk_time_gate_open() -> bool:
        # Must match MapNode._rtk_time_gate_open in tinynav/core/map_node.py:
        # Oct-Mar effective after 18:00, Apr-Sep effective after 19:00.
        from datetime import datetime
        now = datetime.now()
        if now.month in (10, 11, 12, 1, 2, 3):
            return now.hour >= 18
        return now.hour >= 19

    def _load_nav_flow_enable_first_done(self) -> bool:
        config_path = os.path.join(self.map_path, 'nav_flow.json')
        if not os.path.exists(config_path):
            return False
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            self.get_logger().error(f'Failed to read nav_flow.json: {e}')
            return False
        if not isinstance(config, dict):
            return False
        value = config.get('enable_first_done', False)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'on'}
        if isinstance(value, (int, float)):
            return bool(value)
        self.get_logger().warn(f'Invalid nav_flow enable_first_done value: {value!r}')
        return False

    def _load_nav_flow_rtk_mode(self) -> str:
        config_path = os.path.join(self.map_path, 'nav_flow.json')
        if not os.path.exists(config_path):
            return 'off'
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            self.get_logger().error(f'Failed to read nav_flow.json: {e}')
            return 'off'
        if not isinstance(config, dict):
            return 'off'
        rtk_config = config.get('rtk', {})
        if isinstance(rtk_config, bool):
            mode = 'replace' if rtk_config else 'off'
        elif isinstance(rtk_config, str):
            mode = rtk_config.strip().lower()
        elif isinstance(rtk_config, dict):
            mode = str(rtk_config.get('mode', 'off')).strip().lower()
        else:
            self.get_logger().warn(f'Invalid nav_flow rtk config: {rtk_config!r}; using off')
            return 'off'
        if mode in {'replace', 'on', 'true', '1', 'yes'}:
            # rtk comment out
            # return 'replace' if self._rtk_time_gate_open() else 'off'
            return 'replace'
        if mode in {'off', 'false', '0', 'no', ''}:
            return 'off'
        self.get_logger().warn(f'Invalid nav_flow rtk.mode={mode!r}; using off')
        return 'off'

    def _publish_current_map_for_rtk(self):
        # Uses the value cached at nav start (self._nav_rtk_mode), not a live
        # re-read -- must be refreshed by the caller before this runs.
        mode = self._nav_rtk_mode
        msg = String()
        if mode == 'replace':
            msg.data = self.map_path
            self.get_logger().info(f'RTK enabled for current map: publishing /map/current_map={msg.data}')
        else:
            msg.data = ''
            self.get_logger().info('RTK disabled for current map: clearing /map/current_map')
        self._current_map_pub.publish(msg)

    def _load_map_handoff_rule(
        self,
        poi_index: int,
        *,
        poi_id: int | None = None,
        poi_name: str | None = None,
    ) -> dict | None:
        config_path = None
        for filename in ('nav_flow.json', 'map_handoff.json'):
            candidate = os.path.join(self.map_path, filename)
            if os.path.exists(candidate):
                config_path = candidate
                break
        if config_path is None:
            return None
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            self.get_logger().error(f'Failed to read {os.path.basename(config_path)}: {e}')
            return None

        rule = None
        if poi_name:
            if isinstance(config.get('by_name'), dict):
                rule = config['by_name'].get(poi_name)
            if rule is None:
                rule = config.get(poi_name)
        if rule is None and poi_id is not None:
            if isinstance(config.get('by_id'), dict):
                rule = config['by_id'].get(str(poi_id))
            if rule is None:
                rule = config.get(str(poi_id))
        if rule is None and isinstance(config.get('by_index'), dict):
            rule = config['by_index'].get(str(poi_index))
        if rule is None and isinstance(config.get('handoffs'), dict):
            rule = config['handoffs'].get(str(poi_index))
        if rule is None:
            rule = config.get(str(poi_index))
        if not isinstance(rule, dict):
            return None
        target_map = rule.get('target_map') or rule.get('map')
        poi_list = rule.get('poi_list', [])
        if not isinstance(target_map, str) or not re.match(r'^[a-zA-Z0-9_\-]+$', target_map):
            self.get_logger().error(f'Invalid map handoff target_map: {target_map!r}')
            return None
        if not isinstance(poi_list, list) or not all(isinstance(p, (int, str)) for p in poi_list):
            self.get_logger().error(f'Invalid map handoff poi_list: {poi_list!r}')
            return None
        zupt = rule.get('zupt')
        if zupt is not None and not isinstance(zupt, bool):
            self.get_logger().warn(f'Invalid zupt value: {zupt!r}, ignoring')
            zupt = None
        lookat = rule.get('lookat')
        if lookat is not None and not isinstance(lookat, (int, str)):
            self.get_logger().warn(f'Invalid lookat value: {lookat!r}, ignoring')
            lookat = None
        lookat_timeout_s = self._float_rule_value(rule, 'lookat_timeout_s', 6.0, minimum=0.5, maximum=30.0)
        lookat_yaw_tolerance_deg = self._float_rule_value(
            rule, 'lookat_yaw_tolerance_deg', 10.0, minimum=1.0, maximum=45.0
        )
        return {
            'target_map': target_map,
            'poi_list': poi_list,
            'zupt': zupt,
            'lookat': lookat,
            'lookat_timeout_s': lookat_timeout_s,
            'lookat_yaw_tolerance_deg': lookat_yaw_tolerance_deg,
        }

    def _float_rule_value(
        self,
        rule: dict,
        key: str,
        default: float,
        *,
        minimum: float,
        maximum: float,
    ) -> float:
        if key not in rule:
            return default
        try:
            value = float(rule[key])
        except (TypeError, ValueError):
            self.get_logger().warn(f'Invalid {key} value: {rule.get(key)!r}, using {default}')
            return default
        return max(minimum, min(maximum, value))

    def _set_active_map_link(self, map_name: str):
        import shutil
        root = self.tinynav_db_path
        src = os.path.join(root, 'maps', map_name)
        if not os.path.isdir(src):
            raise FileNotFoundError(f'Map {map_name!r} not found')
        link = self.map_path
        if os.path.islink(link) or os.path.isfile(link):
            os.remove(link)
        elif os.path.isdir(link):
            shutil.rmtree(link)
        os.symlink(src, link)

    def _wait_for_map_handoff_localization(self, target_map: str) -> bool:
        """Wait until the target map is localized before continuing nav_flow.

        By default we do not time out. Relocalization can legitimately take more
        than a minute in field tests, and dropping the pending poi_list at that
        point makes the nav_flow silently stop after the map switch.
        Set TINYNAV_MAP_HANDOFF_LOCALIZATION_TIMEOUT_S to a positive value if a
        deployment wants an explicit failure timeout.
        """
        timeout_s = _MAP_HANDOFF_LOCALIZATION_TIMEOUT_S
        deadline = time.time() + timeout_s if timeout_s > 0 else None
        next_log_time = time.time() + 10.0

        while True:
            with self._lock:
                localized = self._localized
                nav_nodes_running = self._nav_nodes_running
            if localized:
                return True
            if not nav_nodes_running:
                self.get_logger().warn(
                    f'Map handoff cancelled while waiting for localization on {target_map}'
                )
                return False
            if deadline is not None and time.time() >= deadline:
                self.get_logger().error(
                    f'Map handoff timed out waiting for localization on {target_map}'
                )
                return False
            if time.time() >= next_log_time:
                self.get_logger().info(
                    f'Map handoff waiting for localization on {target_map}; nav_flow POIs remain pending'
                )
                next_log_time = time.time() + 30.0
            time.sleep(0.2)

    def _load_pois_by_ref(self) -> dict[int | str, np.ndarray]:
        pois_file = os.path.join(self.map_path, 'pois.json')
        if not os.path.exists(pois_file):
            raise FileNotFoundError('No pois.json found for lookat')
        with open(pois_file) as f:
            pois = json.load(f)
        result: dict[int | str, np.ndarray] = {}
        for key, poi in pois.items():
            if not isinstance(poi, dict):
                continue
            position = poi.get('position')
            if not isinstance(position, list) or len(position) < 2:
                continue
            try:
                point = np.array(position[:3], dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if len(point) < 3:
                point = np.pad(point, (0, 3 - len(point)))
            try:
                result[int(key)] = point
            except (TypeError, ValueError):
                pass
            result[str(key)] = point
            name = poi.get('name')
            if isinstance(name, str):
                result[name] = point
        return result

    def _turn_relative_by_odom_for_lookat(
        self,
        target_delta: float,
        *,
        timeout_s: float,
        yaw_tolerance: float,
        angular_speed: float = 0.4,
        cmd_rate_hz: float = 10.0,
    ) -> bool:
        interval = 1.0 / max(cmd_rate_hz, 1.0)
        start_wait = time.monotonic()
        start_yaw = self._latest_odom_yaw()
        while start_yaw is None:
            self._publish_cmd_vel(0.0, 0.0)
            if time.monotonic() - start_wait > min(2.0, timeout_s):
                self.get_logger().warn('Map handoff lookat skipped: no fresh odometry yaw')
                return False
            time.sleep(interval)
            start_yaw = self._latest_odom_yaw()

        if abs(target_delta) <= yaw_tolerance:
            self._publish_cmd_vel(0.0, 0.0)
            return True

        angular_z = math.copysign(abs(angular_speed), target_delta)
        start_time = time.monotonic()
        previous_yaw = start_yaw
        accumulated_delta = 0.0

        while True:
            current_yaw = self._latest_odom_yaw()
            if current_yaw is None:
                self._publish_cmd_vel(0.0, 0.0)
                if time.monotonic() - start_time > timeout_s:
                    self.get_logger().warn('Map handoff lookat timed out: odometry yaw disappeared')
                    return False
                time.sleep(interval)
                continue

            accumulated_delta += self._wrap_angle(current_yaw - previous_yaw)
            previous_yaw = current_yaw
            remaining = target_delta - accumulated_delta
            if abs(remaining) <= yaw_tolerance:
                self._publish_cmd_vel(0.0, 0.0)
                return True
            if math.copysign(1.0, remaining) != math.copysign(1.0, target_delta):
                self._publish_cmd_vel(0.0, 0.0)
                return True
            if time.monotonic() - start_time > timeout_s:
                self.get_logger().warn(
                    f'Map handoff lookat timed out: target_delta={target_delta:.3f} '
                    f'accumulated_delta={accumulated_delta:.3f} remaining={remaining:.3f}'
                )
                self._publish_cmd_vel(0.0, 0.0)
                return False

            self._publish_cmd_vel(0.0, angular_z)
            time.sleep(interval)

    def _get_pose_for_lookat(self, *, max_wait_s: float = 2.0) -> dict | None:
        """Return map-frame pose for lookat, waiting briefly for a fresh /mapping/current_pose_in_map update."""
        deadline = time.monotonic() + max(0.0, max_wait_s)
        while True:
            with self._lock:
                if self._map_pose is not None:
                    return dict(self._map_pose)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def _run_map_handoff_lookat(self, lookat: int | str, timeout_s: float, yaw_tolerance_deg: float) -> None:
        try:
            pois = self._load_pois_by_ref()
        except Exception as exc:
            self.get_logger().warn(f'Map handoff lookat skipped: {exc}')
            return
        if lookat not in pois:
            self.get_logger().warn(f'Map handoff lookat POI {lookat!r} not found in current map')
            return

        # Stop path following before taking direct /cmd_vel ownership for the in-place turn.
        self._set_nav_active(False)
        self._publish_cmd_vel(0.0, 0.0)

        pose = self._get_pose_for_lookat()
        if pose is None:
            self.get_logger().warn('Map handoff lookat skipped: no current map pose')
            return

        with self._lock:
            cmd_vel_proc = self._cmd_vel_proc
            if cmd_vel_proc is not None and cmd_vel_proc.poll() is None:
                self._cmd_vel_proc = None
            else:
                cmd_vel_proc = None
        if cmd_vel_proc is not None:
            self._kill_proc(cmd_vel_proc)
            self.get_logger().info('Map handoff lookat stopped cmd_vel_control for direct /cmd_vel ownership')

        target = pois[lookat]
        dx = float(target[0] - pose['x'])
        dy = float(target[1] - pose['y'])
        if math.hypot(dx, dy) < 1e-3:
            self.get_logger().warn(f'Map handoff lookat skipped: POI {lookat!r} is too close')
            return
        target_yaw = math.atan2(dy, dx)
        current_yaw = float(pose.get('yaw', 0.0))
        target_delta = self._wrap_angle(target_yaw - current_yaw)
        self.get_logger().info(
            f'Map handoff lookat started: target={lookat!r}, '
            f'delta_deg={math.degrees(target_delta):.1f}, timeout_s={timeout_s:.1f}, '
            f'tolerance_deg={yaw_tolerance_deg:.1f}'
        )
        ok = self._turn_relative_by_odom_for_lookat(
            target_delta,
            timeout_s=timeout_s,
            yaw_tolerance=math.radians(yaw_tolerance_deg),
        )
        self._publish_cmd_vel(0.0, 0.0)
        self.get_logger().info(f'Map handoff lookat {"finished" if ok else "continued after timeout"}')

    def _maybe_seed_map_handoff(
        self, source_map: str, target_map: str, T_source_to_odom_live: np.ndarray | None,
    ) -> str | None:
        """If a calibrated map_handoff_from_<source_map>.json edge exists inside the target
        map's own directory (see tool/calibrate_map_transform.py), compute a seed
        T_from_map_to_odom for the target map and write it to a temp file map_node.py can be
        launched with, so nav on the target map doesn't wait on a fresh cold relocalization.
        Returns None (map_node falls back to its normal cold-start behavior) if there's no
        edge for this map pair, or if we don't have a live source-map pose to seed from.
        """
        if T_source_to_odom_live is None:
            self.get_logger().warning(
                f'Map handoff {source_map}->{target_map}: no live source-map pose available, '
                'target map will cold-start relocalization as usual'
            )
            return None
        edge_path = os.path.join(self.tinynav_db_path, 'maps', target_map, f'map_handoff_from_{source_map}.json')
        if not os.path.exists(edge_path):
            return None
        try:
            with open(edge_path) as f:
                edge = json.load(f)
            source_to_target = np.array(edge['mapA_to_mapB'])
            T_target_to_odom_seed = T_source_to_odom_live @ np.linalg.inv(source_to_target)
        except Exception as exc:
            self.get_logger().error(f'Map handoff {source_map}->{target_map}: failed to apply edge {edge_path}: {exc}')
            return None
        seed_path = f'/tmp/map_handoff_seed_{source_map}_to_{target_map}.npy'
        np.save(seed_path, T_target_to_odom_seed)
        self.get_logger().info(
            f'Map handoff {source_map}->{target_map}: seeded T_from_map_to_odom from {edge_path} -> {seed_path}'
        )
        return seed_path

    def _run_map_handoff(self, source_map: str, poi_index: int, rule: dict):
        target_map = rule['target_map']
        poi_list = rule['poi_list']
        self.get_logger().info(
            f'Map handoff triggered: {source_map}[{poi_index}] -> {target_map}, poi_list={poi_list}'
            + (f", lookat={rule.get('lookat')!r}" if rule.get('lookat') is not None else '')
        )
        # Snapshot the live source-map localization NOW, before cmd_stop_nav_nodes/
        # _set_active_map_link tear down this session's state below -- this is the one
        # (map_pose, odom_pose) pair we have to seed the target map's T_from_map_to_odom
        # with, if a calibrated edge exists for this map pair (see _maybe_seed_map_handoff).
        T_source_to_odom_live = self._compute_live_map_to_odom()
        try:
            lookat = rule.get('lookat')
            if lookat is not None:
                self._run_map_handoff_lookat(
                    lookat,
                    timeout_s=float(rule.get('lookat_timeout_s', 6.0)),
                    yaw_tolerance_deg=float(rule.get('lookat_yaw_tolerance_deg', 10.0)),
                )

            # Stop current map_node/control hard before changing the active map.
            self.cmd_stop_nav_nodes()
            self.state = 'idle'
            self._pub_state()

            self._set_active_map_link(target_map)

            with self._lock:
                self._localized = False
                self._map_pose = None
                self._global_path = []
                self._final_global_path = []
                self._nav_target_pose = None
                self._nav_progress = None

            # Apply ZUPT setting before relocalization if specified in nav_flow rule
            zupt = rule['zupt']
            if zupt is not None:
                zupt_cmd = 'enable' if zupt else 'disable'
                try:
                    subprocess.run(
                        ['sshpass', '-p', 'looper@0731', 'python3',
                         '/tinynav/looper_cli/looper_cli.py',
                         'zupt', zupt_cmd, '-y'],
                        check=True, timeout=30,
                    )
                    self.get_logger().info(f'ZUPT {zupt_cmd}d before map handoff to {target_map}')
                except Exception as e:
                    self.get_logger().error(f'Failed to set ZUPT {zupt_cmd}: {e}')
                # Give the device time to apply the ZUPT change before restarting nav nodes.
                time.sleep(3)

            seed_transform_path = self._maybe_seed_map_handoff(source_map, target_map, T_source_to_odom_live)
            self.cmd_start_nav_nodes(initial_map_to_odom_transform_path=seed_transform_path)

            if not self._wait_for_map_handoff_localization(target_map):
                self.state = 'idle'
                self._pub_state()
                return

            if poi_list:
                self.cmd_send_pois(poi_list)
            else:
                self.state = 'idle'
                self._pub_state()
        except Exception as e:
            self.get_logger().error(f'Map handoff failed: {e}')
            self.state = 'error:map_handoff'
            self._pub_state()
        finally:
            with self._lock:
                self._map_handoff_active = False

    def _on_mapping_percent(self, msg: Float32):
        with self._lock:
            self.mapping_percent = float(msg.data)

    def _on_slam_odom(self, msg: Odometry):
        pose = self._odom_to_dict(msg, source='slam')
        with self._lock:
            self.current_pose = pose
            self._odom_pose = pose
            self._odom_pose_received_at = time.monotonic()
        for cb in self.pose_callbacks:
            try:
                cb(pose)
            except Exception:
                pass

    def _on_ekf_odom(self, msg: Odometry):
        pose = self._odom_to_dict(msg, source='ekf')
        with self._lock:
            self._ekf_odom_pose = pose

    def _active_odom_pose_locked(self) -> dict | None:
        if self._odom_source == 'ekf':
            return self._ekf_odom_pose or self._odom_pose_at_kf or self._odom_pose
        return self._odom_pose_at_kf or self._odom_pose

    def _live_odom_pose_locked(self) -> dict | None:
        # Live pose for planning canvas overlays. Must match planning_node odom frame.
        if self._odom_source == 'ekf':
            return self._ekf_odom_pose or self._odom_pose
        return self._odom_pose

    def _on_pose_in_map(self, msg: Odometry):
        pose = self._odom_to_dict(msg, source='map')
        with self._lock:
            was_localized = self._localized
            self.current_pose = pose
            self._map_pose = pose
            self._odom_pose_at_kf = (
                self._ekf_odom_pose if self._odom_source == 'ekf' else self._odom_pose
            )
            self._localized = True
        if not was_localized:
            self._on_localization_achieved()
        for cb in self.pose_callbacks:
            try:
                cb(pose)
            except Exception:
                pass

    def _on_relocalization(self, msg: Odometry):
        pose = self._odom_to_dict(msg, source='map')
        with self._lock:
            was_localized = self._localized
            self._map_pose = pose
            self._localized = True
        if not was_localized:
            self._on_localization_achieved()

    def _on_nav_target_pose(self, msg: Odometry):
        with self._lock:
            self._nav_target_pose = {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
            }

    def _planning_viz_active(self) -> bool:
        return time.monotonic() <= self._planning_viz_wanted_until

    def _on_height_map(self, msg: Image):
        now = time.monotonic()
        if now - self._last_height_map_time < _PLANNING_VIZ_MIN_INTERVAL:
            return
        if not self._planning_viz_active():
            return
        self._last_height_map_time = now
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            if msg.encoding == 'rgb8':
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            # Grid is (X_dim, Y_dim, 3): rows=X, cols=Y.
            # Transpose + flipud → rows=Y(inverted), cols=X, so canvas X=right Y=up matches painter.
            arr = np.flipud(arr.transpose(1, 0, 2))
            # Invert JET colormap so dangerous (near obstacle) = red, safe = blue.
            arr = arr[:, :, ::-1]
            _, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self._lock:
                self._esdf_bytes = buf.tobytes()
        except Exception:
            pass

    def _on_obstacle_mask(self, msg: OccupancyGrid):
        now = time.monotonic()
        if now - self._last_obstacle_mask_time < _PLANNING_VIZ_MIN_INTERVAL:
            return
        if not self._planning_viz_active():
            return
        self._last_obstacle_mask_time = now
        try:
            # Planning node stores OccupancyGrid in Fortran (column-major) order.
            arr = np.array(msg.data, dtype=np.int8)
            grid = arr.reshape(msg.info.height, msg.info.width, order='F')  # (X_dim, Y_dim)
            img = np.where(grid > 50, 255, 0).astype(np.uint8)
            # Transpose + flipud → rows=Y(inverted), cols=X, matching painter (X=right, Y=up).
            img = np.flipud(img.T)
            _, buf = cv2.imencode('.png', img)
            info = {
                'origin_x': float(msg.info.origin.position.x),
                'origin_y': float(msg.info.origin.position.y),
                'resolution': float(msg.info.resolution),
                'width': int(msg.info.height),   # X_dim → image cols (horizontal)
                'height': int(msg.info.width),   # Y_dim → image rows (vertical)
            }
            with self._lock:
                self._obstacle_bytes = buf.tobytes()
                self._grid_info = info
        except Exception:
            pass

    def _on_trajectory_path(self, msg: Path):
        pts = [
            {'x': p.pose.position.x, 'y': p.pose.position.y}
            for p in msg.poses
        ]
        with self._lock:
            self._trajectory = pts

    def _on_global_plan(self, msg: Path):
        pts = [
            {'x': p.pose.position.x, 'y': p.pose.position.y}
            for p in msg.poses
        ]
        with self._lock:
            self._global_path = pts

    def _on_final_global_plan(self, msg: Path):
        pts = [
            {'x': p.pose.position.x, 'y': p.pose.position.y}
            for p in msg.poses
        ]
        with self._lock:
            self._final_global_path = pts

    def _on_footprint(self, msg: PointCloud):
        """Store footprint corner points from PointCloud.

        The planning node publishes 84 points (4 edges × 21 samples per edge).
        We extract the 4 corner points (first of each edge group).
        """
        now = time.monotonic()
        if now - self._last_footprint_time < _PLANNING_VIZ_MIN_INTERVAL:
            return
        if not self._planning_viz_active():
            return
        self._last_footprint_time = now
        n = len(msg.points)
        if n == 0:
            return
        # If 84 points (4 edges × 21), extract corners; otherwise store all unique points
        if n >= 84 and n % 21 == 0:
            edges = n // 21
            corners = []
            for i in range(edges):
                p = msg.points[i * 21]
                corners.append({'x': p.x, 'y': p.y})
        else:
            # Fallback: store all points
            corners = [{'x': p.x, 'y': p.y} for p in msg.points]
        with self._lock:
            self._footprint = corners

    def _on_occupied_voxels(self, msg: PointCloud2):
        """Store a downsampled local 3D occupied voxel cloud for the web UI."""
        now = time.monotonic()
        if now - self._last_voxel_time < _PLANNING_VIZ_MIN_INTERVAL:
            return
        if not self._planning_viz_active():
            return
        if now > self._voxels_wanted_until:
            return
        self._last_voxel_time = now
        try:
            n = int(msg.width) * int(msg.height)
            if n <= 0 or msg.point_step <= 0:
                with self._lock:
                    self._voxel_points = []
                return
            dt = np.dtype({
                'names': ['x', 'y', 'z'],
                'formats': ['<f4', '<f4', '<f4'],
                'offsets': [0, 4, 8],
                'itemsize': int(msg.point_step),
            })
            xyz = np.frombuffer(msg.data, dtype=dt, count=n)
            if n > 800:
                xyz = xyz[:: max(1, n // 800)][:800]
            points = [
                {'x': float(p['x']), 'y': float(p['y']), 'z': float(p['z'])}
                for p in xyz
            ]
            with self._lock:
                self._voxel_points = points
        except Exception:
            pass

    def _on_vio_status(self, msg: String):
        status = msg.data.strip().upper()
        normal = status in _VIO_STATUS_NORMAL

        with self._lock:
            previous_status = self._vio_status
            self._vio_status = status
            already_stopped = self._vio_guard_stopped
            recovering = self._vio_guard_recovering
            nav_running = self._nav_nodes_running

        if normal:
            if already_stopped and not recovering:
                self._recover_from_vio_guard_stop(status)
            return

        if already_stopped or not nav_running:
            return

        self._stop_for_vio_guard(status, previous_status)

    def _stop_for_vio_guard(self, status: str, previous_status: str | None):
        resume_ids = self._remaining_nav_poi_ids_for_resume()
        with self._lock:
            self._vio_guard_stopped = True
            self._vio_guard_recovering = False
            self._vio_resume_poi_ids = resume_ids
            self._localized = False

        self.get_logger().warn(
            f'Insight VIO abnormal ({previous_status!r} -> {status!r}); '
            f'stopping nav nodes, remaining_pois={resume_ids!r}'
        )
        self.cmd_stop_nav_nodes()
        with self._lock:
            if self.state == 'navigation':
                self.state = 'idle'
        self._pub_state()

    def _recover_from_vio_guard_stop(self, status: str):
        with self._lock:
            resume_ids = list(self._vio_resume_poi_ids)
            if not self._vio_guard_stopped or self._vio_guard_recovering:
                return
            self._vio_guard_recovering = True

        self.get_logger().info(
            f'Insight VIO recovered ({status!r}); starting nav nodes before resuming POIs={resume_ids!r}'
        )
        self.cmd_start_nav_nodes()

    def _remaining_nav_poi_ids_for_resume(self) -> list[int | str]:
        with self._lock:
            poi_ids = list(self._active_nav_poi_ids)
            progress = dict(self._nav_progress) if self._nav_progress else None

        if not poi_ids:
            return []

        index = 0
        if progress:
            try:
                index = int(progress.get('poi_index', 0))
                percent = float(progress.get('percent', 0.0))
                if percent >= 100.0:
                    index += 1
            except (TypeError, ValueError):
                index = 0
        index = max(0, min(index, len(poi_ids)))
        return poi_ids[index:]

    def _resume_vio_pois_after_localized(self):
        with self._lock:
            if not self._vio_guard_stopped or not self._vio_guard_recovering:
                return
            resume_ids = list(self._vio_resume_poi_ids)
            self._vio_guard_stopped = False
            self._vio_guard_recovering = False
            self._vio_resume_poi_ids = []

        if resume_ids:
            self.get_logger().info(f'Resuming POIs after VIO recovery localization: {resume_ids!r}')
            self.cmd_send_pois(resume_ids)
        else:
            self.get_logger().info('VIO recovered and localized; no remaining POIs to resume')

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _odom_to_dict(msg: Odometry, source: str) -> dict:
        q = msg.pose.pose.orientation
        # SLAM outputs camera-convention poses (body Z = forward).
        # Project body Z-axis onto world XY to get the true forward heading,
        # which is robust to pitch oscillations during the walking gait.
        fwd_x = 2.0 * (q.x * q.z + q.w * q.y)
        fwd_y = 2.0 * (q.y * q.z - q.w * q.x)
        yaw = math.atan2(fwd_y, fwd_x) if (abs(fwd_x) > 1e-9 or abs(fwd_y) > 1e-9) else 0.0
        return {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z,
            'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w,
            'yaw': yaw,
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'source': source,
        }

    @staticmethod
    def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        return np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [    2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qw*qx)],
            [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
        ])

    @classmethod
    def _pose_dict_to_matrix(cls, pose: dict) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = cls._quat_to_rot(pose['qx'], pose['qy'], pose['qz'], pose['qw'])
        T[:3, 3] = [pose['x'], pose['y'], pose['z']]
        return T

    def _compute_live_map_to_odom(self) -> np.ndarray | None:
        """T_from_map_to_odom for the CURRENTLY active map, from whatever pose_in_map and
        pose_in_odom this session has already observed -- used to seed the next map at a
        handoff (see _run_map_handoff) instead of that map waiting on a fresh cold
        relocalization. _odom_pose_at_kf is odom frozen at the same keyframe _map_pose came
        from (see _on_pose_in_map), so the pair is already time-synchronized.
        """
        with self._lock:
            map_pose = self._map_pose
            odom_source = self._odom_source
            odom_pose = self._active_odom_pose_locked()
        if map_pose is None or odom_pose is None:
            self.get_logger().warning(
                f'Map handoff seed skipped: map_pose={map_pose is not None} '
                f'odom_pose={odom_pose is not None} odom_source={odom_source}'
            )
            return None
        pose_in_map = self._pose_dict_to_matrix(map_pose)
        pose_in_odom = self._pose_dict_to_matrix(odom_pose)
        self.get_logger().info(
            f'Map handoff live T_from_map_to_odom using odom_source={odom_source} '
            f'odom_xyz=({odom_pose["x"]:.2f},{odom_pose["y"]:.2f},{odom_pose["z"]:.2f}) '
            f'map_xyz=({map_pose["x"]:.2f},{map_pose["y"]:.2f},{map_pose["z"]:.2f})'
        )
        return pose_in_odom @ np.linalg.inv(pose_in_map)

    def _transform_path_via_tf(self, path: list) -> list:
        """Transform map-frame path points to odom (world) frame via TF lookup."""
        if not path:
            return path
        try:
            t = self._tf_buffer.lookup_transform('world', 'map', rclpy.time.Time())
            tr = t.transform.translation
            rot = t.transform.rotation
            R = self._quat_to_rot(rot.x, rot.y, rot.z, rot.w)
            trans = np.array([tr.x, tr.y, tr.z])
            result = []
            for pt in path:
                p = R @ np.array([pt['x'], pt['y'], 0.0]) + trans
                result.append({'x': float(p[0]), 'y': float(p[1])})
            return result
        except Exception:
            return path  # TF not yet available — fall back to map-frame coords

    # ------------------------------------------------------------------ #
    # Sensor / camera                                                      #
    # ------------------------------------------------------------------ #

    def _detect_and_init_sensor(self):
        domain = os.environ.get('ROS_DOMAIN_ID', '0')
        self.get_logger().info(f'BackendNode ROS_DOMAIN_ID={domain}')
        try:
            result = subprocess.run(
                ['ros2', 'node', 'list'], capture_output=True, text=True, timeout=3
            )
            nodes = set(result.stdout.splitlines())
            if nodes & _LOOPER_NODE_NAMES:
                self._sensor_mode = 'looper'
                self.get_logger().info(
                    f'Sensor mode: looper ({", ".join(sorted(nodes & _LOOPER_NODE_NAMES))})'
                    ' — launching looper bridge + planning'
                )
            else:
                self._sensor_mode = 'realsense'
                self.get_logger().info('Sensor mode: realsense — launching driver + perception + planning')
            self._sensor_mode = 'looper'
            if self._sensor_mode == 'looper' and self._vio_status_sub is None:
                self._vio_status_sub = self.create_subscription(
                    String, '/insight/vio_status', self._on_vio_status, 10
                )
                self.get_logger().info('Insight VIO guard enabled for looper sensor mode')
            if self._sensor_mode in ('looper', 'realsense'):
                _env = os.environ.copy()
                _env['PYTHONPATH'] = _VENV_SITE + ':' + _env.get('PYTHONPATH', '')
                self._launch_sensor_procs(_env)
        except Exception as e:
            self.get_logger().warn(f'Sensor detection failed: {e}')
            self._sensor_mode = 'unknown'

        topics = self.get_image_topics()
        # logical_topic -> {'left': ndarray|None, 'right': ndarray|None}
        self._preview_sides: dict[str, dict[str, np.ndarray | None]] = {}
        for topic in topics:
            self._last_frame[topic] = b''
            self._last_frame_time[topic] = 0.0
            self.preview_callbacks[topic] = []
            self._preview_sides[topic] = {'left': None, 'right': None}

    def add_preview_callback(self, topic: str, cb) -> bool:
        """Register a frame callback; creates the ROS subscription on the first caller."""
        if topic not in self.preview_callbacks:
            return False
        with self._lock:
            self.preview_callbacks[topic].append(cb)
            first = len(self.preview_callbacks[topic]) == 1
        if first:
            self._create_image_sub(topic)
        return True

    def remove_preview_callback(self, topic: str, cb):
        """Unregister a frame callback; destroys the ROS subscription when the last caller leaves."""
        if topic not in self.preview_callbacks:
            return
        with self._lock:
            try:
                self.preview_callbacks[topic].remove(cb)
            except ValueError:
                pass
            empty = len(self.preview_callbacks[topic]) == 0
        if empty:
            self._destroy_image_sub(topic)

    def _create_image_sub(self, topic: str):
        """Subscribe only to the front /camera/... topic. camera1 is not previewed."""
        if f'{topic}::left' in self._image_subs or topic in self._image_subs:
            return
        self._subscribe_source(topic, logical_topic=topic, side='left')

    def _subscribe_source(self, ros_topic: str, *, logical_topic: str, side: str):
        key = f'{logical_topic}::{side}'
        if key in self._image_subs:
            return
        if ros_topic.endswith('/compressed'):
            self._image_subs[key] = self.create_subscription(
                CompressedImage, ros_topic,
                lambda msg, lt=logical_topic, s=side: self._on_compressed_image(msg, lt, s),
                _SENSOR_QOS,
            )
        else:
            self._image_subs[key] = self.create_subscription(
                Image, ros_topic,
                lambda msg, lt=logical_topic, s=side: self._on_image(msg, lt, s),
                _SENSOR_QOS,
            )

    def _destroy_image_sub(self, topic: str):
        for side in ('left', 'right'):
            key = f'{topic}::{side}'
            sub = self._image_subs.pop(key, None)
            if sub is not None:
                self.destroy_subscription(sub)
        sub = self._image_subs.pop(topic, None)
        if sub is not None:
            self.destroy_subscription(sub)
        sides = self._preview_sides.get(topic)
        if sides is not None:
            sides['left'] = None
            sides['right'] = None

    def _decode_image_msg(self, msg: Image):
        enc = (msg.encoding or '').lower()
        if enc == '32fc1':
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            valid = arr[arr > 0]
            if valid.size > 0:
                p95 = float(np.percentile(valid, 95))
                arr = np.clip(arr / (p95 + 1e-6), 0.0, 1.0)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
            arr = (arr * 255).astype(np.uint8)
            return cv2.applyColorMap(arr, cv2.COLORMAP_JET)
        if enc in ('mono16', '16uc1'):
            arr = np.frombuffer(msg.data, dtype='<u2').reshape(msg.height, msg.width)
            if msg.is_bigendian:
                arr = arr.byteswap()
            depth = arr.astype(np.float32)
            valid = depth[depth > 0]
            if valid.size > 0:
                p95 = float(np.percentile(valid, 95))
                depth = np.clip(depth / (p95 + 1e-6), 0.0, 1.0)
            else:
                depth = np.zeros_like(depth)
            arr = (depth * 255).astype(np.uint8)
            return cv2.applyColorMap(arr, cv2.COLORMAP_JET)
        if enc in ('mono8', '8uc1'):
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        if enc == 'rgb8':
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    def _emit_stitched_preview(self, logical_topic: str):
        sides = self._preview_sides.get(logical_topic) or {}
        left = sides.get('left')
        right = sides.get('right')
        if left is None and right is None:
            return
        try:
            if left is not None and right is not None:
                arr = _stitch_side_by_side(left, right)
            else:
                arr = _to_bgr_preview(left if left is not None else right)
            frame = _encode_preview_jpeg(arr)
        except Exception:
            return
        with self._lock:
            self._last_frame[logical_topic] = frame
            callbacks = list(self.preview_callbacks.get(logical_topic, []))
        for cb in callbacks:
            try:
                cb(frame)
            except Exception:
                pass

    def _on_compressed_image(self, msg: CompressedImage, logical_topic: str, side: str = 'left'):
        now = time.time()
        side_key = f'{logical_topic}::{side}'
        if now - self._last_frame_time.get(side_key, 0.0) < _PREVIEW_MIN_INTERVAL:
            return
        self._last_frame_time[side_key] = now
        try:
            arr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                return
        except Exception:
            return
        if logical_topic not in self._preview_sides:
            self._preview_sides[logical_topic] = {'left': None, 'right': None}
        self._preview_sides[logical_topic][side] = _resize_preview_frame(arr)
        self._emit_stitched_preview(logical_topic)

    def _on_image(self, msg: Image, logical_topic: str, side: str = 'left'):
        now = time.time()
        side_key = f'{logical_topic}::{side}'
        if now - self._last_frame_time.get(side_key, 0.0) < _PREVIEW_MIN_INTERVAL:
            return
        self._last_frame_time[side_key] = now
        try:
            arr = self._decode_image_msg(msg)
            if arr is None:
                return
        except Exception:
            return
        if logical_topic not in self._preview_sides:
            self._preview_sides[logical_topic] = {'left': None, 'right': None}
        self._preview_sides[logical_topic][side] = _resize_preview_frame(arr)
        self._emit_stitched_preview(logical_topic)

    def get_vio_status(self) -> str:
        with self._lock:
            return self._vio_status

    @staticmethod
    def _cap_path_pts(pts: list, max_n: int = 400) -> list:
        if len(pts) <= max_n:
            return list(pts)
        step = max(1, len(pts) // max_n)
        return list(pts[::step][:max_n])

    def get_planning_snapshot(self, include_voxels: bool = False) -> dict:
        self._planning_viz_wanted_until = time.monotonic() + 1.0
        if include_voxels:
            self._voxels_wanted_until = time.monotonic() + 1.0
        with self._lock:
            path_snapshot = self._cap_path_pts(self._global_path)
            final_path_snapshot = self._cap_path_pts(self._final_global_path)
            snapshot = {
                'localized': self._localized,
                'odom_pose': self._live_odom_pose_locked(),
                'odom_pose_at_kf': self._odom_pose_at_kf,
                'map_pose': self._map_pose,
                'esdf_image': base64.b64encode(self._esdf_bytes).decode() if self._esdf_bytes else None,
                'obstacle_image': base64.b64encode(self._obstacle_bytes).decode() if self._obstacle_bytes else None,
                'trajectory': list(self._trajectory),
                'global_path': None,  # filled after TF transform (odom frame)
                'map_global_path': path_snapshot,
                'final_global_path': None,  # filled after TF transform (odom frame)
                'map_final_global_path': final_path_snapshot,
                'grid_info': self._grid_info,
                'nav_target_pose': self._nav_target_pose,
                'active_nav_pois': list(self._active_nav_pois),
                'footprint': list(self._footprint),
                # 3D overlay is off by default; shipping 800 xyz dicts at 5 fps
                # makes Flutter web jsonDecode/GC hitch until the tab freezes.
                'voxel_points': list(self._voxel_points) if include_voxels else [],
            }
        snapshot['global_path'] = self._transform_path_via_tf(path_snapshot)
        snapshot['final_global_path'] = self._transform_path_via_tf(final_path_snapshot)
        return snapshot

    def _start_unitree_if_configured(self):
        _env = os.environ.copy()
        _env['PYTHONPATH'] = _VENV_SITE + ':' + _env.get('PYTHONPATH', '')
        self._unitree_proc = self._launch_proc(
            'unitree',
            ['uv', 'run', 'python', '/tinynav/tinynav/platforms/unitree_control.py'],
            env=_env,
        )
        self.get_logger().info('unitree_control started')

    def get_sensor_mode(self) -> str:
        return self._sensor_mode

    def get_image_topics(self) -> list[str]:
        if self._sensor_mode == 'looper':
            return list(_IMAGE_TOPICS_LOOPER) + list(_IMAGE_TOPICS_LOOPER_CAM1)
        return list(_IMAGE_TOPICS_REALSENSE)

    def get_preview_frame(self, topic: str) -> bytes:
        with self._lock:
            return self._last_frame.get(topic, b'')

    # ------------------------------------------------------------------ #
    # Command API (called from FastAPI handlers — thread-safe enough)     #
    # ------------------------------------------------------------------ #

    def set_active_bag(self, bag_name: str):
        """Select a bag from rosbags/ by name for map building."""
        path = os.path.join(self.tinynav_db_path, 'rosbags', bag_name)
        if os.path.isdir(path):
            with self._lock:
                self._last_verified_bag = path

    @property
    def active_bag_path(self) -> str | None:
        """Most recently verified bag folder, ready for map building."""
        lvb = self._last_verified_bag
        if lvb and os.path.isdir(lvb):
            return lvb
        return None

    def get_status(self) -> dict:
        with self._lock:
            raw = self.state
            pct = self.mapping_percent
            battery = self._battery
            nav_nodes = self._nav_nodes_running
            nav_paused = self._nav_paused
            nav_active = self._nav_active
            loc_assist = self._loc_assist_enabled
            planning_occupancy_source = self._planning_occupancy_source
            odom_source = self._odom_source
            vio_guard_enabled = self._sensor_mode == 'looper'
            vio_status = self._vio_status if vio_guard_enabled else None
            vio_guard_stopped = self._vio_guard_stopped if vio_guard_enabled else False
            active_map = self._active_map_name()
            rtk_mode = self._nav_rtk_mode
            rtk_bridge_status = self._rtk_bridge_status
            rtk_map_status = self._rtk_map_status
        rtk_yaw_init_active = self._rtk_yaw_init_active
        bag_files_exist = self.active_bag_path is not None
        map_files_exist = os.path.exists(os.path.join(self.map_path, 'occupancy_grid.npy'))
        return {
            'battery': battery,
            'bagStatus': 'recording' if raw == 'realsense_bag_record' else 'idle',
            'bagFileReady': bag_files_exist,
            'mapStatus': self._derive_map_status(raw, pct, map_files_exist),
            'mappingPercent': pct,
            'navStatus': 'navigating' if raw == 'navigation' else 'idle',
            'rawState': raw,
            'navNodesRunning': nav_nodes,
            'navPaused': nav_paused,
            'navActive': nav_active,
            'activeMap': active_map,
            'debugRecording': self.debug_recording,
            'locAssistEnabled': loc_assist,
            'planningOccupancySource': planning_occupancy_source,
            'odomSource': odom_source,
            'vioGuardEnabled': vio_guard_enabled,
            'vioStatus': vio_status,
            'vioGuardStopped': vio_guard_stopped,
            'rtkMode': rtk_mode,
            # Receiver fix quality (NO_FIX/SINGLE/DGNSS/RTK_FLOAT/RTK_FIXED/...) —
            # comes from rtk_bridge_node, which runs standalone (scripts/run_rtk.sh)
            # regardless of nav state, so this is available even when nav is off.
            'rtkBridgeOnline': rtk_bridge_status is not None,
            'rtkReceiverStage': (rtk_bridge_status or {}).get('receiver_stage'),
            'rtkBridgeAccepted': (rtk_bridge_status or {}).get('accepted'),
            'rtkCalculateStatusName': (rtk_bridge_status or {}).get('rtk_calculate_status_name'),
            # Map-alignment state (rtk_map_pose_node) — only meaningful once nav
            # has decided to run in 'replace' mode and published /map/current_map.
            'rtkMapState': (rtk_map_status or {}).get('state'),
            'rtkFixOk': (rtk_map_status or {}).get('fix_ok'),
            'rtkYawReady': (rtk_map_status or {}).get('yaw_ready'),
            'rtkYawInitActive': rtk_yaw_init_active,
        }

    @staticmethod
    def _derive_map_status(raw: str, pct: float, files_exist: bool) -> str:
        if raw == 'rosbag_build_map':
            return 'building'
        if raw.startswith('error:'):
            return 'failed'
        if files_exist and raw == 'idle':
            return 'success'
        return 'idle'

    # ------------------------------------------------------------------ #
    # Sensor proc helpers                                                  #
    # ------------------------------------------------------------------ #

    def _kill_proc(self, proc: subprocess.Popen | None):
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), 15)
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _make_log(self, name: str):
        """Open a timestamped log file under tinynav_db/logs/. Safe to close in parent
        after Popen — the child process inherits its own fd copy at fork time."""
        from datetime import datetime
        logs_dir = os.path.join(self.tinynav_db_path, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        path = os.path.join(logs_dir, f'{ts}_{name}.txt')
        return open(path, 'w')

    def _launch_proc(self, name: str, cmd: list[str], env: dict | None = None,
                      cwd: str = '/tinynav') -> subprocess.Popen:
        """Spawn a subprocess with standard logging and process-group setup."""
        lf = self._make_log(name)
        proc = subprocess.Popen(
            cmd, preexec_fn=os.setsid, cwd=cwd,
            env=env or os.environ.copy(),
            stdout=lf, stderr=subprocess.STDOUT,
        )
        log_path = lf.name
        lf.close()
        if name == 'planning':
            self._start_planning_log_monitor(log_path)
        return proc

    def _start_planning_log_monitor(self, log_path: str):
        self._planning_log_stop_event.set()
        stop_event = threading.Event()
        self._planning_log_stop_event = stop_event
        with self._lock:
            self._planning_occupancy_source = 'depth'
        self._planning_log_thread = threading.Thread(
            target=self._monitor_planning_log,
            args=(log_path, stop_event),
            daemon=True,
        )
        self._planning_log_thread.start()

    def _monitor_planning_log(self, log_path: str, stop_event: threading.Event):
        try:
            deadline = time.monotonic() + 5.0
            while not os.path.exists(log_path) and time.monotonic() < deadline:
                if stop_event.is_set():
                    return
                time.sleep(0.1)
            with open(log_path, 'r', errors='replace') as f:
                while not stop_event.is_set():
                    line = f.readline()
                    if not line:
                        time.sleep(0.2)
                        continue
                    source = self._parse_planning_occupancy_source_log(line)
                    if source is not None:
                        with self._lock:
                            self._planning_occupancy_source = source
        except Exception as exc:
            self.get_logger().warning(f'Planning log monitor stopped: {exc}')

    @staticmethod
    def _parse_planning_occupancy_source_log(line: str) -> str | None:
        match = re.search(r'Planning occupancy_source current:\s*(depth|lidar)', line)
        if match:
            return match.group(1)
        match = re.search(r'Updated planning occupancy_source:\s*(depth|lidar)\s*->\s*(depth|lidar)', line)
        if match:
            return match.group(2)
        return None

    def cmd_set_planning_occupancy_source(self, source: str):
        if source not in ('depth', 'lidar'):
            raise ValueError(f"Invalid planning occupancy_source: {source!r}")
        with self._lock:
            self._planning_occupancy_source = source
        self._planning_config_pub.publish(String(data=json.dumps({'occupancy_source': source})))
        self.get_logger().info(f'Planning occupancy_source requested from frontend: {source}')
        return source

    def cmd_set_odom_source(self, source: str):
        if source not in ('vio', 'ekf'):
            raise ValueError(f"Invalid odom_source: {source!r}")
        with self._lock:
            self._odom_source = source
        self._publish_odom_source_config(source)
        self.get_logger().info(f'Localization odom_source requested from frontend: {source}')
        return source

    def _publish_odom_source_config(self, source: str | None = None) -> str:
        with self._lock:
            effective = self._odom_source if source is None else source
        if effective not in ('vio', 'ekf'):
            effective = 'vio'
        self._localization_config_pub.publish(String(data=json.dumps({'odom_source': effective})))
        return effective

    def _sync_odom_source_to_nodes(self) -> None:
        source = self._publish_odom_source_config()
        self.get_logger().info(f'Re-applied localization odom_source to map/planning nodes: {source}')

    def _looper_bridge_cmd(self) -> list[str]:
        return [
            'uv', 'run', 'python', '/tinynav/tool/looper_bridge_node.py',
            '--sync-watchdog-s', os.environ.get('TINYNAV_LOOPER_SYNC_WATCHDOG_S', '12'),
            '--sync-watchdog-grace-s', os.environ.get('TINYNAV_LOOPER_SYNC_WATCHDOG_GRACE_S', '45'),
        ]

    def _launch_looper_bridge(self, env: dict) -> subprocess.Popen:
        return self._launch_proc('looper_bridge', self._looper_bridge_cmd(), env=env)

    def _supervise_looper_bridge(self):
        if self._sensor_mode != 'looper' or not self._looper_bridge_want_running:
            return
        proc = self._looper_bridge_proc
        if proc is not None and proc.poll() is None:
            return
        now = time.monotonic()
        if proc is not None and proc.poll() is not None:
            self.get_logger().warning(
                f'looper_bridge exited (code={proc.returncode}); supervisor will restart'
            )
            self._looper_bridge_proc = None
        if self._looper_bridge_proc is not None:
            return
        if now - self._looper_bridge_last_restart_mono < self._looper_bridge_restart_cooldown_s:
            return
        self._looper_bridge_last_restart_mono = now
        try:
            cyclone_env = self._planning_env(os.environ.copy())
            self._looper_bridge_proc = self._launch_looper_bridge(cyclone_env)
            self.get_logger().info('looper_bridge restarted by supervisor')
        except Exception as exc:
            self.get_logger().error(f'looper_bridge restart failed: {exc}')

    def _stop_sensor_procs(self):
        self._planning_log_stop_event.set()
        with self._lock:
            self._planning_occupancy_source = 'depth'
        self._looper_bridge_want_running = False
        for attr in ('_looper_bridge_proc', '_realsense_proc', '_perception_proc', '_imu_propagation_proc', '_planning_proc'):
            self._kill_proc(getattr(self, attr))
            setattr(self, attr, None)

    def _planning_env(self, env: dict) -> dict:
        """env for planning_node, matching the RMW the current _sensor_mode needs.

        In 'looper' mode planning_node subscribes to the Hesai lidar driver on a
        separate host -- cross-vendor RTPS with FastDDS's default multi-interface
        locator advertisement doesn't reliably reach it on this robot's network.
        Every planning_node launch site (initial start, restart-after-map-build,
        emergency-stop restart) must use this so the CycloneDDS override doesn't
        silently drop out on one of the paths (see cyclonedds_jetson.xml).
        """
        if self._sensor_mode == 'looper':
            cyclone_env = env.copy()
            cyclone_env['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'
            cyclone_env['CYCLONEDDS_URI'] = '/tinynav/cyclonedds_jetson.xml'
            return cyclone_env
        return env

    def _sync_looper_time(self):
        """Sync the Looper module clock before starting Looper-dependent nodes.

        If this fails, do not continue to start looper_bridge/planning_node;
        stale Looper timestamps can poison VIO/ROS message timing.
        """
        max_attempts = 3
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                subprocess.run(
                    ['sshpass', '-p', 'looper@0731', 'python3',
                     '/tinynav/looper_cli/looper_cli.py', 'time', 'sync', '-y'],
                    check=True, timeout=30,
                )
                self.get_logger().info(f'Looper time sync completed on attempt {attempt}/{max_attempts}')
                return
            except Exception as e:
                last_error = e
                self.get_logger().error(f'Looper time sync attempt {attempt}/{max_attempts} failed: {e}')
                if attempt < max_attempts:
                    time.sleep(1.0)
        raise RuntimeError(f'Looper time sync failed after {max_attempts} attempts: {last_error}')

    def _launch_sensor_procs(self, env: dict):
        """Start sensor procs based on current _sensor_mode."""
        if self._sensor_mode == 'looper':
            self._sync_looper_time()
            cyclone_env = self._planning_env(env)
            self._looper_bridge_want_running = True
            self._looper_bridge_last_restart_mono = time.monotonic()
            self._looper_bridge_proc = self._launch_looper_bridge(cyclone_env)
            self._planning_proc = self._launch_proc(
                'planning',
                ['uv', 'run', 'python', '/tinynav/tinynav/core/planning_node.py'],
                env=cyclone_env,
            )
        elif self._sensor_mode == 'realsense':
            self._realsense_proc = self._launch_proc(
                'realsense',
                ['bash', _REALSENSE_SCRIPT],
            )
            self._perception_proc = self._launch_proc(
                'perception',
                ['uv', 'run', 'python', '/tinynav/tinynav/core/perception_node.py'],
                env=env,
            )
            self._imu_propagation_proc = self._launch_proc(
                'perception',
                ['uv', 'run', 'python', '/tinynav/tinynav/core/imu_propagator_node.py'],
                env=env,
            )
            self._planning_proc = self._launch_proc(
                'planning',
                ['uv', 'run', 'python', '/tinynav/tinynav/core/planning_node.py'],
                env=env,
            )

    def _restart_sensor_procs(self):
        _env = os.environ.copy()
        _env['PYTHONPATH'] = _VENV_SITE + ':' + _env.get('PYTHONPATH', '')
        self._launch_sensor_procs(_env)
        self.get_logger().info('Sensor procs restarted after map build')

    # ------------------------------------------------------------------ #
    # Nav nodes toggle                                                     #
    # ------------------------------------------------------------------ #

    def cmd_start_nav_nodes(self, initial_map_to_odom_transform_path: str | None = None):
        self._set_nav_active(False)
        _env = os.environ.copy()
        _env['PYTHONPATH'] = _VENV_SITE + ':' + _env.get('PYTHONPATH', '')
        map_node_cmd = [
            'uv', 'run', 'python', '/tinynav/tinynav/core/map_node.py',
            '--tinynav_map_path', self.map_path,
        ]
        if self._load_nav_flow_enable_first_done():
            map_node_cmd.append('--enable_first_done')
        if initial_map_to_odom_transform_path is not None:
            map_node_cmd += ['--initial_map_to_odom_transform', initial_map_to_odom_transform_path]
        # Decide RTK mode once here, matching MapNode which also decides once at
        # its own startup -- neither side re-checks this while nav is running.
        self._nav_rtk_mode = self._load_nav_flow_rtk_mode()
        self._publish_current_map_for_rtk()
        self._map_node_proc = self._launch_proc(
            'map_node',
            map_node_cmd,
            env=_env,
        )
        rtk_mode = self._nav_rtk_mode
        with self._lock:
            loc_assist_requested = self._loc_assist_enabled
        # RTK-enabled maps still use visual relocalization while RTK is not ACTIVE.
        # If RTK later reaches NEED_YAW_INIT, the yaw-init loop will stop assist and
        # take direct /cmd_vel ownership before RTK replaces the map transform.
        loc_assist = loc_assist_requested
        if loc_assist:
            # Don't start cmd_vel_control yet; start localization assist sweep
            self._start_loc_assist(_env)
        else:
            self._cmd_vel_proc = self._launch_proc(
                'cmd_vel_control',
                ['uv', 'run', 'python', '/tinynav/tinynav/platforms/cmd_vel_control.py'],
                env=_env,
            )
        with self._lock:
            self._nav_nodes_running = True
        self._sync_odom_source_to_nodes()
        self.get_logger().info('Nav nodes started')

    def cmd_stop_nav_nodes(self):
        self._set_nav_active(False)
        self._stop_rtk_yaw_init()
        self._current_map_pub.publish(String(data=''))
        self._stop_loc_assist()
        self._kill_proc(self._map_node_proc)
        self._kill_proc(self._cmd_vel_proc)
        self._map_node_proc = None
        self._cmd_vel_proc = None
        with self._lock:
            self._nav_nodes_running = False
            self._localized = False
            self._map_pose = None
            self._global_path = []
            self._final_global_path = []
            self._nav_target_pose = None
            self._nav_paused = False
        self.get_logger().info('Nav nodes stopped')

    def cmd_restart_nav_nodes(self):
        self._set_nav_active(False)
        self._stop_rtk_yaw_init()
        self._stop_loc_assist()
        self._kill_proc(self._map_node_proc)
        self._kill_proc(self._planning_proc)
        self._kill_proc(self._cmd_vel_proc)
        self._map_node_proc = None
        self._planning_proc = None
        self._cmd_vel_proc = None

        _env = os.environ.copy()
        _env['PYTHONPATH'] = _VENV_SITE + ':' + _env.get('PYTHONPATH', '')

        self._planning_proc = self._launch_proc(
            'planning',
            ['uv', 'run', 'python', '/tinynav/tinynav/core/planning_node.py'],
            env=self._planning_env(_env),
        )
        self._nav_rtk_mode = self._load_nav_flow_rtk_mode()
        self._publish_current_map_for_rtk()
        map_node_cmd = [
            'uv', 'run', 'python', '/tinynav/tinynav/core/map_node.py',
            '--tinynav_map_path', self.map_path,
        ]
        if self._load_nav_flow_enable_first_done():
            map_node_cmd.append('--enable_first_done')
        self._map_node_proc = self._launch_proc(
            'map_node',
            map_node_cmd,
            env=_env,
        )
        self._cmd_vel_proc = self._launch_proc(
            'cmd_vel_control',
            ['uv', 'run', 'python', '/tinynav/tinynav/platforms/cmd_vel_control.py'],
            env=_env,
        )
        with self._lock:
            self._nav_nodes_running = True
            self._localized = False
            self._map_pose = None
            self._global_path = []
            self._final_global_path = []
            self._nav_target_pose = None
        self._sync_odom_source_to_nodes()
        self.state = 'idle'
        self._pub_state()
        self.get_logger().info('Nav nodes restarted (emergency stop)')

    # ------------------------------------------------------------------ #
    # Localization assist: yaw sweep until localized                        #
    # ------------------------------------------------------------------ #

    def cmd_set_loc_assist(self, enabled: bool):
        """Enable or disable the auto-localization assist toggle."""
        with self._lock:
            self._loc_assist_enabled = enabled
        self.get_logger().info(f'Localization assist {"enabled" if enabled else "disabled"}')

    def _start_loc_assist(self, env: dict):
        """Start the yaw sweep thread (no cmd_vel_control process)."""
        if self._loc_assist_thread is not None and self._loc_assist_thread.is_alive():
            self.get_logger().info('Localization assist sweep already running')
            return
        self._loc_assist_stop_event.clear()
        self._loc_assist_thread = threading.Thread(
            target=self._loc_assist_loop, daemon=True
        )
        self._loc_assist_thread.start()
        self.get_logger().info('Localization assist sweep started')

    def _stop_loc_assist(self):
        """Stop the yaw sweep thread if running, publish zero cmd_vel."""
        self._loc_assist_stop_event.set()
        if self._loc_assist_thread is not None and self._loc_assist_thread is not threading.current_thread():
            self._loc_assist_thread.join(timeout=6.0)
            self._loc_assist_thread = None
        # Ensure robot stops
        self._publish_cmd_vel(0.0, 0.0)

    def _publish_cmd_vel(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(msg)

    def _loc_assist_loop(self):
        """
        Fixed yaw assist pattern:
        - Face the initial direction, wait dwell_s
        - Turn left 10° from the initial direction, wait dwell_s
        - Return to the initial direction, wait dwell_s
        - Turn right 10° from the initial direction, wait dwell_s
        - Return to the initial direction, wait dwell_s
        - Repeat until localized

        The turn amount is closed-loop against SLAM odometry yaw. While turning,
        publish cmd_vel continuously so downstream controllers do not need to
        latch a single Twist command.
        """
        dwell_s = 5.0
        angular_speed = 0.4  # rad/s
        cmd_rate_hz = 10.0
        yaw_tolerance = math.radians(2.0)
        side_yaw_rad = math.radians(10.0)
        stop = self._loc_assist_stop_event

        interval = 1.0 / max(cmd_rate_hz, 1.0)
        start_wait = time.monotonic()
        initial_yaw = self._latest_odom_yaw()
        while initial_yaw is None:
            if self._should_stop_loc_assist(stop):
                return
            self._publish_cmd_vel(0.0, 0.0)
            if time.monotonic() - start_wait > 5.0:
                self.get_logger().warn('Localization assist waiting for fresh odometry yaw')
                start_wait = time.monotonic()
            time.sleep(interval)
            initial_yaw = self._latest_odom_yaw()

        yaw_offsets = [0.0, side_yaw_rad, 0.0, -side_yaw_rad, 0.0]

        while not stop.is_set():
            for yaw_offset in yaw_offsets:
                if abs(yaw_offset) > 1e-6:
                    self.get_logger().info(
                        f'Localization assist target yaw offset: {math.degrees(yaw_offset):.1f} deg'
                    )
                if self._turn_to_odom_yaw(
                    target_yaw=self._wrap_angle(initial_yaw + yaw_offset),
                    angular_speed=angular_speed,
                    cmd_rate_hz=cmd_rate_hz,
                    yaw_tolerance=yaw_tolerance,
                    stop=stop,
                ):
                    return
                if self._wait_or_localized(dwell_s, stop):
                    return

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap an angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def _latest_odom_yaw(self, max_age_s: float = 1.0) -> float | None:
        with self._lock:
            pose = self._odom_pose
            received_at = self._odom_pose_received_at
        if pose is None or received_at is None:
            return None
        if time.monotonic() - received_at > max_age_s:
            return None
        yaw = pose.get('yaw')
        return float(yaw) if yaw is not None else None

    def _turn_relative_by_odom(
        self,
        target_delta: float,
        angular_speed: float,
        cmd_rate_hz: float,
        yaw_tolerance: float,
        stop: threading.Event,
    ) -> bool:
        """
        Turn until odometry yaw reaches target_delta relative to the turn start.
        Returns True if the assist loop should stop (localized or stop event set).
        """
        interval = 1.0 / max(cmd_rate_hz, 1.0)
        max_duration = abs(target_delta) / max(angular_speed, 1e-3) + 3.0

        start_wait = time.monotonic()
        start_yaw = self._latest_odom_yaw()
        while start_yaw is None:
            if self._should_stop_loc_assist(stop):
                return True
            # Do not blind-turn without fresh odometry.
            self._publish_cmd_vel(0.0, 0.0)
            if time.monotonic() - start_wait > 5.0:
                self.get_logger().warn('Localization assist waiting for fresh odometry yaw')
                start_wait = time.monotonic()
            time.sleep(interval)
            start_yaw = self._latest_odom_yaw()

        angular_z = math.copysign(abs(angular_speed), target_delta)
        start_time = time.monotonic()
        previous_yaw = start_yaw
        accumulated_delta = 0.0

        while True:
            if self._should_stop_loc_assist(stop):
                return True

            current_yaw = self._latest_odom_yaw()
            if current_yaw is None:
                # Odometry disappeared; stop rather than continuing open-loop.
                self._publish_cmd_vel(0.0, 0.0)
                time.sleep(interval)
                continue

            accumulated_delta += self._wrap_angle(current_yaw - previous_yaw)
            previous_yaw = current_yaw
            remaining = target_delta - accumulated_delta
            if abs(remaining) <= yaw_tolerance:
                self._publish_cmd_vel(0.0, 0.0)
                return False

            # If we overshot, stop this segment instead of commanding a reverse
            # correction sweep. The next sweep segment will continue the pattern.
            if math.copysign(1.0, remaining) != math.copysign(1.0, target_delta):
                self._publish_cmd_vel(0.0, 0.0)
                return False

            if time.monotonic() - start_time > max_duration:
                self.get_logger().warn(
                    f'Localization assist turn timeout: target_delta={target_delta:.3f} '
                    f'accumulated_delta={accumulated_delta:.3f} remaining={remaining:.3f}'
                )
                self._publish_cmd_vel(0.0, 0.0)
                return False

            self._publish_cmd_vel(0.0, angular_z)
            time.sleep(interval)

    def _turn_to_odom_yaw(
        self,
        target_yaw: float,
        angular_speed: float,
        cmd_rate_hz: float,
        yaw_tolerance: float,
        stop: threading.Event,
    ) -> bool:
        """
        Turn to an absolute odometry yaw. Returns True if the assist loop should
        stop because localization succeeded or the stop event was set.
        """
        interval = 1.0 / max(cmd_rate_hz, 1.0)
        start_time = time.monotonic()
        max_duration = math.pi / max(angular_speed, 1e-3) + 3.0

        while True:
            if self._should_stop_loc_assist(stop):
                return True

            current_yaw = self._latest_odom_yaw()
            if current_yaw is None:
                self._publish_cmd_vel(0.0, 0.0)
                time.sleep(interval)
                continue

            error = self._wrap_angle(target_yaw - current_yaw)
            if abs(error) <= yaw_tolerance:
                self._publish_cmd_vel(0.0, 0.0)
                return False

            if time.monotonic() - start_time > max_duration:
                self.get_logger().warn(
                    f'Localization assist turn timeout: target_yaw={target_yaw:.3f} '
                    f'current_yaw={current_yaw:.3f} error={error:.3f}'
                )
                self._publish_cmd_vel(0.0, 0.0)
                return False

            angular_z = math.copysign(abs(angular_speed), error)
            self._publish_cmd_vel(0.0, angular_z)
            time.sleep(interval)

    def _should_stop_loc_assist(self, stop: threading.Event) -> bool:
        if stop.is_set():
            self._publish_cmd_vel(0.0, 0.0)
            return True
        with self._lock:
            localized = self._localized
        if localized:
            self._publish_cmd_vel(0.0, 0.0)
            return True
        return False

    def _wait_or_localized(self, duration: float, stop: threading.Event) -> bool:
        """
        Wait for `duration` seconds, checking localization and stop event
        every 0.1s. Returns True if should stop (localized or event set).
        """
        elapsed = 0.0
        interval = 0.1
        while elapsed < duration:
            if self._should_stop_loc_assist(stop):
                return True
            time.sleep(interval)
            elapsed += interval
        return False

    def _on_localization_achieved(self):
        """
        Called when localization succeeds for the first time.
        Stops the assist sweep and launches cmd_vel_control.
        """
        with self._lock:
            loc_assist = self._loc_assist_enabled
            nav_running = self._nav_nodes_running
            cmd_vel_proc = self._cmd_vel_proc
            if cmd_vel_proc is not None and cmd_vel_proc.poll() is None:
                already_running = True
            else:
                already_running = False
                if cmd_vel_proc is not None:
                    self._cmd_vel_proc = None
        if not nav_running:
            self._resume_vio_pois_after_localized()
            return
        if already_running:
            self._resume_vio_pois_after_localized()
            return
        if not loc_assist:
            self._resume_vio_pois_after_localized()
            return
        # Stop the sweep
        self._stop_loc_assist()
        # Now start cmd_vel_control. Re-check under the lock because both
        # /mapping/current_pose_in_map and /map/relocalization can report the
        # first successful localization close together.
        _env = os.environ.copy()
        _env['PYTHONPATH'] = _VENV_SITE + ':' + _env.get('PYTHONPATH', '')
        with self._lock:
            cmd_vel_proc = self._cmd_vel_proc
            if cmd_vel_proc is not None and cmd_vel_proc.poll() is None:
                return
            self._cmd_vel_proc = self._launch_proc(
                'cmd_vel_control',
                ['uv', 'run', 'python', '/tinynav/tinynav/platforms/cmd_vel_control.py'],
                env=_env,
            )
        self.get_logger().info('Localization achieved — cmd_vel_control started')
        self._resume_vio_pois_after_localized()

    def cmd_bag_start(self):
        if self._sensor_mode == 'looper':
            self._stop_sensor_procs()
        self._stop_all()
        self._start('realsense_bag_record')

    def cmd_bag_stop(self):
        if self.state == 'realsense_bag_record':
            bag_path = self.bag_path
            self._stop_all()
            if self._sensor_mode == 'looper':
                threading.Thread(
                    target=lambda bp: (self._finalize_bag(bp), self._restart_sensor_procs()),
                    args=(bag_path,), daemon=True,
                ).start()
            else:
                threading.Thread(target=self._finalize_bag, args=(bag_path,), daemon=True).start()

    # ── Debug recording (runs alongside navigation, independent state) ── #

    _DEBUG_RECORD_TOPICS = [
        '/camera/camera/imu',
	'/camera/camera/infra2/image_rect_raw',
	'/camera/camera/infra2/camera_info',
        '/camera/camera/infra1/image_rect_raw',
        '/camera/camera/depth/image_rect_raw',
        '/camera/camera/infra1/camera_info',
        '/camera/camera/vio_100hz',
        '/camera/camera/vio_image',
        '/tf_static',
        '/slam/odometry_visual',
        '/slam/depth',
        '/mapping/global_plan',
        '/control/target_pose',
        '/planning/trajectory_path',
        '/planning/occupied_voxels',
        '/planning/footprint',
        '/lidar/points',
    ]

    def cmd_debug_record_start(self):
        """Start a debug rosbag recording (independent of main state machine)."""
        with self._lock:
            if self._debug_record_proc is not None and self._debug_record_proc.poll() is None:
                return  # already recording
            from datetime import datetime
            debug_bags_dir = os.path.join(self.tinynav_db_path, 'debug_bags')
            os.makedirs(debug_bags_dir, exist_ok=True)
            ts = datetime.now().strftime('debug_%Y_%m_%d_%H_%M_%S')
            output_dir = os.path.join(debug_bags_dir, ts)
            cmd = (
                ['ros2', 'bag', 'record',
                 '--output', output_dir,
                 '--max-cache-size', '2147483648']
                + self._DEBUG_RECORD_TOPICS
            )
            # /lidar/points is published by the Hesai driver under CycloneDDS on a
            # separate host (see _launch_sensor_procs); this recorder needs the same
            # RMW to actually reach it.
            self._debug_record_proc = self._spawn(cmd, extra_env={
                'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp',
                'CYCLONEDDS_URI': '/tinynav/cyclonedds_jetson.xml',
            })
            self._debug_record_path = output_dir
            self.get_logger().info(f'Debug recording started → {output_dir}')

    def cmd_debug_record_stop(self):
        """Stop the debug rosbag recording."""
        with self._lock:
            proc = self._debug_record_proc
            self._debug_record_proc = None
            path = self._debug_record_path
            self._debug_record_path = None
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), 15)
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self.get_logger().info(f'Debug recording stopped → {path}')

    @property
    def debug_recording(self) -> bool:
        with self._lock:
            return self._debug_record_proc is not None and self._debug_record_proc.poll() is None

    @property
    def debug_record_path(self) -> str | None:
        with self._lock:
            return self._debug_record_path


    def _finalize_bag(self, bag_path: str):
        import shutil
        from datetime import datetime
        time.sleep(1.5)  # wait for ros2 bag to flush
        if not os.path.isdir(bag_path):
            return
        try:
            result = subprocess.run(
                ['ros2', 'bag', 'info', bag_path],
                capture_output=True,
                timeout=30,
                env={**os.environ},
            )
            if result.returncode != 0:
                return  # bag corrupted — leave in place
            output = result.stdout.decode('utf-8', errors='replace')
            match = re.search(r'Messages:\s+(\d+)', output)
            if not match or int(match.group(1)) == 0:
                return  # empty bag — leave in place
        except Exception:
            return
        rosbags_dir = os.path.join(os.path.dirname(bag_path), 'rosbags')
        os.makedirs(rosbags_dir, exist_ok=True)
        ts = datetime.now().strftime('bag_%Y_%m_%d_%H_%M_%S')
        dest = os.path.join(rosbags_dir, ts)
        shutil.move(bag_path, dest)
        with self._lock:
            self._last_verified_bag = dest

    def _start_rosbag_build_map(self):
        """Override to use the last verified bag instead of the default bag_path."""
        active = self.active_bag_path
        if active is None:
            self.get_logger().warn('No verified bag available for map building')
            return
        bag_file = os.path.join(active, 'bag_0.db3')
        if not os.path.exists(bag_file):
            self.get_logger().warn(f'bag_0.db3 not found in {active}')
            return
        # Remove existing map path so build_map_node creates a fresh real directory.
        # If map_path is a symlink, shutil.move would rename the symlink (not the target),
        # and build_map_node would write through the symlink into the old map directory.
        import shutil as _shutil
        if os.path.islink(self.map_path) or os.path.isfile(self.map_path):
            os.remove(self.map_path)
        elif os.path.isdir(self.map_path):
            _shutil.rmtree(self.map_path)

        _env = os.environ.copy()
        if self._sensor_mode == 'looper':
            _env['ROS_DOMAIN_ID'] = _MAP_BUILD_DOMAIN_LOOPER
        _env['PYTHONPATH'] = _VENV_SITE + ':' + _env.get('PYTHONPATH', '')
        if self._sensor_mode == 'looper':
            source_name = 'looper_bridge'
            source_cmd = ['uv', 'run', 'python', '/tinynav/tool/looper_bridge_node.py']
        else:
            source_name = 'perception'
            source_cmd = ['uv', 'run', 'python', '/tinynav/tinynav/core/perception_node.py']

        self.processes[source_name] = self._launch_proc(
            source_name,
            source_cmd,
            env=_env,
        )
        self.processes['build_map'] = self._launch_proc_tee(
            'build_map_node',
            [
                'uv', 'run', 'python', '/tinynav/tinynav/core/build_map_node.py',
                '--map_save_path', self.map_path,
                '--bag_file', bag_file,
            ],
            env=_env,
        )

        threading.Thread(target=self._on_build_map_done, daemon=True).start()

    def _launch_proc_tee(self, name: str, cmd: list[str], env: dict | None = None,
                          cwd: str = '/tinynav') -> subprocess.Popen:
        """Like _launch_proc, but also tees stdout to a pipe so the caller can
        scan for MAPPING_PERCENT: lines while still logging everything to file."""
        lf = self._make_log(name)
        proc = subprocess.Popen(
            cmd, preexec_fn=os.setsid, cwd=cwd,
            env=env or os.environ.copy(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        threading.Thread(
            target=self._tee_and_read_percent,
            args=(proc, lf),
            daemon=True,
        ).start()
        return proc

    def _tee_and_read_percent(self, proc: subprocess.Popen, log_file):
        """Read lines from proc.stdout, write to log_file, and extract
        MAPPING_PERCENT:<float> values into self.mapping_percent."""
        try:
            for raw in proc.stdout:
                line = raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else raw
                log_file.write(line)
                log_file.flush()
                if _MAPPING_PERCENT_PREFIX in line:
                    try:
                        pct = float(line.split(_MAPPING_PERCENT_PREFIX, 1)[1].strip())
                        with self._lock:
                            self.mapping_percent = pct
                    except (ValueError, AttributeError):
                        pass
        finally:
            log_file.close()

    def _on_build_map_done(self):
        """Wait for build_map to finish, then convert, archive, and restart."""
        import shutil
        from datetime import datetime
        proc_build = self.processes.get('build_map')
        if proc_build:
            proc_build.wait()
        subprocess.run([
            'uv', 'run', 'python', '/tinynav/tool/convert_to_colmap_format.py',
            '--input_dir', self.map_path,
            '--output_dir', self.map_path,
        ])
        # mv map → maps/map_YYYY_MM_DD_HH_MM_SS, symlink back
        maps_dir = os.path.join(self.tinynav_db_path, 'maps')
        os.makedirs(maps_dir, exist_ok=True)
        ts = datetime.now().strftime('map_%Y_%m_%d_%H_%M_%S')
        dest = os.path.join(maps_dir, ts)
        shutil.move(self.map_path, dest)
        os.symlink(dest, self.map_path)

        # Auto-create a home POI at the SLAM origin (0,0,0) if none exist.
        # map_node requires at least one POI as a global localization anchor.
        pois_path = os.path.join(dest, 'pois.json')
        if not os.path.exists(pois_path):
            with open(pois_path, 'w') as _f:
                json.dump(
                    {'0': {'id': 0, 'name': 'home', 'position': [0.0, 0.0, 0.0]}},
                    _f, indent=2,
                )
            self.get_logger().info('Auto-created home POI at (0,0,0)')

        self._stop_all()
        self.state = 'idle'
        self._pub_state()
        self._restart_sensor_procs()


    def cmd_map_build(self):
        self._stop_sensor_procs()
        self._stop_all()
        self._start('rosbag_build_map')

    def _publish_cmd_pois(self, poi_id: int | None) -> bool:
        """Publish the selected POI to map_node as JSON on /mapping/cmd_pois.
        Sending an empty dict clears the current nav target. Returns whether a
        non-empty navigation target was published."""
        if poi_id is None:
            with self._lock:
                self._active_nav_poi_ids = []
                self._active_nav_pois = []
            self._cmd_pois_pub.publish(String(data='{}'))
            return False
        pois_file = os.path.join(self.map_path, 'pois.json')
        if not os.path.exists(pois_file):
            self.get_logger().warn('No pois.json found, cannot publish cmd_pois')
            return False
        with open(pois_file) as f:
            pois = json.load(f)
        key = str(poi_id)
        if key not in pois:
            self.get_logger().warn(f'POI {poi_id} not found in pois.json')
            return False
        # Re-index as "0" to match pub_pois.py convention expected by map_node
        payload = {'0': pois[key]}
        with self._lock:
            self._active_nav_pois = list(payload.values())
        self._cmd_pois_pub.publish(String(data=json.dumps(payload)))
        return True

    def cmd_manual_target_pose(self, x: float, y: float, z: float):
        """Publish a manually selected local-planner target pose.

        planning_node subscribes to /control/target_pose and only reads the
        position vector, so Odometry is used here to match that existing API.
        """
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = float(z)
        msg.pose.pose.orientation.w = 1.0
        self._target_pose_pub.publish(msg)
        with self._lock:
            self._nav_target_pose = {'x': float(x), 'y': float(y)}

    def cmd_send_pois(self, poi_ids: list[int | str]):
        """Publish selected POIs to map_node and transition to navigation state.

        Items may be integer POI IDs or POI names. The payload is re-indexed as
        a dense queue while preserving each POI's original id/name metadata.
        """
        with self._lock:
            self._active_nav_poi_ids = list(poi_ids)
            self._nav_progress = None
        if not poi_ids:
            with self._lock:
                self._active_nav_pois = []
            self._cmd_pois_pub.publish(String(data='{}'))
            self._set_nav_active(False)
        else:
            # New navigation session – allow map handoffs to trigger again.
            with self._lock:
                self._handled_map_handoffs.clear()
            pois_file = os.path.join(self.map_path, 'pois.json')
            if not os.path.exists(pois_file):
                self.get_logger().warn('No pois.json found, cannot publish cmd_pois')
                return
            with open(pois_file) as f:
                all_pois = json.load(f)
            pois_by_name = {
                poi.get('name'): poi
                for poi in all_pois.values()
                if isinstance(poi, dict) and isinstance(poi.get('name'), str)
            }
            # Re-index as a dense queue ("0", "1", ...) so downstream
            # consumers navigate in the same order the UI/nav_flow sent POIs,
            # instead of falling back to the original ids / pois.json order.
            payload = {}
            for poi_ref in poi_ids:
                poi = None
                if isinstance(poi_ref, int):
                    poi = all_pois.get(str(poi_ref))
                elif isinstance(poi_ref, str):
                    poi = pois_by_name.get(poi_ref)
                    if poi is None and poi_ref.isdigit():
                        poi = all_pois.get(poi_ref)
                if poi is not None:
                    payload[str(len(payload))] = poi
                else:
                    self.get_logger().warn(f'POI {poi_ref!r} not found in active map')
            with self._lock:
                self._active_nav_pois = list(payload.values())
            self._cmd_pois_pub.publish(String(data=json.dumps(payload)))
            self._set_nav_active(bool(payload))
        with self._lock:
            nav_running = self._nav_nodes_running
        if nav_running:
            self.state = 'navigation'
            self._pub_state()
        else:
            self._stop_all()
            self._start('navigation')

    def cmd_nav_start(self, poi_id: str | None = None):
        if poi_id is not None:
            # New navigation session – allow map handoffs to trigger again.
            with self._lock:
                self._handled_map_handoffs.clear()
            with self._lock:
                self._active_nav_poi_ids = [int(poi_id)]
                self._nav_progress = None
            self._set_nav_active(self._publish_cmd_pois(int(poi_id)))
        else:
            with self._lock:
                self._active_nav_poi_ids = []
                self._active_nav_pois = []
            self._set_nav_active(False)
        with self._lock:
            nav_running = self._nav_nodes_running
        if nav_running:
            # Nav nodes already running — just send the target, don't spawn duplicates.
            self.state = 'navigation'
            self._pub_state()
        else:
            self._stop_all()
            self._start('navigation')

    def cmd_nav_cancel(self):
        if self.state != 'navigation':
            return
        self._stop_rtk_yaw_init()
        with self._lock:
            self._active_nav_poi_ids = []
            self._active_nav_pois = []
            self._vio_resume_poi_ids = []
            self._vio_guard_stopped = False
            self._vio_guard_recovering = False
            nav_running = self._nav_nodes_running
        if nav_running:
            # Clear the active nav target so map_node stops pathing.
            self._publish_cmd_pois(None)
            self._set_nav_active(False)
            self.state = 'idle'
            self._pub_state()
        else:
            self._stop_all()

    def cmd_nav_pause(self):
        with self._lock:
            self._nav_paused = True
        self._pause_pub.publish(Bool(data=True))

    def cmd_nav_resume(self):
        with self._lock:
            self._nav_paused = False
        self._pause_pub.publish(Bool(data=False))

    def cmd_action(self, action: str):
        self._action_pub.publish(String(data=f'play {action}'))

    def publish_cmd_vel(self, linear_x: float, linear_y: float, angular_z: float):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(msg)


class NodeRunner:
    """Manages the rclpy lifecycle; spins BackendNode in a daemon thread."""

    def __init__(self, tinynav_db_path: str = '/tinynav/tinynav_db'):
        self._db_path = tinynav_db_path
        self.node: BackendNode | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name='rclpy-spin')
        self._thread.start()
        # Looper time sync in BackendNode.__init__ can take 15–30s (20 RTT samples).
        if not self._ready.wait(timeout=60.0):
            raise RuntimeError('rclpy node did not start in time')

    def _run(self):
        rclpy.init()
        self.node = BackendNode(tinynav_db_path=self._db_path)
        self._ready.set()
        try:
            rclpy.spin(self.node)
        except Exception:
            pass
        finally:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            try:
                rclpy.shutdown()
            except Exception:
                pass

    def stop(self):
        if self.node:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            for proc in (self.node._looper_bridge_proc, self.node._realsense_proc, self.node._perception_proc, self.node._imu_propagation_proc, self.node._planning_proc, self.node._unitree_proc, self.node._map_node_proc, self.node._cmd_vel_proc):
                if proc and proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), 15)
                        proc.wait(timeout=2)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
