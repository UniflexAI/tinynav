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
from rclpy.qos import DurabilityPolicy, QoSProfile
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

_IMAGE_TOPICS_REALSENSE = [
    _COLOR_TOPIC_REALSENSE,
    '/camera/camera/infra1/image_rect_raw',
    '/camera/camera/infra2/image_rect_raw',
    '/slam/depth',
]
_IMAGE_TOPICS_LOOPER = [
    _COLOR_TOPIC_LOOPER,
    '/camera/camera/infra1/image_rect_raw',
    '/camera/camera/infra2/image_rect_raw',
    '/slam/depth',
]
_IMAGE_TOPICS_ALL = _IMAGE_TOPICS_REALSENSE  # fallback
_PREVIEW_MIN_INTERVAL = 0.2  # 5 fps
_PREVIEW_MAX_EDGE_PX = int(os.environ.get('TINYNAV_PREVIEW_MAX_EDGE_PX', '320'))
_PREVIEW_JPEG_QUALITY = int(os.environ.get('TINYNAV_PREVIEW_JPEG_QUALITY', '50'))
_MAP_HANDOFF_LOCALIZATION_TIMEOUT_S = float(
    os.environ.get('TINYNAV_MAP_HANDOFF_LOCALIZATION_TIMEOUT_S', '0')
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
        self._odom_pose_at_kf: dict | None = None  # odom pose snapshotted at last mapPose update
        self._map_pose: dict | None = None
        self._localized: bool = False
        self._esdf_bytes: bytes = b''
        self._obstacle_bytes: bytes = b''
        self._trajectory: list = []
        self._global_path: list = []
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

        # Debug recording (independent of main state machine)
        self._debug_record_proc: subprocess.Popen | None = None
        self._debug_record_path: str | None = None

        self.create_subscription(Float32, '/mapping/percent', self._on_mapping_percent, 10)
        self.create_subscription(Odometry, '/slam/odometry_visual', self._on_slam_odom, 10)
        self.create_subscription(
            Odometry, '/mapping/current_pose_in_map', self._on_pose_in_map, 10
        )
        # Mark localized as soon as any relocalization succeeds (published unconditionally
        # by map_node, unlike current_pose_in_map which requires POIs to be set).
        self.create_subscription(
            Odometry, '/map/relocalization', self._on_relocalization, 10
        )
        self.create_subscription(Image, '/planning/height_map', self._on_height_map, 1)
        self.create_subscription(
            OccupancyGrid, '/planning/obstacle_mask', self._on_obstacle_mask, 1
        )
        self.create_subscription(Path, '/planning/trajectory_path', self._on_trajectory_path, 1)
        self.create_subscription(Path, '/mapping/global_plan', self._on_global_plan, 1)
        self.create_subscription(
            Odometry, '/control/target_pose', self._on_nav_target_pose, 1
        )
        self.create_subscription(
            PointCloud, '/planning/footprint', self._on_footprint, 1
        )
        self.create_subscription(
            PointCloud2, '/planning/occupied_voxels', self._on_occupied_voxels, 1
        )
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
        self._nav_paused = False
        self._nav_active = False

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
        self._realsense_proc: subprocess.Popen | None = None
        self._perception_proc: subprocess.Popen | None = None
        self._imu_propagation_proc: subprocess.Popen | None = None
        self._planning_proc: subprocess.Popen | None = None
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

        self._nav_active_pub.publish(Bool(data=False))

        self.create_subscription(Float32, '/battery', self._on_battery, 10)
        self.create_subscription(Bool, '/mapping/nav_done', self._on_nav_done, 10)
        self.create_subscription(String, '/mapping/nav_progress', self._on_nav_progress, 10)
        self._detect_and_init_sensor()
        self._start_unitree_if_configured()

    # ------------------------------------------------------------------ #
    # ROS callbacks                                                        #
    # ------------------------------------------------------------------ #

    def _on_battery(self, msg: Float32):
        with self._lock:
            self._battery = float(msg.data)

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
        return {'target_map': target_map, 'poi_list': poi_list, 'zupt': zupt}

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

    def _run_map_handoff(self, source_map: str, poi_index: int, rule: dict):
        target_map = rule['target_map']
        poi_list = rule['poi_list']
        self.get_logger().info(
            f'Map handoff triggered: {source_map}[{poi_index}] -> {target_map}, poi_list={poi_list}'
        )
        try:
            # Stop current map_node/control hard before changing the active map.
            self.cmd_stop_nav_nodes()
            self.state = 'idle'
            self._pub_state()

            self._set_active_map_link(target_map)

            with self._lock:
                self._localized = False
                self._map_pose = None
                self._global_path = []
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

            self.cmd_start_nav_nodes()

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

    def _on_pose_in_map(self, msg: Odometry):
        pose = self._odom_to_dict(msg, source='map')
        with self._lock:
            was_localized = self._localized
            self.current_pose = pose
            self._map_pose = pose
            self._odom_pose_at_kf = self._odom_pose  # freeze odom at this keyframe
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

    def _on_height_map(self, msg: Image):
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

    def _on_footprint(self, msg: PointCloud):
        """Store footprint corner points from PointCloud.

        The planning node publishes 84 points (4 edges × 21 samples per edge).
        We extract the 4 corner points (first of each edge group).
        """
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
        try:
            step = max(1, len(msg.data) // max(1, msg.point_step) // 2500)
            points = []
            import sensor_msgs_py.point_cloud2 as pc2
            for i, p in enumerate(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)):
                if i % step != 0:
                    continue
                points.append({'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])})
                if len(points) >= 2500:
                    break
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
            if '/insight_full' in result.stdout.splitlines():
                self._sensor_mode = 'looper'
                self.get_logger().info('Sensor mode: looper — launching looper bridge + planning')
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

        topics = _IMAGE_TOPICS_LOOPER if self._sensor_mode == 'looper' else _IMAGE_TOPICS_REALSENSE
        for topic in topics:
            self._last_frame[topic] = b''
            self._last_frame_time[topic] = 0.0
            self.preview_callbacks[topic] = []

    def add_preview_callback(
        self,
        topic: str,
        cb,
        max_edge_px: int = _PREVIEW_MAX_EDGE_PX,
        jpeg_quality: int = _PREVIEW_JPEG_QUALITY,
    ) -> bool:
        """Register a frame callback; creates the ROS subscription on the first caller."""
        if topic not in self.preview_callbacks:
            return False
        with self._lock:
            self.preview_callbacks[topic].append((cb, max_edge_px, jpeg_quality))
            first = len(self.preview_callbacks[topic]) == 1
        if first:
            self._create_image_sub(topic)
        return True

    def remove_preview_callback(self, topic: str, cb):
        """Unregister a frame callback; destroys the ROS subscription when the last caller leaves."""
        if topic not in self.preview_callbacks:
            return
        with self._lock:
            self.preview_callbacks[topic] = [
                registration
                for registration in self.preview_callbacks[topic]
                if registration[0] is not cb
            ]
            empty = len(self.preview_callbacks[topic]) == 0
        if empty:
            self._destroy_image_sub(topic)

    def _create_image_sub(self, topic: str):
        if topic in self._image_subs:
            return
        if topic == _COLOR_TOPIC_LOOPER:
            self._image_subs[topic] = self.create_subscription(
                CompressedImage, topic,
                lambda msg, t=topic: self._on_compressed_image(msg, t),
                1,
            )
        else:
            self._image_subs[topic] = self.create_subscription(
                Image, topic,
                lambda msg, t=topic: self._on_image(msg, t),
                1,
            )

    def _destroy_image_sub(self, topic: str):
        sub = self._image_subs.pop(topic, None)
        if sub is not None:
            self.destroy_subscription(sub)

    def _publish_preview_frame(self, topic: str, arr: np.ndarray):
        with self._lock:
            callbacks = list(self.preview_callbacks.get(topic, []))

        encoded_frames: dict[tuple[int, int], bytes] = {}
        for cb, max_edge_px, jpeg_quality in callbacks:
            profile = (max_edge_px, jpeg_quality)
            try:
                frame = encoded_frames.get(profile)
                if frame is None:
                    frame = _encode_preview_jpeg(arr, max_edge_px, jpeg_quality)
                    encoded_frames[profile] = frame
                cb(frame)
            except Exception:
                pass

        if encoded_frames:
            with self._lock:
                self._last_frame[topic] = next(iter(encoded_frames.values()))

    def _on_compressed_image(self, msg: CompressedImage, topic: str):
        now = time.time()
        if now - self._last_frame_time.get(topic, 0.0) < _PREVIEW_MIN_INTERVAL:
            return
        self._last_frame_time[topic] = now

        try:
            arr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                return
            frame = _encode_preview_jpeg(arr)
        except Exception:
            return

        with self._lock:
            self._last_frame[topic] = frame
        for cb in self.preview_callbacks.get(topic, []):
            try:
                cb(frame)
            except Exception:
                pass

    def _on_image(self, msg: Image, topic: str):
        now = time.time()
        if now - self._last_frame_time.get(topic, 0.0) < _PREVIEW_MIN_INTERVAL:
            return
        self._last_frame_time[topic] = now

        try:
            if msg.encoding == '32FC1':
                arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                valid = arr[arr > 0]
                if valid.size > 0:
                    p95 = float(np.percentile(valid, 95))
                    arr = np.clip(arr / (p95 + 1e-6), 0.0, 1.0)
                arr = (arr * 255).astype(np.uint8)
                arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
            else:
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                if arr.shape[2] == 1:
                    arr = arr[:, :, 0]
                elif msg.encoding == 'rgb8':
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            frame = _encode_preview_jpeg(arr)
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
            'debugRecording': self.debug_recording,
            'locAssistEnabled': loc_assist,
            'vioGuardEnabled': vio_guard_enabled,
            'vioStatus': vio_status,
            'vioGuardStopped': vio_guard_stopped,
        self._kill_proc(self._map_node_proc)
        self._kill_proc(self._cmd_vel_proc)
        self._map_node_proc = None
        self._cmd_vel_proc = None
        with self._lock:
            self._nav_nodes_running = False
            self._localized = False
            self._map_pose = None
            self._global_path = []
            self._nav_target_pose = None
            self._nav_paused = False
        self.get_logger().info('Nav nodes stopped')

    def cmd_restart_nav_nodes(self):
        self._set_nav_active(False)
        self._stop_loc_assist()
            self._set_nav_active(self._publish_cmd_pois(int(poi_id)))
        else:
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
        with self._lock:
            self._active_nav_poi_ids = []
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
        if not self._ready.wait(timeout=15.0):
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
