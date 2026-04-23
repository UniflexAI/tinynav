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
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from tool.ros2_node_manager import Ros2NodeManager

_REALSENSE_SCRIPT = '/tinynav/scripts/run_realsense_sensor.sh'
_VENV_SITE = '/tinynav/.venv/lib/python3.10/site-packages'
_MAP_BUILD_DOMAIN_LOOPER = '231'  # isolated domain to avoid live looper topic collision during map build

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
        self._odom_pose_at_kf: dict | None = None  # odom pose snapshotted at last mapPose update
        self._map_pose: dict | None = None
        self._localized: bool = False
        self._esdf_bytes: bytes = b''
        self._obstacle_bytes: bytes = b''
        self._trajectory: list = []
        self._global_path: list = []
        self._grid_info: dict | None = None

        self.create_subscription(Float32, '/mapping/percent', self._on_mapping_percent, 10)
        self.create_subscription(Odometry, '/slam/odometry', self._on_slam_odom, 10)
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

        # Publisher for POI nav target consumed by map_node via /mapping/cmd_pois
        self._cmd_pois_pub = self.create_publisher(String, '/mapping/cmd_pois', 10)

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

        self.create_subscription(Float32, '/battery', self._on_battery, 10)
        self._detect_and_init_sensor()
        self._start_unitree_if_configured()

    # ------------------------------------------------------------------ #
    # ROS callbacks                                                        #
    # ------------------------------------------------------------------ #

    def _on_battery(self, msg: Float32):
        with self._lock:
            self._battery = float(msg.data)

    def _on_mapping_percent(self, msg: Float32):
        with self._lock:
            self.mapping_percent = float(msg.data)

    def _on_slam_odom(self, msg: Odometry):
        pose = self._odom_to_dict(msg, source='slam')
        with self._lock:
            self.current_pose = pose
            self._odom_pose = pose
        for cb in self.pose_callbacks:
            try:
                cb(pose)
            except Exception:
                pass

    def _on_pose_in_map(self, msg: Odometry):
        pose = self._odom_to_dict(msg, source='map')
        with self._lock:
            self.current_pose = pose
            self._map_pose = pose
            self._odom_pose_at_kf = self._odom_pose  # freeze odom at this keyframe
            self._localized = True
        for cb in self.pose_callbacks:
            try:
                cb(pose)
            except Exception:
                pass

    def _on_relocalization(self, msg: Odometry):
        with self._lock:
            self._localized = True

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

    def _transform_path_to_odom(self, path: list, map_pose: dict, odom_pose_at_kf: dict) -> list:
        """Convert map-frame path points to odom frame using exact T_odom_map."""
        if not path or not map_pose or not odom_pose_at_kf:
            return path
        R_kf  = self._quat_to_rot(odom_pose_at_kf['qx'], odom_pose_at_kf['qy'],
                                   odom_pose_at_kf['qz'], odom_pose_at_kf['qw'])
        R_mp  = self._quat_to_rot(map_pose['qx'], map_pose['qy'],
                                   map_pose['qz'], map_pose['qw'])
        R     = R_kf @ R_mp.T                          # rotation: map → odom
        kf_p  = np.array([odom_pose_at_kf['x'], odom_pose_at_kf['y'], odom_pose_at_kf['z']])
        mp_p  = np.array([map_pose['x'],         map_pose['y'],         map_pose['z']])
        t     = kf_p - R @ mp_p                        # translation: map → odom
        result = []
        for pt in path:
            p = R @ np.array([pt['x'], pt['y'], 0.0]) + t
            result.append({'x': float(p[0]), 'y': float(p[1])})
        return result

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
                self.get_logger().info('Sensor mode: looper — launching looper bridge')
                _env = os.environ.copy()
                _env['PYTHONPATH'] = _VENV_SITE + ':' + _env.get('PYTHONPATH', '')
                lf = self._make_log('looper_bridge')
                self._looper_bridge_proc = subprocess.Popen(
                    ['uv', 'run', 'python', '/tinynav/tool/looper_bridge_node.py'],
                    preexec_fn=os.setsid, cwd='/tinynav', env=_env,
                    stdout=lf, stderr=subprocess.STDOUT,
                )
                lf.close()
                lf = self._make_log('planning')
                self._planning_proc = subprocess.Popen(
                    ['uv', 'run', 'python', '/tinynav/tinynav/core/planning_node.py'],
                    preexec_fn=os.setsid, cwd='/tinynav', env=_env,
                    stdout=lf, stderr=subprocess.STDOUT,
                )
                lf.close()
            else:
                self._sensor_mode = 'realsense'
                self.get_logger().info('Sensor mode: realsense — launching driver and perception')
                lf = self._make_log('realsense')
                self._realsense_proc = subprocess.Popen(
                    ['bash', _REALSENSE_SCRIPT], preexec_fn=os.setsid,
                    stdout=lf, stderr=subprocess.STDOUT,
                )
                lf.close()
                _env = os.environ.copy()
                _env['PYTHONPATH'] = _VENV_SITE + ':' + _env.get('PYTHONPATH', '')
                lf = self._make_log('perception')
                self._perception_proc = subprocess.Popen(
                    ['uv', 'run', 'python', '/tinynav/tinynav/core/perception_node.py'],
                    preexec_fn=os.setsid, cwd='/tinynav', env=_env,
                    stdout=lf, stderr=subprocess.STDOUT,
                )
                self._perception_proc = subprocess.Popen(
                    ['uv', 'run', 'python', '/tinynav/tinynav/core/perception_node.py'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                lf.close()
        except Exception as e:
            self.get_logger().warn(f'Sensor detection failed: {e}')
            self._sensor_mode = 'unknown'

        topics = _IMAGE_TOPICS_LOOPER if self._sensor_mode == 'looper' else _IMAGE_TOPICS_REALSENSE
        for topic in topics:
            self._last_frame[topic] = b''
            self._last_frame_time[topic] = 0.0
            self.preview_callbacks[topic] = []

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

    def _on_compressed_image(self, msg: CompressedImage, topic: str):
        now = time.time()
        if now - self._last_frame_time.get(topic, 0.0) < _PREVIEW_MIN_INTERVAL:
            return
        self._last_frame_time[topic] = now
        frame = bytes(msg.data)
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
            _, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame = buf.tobytes()
        except Exception:
            return

        with self._lock:
            self._last_frame[topic] = frame

        for cb in self.preview_callbacks.get(topic, []):
            try:
                cb(frame)
            except Exception:
                pass

    def get_planning_snapshot(self) -> dict:
        with self._lock:
            return {
                'localized': self._localized,
                'odom_pose': self._odom_pose,
                'odom_pose_at_kf': self._odom_pose_at_kf,
                'map_pose': self._map_pose,
                'esdf_image': base64.b64encode(self._esdf_bytes).decode() if self._esdf_bytes else None,
                'obstacle_image': base64.b64encode(self._obstacle_bytes).decode() if self._obstacle_bytes else None,
                'trajectory': list(self._trajectory),
                'global_path': list(self._global_path),
                'grid_info': self._grid_info,
            }

    def _start_unitree_if_configured(self):
        iface = os.environ.get('UNITREE_NETWORK_INTERFACE')
        if not iface:
            return
        _env = os.environ.copy()
        _env['PYTHONPATH'] = _VENV_SITE + ':' + _env.get('PYTHONPATH', '')
        _env['UNITREE_NETWORK_INTERFACE'] = iface
        self._unitree_proc = subprocess.Popen(
            ['uv', 'run', 'python', '/tinynav/tinynav/platforms/unitree_control.py'],
            preexec_fn=os.setsid,
            cwd='/tinynav',
            env=_env,
        )
        self.get_logger().info(f'unitree_control started (interface={iface})')

    def get_sensor_mode(self) -> str:
        return self._sensor_mode

    def get_image_topics(self) -> list[str]:
        if self._sensor_mode == 'looper':
            return _IMAGE_TOPICS_LOOPER
        return _IMAGE_TOPICS_REALSENSE

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

    def _stop_sensor_procs(self):
        for attr in ('_looper_bridge_proc', '_realsense_proc', '_perception_proc', '_planning_proc'):
            self._kill_proc(getattr(self, attr))
            setattr(self, attr, None)

    def _restart_sensor_procs(self):
        _env = os.environ.copy()
        _env['PYTHONPATH'] = _VENV_SITE + ':' + _env.get('PYTHONPATH', '')
        if self._sensor_mode == 'looper':
            lf = self._make_log('looper_bridge')
            self._looper_bridge_proc = subprocess.Popen(
                ['uv', 'run', 'python', '/tinynav/tool/looper_bridge_node.py'],
                preexec_fn=os.setsid, cwd='/tinynav', env=_env,
                stdout=lf, stderr=subprocess.STDOUT,
            )
            lf.close()
            lf = self._make_log('planning')
            self._planning_proc = subprocess.Popen(
                ['uv', 'run', 'python', '/tinynav/tinynav/core/planning_node.py'],
                preexec_fn=os.setsid, cwd='/tinynav', env=_env,
                stdout=lf, stderr=subprocess.STDOUT,
            )
            lf.close()
            self.get_logger().info('Sensor procs restarted after map build')
            return
        if self._sensor_mode != 'realsense':
            return
        lf = self._make_log('realsense')
        self._realsense_proc = subprocess.Popen(
            ['bash', _REALSENSE_SCRIPT], preexec_fn=os.setsid,
            stdout=lf, stderr=subprocess.STDOUT,
        )
        lf.close()
        lf = self._make_log('perception')
        self._perception_proc = subprocess.Popen(
            ['uv', 'run', 'python', '/tinynav/tinynav/core/perception_node.py'],
            preexec_fn=os.setsid, cwd='/tinynav', env=_env,
            stdout=lf, stderr=subprocess.STDOUT,
        )
        lf.close()
        lf = self._make_log('planning')
        self._planning_proc = subprocess.Popen(
            ['uv', 'run', 'python', '/tinynav/tinynav/core/planning_node.py'],
            preexec_fn=os.setsid, cwd='/tinynav', env=_env,
            stdout=lf, stderr=subprocess.STDOUT,
        )
        lf.close()
        self.get_logger().info('Sensor procs restarted after map build')

    # ------------------------------------------------------------------ #
    # Nav nodes toggle                                                     #
    # ------------------------------------------------------------------ #

    def cmd_start_nav_nodes(self):
        _env = os.environ.copy()
        _env['PYTHONPATH'] = _VENV_SITE + ':' + _env.get('PYTHONPATH', '')
        _env['ROS_DOMAIN_ID'] = self.ros_domain_id
        lf = self._make_log('map_node')
        self._map_node_proc = subprocess.Popen(
            [
                'uv', 'run', 'python', '/tinynav/tinynav/core/map_node.py',
                '--tinynav_map_path', self.map_path,
            ],
            preexec_fn=os.setsid, cwd='/tinynav', env=_env,
            stdout=lf, stderr=subprocess.STDOUT,
        )
        lf.close()
        lf = self._make_log('cmd_vel_control')
        self._cmd_vel_proc = subprocess.Popen(
            ['uv', 'run', 'python', '/tinynav/tinynav/platforms/cmd_vel_control.py'],
            preexec_fn=os.setsid, cwd='/tinynav', env=_env,
            stdout=lf, stderr=subprocess.STDOUT,
        )
        lf.close()
        with self._lock:
            self._nav_nodes_running = True
        self.get_logger().info('Nav nodes started')

    def cmd_stop_nav_nodes(self):
        self._kill_proc(self._map_node_proc)
        self._kill_proc(self._cmd_vel_proc)
        self._map_node_proc = None
        self._cmd_vel_proc = None
        with self._lock:
            self._nav_nodes_running = False
            self._localized = False
            self._map_pose = None
            self._global_path = []
        self.get_logger().info('Nav nodes stopped')

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
                threading.Thread(target=self._finalize_bag_and_restart, args=(bag_path,), daemon=True).start()
            else:
                threading.Thread(target=self._finalize_bag, args=(bag_path,), daemon=True).start()

    def _finalize_bag_and_restart(self, bag_path: str):
        self._finalize_bag(bag_path)
        self._restart_sensor_procs()

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
        elif self.ros_domain_id is not None:
            _env['ROS_DOMAIN_ID'] = self.ros_domain_id
        _env['PYTHONPATH'] = _VENV_SITE + ':' + _env.get('PYTHONPATH', '')
        lf = self._make_log('perception')
        self.processes['perception'] = subprocess.Popen(
            ['uv', 'run', 'python', '/tinynav/tinynav/core/perception_node.py'],
            preexec_fn=os.setsid, cwd='/tinynav', env=_env,
            stdout=lf, stderr=subprocess.STDOUT,
        )
        lf.close()
        lf = self._make_log('build_map_node')
        self.processes['build_map'] = subprocess.Popen(
            [
                'uv', 'run', 'python', '/tinynav/tinynav/core/build_map_node.py',
                '--map_save_path', self.map_path,
                '--bag_file', bag_file,
            ],
            preexec_fn=os.setsid, cwd='/tinynav', env=_env,
            stdout=lf, stderr=subprocess.STDOUT,
        )
        lf.close()

        def wait_and_convert():
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
                import json as _json
                with open(pois_path, 'w') as _f:
                    _json.dump(
                        {'0': {'id': 0, 'name': 'home', 'position': [0.0, 0.0, 0.0]}},
                        _f, indent=2,
                    )
                self.get_logger().info('Auto-created home POI at (0,0,0)')

            self._stop_all()
            self.state = 'idle'
            self._pub_state()
            self._restart_sensor_procs()

        threading.Thread(target=wait_and_convert, daemon=True).start()

    def cmd_map_build(self):
        self._stop_sensor_procs()
        self._stop_all()
        self._start('rosbag_build_map')

    def _publish_cmd_pois(self, poi_id: int | None):
        """Publish the selected POI to map_node as JSON on /mapping/cmd_pois.
        Sending an empty dict clears the current nav target."""
        if poi_id is None:
            self._cmd_pois_pub.publish(String(data='{}'))
            return
        pois_file = os.path.join(self.map_path, 'pois.json')
        if not os.path.exists(pois_file):
            self.get_logger().warn('No pois.json found, cannot publish cmd_pois')
            return
        with open(pois_file) as f:
            pois = json.load(f)
        key = str(poi_id)
        if key not in pois:
            self.get_logger().warn(f'POI {poi_id} not found in pois.json')
            return
        # Re-index as "0" to match pub_pois.py convention expected by map_node
        payload = {'0': pois[key]}
        self._cmd_pois_pub.publish(String(data=json.dumps(payload)))

    def cmd_nav_start(self, poi_id: str | None = None):
        if poi_id is not None:
            self._publish_cmd_pois(int(poi_id))
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
            nav_running = self._nav_nodes_running
        if nav_running:
            # Clear the active nav target so map_node stops pathing.
            self._publish_cmd_pois(None)
            self.state = 'idle'
            self._pub_state()
        else:
            self._stop_all()

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
            for proc in (self.node._looper_bridge_proc, self.node._realsense_proc, self.node._perception_proc, self.node._planning_proc, self.node._unitree_proc, self.node._map_node_proc, self.node._cmd_vel_proc):
                if proc and proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), 15)
                        proc.wait(timeout=2)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
