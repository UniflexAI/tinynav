"""
BackendNode — extends Ros2NodeManager with extra subscriptions for pose and
mapping progress, plus a NodeRunner that spins it in a background thread.
"""
from __future__ import annotations

import math
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import rclpy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String

from tool.ros2_node_manager import Ros2NodeManager


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

        self.create_subscription(Float32, '/mapping/percent', self._on_mapping_percent, 10)
        self.create_subscription(Odometry, '/slam/odometry', self._on_slam_odom, 10)
        self.create_subscription(
            Odometry, '/mapping/current_pose_in_map', self._on_pose_in_map, 10
        )

        # Publisher for nav target (consumed by map_node in the future)
        self._nav_target_pub = self.create_publisher(String, '/service/nav_target', 10)

    # ------------------------------------------------------------------ #
    # ROS callbacks                                                        #
    # ------------------------------------------------------------------ #

    def _on_mapping_percent(self, msg: Float32):
        with self._lock:
            self.mapping_percent = float(msg.data)

    def _on_slam_odom(self, msg: Odometry):
        pose = self._odom_to_dict(msg, source='slam')
        with self._lock:
            self.current_pose = pose
        for cb in self.pose_callbacks:
            try:
                cb(pose)
            except Exception:
                pass

    def _on_pose_in_map(self, msg: Odometry):
        pose = self._odom_to_dict(msg, source='map')
        with self._lock:
            self.current_pose = pose
        for cb in self.pose_callbacks:
            try:
                cb(pose)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _odom_to_dict(msg: Odometry, source: str) -> dict:
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z,
            'yaw': yaw,
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'source': source,
        }

    # ------------------------------------------------------------------ #
    # Command API (called from FastAPI handlers — thread-safe enough)     #
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        with self._lock:
            raw = self.state
            pct = self.mapping_percent
        bag_files_exist = os.path.exists(os.path.join(self.bag_path, 'bag_0.db3'))
        map_files_exist = os.path.exists(os.path.join(self.map_path, 'occupancy_grid.npy'))
        return {
            'bagStatus': 'recording' if raw == 'realsense_bag_record' else 'idle',
            'bagFileReady': bag_files_exist,
            'mapStatus': self._derive_map_status(raw, pct, map_files_exist),
            'mappingPercent': pct,
            'navStatus': 'navigating' if raw == 'navigation' else 'idle',
            'rawState': raw,
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

    def cmd_bag_start(self):
        self._stop_all()
        self._start('realsense_bag_record')

    def cmd_bag_stop(self):
        if self.state == 'realsense_bag_record':
            self._stop_all()

    def cmd_map_build(self):
        self._stop_all()
        self._start('rosbag_build_map')

    def cmd_nav_start(self, poi_id: str | None = None):
        self._stop_all()
        if poi_id is not None:
            self._nav_target_pub.publish(String(data=str(poi_id)))
        self._start('navigation')

    def cmd_nav_cancel(self):
        if self.state == 'navigation':
            self._stop_all()


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
