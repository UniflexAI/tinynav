#!/usr/bin/env python3
"""ROS-backed planning simulator web server.

Web UI edits the scene. This process publishes synthetic /slam/depth,
/slam/odometry(_visual), /control/target_pose, then mirrors outputs from the
real planning_node + cmd_vel_control loop.
"""

from __future__ import annotations

import base64
import copy
import heapq
import math
import os
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path as RosPath
from pydantic import BaseModel
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, PointCloud
from std_msgs.msg import Bool

from tinynav.core.robot_specs import GO2_CONFIG
from tinynav.core import robot_specs as robot_specs_mod
from tool.simulator.map_volume import MapVolume
from tool.simulator.planning_scene import (
    SimObject,
    cam_size,
    image_u8_payload,
    make_camera_pose_from_config,
    render_depth,
)

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "offline_planning_web"
REPO_ROOT = ROOT.parents[1]


def _default_tinynav_db_path() -> Path:
    env = os.environ.get("TINYNAV_DB_PATH")
    if env:
        return Path(env).expanduser()
    repo_db = REPO_ROOT / "tinynav_db"
    if (repo_db / "maps").is_dir():
        return repo_db
    return Path("/tinynav/tinynav_db")


MAPS_ROOT = _default_tinynav_db_path() / "maps"
CAMERA_DEFAULTS = {
    "width": 160,
    "image_height": 100,
    "fx": 80.0,
    "fy": 50.0,
    "max_range": 15.0,
    "mount_height": 0.45,
}
ROBOT_PRESETS = {
    name.removesuffix("_CONFIG").lower(): asdict(getattr(robot_specs_mod, name))
    for name in dir(robot_specs_mod)
    if name.endswith("_CONFIG") and name != "ROBOT_CONFIG"
}


def _map_heuristic(start: tuple[int, int, int], goal: tuple[int, int, int], resolution: float) -> float:
    vec_start = np.array(start)
    vec_goal = np.array(goal)
    return float(np.linalg.norm((vec_start - vec_goal) * resolution) + 20 * abs(vec_start[2] - vec_goal[2]) * resolution)


def _reconstruct_path(parent: dict[tuple[int, int, int], tuple[int, int, int]], current: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    path = []
    while current in parent:
        path.append(current)
        if current == parent[current]:
            break
        current = parent[current]
    return path[::-1]


def _grid_in_bounds(idx: tuple[int, int, int], shape: tuple[int, int, int]) -> bool:
    return 0 <= idx[0] < shape[0] and 0 <= idx[1] < shape[1] and 0 <= idx[2] < shape[2]


def _world_to_grid(point: np.ndarray, volume: MapVolume) -> tuple[int, int, int]:
    idx = ((point - volume.origin) / volume.resolution).astype(np.int32)
    return int(idx[0]), int(idx[1]), int(idx[2])


def _snap_grid_z_to_sdf(
    idx: tuple[int, int, int],
    volume: MapVolume,
) -> tuple[int, int, int]:
    """Keep XY fixed and snap Z to the closest SDF/navigation layer."""
    if volume.sdf is None:
        return idx
    x, y, z = idx
    if x < 0 or x >= volume.grid.shape[0] or y < 0 or y >= volume.grid.shape[1]:
        return idx
    best = None
    best_key = (float("inf"), float("inf"))
    for zi in range(volume.grid.shape[2]):
        candidate = (x, y, zi)
        if volume.grid[candidate] == 2:
            continue
        sdf = float(volume.sdf[candidate])
        if not math.isfinite(sdf):
            continue
        key = (sdf, abs(zi - z))
        if key < best_key:
            best = candidate
            best_key = key
    return best if best is not None else idx


def _grid_to_world(path: list[tuple[int, int, int]], volume: MapVolume) -> np.ndarray:
    if not path:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(path, dtype=np.float32) * float(volume.resolution) + volume.origin.astype(np.float32)


def _search_close_to_sdf_map(
    start_index: tuple[int, int, int],
    sdf_map: np.ndarray,
    occupancy_map: np.ndarray,
    stop_distance: float,
) -> list[tuple[int, int, int]]:
    open_heap = [(float(sdf_map[start_index]), start_index)]
    open_heap_set = {start_index}
    parent = {start_index: start_index}
    visited = set()
    while open_heap:
        current_sdf, current = heapq.heappop(open_heap)
        open_heap_set.remove(current)
        visited.add(current)
        if current_sdf < stop_distance:
            return _reconstruct_path(parent, current)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if _grid_in_bounds(neighbor, sdf_map.shape):
                        if neighbor not in open_heap_set and neighbor not in visited and occupancy_map[neighbor] != 2:
                            open_heap_set.add(neighbor)
                            heapq.heappush(open_heap, (float(sdf_map[neighbor]), neighbor))
                            parent[neighbor] = current
    return []


def _search_within_sdf_map(
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    sdf_map: np.ndarray,
    occupancy_map: np.ndarray,
    resolution: float,
) -> list[tuple[int, int, int]]:
    sdf_bins = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    def queue_index(sdf_value: float) -> int:
        for idx, threshold in enumerate(sdf_bins):
            if sdf_value < threshold:
                return idx
        return len(sdf_bins)

    open_heaps = [[] for _ in range(len(sdf_bins) + 1)]
    open_sets = [set() for _ in range(len(sdf_bins) + 1)]
    start_queue_idx = queue_index(float(sdf_map[start]))
    heapq.heappush(open_heaps[start_queue_idx], (_map_heuristic(start, goal, resolution), start))
    open_sets[start_queue_idx].add(start)
    parent = {start: start}
    visited = set()

    while True:
        queue_idx = -1
        for i, q in enumerate(open_heaps):
            if q:
                queue_idx = i
                break
        if queue_idx == -1:
            break
        _, current = heapq.heappop(open_heaps[queue_idx])
        open_sets[queue_idx].remove(current)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return _reconstruct_path(parent, current)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if not _grid_in_bounds(neighbor, sdf_map.shape):
                        continue
                    if neighbor in visited or occupancy_map[neighbor] == 2:
                        continue
                    neighbor_queue_idx = queue_index(float(sdf_map[neighbor]))
                    if neighbor in open_sets[neighbor_queue_idx]:
                        continue
                    open_sets[neighbor_queue_idx].add(neighbor)
                    heapq.heappush(open_heaps[neighbor_queue_idx], (_map_heuristic(neighbor, goal, resolution), neighbor))
                    parent.setdefault(neighbor, current)
    return []


def _sdf_global_path(volume: MapVolume, robot_xy: list[float], target: list[float]) -> np.ndarray:
    if volume.sdf is None:
        return np.empty((0, 3), dtype=np.float32)
    target_z = float(target[2] if len(target) > 2 else volume.origin[2])
    start_idx = _world_to_grid(np.array([robot_xy[0], robot_xy[1], target_z], dtype=np.float64), volume)
    goal_idx = _world_to_grid(np.array([target[0], target[1], target_z], dtype=np.float64), volume)
    start_idx = _snap_grid_z_to_sdf(start_idx, volume)
    goal_idx = _snap_grid_z_to_sdf(goal_idx, volume)
    if not _grid_in_bounds(start_idx, volume.grid.shape) or not _grid_in_bounds(goal_idx, volume.grid.shape):
        return np.empty((0, 3), dtype=np.float32)
    sdf_start_path = _search_close_to_sdf_map(start_idx, volume.sdf, volume.grid, 0.2)
    sdf_goal_path = _search_close_to_sdf_map(goal_idx, volume.sdf, volume.grid, 0.2)
    if not sdf_start_path or not sdf_goal_path:
        return np.empty((0, 3), dtype=np.float32)
    path_sdf = _search_within_sdf_map(sdf_start_path[-1], sdf_goal_path[-1], volume.sdf, volume.grid, volume.resolution)
    path = sdf_start_path + path_sdf + sdf_goal_path[::-1]
    return _grid_to_world(path, volume)


def _local_target_from_path(path: np.ndarray, robot_xy: list[float], horizon_m: float = 2.5) -> list[float] | None:
    if len(path) == 0:
        return None
    robot = np.array(robot_xy, dtype=np.float32)
    closest_idx = int(np.argmin(np.linalg.norm(path[:, :2] - robot[:2], axis=1)))
    accumulated = 0.0
    start = np.array([robot_xy[0], robot_xy[1]], dtype=np.float32)
    target = path[-1]
    for i in range(closest_idx, len(path) - 1):
        accumulated += float(np.linalg.norm(path[i, :2] - start))
        if accumulated > horizon_m:
            target = path[i]
            break
        start = path[i, :2]
    return [float(target[0]), float(target[1]), float(target[2])]


def _robot_dict(name: str | None = None) -> dict[str, Any]:
    key = (name or GO2_CONFIG.name).strip().lower()
    return copy.deepcopy(ROBOT_PRESETS.get(key, asdict(GO2_CONFIG)))


def default_config(robot_name: str | None = None) -> dict[str, Any]:
    robot = _robot_dict(robot_name)
    return {
        "name": "ros_planning_sim",
        "robot": robot,
        "obstacle": copy.deepcopy(robot.get("obstacle") or {}),
        "camera": copy.deepcopy(CAMERA_DEFAULTS),
        "start": {"xy": [0.0, 0.0], "yaw_deg": 0.0},
        "target": [4.0, 0.0, 0.0],
        "map_path": None,
        "map_name": None,
        "objects": [],
    }


def odom_from_T(T: np.ndarray, stamp, frame_id: str = "world", child_frame_id: str = "camera") -> Odometry:
    quat = R.from_matrix(T[:3, :3]).as_quat()
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = map(float, T[:3, 3])
    msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = map(float, quat)
    return msg


def grid_payload(msg: OccupancyGrid, data_u8: np.ndarray) -> dict[str, Any]:
    return {
        "width": int(data_u8.shape[1]),
        "height": int(data_u8.shape[0]),
        "data": data_u8.ravel().tolist(),
        "origin": [float(msg.info.origin.position.x), float(msg.info.origin.position.y)],
        "resolution": float(msg.info.resolution),
    }


def rgb_payload(image: np.ndarray) -> dict[str, Any]:
    rgb = np.ascontiguousarray(image, dtype=np.uint8)
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("failed to encode map background PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    return {"width": int(w), "height": int(h), "mime": "image/png", "data_url": data_url}


_MAP_CACHE: dict[str, MapVolume] = {}


def load_map_volume(map_path: str | None) -> MapVolume | None:
    if not map_path:
        return None
    path = str(Path(map_path).expanduser().resolve())
    cached = _MAP_CACHE.get(path)
    if cached is not None:
        return cached
    volume = MapVolume.load(path)
    _MAP_CACHE[path] = volume
    return volume


def resolve_map_path(map_name: str | None = None, map_path: str | None = None) -> str:
    if map_name:
        candidate = (MAPS_ROOT / map_name).resolve()
        if MAPS_ROOT.resolve() not in candidate.parents and candidate != MAPS_ROOT.resolve():
            raise ValueError(f"Invalid map name: {map_name!r}")
        if not candidate.is_dir():
            raise FileNotFoundError(f"Map folder not found: {candidate}")
        return str(candidate)
    if map_path:
        return str(Path(map_path).expanduser().resolve())
    raise ValueError("map_name or map_path is required")


def list_map_catalog() -> list[dict[str, Any]]:
    root = MAPS_ROOT
    if not root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if not (child / "occupancy_grid.npy").is_file():
            continue
        entries.append({"name": child.name, "path": str(child.resolve())})
    return entries


def map_config_for_path(map_path: str, start_xy: list[float] | None = None, yaw_deg: float = 0.0) -> dict[str, Any]:
    volume = load_map_volume(map_path)
    if volume is None:
        raise FileNotFoundError(f"Could not load map at {map_path}")
    robot_name = None
    camera = copy.deepcopy(CAMERA_DEFAULTS)
    if SIM_NODE is not None:
        robot_name = SIM_NODE.config.get("robot", {}).get("name")
        camera = copy.deepcopy(SIM_NODE.config.get("camera") or camera)
    config = default_config(robot_name)
    config["camera"] = camera
    if start_xy is None:
        start_xy = list(volume.default_start_xy())
    z = float(volume.origin[2])
    config.update({
        "name": Path(map_path).name,
        "map_path": volume.map_path,
        "map_name": Path(map_path).name,
        "start": {"xy": [float(start_xy[0]), float(start_xy[1])], "yaw_deg": float(yaw_deg)},
        "target": [float(start_xy[0]) + 2.0, float(start_xy[1]), z],
        "objects": [],
    })
    config["camera"]["ground_z"] = z
    return config


class RosPlanningSimNode(Node):
    def __init__(self):
        super().__init__("tinynav_ros_planning_sim")
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.config = default_config()
        self.map_volume: MapVolume | None = None
        self.map_info: dict[str, Any] | None = None
        self.map_background: dict[str, Any] | None = None
        self._apply_map_from_config(self.config)
        self.control_xy = list(self.config["start"]["xy"])
        self.yaw_deg = float(self.config["start"]["yaw_deg"])
        self.last_update = time.monotonic()
        self.last_depth = np.zeros((100, 160), dtype=np.float32)
        self.last_cmd = Twist()
        self.last_path: list[list[float]] = []
        self.last_footprint: list[list[float]] = []
        self.last_obstacle_mask: dict[str, Any] | None = None
        self.last_esdf_grid: dict[str, Any] | None = None
        self.global_path_xy: list[list[float]] = []
        self.global_local_target: list[float] | None = None
        self.running = True

        self.depth_pub = self.create_publisher(Image, "/slam/depth", 10)
        self.odom_visual_pub = self.create_publisher(Odometry, "/slam/odometry_visual", 10)
        self.odom_pub = self.create_publisher(Odometry, "/slam/odometry", 10)
        self.target_pub = self.create_publisher(Odometry, "/control/target_pose", 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, "/camera/camera/infra2/camera_info", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.nav_active_pub = self.create_publisher(Bool, "/nav/active", latched)
        self.nav_paused_pub = self.create_publisher(Bool, "/nav/paused", latched)

        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(RosPath, "/planning/trajectory_path", self.path_callback, 10)
        self.create_subscription(PointCloud, "/planning/footprint", self.footprint_callback, 10)
        self.create_subscription(OccupancyGrid, "/planning/obstacle_mask", self.obstacle_callback, 10)
        self.create_subscription(OccupancyGrid, "/planning/occupancy_grid", self.esdf_callback, 10)
        self.create_timer(1.0 / 8.0, self.tick)

    def _apply_map_from_config(self, config: dict[str, Any]) -> None:
        map_path = config.get("map_path")
        if not map_path:
            self.map_volume = None
            self.map_info = None
            self.map_background = None
            return
        try:
            volume = load_map_volume(str(map_path))
        except (FileNotFoundError, ValueError, OSError) as exc:
            self.get_logger().error(f"Map load failed: {exc}")
            self.map_volume = None
            self.map_info = None
            self.map_background = None
            return
        self.map_volume = volume
        self.map_info = volume.info().as_dict()
        self.map_background = rgb_payload(volume.background_rgb())
        config["map_path"] = volume.map_path
        cam = config.setdefault("camera", {})
        cam["ground_z"] = float(volume.origin[2])

    def set_config(self, config: dict[str, Any], reset: bool = False) -> None:
        with self.lock:
            prev_map = self.config.get("map_path")
            self.config = copy.deepcopy(config)
            robot = self.config.get("robot")
            if not isinstance(robot, dict):
                robot = _robot_dict()
                self.config["robot"] = robot
            self.config["obstacle"] = copy.deepcopy(robot.get("obstacle") or {})
            if self.config.get("map_path") != prev_map:
                self._apply_map_from_config(self.config)
            if reset:
                self.control_xy = list(self.config.get("start", {}).get("xy", [0.0, 0.0]))
                self.yaw_deg = float(self.config.get("start", {}).get("yaw_deg", 0.0))
                self.last_cmd = Twist()
                self.last_path = []
                self.last_footprint = []
                self.last_obstacle_mask = None
                self.last_esdf_grid = None
                self.global_path_xy = []
                self.global_local_target = None

    def cmd_callback(self, msg: Twist) -> None:
        with self.lock:
            self.last_cmd = msg

    def path_callback(self, msg: RosPath) -> None:
        with self.lock:
            self.last_path = [[float(p.pose.position.x), float(p.pose.position.y)] for p in msg.poses]

    def footprint_callback(self, msg: PointCloud) -> None:
        with self.lock:
            pts = msg.points
            # PlanningNode publishes 21 samples/edge; keep corners for UI.
            idxs = [0, 21, 42, 63, 0] if len(pts) >= 64 else range(len(pts))
            self.last_footprint = [[float(pts[i].x), float(pts[i].y)] for i in idxs]

    def obstacle_callback(self, msg: OccupancyGrid) -> None:
        data = np.asarray(msg.data, dtype=np.int16).reshape((msg.info.width, msg.info.height), order="F")
        with self.lock:
            self.last_obstacle_mask = grid_payload(msg, np.where(data > 0, 255, 0).astype(np.uint8))

    def esdf_callback(self, msg: OccupancyGrid) -> None:
        data = np.asarray(msg.data, dtype=np.int16).reshape((msg.info.width, msg.info.height), order="F")
        clearance = np.round((1.0 - np.clip(data, 0, 120).astype(np.float32) / 120.0) * 255.0).astype(np.uint8)
        with self.lock:
            self.last_esdf_grid = grid_payload(msg, clearance)

    def publish_camera_info(self, stamp, config: dict[str, Any]) -> None:
        cam = config["camera"]
        width, height = cam_size(cam)
        fx, fy = float(cam["fx"]), float(cam["fy"])
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        msg.width, msg.height = width, height
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, -0.06 * fx, 0.0, 0.0, 1.0, 0.0]
        self.camera_info_pub.publish(msg)

    def publish_target(self, stamp, config: dict[str, Any], target_override: list[float] | None = None) -> None:
        target = target_override if target_override is not None else config.get("target", [4.0, 0.0, 0.0])
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.child_frame_id = "target"
        msg.pose.pose.position.x = float(target[0])
        msg.pose.pose.position.y = float(target[1])
        msg.pose.pose.position.z = float(target[2] if len(target) > 2 else 0.0)
        msg.pose.pose.orientation.w = 1.0
        self.target_pub.publish(msg)

    def integrate_cmd(self, dt: float) -> None:
        yaw = math.radians(self.yaw_deg)
        self.control_xy[0] += math.cos(yaw) * self.last_cmd.linear.x * dt
        self.control_xy[1] += math.sin(yaw) * self.last_cmd.linear.x * dt
        self.yaw_deg = (self.yaw_deg + math.degrees(self.last_cmd.angular.z * dt) + 180.0) % 360.0 - 180.0

    def tick(self) -> None:
        with self.lock:
            if not self.running:
                return
            now = time.monotonic()
            dt = max(1e-3, min(0.2, now - self.last_update))
            self.last_update = now
            self.integrate_cmd(dt)
            config = copy.deepcopy(self.config)
            config.setdefault("start", {})["xy"] = [float(self.control_xy[0]), float(self.control_xy[1])]
            config["start"]["yaw_deg"] = float(self.yaw_deg)
            yaw_deg = float(self.yaw_deg)
            map_volume = self.map_volume

        target_override = None
        global_path_xy: list[list[float]] = []
        if map_volume is not None and map_volume.sdf is not None:
            global_path = _sdf_global_path(map_volume, config["start"]["xy"], config.get("target", [4.0, 0.0, 0.0]))
            if len(global_path) > 0:
                target_override = _local_target_from_path(global_path, config["start"]["xy"])
                global_path_xy = [[float(p[0]), float(p[1])] for p in global_path]
        with self.lock:
            self.global_path_xy = global_path_xy
            self.global_local_target = target_override

        objects = [SimObject(**obj) for obj in config.get("objects", [])]
        T_cam = make_camera_pose_from_config(config["start"]["xy"], yaw_deg, config["robot"], config["camera"])
        depth = render_depth(objects, T_cam, config["camera"], map_volume=map_volume)
        with self.lock:
            self.last_depth = depth

        stamp = self.get_clock().now().to_msg()
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = "camera"
        odom_msg = odom_from_T(T_cam, stamp)
        self.depth_pub.publish(depth_msg)
        self.odom_visual_pub.publish(odom_msg)
        self.odom_pub.publish(odom_msg)
        self.publish_camera_info(stamp, config)
        self.publish_target(stamp, config, target_override=target_override)
        self.nav_active_pub.publish(Bool(data=True))
        self.nav_paused_pub.publish(Bool(data=False))

    def frame(self) -> dict[str, Any]:
        with self.lock:
            xy = [float(self.control_xy[0]), float(self.control_xy[1])]
            return {
                "robot_xy": xy,
                "robot_yaw_deg": float(self.yaw_deg),
                "robot_footprint_xy": copy.deepcopy(self.last_footprint),
                "selected_trajectory_xy": copy.deepcopy(self.last_path),
                "global_path_xy": copy.deepcopy(self.global_path_xy),
                "global_local_target": copy.deepcopy(self.global_local_target),
                "selected_param": [float(self.last_cmd.linear.x), float(self.last_cmd.angular.z)],
                "depth_u8": image_u8_payload(self.last_depth, 0.0, float(self.config["camera"]["max_range"])),
                "obstacle_u8": self.last_obstacle_mask,
                "esdf_u8": self.last_esdf_grid,
                "next_start": {"xy": xy, "yaw_deg": float(self.yaw_deg)},
            }


class RunRequest(BaseModel):
    config: dict[str, Any]
    reset: bool | None = None


class LoadMapRequest(BaseModel):
    map_name: str | None = None
    map_path: str | None = None
    start_xy: list[float] | None = None
    yaw_deg: float = 0.0


app = FastAPI(title="TinyNav ROS Planning Simulator")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
SIM_NODE: RosPlanningSimNode | None = None
EXECUTOR: MultiThreadedExecutor | None = None
PROCS: list[subprocess.Popen] = []
CHILD_SCRIPTS = (
    "tinynav/core/planning_node.py",
    "tinynav/platforms/cmd_vel_control.py",
)
_LAST_PLANNING_RESET = 0.0
_PLANNING_RESET_COOLDOWN_S = 1.0


def _require_sim() -> RosPlanningSimNode:
    if SIM_NODE is None:
        raise HTTPException(status_code=503, detail="ROS simulator is not ready")
    return SIM_NODE


def start_ros() -> None:
    global SIM_NODE, EXECUTOR
    if SIM_NODE is not None:
        return
    rclpy.init(args=None)
    SIM_NODE = RosPlanningSimNode()
    EXECUTOR = MultiThreadedExecutor(num_threads=4)
    EXECUTOR.add_node(SIM_NODE)
    threading.Thread(target=EXECUTOR.spin, daemon=True).start()


def _active_robot_type() -> str:
    if SIM_NODE is None:
        return "go2"
    return str(SIM_NODE.config.get("robot", {}).get("name", "go2")).strip().lower()


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(key, "1")
    env["ROBOT_TYPE"] = _active_robot_type()
    return env


def _prune_procs() -> None:
    alive = []
    for proc in PROCS:
        if proc.poll() is None:
            alive.append(proc)
    PROCS[:] = alive


def _stop_script(script: str, grace_s: float = 1.0) -> None:
    _prune_procs()
    victims = [p for p in PROCS if script in " ".join(p.args)]
    for proc in victims:
        proc.terminate()
    deadline = time.monotonic() + grace_s
    for proc in victims:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)
    _prune_procs()


def _spawn_if_needed(script: str) -> None:
    _prune_procs()
    if any(script in " ".join(p.args) for p in PROCS):
        return
    PROCS.append(subprocess.Popen(["uv", "run", "python", script], cwd=str(REPO_ROOT), env=_child_env()))


def ensure_ros_loop(reset_planning: bool = False, force: bool = False) -> int:
    """Start planning/control children. Optionally restart planning to clear occupancy."""
    global _LAST_PLANNING_RESET
    if reset_planning:
        now = time.monotonic()
        if force or (now - _LAST_PLANNING_RESET) >= _PLANNING_RESET_COOLDOWN_S:
            _stop_script("tinynav/core/planning_node.py")
            _LAST_PLANNING_RESET = now
    for script in CHILD_SCRIPTS:
        _spawn_if_needed(script)
    return sum(1 for p in PROCS if p.poll() is None)


def stop_children() -> None:
    _prune_procs()
    for proc in list(PROCS):
        proc.terminate()
    for proc in list(PROCS):
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)
    PROCS.clear()


@app.on_event("startup")
def startup() -> None:
    start_ros()


@app.on_event("shutdown")
def shutdown() -> None:
    stop_children()
    if EXECUTOR is not None:
        EXECUTOR.shutdown()
    if SIM_NODE is not None:
        SIM_NODE.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/default-config")
def get_default_config() -> dict[str, Any]:
    return default_config()


@app.get("/api/map-catalog")
def map_catalog() -> dict[str, Any]:
    root = MAPS_ROOT
    maps = list_map_catalog()
    return {
        "maps_root": str(root.resolve() if root.exists() else root),
        "maps_root_exists": root.is_dir(),
        "maps": maps,
    }


@app.post("/api/load-map")
def load_map(request: LoadMapRequest) -> dict[str, Any]:
    try:
        resolved = resolve_map_path(request.map_name, request.map_path)
        config = map_config_for_path(resolved, request.start_xy, request.yaw_deg)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if SIM_NODE is not None:
        SIM_NODE.set_config(config, reset=True)
    volume = load_map_volume(config["map_path"])
    return {
        "config": config,
        "map_info": volume.info().as_dict() if volume else None,
        "map_background": rgb_payload(volume.background_rgb()) if volume else None,
    }


@app.get("/api/map-info")
def map_info(map_path: str) -> dict[str, Any]:
    try:
        volume = load_map_volume(map_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if volume is None:
        raise HTTPException(status_code=400, detail="map_path is required")
    return {
        "map_info": volume.info().as_dict(),
        "map_background": rgb_payload(volume.background_rgb()),
    }


@app.get("/api/robot-presets")
def get_robot_presets() -> dict[str, Any]:
    return {"presets": [{"name": name, "robot": copy.deepcopy(robot)} for name, robot in ROBOT_PRESETS.items()]}


@app.post("/api/update-config")
def update_config(request: RunRequest) -> dict[str, Any]:
    node = _require_sim()
    if not isinstance(request.config, dict):
        raise HTTPException(status_code=400, detail="config must be an object")
    reset = bool(request.reset)
    prev_robot = node.config.get("robot", {}).get("name")
    node.set_config(copy.deepcopy(request.config), reset=reset)
    robot_changed = prev_robot != node.config.get("robot", {}).get("name")
    if robot_changed:
        _stop_script("tinynav/platforms/cmd_vel_control.py")
    if reset or robot_changed:
        ensure_ros_loop(reset_planning=True, force=robot_changed)
    return {"ok": True, "robot_changed": robot_changed}


@app.post("/api/start-ros-loop")
def start_ros_loop() -> dict[str, Any]:
    return {"ok": True, "process_count": ensure_ros_loop(reset_planning=True, force=True)}


@app.get("/api/sim-state")
def sim_state() -> dict[str, Any]:
    return {"frame": _require_sim().frame()}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8766)


if __name__ == "__main__":
    main()
