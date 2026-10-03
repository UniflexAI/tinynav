#!/usr/bin/env python3
"""ROS-backed planning simulator web server.

Web UI edits the scene. This process publishes synthetic /slam/depth,
/slam/odometry(_visual), /control/target_pose, then mirrors outputs from the
real planning_node + simulator_control loop.
"""

from __future__ import annotations

import base64
import copy
import math
import json
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
from std_msgs.msg import Bool, String

from tool.simulator.navigation_lab import NavigationLab
from tool.simulator.decision_observer import DecisionObserver
from tool.simulator.candidate_branches import compact_report, compare
from tool.simulator.recovery_strategies import RecoveryExecutor, proposals, validate_strategy
from tool.simulator.decision_information import enrich
from tool.simulator.closed_loop_results import results as closed_loop_results
from tinynav.core.robot_specs import GO2_CONFIG
from tinynav.core import robot_specs as robot_specs_mod
from tool.simulator.map_volume import MapVolume
from tool.simulator.planning_scene import (
    SimObject,
    cam_size,
    footprint_polygon_xy,
    image_u8_payload,
    make_camera_pose_from_config,
    render_depth,
    robot_hits_objects,
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
    "max_range": 8.0,
    "mount_height": 0.45,
}
ROBOT_PRESETS = {
    name.removesuffix("_CONFIG").lower(): asdict(getattr(robot_specs_mod, name))
    for name in dir(robot_specs_mod)
    if name.endswith("_CONFIG") and name != "ROBOT_CONFIG"
}


def l_corridor_objects() -> list[dict[str, Any]]:
    """L-shaped corridor used as the default web-sim planning scene."""
    return [
        {"name": "lower_horizontal_wall", "kind": "box", "center": [1.8, -0.85, 0.65], "size": [5.6, 0.3, 1.3]},
        {"name": "upper_horizontal_wall_before_turn", "kind": "box", "center": [1.15, 0.85, 0.65], "size": [4.3, 0.3, 1.3]},
        {"name": "inside_corner_block", "kind": "box", "center": [3.45, 0.85, 0.65], "size": [0.3, 0.3, 1.3]},
        {"name": "left_vertical_wall_after_turn", "kind": "box", "center": [3.15, 2.8, 0.65], "size": [0.3, 3.6, 1.3]},
        {"name": "right_vertical_wall", "kind": "box", "center": [4.85, 2.55, 0.65], "size": [0.3, 5.1, 1.3]},
        {"name": "entry_left_stub", "kind": "box", "center": [-1.05, 0.85, 0.65], "size": [0.8, 0.3, 1.3]},
        {"name": "entry_right_stub", "kind": "box", "center": [-1.05, -0.85, 0.65], "size": [0.8, 0.3, 1.3]},
        {"name": "far_end_cap", "kind": "box", "center": [4.0, 5.25, 0.65], "size": [2.0, 0.3, 1.3]},
    ]


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
        "target": [3.9, 4.4, 0.0],
        "map_path": None,
        "map_name": None,
        "objects": l_corridor_objects(),
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
        self.collision = False
        self.geom_footprint: list[list[float]] = []
        self.running = True
        self.lab = NavigationLab()
        self.world_mode = "observed"
        self.saved_run_id = None
        self.config_generation = 0
        self.plan_report = None
        self.plan_received_at = None
        self.experiment_mode = os.getenv("TINYNAV_EXPERIMENT_MODE", "off")
        if self.experiment_mode not in ("off", "baseline", "model") or (self.experiment_mode != "off" and (not 1 <= int(os.getenv("ROS_DOMAIN_ID", "0")) <= 229 or os.getenv("ROS_LOCALHOST_ONLY") != "1" or os.getenv("TINYNAV_WEB_HOST") != "127.0.0.1")):
            raise ValueError("Experiments require a local-only ROS domain 1..229 and loopback HTTP")
        self.experiment_events = []
        self.experiment_pending = None
        self.experiment_next_call = 0.0
        self.override = None
        self.override_until = 0.0
        self.recovery = RecoveryExecutor()
        self.information_version = os.getenv('TINYNAV_DECISION_INFORMATION', 'basic')
        if self.information_version not in ('basic','rich'):raise ValueError('Unknown information version')

        self.depth_pub = self.create_publisher(Image, "/slam/depth", 10)
        self.odom_visual_pub = self.create_publisher(Odometry, "/slam/odometry_visual", 10)
        self.odom_pub = self.create_publisher(Odometry, "/slam/odometry", 10)
        self.target_pub = self.create_publisher(Odometry, "/control/target_pose", 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, "/camera/camera/infra2/camera_info", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.nav_active_pub = self.create_publisher(Bool, "/nav/active", latched)
        self.nav_paused_pub = self.create_publisher(Bool, "/nav/paused", latched)

        self.create_subscription(String, "/planning/report", self.report_callback, 1)
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

    def save_result(self) -> None:
        if not self.lab.run or self.lab.run["status"] == "running":
            return
        run_id = self.lab.run["id"]
        if self.saved_run_id == run_id:
            return
        folder = REPO_ROOT / "tinynav_temp" / "navigation_lab"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{run_id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.lab.export()), encoding="utf-8")
        temporary.replace(path)
        self.saved_run_id = run_id

    def set_config(self, config: dict[str, Any], reset: bool = False) -> None:
        with self.lock:
            if self.lab.run and self.lab.run["status"] == "running":
                self.lab.run["status"] = "config_changed"
                self.save_result()
            self.lab.cells.clear()
            self.lab.history.clear()
            self.experiment_events = []
            self.experiment_pending = None
            self.experiment_next_call = time.monotonic() + 6
            self.override = None
            self.recovery = RecoveryExecutor()
            self.config_generation += 1
            self.config_changed_at = time.time()
            self.plan_report = None
            self.plan_received_at = None
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
                self.collision = False
                self.geom_footprint = []

    def report_callback(self, msg: String) -> None:
        report = json.loads(msg.data)
        with self.lock:
            if report["stamp_unix"] >= getattr(self, "config_changed_at", 0):
                self.plan_report = report
                self.plan_received_at = time.monotonic()

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

    def publish_target(self, stamp, config: dict[str, Any]) -> None:
        target = config.get("target", [4.0, 0.0, 0.0])
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
        if self.collision:
            self.last_cmd = Twist()
            return
        if self.recovery.active:
            age = time.monotonic()-self.plan_received_at if self.plan_received_at is not None else float('inf')
            objects = [SimObject(**obj) for obj in self.config.get('objects', [])]
            move = self.recovery.step(self.control_xy, self.yaw_deg, self.config['target'], dt, time.monotonic(), age,
                                      lambda xy,yaw: robot_hits_objects(xy,yaw,self.config['robot'],objects))
            if move:
                v,w,used = move
                angle = math.radians(self.yaw_deg)
                self.control_xy[0] += math.cos(angle)*v*used
                self.control_xy[1] += math.sin(angle)*v*used
                self.yaw_deg = (self.yaw_deg+math.degrees(w*used)+180)%360-180
                return
        command = self.override if self.override and time.monotonic() < self.override_until else None
        if command:
            robot = self.config["robot"]
            v = float(np.clip(command["linear_mps"], -robot["max_linear_vel"], robot["max_linear_vel"]))
            w = float(np.clip(command["yaw_radps"], -robot["max_angular_vel"], robot["max_angular_vel"]))
        else:
            self.override = None
            v, w = self.last_cmd.linear.x, self.last_cmd.angular.z
        yaw = math.radians(self.yaw_deg)
        self.control_xy[0] += math.cos(yaw) * v * dt
        self.control_xy[1] += math.sin(yaw) * v * dt
        self.yaw_deg = (self.yaw_deg + math.degrees(w * dt) + 180.0) % 360.0 - 180.0

    def tick(self) -> None:
        with self.lock:
            if not self.running:
                self.last_update = time.monotonic()
                self.nav_paused_pub.publish(Bool(data=True))
                return
            now = time.monotonic()
            dt = max(1e-3, min(0.2, now - self.last_update))
            self.last_update = now
            prev_xy = [float(self.control_xy[0]), float(self.control_xy[1])]
            prev_yaw = float(self.yaw_deg)
            self.integrate_cmd(dt)
            generation = self.config_generation
            config = copy.deepcopy(self.config)
            robot = config.get("robot") or _robot_dict()
            objects = [SimObject(**obj) for obj in config.get("objects", [])]
            if robot_hits_objects(self.control_xy, self.yaw_deg, robot, objects):
                self.collision = True
                self.control_xy = prev_xy
                self.yaw_deg = prev_yaw
                self.last_cmd = Twist()
            config.setdefault("start", {})["xy"] = [float(self.control_xy[0]), float(self.control_xy[1])]
            config["start"]["yaw_deg"] = float(self.yaw_deg)
            yaw_deg = float(self.yaw_deg)
            map_volume = self.map_volume
            geom = footprint_polygon_xy(self.control_xy, math.radians(yaw_deg), robot)
            self.geom_footprint = geom.tolist() + [geom[0].tolist()]

        T_cam = make_camera_pose_from_config(config["start"]["xy"], yaw_deg, config["robot"], config["camera"])
        depth = render_depth(objects, T_cam, config["camera"], map_volume=map_volume)
        with self.lock:
            if generation != self.config_generation:
                return
            self.last_depth = depth
            self.lab.observe_depth(depth, T_cam, config["camera"], robot)
            was_measuring = self.lab.run and self.lab.run["status"] == "running"
            self.lab.update(self.control_xy, self.yaw_deg, config["target"], self.collision)
            if was_measuring and self.lab.run["status"] != "running":
                self.running = False
                self.last_cmd = Twist()
                self.save_result()

        stamp = self.get_clock().now().to_msg()
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = "camera"
        odom_msg = odom_from_T(T_cam, stamp)
        self.depth_pub.publish(depth_msg)
        self.odom_visual_pub.publish(odom_msg)
        self.odom_pub.publish(odom_msg)
        self.publish_camera_info(stamp, config)
        self.publish_target(stamp, config)
        self.nav_active_pub.publish(Bool(data=True))
        self.nav_paused_pub.publish(Bool(data=not self.running))

    def frame(self) -> dict[str, Any]:
        with self.lock:
            xy = [float(self.control_xy[0]), float(self.control_xy[1])]
            footprint = copy.deepcopy(self.geom_footprint) or copy.deepcopy(self.last_footprint)
            return {
                "planning_report": copy.deepcopy(self.plan_report),
                "metrics": self.lab.metrics(),
                "world_state": self.lab.world_state(xy, self.yaw_deg, self.config, self.world_mode),
                "robot_xy": xy,
                "robot_yaw_deg": float(self.yaw_deg),
                "robot_footprint_xy": footprint,
                "selected_trajectory_xy": copy.deepcopy(self.last_path),
                "selected_param": [float(self.last_cmd.linear.x), float(self.last_cmd.angular.z)],
                "collision": bool(self.collision),
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
OBSERVER = DecisionObserver(REPO_ROOT / "tinynav_temp" / "decision_observer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
SIM_NODE: RosPlanningSimNode | None = None
EXECUTOR: MultiThreadedExecutor | None = None
PROCS: list[subprocess.Popen] = []
CHILD_SCRIPTS = (
    "tinynav/core/planning_node.py",
    "tinynav/platforms/simulator_control.py",
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
    env["TINYNAV_PLAN_REPORT"] = "1"
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
    if SIM_NODE.experiment_mode == "model":
        threading.Thread(target=experiment_worker, daemon=True).start()


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
    node.running = True
    robot_changed = prev_robot != node.config.get("robot", {}).get("name")
    if robot_changed:
        _stop_script("tinynav/platforms/simulator_control.py")
    if reset or robot_changed:
        ensure_ros_loop(reset_planning=True, force=robot_changed)
    return {"ok": True, "robot_changed": robot_changed}


@app.post("/api/start-ros-loop")
def start_ros_loop() -> dict[str, Any]:
    return {"ok": True, "process_count": ensure_ros_loop(reset_planning=True, force=True)}


@app.get("/api/sim-state")
def sim_state() -> dict[str, Any]:
    return {"frame": _require_sim().frame()}


@app.get("/decision", response_class=HTMLResponse)
def decision_page() -> str:
    return (STATIC_DIR / "decision.html").read_text()


@app.post("/api/decision/evaluate")
def decision_evaluate() -> dict[str, Any]:
    node = _require_sim()
    with node.lock:
        xy = [float(node.control_xy[0]), float(node.control_xy[1])]
        state = node.lab.world_state(xy, node.yaw_deg, node.config, node.world_mode)
        if node.plan_report is None or time.monotonic() - node.plan_received_at > 3:
            raise HTTPException(409, "No fresh planning report; start the ROS loop and retry")
        report = copy.deepcopy(node.plan_report)
        compact = compact_report(report)
        compact["snapshot_age_ms"] = round((time.monotonic() - node.plan_received_at) * 1000, 1)
        state["planning"] = compact
        if node.experiment_mode == 'model':
            if node.config.get('map_path') or node.world_mode != 'observed':
                raise HTTPException(409, 'Recovery experiments require observed synthetic scenes')
            state['recovery'] = {'strategies':proposals(xy,node.yaw_deg,node.config['target'],node.config['robot'],
                                                       node.lab.cells,node.lab.resolution,node.recovery.memory),
                                 'previous_attempts':copy.deepcopy(node.recovery.memory),
                                 'notes':['No synthetic object geometry is used to build model proposals.',
                                          'Execution collision checks use simulator geometry; unknown remains unknown.']}
            if node.information_version == 'rich':
                state = enrich(state,xy,node.yaw_deg,node.config['target'],node.config['robot'],node.lab.cells,node.lab.resolution,node.lab.samples)
        state["robot"]["recent"].pop("pattern", None)
        state["navigation"] = {"running": bool(node.running), "collision": bool(node.collision)}
        context = {"scenario": node.config.get("scenario_id", node.config.get("name")),
                   "world_mode": node.world_mode, "config_generation": node.config_generation, "planning_report": report, "scene_config": copy.deepcopy(node.config), "closed_loop_experiment": node.experiment_mode == "model", "information_version":node.information_version, "metrics": node.lab.metrics()}
    try:
        return OBSERVER.submit(state, context)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


def experiment_worker() -> None:
    node = _require_sim()
    while rclpy.ok():
        time.sleep(.25)
        with node.lock:
            if node.recovery.events:
                for event in node.recovery.events:
                    node.experiment_events.append({'t':node.lab.metrics()['elapsed_s'],**event})
                node.recovery.events.clear()
            if not node.running or not node.lab.run or node.lab.run["status"] != "running":
                node.override = None
                node.recovery.finish('run_stopped',node.control_xy,node.config['target'],time.monotonic())
                continue
            if node.recovery.active or time.monotonic() < node.recovery.settle_until:
                continue
            if node.recovery.memory and 'post_planner_goal_progress_m' not in node.recovery.memory[-1]:
                attempt = node.recovery.memory[-1]
                attempt['post_planner_goal_progress_m'] = round(math.dist(attempt['start_xy'],node.config['target'][:2])-math.dist(node.control_xy,node.config['target'][:2]),3)
            pending = node.experiment_pending
            if pending:
                record = OBSERVER.status()
                if record is None or record["id"] != pending:
                    node.experiment_pending = None
                    continue
                if record["status"] == "running":
                    continue
                age = time.monotonic()-node.plan_received_at if node.plan_received_at is not None else float("inf")
                command, reason = validate_strategy(record, node.control_xy, node.yaw_deg, node.config_generation, time.time(), age)
                if command:
                    node.recovery.start(command,node.control_xy,node.yaw_deg,node.config['target'],time.monotonic())
                node.experiment_events.append({"t":node.lab.metrics()["elapsed_s"], "event":reason,"decision_id":record["id"],"latency_ms":record.get("latency_ms"),"command":command,
                                               "confidence":record.get("response",{}).get("answers",{}).get("alternative",{}).get("confidence")})
                node.experiment_pending = None
                node.experiment_next_call = time.monotonic()+2
                continue
            if time.monotonic() < node.experiment_next_call or node.override:
                continue
            recent = node.lab.recent()
            if recent["window_s"] < 3 or recent["moved_m"] >= .1 or recent["turned_deg"] >= 15:
                continue
            current = OBSERVER.status()
            if current and current["status"] == "running":
                continue
        try:
            record = decision_evaluate()
        except HTTPException as exc:
            if exc.status_code not in (409,503):raise
            with node.lock:
                node.experiment_next_call = time.monotonic()+1
            continue
        with node.lock:
            node.experiment_pending = record["id"]
            node.experiment_events.append({"t":node.lab.metrics()["elapsed_s"],"event":"request","decision_id":record["id"]})


@app.get("/api/experiment/results")
def experiment_results(run: str = 'strategies') -> dict[str, Any]:
    folders = {'strategies':'closed_loop_strategies_20261003', 'short_actions':'closed_loop_20261003','rich_information':'information_rich_20261003'}
    if run not in folders:raise HTTPException(400, 'Unknown experiment')
    data = closed_loop_results(REPO_ROOT / 'tinynav_temp' / folders[run])
    data['expected_runs'] = 8 if run == 'rich_information' else 20
    if run == 'short_actions':data['live'] = []
    return data


@app.get("/api/experiment/status")
def experiment_status() -> dict[str, Any]:
    node = _require_sim()
    with node.lock:
        return {"mode":node.experiment_mode, "events":copy.deepcopy(node.experiment_events),"active_override":copy.deepcopy(node.override),"pending":node.experiment_pending,
                "policy_version":"recovery_sequences_v1", "information_version":node.information_version, "active_strategy":copy.deepcopy(node.recovery.active),
                "attempts":copy.deepcopy(node.recovery.memory)}


@app.post("/api/decision/compare")
def decision_compare(decision_id: str) -> dict[str, Any]:
    record = OBSERVER.status()
    if not record or record["status"] != "complete" or record["id"] != decision_id:
        raise HTTPException(409, "No completed decision to compare")
    try:
        result = compare(record)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    folder = REPO_ROOT / "tinynav_temp" / "candidate_branches"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (record["id"] + ".json")).write_text(json.dumps(result, indent=2))
    return result


@app.get("/api/planning/report")
def planning_report() -> dict[str, Any]:
    node = _require_sim()
    with node.lock:
        return {"report": copy.deepcopy(node.plan_report), "age_ms": round((time.monotonic()-node.plan_received_at)*1000, 1) if node.plan_received_at is not None else None,
                "config_generation": node.config_generation, "navigation_running": bool(node.running)}


@app.get("/api/decision/status")
def decision_status() -> dict[str, Any]:
    return {"record": OBSERVER.status(), "navigation_running": bool(SIM_NODE and SIM_NODE.running)}



class BaselineRequest(BaseModel):
    config: dict[str, Any]
    timeout_s: float = 120.0


@app.post("/api/baseline/start")
def baseline_start(request: BaselineRequest) -> dict[str, Any]:
    if not 5 <= request.timeout_s <= 600:
        raise HTTPException(400, "timeout_s must be between 5 and 600")
    node = _require_sim()
    with node.lock:
        node.running = False
    stop_children()
    with node.lock:
        node.set_config(request.config, reset=True)
        node.lab.begin(node.config, node.control_xy, node.yaw_deg, request.timeout_s)
        node.last_update = time.monotonic()
    ensure_ros_loop(reset_planning=True, force=True)
    with node.lock:
        node.running = True
    return {"ok": True, "metrics": node.lab.metrics()}


@app.post("/api/baseline/stop")
def baseline_stop() -> dict[str, Any]:
    node = _require_sim()
    with node.lock:
        if node.lab.run and node.lab.run["status"] == "running":
            node.lab.run["status"] = "cancelled"
            node.save_result()
        node.running = False
        node.last_cmd = Twist()
    return {"ok": True}


@app.get("/api/baseline/scenarios")
def baseline_scenarios() -> dict[str, Any]:
    return {"scenarios": json.loads((ROOT / "navigation_scenarios.json").read_text())}


@app.get("/api/baseline/results")
def baseline_results() -> dict[str, Any]:
    folder = REPO_ROOT / "tinynav_temp" / "navigation_lab"
    files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]
    records = [json.loads(path.read_text()) for path in files]
    results = [{**record["metrics"], "scenario": record["config"].get("scenario_id", record["config"].get("name", "custom"))} for record in records]
    groups: dict[str, Any] = {}
    for result in results:
        if result["status"] not in ("arrived", "collision", "timeout"):
            continue
        key = result["config_hash"] + ":" + str(result["timeout_s"])
        group = groups.setdefault(key, {"scenario": result.get("scenario", "custom"), "config_hash": result["config_hash"], "timeout_s": result["timeout_s"], "runs": 0, "arrived": 0})
        group["runs"] += 1
        group["arrived"] += int(result["status"] == "arrived")
        group["arrival_rate"] = group["arrived"] / group["runs"]
    return {"results": results, "groups": list(groups.values()), "limit": 100}


@app.get("/api/baseline/status")
def baseline_status() -> dict[str, Any]:
    node = _require_sim()
    with node.lock:
        return node.lab.metrics()


@app.get("/api/baseline/export")
def baseline_export() -> dict[str, Any]:
    node = _require_sim()
    with node.lock:
        return node.lab.export()


@app.post("/api/world-state/mode")
def world_state_mode(mode: str = "observed") -> dict[str, Any]:
    if mode not in ("observed", "full_scene"):
        raise HTTPException(400, "mode must be observed or full_scene")
    node = _require_sim()
    with node.lock:
        node.world_mode = mode
    return {"ok": True, "mode": mode}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=os.getenv("TINYNAV_WEB_HOST", "0.0.0.0"), port=int(os.getenv("TINYNAV_WEB_PORT", "8766")))


if __name__ == "__main__":
    main()
