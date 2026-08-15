#!/usr/bin/env python3
"""ROS-backed planning simulator web server.

Web UI edits the scene. This process publishes synthetic /slam/depth,
/slam/odometry(_visual), /control/target_pose, then mirrors outputs from the
real planning_node + cmd_vel_control loop.
"""

from __future__ import annotations

import copy
import math
import os
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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

from tinynav.core.planning_node import GO2_CONFIG, ObstacleConfig
from tool.simulator.planning_scene import (
    SimObject,
    box,
    cam_size,
    image_u8_payload,
    make_camera_pose_from_config,
    render_depth,
)

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "offline_planning_web"
PLANNING_ROBOT_DEFAULT = asdict(GO2_CONFIG)
PLANNING_OBSTACLE_DEFAULT = asdict(ObstacleConfig())


def default_config() -> dict[str, Any]:
    # Fences + ground plane give valid depth so free-space carving works when
    # boxes move (planning core stays unchanged).
    return {
        "name": "ros_planning_sim",
        "robot": copy.deepcopy(PLANNING_ROBOT_DEFAULT),
        "camera": {
            "width": 160,
            "image_height": 100,
            "fx": 80.0,
            "fy": 25.0,
            "max_range": 15.0,
            "mount_height": 0.45,
        },
        "start": {"xy": [0.0, 0.0], "yaw_deg": 0.0},
        "target": [4.0, 0.0, 0.0],
        "obstacle": copy.deepcopy(PLANNING_OBSTACLE_DEFAULT),
        "objects": [
            box("fence_back", [-0.35, 0.0, 0.6], [0.2, 10.2, 1.2]),
            box("fence_front", [9.35, 0.0, 0.6], [0.2, 10.2, 1.2]),
            box("fence_left", [4.5, 5.05, 0.6], [10.0, 0.2, 1.2]),
            box("fence_right", [4.5, -5.05, 0.6], [10.0, 0.2, 1.2]),
            box("left_wall", [2.0, 1.0, 0.35], [3.2, 0.25, 1.3]),
            box("right_wall", [2.0, -1.0, 0.35], [3.2, 0.25, 1.3]),
            box("center_box", [1.65, 0.0, 0.25], [0.45, 0.55, 0.5]),
        ],
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


class RosPlanningSimNode(Node):
    def __init__(self):
        super().__init__("tinynav_ros_planning_sim")
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.config = default_config()
        self.control_xy = list(self.config["start"]["xy"])
        self.yaw_deg = float(self.config["start"]["yaw_deg"])
        self.last_update = time.monotonic()
        self.last_depth = np.zeros((100, 160), dtype=np.float32)
        self.last_cmd = Twist()
        self.last_path: list[list[float]] = []
        self.last_footprint: list[list[float]] = []
        self.last_obstacle_mask: dict[str, Any] | None = None
        self.last_esdf_grid: dict[str, Any] | None = None
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

    def set_config(self, config: dict[str, Any], reset: bool = False) -> None:
        with self.lock:
            self.config = copy.deepcopy(config)
            self.config["robot"] = copy.deepcopy(PLANNING_ROBOT_DEFAULT)
            self.config["obstacle"] = copy.deepcopy(PLANNING_OBSTACLE_DEFAULT)
            if reset:
                self.control_xy = list(self.config.get("start", {}).get("xy", [0.0, 0.0]))
                self.yaw_deg = float(self.config.get("start", {}).get("yaw_deg", 0.0))
                self.last_cmd = Twist()
                self.last_path = []
                self.last_footprint = []
                self.last_obstacle_mask = None
                self.last_esdf_grid = None

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

        objects = [SimObject(**obj) for obj in config.get("objects", [])]
        T_cam = make_camera_pose_from_config(config["start"]["xy"], yaw_deg, config["robot"], config["camera"])
        depth = render_depth(objects, T_cam, config["camera"])
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
        self.publish_target(stamp, config)
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
                "selected_param": [float(self.last_cmd.linear.x), float(self.last_cmd.angular.z)],
                "depth_u8": image_u8_payload(self.last_depth, 0.0, float(self.config["camera"]["max_range"])),
                "obstacle_u8": self.last_obstacle_mask,
                "esdf_u8": self.last_esdf_grid,
                "next_start": {"xy": xy, "yaw_deg": float(self.yaw_deg)},
            }


class RunRequest(BaseModel):
    config: dict[str, Any]
    reset: bool | None = None


app = FastAPI(title="TinyNav ROS Planning Simulator")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
SIM_NODE: RosPlanningSimNode | None = None
EXECUTOR: MultiThreadedExecutor | None = None
PROCS: list[subprocess.Popen] = []
REPO_ROOT = ROOT.parents[1]
CHILD_SCRIPTS = (
    "tinynav/core/planning_node.py",
    "tinynav/platforms/cmd_vel_control.py",
)
_LAST_PLANNING_RESET = 0.0
_PLANNING_RESET_COOLDOWN_S = 1.0


def start_ros() -> None:
    global SIM_NODE, EXECUTOR
    if SIM_NODE is not None:
        return
    rclpy.init(args=None)
    SIM_NODE = RosPlanningSimNode()
    EXECUTOR = MultiThreadedExecutor(num_threads=4)
    EXECUTOR.add_node(SIM_NODE)
    threading.Thread(target=EXECUTOR.spin, daemon=True).start()


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(key, "1")
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


@app.post("/api/realtime-step")
def realtime_step(request: RunRequest) -> dict[str, Any]:
    if SIM_NODE is None:
        raise HTTPException(status_code=503, detail="ROS simulator is not ready")
    if not isinstance(request.config, dict):
        raise HTTPException(status_code=400, detail="config must be an object")
    reset = bool(request.reset)
    SIM_NODE.set_config(copy.deepcopy(request.config), reset=reset)
    # Scene edits set reset=true; restart planning so occupancy/ESDF is not sticky.
    if reset:
        ensure_ros_loop(reset_planning=True, force=False)
    return {"frame": SIM_NODE.frame()}


@app.post("/api/start-ros-loop")
def start_ros_loop() -> dict[str, Any]:
    return {"ok": True, "process_count": ensure_ros_loop(reset_planning=True, force=True)}


@app.get("/api/sim-state")
def sim_state() -> dict[str, Any]:
    if SIM_NODE is None:
        raise HTTPException(status_code=503, detail="ROS simulator is not ready")
    return {"frame": SIM_NODE.frame()}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8766)


if __name__ == "__main__":
    main()
