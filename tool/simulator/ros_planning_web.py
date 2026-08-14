#!/usr/bin/env python3
"""ROS-backed planning simulator web server.

The web UI owns the editable scene. This process publishes synthetic
/slam/depth, /slam/odometry_visual, /slam/odometry and /control/target_pose,
then renders outputs from the real planning_node + cmd_vel_control loop.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from cv_bridge import CvBridge
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from geometry_msgs.msg import Point32, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path as RosPath
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, PointCloud
from std_msgs.msg import Bool
from pydantic import BaseModel

from tinynav.core.planning_node import GO2_CONFIG, ObstacleConfig


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "offline_planning_web"
PLANNING_ROBOT_DEFAULT = asdict(GO2_CONFIG)
PLANNING_OBSTACLE_DEFAULT = asdict(ObstacleConfig())


@dataclass
class SimObject:
    name: str
    kind: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(self.center, dtype=np.float64)
        half = np.asarray(self.size, dtype=np.float64) / 2.0
        return center - half, center + half


def default_config() -> dict[str, Any]:
    return {
        "name": "ros_planning_sim",
        "robot": copy.deepcopy(PLANNING_ROBOT_DEFAULT),
        "camera": {
            "width": 160,
            "image_height": 100,
            "fx": 120.0,
            "fy": 120.0,
            "max_range": 6.0,
            "mount_height": 0.45,
        },
        "start": {"xy": [0.0, 0.0], "yaw_deg": 0.0},
        "target": [4.0, 0.0, 0.0],
        "obstacle": copy.deepcopy(PLANNING_OBSTACLE_DEFAULT),
        "objects": [
            {"name": "left_wall", "kind": "box", "center": [2.0, 1.0, 0.35], "size": [3.2, 0.25, 1.3]},
            {"name": "right_wall", "kind": "box", "center": [2.0, -1.0, 0.35], "size": [3.2, 0.25, 1.3]},
            {"name": "center_box", "kind": "box", "center": [1.65, 0.0, 0.25], "size": [0.45, 0.55, 0.5]},
        ],
    }


def make_camera_pose(control_xy: list[float], yaw_deg: float, robot: dict[str, Any], cam: dict[str, Any]) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0], dtype=np.float64)
    down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    rot = np.column_stack([right, down, forward])
    pos = np.array([control_xy[0], control_xy[1], 0.0], dtype=np.float64)
    pos += forward * (float(robot.get("camera_x", 0.0)) - float(robot.get("control_x", 0.0)))
    pos += left * (float(robot.get("camera_y", 0.0)) - float(robot.get("control_y", 0.0)))
    pos[2] = float(cam.get("mount_height", 0.45))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def render_depth(objects: list[SimObject], T_cam_to_world: np.ndarray, cam: dict[str, Any]) -> np.ndarray:
    width = int(cam["width"])
    height = int(cam.get("image_height", cam.get("height", 100)))
    fx = float(cam["fx"])
    fy = float(cam["fy"])
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    max_range = float(cam["max_range"])
    us, vs = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    rays_cam = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], axis=-1)
    rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    rays_world = rays_cam @ T_cam_to_world[:3, :3].T
    rays_flat = rays_world.reshape((-1, 3))
    z_flat = rays_cam[..., 2].reshape(-1)
    best = np.full(rays_flat.shape[0], np.inf, dtype=np.float64)
    origin = T_cam_to_world[:3, 3]

    for obj in objects:
        box_min, box_max = obj.bounds
        inv_d = np.divide(1.0, rays_flat, out=np.full_like(rays_flat, np.inf), where=np.abs(rays_flat) > 1e-9)
        t0 = (box_min - origin) * inv_d
        t1 = (box_max - origin) * inv_d
        t_near = np.maximum.reduce(np.minimum(t0, t1), axis=1)
        t_far = np.minimum.reduce(np.maximum(t0, t1), axis=1)
        hit = np.where(t_near > 0.0, t_near, t_far)
        valid = (t_far >= 0.0) & (t_near <= t_far) & (hit > 0.0) & (hit <= max_range)
        best = np.where(valid & (hit < best), hit, best)

    depth = np.zeros(best.shape[0], dtype=np.float32)
    finite = np.isfinite(best)
    depth[finite] = (best[finite] * z_flat[finite]).astype(np.float32)
    return depth.reshape((height, width))


def image_u8_payload(image: np.ndarray, vmin: float, vmax: float) -> dict[str, Any]:
    clipped = np.clip((image.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    u8 = np.round(clipped * 255.0).astype(np.uint8)
    return {"width": int(u8.shape[1]), "height": int(u8.shape[0]), "data": u8.ravel().tolist()}


def odom_from_T(T: np.ndarray, stamp, frame_id: str = "world", child_frame_id: str = "camera") -> Odometry:
    quat = R.from_matrix(T[:3, :3]).as_quat()
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    msg.pose.pose.position.x = float(T[0, 3])
    msg.pose.pose.position.y = float(T[1, 3])
    msg.pose.pose.position.z = float(T[2, 3])
    msg.pose.pose.orientation.x = float(quat[0])
    msg.pose.pose.orientation.y = float(quat[1])
    msg.pose.pose.orientation.z = float(quat[2])
    msg.pose.pose.orientation.w = float(quat[3])
    return msg


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
        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.nav_active_pub = self.create_publisher(Bool, "/nav/active", latched_qos)
        self.nav_paused_pub = self.create_publisher(Bool, "/nav/paused", latched_qos)

        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(RosPath, "/planning/trajectory_path", self.path_callback, 10)
        self.create_subscription(PointCloud, "/planning/footprint", self.footprint_callback, 10)
        self.create_subscription(OccupancyGrid, "/planning/obstacle_mask", self.obstacle_callback, 10)
        self.create_subscription(OccupancyGrid, "/planning/occupancy_grid", self.esdf_callback, 10)

        self.timer = self.create_timer(1.0 / 8.0, self.tick)

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
            # PlanningNode publishes 21 points per edge. Keep corners for UI.
            points = msg.points
            if len(points) >= 64:
                idxs = [0, 21, 42, 63, 0]
                self.last_footprint = [[float(points[i].x), float(points[i].y)] for i in idxs]
            else:
                self.last_footprint = [[float(p.x), float(p.y)] for p in points]

    def obstacle_callback(self, msg: OccupancyGrid) -> None:
        data = np.asarray(msg.data, dtype=np.int16).reshape((msg.info.width, msg.info.height), order="F")
        u8 = np.where(data > 0, 255, 0).astype(np.uint8)
        with self.lock:
            self.last_obstacle_mask = {
                "width": int(u8.shape[1]),
                "height": int(u8.shape[0]),
                "data": u8.ravel().tolist(),
                "origin": [float(msg.info.origin.position.x), float(msg.info.origin.position.y)],
                "resolution": float(msg.info.resolution),
            }

    def esdf_callback(self, msg: OccupancyGrid) -> None:
        data = np.asarray(msg.data, dtype=np.int16).reshape((msg.info.width, msg.info.height), order="F")
        risk = np.clip(data, 0, 120).astype(np.float32) / 120.0
        clearance_u8 = np.round((1.0 - risk) * 255.0).astype(np.uint8)
        with self.lock:
            self.last_esdf_grid = {
                "width": int(clearance_u8.shape[1]),
                "height": int(clearance_u8.shape[0]),
                "data": clearance_u8.ravel().tolist(),
                "origin": [float(msg.info.origin.position.x), float(msg.info.origin.position.y)],
                "resolution": float(msg.info.resolution),
            }

    def publish_camera_info(self, stamp, config: dict[str, Any]) -> None:
        cam = config["camera"]
        width = int(cam["width"])
        height = int(cam.get("image_height", cam.get("height", 100)))
        fx = float(cam["fx"])
        fy = float(cam["fy"])
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        msg.width = width
        msg.height = height
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
        vx = float(self.last_cmd.linear.x)
        wz = float(self.last_cmd.angular.z)
        yaw = math.radians(self.yaw_deg)
        self.control_xy[0] += math.cos(yaw) * vx * dt
        self.control_xy[1] += math.sin(yaw) * vx * dt
        self.yaw_deg = (self.yaw_deg + math.degrees(wz * dt) + 180.0) % 360.0 - 180.0

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
        T_cam = make_camera_pose(config["start"]["xy"], yaw_deg, config["robot"], config["camera"])
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
            cam = self.config["camera"]
            return {
                "robot_xy": [float(self.control_xy[0]), float(self.control_xy[1])],
                "robot_yaw_deg": float(self.yaw_deg),
                "robot_footprint_xy": copy.deepcopy(self.last_footprint),
                "selected_trajectory_xy": copy.deepcopy(self.last_path),
                "candidate_trajectories_xy": [],
                "selected_param": [float(self.last_cmd.linear.x), float(self.last_cmd.angular.z)],
                "front_clearance": 0.0,
                "valid_trajectories": len(self.last_path),
                "obstacle_cells": int(sum(1 for v in (self.last_obstacle_mask or {}).get("data", []) if v)),
                "depth_u8": image_u8_payload(self.last_depth, 0.0, float(cam["max_range"])),
                "obstacle_u8": self.last_obstacle_mask,
                "esdf_u8": self.last_esdf_grid,
                "next_start": {"xy": [float(self.control_xy[0]), float(self.control_xy[1])], "yaw_deg": float(self.yaw_deg)},
            }


class RunRequest(BaseModel):
    config: dict[str, Any]
    reset: bool | None = None
    advance_step: int | None = None


app = FastAPI(title="TinyNav ROS Planning Simulator")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
SIM_NODE: RosPlanningSimNode | None = None
EXECUTOR: MultiThreadedExecutor | None = None
ROS_THREAD: threading.Thread | None = None
PROCS: list[subprocess.Popen] = []


def start_ros() -> None:
    global SIM_NODE, EXECUTOR, ROS_THREAD
    if SIM_NODE is not None:
        return
    rclpy.init(args=None)
    SIM_NODE = RosPlanningSimNode()
    EXECUTOR = MultiThreadedExecutor(num_threads=4)
    EXECUTOR.add_node(SIM_NODE)
    ROS_THREAD = threading.Thread(target=EXECUTOR.spin, daemon=True)
    ROS_THREAD.start()


@app.on_event("startup")
def startup() -> None:
    start_ros()


@app.on_event("shutdown")
def shutdown() -> None:
    for proc in PROCS:
        proc.terminate()
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
    SIM_NODE.set_config(json.loads(json.dumps(request.config)), reset=bool(request.reset))
    return {"frame": SIM_NODE.frame()}


@app.post("/api/start-ros-loop")
def start_ros_loop() -> dict[str, Any]:
    cwd = ROOT.parents[1]
    child_env = os.environ.copy()
    child_env.setdefault("OPENBLAS_NUM_THREADS", "1")
    child_env.setdefault("OMP_NUM_THREADS", "1")
    child_env.setdefault("MKL_NUM_THREADS", "1")
    child_env.setdefault("NUMEXPR_NUM_THREADS", "1")
    if not any(proc.poll() is None and "planning_node.py" in " ".join(proc.args) for proc in PROCS):
        PROCS.append(subprocess.Popen(["uv", "run", "python", "tinynav/core/planning_node.py"], cwd=str(cwd), env=child_env))
    if not any(proc.poll() is None and "cmd_vel_control.py" in " ".join(proc.args) for proc in PROCS):
        PROCS.append(subprocess.Popen(["uv", "run", "python", "tinynav/platforms/cmd_vel_control.py"], cwd=str(cwd), env=child_env))
    return {"ok": True, "process_count": sum(1 for proc in PROCS if proc.poll() is None)}


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
