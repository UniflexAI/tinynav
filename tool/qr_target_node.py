#!/usr/bin/env python3
"""Navigate to a fixed position relative to a 4-tag AprilTag board.

Two procedures
--------------
  Record (python qr_target_node.py record):
      Collects INIT_FRAMES board detections, pose-graph solves T_world_qrworld, saves T_qrworld_robot
      to tag_target.json, and exits.

  Navigate (python qr_target_node.py):
      Collects INIT_FRAMES board detections to initialize T_world_goal, then
      drives the robot toward the goal using a proportional controller on
      /cmd_vel. Control runs at odom rate — board visibility not required after
      initialization.

Frame conventions
-----------------
  world          — SLAM/odometry world frame
  qr_world       — fixed frame anchored to the physical board
  T_world_qrworld  — where the board is in odometry world
  T_qrworld_robot  — goal robot pose in the board frame
  T_world_goal     — T_world_qrworld @ T_qrworld_robot  (fixed after init)

Topics
------
  Subscribed:  /slam/odometry                         nav_msgs/Odometry
               /camera/camera/infra1/image_rect_raw   sensor_msgs/Image
               /camera/camera/infra1/camera_info      sensor_msgs/CameraInfo
  Published:   /cmd_vel                               geometry_msgs/Twist
               /mapping/nav_done                      std_msgs/Bool
  TF:          world → qr_world  (nav mode, after INIT_FRAMES)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster

from tinynav.core.math_utils import msg2np, np2msg, np2tf
from tinynav.tinynav_cpp_bind import pose_graph_solve

DB_DIR          = Path("tinynav_db/qrcode")
TAG_PARAMS_PATH = DB_DIR / "tag_satellite.json"
TARGET_PATH     = DB_DIR / "tag_target.json"

ODOM_TOPIC    = "/slam/odometry"
IMAGE_TOPIC   = "/camera/camera/infra1/image_rect_raw"
INFO_TOPIC    = "/camera/camera/infra1/camera_info"
CMD_VEL_TOPIC = "/cmd_vel"
TARGET_POSE_TOPIC = "/qr_world/target_pose"
NAV_DONE_TOPIC = "/qr_world/nav_done"

ARUCO_DICT  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
FAMILY_NAME = "DICT_APRILTAG_36h11"

INIT_FRAMES = 10

# PI controller gains, limits, and drivetrain deadband compensation.
K_LINEAR        = 0.5   # (m/s) / m
K_LINEAR_I      = 0.05  # (m/s) / (m*s)
K_ANGULAR       = 1.0   # (rad/s) / rad
K_ANGULAR_I     = 0.10  # (rad/s) / (rad*s)
MAX_LINEAR      = 0.3   # m/s
MAX_LINEAR_I    = 0.05  # m/s — max integral contribution
MAX_ANGULAR     = 0.5   # rad/s
MAX_ANGULAR_I   = 0.10  # rad/s — max integral contribution
MIN_LINEAR      = 0.15  # m/s — below this the base may not move
MIN_ANGULAR     = 0.15  # rad/s — below this the base may not rotate
DIST_THRESH     = 0.06  # m  — switch from approach to heading-align
HEADING_THRESH  = 0.06  # rad — stop when aligned
CMD_DEADBAND    = 1e-3
MAX_CONTROL_DT  = 0.1   # s — cap integral step after odom stalls

# LooperBridge publishes /slam/odometry as T_world_camera:
#   tool/looper_bridge_node.py uses np2msg(T_world_camera, ..., "world", "camera").
# Convert it to a robot/control frame before recording or controlling.
#
# Frame conventions:
#   camera (OpenCV/Looper): +x right, +y down, +z forward
#   robot/control:         +x forward, +y left, +z up
#
# T_CAMERA_ROBOT maps robot-frame points into the camera frame. Its rotation is
# the same convention used by tinynav/platforms/cmd_vel_control.py when it
# right-multiplies a camera pose to obtain a robot pose.
CAMERA_FORWARD_FROM_ROBOT_M = 0.00
CAMERA_LEFT_FROM_ROBOT_M = 0.0
CAMERA_UP_FROM_ROBOT_M = 0.0
R_CAMERA_ROBOT = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
T_CAMERA_ROBOT = np.eye(4, dtype=np.float64)
T_CAMERA_ROBOT[:3, :3] = R_CAMERA_ROBOT
T_CAMERA_ROBOT[:3, 3] = -R_CAMERA_ROBOT @ np.array(
    [
        CAMERA_FORWARD_FROM_ROBOT_M,
        CAMERA_LEFT_FROM_ROBOT_M,
        CAMERA_UP_FROM_ROBOT_M,
    ],
    dtype=np.float64,
)


def _make_board(tag_ids: list[int], size_m: float,
                spacing_m: float) -> cv2.aruco.GridBoard:
    return cv2.aruco.GridBoard(
        size=(2, 2),
        markerLength=size_m,
        markerSeparation=spacing_m,
        dictionary=ARUCO_DICT,
        ids=np.array(tag_ids, dtype=np.int32),
    )


def _load_tag_params() -> tuple[str, list[int], float, float]:
    if not TAG_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"{TAG_PARAMS_PATH} not found. Run apriltag_generate.py first."
        )
    d = json.loads(TAG_PARAMS_PATH.read_text())
    return d["tag_family"], d["tag_ids"], float(d["size_m"]), float(d["spacing_m"])


def _save_target(family: str, tag_ids: list[int], size_m: float,
                 spacing_m: float, T_qrworld_robot: np.ndarray) -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_PATH.write_text(json.dumps({
        "tag_family":      family,
        "tag_ids":         tag_ids,
        "size_m":          size_m,
        "spacing_m":       spacing_m,
        "T_qrworld_robot": T_qrworld_robot.tolist(),
    }, indent=2))


def _load_target() -> tuple[str, list[int], float, float, np.ndarray]:
    if not TARGET_PATH.exists():
        raise FileNotFoundError(
            f"{TARGET_PATH} not found. Run in record mode first."
        )
    d = json.loads(TARGET_PATH.read_text())
    return (d["tag_family"], d["tag_ids"], float(d["size_m"]),
            float(d["spacing_m"]), np.array(d["T_qrworld_robot"]))


def _camera_pose_to_robot_pose(T_world_camera: np.ndarray) -> np.ndarray:
    return T_world_camera @ T_CAMERA_ROBOT


def _clip_with_min(value: float, min_abs: float, max_abs: float) -> float:
    if abs(value) < CMD_DEADBAND:
        return 0.0
    clipped = float(np.clip(value, -max_abs, max_abs))
    if abs(clipped) < min_abs:
        return float(np.sign(clipped) * min_abs)
    return clipped


def _solve_T_world_qrworld(T_world_board_samples: list[np.ndarray]) -> np.ndarray:
    """Estimate T_world_qrworld via pose_graph_solve over N board observations.

    Mirrors map_node.compute_transform_from_map_to_odom:
      node 0 = T_world_qrworld (unknown)
      node 1 = eye             (fixed odom-world anchor)
    Each detection contributes one edge (0, 1, T_obs).
    """
    optimized = {0: np.eye(4), 1: np.eye(4)}
    constraints = [
        (0, 1, T_obs, np.array([10.0, 10.0, 10.0]), np.array([10.0, 10.0, 10.0]))
        for T_obs in T_world_board_samples
    ]
    return pose_graph_solve(optimized, constraints, {1: True}, max_iteration_num=100)[0]


def _estimate_board_pose(corners, ids, board, K, dist) -> np.ndarray | None:
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) == 0:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R_mat, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3]  = tvec.ravel()
    return T


class BoardBaseNode(Node):
    def __init__(self, node_name: str, tag_ids: list[int],
                 size_m: float, spacing_m: float):
        super().__init__(node_name)
        self._board = _make_board(tag_ids, size_m, spacing_m)
        self._K:              np.ndarray | None = None
        self._dist:           np.ndarray | None = None
        self._T_world_camera: np.ndarray | None = None
        self._bridge = CvBridge()

        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self._detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)

        self.create_subscription(Odometry, ODOM_TOPIC, self._odom_callback, 100)
        self._info_sub = self.create_subscription(CameraInfo, INFO_TOPIC, self._info_callback, 10)
        self.create_subscription(Image, IMAGE_TOPIC, self._image_callback, 10)

    def _odom_callback(self, msg: Odometry) -> None:
        self._T_world_camera, _ = msg2np(msg)
        self._on_odom()

    def _on_odom(self) -> None:
        pass

    def _info_callback(self, msg: CameraInfo) -> None:
        self._K    = np.array(msg.k).reshape(3, 3)
        self._dist = np.array(msg.d)
        self.destroy_subscription(self._info_sub)
        self.get_logger().info("Camera intrinsics received.")

    def _image_callback(self, msg: Image) -> None:
        log = self.get_logger()
        if self._K is None:
            log.warn("Waiting for camera_info ...", throttle_duration_sec=5.0)
            return
        if self._T_world_camera is None:
            log.warn("Waiting for odometry ...", throttle_duration_sec=5.0)
            return

        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        corners, ids, _ = self._detector.detectMarkers(img)

        if ids is None:
            log.warn("No AprilTags detected.", throttle_duration_sec=5.0)
            return

        board_ids = set(self._board.getIds().flatten().tolist())
        ids_flat  = ids.flatten().tolist()
        n_board_visible = sum(1 for tid in ids_flat if tid in board_ids)
        if n_board_visible == 0:
            log.warn(
                f"No board tags visible (saw ids={ids_flat}, board ids={sorted(board_ids)})",
                throttle_duration_sec=5.0,
            )
            return

        T_camera_board = _estimate_board_pose(corners, ids, self._board, self._K, self._dist)
        if T_camera_board is None:
            log.warn("Board pose estimation failed.", throttle_duration_sec=5.0)
            return

        self._on_board_detected(T_camera_board, n_board_visible, msg.header.stamp)

    def _on_board_detected(self, T_camera_board: np.ndarray,
                           n_tags_used: int, stamp) -> None:
        raise NotImplementedError


class RecordNode(BoardBaseNode):
    def __init__(self, tag_ids: list[int], size_m: float, spacing_m: float):
        super().__init__("qr_target_record", tag_ids, size_m, spacing_m)
        self._tag_ids             = tag_ids
        self._size_m              = size_m
        self._spacing_m           = spacing_m
        self._T_world_board_samples: list[np.ndarray] = []
        self._done = False
        self.get_logger().info(
            f"Record mode: point camera at the board (ids={tag_ids}). "
            f"Collecting {INIT_FRAMES} frames ..."
        )

    def _on_board_detected(self, T_camera_board: np.ndarray,
                           n_tags_used: int, stamp) -> None:
        self._T_world_board_samples.append(self._T_world_camera @ T_camera_board)
        n = len(self._T_world_board_samples)
        self.get_logger().info(f"Record: {n}/{INIT_FRAMES} frames ({n_tags_used} tags).")
        if n < INIT_FRAMES:
            return
        T_world_qrworld = _solve_T_world_qrworld(self._T_world_board_samples)
        T_world_robot = _camera_pose_to_robot_pose(self._T_world_camera)
        T_qrworld_robot = np.linalg.inv(T_world_qrworld) @ T_world_robot
        _save_target(FAMILY_NAME, self._tag_ids, self._size_m,
                     self._spacing_m, T_qrworld_robot)
        self._done = True
        self.get_logger().info(
            f"Recorded robot pose in qr_world frame ({INIT_FRAMES} frames, pose-graph solved) -> {TARGET_PATH}"
        )


def run_record() -> None:
    _, tag_ids, size_m, spacing_m = _load_tag_params()
    rclpy.init()
    node = RecordNode(tag_ids, size_m, spacing_m)
    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


class NavNode(BoardBaseNode):
    def __init__(self, tag_ids: list[int], size_m: float,
                 spacing_m: float, T_qrworld_robot: np.ndarray):
        super().__init__("qr_target_nav", tag_ids, size_m, spacing_m)
        self._T_qrworld_robot          = T_qrworld_robot
        self._T_world_qrworld_samples: list[np.ndarray] = []
        self._T_world_goal:            np.ndarray | None = None
        self._linear_error_i = np.zeros(2, dtype=np.float64)
        self._heading_error_i = 0.0
        self._last_control_time: float | None = None
        self._reached = False
        self._done = False
        self._cmd_pub        = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self._nav_done_pub = self.create_publisher(Bool, NAV_DONE_TOPIC, 10)
        self._target_pose_pub = self.create_publisher(Odometry, TARGET_POSE_TOPIC, 10)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info(
            f"Navigation mode: board ids={tag_ids}. "
            f"Collecting {INIT_FRAMES} initial frames to initialize qr_world ..."
        )

    def _publish_reached_once(self) -> None:
        if self._reached:
            return
        self._reached = True
        self._nav_done_pub.publish(Bool(data=True))
        self._done = True
        self.get_logger().info("qr target reached.")

    def _control_dt(self) -> float:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_control_time is None:
            self._last_control_time = now
            return 0.0
        dt = min(max(now - self._last_control_time, 0.0), MAX_CONTROL_DT)
        self._last_control_time = now
        return dt

    def _linear_pi_cmd(self, dx: float, dy: float, dt: float) -> tuple[float, float]:
        self._linear_error_i += np.array([dx, dy], dtype=np.float64) * dt
        linear_i_limit = MAX_LINEAR_I / K_LINEAR_I
        self._linear_error_i = np.clip(
            self._linear_error_i,
            -linear_i_limit,
            linear_i_limit,
        )
        return (
            _clip_with_min(
                K_LINEAR * dx + K_LINEAR_I * self._linear_error_i[0],
                MIN_LINEAR,
                MAX_LINEAR,
            ),
            _clip_with_min(
                K_LINEAR * dy + K_LINEAR_I * self._linear_error_i[1],
                MIN_LINEAR,
                MAX_LINEAR,
            ),
        )

    def _heading_pi_cmd(self, heading_error: float, dt: float) -> float:
        self._heading_error_i += heading_error * dt
        angular_i_limit = MAX_ANGULAR_I / K_ANGULAR_I
        self._heading_error_i = float(np.clip(
            self._heading_error_i,
            -angular_i_limit,
            angular_i_limit,
        ))
        return _clip_with_min(
            K_ANGULAR * heading_error + K_ANGULAR_I * self._heading_error_i,
            MIN_ANGULAR,
            MAX_ANGULAR,
        )

    def _on_board_detected(self, T_camera_board: np.ndarray,
                           n_tags_used: int, stamp) -> None:
        if self._T_world_goal is not None:
            return
        self._T_world_qrworld_samples.append(self._T_world_camera @ T_camera_board)
        n = len(self._T_world_qrworld_samples)
        self.get_logger().info(
            f"qr_world init: {n}/{INIT_FRAMES} frames ({n_tags_used} tags)."
        )
        if n < INIT_FRAMES:
            return
        T_world_qrworld = _solve_T_world_qrworld(self._T_world_qrworld_samples)
        self._T_world_goal = T_world_qrworld @ self._T_qrworld_robot
        self._target_pose_pub.publish(
            np2msg(self._T_world_goal, stamp, "world", "qr_target_pose")
        )
        self._tf_broadcaster.sendTransform(
            np2tf(T_world_qrworld, stamp, "world", "qr_world")
        )
        self.get_logger().info(f"qr_world initialized — driving to goal via {CMD_VEL_TOPIC}")

    def _on_odom(self) -> None:
        if self._T_world_goal is None:
            return

        # error in robot frame
        T_world_robot = _camera_pose_to_robot_pose(self._T_world_camera)
        T_robot_goal = np.linalg.inv(T_world_robot) @ self._T_world_goal
        dx      = T_robot_goal[0, 3]
        dy      = T_robot_goal[1, 3]
        dist    = np.hypot(dx, dy)
        print(f"dist: {dist}")
        heading_error = np.arctan2(T_robot_goal[1, 0], T_robot_goal[0, 0])
        print(f"heading_error: {heading_error}")
        dt = self._control_dt()

        cmd = Twist()
        if dist > DIST_THRESH:
            self._heading_error_i = 0.0
            cmd.linear.x, cmd.linear.y = self._linear_pi_cmd(dx, dy, dt)
        elif abs(heading_error) > HEADING_THRESH:
            self._linear_error_i[:] = 0.0
            cmd.angular.z = self._heading_pi_cmd(heading_error, dt)
        else:
            self._linear_error_i[:] = 0.0
            self._heading_error_i = 0.0
            self._publish_reached_once()
        self._cmd_pub.publish(cmd)


def run_nav() -> None:
    _, tag_ids, size_m, spacing_m, T_qrworld_robot = _load_target()
    rclpy.init()
    node = NavNode(tag_ids, size_m, spacing_m, T_qrworld_robot)
    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        run_record()
    else:
        run_nav()


main()
