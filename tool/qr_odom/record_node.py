#!/usr/bin/env python3
"""
tool/qr_odom/record_node.py

One-shot calibration node: point the camera at the AprilTag board and run this
to write two calibration files used by odom_node.py and nav_node.py.

Outputs
-------
  tinynav_db/qrcode/tag_target.json
      T_qrworld_robot — goal robot pose in board frame
      (used by nav_node.py to define the target)

  tinynav_db/qrcode/tag_mappose.json
      T_map_qrworld   — board pose in map frame
      (used by odom_node.py to compute T_world_camera from detections)
      Written only when the world→map TF is available (map_node must be running).

Frame chain
-----------
  Collected over RECORD_FRAMES valid detections:
    T_world_board[i] = T_world_camera[i] @ T_camera_board[i]

  Averaged:
    T_world_qrworld  = mean(T_world_board[i])

  At collection end (latest odometry):
    T_world_robot    = T_world_camera_latest @ T_CAMERA_ROBOT
    T_qrworld_robot  = inv(T_world_qrworld) @ T_world_robot

  From TF (if available):
    T_world_map      = lookup world→map
    T_map_qrworld    = inv(T_world_map) @ T_world_qrworld

Detection quality gate
----------------------
  - Visible board tags   ≥ MIN_TAGS   (2)
  - solvePnPRansac reproj error < MAX_REPROJ_PX (3.0 px)

Usage
-----
  python tool/qr_odom/record_node.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from tinynav.core.math_utils import msg2np, tf2np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DB_DIR           = Path("tinynav_db/qrcode")
TAG_PARAMS_PATH  = DB_DIR / "tag_satellite.json"
TARGET_PATH      = DB_DIR / "tag_target.json"
MAPPOSE_PATH     = DB_DIR / "tag_mappose.json"

# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------

ODOM_TOPIC   = "/slam/odometry"
IMAGE_TOPIC  = "/camera/camera/infra1/image_rect_raw"
INFO_TOPIC   = "/camera/camera/infra1/camera_info"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

RECORD_FRAMES = 20      # valid detections to collect before solving
MIN_TAGS      = 2       # minimum visible board tags per frame
MAX_REPROJ_PX = 3.0     # reprojection error threshold (px)

# camera → robot frame (same convention as target_node.py / nav_node.py)
R_CAMERA_ROBOT = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=np.float64)
T_CAMERA_ROBOT = np.eye(4, dtype=np.float64)
T_CAMERA_ROBOT[:3, :3] = R_CAMERA_ROBOT


# ---------------------------------------------------------------------------
# Rotation averaging (simple, no C++ dependency)
# ---------------------------------------------------------------------------

def _mean_T(samples: list[np.ndarray]) -> np.ndarray:
    """Average a list of 4×4 SE3 matrices: mean position + mean rotation."""
    p_mean = np.mean([T[:3, 3] for T in samples], axis=0)
    rots   = Rotation.concatenate([Rotation.from_matrix(T[:3, :3]) for T in samples])
    R_mean = rots.mean().as_matrix()
    T_mean = np.eye(4)
    T_mean[:3, :3] = R_mean
    T_mean[:3,  3] = p_mean
    return T_mean


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------

class RecordNode(Node):
    def __init__(self):
        super().__init__("qr_record_node")

        if not TAG_PARAMS_PATH.exists():
            raise FileNotFoundError(
                f"{TAG_PARAMS_PATH} not found — run generate.py first."
            )
        d = json.loads(TAG_PARAMS_PATH.read_text())
        self._tag_ids   = d["tag_ids"]
        self._size_m    = float(d["size_m"])
        self._spacing_m = float(d["spacing_m"])

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._board = cv2.aruco.GridBoard(
            size=(2, 2),
            markerLength=self._size_m,
            markerSeparation=self._spacing_m,
            dictionary=aruco_dict,
            ids=np.array(self._tag_ids, dtype=np.int32),
        )
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        self._K:    np.ndarray | None = None
        self._dist: np.ndarray | None = None
        self._T_world_camera: np.ndarray | None = None
        self._bridge = CvBridge()

        self._samples: list[np.ndarray] = []   # T_world_board per valid frame
        self.done = False

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._info_sub = self.create_subscription(
            CameraInfo, INFO_TOPIC, self._info_cb, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self._odom_cb, 100)
        self.create_subscription(Image,    IMAGE_TOPIC, self._image_cb, 10)

        self.get_logger().info(
            f"qr_record_node ready — collecting {RECORD_FRAMES} frames "
            f"(tag_ids={self._tag_ids}). Point camera at the board."
        )

    # ---- subscribers ----

    def _info_cb(self, msg: CameraInfo) -> None:
        self._K    = np.array(msg.k).reshape(3, 3)
        self._dist = np.array(msg.d)
        self.destroy_subscription(self._info_sub)
        self.get_logger().info("Camera intrinsics received.")

    def _odom_cb(self, msg: Odometry) -> None:
        self._T_world_camera, _ = msg2np(msg)

    def _image_cb(self, msg: Image) -> None:
        if self.done or self._K is None or self._T_world_camera is None:
            return

        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        corners, ids, _ = self._detector.detectMarkers(img)
        if ids is None:
            return

        board_ids = set(self._board.getIds().flatten().tolist())
        n_visible = sum(1 for tid in ids.flatten() if tid in board_ids)
        if n_visible < MIN_TAGS:
            return

        obj_pts, img_pts = self._board.matchImagePoints(corners, ids)
        if obj_pts is None or len(obj_pts) < 4:
            return

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, self._K, self._dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None:
            return

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, self._K, self._dist)
        reproj_err = float(np.mean(
            np.linalg.norm(img_pts.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        ))
        if reproj_err > MAX_REPROJ_PX:
            self.get_logger().warn(
                f"reproj={reproj_err:.2f}px > {MAX_REPROJ_PX}px, skip",
                throttle_duration_sec=1.0,
            )
            return

        R_mat, _ = cv2.Rodrigues(rvec)
        T_camera_board = np.eye(4)
        T_camera_board[:3, :3] = R_mat
        T_camera_board[:3,  3] = tvec.ravel()

        T_world_board = self._T_world_camera @ T_camera_board
        self._samples.append(T_world_board)

        n = len(self._samples)
        self.get_logger().info(
            f"Collected {n}/{RECORD_FRAMES} "
            f"(reproj={reproj_err:.2f}px, n_tags={n_visible})"
        )

        if n >= RECORD_FRAMES:
            self._finish()

    # ---- calibration solve ----

    def _finish(self) -> None:
        self.done = True

        T_world_qrworld = _mean_T(self._samples)
        T_world_robot   = self._T_world_camera @ T_CAMERA_ROBOT
        T_qrworld_robot = np.linalg.inv(T_world_qrworld) @ T_world_robot

        # Save tag_target.json
        DB_DIR.mkdir(parents=True, exist_ok=True)
        TARGET_PATH.write_text(json.dumps({
            "tag_ids":         self._tag_ids,
            "size_m":          self._size_m,
            "spacing_m":       self._spacing_m,
            "T_qrworld_robot": T_qrworld_robot.tolist(),
        }, indent=2))
        self.get_logger().info(f"Saved T_qrworld_robot → {TARGET_PATH}")

        # Save tag_mappose.json (requires world→map TF from map_node)
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                "world", "map", rclpy.time.Time()
            )
            _, _, T_world_map = tf2np(tf_msg)
            T_map_qrworld = np.linalg.inv(T_world_map) @ T_world_qrworld
            MAPPOSE_PATH.write_text(json.dumps({
                "T_map_qrworld": T_map_qrworld.tolist(),
            }, indent=2))
            self.get_logger().info(f"Saved T_map_qrworld  → {MAPPOSE_PATH}")
        except Exception as e:
            self.get_logger().warn(
                f"world→map TF not available ({e}); "
                f"{MAPPOSE_PATH.name} NOT written — "
                "run map_node and retry if odom_node.py is needed."
            )

        self.get_logger().info("Calibration complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = RecordNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
