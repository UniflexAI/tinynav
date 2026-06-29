#!/usr/bin/env python3
"""
tool/qr_odom.py

Detects an AprilTag board and publishes the camera pose as /qr/odom.

Frame chain
-----------
  T_camera_board   ← solvePnPRansac (per detection)
  T_qrworld_camera = inv(T_camera_board)         (board IS qrworld origin)
  T_map_camera     = T_map_qrworld @ T_qrworld_camera   (T_map_qrworld predefined)
  T_world_camera   = T_world_map @ T_map_camera   (T_world_map from TF world→map)

Calibration files
-----------------
  tinynav_db/qrcode/tag_satellite.json   tag ids, size, spacing
  tinynav_db/qrcode/tag_mappose.json     T_map_qrworld  ← NEW, created at map-build time

Topics
------
  Subscribed:  /camera/camera/infra1/image_rect_raw   sensor_msgs/Image
               /camera/camera/infra1/camera_info      sensor_msgs/CameraInfo
  TF lookup:   world → map                             (broadcast by map_node)
  Published:   /qr/odom                               nav_msgs/Odometry
"""

import json
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from tinynav.core.math_utils import np2msg, tf2np

DB_DIR         = Path("tinynav_db/qrcode")
TAG_PARAMS_PATH = DB_DIR / "tag_satellite.json"
TAG_MAPPOSE_PATH = DB_DIR / "tag_mappose.json"

IMAGE_TOPIC  = "/camera/camera/infra1/image_rect_raw"
INFO_TOPIC   = "/camera/camera/infra1/camera_info"
QR_ODOM_TOPIC = "/qr/odom"

MIN_TAGS      = 2      # minimum visible tags to publish
MAX_REPROJ_PX = 3.0    # reprojection error threshold


class QROdomNode(Node):
    def __init__(self):
        super().__init__("qr_odom_node")

        d_params = json.loads(TAG_PARAMS_PATH.read_text())
        d_map    = json.loads(TAG_MAPPOSE_PATH.read_text())

        self._T_map_qrworld: np.ndarray = np.array(d_map["T_map_qrworld"])

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._board = cv2.aruco.GridBoard(
            size=(2, 2),
            markerLength=float(d_params["size_m"]),
            markerSeparation=float(d_params["spacing_m"]),
            dictionary=aruco_dict,
            ids=np.array(d_params["tag_ids"], dtype=np.int32),
        )
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        self._K:    np.ndarray | None = None
        self._dist: np.ndarray | None = None
        self._bridge = CvBridge()

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._info_sub = self.create_subscription(
            CameraInfo, INFO_TOPIC, self._info_cb, 10)
        self.create_subscription(Image, IMAGE_TOPIC, self._image_cb, 10)
        self._pub = self.create_publisher(Odometry, QR_ODOM_TOPIC, 10)

        self.get_logger().info(
            f"qr_odom_node: tag ids={d_params['tag_ids']}, publishing {QR_ODOM_TOPIC}"
        )

    def _info_cb(self, msg: CameraInfo) -> None:
        self._K    = np.array(msg.k).reshape(3, 3)
        self._dist = np.array(msg.d)
        self.destroy_subscription(self._info_sub)
        self.get_logger().info("Camera intrinsics received.")

    def _image_cb(self, msg: Image) -> None:
        if self._K is None:
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
                f"reproj_err={reproj_err:.2f}px > {MAX_REPROJ_PX}px, skip",
                throttle_duration_sec=1.0,
            )
            return

        # T_camera_board from PnP
        R_mat, _ = cv2.Rodrigues(rvec)
        T_camera_board = np.eye(4)
        T_camera_board[:3, :3] = R_mat
        T_camera_board[:3, 3]  = tvec.ravel()

        # T_world_map from TF (broadcast by map_node)
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                "world", "map", rclpy.time.Time()
            )
            _, _, T_world_map = tf2np(tf_msg)
        except Exception:
            self.get_logger().warn(
                "TF world→map not available yet", throttle_duration_sec=2.0)
            return

        # Full chain: world ← map ← qrworld ← camera
        T_qrworld_camera = np.linalg.inv(T_camera_board)
        T_map_camera     = self._T_map_qrworld @ T_qrworld_camera
        T_world_camera   = T_world_map @ T_map_camera

        odom_msg = np2msg(T_world_camera, msg.header.stamp, "world", "camera")

        # Covariance: scale with reprojection error
        var_pos = max(reproj_err / MAX_REPROJ_PX, 0.1) * 0.01
        var_rot = max(reproj_err / MAX_REPROJ_PX, 0.1) * 0.001
        cov = np.zeros(36)
        cov[0] = cov[7] = cov[14] = var_pos
        cov[21] = cov[28] = cov[35] = var_rot
        odom_msg.pose.covariance = cov.tolist()

        self._pub.publish(odom_msg)
        self.get_logger().info(
            f"qr/odom published: reproj={reproj_err:.2f}px, n_tags={n_visible}",
            throttle_duration_sec=0.5,
        )


def main(args=None):
    rclpy.init(args=args)
    node = QROdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
