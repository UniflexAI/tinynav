import argparse
import json
import logging
import os
import sys
import time
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer, InputAligner, SimpleFilter
import numpy as np
import rclpy
from codetiming import Timer
from cv_bridge import CvBridge
from tinynav.core.models_trt import LightGlueTRT, SuperPointTRT, StereoEngineTRT
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, CameraInfo
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.duration import Duration
from tinynav.core.math_utils import rot_from_two_vector, np2msg, np2tf, estimate_pose
from tinynav.core.math_utils import uf_init, uf_union, uf_all_sets_list
from tf2_ros import TransformBroadcaster
import asyncio
import gtsam
import gtsam_unstable
from collections import deque
from dataclasses import dataclass

from gtsam.symbol_shorthand import X, B, V
from tinynav.core.imu_propagator_node import ImuPropagatorNode

_N = 5
_M = 1000

_MIN_FEATURES = 20
_KEYFRAME_MIN_DISTANCE = 0.1    # unit: meter
_KEYFRAME_MIN_ROTATE_DEGREE = 0.1 # unit: degree

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def keyframe_check(T_i, T_j):
    T_ij = np.linalg.inv(T_i) @ T_j
    t_diff = np.linalg.norm(T_ij[:3, 3])
    cos_theta = (np.trace(T_ij[:3, :3]) - 1) / 2
    r_diff = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    return t_diff > _KEYFRAME_MIN_DISTANCE or r_diff > _KEYFRAME_MIN_ROTATE_DEGREE


def Matrix4x4ToGtsamPose3(T: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(T[:3, 3]))

def depth_to_point(kp, depth, K):
    u, v = int(kp[0]), int(kp[1])
    Z = depth
    X = (u - K[0,2]) * Z / K[0,0]
    Y = (v - K[1,2]) * Z / K[1,1]
    return np.array([X, Y, Z])

def stamp2second(stamp):
    nano_s = np.int64(stamp.sec) * 1_000_000_000 + np.int64(stamp.nanosec)
    return nano_s * 1e-9


@dataclass
class StereoPairMsg:
    header: object
    left_msg: Image
    right_msg: Image


# keyframe dataclass
@dataclass
class Keyframe:
    timestamp: float
    image: np.ndarray
    disparity: np.ndarray
    depth: np.ndarray
    pose: np.ndarray
    velocity: np.ndarray
    bias: gtsam.imuBias.ConstantBias
    preintegrated_imu: gtsam.PreintegratedCombinedMeasurements
    latest_imu_timestamp: float
    imu_measurement_count: int = 0

class PerceptionNode(Node):
    def __init__(self, verbose_timer: bool = True):
        super().__init__("perception_node")
        self.verbose_timer = verbose_timer
        self.logger = logging.getLogger(__name__)
        # self.timer_logger = self.logger.info if verbose_timer else self.logger.debug
        # model
        self.superpoint = SuperPointTRT()
        self.light_glue = LightGlueTRT()

        self.last_keyframe_img = None
        self.last_keyframe_features = None

        self.stereo_engine = StereoEngineTRT()
        # intrinsic
        self.baseline = None
        self.K = None
        self.image_shape = None

        self.T_body_last = None
        self.V_last = None
        self.B_last = None

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=500)

        # use a single topic to handle the imu data.
        self.imu_sub = self.create_subscription(Imu, "/camera/camera/imu", self.imu_callback, qos_profile)
        self.imu_last_received_timestamp = None


        self.camerainfo_sub = self.create_subscription(CameraInfo, "/camera/camera/infra2/camera_info", self.info_callback, 10)
        self.left_sub = Subscriber(self, Image, "/camera/camera/infra1/image_rect_raw")
        self.right_sub = Subscriber(self, Image, "/camera/camera/infra2/image_rect_raw")
        self.ts = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=10, slop=0.02)
        self.ts.registerCallback(self.images_callback)

        self.input_aligner_imu_filter = SimpleFilter()
        self.input_aligner_stereo_filter = SimpleFilter()
        self.input_aligner = InputAligner(Duration(seconds=1.000), self.input_aligner_imu_filter, self.input_aligner_stereo_filter)
        self.input_aligner.setInputPeriod(0, Duration(seconds=0.005))
        self.input_aligner.setInputPeriod(1, Duration(seconds=0.01))
        self.input_aligner.registerCallback(0, self._aligned_imu_callback)
        self.input_aligner.registerCallback(1, self._aligned_stereo_callback)
        self.input_aligner_seen_imu = False
        self.input_aligner_seen_stereo = False
        self.odom_pub = self.create_publisher(Odometry, "/slam/odometry_visual", 10)
        self.slam_camera_info_pub = self.create_publisher(CameraInfo, "/slam/camera_info", 10)
        self.depth_pub = self.create_publisher(Image, "/slam/depth", 10)
        self.disparity_pub_vis = self.create_publisher(Image, '/slam/disparity_vis', 10)
        self.keyframe_pose_pub = self.create_publisher(Odometry, "/slam/keyframe_odom", 10)
        self.keyframe_image_pub = self.create_publisher(Image, "/slam/keyframe_image", 10)
        self.keyframe_depth_pub = self.create_publisher(Image, "/slam/keyframe_depth", 10)
        self.stats_pub = self.create_publisher(String, "/slam/data", 10)

        self.accel_readings = []
        self.last_processed_timestamp = 0.0

        self.camera_info_msg = None

        # Noise model (continuous-time). Defaults preserve the previous D435i tuning.
        accel_noise_density = self.declare_parameter("imu.accel_noise_density", 0.50).value
        gyro_noise_density = self.declare_parameter("imu.gyro_noise_density", 0.50).value
        bias_acc_rw_sigma = self.declare_parameter("imu.bias_acc_random_walk", 0.001).value
        bias_gyro_rw_sigma = self.declare_parameter("imu.bias_gyro_random_walk", 0.0001).value
        self.min_imu_measurements_between_keyframes = int(
            self.declare_parameter("imu.min_measurements_between_keyframes", 26).value
        )
        self.bias_prior_sigma_accel = float(self.declare_parameter("imu.bias_prior_sigma_accel", 1e-2).value)
        self.bias_prior_sigma_gyro = float(self.declare_parameter("imu.bias_prior_sigma_gyro", 1e-2).value)
        self.landmark_overlay_enabled = bool(
            self.declare_parameter("debug.landmark_overlay.enabled", False).value
        )
        self.landmark_overlay_dir = str(
            self.declare_parameter(
                "debug.landmark_overlay.dir",
                "tinynav_temp/sliding_window_landmarks",
            ).value
        )
        if self.landmark_overlay_enabled:
            os.makedirs(self.landmark_overlay_dir, exist_ok=True)
        self.stereo_feature_debug_enabled = bool(
            self.declare_parameter("debug.stereo_feature.enabled", False).value
        )
        self.stereo_feature_debug_dir = str(
            self.declare_parameter(
                "debug.stereo_feature.dir",
                "tinynav_temp/stereo_feature_debug",
            ).value
        )
        self.stereo_feature_debug_max_matches = int(
            self.declare_parameter("debug.stereo_feature.max_matches", 200).value
        )
        if self.stereo_feature_debug_enabled:
            os.makedirs(self.stereo_feature_debug_dir, exist_ok=True)
        self.pre_integration_params = gtsam.PreintegrationCombinedParams.MakeSharedU()
        self.pre_integration_params.setAccelerometerCovariance((accel_noise_density**2) * np.eye(3))
        self.pre_integration_params.setGyroscopeCovariance((gyro_noise_density**2) * np.eye(3))
        self.pre_integration_params.setIntegrationCovariance(1e-8 * np.eye(3))
        self.pre_integration_params.setBiasAccCovariance(np.eye(3) * bias_acc_rw_sigma**2)
        self.pre_integration_params.setBiasOmegaCovariance(np.eye(3) * bias_gyro_rw_sigma**2)
        self.pre_integration_params.setUse2ndOrderCoriolis(False)
        self.pre_integration_params.setOmegaCoriolis(np.array([0.0, 0.0, 0.0]))

        self.T_imu_body_to_camera = np.array(
                            [[1, 0, 0, 0],
                             [0, 0, -1, 0], 
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]])

        self.imu_measurements = deque(maxlen=1000)

        self.keyframe_queue = []
        self._async_loop = asyncio.new_event_loop()
        self.logger.info("PerceptionNode initialized.")
        self.process_cnt = 0

    def _track_color(self, track_idx):
        rng = np.random.default_rng(track_idx + 17)
        return tuple(int(c) for c in rng.integers(64, 256, size=3))

    def _save_sliding_window_landmarks(self, tracks, extract_info, stamp):
        if not self.landmark_overlay_enabled:
            return

        window = self.keyframe_queue[-_N:]
        if not window:
            return

        tiles = []
        for pose_idx, keyframe in enumerate(window):
            image = keyframe.image
            if image.ndim == 2:
                tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                tile = image.copy()
            label = f"{pose_idx} t={keyframe.timestamp:.3f}"
            cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            tiles.append(tile)

        tile_widths = [tile.shape[1] for tile in tiles]
        x_offsets = np.cumsum([0] + tile_widths[:-1])
        points_by_track = []

        for track_idx, landmark in enumerate(tracks):
            color = self._track_color(track_idx)
            points = []
            for projection in landmark:
                pose_idx = projection // _M
                feature_idx = projection % _M
                if pose_idx >= len(window) or pose_idx >= len(extract_info):
                    continue

                keypoints = extract_info[pose_idx]["kpts"][0]
                if feature_idx >= len(keypoints):
                    continue

                x, y = keypoints[feature_idx]
                x_i = int(round(float(x)))
                y_i = int(round(float(y)))
                if not (0 <= y_i < tiles[pose_idx].shape[0] and 0 <= x_i < tiles[pose_idx].shape[1]):
                    continue

                cv2.circle(tiles[pose_idx], (x_i, y_i), 3, color, -1)
                points.append((pose_idx, int(x_offsets[pose_idx]) + x_i, y_i))
            if len(points) >= 2:
                ordered_points = [(x, y) for _, x, y in sorted(points)]
                points_by_track.append((color, ordered_points))

        mosaic = cv2.hconcat(tiles)
        for color, points in points_by_track:
            for p0, p1 in zip(points[:-1], points[1:]):
                cv2.line(mosaic, p0, p1, color, 1, cv2.LINE_AA)

        timestamp_ns = np.int64(stamp.sec) * 1_000_000_000 + np.int64(stamp.nanosec)
        filename = f"window_{self.process_cnt:06d}_{timestamp_ns}.png"
        output_path = os.path.join(self.landmark_overlay_dir, filename)
        if not cv2.imwrite(output_path, mosaic):
            self.logger.warning(f"Failed to write landmark overlay image: {output_path}")

    def _draw_keypoints(self, image, keypoints, color):
        if image.ndim == 2:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            output = image.copy()
        for x, y in keypoints:
            x_i = int(round(float(x)))
            y_i = int(round(float(y)))
            if 0 <= y_i < output.shape[0] and 0 <= x_i < output.shape[1]:
                cv2.circle(output, (x_i, y_i), 2, color, -1, cv2.LINE_AA)
        return output

    def _disparity_color(self, disparity):
        valid = np.isfinite(disparity) & (disparity > 0)
        if valid.any():
            disp_min = float(np.nanpercentile(disparity[valid], 2))
            disp_max = float(np.nanpercentile(disparity[valid], 98))
            if disp_max > disp_min:
                disp_norm = (np.clip(disparity, disp_min, disp_max) - disp_min) / (disp_max - disp_min)
            else:
                disp_norm = np.zeros_like(disparity, dtype=np.float32)
        else:
            disp_min = 0.0
            disp_max = 0.0
            disp_norm = np.zeros_like(disparity, dtype=np.float32)

        disp_u8 = np.clip(disp_norm * 255.0, 0, 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_PLASMA)
        cv2.putText(
            disp_color,
            f"Retinify disparity p02={disp_min:.2f} p98={disp_max:.2f}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return disp_color

    def _save_stereo_feature_debug(self, left_img, right_img, disparity, left_extract, right_extract, match_result, stamp):
        if not self.stereo_feature_debug_enabled:
            return

        left_keypoints = left_extract["kpts"][0]
        right_keypoints = right_extract["kpts"][0]
        match_indices = match_result["match_indices"][0]
        valid_left_idx = np.flatnonzero(match_indices != -1)
        if len(valid_left_idx) > self.stereo_feature_debug_max_matches:
            step = max(1, len(valid_left_idx) // self.stereo_feature_debug_max_matches)
            valid_left_idx = valid_left_idx[::step][: self.stereo_feature_debug_max_matches]

        rows = []
        for left_idx in valid_left_idx:
            right_idx = int(match_indices[left_idx])
            if right_idx < 0 or right_idx >= len(right_keypoints):
                continue
            xl, yl = left_keypoints[left_idx]
            xr, yr = right_keypoints[right_idx]
            x_i = int(round(float(xl)))
            y_i = int(round(float(yl)))
            disp_value = np.nan
            if 0 <= y_i < disparity.shape[0] and 0 <= x_i < disparity.shape[1]:
                disp_value = float(disparity[y_i, x_i])
            lg_disp = float(xl - xr)
            rows.append(
                {
                    "left_idx": int(left_idx),
                    "right_idx": right_idx,
                    "xl": float(xl),
                    "yl": float(yl),
                    "xr": float(xr),
                    "yr": float(yr),
                    "lightglue_disparity": lg_disp,
                    "retinify_disparity": disp_value,
                    "difference": lg_disp - disp_value,
                }
            )

        left_sp = self._draw_keypoints(left_img, left_keypoints, (0, 255, 0))
        right_sp = self._draw_keypoints(right_img, right_keypoints, (0, 255, 255))
        cv2.putText(left_sp, f"left SuperPoint: {len(left_keypoints)}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(right_sp, f"right SuperPoint: {len(right_keypoints)}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        match_vis = np.hstack([cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)])
        right_offset = left_img.shape[1]
        finite_diffs = []
        for row in rows:
            color = self._track_color(row["left_idx"])
            pt_left = (int(round(row["xl"])), int(round(row["yl"])))
            pt_right = (right_offset + int(round(row["xr"])), int(round(row["yr"])))
            cv2.circle(match_vis, pt_left, 3, color, -1, cv2.LINE_AA)
            cv2.circle(match_vis, pt_right, 3, color, -1, cv2.LINE_AA)
            cv2.line(match_vis, pt_left, pt_right, color, 1, cv2.LINE_AA)
            if np.isfinite(row["difference"]):
                finite_diffs.append(row["difference"])

        if finite_diffs:
            diff_arr = np.asarray(finite_diffs, dtype=np.float32)
            diff_text = (
                f"matches={len(rows)} dx-disp mean={float(np.mean(diff_arr)):.2f} "
                f"median={float(np.median(diff_arr)):.2f} abs95={float(np.percentile(np.abs(diff_arr), 95)):.2f}"
            )
        else:
            diff_text = f"matches={len(rows)} no finite disparity samples"
        cv2.putText(match_vis, diff_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        disp_color = self._disparity_color(disparity)
        blank = np.zeros_like(disp_color)
        top = np.hstack([left_sp, right_sp])
        middle = match_vis
        bottom = np.hstack([disp_color, blank])
        mosaic = np.vstack([top, middle, bottom])

        timestamp_ns = np.int64(stamp.sec) * 1_000_000_000 + np.int64(stamp.nanosec)
        basename = f"stereo_{self.process_cnt:06d}_{timestamp_ns}"
        image_path = os.path.join(self.stereo_feature_debug_dir, f"{basename}.png")
        csv_path = os.path.join(self.stereo_feature_debug_dir, f"{basename}.csv")
        if not cv2.imwrite(image_path, mosaic):
            self.logger.warning(f"Failed to write stereo feature debug image: {image_path}")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("left_idx,right_idx,xl,yl,xr,yr,lightglue_disparity,retinify_disparity,difference\n")
            for row in rows:
                f.write(
                    f"{row['left_idx']},{row['right_idx']},{row['xl']:.3f},{row['yl']:.3f},"
                    f"{row['xr']:.3f},{row['yr']:.3f},{row['lightglue_disparity']:.3f},"
                    f"{row['retinify_disparity']:.3f},{row['difference']:.3f}\n"
                )

    def info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            fx = self.K[0, 0]
            Tx = msg.p[3]  # From the right camera's projection matrix
            self.baseline = -Tx / fx
            self.get_logger().info(f"Camera intrinsics and baseline received. Baseline: {self.baseline:.4f}m")
            self.camera_info_msg = msg
            self.destroy_subscription(self.camerainfo_sub)

    def _process_imu_msg(self, imu_msg):
        current_timestamp = stamp2second(imu_msg.header.stamp)
        if len(self.accel_readings) >= 10 and self.T_body_last is None:
            accel_data = np.array([(a.x, a.y, a.z) for a in self.accel_readings])
            gravity_cam = np.mean(accel_data, axis=0)
            gravity_cam /= np.linalg.norm(gravity_cam)
            gravity_world = np.array([0.0, 0.0, 1.0])

            self.T_body_last = np.eye(4)
            self.T_body_last[:3, :3] = rot_from_two_vector(gravity_cam, gravity_world)
            self.get_logger().info("Initial pose set from accelerometer data.")
            self.get_logger().info(f"Initial rotation matrix:\n{self.T_body_last}")
        elif len(self.accel_readings) < 10:
            self.accel_readings.append(imu_msg.linear_acceleration)

        # if the timestamp jump is too large, it means the IMU is not working properly
        if self.imu_last_received_timestamp is not None and current_timestamp - self.imu_last_received_timestamp > 0.1:
            delta_timestamp = current_timestamp - self.imu_last_received_timestamp
            self.get_logger().warning(f"IMU timestamp jump {delta_timestamp} s is too large, it means the IMU is not working properly")
        self.imu_last_received_timestamp = current_timestamp
        accel_data = np.array([[imu_msg.linear_acceleration.x], [imu_msg.linear_acceleration.y], [imu_msg.linear_acceleration.z]])
        gyro_data = np.array([[imu_msg.angular_velocity.x], [imu_msg.angular_velocity.y], [imu_msg.angular_velocity.z]])
        self.imu_measurements.append([current_timestamp, accel_data.flatten(), gyro_data.flatten()])

    def _aligned_imu_callback(self, imu_msg):
        self._process_imu_msg(imu_msg)

    def _aligned_stereo_callback(self, stereo_pair_msg):
        left_msg = stereo_pair_msg.left_msg
        right_msg = stereo_pair_msg.right_msg
        image_timestamp = stamp2second(left_msg.header.stamp)
        if image_timestamp - self.last_processed_timestamp < 0.1333:
            return

        self.last_processed_timestamp = image_timestamp
        loop_start = time.perf_counter()
        with Timer(name="Perception Loop", text="[{name}] Elapsed time: {milliseconds:.0f} ms\n\n", logger=self.logger.info):
            processed = self._async_loop.run_until_complete(self.process(left_msg, right_msg))
        if processed:
            processed["stats"]["loop_ms"] = (time.perf_counter() - loop_start) * 1000.0
            self.stats_pub.publish(String(data=json.dumps(processed)))

    def destroy_node(self):
        if self._async_loop is not None:
            self._async_loop.close()
            self._async_loop = None
        return super().destroy_node()

    def imu_callback(self, imu_msg):
        self.input_aligner_imu_filter.signalMessage(imu_msg)
        self.input_aligner_seen_imu = True
        if self.input_aligner_seen_stereo:
            self.input_aligner.dispatchMessages()

    def images_callback(self, left_msg, right_msg):
        stereo_pair_msg = StereoPairMsg(header=left_msg.header, left_msg=left_msg, right_msg=right_msg)
        self.input_aligner_stereo_filter.signalMessage(stereo_pair_msg)
        self.input_aligner_seen_stereo = True
        if self.input_aligner_seen_imu:
            self.input_aligner.dispatchMessages()

    async def process(self, left_msg, right_msg):
        if self.K is None or self.T_body_last is None:
            return {
            "stats": {"process_cnt": 0},
            "metrics": {"num_keyframes": 0, "num_tracks": 0, "num_factors": 0, "num_variables": 0, "initial_error": 0.0, "final_error": 0.0}
        }
        self.process_cnt += 1
        left_img = self.bridge.imgmsg_to_cv2(left_msg, "mono8")
        right_img = self.bridge.imgmsg_to_cv2(right_msg, "mono8")
        current_timestamp = stamp2second(left_msg.header.stamp)
        if len(self.keyframe_queue) == 0: # first frame
            disparity, depth = await self.stereo_engine.infer(left_img, right_img, np.array([[self.baseline]]), np.array([[self.K[0,0]]]))
            if self.stereo_feature_debug_enabled:
                left_extract_result = await self.superpoint.infer(left_img)
                right_extract_result = await self.superpoint.infer(right_img)
                stereo_match_result = await self.light_glue.infer(
                    left_extract_result["kpts"],
                    right_extract_result["kpts"],
                    left_extract_result["descps"],
                    right_extract_result["descps"],
                    left_extract_result["mask"],
                    right_extract_result["mask"],
                    left_img.shape,
                    right_img.shape,
                )
                self._save_stereo_feature_debug(
                    left_img,
                    right_img,
                    disparity,
                    left_extract_result,
                    right_extract_result,
                    stereo_match_result,
                    left_msg.header.stamp,
                )
            initial_bias = gtsam.imuBias.ConstantBias()
            self.keyframe_queue.append(
                Keyframe(
                    timestamp=current_timestamp,
                    image=left_img,
                    disparity=disparity,
                    depth=depth,
                    pose=self.T_body_last,
                    velocity=np.zeros(3),
                    bias=initial_bias,
                    preintegrated_imu=gtsam.PreintegratedCombinedMeasurements(self.pre_integration_params, initial_bias),
                    latest_imu_timestamp=current_timestamp
                )
            )
            return {
            "stats": {"process_cnt": 0},
            "metrics": {"num_keyframes": 0, "num_tracks": 0, "num_factors": 0, "num_variables": 0, "initial_error": 0.0, "final_error": 0.0}
        }

        with Timer(name="[Stereo Inference]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            disparity, depth = await self.stereo_engine.infer(left_img, right_img, np.array([[self.baseline]]), np.array([[self.K[0,0]]]))
            kf_prev = self.keyframe_queue[-1]
            prev_left_extract_result = await self.superpoint.infer(kf_prev.image)
            current_left_extract_result = await self.superpoint.infer(left_img)

            match_result = await self.light_glue.infer(
                prev_left_extract_result["kpts"],
                current_left_extract_result["kpts"],
                prev_left_extract_result["descps"],
                current_left_extract_result["descps"],
                prev_left_extract_result["mask"],
                current_left_extract_result["mask"],
                kf_prev.image.shape,
                left_img.shape)

        if self.stereo_feature_debug_enabled:
            right_extract_result = await self.superpoint.infer(right_img)
            stereo_match_result = await self.light_glue.infer(
                current_left_extract_result["kpts"],
                right_extract_result["kpts"],
                current_left_extract_result["descps"],
                right_extract_result["descps"],
                current_left_extract_result["mask"],
                right_extract_result["mask"],
                left_img.shape,
                right_img.shape,
            )
            self._save_stereo_feature_debug(
                left_img,
                right_img,
                disparity,
                current_left_extract_result,
                right_extract_result,
                stereo_match_result,
                left_msg.header.stamp,
            )

        # propagate IMU measurements
        while len(self.imu_measurements) > 0 and self.imu_measurements[0][0] <= current_timestamp:
            timestamp, accel, gyro = self.imu_measurements[0]
            dt = timestamp - self.keyframe_queue[-1].latest_imu_timestamp

            if timestamp <= self.keyframe_queue[-1].latest_imu_timestamp:
                self.imu_measurements.popleft()
                self.logger.warning("should only happen at beginning")
                continue

            self.keyframe_queue[-1].preintegrated_imu.integrateMeasurement(accel, gyro, dt) #todo
            self.keyframe_queue[-1].latest_imu_timestamp = timestamp
            self.keyframe_queue[-1].imu_measurement_count += 1

            self.imu_measurements.popleft()
        # specially process the last imu
        if len(self.imu_measurements) > 0 and current_timestamp - self.keyframe_queue[-1].latest_imu_timestamp > 0.001:
            timestamp, accel, gyro = self.imu_measurements[0]
            dt = current_timestamp - self.keyframe_queue[-1].latest_imu_timestamp
            self.keyframe_queue[-1].preintegrated_imu.integrateMeasurement(accel, gyro, dt)
            self.keyframe_queue[-1].imu_measurement_count += 1

        with Timer(name="[PnP]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
        # do simple pose estimation between last keyframe and current frame
            prev_keypoints = prev_left_extract_result["kpts"][0]  # (n, 2)
            current_keypoints = current_left_extract_result["kpts"][0]  # (n, 2)
            match_indices = match_result["match_indices"][0]
            idx_to_origial = range(len(prev_keypoints))
            valid_mask = match_indices != -1
            kpt_pre = prev_keypoints[valid_mask]
            kpt_cur = current_keypoints[match_indices[valid_mask]]
            idx_valid = np.array(idx_to_origial)[valid_mask]
            logging.debug(f"match cnt: {len(kpt_pre)}")
            state, T_kf_curr, _, _, _ = estimate_pose(
                kpt_pre,
                kpt_cur,
                depth,
                self.K,
                idx_valid
            )
            self.logger.debug("Estimated T_kf_curr:\n", T_kf_curr)
        # for new frame, we first add it as keyframe, if not, we pop it later
        previous_bias = self.keyframe_queue[-1].bias
        self.keyframe_queue.append(
            Keyframe(
                timestamp=current_timestamp,
                image=left_img,
                disparity=disparity,
                depth=depth,
                pose=self.keyframe_queue[-1].pose @ T_kf_curr,
                velocity=self.keyframe_queue[-1].velocity,
                bias=previous_bias,
                preintegrated_imu=gtsam.PreintegratedCombinedMeasurements(self.pre_integration_params, previous_bias),
                latest_imu_timestamp=current_timestamp
            )
        )
        if len(self.keyframe_queue) > _N:
            self.keyframe_queue.pop(0)
        with Timer(name="[ISAM Processing]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.info):
            with Timer(name="[adding imu]", text="[{name}] Elapsed time: {milliseconds:.03f} ms", logger=self.logger.debug):
                # we have new graph each time
                graph = gtsam.NonlinearFactorGraph()
                initial_estimate = gtsam.Values()
                # process previous keyframes' factors
                for i, keyframe in enumerate(self.keyframe_queue[-_N:]):
                    # per pose -- bias
                    initial_estimate.insert(B(i), keyframe.bias)

                    initial_estimate.insert(V(i), keyframe.velocity)
                    initial_estimate.insert(X(i), Matrix4x4ToGtsamPose3(keyframe.pose))
                    if i == 0:
                        bias_sigmas = np.array(
                            [
                                self.bias_prior_sigma_accel,
                                self.bias_prior_sigma_accel,
                                self.bias_prior_sigma_accel,
                                self.bias_prior_sigma_gyro,
                                self.bias_prior_sigma_gyro,
                                self.bias_prior_sigma_gyro,
                            ]
                        )
                        graph.add(
                            gtsam.PriorFactorConstantBias(
                                B(i),
                                keyframe.bias,
                                gtsam.noiseModel.Diagonal.Sigmas(bias_sigmas),
                            )
                        )
                        ## per pose -- velocity
                        #graph.add(gtsam.PriorFactorVector(V(i), np.zeros(3), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2, 1e-2, 1e-2]))))

                        # per pose -- pose, could only be applied to the first keyframe
                        graph.add(gtsam.PriorFactorPose3(X(i), Matrix4x4ToGtsamPose3(keyframe.pose), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1]))))

                    # per pose -- preintegrated IMU factor, only between two keyframes
                    if i != len(self.keyframe_queue[-_N:]) - 1:
                        if keyframe.imu_measurement_count < self.min_imu_measurements_between_keyframes:
                            self.logger.warning(
                                f"keyframe {i} at {keyframe.timestamp} only used "
                                f"{keyframe.imu_measurement_count} imu measurements; expected at least "
                                f"{self.min_imu_measurements_between_keyframes}"
                            )
                        imu_factor = gtsam.CombinedImuFactor(X(i), V(i), X(i+1), V(i+1), B(i), B(i+1), keyframe.preintegrated_imu)
                        graph.add(imu_factor)
                    self.logger.debug(
                        f"for frame {i} at {keyframe.timestamp}, added imufactor up to "
                        f"{keyframe.latest_imu_timestamp} using {keyframe.imu_measurement_count} imu measurements"
                    )

            #with Timer(name="[stats]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            #    self.frame_diff_t = []
            #    for i in range(max(0, len(self.keyframe_queue) - _N), len(self.keyframe_queue) - 1):
            #        j = i + 1
            #        kf_prev_timestamp, kf_prev_image, kf_prev_disparity, kf_prev_depth, kf_prev_P, kf_prev_V, kf_prev_B, kf_prev_factor, _  = astuple(self.keyframe_queue[i])
            #        kf_curr_timestamp, kf_curr_image, kf_curr_disparity, kf_curr_depth, kf_curr_P, kf_curr_V, kf_curr_B, kf_curr_factor, _  = astuple(self.keyframe_queue[i + 1])
            #        self.frame_diff_t.append(kf_curr_timestamp - kf_prev_timestamp)

            #for i, keyframe in enumerate(self.keyframe_queue[-_N:]):
            #    kf_timestamp, kf_image, kf_disparity, kf_depth, kf_P, kf_V, kf_B, kf_factor, latest_imu_timestamp = astuple(keyframe)
            #    if i != len(self.keyframe_queue[-_N:]) - 1:
            #        imu_factor = gtsam.CombinedImuFactor(X(i), V(i), X(i+1), V(i+1), B(i), B(i+1), kf_factor) 

            #        print("processing imu factor between ", i, " and ", i+1)
            #        print("error: ", imu_factor.error(initial_estimate))
            #        print("frame_diff_t: ", self.frame_diff_t[i])
            #        print("kf_factor: ", kf_factor)
            #current_i = len(self.keyframe_queue[-_N:])

            with Timer(name="[init extract info]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
                extract_info = [await self.superpoint.infer(kf.image) for kf in self.keyframe_queue[-_N:]]
                uf = uf_init(len(self.keyframe_queue[-_N:]) * _M)

            self.logger.debug(f"Processing {len(self.keyframe_queue)} keyframes for data association.")
            # Process pairs of keyframes from last _N keyframes: extract features (SuperPoint),
            # match by LightGlue, filter by geometric consistency (pose estimation), 
            # and build tracks via Union-Find
            with Timer(name="[cached result]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
                for i in range(max(0, len(self.keyframe_queue) - _N), len(self.keyframe_queue) - 1):
                    with Timer(name="[cached result[1/3]]", text="[{name}] Elapsed time: {milliseconds:.03f} ms", logger=self.logger.debug):
                        j = i + 1
                        kf_prev = self.keyframe_queue[i]
                        kf_curr = self.keyframe_queue[j]

                    self.logger.debug("timestamp prev: ", kf_prev.timestamp)
                    self.logger.debug("timestamp curr: ", kf_curr.timestamp)
                    with Timer(name="[cached result[1.1/3]]", text="[{name}] Elapsed time: {milliseconds:.03f} ms", logger=self.logger.debug):
                        prev_left_extract_result = await self.superpoint.infer(kf_prev.image)
                    with Timer(name="[cached result[1.2/3]]", text="[{name}] Elapsed time: {milliseconds:.03f} ms", logger=self.logger.debug):
                        current_left_extract_result = await self.superpoint.infer(kf_curr.image)

                    with Timer(name="[cached result[1.3/3]]", text="[{name}] Elapsed time: {milliseconds:.03f} ms", logger=self.logger.debug):
                        match_result = await self.light_glue.infer(
                            prev_left_extract_result["kpts"],
                            current_left_extract_result["kpts"],
                            prev_left_extract_result["descps"],
                            current_left_extract_result["descps"],
                            prev_left_extract_result["mask"],
                            current_left_extract_result["mask"],
                            kf_prev.image.shape,
                            kf_curr.image.shape,
                        )
                    with Timer(name="[cached result[2/3]]", text="[{name}] Elapsed time: {milliseconds:.03f} ms", logger=self.logger.debug):
                        prev_keypoints = prev_left_extract_result["kpts"][0]  # (n, 2)
                        current_keypoints = current_left_extract_result["kpts"][0]  # (n, 2)
                        match_indices = match_result["match_indices"][0].copy()
                        idx_to_origial = range(len(prev_keypoints))

                        valid_mask = match_indices != -1
                        kpt_pre = prev_keypoints[valid_mask]
                        kpt_cur = current_keypoints[match_indices[valid_mask]]
                        idx_valid = np.array(idx_to_origial)[valid_mask]

                        depth = kf_curr.depth

                        logging.debug(f"match cnt: {len(kpt_pre)}")
                        state, _, _, _, inliers = estimate_pose(
                            kpt_pre,
                            kpt_cur,
                            depth,
                            self.K,
                            idx_valid
                        )
                        inlier_set = set(inliers)
                        if len(inlier_set) > 20:
                            for idx in range(len(match_indices)):
                                if idx not in inlier_set:
                                    match_indices[idx] = -1
                        else:
                            for idx in range(len(match_indices)):
                                match_indices[idx] = -1
                            self.logger.warning(f"match cnt: {len(kpt_pre)} is too small, {len(inlier_set)} inliers.enable velocity constraint")
                            velocity_constraint = gtsam.PriorFactorVector(V(i), np.zeros(3), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.25, 0.25, 0.25])))
                            graph.add(velocity_constraint)

                    with Timer(name="[cached result[3/3]]", text="[{name}] Elapsed time: {milliseconds:.03f} ms", logger=self.logger.debug):
                        count = 0
                        for k, match_idx in enumerate(match_indices):
                            if match_idx != -1:
                                idx_prev = i * _M + k
                                idx_curr = j * _M + match_idx
                                uf_union(idx_prev, idx_curr, uf)
                                count += 1
                        self.logger.debug(f"{i} match {j} after Pnp filter count: {count}")

            with Timer(name="[found track]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
                tracks = uf_all_sets_list(uf, min_component_size=2)
                self.logger.debug(f"Found {len(tracks)} tracks after data association.")
                self._save_sliding_window_landmarks(tracks, extract_info, left_msg.header.stamp)

            with Timer(name="[add track]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
                for landmark in tracks[::1]:
                    # Build a smart factor per track (no explicit landmark variable)
                    disparity_valid = True
                    observations = []
                    for projection in landmark:
                        pose_idx = projection // _M
                        feature_idx = projection % _M
                        disparity = self.keyframe_queue[pose_idx].disparity
                        kpt = extract_info[pose_idx]['kpts'][0][feature_idx]
                        if disparity[int(kpt[1]), int(kpt[0])] < 0.1:
                            disparity_valid = False
                            break
                        observations.append((pose_idx, kpt, disparity))

                    if not disparity_valid or len(observations) < 2:
                        continue

                    # Smart factors require isotropic pixel noise
                    noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
                    params = gtsam.SmartProjectionParams()
                    smart_factor = gtsam_unstable.SmartStereoProjectionPoseFactor(noise, params)

                    calib = gtsam.Cal3_S2Stereo(
                        self.K[0, 0], self.K[1, 1], 0, self.K[0, 2], self.K[1, 2], self.baseline
                    )
                    for pose_idx, kpt, disparity in observations:
                        stereo_meas = gtsam.StereoPoint2(
                            kpt[0],
                            kpt[0] - disparity[int(kpt[1]), int(kpt[0])],
                            kpt[1],
                        )
                        smart_factor.add(stereo_meas, X(pose_idx), calib)
                    graph.add(smart_factor)

        with Timer(name="[Solver]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            params = gtsam.LevenbergMarquardtParams()
            # set iteration limit
            params.setMaxIterations(3)
            params.setVerbosityLM("DEBUG")
            lm = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
            result = lm.optimize()

            self.logger.info(f"ISAM optimization done with {graph.size()} factors and {initial_estimate.size()} variables.")
            self.logger.info(f"Initial error: {graph.error(initial_estimate):.4f}, Final error: {graph.error(result):.4f}")

            for i, keyframe in enumerate(self.keyframe_queue[-_N:]):
                T_i = result.atPose3(X(i)).matrix()
                keyframe.pose = T_i
                keyframe.velocity = result.atVector(V(i))
                keyframe.bias = result.atConstantBias(B(i))
                self.logger.debug(f"Keyframe {i} pose updated:\n{T_i}, at timestamp {keyframe.timestamp}")
                self.logger.debug(f"Bias {i} updated:\n{keyframe.bias}")
                #print("imu error: ", keyframe.preintegrated_imu.error(initial_estimate))

        with Timer(text="[Depth as Color] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            disp_vis = disparity.copy().astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_vis * 4, cv2.COLORMAP_PLASMA)
            disp_color_msg = self.bridge.cv2_to_imgmsg(disp_color, encoding='bgr8')
            disp_color_msg.header = left_msg.header
            self.disparity_pub_vis.publish(disp_color_msg)

        with Timer(name='[Depth as Cloud', text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            # publish depth image and camera info for depth topic (required by DepthCloud)
            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
            depth_msg.header.stamp = left_msg.header.stamp
            depth_msg.header.frame_id = "camera"  # Match TF frame
            self.camera_info_msg.header.stamp = left_msg.header.stamp
            self.camera_info_msg.header.frame_id = "camera"  # Match TF frame
            self.slam_camera_info_pub.publish(self.camera_info_msg)
            self.depth_pub.publish(depth_msg)
        self.logger.debug(f"superpoint cache info: {self.superpoint.infer.cache_info()}")
        self.logger.debug(f"lightglue cache info: {self.light_glue.infer.cache_info()}")
        self.logger.debug(f"estimate_pose cache info: {estimate_pose.cache_info()}")

        with Timer(name="[Publish Odometry]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            self.T_body_last = result.atPose3(X(len(self.keyframe_queue) - 1)).matrix()
            self.V_last = result.atVector(V(len(self.keyframe_queue) - 1))
            self.B_last = result.atConstantBias(B(len(self.keyframe_queue) - 1))
            # publish odometry
            self.odom_pub.publish(np2msg(self.T_body_last, left_msg.header.stamp, "world", "camera", self.V_last))
            # publish TF
            self.tf_broadcaster.sendTransform(np2tf(self.T_body_last, left_msg.header.stamp, "world", "camera"))

            last_keyframe = self.keyframe_queue[-2]
            current_keyframe = self.keyframe_queue[-1]
            if keyframe_check(last_keyframe.pose, current_keyframe.pose) or current_keyframe.timestamp - last_keyframe.timestamp > 3.0:
                self.keyframe_pose_pub.publish(np2msg(current_keyframe.pose, left_msg.header.stamp, "world", "camera", current_keyframe.velocity))
                self.keyframe_image_pub.publish(left_msg)
                self.keyframe_depth_pub.publish(depth_msg)
            else:
                self.keyframe_queue.pop()

        return {
            "stats": {
                "process_cnt": self.process_cnt,
            },
            "metrics": {
                "num_keyframes": len(self.keyframe_queue),
                "num_tracks": len(tracks),
                "num_factors": graph.size(),
                "num_variables": initial_estimate.size(),
                "initial_error": graph.error(initial_estimate),
                "final_error": graph.error(result),
            },
        }


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description='Run TinyNav perception node.')
    parser.add_argument('--verbose_timer', action='store_true', help='Print timing for key pipeline stages.')
    parsed_args, _ = parser.parse_known_args(args=sys.argv[1:] if args is None else args)

    perception_node = PerceptionNode(verbose_timer=parsed_args.verbose_timer)
    #imu_propagator_node = ImuPropagatorNode()

    executor = rclpy.executors.MultiThreadedExecutor(1)
    executor.add_node(perception_node)
    #executor.add_node(imu_propagator_node)
    executor.spin()
    perception_node.destroy_node()
    #imu_propagator_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
