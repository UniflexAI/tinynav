import argparse
import json
import logging
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
from tinynav.core.math_utils import rot_from_two_vector, np2msg, np2tf, estimate_pose, se3_inv
from tinynav.core.math_utils import uf_init, uf_union, uf_all_sets_list
from tf2_ros import TransformBroadcaster
import asyncio
import gtsam
import gtsam_unstable
from collections import deque
from dataclasses import dataclass

from gtsam.symbol_shorthand import X, B, V

_N = 5
_M = 1000

_MIN_FEATURES = 20
_KEYFRAME_MIN_DISTANCE = 0.1    # unit: meter
_KEYFRAME_MIN_ROTATE_DEGREE = 0.1 # unit: degree

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def keyframe_check(T_i, T_j):
    T_ij = se3_inv(T_i) @ T_j
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
        self.odom_pub = self.create_publisher(Odometry, "/slam/odometry", 10)
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

        # Noise model (continuous-time)
        # for Realsense D435i
        accel_noise_density = 0.25     # [m/s^2/√Hz]
        gyro_noise_density = 0.00005 # [rad/s/√Hz]
        bias_acc_rw_sigma = 0.001
        bias_gyro_rw_sigma = 0.0001
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
        self.logger.info("PerceptionNode initialized.")
        self.process_cnt = 0

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
            processed = asyncio.run(self.process(left_msg, right_msg))
        if processed:
            processed["stats"]["loop_ms"] = (time.perf_counter() - loop_start) * 1000.0
            self.stats_pub.publish(String(data=json.dumps(processed)))

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
            self.keyframe_queue.append(
                Keyframe(
                    timestamp=current_timestamp,
                    image=left_img,
                    disparity=disparity,
                    depth=depth,
                    pose=self.T_body_last,
                    velocity=np.zeros(3),
                    bias=gtsam.imuBias.ConstantBias(),
                    preintegrated_imu=gtsam.PreintegratedCombinedMeasurements(self.pre_integration_params, gtsam.imuBias.ConstantBias()),
                    latest_imu_timestamp=current_timestamp
                )
            )
            return {
            "stats": {"process_cnt": 0},
            "metrics": {"num_keyframes": 0, "num_tracks": 0, "num_factors": 0, "num_variables": 0, "initial_error": 0.0, "final_error": 0.0}
        }

        with Timer(name="[Stereo Inference]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            disparity, depth = await self.stereo_engine.infer(left_img, right_img, np.array([[self.baseline]]), np.array([[self.K[0,0]]]))
            self.image_shape = (
                left_img.shape)

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

            self.imu_measurements.popleft()
        # specially process the last imu
        if len(self.imu_measurements) > 0 and current_timestamp - self.keyframe_queue[-1].latest_imu_timestamp > 0.001:
            timestamp, accel, gyro = self.imu_measurements[0]
            dt = current_timestamp - self.keyframe_queue[-1].latest_imu_timestamp
            self.keyframe_queue[-1].preintegrated_imu.integrateMeasurement(accel, gyro, dt)

        with Timer(name="[PnP]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
        # do simple pose estimation between last keyframe and current frame
            if self.last_keyframe_features is None:
                kpts_curr, desc_curr = self.superpoint.detectAndCompute(left_img)
                self.last_keyframe_features = (kpts_curr, desc_curr)
                self.last_keyframe_img = left_img
            with Timer(name="  [LightGlue]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
                kpts_prev, desc_prev = self.last_keyframe_features
                idx_prev, idx_curr = self.light_glue.match(desc_prev, desc_curr := self.superpoint.detectAndCompute(left_img)[1], kpts_prev, kpts_curr := self.superpoint.detectAndCompute(left_img)[0])
            with Timer(name="  [Index Features]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
                kpts_prev = kpts_prev[idx_prev]
                kpts_curr = kpts_curr[idx_curr]
            with Timer(name="  [Estimate Pose]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
                T_kf_curr, inliers, pnp_diagnostics = estimate_pose(kpts_prev, kpts_curr, self.keyframe_queue[-1].depth, self.K, idx_curr)
            if pnp_diagnostics is None:
                self.logger.warning("PnP failed, skipping frame.")
                return {
                    "stats": {"process_cnt": self.process_cnt},
                    "metrics": {
                        "num_keyframes": len(self.keyframe_queue),
                        "num_tracks": 0,
                        "num_factors": 0,
                        "num_variables": 0,
                        "initial_error": 0.0,
                        "final_error": 0.0,
                        "pnp_success": False,
                        "pnp_inlier_count": 0,
                        "pnp_inlier_ratio": 0.0,
                        "pnp_match_count": 0,
                    }
                }

            metrics = {
                "pnp_success": True,
                "pnp_inlier_count": pnp_diagnostics["inlier_count"],
                "pnp_inlier_ratio": pnp_diagnostics["inlier_ratio"],
                "pnp_match_count": pnp_diagnostics["match_count"],
            }

            self.logger.debug(f"estimate_pose cache info: {estimate_pose.cache_info()}")
            if np.linalg.norm(T_kf_curr[:3, 3]) > 10.0:
                self.logger.warning("Pose estimation failed, skipping frame.")
                return

        # check if is keyframe
        if keyframe_check(np.eye(4), T_kf_curr):
            self.last_keyframe_features = (kpts_curr, desc_curr)
            self.last_keyframe_img = left_img
            self.keyframe_queue.append(
                Keyframe(
                    timestamp=current_timestamp,
                    image=left_img,
                    disparity=disparity,
                    depth=depth,
                    pose=self.keyframe_queue[-1].pose @ T_kf_curr,
                    velocity=self.keyframe_queue[-1].velocity,
                    bias=gtsam.imuBias.ConstantBias(),
                    preintegrated_imu=gtsam.PreintegratedCombinedMeasurements(self.pre_integration_params, gtsam.imuBias.ConstantBias()),
                    latest_imu_timestamp=current_timestamp
                )
            )
            if len(self.keyframe_queue) > _N:
                self.keyframe_queue.pop(0)
        if len(self.keyframe_queue) < _N:
            self.logger.info(f"Not enough keyframes yet. Current: {len(self.keyframe_queue)}, Required: {_N}")
            return {
                "stats": {"process_cnt": self.process_cnt},
                "metrics": {
                    "num_keyframes": len(self.keyframe_queue),
                    "num_tracks": 0,
                    "num_factors": 0,
                    "num_variables": 0,
                    "initial_error": 0.0,
                    "final_error": 0.0,
                    **metrics,
                }
            }

        graph = gtsam.NonlinearFactorGraph()
        initial_values = gtsam.Values()

        self.logger.debug("Preintegration params:")
        self.logger.debug(f"Accelerometer Covariance: {self.pre_integration_params.getAccelerometerCovariance()}")
        self.logger.debug(f"Gyroscope Covariance: {self.pre_integration_params.getGyroscopeCovariance()}")
        self.logger.debug(f"Integration Covariance: {self.pre_integration_params.getIntegrationCovariance()}")
        self.logger.debug(f"Bias Acc Covariance: {self.pre_integration_params.getBiasAccCovariance()}")
        self.logger.debug(f"Bias Omega Covariance: {self.pre_integration_params.getBiasOmegaCovariance()}")
        self.logger.debug(f"Omega Coriolis: {self.pre_integration_params.getOmegaCoriolis()}")

        # 1. add variables, priors, imu factors
        with Timer(name="[IMU Factor]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            # Prior noise models
            pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]))
            velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
            bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)

            # Prior for the first pose, velocity, and bias
            first_pose = self.keyframe_queue[0].pose
            first_vel = self.keyframe_queue[0].velocity
            first_bias = self.keyframe_queue[0].bias

            graph.add(gtsam.PriorFactorPose3(X(0), Matrix4x4ToGtsamPose3(first_pose), pose_noise))
            graph.add(gtsam.PriorFactorVector(V(0), first_vel, velocity_noise))
            graph.add(gtsam.PriorFactorConstantBias(B(0), first_bias, bias_noise))

            # Insert initial values
            for i, kf in enumerate(self.keyframe_queue):
                initial_values.insert(X(i), Matrix4x4ToGtsamPose3(kf.pose))
                initial_values.insert(V(i), kf.velocity)
                initial_values.insert(B(i), kf.bias)

            imu_bias_rw_sigma = np.array([0.001, 0.001, 0.001, 1e-4, 1e-4, 1e-4])
            bias_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.sqrt(kf.latest_imu_timestamp - self.keyframe_queue[i - 1].latest_imu_timestamp) * imu_bias_rw_sigma)
            for i in range(1, len(self.keyframe_queue)):
                prev_kf = self.keyframe_queue[i - 1]
                curr_kf = self.keyframe_queue[i]

                self.logger.debug(f"Preintegrated IMU measurements between keyframes {i-1} and {i}:")
                self.logger.debug(f"  Delta t: {curr_kf.preintegrated_imu.deltaTij()}")
                self.logger.debug(f"  Delta P: {curr_kf.preintegrated_imu.deltaPij()}")
                self.logger.debug(f"  Delta V: {curr_kf.preintegrated_imu.deltaVij()}")
                self.logger.debug(f"  Delta R: {curr_kf.preintegrated_imu.deltaRij().matrix()}")

                imu_factor = gtsam.CombinedImuFactor(
                    X(i - 1), V(i - 1), X(i), V(i),
                    B(i - 1), B(i),
                    curr_kf.preintegrated_imu
                )
                graph.add(imu_factor)
                graph.add(gtsam.BetweenFactorConstantBias(B(i - 1), B(i), gtsam.imuBias.ConstantBias(), bias_noise_model))

        # 2. feature track across keyframes
        max_dist = 5.0 # meters
        feature_tracks = []
        landmark_positions = []
        correspondence_info = []

        with Timer(name="[Feature Matching]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            all_kpts = []
            all_desc = []
            for kf in self.keyframe_queue:
                kpts, desc = self.superpoint.detectAndCompute(kf.image)
                all_kpts.append(kpts)
                all_desc.append(desc)
            for i in range(len(self.keyframe_queue)):
                for j in range(i + 1, len(self.keyframe_queue)):
                    idx_i, idx_j = self.light_glue.match(all_desc[i], all_desc[j], all_kpts[i], all_kpts[j])
                    kpts_i = all_kpts[i][idx_i]
                    kpts_j = all_kpts[j][idx_j]
                    pts_3d = []
                    pts_2d = []
                    valid_idx_i = []
                    for m, (kp_i, kp_j) in enumerate(zip(kpts_i, kpts_j)):
                        u, v = int(kp_i[0]), int(kp_i[1])
                        if 0 <= v < self.keyframe_queue[i].depth.shape[0] and 0 <= u < self.keyframe_queue[i].depth.shape[1]:
                            Z = self.keyframe_queue[i].depth[v, u]
                            if Z > 0.1 and Z < max_dist:
                                pts_3d.append(depth_to_point(kp_i, Z, self.K))
                                pts_2d.append(kp_j)
                                valid_idx_i.append(idx_i[m])
                    if len(pts_3d) == 0:
                        continue
                    T_iw = self.keyframe_queue[i].pose
                    landmarks_i = [T_iw[:3, :3] @ pt + T_iw[:3, 3] for pt in pts_3d]
                    feature_tracks.append((i, j, valid_idx_i, idx_j[:len(valid_idx_i)], landmarks_i, pts_2d))
                    correspondence_info.append((i, j, len(pts_3d)))

        # 3. robust union-find on landmarks with outlier rejection
        track_map = {}
        point_records = []
        for track_id, (i, j, idx_i_list, idx_j_list, landmarks_i, pts_2d) in enumerate(feature_tracks):
            for n, (idx_i, idx_j, landmark) in enumerate(zip(idx_i_list, idx_j_list, landmarks_i)):
                key_i = (i, idx_i)
                key_j = (j, idx_j)
                point_records.append((track_id, n, key_i, key_j, landmark))

        uf = uf_init(len(point_records))
        key_to_indices = {}
        for idx, (_, _, key_i, key_j, _) in enumerate(point_records):
            key_to_indices.setdefault(key_i, []).append(idx)
            key_to_indices.setdefault(key_j, []).append(idx)
        for indices in key_to_indices.values():
            for k in range(1, len(indices)):
                uf_union(uf, indices[0], indices[k])
        sets = uf_all_sets_list(uf)

        for s in sets:
            if len(s) < 2:
                continue
            pts = np.array([point_records[idx][4] for idx in s])
            mean = np.mean(pts, axis=0)
            dists = np.linalg.norm(pts - mean, axis=1)
            inliers = [s[idx] for idx, d in enumerate(dists) if d < 0.3]
            if len(inliers) < 2:
                continue
            landmark_positions.append(np.mean([point_records[idx][4] for idx in inliers], axis=0))
            lm_id = len(landmark_positions) - 1
            for idx in inliers:
                _, _, key_i, key_j, _ = point_records[idx]
                track_map[key_i] = lm_id
                track_map[key_j] = lm_id

        # 4. add projection factors
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        K_cal = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
        meas_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

        with Timer(name="[Projection Factors]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            for track_id, (i, j, idx_i_list, idx_j_list, landmarks_i, pts_2d) in enumerate(feature_tracks):
                kpts_i = all_kpts[i][idx_i_list]
                kpts_j = all_kpts[j][idx_j_list]
                for idx_i, idx_j, pt_j in zip(idx_i_list, idx_j_list, pts_2d):
                    key_i = (i, idx_i)
                    key_j = (j, idx_j)
                    if key_i in track_map and key_j in track_map and track_map[key_i] == track_map[key_j]:
                        lm_id = track_map[key_i]
                        landmark = landmark_positions[lm_id]
                        point3 = gtsam.Point3(landmark)
                        if not initial_values.exists(gtsam.symbol('l', lm_id)):
                            initial_values.insert(gtsam.symbol('l', lm_id), point3)
                        factor = gtsam.GenericProjectionFactorCal3_S2(
                            gtsam.Point2(pt_j[0], pt_j[1]),
                            meas_noise,
                            X(j),
                            gtsam.symbol('l', lm_id),
                            K_cal,
                            Matrix4x4ToGtsamPose3(self.T_imu_body_to_camera)
                        )
                        graph.add(factor)

        with Timer(name="[Optimize]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
            initial_error = graph.error(initial_values)
            result = optimizer.optimize()
            final_error = graph.error(result)

        self.logger.debug(f"Correspondence info (kf_i, kf_j, num_matches): {correspondence_info}")
        self.logger.debug(f"Number of landmarks: {len(landmark_positions)}")
        self.logger.debug(f"Number of factors: {graph.size()}")
        self.logger.debug(f"Initial error: {initial_error}")
        self.logger.debug(f"Final error: {final_error}")
        self.logger.debug(f"estimate_pose cache info: {estimate_pose.cache_info()}")

        with Timer(name="[Publish Odometry]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            self.T_body_last = result.atPose3(X(len(self.keyframe_queue) - 1)).matrix()
            self.V_last = result.atVector(V(len(self.keyframe_queue) - 1))
            # publish odometry
            self.odom_pub.publish(np2msg(self.T_body_last, left_msg.header.stamp, "world", "camera", self.V_last))
            # publish TF
            self.tf_broadcaster.sendTransform(np2tf(self.T_body_last, left_msg.header.stamp, "world", "camera"))
            # publish camera info
            self.camera_info_msg.header.stamp = left_msg.header.stamp
            self.camera_info_msg.header.frame_id = "camera"
            self.slam_camera_info_pub.publish(self.camera_info_msg)
            # publish depth
            depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding='32FC1')
            depth_msg.header.stamp = left_msg.header.stamp
            depth_msg.header.frame_id = "camera"
            self.depth_pub.publish(depth_msg)
            # publish depth_vis 0->255, uint8
            disparity_vis = (disparity - disparity.min()) / (disparity.max() - disparity.min()) * 255
            disparity_vis = disparity_vis.astype(np.uint8)
            disparity_vis = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
            disparity_msg = self.bridge.cv2_to_imgmsg(disparity_vis, encoding='bgr8')
            disparity_msg.header.stamp = left_msg.header.stamp
            disparity_msg.header.frame_id = "camera"
            self.disparity_pub_vis.publish(disparity_msg)

            # publish latest keyframe pose, image, depth if available
            if self.keyframe_queue:
                kf = self.keyframe_queue[-1]
                self.keyframe_pose_pub.publish(np2msg(kf.pose, left_msg.header.stamp, "world", "camera", kf.velocity))
                keyframe_image_msg = self.bridge.cv2_to_imgmsg(kf.image, encoding='mono8')
                keyframe_image_msg.header.stamp = left_msg.header.stamp
                keyframe_image_msg.header.frame_id = "camera"
                self.keyframe_image_pub.publish(keyframe_image_msg)
                keyframe_depth_msg = self.bridge.cv2_to_imgmsg(kf.depth.astype(np.float32), encoding='32FC1')
                keyframe_depth_msg.header.stamp = left_msg.header.stamp
                keyframe_depth_msg.header.frame_id = "camera"
                self.keyframe_depth_pub.publish(keyframe_depth_msg)

        return {
            "stats": {"process_cnt": self.process_cnt},
            "metrics": {
                "num_keyframes": len(self.keyframe_queue),
                "num_tracks": len(feature_tracks),
                "num_factors": graph.size(),
                "num_variables": initial_values.size(),
                "initial_error": initial_error,
                "final_error": final_error,
                **metrics,
            }
        }

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description='Run TinyNav perception node.')
    parser.add_argument('--verbose_timer', action='store_true', help='Print timing for key pipeline stages.')
    parsed_args = parser.parse_args(args=sys.argv[1:] if args is None else args)

    perception_node = PerceptionNode(verbose_timer=parsed_args.verbose_timer)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(perception_node)
    executor.spin()
    perception_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
