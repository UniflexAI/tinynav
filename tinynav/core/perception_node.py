import argparse
import logging
import sys
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import rclpy
from codetiming import Timer
from cv_bridge import CvBridge
from models_trt import LightGlueTRT, SuperPointTRT, StereoEngineTRT, TRTFusionModel
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, CameraInfo
from rclpy.qos import QoSProfile, ReliabilityPolicy
from math_utils import rot_from_two_vector, np2msg, np2tf, estimate_pose
from tf2_ros import TransformBroadcaster
import asyncio
import gtsam
from collections import deque

_MIN_FEATURES = 20
_KEYFRAME_MIN_DISTANCE = 1.0 # uint: meter
_KEYFRAME_MIN_ROTATE_DEGREE = 5 # uint: degree

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def Matrix4x4ToGtsamPose3(T: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(T[:3, 3]))

def depth_to_point(kp, depth, K):
    u, v = int(kp[0]), int(kp[1])
    Z = depth
    X = (u - K[0,2]) * Z / K[0,0]
    Y = (v - K[1,2]) * Z / K[1,1]
    return np.array([X, Y, Z])

def stamp_to_index(stamp):
    nano_s = np.int64(stamp.sec) * 1_000_000_000 + np.int64(stamp.nanosec)
    masked = np.int64(nano_s & 0x00FFFFFFFFFFFFFF)
    return masked

class PerceptionNode(Node):
    def __init__(self, verbose_timer: bool = True):
        super().__init__("perception_node")
        self.verbose_timer = verbose_timer
        self.logger = logging.getLogger(__name__)
        # self.timer_logger = self.logger.info if verbose_timer else self.logger.debug
        # model
        self.superpoint = SuperPointTRT()
        self.light_glue = LightGlueTRT()
        self.trt_fusion_model = TRTFusionModel()

        self.last_keyframe_img = None
        self.last_keyframe_features = None
        self.has_first_keyframe = False

        self.stereo_engine = StereoEngineTRT()
        # intrinsic
        self.baseline = None
        self.K = None
        self.image_shape = None

        self.T_body_last = None
        self.V_last = None
        self.B_last = None

        self.Tcb = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=3000)
        
        self.imu_accel_sub = Subscriber(self, Imu, "/camera/camera/accel/sample", qos_profile=qos_profile)
        self.imu_gyro_sub = Subscriber(self, Imu, "/camera/camera/gyro/sample", qos_profile=qos_profile)
        self.imu_ts = ApproximateTimeSynchronizer([self.imu_accel_sub, self.imu_gyro_sub], queue_size=10, slop=0.005)
        self.imu_ts.registerCallback(self.imu_callback)
        self.imu_last_received_timestamp = None
        
        self.accel_sub = self.create_subscription(Imu, "/camera/camera/accel/sample", self.accel_callback, qos_profile)
        self.camerainfo_sub = self.create_subscription(CameraInfo, "/camera/camera/infra2/camera_info", self.info_callback, 10)

        self.left_sub = Subscriber(self, Image, "/camera/camera/infra1/image_rect_raw")
        self.right_sub = Subscriber(self, Image, "/camera/camera/infra2/image_rect_raw")
        self.ts = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=10, slop=0.02)
        self.ts.registerCallback(self.images_callback)
        self.odom_pub = self.create_publisher(Odometry, "/slam/odometry", 10)
        self.slam_camera_info_pub = self.create_publisher(CameraInfo, "/slam/camera_info", 10)
        self.depth_pub = self.create_publisher(Image, "/slam/depth", 10)
        self.disparity_pub_vis = self.create_publisher(Image, '/slam/disparity_vis', 10)
        self.keyframe_pose_pub = self.create_publisher(Odometry, "/slam/keyframe_odom", 10)
        self.keyframe_image_pub = self.create_publisher(Image, "/slam/keyframe_image", 10)
        self.keyframe_depth_pub = self.create_publisher(Image, "/slam/keyframe_depth", 10)

        self.ros_timestamp = None
        self.last_ros_timestamp = None

        self.accel_readings = []
        self.last_processed_timestamp = 0.0

        # 200Hz IMU
        self.gravity_in_camera_frame = None

        self.camera_info_msg = None

        self.isam_params = gtsam.ISAM2Params()
        self.isam_params.setRelinearizeThreshold(0.01)
        self.isam_params.relinearizeSkip = 1
        # the default factorization is Cholesky
        # the QR factorization is more numerically stable but slower
        self.isam_params.factorization = gtsam.ISAM2Params.Factorization.QR

        # lag is 2.0 seconds, since we limit the pose update rate to 10Hz
        self.fixed_lag_isam = gtsam.IncrementalFixedLagSmoother(2.0, self.isam_params)

        self.timestamp_last = None
        self.landmark_id = None
        self.kGravity = 9.81

        self.update_explicitly_iterations = 5
        # Noise model (continuous-time)
        # for Realsense D435i
        accel_noise_density = 0.25     # [m/s^2/√Hz]
        gyro_noise_density = 0.00005 # [rad/s/√Hz]
        bias_acc_rw_sigma = 0.001
        bias_gyro_rw_sigma = 0.0001
        self.pre_integration_params = gtsam.PreintegrationCombinedParams.MakeSharedU(self.kGravity)
        self.pre_integration_params.setAccelerometerCovariance((accel_noise_density**2) * np.eye(3))
        self.pre_integration_params.setGyroscopeCovariance((gyro_noise_density**2) * np.eye(3))
        self.pre_integration_params.setIntegrationCovariance(1e-8 * np.eye(3))
        self.pre_integration_params.setBiasAccCovariance(np.eye(3) * bias_acc_rw_sigma**2)
        self.pre_integration_params.setBiasOmegaCovariance(np.eye(3) * bias_gyro_rw_sigma**2)
        self.pre_integration_params.setUse2ndOrderCoriolis(False)
        self.pre_integration_params.setOmegaCoriolis(np.array([0.0, 0.0, 0.0]))

        self.pre_integrated_imu_measurements = gtsam.PreintegratedCombinedMeasurements(self.pre_integration_params, gtsam.imuBias.ConstantBias())

        self.last_disparity = None
        self.T_imu_body_to_camera = np.array(
                            [[1, 0, 0, 0],
                             [0, 0, -1, 0], 
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]])
        self.T_camera_to_imu = np.linalg.inv(self.T_imu_body_to_camera)
        self.imu_measurements = deque(maxlen=1000)

        self.match_tracker = {}
        self.graph = None
        self.initial_estimate = None
        self.symbol_to_timestamp = None
        self.last_keyframe_pose = None
        self.logger.info("PerceptionNode initialized.")

    async def feature_matching(self, prev_features, current_image, image_shape):
            match_task = asyncio.create_task(self.trt_fusion_model.infer(
                prev_features["kpts"],
                prev_features["descps"],
                prev_features["mask"],
                current_image,
                image_shape,
                image_shape))
            return await match_task

    def info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            fx = self.K[0, 0]
            Tx = msg.p[3]  # From the right camera's projection matrix
            self.baseline = -Tx / fx
            self.image_shape = np.array([msg.width, msg.height], dtype=np.int64)
            self.get_logger().info(f"Camera intrinsics and baseline received. Baseline: {self.baseline:.4f}m")
            self.camera_info_msg = msg
            self.destroy_subscription(self.camerainfo_sub)

    def accel_callback(self, msg):
        self.accel_readings.append(msg.linear_acceleration)
        if len(self.accel_readings) >= 10 and self.T_body_last is None:
            self.T_body_last = np.eye(4)
            accel_data = np.array([(a.x, a.y, a.z) for a in self.accel_readings])
            gravity_cam = np.mean(accel_data, axis=0)
            gravity_cam /= np.linalg.norm(gravity_cam)
            gravity_world = np.array([0.0, 0.0, 1.0])
            self.T_body_last[:3, :3] = rot_from_two_vector(gravity_cam, gravity_world)
            self.last_keyframe_pose = self.T_body_last
            self.get_logger().info("Initial pose set from accelerometer data.")
            self.get_logger().info(f"Initial rotation matrix:\n{self.T_body_last}")
            self.destroy_subscription(self.accel_sub)
            self.gravity_in_camera_frame = gravity_cam
            self.T_body_last = self.T_body_last @ self.T_imu_body_to_camera

    def imu_callback(self, accel_msg, gyro_msg):
        current_timestamp = stamp_to_index(accel_msg.header.stamp)

        # if the timestamp jump is too large, it means the IMU is not working properly
        if self.imu_last_received_timestamp is not None and current_timestamp - self.imu_last_received_timestamp > 10_000_000:
            delta_timestamp = current_timestamp - self.imu_last_received_timestamp
            self.get_logger().warning(f"IMU timestamp jump {delta_timestamp * 1e-6} ms is too large, it means the IMU is not working properly")
        self.imu_last_received_timestamp = current_timestamp

        T_camera_to_imu = self.T_imu_body_to_camera[:3, :3].T
        accel_data = np.array([[accel_msg.linear_acceleration.x], [accel_msg.linear_acceleration.y], [accel_msg.linear_acceleration.z]])
        gyro_data = np.array([[gyro_msg.angular_velocity.x], [gyro_msg.angular_velocity.y], [gyro_msg.angular_velocity.z]])
        accel_data_imu = T_camera_to_imu @ accel_data
        gyro_data_imu = T_camera_to_imu @ gyro_data
        self.imu_measurements.append([current_timestamp, accel_data_imu.flatten(), gyro_data_imu.flatten()])



    def images_callback(self, left_msg, right_msg):
        current_timestamp = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9
        if current_timestamp - self.last_processed_timestamp < 0.1333:
            return
        self.last_processed_timestamp = current_timestamp
        self.ros_timestamp = left_msg.header.stamp
        with Timer(name="Perception Loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.info):
            asyncio.run(self.process(left_msg, right_msg))

    async def process(self, left_msg, right_msg):
        if self.K is None or self.T_body_last is None or self.image_shape is None:
            return
        left_img = self.bridge.imgmsg_to_cv2(left_msg, "mono8")
        right_img = self.bridge.imgmsg_to_cv2(right_msg, "mono8")
        current_timestamp = stamp_to_index(left_msg.header.stamp)

        with Timer(name="[Model Inference]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            if self.last_keyframe_img is None:
                stereo_task = asyncio.create_task(self.stereo_engine.infer(left_img, right_img, np.array([[self.baseline]]), np.array([[self.K[0,0]]])))

                symbol_to_timestamp = {}
                graph = gtsam.NonlinearFactorGraph()

                pose_symbol= gtsam.symbol_shorthand.X(current_timestamp)
                prior_factor = gtsam.PriorFactorPose3(pose_symbol, Matrix4x4ToGtsamPose3(self.T_body_last), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])))
                graph.add(prior_factor)

                initial_estimate = gtsam.Values()
                initial_estimate.insert(pose_symbol, Matrix4x4ToGtsamPose3(self.T_body_last))
                symbol_to_timestamp[pose_symbol] = current_timestamp * 1e-9

                self.landmark_id = 0
                bias_symbol = gtsam.symbol_shorthand.B(current_timestamp)
                initial_estimate.insert(bias_symbol, gtsam.imuBias.ConstantBias())
                self.B_last = gtsam.imuBias.ConstantBias()
                bias_initialization_factor = gtsam.PriorFactorConstantBias(bias_symbol, gtsam.imuBias.ConstantBias(), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])))
                graph.add(bias_initialization_factor)
                symbol_to_timestamp[bias_symbol] = current_timestamp * 1e-9

                self.V_last = np.zeros(3)
                velocity_symbol = gtsam.symbol_shorthand.V(current_timestamp)
                prior_velocity_factor = gtsam.PriorFactorVector(velocity_symbol, np.zeros(3), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2, 1e-2, 1e-2])))
                graph.add(prior_velocity_factor)
                initial_estimate.insert(velocity_symbol, self.V_last)
                symbol_to_timestamp[velocity_symbol] = current_timestamp * 1e-9

                self.graph = graph
                self.initial_estimate = initial_estimate
                self.symbol_to_timestamp = symbol_to_timestamp
                self.last_keyframe_img = left_img
                self.last_keyframe_features = await self.superpoint.infer(left_img)
                disparity, depth = await stereo_task
                self.last_disparity = disparity

                self.timestamp_last = current_timestamp
                self.last_ros_timestamp = self.ros_timestamp
                # print(f"T_body_last: {self.T_body_last}")
                return

            # Create tasks for all three operations to run concurrently
            feature_matching_task = asyncio.create_task(self.feature_matching(self.last_keyframe_features, left_img, self.image_shape))
            stereo_task = asyncio.create_task(self.stereo_engine.infer(left_img, right_img, np.array([[self.baseline]]), np.array([[self.K[0,0]]])))
            
            # Create task for ISAM Update to run concurrently
            def isam_update_func():
                with Timer(name="[ISAM Update]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.info):
                    self.fixed_lag_isam.update(self.graph, self.initial_estimate, self.symbol_to_timestamp)
                    for _ in range(self.update_explicitly_iterations - 1):
                        self.fixed_lag_isam.update(gtsam.NonlinearFactorGraph(), gtsam.Values(), {})

                    result = self.fixed_lag_isam.calculateEstimate()
                    pose_last_symbol = gtsam.symbol_shorthand.X(self.timestamp_last)
                    pose = result.atPose3(pose_last_symbol)
                    self.T_body_last = np.array(pose.matrix())
                    bias_last_symbol = gtsam.symbol_shorthand.B(self.timestamp_last)
                    self.B_last = result.atConstantBias(bias_last_symbol)
                    velocity_last_symbol = gtsam.symbol_shorthand.V(self.timestamp_last)
                    self.V_last = np.array(result.atVector(velocity_last_symbol)).reshape(-1)
                    self.pre_integrated_imu_measurements = gtsam.PreintegratedCombinedMeasurements(self.pre_integration_params, self.B_last)
            isam_task = asyncio.to_thread(isam_update_func)
            # Wait for all tasks concurrently using gather
            with Timer(name="[Wait for tasks]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
                prev_curr_match_result, (disparity, depth), _ = await asyncio.gather(
                    feature_matching_task,
                    stereo_task,
                    isam_task
                )

            prev_keypoints = self.last_keyframe_features["kpts"][0]  # (n, 2)
            current_keypoints = prev_curr_match_result["kpts"][0]  # (n, 2)
            match_indices = prev_curr_match_result["match_indices"][0]
            valid_mask = match_indices != -1
            kpt_pre = prev_keypoints[valid_mask]
            kpt_cur = current_keypoints[match_indices[valid_mask]]
            matches = [[i, match_indices[i]] for i in range(len(match_indices)) if match_indices[i] != -1]
            self.last_keyframe_features = {
                "kpts": prev_curr_match_result["kpts"],
                "descps": prev_curr_match_result["descps"],
                "mask": prev_curr_match_result["mask"],
            }

        with Timer(text="[Depth as Color] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            disp_vis = disparity.copy().astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_vis * 4, cv2.COLORMAP_PLASMA)
            disp_color_msg = self.bridge.cv2_to_imgmsg(disp_color, encoding='bgr8')
            disp_color_msg.header = left_msg.header
            self.disparity_pub_vis.publish(disp_color_msg)

        with Timer(name='[Depth as Cloud', text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            # publish depth image and camera info for depth topic (required by DepthCloud)
            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
            depth_msg.header.stamp = self.ros_timestamp
            depth_msg.header.frame_id = "camera"  # Match TF frame
            self.camera_info_msg.header.stamp = self.ros_timestamp
            self.camera_info_msg.header.frame_id = "camera"  # Match TF frame
            self.slam_camera_info_pub.publish(self.camera_info_msg)
            self.depth_pub.publish(depth_msg)
        with Timer(name="Compute Pose", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            success, T_pre_curr, inliers_2d, inliers_3d, inliers = estimate_pose(kpt_pre, kpt_cur, depth, self.K)
            if not success:
                self.logger.warning("Failed to estimate pose")
                return
            T_body_curr = self.T_body_last @ np.linalg.inv(self.T_imu_body_to_camera) @ T_pre_curr @ self.T_imu_body_to_camera


        with Timer(name="[ISAM Processing]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.info):
            symbol_to_timestamp = {}
            imu_exists = False
            while len(self.imu_measurements) > 0:
                timestamp, accel, gyro = self.imu_measurements[0]
                if timestamp < self.timestamp_last:
                    self.imu_measurements.popleft()
                elif timestamp > current_timestamp:
                    break
                else:
                    self.pre_integrated_imu_measurements.integrateMeasurement(accel, gyro, 0.005)
                    self.imu_measurements.popleft()
                    imu_exists = True

            with Timer(name="[Factor Graph Construction]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
                nav_state_last = gtsam.NavState(Matrix4x4ToGtsamPose3(self.T_body_last), self.V_last)
                nav_state_curr = self.pre_integrated_imu_measurements.predict(nav_state_last, self.B_last)
                T_body_curr = np.array(nav_state_curr.pose().matrix())
                V_curr_predicted = np.array(nav_state_curr.velocity())
                cal3_s2_intrinsics = gtsam.Cal3_S2Stereo(self.K[0,0], self.K[1,1], 0, self.K[0,2], self.K[1,2], self.baseline)
                graph = gtsam.NonlinearFactorGraph()
                initial_estimate = gtsam.Values()
                pose_pre_symbol = gtsam.symbol_shorthand.X(self.timestamp_last)
                pose_curr_symbol = gtsam.symbol_shorthand.X(current_timestamp)

                initial_estimate.insert(pose_curr_symbol, Matrix4x4ToGtsamPose3(T_body_curr))
                new_match_tracker = {}
                body_P_sensor = Matrix4x4ToGtsamPose3(self.T_camera_to_imu)
                for i in range(0, len(kpt_pre), 4):
                    if self.last_disparity[int(kpt_pre[i, 1]), int(kpt_pre[i, 0])] < 0.1 or disparity[int(kpt_cur[i, 1]), int(kpt_cur[i, 0])] < 0.1:
                        continue
                    left_feature_index, right_feature_index = matches[i]
                    is_new_landmark = False
                    if left_feature_index in self.match_tracker:
                        landmark_id = self.match_tracker[left_feature_index]
                        new_match_tracker[right_feature_index] = landmark_id
                    else:
                        is_new_landmark = True
                        landmark_id = self.landmark_id
                        new_match_tracker[right_feature_index] = landmark_id
                        self.landmark_id += 1
                    landmark_symbol = gtsam.symbol_shorthand.L(landmark_id)

                    robust_model = gtsam.noiseModel.Robust.Create(
                        gtsam.noiseModel.mEstimator.Huber(1.41),       # delta parameter, e.g., 1 pixel
                        gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))      # base noise (σ in pixels)
                    )

                    # it's not a bug, it's a feature.
                    # But I don't really understand why it's got the best results.
                    if not is_new_landmark:
                        factor = gtsam.GenericStereoFactor3D(
                                    gtsam.StereoPoint2(
                                        kpt_pre[i, 0],
                                        kpt_pre[i, 0] - self.last_disparity[int(kpt_pre[i, 1]), int(kpt_pre[i, 0])],
                                        kpt_pre[i, 1]
                                    ),
                                    robust_model,
                                    pose_pre_symbol,
                                    landmark_symbol,
                                    cal3_s2_intrinsics,
                                    body_P_sensor
                                )
                        graph.add(factor)
                    factor = gtsam.GenericStereoFactor3D(
                        gtsam.StereoPoint2(
                            kpt_cur[i, 0],
                            kpt_cur[i, 0] - disparity[int(kpt_cur[i, 1]), int(kpt_cur[i, 0])],
                            kpt_cur[i, 1]
                        ),
                        robust_model,
                        pose_curr_symbol,
                        landmark_symbol,
                        cal3_s2_intrinsics,
                        body_P_sensor
                    )
                    graph.add(factor)

                    if is_new_landmark:
                        prev_depth = self.K[0, 0] * self.baseline / self.last_disparity[int(kpt_pre[i, 1]), int(kpt_pre[i, 0])]
                        prev_point = depth_to_point(kpt_pre[i], prev_depth, self.K)
                        T_camera_last = self.T_body_last @ self.T_camera_to_imu
                        prev_point_world = T_camera_last[:3, :3] @ prev_point + T_camera_last[:3, 3]
                        initial_estimate.insert(landmark_symbol, gtsam.Point3(prev_point_world))
                        prior_factor = gtsam.PriorFactorPoint3(landmark_symbol, gtsam.Point3(prev_point_world), gtsam.noiseModel.Isotropic.Sigma(3, 4.0))
                        graph.add(prior_factor)
                    symbol_to_timestamp[landmark_symbol] = current_timestamp * 1e-9

                self.match_tracker = new_match_tracker
                symbol_to_timestamp[pose_curr_symbol] = current_timestamp * 1e-9
                symbol_to_timestamp[pose_pre_symbol] = self.timestamp_last * 1e-9

                bias_pre_symbol = gtsam.symbol_shorthand.B(self.timestamp_last)
                bias_curr_symbol = gtsam.symbol_shorthand.B(current_timestamp)
                velocity_pre_symbol = gtsam.symbol_shorthand.V(self.timestamp_last)
                velocity_curr_symbol = gtsam.symbol_shorthand.V(current_timestamp)
                initial_estimate.insert(bias_curr_symbol, self.B_last)
                initial_estimate.insert(velocity_curr_symbol, V_curr_predicted)
                if imu_exists:
                    imufac = gtsam.CombinedImuFactor(pose_pre_symbol, velocity_pre_symbol, pose_curr_symbol, velocity_curr_symbol, bias_pre_symbol, bias_curr_symbol, self.pre_integrated_imu_measurements)
                    graph.add(imufac)
                    symbol_to_timestamp[bias_pre_symbol] = self.timestamp_last * 1e-9
                    symbol_to_timestamp[bias_curr_symbol] = current_timestamp * 1e-9
                    symbol_to_timestamp[velocity_pre_symbol] = self.timestamp_last * 1e-9
                    symbol_to_timestamp[velocity_curr_symbol] = current_timestamp * 1e-9
                else:
                    self.logger.warning(f"no imu measurements between {(current_timestamp - self.timestamp_last) * 1e-6} ms")
                self.last_disparity = disparity
                self.graph = graph
                self.initial_estimate = initial_estimate
                self.symbol_to_timestamp = symbol_to_timestamp


        with Timer(name="[Publish Odometry]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            # publish odometry
            self.odom_pub.publish(np2msg(self.T_body_last @ self.T_camera_to_imu, self.last_ros_timestamp, "world", "camera"))
            # publish TF
            self.tf_broadcaster.sendTransform(np2tf(self.T_body_last @ self.T_camera_to_imu, self.last_ros_timestamp, "world", "camera"))
            # keyframe checking
            self.last_keyframe_img = left_img
            self.timestamp_last = current_timestamp
            self.last_ros_timestamp = self.ros_timestamp

            def keyframe_check(T_ij):
                t_diff = np.linalg.norm(T_ij[:3, 3])
                cos_theta = (np.trace(T_ij[:3, :3]) - 1) / 2
                r_diff = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
                return t_diff > _KEYFRAME_MIN_DISTANCE or r_diff > _KEYFRAME_MIN_ROTATE_DEGREE
            T_pre_curr = np.linalg.inv(self.last_keyframe_pose) @ self.T_body_last
            if not self.has_first_keyframe or keyframe_check(T_pre_curr):
                    self.last_keyframe_pose = self.T_body_last
                    self.has_first_keyframe = True
                    self.keyframe_pose_pub.publish(np2msg(self.last_keyframe_pose @ self.T_camera_to_imu, self.last_ros_timestamp, "world", "camera"))
                    self.keyframe_image_pub.publish(left_msg)
                    self.keyframe_depth_pub.publish(depth_msg)


def main(args=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("odom.log")],
    )

    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.set_defaults(verbose_timer=True)
    parser.add_argument("--verbose_timer", action="store_true", help="Enable verbose timer output")
    parser.add_argument("--no_verbose_timer", dest="verbose_timer", action="store_false", help="Disable verbose timer output")
    parsed_args, unknown_args = parser.parse_known_args(sys.argv[1:])
    print(f"Verbose timer: {parsed_args.verbose_timer}")

    perception_node = PerceptionNode(verbose_timer=parsed_args.verbose_timer)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(perception_node)
    try:
        executor.spin()
        perception_node.destroy_node()
        executor.shutdown()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, perception node is shut down")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
