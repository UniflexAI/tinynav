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
from math_utils import rot_from_two_vector, np2msg, np2tf, estimate_pose2
from math_utils import uf_init, uf_union, uf_all_sets_list
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
logger.setLevel(logging.DEBUG)


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
        self.trt_fusion_model = TRTFusionModel()

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

        self.Tcb = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=3000)
        
        self.imu_accel_sub = Subscriber(self, Imu, "/camera/camera/accel/sample", qos_profile=qos_profile)
        self.imu_gyro_sub = Subscriber(self, Imu, "/camera/camera/gyro/sample", qos_profile=qos_profile)
        self.imu_ts = ApproximateTimeSynchronizer([self.imu_accel_sub, self.imu_gyro_sub], queue_size=1000, slop=0.005)
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

    def accel_callback(self, msg):
        self.accel_readings.append(msg.linear_acceleration)
        if len(self.accel_readings) >= 10 and self.T_body_last is None:
            accel_data = np.array([(a.x, a.y, a.z) for a in self.accel_readings])
            gravity_cam = np.mean(accel_data, axis=0)
            gravity_cam /= np.linalg.norm(gravity_cam)
            gravity_world = np.array([0.0, 0.0, 1.0])

            self.T_body_last = np.eye(4)
            self.T_body_last[:3, :3] = rot_from_two_vector(gravity_cam, gravity_world)

            self.get_logger().info("Initial pose set from accelerometer data.")
            self.get_logger().info(f"Initial rotation matrix:\n{self.T_body_last}")
            self.destroy_subscription(self.accel_sub)

    def imu_callback(self, accel_msg, gyro_msg):
        current_timestamp = stamp2second(accel_msg.header.stamp)

        # if the timestamp jump is too large, it means the IMU is not working properly
        if self.imu_last_received_timestamp is not None and current_timestamp - self.imu_last_received_timestamp > 0.1:
            delta_timestamp = current_timestamp - self.imu_last_received_timestamp
            self.get_logger().warning(f"IMU timestamp jump {delta_timestamp} s is too large, it means the IMU is not working properly")
        self.imu_last_received_timestamp = current_timestamp

        accel_data = np.array([[accel_msg.linear_acceleration.x], [accel_msg.linear_acceleration.y], [accel_msg.linear_acceleration.z]])
        gyro_data = np.array([[gyro_msg.angular_velocity.x], [gyro_msg.angular_velocity.y], [gyro_msg.angular_velocity.z]])
        self.imu_measurements.append([current_timestamp, accel_data.flatten(), gyro_data.flatten()])

    def images_callback(self, left_msg, right_msg):
        current_timestamp = stamp2second(left_msg.header.stamp)
        if current_timestamp - self.last_processed_timestamp < 0.1333:
            return
        self.last_processed_timestamp = current_timestamp
        with Timer(name="Perception Loop", text="[{name}] Elapsed time: {milliseconds:.0f} ms\n\n", logger=self.logger.info):
            asyncio.run(self.process(left_msg, right_msg))

    async def process(self, left_msg, right_msg):
        if self.K is None or self.T_body_last is None:
            return
        self.process_cnt += 1
        left_img = self.bridge.imgmsg_to_cv2(left_msg, "mono8")
        right_img = self.bridge.imgmsg_to_cv2(right_msg, "mono8")
        current_timestamp = stamp2second(left_msg.header.stamp)
        if len(self.keyframe_queue) == 0: # first frame
            disparity, depth = await asyncio.create_task(self.stereo_engine.infer(left_img, right_img, np.array([[self.baseline]]), np.array([[self.K[0,0]]])))
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
            return


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

        # propagate IMU measurements
        while len(self.imu_measurements) > 0 and self.imu_measurements[0][0] <= current_timestamp:
            timestamp, accel, gyro = self.imu_measurements[0]
            dt = timestamp - self.keyframe_queue[-1].latest_imu_timestamp

            if timestamp < self.keyframe_queue[-1].latest_imu_timestamp:
                self.imu_measurements.popleft()
                print("should only happen at beginning")
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
            prev_keypoints = prev_left_extract_result["kpts"][0]  # (n, 2)
            current_keypoints = current_left_extract_result["kpts"][0]  # (n, 2)
            match_indices = match_result["match_indices"][0]
            idx_to_origial = range(len(prev_keypoints))
            valid_mask = match_indices != -1
            kpt_pre = prev_keypoints[valid_mask]
            kpt_cur = current_keypoints[match_indices[valid_mask]]
            idx_valid = np.array(idx_to_origial)[valid_mask]
            logging.debug(f"match cnt: {len(kpt_pre)}")
            state, T_kf_curr, _, _, _ = await estimate_pose2(
                kpt_pre,
                kpt_cur,
                idx_valid,
                depth,
                self.K
            )
            print("Estimated T_kf_curr:\n", T_kf_curr)

        # for new frame, we first add it as keyframe, if not, we pop it later
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

        with Timer(name="[ISAM Processing]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.info):
            with Timer(name="[adding imu]", text="[{name}] Elapsed time: {milliseconds:.03f} ms", logger=self.logger.debug):
                # we have new graph each time
                graph = gtsam.NonlinearFactorGraph()
                initial_estimate = gtsam.Values()
                # process previous keyframes' factors
                for i, keyframe in enumerate(self.keyframe_queue[-_N:]):
                    # per pose -- bias
                    initial_estimate.insert(B(i), gtsam.imuBias.ConstantBias())
                    graph.add(gtsam.PriorFactorConstantBias(B(i), gtsam.imuBias.ConstantBias(), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]))))

                    initial_estimate.insert(V(i), keyframe.velocity)
                    initial_estimate.insert(X(i), Matrix4x4ToGtsamPose3(keyframe.pose))
                    if i == 0:
                        ## per pose -- velocity
                        #graph.add(gtsam.PriorFactorVector(V(i), np.zeros(3), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2, 1e-2, 1e-2]))))

                        # per pose -- pose, could only be applied to the first keyframe
                        graph.add(gtsam.PriorFactorPose3(X(i), Matrix4x4ToGtsamPose3(keyframe.pose), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1]))))

                    # per pose -- preintegrated IMU factor, only between two keyframes
                    if i != len(self.keyframe_queue[-_N:]) - 1:
                        imu_factor = gtsam.CombinedImuFactor(X(i), V(i), X(i+1), V(i+1), B(i), B(i+1), keyframe.preintegrated_imu)
                        graph.add(imu_factor)
                    print(f"for frame {i} at {keyframe.timestamp}, added imufactor up to {keyframe.latest_imu_timestamp}")

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
            parent, rank = uf_init(len(self.keyframe_queue[-_N:]) * _M)

            self.logger.debug(f"Processing {len(self.keyframe_queue)} keyframes for data association.")
            
            with Timer(name="[cached result]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
                for i in range(max(0, len(self.keyframe_queue) - _N), len(self.keyframe_queue) - 1):
                    with Timer(name="[cached result[1/3]]", text="[{name}] Elapsed time: {milliseconds:.03f} ms", logger=self.logger.debug):
                        j = i + 1
                        kf_prev = self.keyframe_queue[i]
                        kf_curr = self.keyframe_queue[j]

                    print("timestamp prev: ", kf_prev.timestamp)
                    print("timestamp curr: ", kf_curr.timestamp)
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
                            kf_curr.image.shape)
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
                        state, _, _, _, inliers = await estimate_pose2(
                            kpt_pre,
                            kpt_cur,
                            idx_valid,
                            depth,
                            self.K
                        )

                        inlier_set = set(inliers)
                        for idx in range(len(match_indices)):
                            if idx not in inlier_set:
                                match_indices[idx] = -1

                    with Timer(name="[cached result[3/3]]", text="[{name}] Elapsed time: {milliseconds:.03f} ms", logger=self.logger.debug):
                        for k, match_idx in enumerate(match_indices):
                            if match_idx != -1:
                                idx_prev = i * _M + k
                                idx_curr = j * _M + match_idx
                                uf_union(idx_prev, idx_curr, parent, rank)

            with Timer(name="[found track]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
                tracks = [track for track in uf_all_sets_list(parent) if len(track) >= 2]
                self.logger.debug(f"Found {len(tracks)} tracks after data association.")

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
                print(f"Keyframe {i} pose updated:\n{T_i}, at timestamp {keyframe.timestamp}")
                print(f"Bias {i} updated:\n{result.atConstantBias(B(i))}")
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
        self.logger.debug(f"estimate_pose cache info: {estimate_pose2.cache_info()}")

        with Timer(name="[Publish Odometry]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.logger.debug):
            self.T_body_last = result.atPose3(X(len(self.keyframe_queue) - 1)).matrix()
            # publish odometry
            self.odom_pub.publish(np2msg(self.T_body_last, left_msg.header.stamp, "world", "camera"))
            # publish TF
            self.tf_broadcaster.sendTransform(np2tf(self.T_body_last, left_msg.header.stamp, "world", "camera"))

            def keyframe_check(T_i, T_j):
                T_ij = np.linalg.inv(T_i) @ T_j
                t_diff = np.linalg.norm(T_ij[:3, 3])
                cos_theta = (np.trace(T_ij[:3, :3]) - 1) / 2
                r_diff = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
                return t_diff > _KEYFRAME_MIN_DISTANCE or r_diff > _KEYFRAME_MIN_ROTATE_DEGREE


            if keyframe_check(self.keyframe_queue[-2].pose, self.keyframe_queue[-1].pose):
                self.keyframe_pose_pub.publish(np2msg(self.keyframe_queue[-1].pose, left_msg.header.stamp, "world", "camera"))
                self.keyframe_image_pub.publish(left_msg)
                self.keyframe_depth_pub.publish(depth_msg)
            else:
                self.keyframe_queue.pop()



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
    executor.spin()
    perception_node.destroy_node()
    executor.shutdown()
    #try:
    #    executor.spin()
    #    perception_node.destroy_node()
    #    executor.shutdown()
    #except KeyboardInterrupt:
    #    logging.info("Keyboard interrupt received, perception node is shut down")
    #except Exception as e:
    #    logging.error(f"Error occurred: {e}")
    #finally:
    #    rclpy.shutdown()

if __name__ == "__main__":
    main()
