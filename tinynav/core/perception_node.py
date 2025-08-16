import logging
import sys
import cv2
from message_filters import TimeSynchronizer, Subscriber
import numpy as np
import rclpy
from codetiming import Timer
from cv_bridge import CvBridge
from models_trt import LightGlueTRT, SuperPointTRT, StereoEngineTRT
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, CameraInfo, PointCloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy
from math_utils import rot_from_two_vector, np2msg, np2tf, estimate_pose, disparity_to_pointcloud
from tf2_ros import TransformBroadcaster
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pc2

import asyncio

_MIN_FEATURES = 20
_KEYFRAME_MIN_DISTANCE = 0.01 # uint: meter
_KEYFRAME_MIN_ROTATE_DEGREE = 5 # uint: degree

logger = logging.getLogger(__name__)

class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        # model
        self.superpoint = SuperPointTRT()
        self.light_glue = LightGlueTRT()

        self.last_keyframe_img = None
        self.has_first_keyframe = False

        self.stereo_engine = StereoEngineTRT()
        # intrinsic
        self.baseline = None
        self.K = None
        self.image_shape = None
        self.T_last = None
        self.Tcb = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.accel_sub = self.create_subscription(Imu, "/camera/camera/accel/sample", self.accel_callback, qos_profile)
        self.camerainfo_sub = self.create_subscription(CameraInfo, "/camera/camera/infra2/camera_info", self.info_callback, 10)
        self.left_sub = Subscriber(self, Image, "/camera/camera/infra1/image_rect_raw")
        self.right_sub = Subscriber(self, Image, "/camera/camera/infra2/image_rect_raw")
        self.ts = TimeSynchronizer([self.left_sub, self.right_sub], queue_size=1)
        self.ts.registerCallback(self.images_callback)
        self.odom_pub = self.create_publisher(Odometry, "/slam/odometry", 10)
        self.disparity_pub = self.create_publisher(Image, "/slam/disparity", 10)
        self.disparity_pub_vis = self.create_publisher(Image, '/slam/disparity_vis', 10)
        self.dispairty_pointcloud_pub = self.create_publisher(PointCloud2, '/slam/disparity_pointcloud', 10)
        self.keyframe_pose_pub = self.create_publisher(Odometry, "/slam/keyframe_odom", 10)
        self.keyframe_image_pub = self.create_publisher(Image, "/slam/keyframe_image", 10)
        self.keyframe_disparity_pub = self.create_publisher(Image, "/slam/keyframe_disparity", 10)

        self.accel_readings = []
        self.last_processed_timestamp = 0.0

        # 200Hz IMU
        self.gravity_in_camera_frame = None
        self.is_static = False

    def info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            fx = self.K[0, 0]
            Tx = msg.p[3]  # From the right camera's projection matrix
            self.baseline = -Tx / fx
            self.image_shape = np.array([msg.width, msg.height], dtype=np.int64)
            self.get_logger().info(f"Camera intrinsics and baseline received. Baseline: {self.baseline:.4f}m")
            self.destroy_subscription(self.camerainfo_sub)

    def accel_callback(self, msg):
        self.accel_readings.append(msg.linear_acceleration)
        if len(self.accel_readings) >= 10 and self.T_last is None:
            self.T_last = np.eye(4)

            accel_data = np.array([(a.x, a.y, a.z) for a in self.accel_readings])
            gravity_cam = np.mean(accel_data, axis=0)
            gravity_cam /= np.linalg.norm(gravity_cam)

            gravity_world = np.array([0.0, 0.0, 1.0])

            self.T_last[:3, :3] = rot_from_two_vector(gravity_cam, gravity_world)
            self.get_logger().info("Initial pose set from accelerometer data.")
            self.get_logger().info(f"Initial rotation matrix:\n{self.T_last}")
            #self.destroy_subscription(self.accel_sub)
            self.gravity_in_camera_frame = gravity_cam
        
        if len(self.accel_readings) > 200:
            self.accel_readings.pop(0)
            accel_data = np.array([(a.x, a.y, a.z) for a in self.accel_readings])
            norms = np.linalg.norm(accel_data, axis=1)
            norms_std = np.std(norms)
            if norms_std > 0.3:
                #self.get_logger().info(f"Norms std: {norms_std}")
                self.is_static = False
            else:
                if not self.is_static:
                    self.gravity_in_camera_frame = np.mean(accel_data, axis=0)
                self.is_static = True


        


    def images_callback(self, left_msg, right_msg):
        current_timestamp = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9
        if current_timestamp - self.last_processed_timestamp < 0.1:
            return
        
        self.last_processed_timestamp = current_timestamp
        
        self.timestamp = left_msg.header.stamp

        with Timer(name="Perception Loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms"):
            asyncio.run(self.process(left_msg, right_msg))

    async def process(self, left_msg, right_msg):
        if self.K is None or self.T_last is None or self.image_shape is None:
            return

        with Timer(name="[Model Inference]", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=logger.info):
            left_img = self.bridge.imgmsg_to_cv2(left_msg, "mono8")
            right_img = self.bridge.imgmsg_to_cv2(right_msg, "mono8")
            if self.last_keyframe_img is None:
                self.last_keyframe_img = left_img
                return

            stereo_task = asyncio.create_task(self.stereo_engine.infer(left_img, right_img))

            prev_left_extract_result = await self.superpoint.memorized_infer(self.last_keyframe_img)
            current_left_extract_result = await self.superpoint.memorized_infer(left_img)

            match_result = await self.light_glue.infer(
                    prev_left_extract_result["kpts"],
                    current_left_extract_result["kpts"],
                    prev_left_extract_result["descps"],
                    current_left_extract_result["descps"],
                    self.image_shape,
                    self.image_shape)

            prev_keypoints = prev_left_extract_result["kpts"][0]  # (n, 2)
            current_keypoints = current_left_extract_result["kpts"][0]  # (n, 2)
            match_indices = match_result["match_indices"][0]
            valid_mask = match_indices != -1
            kpt_pre = prev_keypoints[valid_mask]
            kpt_cur = current_keypoints[match_indices[valid_mask]]

            logging.info(f"prev_pts current_pts, match cnt: {len(prev_keypoints)}, {len(current_keypoints)}, {len(kpt_pre)}")
            disparity = await stereo_task
            disparity[disparity < 0] = 0

        # publish dispairty   
        with Timer(text="[ComputeDisparity] Elapsed time: {milliseconds:.0f} ms", logger=logger.info):
            disparity_msg = self.bridge.cv2_to_imgmsg(disparity, encoding="32FC1")
            disparity_msg.header = left_msg.header
            self.disparity_pub.publish(disparity_msg)

        with Timer(text="[Depth as Color] Elapsed time: {milliseconds:.0f} ms", logger=logger.info):
            disp_vis = disparity.copy().astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_vis * 4, cv2.COLORMAP_PLASMA)
            disp_color_msg = self.bridge.cv2_to_imgmsg(disp_color, encoding='bgr8')
            disp_color_msg.header = left_msg.header
            self.disparity_pub_vis.publish(disp_color_msg)

        with Timer(name='[Depth as Cloud', text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=logger.info):
            self.pub_disp_as_pointcloud(disparity, self.timestamp)

        with Timer(text="[ComputePose] Elapsed time: {milliseconds:.0f} ms", logger=logger.info):
            state, T_pre_curr, _, _, _ = estimate_pose(kpt_pre, kpt_cur, disparity, self.K, self.baseline)

        if not state:
            return

        # final pose
        T_curr = self.T_last @ T_pre_curr

        self.T_last, T_curr = self.gravity_correction(self.T_last, T_curr, T_pre_curr)

        # publish odometry 
        self.odom_pub.publish(np2msg(T_curr, self.timestamp, "world", "camera"))
        # publish TF
        self.tf_broadcaster.sendTransform(np2tf(T_curr, self.timestamp, "world", "camera"))

        # keyframe checking
        def keyframe_check(T_ij):
           t_diff = np.linalg.norm(T_ij[:3, 3])
           cos_theta = (np.trace(T_ij[:3, :3]) - 1) / 2
           r_diff = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
           return t_diff > _KEYFRAME_MIN_DISTANCE or r_diff > _KEYFRAME_MIN_ROTATE_DEGREE

        if not self.has_first_keyframe or keyframe_check(T_pre_curr):
            self.last_keyframe_img = left_img
            self.T_last = T_curr
            self.has_first_keyframe = True
            self.keyframe_pose_pub.publish(np2msg(T_curr, left_msg.header.stamp, "world", "camera"))
            self.keyframe_image_pub.publish(left_msg)
            self.keyframe_disparity_pub.publish(disparity_msg)

    # ===publish utils functions===
    def pub_disp_as_pointcloud(self, disparity, timestamp):
        points = disparity_to_pointcloud(disparity, self.K, self.baseline)
        header = std_msgs.msg.Header(stamp=timestamp, frame_id="camera")
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.dispairty_pointcloud_pub.publish(pc2_msg)

    def gravity_correction(self, T_last, T_curr, T_pre_curr):
        if self.is_static:
             target_gravity_world = np.array([0.0, 0.0, 1.0])
             gravity_world = T_curr[:3, :3] @ self.gravity_in_camera_frame
             gravity_world /= np.linalg.norm(gravity_world)
             diff_angle = np.arccos((target_gravity_world.T @ gravity_world).item())
             diff_angle = np.degrees(diff_angle)
             if diff_angle > 2:
                 T_rot_corrector = rot_from_two_vector(self.gravity_in_camera_frame, np.linalg.inv(T_curr[:3, :3]) @ target_gravity_world)
                 T_curr[:3, :3] = T_curr[:3, :3] @ T_rot_corrector
                 T_last = T_curr @ np.linalg.inv(T_pre_curr)
                 self.get_logger().info(f"diff_angle: {diff_angle}, T_rot_corrector: {T_rot_corrector}, self.gravity_in_camera_frame: {self.gravity_in_camera_frame}")
        return T_last, T_curr
        

def main(args=None):
    rclpy.init(args=args)
    perception_node = PerceptionNode()
    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        logging.info("exit")
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("odom.log")],
    )
    main()
