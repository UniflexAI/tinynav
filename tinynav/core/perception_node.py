import logging
import sys
from collections import deque
import cv2
from message_filters import TimeSynchronizer, Subscriber
import numpy as np
import rclpy
from codetiming import Timer
from cv_bridge import CvBridge
from models_trt import LightGlueTRT, SuperPointTRT
from stereo_engine import StereoEngine # noqa: F401
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, CameraInfo, PointCloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy
from math_utils import rot_from_two_vector, np2msg, np2tf, msg2np
from tf2_ros import TransformBroadcaster
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pc2

_MIN_FEATURES = 20
_KEYFRAME_MIN_DISTANCE = 1 # uint: meter
_KEYFRAME_MIN_ROTATE_DEGREE = 5 # uint: degree

logger = logging.getLogger(__name__)

class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        # model
        self.superpoint = SuperPointTRT()
        self.light_glue = LightGlueTRT()

        self.left0_extract_result = None
        self.left1_extract_result = None
        self.frame_index = 0
        self.keyframe_data = {
            "timestamp": deque(maxlen=10),
            "node": deque(maxlen=10),
        }

        # stereo match: sgbm or foundation_stereo
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        #self.stereo_engine = StereoEngine()
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
        self.relocalization_sub = self.create_subscription(Odometry, "/map/relocalization", self.relocalization_callback, 10)
        self.Tmap_odom = np.eye(4)
        self.odom_pub = self.create_publisher(Odometry, "/slam/odometry", 10)
        self.disparity_pub = self.create_publisher(Image, "/slam/disparity", 10)
        self.disparity_pub_vis = self.create_publisher(Image, '/slam/disparity_vis', 10)
        self.dispairty_pointcloud_pub = self.create_publisher(PointCloud2, '/slam/disparity_pointcloud', 10)
        self.keyframe_pose_pub = self.create_publisher(Odometry, "/slam/keyframe_odom", 10)
        self.keyframe_image_pub = self.create_publisher(Image, "/slam/keyframe_image", 10)
        self.keyframe_disparity_pub = self.create_publisher(Image, "/slam/keyframe_disparity", 10)

        self.accel_readings = []
        self.last_processed_timestamp = 0.0

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
        if self.T_last is not None:
            return
        self.accel_readings.append(msg.linear_acceleration)
        if len(self.accel_readings) >= 10:
            self.T_last = np.eye(4)

            accel_data = np.array([(a.x, a.y, a.z) for a in self.accel_readings])
            gravity_cam = np.mean(accel_data, axis=0)
            gravity_cam /= np.linalg.norm(gravity_cam)

            gravity_world = np.array([0.0, 0.0, 1.0])

            self.T_last[:3, :3] = rot_from_two_vector(gravity_cam, gravity_world)
            self.get_logger().info("Initial pose set from accelerometer data.")
            self.get_logger().info(f"Initial rotation matrix:\n{self.T_last}")
            self.destroy_subscription(self.accel_sub)

    def relocalization_callback(self, msg):
        try:
            index = self.keyframe_data["timestamp"].index(msg.header.stamp)
            Todom_i = self.keyframe_data["node"][index]
            self.Tmap_odom = msg2np(msg) @ np.linalg.inv(Todom_i)
        except Exception:
            logging.error(f"keyframe is delay too much {self.keyframe_data['timestamp'][0]} > {msg.header.stamp}")

    def images_callback(self, left_msg, right_msg):
        current_timestamp = left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9
        if current_timestamp - self.last_processed_timestamp < 0.2:
            return
        
        self.last_processed_timestamp = current_timestamp
        
        self.timestamp = left_msg.header.stamp
        self.process(left_msg, right_msg)

    @Timer(name="Perception Loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms")
    def process(self, left_msg, right_msg):
        if self.K is None or self.T_last is None or self.image_shape is None:
            return
        
        left_img = self.bridge.imgmsg_to_cv2(left_msg, "mono8")
        right_img = self.bridge.imgmsg_to_cv2(right_msg, "mono8")

        with Timer(text="[extract features + match] Elapsed time: {milliseconds:.0f} ms", logger=logger.info):
            if self.left0_extract_result is None:
                extractor_result = self.superpoint.infer(left_img)
                if len(extractor_result["kpts"][0]) < _MIN_FEATURES:
                    return
                self.left0_extract_result = extractor_result
                return

            self.left1_extract_result = self.superpoint.infer(left_img)
            left1_keypoints = self.left1_extract_result["kpts"][0]  # (n, 2)
            if len(left1_keypoints) < _MIN_FEATURES:
                return
            match_result = self.light_glue.infer(
                self.left0_extract_result["kpts"],
                self.left1_extract_result["kpts"],
                self.left0_extract_result["descps"],
                self.left1_extract_result["descps"],
                self.image_shape,
                self.image_shape,
            )
            left0_keypoints = self.left0_extract_result["kpts"][0]  # (n, 2)
            left1_keypoints = self.left1_extract_result["kpts"][0]  # (n, 2)
            match_indices = match_result["match_indices"][0]
            valid_mask = match_indices != -1
            kpt_pre = left0_keypoints[valid_mask]
            kpt_cur = left1_keypoints[match_indices[valid_mask]]

            logging.info(f"left0_pts left1_pts, match cnt: {len(left0_keypoints)}, {len(left1_keypoints)}, {len(kpt_pre)}")
            self.left0_extract_result = self.left1_extract_result

        with Timer(text="[ComputeDisparity] Elapsed time: {milliseconds:.0f} ms", logger=logger.info):
            disparity = self.sgbm.compute(left_img, right_img).astype(np.float32) / 16.0
            disparity[disparity < 0] = 0

        # compute disparity by foundation_stereo
        #with Timer(name='stereo', text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=logger.info):
        #    outputs, _ = self.stereo_engine.infer(left_img, right_img)
        #    disparity = outputs[0]  # Already resized/scaled to original size

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

        with Timer(name='[Depth as Cloud', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            self.pub_disp_as_pointcloud(disparity, self.timestamp)

        with Timer(text="[ComputePose] Elapsed time: {milliseconds:.0f} ms", logger=logger.info):
            state, T_pre_curr = self.estimate_pose(kpt_pre, kpt_cur, disparity)

        if not state:
            return

        # final pose
        T_curr = self.T_last @ T_pre_curr
        self.T_last = T_curr

        # publish odometry 
        if self.Tmap_odom is None:
            self.odom_pub.publish(np2msg(T_curr, self.timestamp, "world", "camera"))
        else:
            T_curr = self.Tmap_odom @ T_curr
            self.odom_pub.publish(np2msg(T_curr, self.timestamp, "world", "camera"))
        # publish TF
        self.tf_broadcaster.sendTransform(np2tf(T_curr, self.timestamp, "world", "camera"))

        # keyframe checking
        def keyframe_check(T_curr):
                if len(self.keyframe_data["node"]) == 0:
                    return True
                last_keyframe = self.keyframe_data["node"][-1]
                T_ij = np.linalg.inv(last_keyframe) @ T_curr
                t_diff = np.linalg.norm(T_ij[:3, 3])
                cos_theta = (np.trace(T_ij[:3, :3]) - 1) / 2
                r_diff = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
                return t_diff > _KEYFRAME_MIN_DISTANCE or r_diff > _KEYFRAME_MIN_ROTATE_DEGREE

        if keyframe_check(T_curr):
            self.keyframe_data["node"].append(T_curr)  # T_wi
            self.keyframe_data["timestamp"].append(left_msg.header.stamp)
            self.keyframe_pose_pub.publish(np2msg(T_curr, left_msg.header.stamp, "world", "camera"))
            self.keyframe_image_pub.publish(left_msg)
            self.keyframe_disparity_pub.publish(disparity_msg)



    def estimate_pose(self, kpts_prev, kpts_curr, disparity):
        points_3d, points_2d = [], []
        for pt_prev, pt_curr in zip(kpts_prev, kpts_curr):
            u, v = int(pt_curr[0]), int(pt_curr[1])
            if 0 <= v < disparity.shape[0] and 0 <= u < disparity.shape[1]:
                disp = disparity[v, u]
                if disp > 1:
                    Z = self.K[0, 0] * self.baseline / disp
                    X = (pt_curr[0] - self.K[0, 2]) * Z / self.K[0, 0]
                    Y = (pt_curr[1] - self.K[1, 2]) * Z / self.K[1, 1]
                    points_3d.append([X, Y, Z])
                    points_2d.append(pt_prev)
        if len(points_3d) < 6:
            return False, np.eye(4)
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)
        success, rvec, tvec, _ = cv2.solvePnPRansac(points_3d, points_2d, self.K, None)
        if not success:
            return False, np.eye(4)
        R_mat, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = tvec.ravel()
        return True, T

    def convert_to_odom_msg(self, T_curr, timestamp):
        """Convert the transformation matrix to an Odometry message."""
        odom_msg = Odometry()
        odom_msg.header.stamp = timestamp
        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = "camera"
        
        R_curr = T_curr[:3, :3]
        t_curr = T_curr[:3, 3]
        
        quat = R.from_matrix(R_curr).as_quat()
        odom_msg.pose.pose.position.x = t_curr[0]
        odom_msg.pose.pose.position.y = t_curr[1]
        odom_msg.pose.pose.position.z = t_curr[2]
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]
        return odom_msg
    # ===publish utils functions===
    def pub_disp_as_pointcloud(self, disparity, timestamp):
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        points = []
        h, w = disparity.shape
        for v in range(0, h, 16):
            for u in range(0 , w, 16):
                d = disparity[v, u]
                if d > 1:
                    Z = fx * self.baseline / d
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    points.append((X, Y, Z))
        header = std_msgs.msg.Header(stamp=timestamp, frame_id="camera")
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.dispairty_pointcloud_pub.publish(pc2_msg)

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
