import argparse
import copy
from collections import deque

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

from tinynav.core.math_utils import np2msg, pose_msg2np, tf2np


class LooperBridgeNode(Node):
    def __init__(self, args):
        super().__init__("looper_bridge_node")
        self.args = args
        self.bridge = CvBridge()

        self.depth_frame_id = None
        self.cached_camera_info = None
        self.T_i_depth = None
        self.tf_edges = {}
        self.last_keyframe_pose = None
        self.last_odom_pose = None
        self.last_odom_time = None
        self.recent_odom_stamp_ns = deque(maxlen=512)
        self.recent_odom_stamp_ns_set = set()
        self._missing_input_counter = 0

        self.sensor_qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        self.tf_static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.camera_info_sub = self.create_subscription(CameraInfo, "/camera/camera/infra1/camera_info", self.camera_info_callback, self.sensor_qos)
        self.tf_static_sub = self.create_subscription(TFMessage, "/tf_static", self.tf_callback, self.tf_static_qos)
        self.pose_forward_sub = self.create_subscription(
            PoseStamped, "/insight/vio_pose", self.pose_callback, self.sensor_qos
        )

        self.depth_sub = message_filters.Subscriber(self, Image, "/camera/camera/depth/image_rect_raw", qos_profile=self.sensor_qos)
        self.pose_sub = message_filters.Subscriber(self, PoseStamped, "/insight/vio_pose")
        self.image_sub = message_filters.Subscriber(self, Image, "/camera/camera/infra1/image_rect_raw", qos_profile=self.sensor_qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.pose_sub, self.image_sub], queue_size=20, slop=0.08
        )
        self.sync.registerCallback(self.sync_callback)

        self.odom_pub = self.create_publisher(Odometry, "/slam/odometry", 10)
        self.depth_pub = self.create_publisher(Image, "/slam/depth", 10)
        self.disparity_pub_vis = self.create_publisher(Image, "/slam/disparity_vis", 10)
        self.slam_camera_info_pub = self.create_publisher(CameraInfo, "/slam/camera_info", 10)
        self.camera_info_alias_pub = self.create_publisher(
            CameraInfo, "/camera/camera/infra2/camera_info", 10
        )
        self.keyframe_pose_pub = self.create_publisher(Odometry, "/slam/keyframe_odom", 10)
        self.keyframe_image_pub = self.create_publisher(Image, "/slam/keyframe_image", 10)
        self.keyframe_depth_pub = self.create_publisher(Image, "/slam/keyframe_depth", 10)

        self.get_logger().info(
            "Bridging /insight/vio_pose + /camera/camera/depth/image_rect_raw + /camera/camera/infra1/image_rect_raw into TinyNav /slam topics."
        )

    def camera_info_callback(self, msg: CameraInfo):
        self.cached_camera_info = msg
        if msg.header.frame_id:
            self.depth_frame_id = msg.header.frame_id
        self.try_update_depth_transform()
        self.get_logger().info(
            f"Received camera info from /camera/camera/infra1/camera_info with frame {msg.header.frame_id}.",
            once=True,
        )

    def tf_callback(self, msg: TFMessage):
        for transform in msg.transforms:
            frame_id, child_frame_id, T = tf2np(transform)
            self.store_transform(frame_id, child_frame_id, T)
        self.try_update_depth_transform()
        self.get_logger().info("Received TF_STATIC for Looper bridge.", once=True)

    def store_transform(self, parent_frame: str, child_frame: str, T: np.ndarray):
        self.tf_edges.setdefault(parent_frame, {})[child_frame] = T.astype(np.float32, copy=False)
        self.tf_edges.setdefault(child_frame, {})[parent_frame] = np.linalg.inv(T).astype(
            np.float32, copy=False
        )

    def lookup_transform(self, source_frame: str, target_frame: str):
        if source_frame == target_frame:
            return np.eye(4, dtype=np.float32)
        if source_frame not in self.tf_edges or target_frame not in self.tf_edges:
            return None

        queue = deque([(source_frame, np.eye(4, dtype=np.float32))])
        visited = {source_frame}
        while queue:
            frame, T_source_frame = queue.popleft()
            for next_frame, T_frame_next in self.tf_edges.get(frame, {}).items():
                if next_frame in visited:
                    continue
                T_source_next = T_source_frame @ T_frame_next
                if next_frame == target_frame:
                    return T_source_next.astype(np.float32, copy=False)
                visited.add(next_frame)
                queue.append((next_frame, T_source_next))
        return None

    def try_update_depth_transform(self):
        if self.depth_frame_id is None:
            return
        T_i_depth = self.lookup_transform("imu", self.depth_frame_id)
        if T_i_depth is not None:
            self.T_i_depth = T_i_depth
            self.get_logger().info(
                f"Resolved imu -> {self.depth_frame_id} transform for Looper bridge.",
                once=True,
            )

    def log_missing_inputs(self):
        self._missing_input_counter += 1
        if self._missing_input_counter % 30 != 1:
            return
        missing = []
        if self.cached_camera_info is None:
            missing.append("/camera/camera/infra1/camera_info")
        if self.T_i_depth is None:
            missing.append(f"imu->{self.depth_frame_id or 'depth'} TF")
        self.get_logger().info(f"Waiting for Looper bridge inputs: {', '.join(missing)}")

    def should_add_keyframe(self, T_world_camera: np.ndarray) -> bool:
        if self.last_keyframe_pose is None:
            return True
        translation = np.linalg.norm(
            T_world_camera[:3, 3] - self.last_keyframe_pose[:3, 3]
        )
        relative_rotation = self.last_keyframe_pose[:3, :3].T @ T_world_camera[:3, :3]
        rotation_angle = np.arccos(
            np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        )
        return (
            translation >= self.args.keyframe_translation
            or rotation_angle >= np.deg2rad(self.args.keyframe_rotation_deg)
        )

    @staticmethod
    def stamp_to_ns(stamp) -> int:
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def build_odom(self, T_world_camera: np.ndarray, stamp) -> Odometry:
        velocity = None
        current_time = stamp.sec + stamp.nanosec * 1e-9
        if self.last_odom_pose is not None and self.last_odom_time is not None:
            dt = current_time - self.last_odom_time
            if dt > 1e-3:
                velocity = (T_world_camera[:3, 3] - self.last_odom_pose[:3, 3]) / dt

        odom_msg = np2msg(
            T_world_camera,
            stamp,
            "world",
            "camera",
            velocity=velocity,
        )
        self.last_odom_pose = T_world_camera.copy()
        self.last_odom_time = current_time
        return odom_msg

    def publish_odom_if_new(self, T_world_camera: np.ndarray, stamp) -> Odometry | None:
        stamp_ns = self.stamp_to_ns(stamp)
        if stamp_ns in self.recent_odom_stamp_ns_set:
            return None
        odom_msg = self.build_odom(T_world_camera, stamp)
        self.odom_pub.publish(odom_msg)
        if len(self.recent_odom_stamp_ns) == self.recent_odom_stamp_ns.maxlen:
            oldest_stamp_ns = self.recent_odom_stamp_ns.popleft()
            self.recent_odom_stamp_ns_set.discard(oldest_stamp_ns)
        self.recent_odom_stamp_ns.append(stamp_ns)
        self.recent_odom_stamp_ns_set.add(stamp_ns)
        return odom_msg

    def pose_callback(self, pose_msg: PoseStamped):
        if self.T_i_depth is None:
            self.log_missing_inputs()
            return
        T_world_imu = pose_msg2np(pose_msg)
        T_world_camera = T_world_imu @ self.T_i_depth
        self.publish_odom_if_new(T_world_camera, pose_msg.header.stamp)

    def decode_depth_meters(self, depth_msg: Image) -> np.ndarray:
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = np.asarray(depth)
        if depth_msg.encoding in ("mono16", "16UC1"):
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)
        return depth

    def build_depth_msg(self, depth_m: np.ndarray, stamp) -> Image:
        depth_out = self.bridge.cv2_to_imgmsg(depth_m, encoding="32FC1")
        depth_out.header.stamp = stamp
        depth_out.header.frame_id = "camera"
        return depth_out

    def build_disparity_vis(self, depth_m: np.ndarray, stamp) -> Image:
        depth = np.asarray(depth_m, dtype=np.float32)

        valid = np.isfinite(depth) & (depth > 1e-3)
        disparity_u8 = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(valid):
            inv_depth = np.zeros(depth.shape, dtype=np.float32)
            inv_depth[valid] = 1.0 / depth[valid]
            disp_min = float(np.min(inv_depth[valid]))
            disp_max = float(np.max(inv_depth[valid]))
            if disp_max > disp_min:
                disparity_u8[valid] = np.clip(
                    255.0 * (inv_depth[valid] - disp_min) / (disp_max - disp_min),
                    0.0,
                    255.0,
                ).astype(np.uint8)
            else:
                disparity_u8[valid] = 255

        disp_color = cv2.applyColorMap(disparity_u8, cv2.COLORMAP_PLASMA)
        disp_color[~valid] = 0
        disp_color_msg = self.bridge.cv2_to_imgmsg(disp_color, encoding="bgr8")
        disp_color_msg.header.stamp = stamp
        disp_color_msg.header.frame_id = "camera"
        return disp_color_msg

    def sync_callback(self, depth_msg: Image, pose_msg: PoseStamped, image_msg: Image):
        if self.cached_camera_info is None or self.T_i_depth is None:
            self.log_missing_inputs()
            return

        T_world_imu = pose_msg2np(pose_msg)
        T_world_camera = T_world_imu @ self.T_i_depth
        stamp = pose_msg.header.stamp

        odom_msg = self.publish_odom_if_new(T_world_camera, stamp)
        if odom_msg is None:
            odom_msg = np2msg(T_world_camera, stamp, "world", "camera")
        depth_m = self.decode_depth_meters(depth_msg)
        depth_out = self.build_depth_msg(depth_m, stamp)
        disparity_vis_msg = self.build_disparity_vis(depth_m, stamp)

        image_out = copy.deepcopy(image_msg)
        image_out.header.stamp = stamp
        image_out.header.frame_id = "camera"

        camera_info_out = copy.deepcopy(self.cached_camera_info)
        camera_info_out.header.stamp = stamp
        camera_info_out.header.frame_id = "camera"

        self.depth_pub.publish(depth_out)
        self.disparity_pub_vis.publish(disparity_vis_msg)
        self.slam_camera_info_pub.publish(camera_info_out)
        self.camera_info_alias_pub.publish(camera_info_out)

        if self.should_add_keyframe(T_world_camera):
            self.keyframe_pose_pub.publish(odom_msg)
            self.keyframe_image_pub.publish(image_out)
            self.keyframe_depth_pub.publish(depth_out)
            self.last_keyframe_pose = T_world_camera.copy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframe-translation", type=float, default=0.03)
    parser.add_argument("--keyframe-rotation-deg", type=float, default=1.0)
    return parser.parse_args()


def main(args=None):
    rclpy.init(args=args)
    node = LooperBridgeNode(parse_args())
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
