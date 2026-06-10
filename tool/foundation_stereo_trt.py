import argparse
import asyncio
import logging
import math
import platform
import sys

import cv2
import numpy as np
from codetiming import Timer
from tinynav.core.models_trt import FoundationStereoTRT

try:
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import TransformStamped
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
    from tf2_ros import TransformBroadcaster

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = object

class FoundationStereoDepthNode(Node):
    def __init__(
        self,
        engine_path=f"/tinynav/tinynav/models/foundation_stereo_11-33-40_256x320_4_{platform.machine()}.plan",
        left_topic="/camera/camera/infra1/image_rect_raw",
        right_topic="/camera/camera/infra2/image_rect_raw",
        camera_info_topic="/camera/camera/infra2/camera_info",
        keyframe_depth_topic="/keyframe/depth",
        pointcloud_topic="/keyframe/pointcloud",
        frame_id="camera",
        world_frame_id="world",
        publish_pointcloud=False,
        min_depth=0.05,
        max_depth=20.0,
        queue_size=10,
        slop=0.02,
    ):
        super().__init__("foundation_stereo_depth_node")
        self.logger = logging.getLogger(__name__)
        self.bridge = CvBridge()
        self.stereo_engine = FoundationStereoTRT(engine_path=engine_path)

        self.baseline = None
        self.focal_length = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_info_msg = None
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.publish_pointcloud = bool(publish_pointcloud)
        self.frame_id = frame_id
        self.world_frame_id = world_frame_id
        self.tf_broadcaster = TransformBroadcaster(self)

        self.camerainfo_sub = self.create_subscription(CameraInfo, camera_info_topic, self.info_callback, 10)
        self.left_sub = Subscriber(self, Image, left_topic)
        self.right_sub = Subscriber(self, Image, right_topic)
        self.ts = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=queue_size, slop=slop)
        self.ts.registerCallback(self.images_callback)

        self.keyframe_depth_pub = self.create_publisher(Image, keyframe_depth_topic, 10)
        self.pointcloud_pub = None
        if self.publish_pointcloud:
            self.pointcloud_pub = self.create_publisher(PointCloud2, pointcloud_topic, 10)
        self.logger.info(
            "FoundationStereoDepthNode initialized. left=%s right=%s info=%s depth=%s pointcloud=%s enabled=%s",
            left_topic,
            right_topic,
            camera_info_topic,
            keyframe_depth_topic,
            pointcloud_topic,
            self.publish_pointcloud,
        )

    def _publish_world_to_foundation_tf(self, stamp):
        # Static orientation offset around X so world Z points upward in visualization.
        qx = math.sin(math.radians(-90.0) * 0.5)
        qw = math.cos(math.radians(-90.0) * 0.5)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.world_frame_id
        tf_msg.child_frame_id = self.frame_id
        tf_msg.transform.translation.x = 0.0
        tf_msg.transform.translation.y = 0.0
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = 0.0
        tf_msg.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf_msg)

    def info_callback(self, msg: CameraInfo):
        if self.focal_length is not None:
            return
        k = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        fx = float(k[0, 0])
        fy = float(k[1, 1])
        cx = float(k[0, 2])
        cy = float(k[1, 2])
        tx = float(msg.p[3])
        baseline = -tx / fx
        if baseline <= 0.0:
            self.get_logger().warning(f"Baseline from CameraInfo is non-positive: {baseline:.6f}")
        self.focal_length = fx
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.baseline = baseline
        self.camera_info_msg = msg
        self.get_logger().info(
            f"Camera intrinsics received. fx={self.fx:.3f} fy={self.fy:.3f} "
            f"cx={self.cx:.3f} cy={self.cy:.3f} baseline={self.baseline:.5f} m"
        )
        self.destroy_subscription(self.camerainfo_sub)

    def _build_colorized_pointcloud(self, depth: np.ndarray, left_img: np.ndarray, stamp) -> PointCloud2:
        h, w = depth.shape
        valid = np.isfinite(depth) & (depth > self.min_depth) & (depth < self.max_depth)
        ys, xs = np.where(valid)
        z = depth[ys, xs].astype(np.float32)

        x = ((xs.astype(np.float32) - self.cx) * z) / self.fx
        y = ((ys.astype(np.float32) - self.cy) * z) / self.fy

        color_rgb = cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB)
        r = color_rgb[ys, xs, 0].astype(np.uint32)
        g = color_rgb[ys, xs, 1].astype(np.uint32)
        b = color_rgb[ys, xs, 2].astype(np.uint32)
        rgb_u32 = (r << 16) | (g << 8) | b
        rgb_f32 = rgb_u32.view(np.float32)

        cloud = np.empty(
            z.shape[0],
            dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)],
        )
        cloud["x"] = x
        cloud["y"] = y
        cloud["z"] = z
        cloud["rgb"] = rgb_f32

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = 1
        msg.width = int(cloud.shape[0])
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False
        msg.data = cloud.tobytes()
        return msg

    def images_callback(self, left_msg: Image, right_msg: Image):
        if self.focal_length is None or self.baseline is None:
            self.get_logger().warning("Skipping stereo frame: waiting for camera_info.")
            return

        left_img = self.bridge.imgmsg_to_cv2(left_msg, "mono8")
        right_img = self.bridge.imgmsg_to_cv2(right_msg, "mono8")

        with Timer(
            name="FoundationStereo Inference",
            text="[{name}] Elapsed time: {milliseconds:.0f} ms",
            logger=self.logger.info,
        ):
            _, depth = asyncio.run(
                self.stereo_engine.infer(
                    left_img,
                    right_img,
                    np.array([[self.baseline]], dtype=np.float32),
                    np.array([[self.focal_length]], dtype=np.float32),
                )
            )

        with Timer(
            name="Publish Keyframe Depth",
            text="[{name}] Elapsed time: {milliseconds:.0f} ms",
            logger=self.logger.info,
        ):
            depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding="32FC1")
            depth_msg.header.stamp = left_msg.header.stamp
            depth_msg.header.frame_id = self.frame_id
            self.keyframe_depth_pub.publish(depth_msg)
            #self._publish_world_to_foundation_tf(left_msg.header.stamp)

        if self.publish_pointcloud:
            with Timer(
                name="Build+Publish Keyframe PointCloud",
                text="[{name}] Elapsed time: {milliseconds:.0f} ms",
                logger=self.logger.info,
            ):
                cloud_msg = self._build_colorized_pointcloud(depth, left_img, left_msg.header.stamp)
                self.pointcloud_pub.publish(cloud_msg)


def main(args=None):
    if not ROS_AVAILABLE:
        raise RuntimeError("ROS dependencies are unavailable. Please run in the TinyNav ROS2 environment.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine_path",
        type=str,
        default=f"/tinynav/tinynav/models/foundation_stereo_11-33-40_256x320_4_{platform.machine()}.plan",
    )
    parser.add_argument("--left_topic", type=str, default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--right_topic", type=str, default="/camera/camera/infra2/image_rect_raw")
    parser.add_argument("--camera_info_topic", type=str, default="/camera/camera/infra2/camera_info")
    parser.add_argument("--keyframe_depth_topic", type=str, default="/keyframe/depth")
    parser.add_argument("--pointcloud_topic", type=str, default="/keyframe/pointcloud")
    parser.add_argument("--frame_id", type=str, default="camera")
    parser.add_argument("--world_frame_id", type=str, default="world")
    parser.add_argument("--publish_pointcloud", action="store_true")
    parser.add_argument("--min_depth", type=float, default=0.05)
    parser.add_argument("--max_depth", type=float, default=20.0)
    parser.add_argument("--sync_queue_size", type=int, default=10)
    parser.add_argument("--sync_slop", type=float, default=0.02)
    parser.add_argument("--log_level", type=str, default="INFO")
    parsed_args, _ = parser.parse_known_args(args if args is not None else sys.argv[1:])

    logging.basicConfig(
        level=getattr(logging, parsed_args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    rclpy.init(args=args)
    node = FoundationStereoDepthNode(
        engine_path=parsed_args.engine_path,
        left_topic=parsed_args.left_topic,
        right_topic=parsed_args.right_topic,
        camera_info_topic=parsed_args.camera_info_topic,
        keyframe_depth_topic=parsed_args.keyframe_depth_topic,
        pointcloud_topic=parsed_args.pointcloud_topic,
        frame_id=parsed_args.frame_id,
        world_frame_id=parsed_args.world_frame_id,
        publish_pointcloud=parsed_args.publish_pointcloud,
        min_depth=parsed_args.min_depth,
        max_depth=parsed_args.max_depth,
        queue_size=parsed_args.sync_queue_size,
        slop=parsed_args.sync_slop,
    )
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    import os

    left_path = "/tinynav/tests/data/looper/left.png"
    right_path = "/tinynav/tests/data/looper/right.png"
    run_local_infer = (
        len(sys.argv) > 1 and sys.argv[1] == "--local_infer"
    ) or (not ROS_AVAILABLE and os.path.exists(left_path) and os.path.exists(right_path))
    if run_local_infer:
        left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        model = FoundationStereoTRT()
        disp, depth = asyncio.run(
            model.infer(
                left,
                right,
                np.array([[0.056]], dtype=np.float32),
                np.array([[323.0]], dtype=np.float32),
            )
        )
        print("disp", disp.shape, disp.dtype, float(np.nanmean(disp)))
        print("depth", depth.shape, depth.dtype, float(np.nanmean(depth[np.isfinite(depth)])))
    else:
        main()
