import argparse
import sys
from collections import deque
from pathlib import Path

import message_filters
import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

from tinynav.core.math_utils import tf2np


def pose_msg_to_np(msg: PoseStamped) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    quat = [
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
    ]
    T[:3, :3] = R.from_quat(quat).as_matrix().astype(np.float32)
    T[:3, 3] = np.array(
        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
        dtype=np.float32,
    )
    return T


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3)
    rotation = T[:3, :3]
    translation = T[:3, 3]
    return points @ rotation.T + translation


def crop_mask(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    dist2 = np.sum((points - center[None, :]) ** 2, axis=1)
    return dist2 <= radius * radius


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0 or voxel_size <= 0.0:
        return points.reshape(-1, 3), colors.reshape(-1)
    coords = np.floor(points / voxel_size).astype(np.int32)
    coord_view = (
        np.ascontiguousarray(coords)
        .view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1])))
        .reshape(-1)
    )
    _, unique_idx = np.unique(coord_view, return_index=True)
    unique_idx.sort()
    return points[unique_idx], colors[unique_idx]


def depth_to_grayscale_cloud(depth, gray_image, K, u_grid, v_grid, pixel_step, max_depth):
    sampled_depth = depth[::pixel_step, ::pixel_step]
    sampled_gray = gray_image[::pixel_step, ::pixel_step]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid_mask = (sampled_depth > 0.0) & (sampled_depth <= max_depth)
    stats = {
        "sampled": int(sampled_depth.size),
        "valid_depth": int(np.count_nonzero(valid_mask)),
        "colored": int(np.count_nonzero(valid_mask)),
    }
    if not np.any(valid_mask):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.uint32),
            stats,
            [],
        )

    z = sampled_depth[valid_mask].astype(np.float32)
    u = u_grid[valid_mask]
    v = v_grid[valid_mask]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.column_stack((x, y, z)).astype(np.float32)

    intensity = sampled_gray[valid_mask].astype(np.uint32)
    colors = (intensity << 16) | (intensity << 8) | intensity
    sample_count = min(5, points.shape[0])
    if sample_count == 0:
        sample_points = []
    else:
        sample_idx = np.linspace(0, points.shape[0] - 1, sample_count, dtype=int)
        sample_points = [
            (
                float(u[idx]),
                float(v[idx]),
                float(z[idx]),
                int(intensity[idx]),
                float(points[idx, 0]),
                float(points[idx, 1]),
            )
            for idx in sample_idx
        ]
    return points, colors, stats, sample_points


def depth_to_color_cloud(
    depth, color_image, depth_K, color_K, T_depth_color, u_grid, v_grid, pixel_step, max_depth
):
    sampled_depth = depth[::pixel_step, ::pixel_step]
    fx, fy = depth_K[0, 0], depth_K[1, 1]
    cx, cy = depth_K[0, 2], depth_K[1, 2]

    valid_mask = (sampled_depth > 0.0) & (sampled_depth <= max_depth)
    stats = {
        "sampled": int(sampled_depth.size),
        "valid_depth": int(np.count_nonzero(valid_mask)),
        "colored": int(np.count_nonzero(valid_mask)),
    }
    if not np.any(valid_mask):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.uint32),
            stats,
            [],
        )

    z = sampled_depth[valid_mask].astype(np.float32)
    u = u_grid[valid_mask]
    v = v_grid[valid_mask]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.column_stack((x, y, z)).astype(np.float32)

    color_points = transform_points(points, T_depth_color)
    valid_color_depth = color_points[:, 2] > 1e-6
    if not np.any(valid_color_depth):
        stats["colored"] = 0
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.uint32),
            stats,
            [],
        )

    points = points[valid_color_depth]
    color_points = color_points[valid_color_depth]
    u = u[valid_color_depth]
    v = v[valid_color_depth]
    z = z[valid_color_depth]

    color_u = np.rint(
        color_K[0, 0] * color_points[:, 0] / color_points[:, 2] + color_K[0, 2]
    ).astype(np.int32)
    color_v = np.rint(
        color_K[1, 1] * color_points[:, 1] / color_points[:, 2] + color_K[1, 2]
    ).astype(np.int32)
    in_bounds = (
        (color_u >= 0)
        & (color_u < color_image.shape[1])
        & (color_v >= 0)
        & (color_v < color_image.shape[0])
    )
    stats["colored"] = int(np.count_nonzero(in_bounds))
    if not np.any(in_bounds):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.uint32),
            stats,
            [],
        )

    points = points[in_bounds]
    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    color_u = color_u[in_bounds]
    color_v = color_v[in_bounds]

    rgb = color_image[color_v, color_u].astype(np.uint32)
    colors = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    sample_count = min(5, points.shape[0])
    if sample_count == 0:
        sample_points = []
    else:
        sample_idx = np.linspace(0, points.shape[0] - 1, sample_count, dtype=int)
        sample_points = [
            (
                float(u[idx]),
                float(v[idx]),
                float(z[idx]),
                int(rgb[idx, 0]),
                int(rgb[idx, 1]),
                int(rgb[idx, 2]),
                float(points[idx, 0]),
                float(points[idx, 1]),
            )
            for idx in sample_idx
        ]
    return points, colors, stats, sample_points


class GlobalPointCloudPublisher(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("global_pointcloud_publisher")
        self.args = args
        self.bridge = CvBridge()
        self.K: np.ndarray | None = None
        self.color_K: np.ndarray | None = None
        self.T_i_depth: np.ndarray | None = None
        self.T_depth_color: np.ndarray | None = None
        self.depth_frame_id: str | None = None
        self.color_frame_id: str | None = None
        self.tf_edges: dict[str, dict[str, np.ndarray]] = {}
        self.global_cloud_buffer: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = (
            deque()
        )
        self.path_msg = Path()
        self.merged_cloud_cache = np.empty((0, 3), dtype=np.float32)
        self.merged_color_cache = np.empty((0,), dtype=np.uint32)
        self.sample_grid_key: tuple[int, int, int] | None = None
        self.sample_u_grid = np.empty((0, 0), dtype=np.float32)
        self.sample_v_grid = np.empty((0, 0), dtype=np.float32)
        self.last_keyframe_pose: np.ndarray | None = None
        self._missing_input_counter = 0
        self._published_once = False
        self._saw_depth = False
        self._saw_pose = False
        self._saw_image = False
        self._saw_tf = False
        self._saw_tf_static = False
        self._logged_depth_tf = False
        self._logged_color_tf = False
        self._sync_count = 0
        self._projection_debug_counter = 0
        self.image_topic = "/camera/camera/infra1/image_rect_raw" if args.image_mode == "grayscale" else args.color_image_topic
        self.sensor_qos = QoSProfile(
            depth=50, reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self.camera_info_callback,
            self.sensor_qos,
        )
        self.color_camera_info_sub = None
        if args.image_mode == "color":
            self.color_camera_info_sub = self.create_subscription(
                CameraInfo,
                args.color_camera_info_topic,
                self.color_camera_info_callback,
                self.sensor_qos,
            )
        self.tf_sub = self.create_subscription(TFMessage, "/tf", self.tf_callback, 10)
        self.tf_static_sub = self.create_subscription(
            TFMessage, "/tf_static", self.tf_callback, 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.cloud_pub = self.create_publisher(PointCloud2, args.output_topic, 10)
        self.path_pub = self.create_publisher(Path, args.path_topic, 10)

        self.depth_log_sub = self.create_subscription(
            Image, args.depth_topic, self.depth_log_callback, self.sensor_qos
        )
        self.pose_log_sub = self.create_subscription(
            PoseStamped, args.pose_topic, self.pose_log_callback, 10
        )
        image_msg_type = (
            CompressedImage
            if args.image_mode == "color" and args.color_image_compressed
            else Image
        )
        self.image_log_sub = self.create_subscription(
            image_msg_type, self.image_topic, self.image_log_callback, self.sensor_qos
        )

        self.depth_sub = message_filters.Subscriber(
            self, Image, args.depth_topic, qos_profile=self.sensor_qos
        )
        self.pose_sub = message_filters.Subscriber(self, PoseStamped, args.pose_topic)
        self.image_sub = message_filters.Subscriber(
            self, image_msg_type, self.image_topic, qos_profile=self.sensor_qos
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.pose_sub, self.image_sub],
            queue_size=20,
            slop=args.sync_slop,
        )
        self.sync.registerCallback(self.sync_callback)

        self.get_logger().info(
            f"Publishing global cloud on {args.output_topic} from {args.depth_topic} + {args.pose_topic} + {self.image_topic} ({args.image_mode})"
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.depth_frame_id = msg.header.frame_id or self.depth_frame_id
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self.get_logger().info(
                f"Received depth camera intrinsics from {self.args.camera_info_topic} with frame {msg.header.frame_id}."
            )
        self.try_update_frame_transforms()

    def color_camera_info_callback(self, msg: CameraInfo) -> None:
        self.color_frame_id = msg.header.frame_id or self.color_frame_id
        if self.color_K is None:
            self.color_K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self.get_logger().info(
                f"Received color camera intrinsics from {self.args.color_camera_info_topic} with frame {msg.header.frame_id}."
            )
        self.try_update_frame_transforms()

    def depth_log_callback(self, msg: Image) -> None:
        if not self._saw_depth:
            self._saw_depth = True
            self.get_logger().info(
                f"Received first depth frame on {self.args.depth_topic}."
            )

    def pose_log_callback(self, msg: PoseStamped) -> None:
        if not self._saw_pose:
            self._saw_pose = True
            self.get_logger().info(
                f"Received first pose frame on {self.args.pose_topic}."
            )

    def image_log_callback(self, msg: Image) -> None:
        if not self._saw_image:
            self._saw_image = True
            self.get_logger().info(
                f"Received first {self.args.image_mode} image on {self.image_topic}."
            )

    def log_missing_inputs(self) -> None:
        self._missing_input_counter += 1
        if self._missing_input_counter % 30 != 1:
            return
        missing = []
        if self.K is None:
            missing.append(self.args.camera_info_topic)
        if self.args.image_mode == "color" and self.color_K is None:
            missing.append(self.args.color_camera_info_topic)
        if self.T_i_depth is None:
            missing.append(
                f"{self.args.pose_child_frame}->{self.depth_frame_id or 'depth'} TF"
            )
        if self.args.image_mode == "color" and self.T_depth_color is None:
            missing.append(
                f"{self.depth_frame_id or 'depth'}->{self.color_frame_id or 'color'} TF"
            )
        self.get_logger().info(
            f"Waiting for inputs before publishing {self.args.image_mode} cloud: {', '.join(missing)}"
        )

    def store_transform(
        self, parent_frame: str, child_frame: str, T: np.ndarray
    ) -> None:
        self.tf_edges.setdefault(parent_frame, {})[child_frame] = T.astype(
            np.float32, copy=False
        )
        self.tf_edges.setdefault(child_frame, {})[parent_frame] = np.linalg.inv(
            T
        ).astype(np.float32, copy=False)

    def lookup_transform(
        self, source_frame: str, target_frame: str
    ) -> np.ndarray | None:
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

    def try_update_frame_transforms(self) -> None:
        if self.depth_frame_id is not None:
            T_pose_depth = self.lookup_transform(
                self.args.pose_child_frame, self.depth_frame_id
            )
            if T_pose_depth is not None:
                self.T_i_depth = T_pose_depth
                if not self._logged_depth_tf:
                    self._logged_depth_tf = True
                    self.get_logger().info(
                        f"Resolved {self.args.pose_child_frame} -> {self.depth_frame_id} transform."
                    )

        if (
            self.args.image_mode == "color"
            and self.depth_frame_id
            and self.color_frame_id
        ):
            T_depth_color = self.lookup_transform(
                self.depth_frame_id, self.color_frame_id
            )
            if T_depth_color is not None:
                self.T_depth_color = T_depth_color
                if not self._logged_color_tf:
                    self._logged_color_tf = True
                    self.get_logger().info(
                        f"Resolved {self.depth_frame_id} -> {self.color_frame_id} transform."
                    )

    def tf_callback(self, msg: TFMessage) -> None:
        has_static = any(transform.header.frame_id for transform in msg.transforms)
        if any(transform.child_frame_id for transform in msg.transforms):
            if not self._saw_tf:
                self._saw_tf = True
                self.get_logger().info("Received first TF message.")
        if has_static and not self._saw_tf_static:
            self._saw_tf_static = True
            self.get_logger().info("Received first TF/TF_STATIC message.")

        for transform in msg.transforms:
            frame_id, child_frame_id, T = tf2np(transform)
            self.store_transform(frame_id, child_frame_id, T)
        self.try_update_frame_transforms()

    def prune_buffer(self, current_position: np.ndarray) -> None:
        kept = deque()
        removed = False
        for sensor_position, cloud, colors in self.global_cloud_buffer:
            if (
                np.linalg.norm(sensor_position - current_position)
                <= self.args.global_radius
            ):
                kept.append((sensor_position, cloud, colors))
            else:
                removed = True
        self.global_cloud_buffer = kept
        if removed:
            self.rebuild_merged_cache()

    def rebuild_merged_cache(self) -> None:
        if not self.global_cloud_buffer:
            self.merged_cloud_cache = np.empty((0, 3), dtype=np.float32)
            self.merged_color_cache = np.empty((0,), dtype=np.uint32)
            return
        self.merged_cloud_cache = np.vstack(
            [cloud for _, cloud, _ in self.global_cloud_buffer]
        ).astype(np.float32, copy=False)
        self.merged_color_cache = np.concatenate(
            [colors for _, _, colors in self.global_cloud_buffer]
        ).astype(np.uint32, copy=False)

    def append_to_merged_cache(self, cloud: np.ndarray, colors: np.ndarray) -> None:
        if cloud.size == 0:
            return
        if self.merged_cloud_cache.size == 0:
            self.merged_cloud_cache = cloud.astype(np.float32, copy=False)
            self.merged_color_cache = colors.astype(np.uint32, copy=False)
            return
        self.merged_cloud_cache = np.concatenate(
            (self.merged_cloud_cache, cloud.astype(np.float32, copy=False)), axis=0
        )
        self.merged_color_cache = np.concatenate(
            (self.merged_color_cache, colors.astype(np.uint32, copy=False)), axis=0
        )

    def get_sample_grid(
        self, depth_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = depth_shape
        grid_key = (h, w, self.args.pixel_step)
        if self.sample_grid_key != grid_key:
            u_coords = np.arange(0, w, self.args.pixel_step, dtype=np.float32)
            v_coords = np.arange(0, h, self.args.pixel_step, dtype=np.float32)
            self.sample_u_grid, self.sample_v_grid = np.meshgrid(u_coords, v_coords)
            self.sample_grid_key = grid_key
        return self.sample_u_grid, self.sample_v_grid

    def publish_cloud(
        self, points: np.ndarray, colors: np.ndarray, stamp, frame_id: str = "world"
    ) -> None:
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        dtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.float32),
            ]
        )
        structured = np.zeros(points.shape[0], dtype=dtype)
        structured["x"], structured["y"], structured["z"] = (
            (
                points[:, 0],
                points[:, 1],
                points[:, 2],
            )
            if points.size
            else (np.empty((0,), dtype=np.float32),) * 3
        )
        structured["rgb"] = colors.view(np.float32)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.cloud_pub.publish(pc2.create_cloud(header, fields, structured))

    def publish_pose_tf(
        self, T_world_imu: np.ndarray, stamp, parent_frame: str
    ) -> None:
        rotation = R.from_matrix(T_world_imu[:3, :3]).as_quat()
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = parent_frame or "world"
        tf_msg.child_frame_id = self.args.pose_child_frame
        tf_msg.transform.translation.x = float(T_world_imu[0, 3])
        tf_msg.transform.translation.y = float(T_world_imu[1, 3])
        tf_msg.transform.translation.z = float(T_world_imu[2, 3])
        tf_msg.transform.rotation.x = float(rotation[0])
        tf_msg.transform.rotation.y = float(rotation[1])
        tf_msg.transform.rotation.z = float(rotation[2])
        tf_msg.transform.rotation.w = float(rotation[3])
        self.tf_broadcaster.sendTransform(tf_msg)

    def publish_path(self, pose_msg: PoseStamped) -> None:
        path_pose = PoseStamped()
        path_pose.header = pose_msg.header
        path_pose.pose = pose_msg.pose
        self.path_msg.header = pose_msg.header
        self.path_msg.poses.append(path_pose)
        if (
            self.args.max_path_length > 0
            and len(self.path_msg.poses) > self.args.max_path_length
        ):
            self.path_msg.poses = self.path_msg.poses[-self.args.max_path_length :]
        self.path_pub.publish(self.path_msg)

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

    def sync_callback(
        self, depth_msg: Image, pose_msg: PoseStamped, image_msg: Image
    ) -> None:
        self._sync_count += 1
        if self._sync_count == 1:
            self.get_logger().info(
                f"Received first synchronized triplet: depth={depth_msg.header.stamp.sec}.{depth_msg.header.stamp.nanosec:09d}, "
                f"pose={pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec:09d}, "
                f"image={image_msg.header.stamp.sec}.{image_msg.header.stamp.nanosec:09d}"
            )
        if self.K is None or self.T_i_depth is None:
            self.log_missing_inputs()
            return
        if self.args.image_mode == "color" and (
            self.color_K is None or self.T_depth_color is None
        ):
            self.log_missing_inputs()
            return

        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if self.args.image_mode == "grayscale":
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="mono8")
        else:
            if self.args.color_image_compressed:
                image = self.bridge.compressed_imgmsg_to_cv2(
                    image_msg, desired_encoding="rgb8"
                )
            else:
                image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        if depth is None or image is None:
            return
        depth = np.asarray(depth)
        if depth_msg.encoding in ("mono16", "16UC1"):
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)
        image = np.asarray(image, dtype=np.uint8)
        if self.args.image_mode == "grayscale" and image.shape[:2] != depth.shape[:2]:
            if self._projection_debug_counter % 20 == 0:
                self.get_logger().info(
                    f"Skipping grayscale projection because image and depth shapes differ: "
                    f"depth_shape={tuple(depth.shape)}, image_shape={tuple(image.shape)}"
                )
            self._projection_debug_counter += 1
            return
        sample_u_grid, sample_v_grid = self.get_sample_grid(depth.shape)

        # `/insight/vio_pose` provides T_w_i, where the world frame is z-up.
        # Depth backprojection yields points in the camera optical frame, so we
        # first map them into the IMU/depth rig frame before applying T_w_i.
        T_w_i = pose_msg_to_np(pose_msg)
        T_world_camera = T_w_i @ self.T_i_depth
        current_position = T_w_i[:3, 3].copy()
        self.publish_pose_tf(T_w_i, pose_msg.header.stamp, pose_msg.header.frame_id)
        self.publish_path(pose_msg)

        if self.args.image_mode == "grayscale":
            cloud_camera, cloud_colors, projection_stats, projection_samples = (
                depth_to_grayscale_cloud(
                    depth,
                    image,
                    self.K,
                    sample_u_grid,
                    sample_v_grid,
                    self.args.pixel_step,
                    self.args.max_depth,
                )
            )
        else:
            cloud_camera, cloud_colors, projection_stats, projection_samples = (
                depth_to_color_cloud(
                    depth,
                    image,
                    self.K,
                    self.color_K,
                    self.T_depth_color,
                    sample_u_grid,
                    sample_v_grid,
                    self.args.pixel_step,
                    self.args.max_depth,
                )
            )
        self._projection_debug_counter += 1
        if cloud_camera.size == 0:
            if self._projection_debug_counter % 20 == 1:
                self.get_logger().info(
                    f"No valid {self.args.image_mode} depth points for current synchronized frame. "
                    f"Projection stats: {projection_stats}, "
                    f"depth_encoding={depth_msg.encoding}, "
                    f"depth_shape={tuple(depth.shape)}, image_shape={tuple(image.shape)}, "
                    f"depth_K={np.round(self.K, 2).tolist()}, "
                    f"imu_to_depth_t={np.round(self.T_i_depth[:3, 3], 4).tolist()}"
                )
        elif self._projection_debug_counter % 20 == 1:
            if self.args.image_mode == "grayscale":
                sample_text = "; ".join(
                    [
                        f"uv=({u:.0f},{v:.0f}) z={z:.3f} gray={gray} xy=({x:.3f},{y:.3f})"
                        for u, v, z, gray, x, y in projection_samples
                    ]
                )
            else:
                sample_text = "; ".join(
                    [
                        f"uv=({u:.0f},{v:.0f}) z={z:.3f} rgb=({r},{g},{b}) xy=({x:.3f},{y:.3f})"
                        for u, v, z, r, g, b, x, y in projection_samples
                    ]
                )
            self.get_logger().info(
                f"{self.args.image_mode.capitalize()} sample points: {sample_text}"
            )
        cloud_world = transform_points(cloud_camera, T_world_camera)
        keep_mask = crop_mask(cloud_world, current_position, self.args.global_radius)
        cloud_world = cloud_world[keep_mask]
        cloud_colors = cloud_colors[keep_mask]

        if self.should_add_keyframe(T_world_camera):
            cloud_world, cloud_colors = voxel_downsample(
                cloud_world, cloud_colors, self.args.voxel_size
            )
            self.global_cloud_buffer.append(
                (current_position, cloud_world, cloud_colors)
            )
            self.append_to_merged_cache(cloud_world, cloud_colors)
            self.last_keyframe_pose = T_world_camera.copy()
            self.prune_buffer(current_position)

        if not self.global_cloud_buffer:
            self.publish_cloud(
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.uint32),
                depth_msg.header.stamp,
                pose_msg.header.frame_id,
            )
            return

        keep_mask = crop_mask(
            self.merged_cloud_cache, current_position, self.args.global_radius
        )
        merged_cloud = self.merged_cloud_cache[keep_mask]
        merged_colors = self.merged_color_cache[keep_mask]

        self.publish_cloud(
            merged_cloud.astype(np.float32),
            merged_colors.astype(np.uint32),
            depth_msg.header.stamp,
            pose_msg.header.frame_id or "world",
        )
        if not self._published_once:
            self._published_once = True
            self.get_logger().info(
                f"Published first {self.args.image_mode} global cloud with {merged_cloud.shape[0]} points in frame {pose_msg.header.frame_id or 'world'}."
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project SLAM depth into a global world-frame point cloud."
    )
    parser.add_argument("--depth-topic", default="/camera/camera/depth/image_rect_raw")
    parser.add_argument(
        "--camera-info-topic", default="/camera/camera/infra1/camera_info"
    )
    parser.add_argument("--pose-topic", default="/insight/vio_pose")
    parser.add_argument("--image-mode", choices=("grayscale", "color"), default="color")
    # grayscale image topic is fixed to /camera/camera/infra1/image_rect_raw
    parser.add_argument(
        "--color-image-topic",
        default="/camera/camera/color/image_rect_raw/compressed",
    )
    parser.add_argument(
        "--color-camera-info-topic", default="/camera/camera/color/camera_info"
    )
    parser.add_argument(
        "--color-image-raw", dest="color_image_compressed", action="store_false"
    )
    parser.set_defaults(color_image_compressed=True)
    parser.add_argument("--output-topic", default="/global_pointcloud")
    parser.add_argument("--path-topic", default="/global_pointcloud_path")
    parser.add_argument("--max-path-length", type=int, default=0)
    parser.add_argument("--pose-child-frame", default="imu")
    parser.add_argument("--sync-slop", type=float, default=0.08)
    parser.add_argument("--pixel-step", type=int, default=2)
    parser.add_argument("--keyframe-translation", type=float, default=0.03)
    parser.add_argument("--keyframe-rotation-deg", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=2)
    parser.add_argument("--global-radius", type=float, default=100.0)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    return parser


def main(args=None) -> None:
    parser = build_parser()
    parsed_args, ros_args = parser.parse_known_args(
        sys.argv[1:] if args is None else args
    )
    rclpy.init(args=ros_args)
    node = GlobalPointCloudPublisher(parsed_args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
