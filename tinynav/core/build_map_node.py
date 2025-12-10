import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool
import numpy as np
import sys

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from math_utils import matrix_to_quat, msg2np, estimate_pose, tf2np, depth_to_cloud
from sensor_msgs.msg import Image, CameraInfo
from message_filters import TimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
from codetiming import Timer
import os
import argparse

from tinynav.tinynav_cpp_bind import pose_graph_solve
from models_trt import LightGlueTRT, Dinov2TRT, SuperPointTRT
from planning_node import run_raycasting_loopy
import logging
import asyncio
import shelve
from tqdm import tqdm
import einops
from tf2_msgs.msg import TFMessage
from typing import Tuple, Dict
import json

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped,Point
from scipy.spatial.transform import Rotation as R

from rclpy.executors import SingleThreadedExecutor
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from rosgraph_msgs.msg import Clock
from rclpy.serialization import deserialize_message



logger = logging.getLogger(__name__)

def z_value_to_color(z, z_min, z_max):
    color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)
    normalized_z = (z - z_min) / (z_max - z_min)
    if normalized_z < 0.25:
        color.g = normalized_z * 4.0
        color.b = 1.0
    elif normalized_z < 0.5:
        color.g = 1.0
        color.b = 1.0 - (normalized_z - 0.25) * 4.0
    elif normalized_z < 0.75:
        color.r = (normalized_z - 0.5) * 4.0
        color.g = 1.0
    else:
        color.r = 1.0
        color.g = 1.0 - (normalized_z - 0.75) * 4.0
    return color

def convert_nerf_format(output_dir: str, infra1_poses: dict, rgb_intrinscis:np.ndarray, image_size: Tuple[int, int], T_rgb_to_infra1:np.ndarray):
    camera_model = "PINHOLE"
    fl_x = rgb_intrinscis[0, 0]
    fl_y = rgb_intrinscis[1, 1]
    cx = rgb_intrinscis[0, 2]
    cy = rgb_intrinscis[1, 2]
    w = image_size[1]
    h = image_size[0]
    frames = []
    opencv_to_opengl_convention = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    for timestmap, camera_to_world_pose in infra1_poses.items():
        camera_to_world_opengl = camera_to_world_pose @ T_rgb_to_infra1 @ opencv_to_opengl_convention
        frame = {
            "file_path": f"images/image_{timestmap}.png",
            "transform_matrix": camera_to_world_opengl.tolist(),
        }
        frames.append(frame)

    data = {
        "camera_model": camera_model,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "frames": frames
    }

    with open(f"{output_dir}/transforms.json", "w") as f:
        json.dump(data, f, indent=4)

def merge_local_into_global(global_grid:np.ndarray, global_origin:np.ndarray, local_grid:np.ndarray, local_origin:np.ndarray, resolution:float) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge a local grid into a global grid.
    """
    resolution_half = np.array([resolution / 2.0, resolution / 2.0, resolution / 2.0], dtype=np.float32)
    local_origin_offset = ((local_origin - global_origin + resolution_half) / resolution).astype(np.int32)
    global_grid[local_origin_offset[0]:local_origin_offset[0] + local_grid.shape[0],
                local_origin_offset[1]:local_origin_offset[1] + local_grid.shape[1],
                local_origin_offset[2]:local_origin_offset[2] + local_grid.shape[2]] += local_grid
    return global_grid, global_origin

def solve_pose_graph(pose_graph_used_pose:dict, relative_pose_constraint:list, max_iteration_num:int = 1024) -> dict:
    """
    Solve the bundle adjustment problem.
    """
    if len(relative_pose_constraint) == 0:
        return pose_graph_used_pose
    min_timestamp = min(pose_graph_used_pose.keys())
    constant_pose_index_dict = { min_timestamp : True }

    relative_pose_constraint = [
        (curr_timestamp, prev_timestamp, T_prev_curr, np.array([10.0, 10.0, 10.0]), np.array([30.0, 30.0, 30.0]))
        for curr_timestamp, prev_timestamp, T_prev_curr in relative_pose_constraint]
    optimized_camera_poses = pose_graph_solve(pose_graph_used_pose, relative_pose_constraint, constant_pose_index_dict, max_iteration_num)
    return {t: optimized_camera_poses[t] for t in sorted(optimized_camera_poses.keys())}

def find_loop(target_embedding:np.ndarray, embeddings:np.ndarray, loop_similarity_threshold:float, loop_top_k:int) -> list[tuple[int, float]]:
    if len(embeddings) == 0:
        return []
    similarity_array = einops.einsum(target_embedding, embeddings, "d, n d -> n")
    top_k_indices = np.argsort(similarity_array, axis = 0)
    loop_list = []
    for idx in top_k_indices:
        if similarity_array[idx] > loop_similarity_threshold:
            loop_list.append((idx, similarity_array[idx]))
    return loop_list[-loop_top_k:]

def generate_occupancy_map(poses, db, K, baseline, resolution = 0.05, step = 10):
    """
        Generate a occupancy grid map from the depth images.
        The occupancy grid map is a 3D grid with the following values:
            0 : Unknown
            1 : Free
            2 : Occupied
    """
    raycast_shape = (100, 100, 20)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    odom_pose_min_position = np.array([np.inf, np.inf, np.inf])
    odom_pose_max_position = np.array([-np.inf, -np.inf, -np.inf])
    for timestamp, odom_pose in poses.items():
        odom_translation = odom_pose[:3, 3]
        odom_pose_min_position = np.minimum(odom_pose_min_position, odom_translation)
        odom_pose_max_position = np.maximum(odom_pose_max_position, odom_translation)
    odom_pose_min_position = np.floor(odom_pose_min_position / resolution) * resolution
    odom_pose_max_position = np.ceil(odom_pose_max_position / resolution) * resolution
    global_grid_shape = (np.ceil((odom_pose_max_position - odom_pose_min_position) / resolution) + raycast_shape).astype(np.int32)
    global_origin = odom_pose_min_position - 0.5 * np.array(raycast_shape) * resolution
    global_grid = np.zeros(global_grid_shape, dtype=np.float32)
    for timestamp, odom_pose in tqdm(poses.items()):
        depth, _, _, _, _ = db.get_depth_embedding_features_images(timestamp)
        odom_translation = odom_pose[:3, 3]
        local_origin = np.floor(odom_translation / resolution) * resolution - 0.5 * np.array(raycast_shape) * resolution
        local_grid = run_raycasting_loopy(depth, odom_pose, raycast_shape, fx, fy, cx, cy, local_origin, step, resolution, filter_ground = True)
        global_grid, global_origin = merge_local_into_global(global_grid, global_origin, local_grid, local_origin, resolution)

    grid_type = np.zeros_like(global_grid, dtype=np.uint8)
    grid_type[global_grid > 0] = 2  # Occupied
    grid_type[global_grid < 0] = 1  # Free
    x_y_plane = np.max(grid_type, axis=2)
    x_y_plane_image = np.zeros_like(x_y_plane, dtype=np.float32)
    x_y_plane_image[x_y_plane == 2] = 1.0
    x_y_plane_image[x_y_plane == 1] = 0.5
    x_y_plane_image = (x_y_plane_image * 255).astype(np.uint8)
    return grid_type, global_origin, x_y_plane_image

class IntKeyShelf:
    def __init__(self, filename):
        self.db = shelve.open(filename)

    def __getitem__(self, key: int):
        return self.db[str(key)]

    def __setitem__(self, key: int, value):
        self.db[str(key)] = value

    def __delitem__(self, key: int):
        del self.db[str(key)]

    def __contains__(self, key: int):
        return str(key) in self.db

    def keys(self):
        return [int(k) for k in self.db.keys()]

    def close(self):
        self.db.close()


class OdomPoseRecorder:
    """
    Utility class to record continuous odometry data to disk.
    Saves timestamp-pose pairs for later timestamp-based queries.
    """

    def __init__(self, save_path: str, prefix: str = "poses"):
        self.save_path = save_path
        self.prefix = prefix
        self.file_save_path = os.path.join(save_path, f"{prefix}_continuous_odom.npy")
        self.poses: Dict[int, np.ndarray] = {}  # timestamp_ns -> 4x4 pose matrix

        os.makedirs(save_path, exist_ok=True)

    def record_odometry_msg(self, odom_msg: Odometry) -> None:
        timestamp_ns = int(odom_msg.header.stamp.sec * 1e9) + int(
            odom_msg.header.stamp.nanosec
        )
        pose_matrix = msg2np(odom_msg)
        self.poses[timestamp_ns] = pose_matrix

    def save_to_disk(self) -> None:
        if not self.poses:
            logger.warning(f"No continuous odom poses to save for {self.prefix}")
            return

        logger.info(f"{self.prefix}: Saved {len(self.poses)} continuous odom poses")
        # Create a copy of the dict for saving to avoid any typing issues
        poses_to_save = dict(self.poses)
        np.save(self.file_save_path, poses_to_save, allow_pickle=True)  # type: ignore

        logger.info(f"Saved {len(self.poses)} poses to {self.file_save_path}")

    def load_from_disk(self) -> bool:
        if not os.path.exists(self.file_save_path):
            logger.warning(f"Pose file not found: {self.file_save_path}")
            return False

        try:
            self.poses = np.load(self.file_save_path, allow_pickle=True).item()
            logger.info(
                f"[PoseRecorder] Loaded {len(self.poses)} poses from {self.file_save_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load poses from {self.file_save_path}: {e}")
            return False

    def clear(self) -> None:
        self.poses.clear()


class TinyNavDB():
    def __init__(self, map_save_path:str, is_scratch:bool = True):
        self.map_save_path = map_save_path
        if is_scratch:
            if os.path.exists(f"{map_save_path}/features.db"):
                os.remove(f"{map_save_path}/features.db")
            if os.path.exists(f"{map_save_path}/infra1_images.db"):
                os.remove(f"{map_save_path}/infra1_images.db")
            if os.path.exists(f"{map_save_path}/depths.db"):
                os.remove(f"{map_save_path}/depths.db")
            if os.path.exists(f"{map_save_path}/rgb_images.db"):
                os.remove(f"{map_save_path}/rgb_images.db")
            if os.path.exists(f"{map_save_path}/embeddings.db"):
                os.remove(f"{map_save_path}/embeddings.db")
        self.features = IntKeyShelf(f"{map_save_path}/features")
        self.embeddings = IntKeyShelf(f"{map_save_path}/embeddings")
        self.infra1_images = IntKeyShelf(f"{map_save_path}/infra1_images")
        self.depths = IntKeyShelf(f"{map_save_path}/depths")
        self.rgb_images = IntKeyShelf(f"{map_save_path}/rgb_images")

    def set_entry(self, key:int,   depth:np.ndarray = None, embedding:np.ndarray = None, features:dict = None,  infra1_image:np.ndarray = None, rgb_image:np.ndarray = None):
        if infra1_image is not None:
            self.infra1_images[key] = infra1_image
        if rgb_image is not None:
            self.rgb_images[key] = rgb_image
        if depth is not None:
            self.depths[key] = depth
        if embedding is not None:
            self.embeddings[key] = embedding
        if features is not None:
            self.features[key] = features

    def get_depth_embedding_features_images(self, key:int):
        return self.depths[key], self.embeddings[key], self.features[key], self.rgb_images[key], self.infra1_images[key]

    def get_embedding(self, key:int):
        return self.embeddings[key]

    def close(self):
        self.features.close()
        self.embeddings.close()
        self.infra1_images.close()
        self.depths.close()
        self.rgb_images.close()

class BagPlayer(Node):
    def __init__(self, bag_uri: str, storage_id: str = "sqlite3", serialization_format: str = "cdr",
    ):
        super().__init__("rosbag_player")

        self._storage_options = StorageOptions(uri=bag_uri, storage_id="sqlite3",)
        self._converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr",)

        self._reader = SequentialReader()
        self._reader.open(self._storage_options, self._converter_options)

        # topic -> (publisher, msg_type)
        self._topic_publishers = {}

        # Build publishers for all topics in the bag
        for topic_info in self._reader.get_all_topics_and_types():
            msg_type = get_message(topic_info.type)
            pub = self.create_publisher(msg_type, topic_info.name, 10)
            self._topic_publishers[topic_info.name] = (pub, msg_type)

        # /clock publisher (for use_sim_time)
        self._clock_pub = self.create_publisher(Clock, "/clock", 10)

        self.get_logger().info(f"BagPlayer opened bag: {bag_uri}")

    def play_next(self) -> bool:
        """
        Publish the next message from the bag.
        Returns False when there are no more messages.
        """
        if not self._reader.has_next():
            return False

        topic, serialized_msg, timestamp_ns = self._reader.read_next()

        # Find publisher + msg type for this topic
        pub_and_type = self._topic_publishers.get(topic)
        if pub_and_type is None:
            # No publisher (should not really happen, but don't crash playback)
            self.get_logger().warn(f"No publisher for topic '{topic}'")
            return True

        pub, msg_type = pub_and_type

        # Deserialize and publish actual message
        msg = deserialize_message(serialized_msg, msg_type)
        pub.publish(msg)

        # Publish /clock with the same timestamp (for use_sim_time)
        if self._clock_pub is not None:
            clock_msg = Clock()
            clock_msg.clock.sec = int(timestamp_ns // 1_000_000_000)
            clock_msg.clock.nanosec = int(timestamp_ns % 1_000_000_000)
            self._clock_pub.publish(clock_msg)

        return True

class BuildMapNode(Node):
    def __init__(self, map_save_path:str, verbose_timer: bool = True):
        super().__init__('map_node')
        self.verbose_timer = verbose_timer
        self.logger = logging.getLogger(__name__)
        self.timer_logger = self.logger.info if verbose_timer else self.logger.debug
        self.super_point_extractor = SuperPointTRT()
        self.light_glue_matcher = LightGlueTRT()
        self.dinov2_model = Dinov2TRT()

        self.bridge = CvBridge()

        self.tf_broadcaster = TransformBroadcaster(self)

        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)
        self.depth_sub = Subscriber(self, Image, '/slam/keyframe_depth')
        self.keyframe_image_sub = Subscriber(self, Image, '/slam/keyframe_image')
        self.keyframe_odom_sub = Subscriber(self, Odometry, '/slam/keyframe_odom')
        self.rgb_image_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.continuous_odom_sub = self.create_subscription(Odometry, '/slam/odometry', self.continuous_odom_callback, 100)

        self.marker_pub = self.create_publisher(MarkerArray, '/mapping/pointcloud_markers', 10)
        self.local_map_pub = self.create_publisher(PointCloud2, "/mapping/local_map", 10)
        self.pose_graph_trajectory_pub = self.create_publisher(Path, "/mapping/pose_graph_trajectory", 10)
        self.project_3d_to_2d_pub = self.create_publisher(Image, "/mapping/project_3d_to_2d", 10)
        self.matches_image_pub = self.create_publisher(Image, "/mapping/keyframe_matches_images", 10)
        self.loop_matches_image_pub = self.create_publisher(Image, "/mapping/loop_matches_images", 10)
        self.global_map_marker_pub = self.create_publisher(MarkerArray, "/mapping/global_map_marker", 10)

        # Add stop signal subscription and save finished publisher
        self.mapping_stop_sub = self.create_subscription(Bool, '/benchmark/stop', self.mapping_stop_callback, 10)
        self.mapping_save_finished_pub = self.create_publisher(Bool, '/benchmark/data_saved', 10)
        self.ts = TimeSynchronizer([self.keyframe_image_sub, self.keyframe_odom_sub, self.depth_sub, self.rgb_image_sub], 1000)
        self.ts.registerCallback(self.keyframe_callback)

        self.K = None
        self.baseline = None
        self.odom = {}
        self.pose_graph_used_pose = {}
        self.relative_pose_constraint = []
        self.last_keyframe_timestamp = None
        self.continuous_odom_recorder = OdomPoseRecorder(map_save_path, "mapping")

        os.makedirs(f"{map_save_path}", exist_ok=True)
        self.db = TinyNavDB(map_save_path)

        self.marker_id = 0

        self.loop_similarity_threshold = 0.90
        self.loop_top_k = 1

        self.map_save_path = map_save_path
        self._save_completed = False
        self.tf_sub = Subscriber(self, TFMessage, "/tf")
        self.tf_sub.registerCallback(self.tf_callback)
        self.T_rgb_to_infra1 = None
        self.rgb_camera_info_sub = Subscriber(self, CameraInfo, "/camera/camera/color/camera_info")
        self.rgb_camera_info_sub.registerCallback(self.rgb_camera_info_callback)
        self.rgb_camera_K = None

        self.edges = set()

    def tf_callback(self, msg:TFMessage):
        T_infra1_to_link = None
        T_infra1_optical_to_infra1 = None
        T_rgb_to_link = None
        T_rgb_optical_to_rgb = None
        for t in msg.transforms:
            frame_id, child_frame_id, T = tf2np(t)
            if frame_id == "camera_link" and child_frame_id == "camera_infra1_frame":
                T_infra1_to_link = T
            if frame_id == "camera_infra1_frame" and child_frame_id == "camera_infra1_optical_frame":
                T_infra1_optical_to_infra1 = T
            if frame_id == "camera_color_frame" and child_frame_id == "camera_color_optical_frame":
                T_rgb_optical_to_rgb = T
            if frame_id == "camera_link" and child_frame_id == "camera_color_frame":
                T_rgb_to_link = T
        if T_infra1_optical_to_infra1 is None or T_rgb_optical_to_rgb is None or T_infra1_to_link is None or T_rgb_to_link is None:
            return
        self.T_rgb_to_infra1 = np.linalg.inv(T_infra1_optical_to_infra1) @ np.linalg.inv(T_infra1_to_link) @ T_rgb_to_link @ T_rgb_optical_to_rgb

    def rgb_camera_info_callback(self, msg:CameraInfo):
        if self.rgb_camera_K is None:
            self.rgb_camera_K = np.array(msg.k).reshape(3, 3)

    def info_callback(self, msg:CameraInfo):
        if self.K is None:
            self.get_logger().info("Camera intrinsics received.")
            self.K = np.array(msg.k).reshape(3, 3)
            fx = self.K[0, 0]
            Tx = msg.p[3]
            self.baseline = -Tx / fx
            self.destroy_subscription(self.camera_info_sub)

    def continuous_odom_callback(self, odom_msg: Odometry):
        self.continuous_odom_recorder.record_odometry_msg(odom_msg)

    def mapping_stop_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Received benchmark stop signal, starting save process...")
            try:
                self.save_mapping()
                self.get_logger().info("Mapping save completed successfully")

                # Publish save finished signal
                save_finished_msg = Bool()
                save_finished_msg.data = True
                self.mapping_save_finished_pub.publish(save_finished_msg)
                self.get_logger().info("Published data save finished signal")

            except Exception as e:
                self.get_logger().error(f"Error during mapping save: {e}")
                # Still publish completion signal even if there was an error
                save_finished_msg = Bool()
                save_finished_msg.data = False
                self.mapping_save_finished_pub.publish(save_finished_msg)

    def keyframe_callback(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, depth_msg:Image, rgb_image_msg:Image):
        with Timer(name="Mapping Loop", text="[{name}] Elapsed time: {milliseconds:.0f} ms\n\n", logger=self.timer_logger):
            if self.K is None:
                return
            self.process(keyframe_image_msg, keyframe_odom_msg, depth_msg, rgb_image_msg)

    def process(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, depth_msg:Image, rgb_image_msg:Image):
        with Timer(name = "Msg decode", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            keyframe_image_timestamp = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(keyframe_image_msg.header.stamp.nanosec)
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            odom = msg2np(keyframe_odom_msg)
            infra1_image = self.bridge.imgmsg_to_cv2(keyframe_image_msg, desired_encoding="mono8")
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, desired_encoding="bgr8")

        with Timer(name = "save image and depth", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            self.db.set_entry(keyframe_image_timestamp, depth = depth, infra1_image = infra1_image, rgb_image = rgb_image)

        with Timer(name = "get embeddings", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            embedding = self.get_embeddings(infra1_image)
            self.db.set_entry(keyframe_image_timestamp, embedding = embedding)
        with Timer(name = "super point extractor", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            features = asyncio.run(self.super_point_extractor.infer(infra1_image))
            self.db.set_entry(keyframe_image_timestamp, features = features)

        with Timer(name = "loop and pose graph solve", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            if len(self.odom) == 0 and self.last_keyframe_timestamp is None:
                self.odom[keyframe_image_timestamp] = odom
                self.pose_graph_used_pose[keyframe_image_timestamp] = odom
            else:
                last_keyframe_odom_pose = self.odom[self.last_keyframe_timestamp]
                T_prev_curr = np.linalg.inv(last_keyframe_odom_pose) @ odom
                self.relative_pose_constraint.append((keyframe_image_timestamp, self.last_keyframe_timestamp, T_prev_curr))
                self.edges.add((self.last_keyframe_timestamp, keyframe_image_timestamp))
                self.pose_graph_used_pose[keyframe_image_timestamp] = odom
                self.odom[keyframe_image_timestamp] = odom


                def find_loop_and_pose_graph(timestamp):
                    target_embedding = self.db.get_embedding(timestamp)
                    valid_timestamp = [t for t in self.pose_graph_used_pose.keys() if t + 10 * 1e9 < timestamp]
                    valid_embeddings = np.array([self.db.get_embedding(t) for t in valid_timestamp])

                    idx_to_timestamp = {i:t for i, t in enumerate(valid_timestamp)}
                    with Timer(name = "find loop", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        loop_list = find_loop(target_embedding, valid_embeddings, self.loop_similarity_threshold, self.loop_top_k)
                    with Timer(name = "Relative pose estimation", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        for idx, similarity in loop_list:
                            prev_timestamp = idx_to_timestamp[idx]
                            curr_timestamp = timestamp
                            prev_depth, _, prev_features, _, _ = self.db.get_depth_embedding_features_images(prev_timestamp)
                            curr_depth, _, curr_features, _, _ = self.db.get_depth_embedding_features_images(curr_timestamp)
                            prev_matched_keypoints, curr_matched_keypoints, matches = self.match_keypoints(prev_features, curr_features)
                            success, T_prev_curr, _, _, inliers = estimate_pose(prev_matched_keypoints, curr_matched_keypoints, curr_depth, self.K)
                            if success and len(inliers) >= 100:
                                self.relative_pose_constraint.append((curr_timestamp, prev_timestamp, T_prev_curr))
                                print(f"Added loop relative pose constraint: {curr_timestamp} -> {prev_timestamp}")
                                self.edges.add((prev_timestamp, curr_timestamp))
                    with Timer(name = "solve pose graph", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        self.pose_graph_used_pose = solve_pose_graph(self.pose_graph_used_pose, self.relative_pose_constraint, max_iteration_num = 5)
                find_loop_and_pose_graph(keyframe_image_timestamp)

        with Timer(name = "publish local pointcloud", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            cloud = depth_to_cloud(depth, self.K, 30, 3)
            self.publish_local_map(cloud, 'camera_'+str(keyframe_image_timestamp))

        with Timer(name = "tf publish", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            self.publish_all_transforms()

        with Timer(name = "pose graph trajectory publish", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            self.pose_graph_trajectory_publish(keyframe_image_timestamp)
        self.last_keyframe_timestamp = keyframe_image_timestamp

    def get_embeddings(self, image: np.ndarray) -> np.ndarray:
        # shape: (1, 768)
        return asyncio.run(self.dinov2_model.infer(image))

    def match_keypoints(self, feats0:dict, feats1:dict, image_shape = np.array([848, 480], dtype = np.int64)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        match_result = asyncio.run(self.light_glue_matcher.infer(feats0["kpts"], feats1["kpts"], feats0['descps'], feats1['descps'], feats0['mask'], feats1['mask'], image_shape, image_shape))
        match_indices = match_result["match_indices"][0]
        valid_mask = match_indices != -1
        keypoints0 = feats0["kpts"][0][valid_mask]
        keypoints1 = feats1["kpts"][0][match_indices[valid_mask]]
        matches = []
        for i, index in enumerate(match_indices):
            if index != -1:
                matches.append([i, index])
        return keypoints0, keypoints1, np.array(matches, dtype=np.int64)

    def pose_graph_trajectory_publish(self, timestamp):
        path_msg = Path()
        path_msg.header.stamp.sec = int(timestamp / 1e9)
        path_msg.header.stamp.nanosec = int(timestamp % 1e9)
        path_msg.header.frame_id = "world"
        for t, pose_in_world in self.pose_graph_used_pose.items():
            pose = PoseStamped()
            pose.header = path_msg.header
            t = pose_in_world[:3, 3]
            quat = matrix_to_quat(pose_in_world[:3, :3])
            pose.pose.position.x = t[0]
            pose.pose.position.y = t[1]
            pose.pose.position.z = t[2]
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            path_msg.poses.append(pose)
        self.pose_graph_trajectory_pub.publish(path_msg)

    def save_mapping(self):
        if self._save_completed:
            self.get_logger().info("Mapping data already saved, skipping duplicate save")
            return

        if self.K is None:
            self.get_logger().info("No camera intrinsics available, skipping save")
            return

        self.get_logger().info("Saving mapping data...")

        # Save continuous poses
        self.continuous_odom_recorder.save_to_disk()

        with Timer(name = "final pose graph", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            self.pose_graph_used_pose = solve_pose_graph(self.pose_graph_used_pose, self.relative_pose_constraint)

        np.save(f"{self.map_save_path}/poses.npy", self.pose_graph_used_pose, allow_pickle = True)
        np.save(f"{self.map_save_path}/intrinsics.npy", self.K)
        np.save(f"{self.map_save_path}/baseline.npy", self.baseline)
        print(f"T_rgb_to_infra1: {self.T_rgb_to_infra1}")
        np.save(f"{self.map_save_path}/T_rgb_to_infra1.npy", self.T_rgb_to_infra1, allow_pickle = True)
        np.save(f"{self.map_save_path}/rgb_camera_intrinsics.npy", self.rgb_camera_K, allow_pickle = True)
        np.save(f"{self.map_save_path}/edges.npy", list(self.edges), allow_pickle = True)

        # Generate occupancy map
        occupancy_resolution = 0.05
        occupancy_step = 10
        occupancy_grid, occupancy_origin, occupancy_2d_image = generate_occupancy_map(self.pose_graph_used_pose, self.db, self.K, self.baseline, occupancy_resolution, occupancy_step)
        occupancy_meta = np.array([occupancy_origin[0], occupancy_origin[1], occupancy_origin[2], occupancy_resolution], dtype=np.float32)
        np.save(f"{self.map_save_path}/occupancy_grid.npy", occupancy_grid)
        np.save(f"{self.map_save_path}/occupancy_meta.npy", occupancy_meta)
        cv2.imwrite(f"{self.map_save_path}/occupancy_2d_image.png", occupancy_2d_image)

        image_size = None
        os.makedirs(f"{self.map_save_path}/images", exist_ok=True)
        for timestamp, infra1_pose in self.pose_graph_used_pose.items():
            _, _, _, rgb_image, _ = self.db.get_depth_embedding_features_images(timestamp)
            if image_size is None:
                image_size = rgb_image.shape[:2]
            cv2.imwrite(f"{self.map_save_path}/images/image_{timestamp}.png", rgb_image)
        convert_nerf_format(self.map_save_path, self.pose_graph_used_pose, self.rgb_camera_K, image_size, self.T_rgb_to_infra1)
        self.db.close()

        self._save_completed = True
        self.get_logger().info("Full mapping data saved successfully")


    def pointcloud_to_marker_array(self, points, frame_id='camera',colors=None):
        marker_array = MarkerArray()
        
        # Create point cloud Marker
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pointcloud"
        marker.id = self.marker_id
        self.marker_id = self.marker_id + 1

        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # Set Marker properties
        marker.scale.x = 0.03  # Point width
        marker.scale.y = 0.03  # Point height
        marker.scale.z = 0.0   # For POINTS type, z is not used
        
        # Set orientation (unit quaternion)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Set position
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        
        # Set points
        marker.points = []
        for point in points:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = float(point[2])
            if (p.y > 0):
                marker.points.append(p)
                c = z_value_to_color(float(point[1]), -3, 1)
                marker.colors.append(c)
        
        # Set lifetime (0 means never expire)
        marker.lifetime.sec = 0
        marker.frame_locked = True
        
        marker_array.markers.append(marker) 
        
        return marker_array

    def publish_local_map(self, point_cloud, frame_id):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        marker_array = self.pointcloud_to_marker_array(point_cloud.tolist(), frame_id)
        self.marker_pub.publish(marker_array)

    def publish_all_transforms(self):
        """Publish all pose TF transforms"""
        if not self.pose_graph_used_pose:
            return
            
        transforms = []        
                
        for time, pose_in_world in self.pose_graph_used_pose.items():
            transform = TransformStamped()
            
            # Set header
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = 'world'
            transform.child_frame_id = 'camera_' + str(time)
            
            # Set position
            t = pose_in_world[:3, 3]
            transform.transform.translation.x = t[0]
            transform.transform.translation.y = t[1]
            transform.transform.translation.z = t[2]
            qx,qy,qz,qw =  R.from_matrix(pose_in_world[:3, :3]).as_quat()
            transform.transform.rotation.x = qx
            transform.transform.rotation.y = qy
            transform.transform.rotation.z = qz
            transform.transform.rotation.w = qw

            transforms.append(transform)
        
        # Publish all TF transforms
        self.tf_broadcaster.sendTransform(transforms)
        

    def destroy_node(self):
        try:
            self.save_mapping()
            super().destroy_node()
        except Exception:
            # Ignore errors during destruction as resources may already be freed
            pass


def main(args=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    rclpy.init(args=args)


    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_file", type=str, default="tinynav_db")
    parser.add_argument("--map_save_path", type=str, default="tinynav_db")
    parser.add_argument("--verbose_timer", action="store_true", default=True, help="Enable verbose timer output")
    parser.add_argument("--no_verbose_timer", dest="verbose_timer", action="store_false", help="Disable verbose timer output")
    parsed_args, unknown_args = parser.parse_known_args(sys.argv[1:])

    exec_ = SingleThreadedExecutor()
    player_node = BagPlayer(parsed_args.bag_file)
    map_node = BuildMapNode(parsed_args.map_save_path, verbose_timer=parsed_args.verbose_timer)
    exec_.add_node(player_node)
    exec_.add_node(map_node)
    while rclpy.ok() and player_node.play_next():
        exec_.spin_once(timeout_sec=0.001)
    map_node.save_mapping()

if __name__ == '__main__':
    main()
