import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
import numpy as np
import sys

from math_utils import matrix_to_quat, msg2np, estimate_pose, tf2np
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
from numba import njit
from visualization_msgs.msg import MarkerArray
from tf2_msgs.msg import TFMessage

logger = logging.getLogger(__name__)
TINYNAV_DB = "tinynav_db"
@njit(cache=True)
def depth_to_cloud_in_world(depth: np.ndarray, K: np.ndarray, baseline: float, pose_in_world: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    points_3d = []
    for u in range(w):
        for v in range(h):
            if 0.1 < depth[v, u] and depth[v, u] < 10.0:
                z = depth[v, u]
                x = (u - K[0, 2]) * z / K[0, 0]
                y = (v - K[1, 2]) * z / K[1, 1]
                norm = np.linalg.norm(np.array([x, y, z]))
                if norm > 10.0 or y < -1.0:
                    continue
                world_x = pose_in_world[0, 0] * x + pose_in_world[0, 1] * y + pose_in_world[0, 2] * z + pose_in_world[0, 3]
                world_y = pose_in_world[1, 0] * x + pose_in_world[1, 1] * y + pose_in_world[1, 2] * z + pose_in_world[1, 3]
                world_z = pose_in_world[2, 0] * x + pose_in_world[2, 1] * y + pose_in_world[2, 2] * z + pose_in_world[2, 3]
                points_3d.append([world_x, world_y, world_z])
    return np.array(points_3d)

def merge_grids(grid1:np.ndarray, grid1_origin:np.ndarray, grid2:np.ndarray, grid2_origin:np.ndarray, resolution:float) -> tuple[np.ndarray, np.ndarray]:
        """
        Merge two grids into one.
        """
        min_x = min(grid1_origin[0], grid2_origin[0])
        min_y = min(grid1_origin[1], grid2_origin[1])
        min_z = min(grid1_origin[2], grid2_origin[2])

        max_x = max(grid1_origin[0] + grid1.shape[0] * resolution, grid2_origin[0] + grid2.shape[0] * resolution)
        max_y = max(grid1_origin[1] + grid1.shape[1] * resolution, grid2_origin[1] + grid2.shape[1] * resolution)
        max_z = max(grid1_origin[2] + grid1.shape[2] * resolution, grid2_origin[2] + grid2.shape[2] * resolution)

        new_shape_x = int((max_x - min_x) / resolution) + 1
        new_shape_y = int((max_y - min_y) / resolution) + 1
        new_shape_z = int((max_z - min_z) / resolution) + 1

        new_grid = np.zeros((new_shape_x, new_shape_y, new_shape_z), dtype=np.float32)
        new_origin = np.array([min_x, min_y, min_z], dtype=np.float32)
        grid1_x_start = int((grid1_origin[0] - min_x) / resolution)
        grid1_y_start = int((grid1_origin[1] - min_y) / resolution)
        grid1_z_start = int((grid1_origin[2] - min_z) / resolution)
        grid2_x_start = int((grid2_origin[0] - min_x) / resolution)
        grid2_y_start = int((grid2_origin[1] - min_y) / resolution)
        grid2_z_start = int((grid2_origin[2] - min_z) / resolution)
        # Merge the grids
        new_grid[grid1_x_start:grid1_x_start + grid1.shape[0],
                  grid1_y_start:grid1_y_start + grid1.shape[1],
                  grid1_z_start:grid1_z_start + grid1.shape[2]] += grid1

        new_grid[grid2_x_start:grid2_x_start + grid2.shape[0],
                  grid2_y_start:grid2_y_start + grid2.shape[1],
                  grid2_z_start:grid2_z_start + grid2.shape[2]] += grid2
        return new_grid, new_origin

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
    top_k_indices = np.argsort(similarity_array, axis = 0)[-loop_top_k:]
    loop_list = []
    for idx in top_k_indices:
        if similarity_array[idx] > loop_similarity_threshold:
            loop_list.append((idx, similarity_array[idx]))
        if len(loop_list) >= loop_top_k:
            break
    return loop_list

def generate_occupancy_map(poses, db, K, baseline, resolution = 0.05, step = 10):
    """
        Genereate a occupancy grid map from the depth images.
        The occupancy grid map is a 3D grid with the following values:
            0 : Unknown
            1 : Free
            2 : Occupied
    """
    raycast_shape = (100, 100, 20)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    global_grid = None
    global_origin = None
    for timestamp, odom_pose in tqdm(poses.items()):
        depth, _, _, _, _ = db.get_depth_embedding_features_images(timestamp)
        odom_translation = odom_pose[:3, 3]
        local_origin = odom_translation - 0.5 * np.array(raycast_shape) * resolution
        local_grid = run_raycasting_loopy(depth, odom_pose, raycast_shape, fx, fy, cx, cy, local_origin, step, resolution, filter_ground = True)
        if global_grid is None:
            global_grid = local_grid
            global_origin = local_origin
        else:
            global_grid, global_origin = merge_grids(global_grid, global_origin, local_grid, local_origin, resolution)
    grid_type = np.zeros_like(global_grid, dtype=np.uint8)
    grid_type[global_grid > 0] = 2  # Occupied
    grid_type[global_grid < 0] = 1  # Free
    x_y_plane = np.max(grid_type, axis=2)
    x_y_plane_image = np.zeros_like(x_y_plane, dtype=np.float32)
    x_y_plane_image[x_y_plane == 2] = 1.0
    x_y_plane_image[x_y_plane == 1] = 0.5
    x_y_plane_image = (x_y_plane_image * 255).astype(np.uint8)
    return global_grid, global_origin, x_y_plane_image

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


class BuildMapNode(Node):
    def __init__(self, map_save_path:str):
        super().__init__('map_node')
        self.super_point_extractor = SuperPointTRT()
        self.light_glue_matcher = LightGlueTRT()
        self.dinov2_model = Dinov2TRT()

        self.bridge = CvBridge()

        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)
        self.depth_sub = Subscriber(self, Image, '/slam/keyframe_depth')
        self.keyframe_image_sub = Subscriber(self, Image, '/slam/keyframe_image')
        self.keyframe_odom_sub = Subscriber(self, Odometry, '/slam/keyframe_odom')
        self.rgb_image_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')

        self.pose_graph_trajectory_pub = self.create_publisher(Path, "/mapping/pose_graph_trajectory", 10)
        self.project_3d_to_2d_pub = self.create_publisher(Image, "/mapping/project_3d_to_2d", 10)
        self.matches_image_pub = self.create_publisher(Image, "/mapping/keyframe_matches_images", 10)
        self.loop_matches_image_pub = self.create_publisher(Image, "/mapping/loop_matches_images", 10)
        self.global_map_marker_pub = self.create_publisher(MarkerArray, "/mapping/global_map_marker", 10)
        self.ts = TimeSynchronizer([self.keyframe_image_sub, self.keyframe_odom_sub, self.depth_sub, self.rgb_image_sub], 1000)
        self.ts.registerCallback(self.keyframe_callback)

        self.K = None
        self.baseline = None
        self.odom = {}
        self.pose_graph_used_pose = {}
        self.relative_pose_constraint = []
        self.last_keyframe_timestamp = None

        os.makedirs(f"{map_save_path}", exist_ok=True)
        self.db = TinyNavDB(map_save_path)

        self.loop_similarity_threshold = 0.90
        self.loop_top_k = 1

        self.map_save_path = map_save_path
        self._on_shutdown_callback_registered = True
        rclpy.get_default_context().on_shutdown(self.on_shutdown)
        self.window_size = 5
        self.keyframe_timestamp_windows = []
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
            logger.info("Camera intrinsics received.")
            self.K = np.array(msg.k).reshape(3, 3)
            fx = self.K[0, 0]
            Tx = msg.p[3]
            self.baseline = -Tx / fx
            self.destroy_subscription(self.camera_info_sub)

    @Timer(name="Mapping Loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms")
    def keyframe_callback(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, depth_msg:Image, rgb_image_msg:Image):
        if self.K is None:
            return
        self.process(keyframe_image_msg, keyframe_odom_msg, depth_msg, rgb_image_msg)

    def process(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, depth_msg:Image, rgb_image_msg:Image):
        with Timer(name = "Msg decode", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            keyframe_image_timestamp = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(keyframe_image_msg.header.stamp.nanosec)
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            odom = msg2np(keyframe_odom_msg)
            infra1_image = self.bridge.imgmsg_to_cv2(keyframe_image_msg, desired_encoding="mono8")
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, desired_encoding="bgr8")

        with Timer(name = "save image and depth", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            self.db.set_entry(keyframe_image_timestamp, depth = depth, infra1_image = infra1_image, rgb_image = rgb_image)

        with Timer(name = "get embeddings", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            embedding = self.get_embeddings(infra1_image)
            self.db.set_entry(keyframe_image_timestamp, embedding = embedding)
        with Timer(name = "super point extractor", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            features = asyncio.run(self.super_point_extractor.infer(infra1_image))
            self.db.set_entry(keyframe_image_timestamp, features = features)

        with Timer(name = "loop and pose graph solve", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            if len(self.odom) == 0 and self.last_keyframe_timestamp is None:
                self.odom[keyframe_image_timestamp] = odom
                self.pose_graph_used_pose[keyframe_image_timestamp] = odom
                self.keyframe_timestamp_windows.append(keyframe_image_timestamp)
            else:
                last_keyframe_odom_pose = self.odom[self.last_keyframe_timestamp]
                T_prev_curr = np.linalg.inv(last_keyframe_odom_pose) @ odom
                self.relative_pose_constraint.append((keyframe_image_timestamp, self.last_keyframe_timestamp, T_prev_curr))
                self.edges.add((self.last_keyframe_timestamp, keyframe_image_timestamp))
                self.pose_graph_used_pose[keyframe_image_timestamp] = odom
                self.odom[keyframe_image_timestamp] = odom
                self.keyframe_timestamp_windows.append(keyframe_image_timestamp)

                for prev_timestamp in self.keyframe_timestamp_windows[-self.window_size:-1]:
                    _,_, prev_features, _, _= self.db.get_depth_embedding_features_images(prev_timestamp)
                    curr_depth, _, curr_features, _, _ = self.db.get_depth_embedding_features_images(keyframe_image_timestamp)

                    prev_matched_keypoints, curr_matched_keypoints, matches = self.match_keypoints(prev_features, curr_features)
                    success, T_prev_curr_features, _, _, inliers = estimate_pose(prev_matched_keypoints, curr_matched_keypoints, curr_depth, self.K)
                    if success and len(inliers) >= 100:
                        self.relative_pose_constraint.append((keyframe_image_timestamp, prev_timestamp, T_prev_curr_features))
                        print(f"Added relative pose constraint: {keyframe_image_timestamp} -> {prev_timestamp}")

                def find_loop_and_pose_graph(timestamp):
                    target_embedding = self.db.get_embedding(timestamp)
                    valid_timestamp = [t for t in self.pose_graph_used_pose.keys() if t + 10 * 1e9 < timestamp]
                    valid_embeddings = np.array([self.db.get_embedding(t) for t in valid_timestamp])

                    idx_to_timestamp = {i:t for i, t in enumerate(valid_timestamp)}
                    with Timer(name = "find loop", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
                        loop_list = find_loop(target_embedding, valid_embeddings, self.loop_similarity_threshold, self.loop_top_k)
                    with Timer(name = "Relative pose estimation", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
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
                    with Timer(name = "solve pose graph", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
                        self.pose_graph_used_pose = solve_pose_graph(self.pose_graph_used_pose, self.relative_pose_constraint, max_iteration_num = 5)
                find_loop_and_pose_graph(keyframe_image_timestamp)

        with Timer(name = "pose graph trajectory publish", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
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
        if self.K is None:
            return
        with Timer(name = "final pose graph", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            self.pose_graph_used_pose = solve_pose_graph(self.pose_graph_used_pose, self.relative_pose_constraint)

        np.save(f"{self.map_save_path}/poses.npy", self.pose_graph_used_pose, allow_pickle = True)
        np.save(f"{self.map_save_path}/intrinsics.npy", self.K)
        np.save(f"{self.map_save_path}/baseline.npy", self.baseline)
        occupancy_resolution = 0.05
        occupancy_step = 10
        occupancy_grid, occupancy_origin, occupancy_2d_image = generate_occupancy_map(self.pose_graph_used_pose, self.db, self.K, self.baseline, occupancy_resolution, occupancy_step)
        occupancy_meta = np.array([occupancy_origin[0], occupancy_origin[1], occupancy_origin[2], occupancy_resolution], dtype=np.float32)
        np.save(f"{self.map_save_path}/occupancy_grid.npy", occupancy_grid)
        np.save(f"{self.map_save_path}/occupancy_meta.npy", occupancy_meta)
        cv2.imwrite(f"{self.map_save_path}/occupancy_2d_image.png", occupancy_2d_image)
        print(f"T_rgb_to_infra1: {self.T_rgb_to_infra1}")
        np.save(f"{self.map_save_path}/T_rgb_to_infra1.npy", self.T_rgb_to_infra1, allow_pickle = True)
        np.save(f"{self.map_save_path}/rgb_camera_intrinsics.npy", self.rgb_camera_K, allow_pickle = True)
        np.save(f"{self.map_save_path}/edges.npy", list(self.edges), allow_pickle = True)
        self.db.close()

    def on_shutdown(self):
        print("Shutdown callback triggered - saving mapping data...")
        self.global_map_pub_thread_running = False
        # Only join if thread was actually started
        if hasattr(self, 'global_map_pub_thread') and self.global_map_pub_thread.is_alive():
            self.global_map_pub_thread.join()
        self.save_mapping()
        print("Mapping data saved successfully.")

def main(args=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_save_path", type=str, default=TINYNAV_DB)
    parsed_args, unknown_args = parser.parse_known_args(sys.argv[1:])
    node = BuildMapNode(parsed_args.map_save_path)
    try:
        rclpy.spin(node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C pressed, shutting down...")
    finally:
        node.on_shutdown()  # <-- you can implement a shutdown method in your node
        node.destroy_node()

if __name__ == '__main__':
    main()
