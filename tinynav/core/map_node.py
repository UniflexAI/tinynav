import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool
import numpy as np
import sys
import json

from math_utils import matrix_to_quat, msg2np, np2msg, estimate_pose, np2tf, theta_star
from sensor_msgs.msg import Image, CameraInfo
from message_filters import TimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
from codetiming import Timer
import os
import argparse

from tinynav.tinynav_cpp_bind import pose_graph_solve
from models_trt import LightGlueTRT, Dinov2TRT, SuperPointTRT
import logging
from scipy.ndimage import gaussian_filter
import asyncio
from tf2_ros import TransformBroadcaster
from build_map_node import TinyNavDB
from build_map_node import find_loop, solve_pose_graph
import einops
from build_map_node import OdomPoseRecorder

logger = logging.getLogger(__name__)



def draw_image_match_origin(prev_image: np.ndarray, curr_image: np.ndarray, prev_keypoints: np.ndarray, curr_keypoints: np.ndarray, matches: np.ndarray):
    cv_matches = [cv2.DMatch(_queryIdx=matches[index, 0].item(), _trainIdx=matches[index, 1].item(), _imgIdx=0, _distance=0) for index in range(matches.shape[0])]
    # convert kpts_prev and kpts_curr to cv2.KeyPoint
    cv_kpts_prev = [cv2.KeyPoint(x=prev_keypoints[index, 0].item(), y=prev_keypoints[index, 1].item(), size=20) for index in range(prev_keypoints.shape[0])]
    cv_kpts_curr = [cv2.KeyPoint(x=curr_keypoints[index, 0].item(), y=curr_keypoints[index, 1].item(), size=20) for index in range(curr_keypoints.shape[0])]
    output_image = cv2.drawMatches(prev_image, cv_kpts_prev, curr_image, cv_kpts_curr, cv_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return output_image

def depth_to_cloud(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert depth image to point cloud.
    :param depth: (H, W) depth image.
    :param K: (3, 3) camera intrinsic matrix.
    :return: (N, 3) point cloud in camera coordinates.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.flatten()

    x = (u.flatten() - K[0, 2]) * z / K[0, 0]
    y = (v.flatten() - K[1, 2]) * z / K[1, 1]

    points_3d = np.vstack((x, y, z)).T
    return points_3d[~np.isnan(points_3d).any(axis=1)]

def transform_point_cloud(point_cloud: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform a point cloud with a transformation matrix.
    :param point_cloud: (N, 3) numpy array of points in the point cloud.
    :param T: (4, 4) transformation matrix.
    :return: (N, 3) transformed point cloud.
    """
    assert point_cloud.shape[1] == 3, "Point cloud must be of shape (N, 3)"
    assert T.shape == (4, 4), "Transformation matrix must be of shape (4, 4)"

    # Convert to homogeneous coordinates
    ones = np.ones((point_cloud.shape[0], 1))
    homogeneous_points = np.hstack((point_cloud, ones))
    # Apply transformation
    transformed_points = homogeneous_points @ T.T
    return transformed_points[:, :3]

def compute_cost_map(occupancy_map: np.ndarray, Unknown_cost: float = 15.0, Free_cost: float = 0.0, Occupied_cost: float = 10.0, sigma: float = 5.0) -> np.ndarray:
    x_y_plane = np.max(occupancy_map, axis = 2)
    cost_map = x_y_plane.copy().astype(np.float32)
    cost_map[x_y_plane == 2] = Occupied_cost
    cost_map[x_y_plane == 1] = Free_cost
    cost_map[x_y_plane == 0] = Unknown_cost
    return gaussian_filter(cost_map, sigma=sigma)

class MapNode(Node):
    def __init__(self, tinynav_db_path: str, tinynav_map_path: str, verbose_timer: bool = True):
        """Initialization

        Args:
            tinynav_db_path (str): Directory to store output data.
            tinynav_map_path (str): Directory to load the pre-built map.
            verbose_timer (bool): Whether to use verbose timer output.
        """
        super().__init__('map_node')
        self.logger = logging.getLogger(__name__)
        self.timer_logger = self.logger.info if verbose_timer else self.logger.debug
        self.super_point_extractor = SuperPointTRT()
        self.light_glue_matcher = LightGlueTRT()
        self.dinov2_model = Dinov2TRT()
        self.tinynav_db_path = tinynav_db_path

        self.bridge = CvBridge()

        self.depth_sub = Subscriber(self, Image, '/slam/keyframe_depth')
        self.keyframe_image_sub = Subscriber(self, Image, '/slam/keyframe_image')
        self.keyframe_odom_sub = Subscriber(self, Odometry, '/slam/keyframe_odom')
        self.continuous_odom_sub = self.create_subscription(Odometry, '/slam/odometry', self.continuous_odom_callback, 100)
        self.pose_graph_trajectory_pub = self.create_publisher(Path, "/mapping/pose_graph_trajectory", 10)
        self.relocation_pub = self.create_publisher(Odometry, '/map/relocalization', 10)
        self.current_pose_in_map_pub = self.create_publisher(Odometry, "/mapping/current_pose_in_map", 10)

        # Add stop signal subscription and data saved publisher
        self.localization_stop_sub = self.create_subscription(Bool, '/benchmark/stop', self.localization_stop_callback, 10)
        self.localization_data_saved_pub = self.create_publisher(Bool, '/benchmark/data_saved', 10)
        self.ts = TimeSynchronizer([self.keyframe_image_sub, self.keyframe_odom_sub, self.depth_sub], 1000)
        self.ts.registerCallback(self.keyframe_callback)

        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)
        self.K = None
        self.baseline = None
        self.last_keyframe_image = None
        self.continuous_odom_recorder = OdomPoseRecorder(tinynav_db_path, "localization")

        self.odom = {}
        self.pose_graph_used_pose = {}
        self.relative_pose_constraint = []
        self.last_keyframe_timestamp = None

        self.loop_similarity_threshold = 0.90
        self.loop_top_k = 1

        self.relocalization_threshold = 0.85
        self.relocalization_loop_top_k = 1

        os.makedirs(f"{tinynav_db_path}/nav_temp", exist_ok=True)
        self.nav_temp_db = TinyNavDB(f"{tinynav_db_path}/nav_temp", is_scratch=True)
        self.map_poses = np.load(f"{tinynav_map_path}/poses.npy", allow_pickle=True).item()
        self.map_K = np.load(f"{tinynav_map_path}/intrinsics.npy")
        self.db = TinyNavDB(tinynav_map_path, is_scratch=False)
        self.map_embeddings_idx_to_timestamp = {idx: timestamp for idx, timestamp in enumerate(self.map_poses.keys())}
        self.map_embeddings = np.stack([self.db.get_embedding(timestamp) for idx, timestamp in self.map_embeddings_idx_to_timestamp.items()])
        self.occupancy_map = np.load(f"{tinynav_map_path}/occupancy_grid.npy")
        self.occupancy_map_meta = np.load(f"{tinynav_map_path}/occupancy_meta.npy")
        self.cost_map = compute_cost_map(self.occupancy_map, Unknown_cost=15.0, Free_cost=0.0, Occupied_cost=10.0, sigma=5.0)

        self.relocalization_poses = {}
        self.relocalization_pose_weights = {}
        self.failed_relocalizations = []

        self.T_from_map_to_odom = None

        if os.path.exists(f"{tinynav_map_path}/pois.json"):
            self.pois = json.load(open(f"{tinynav_map_path}/pois.json"))
        else:
            self.pois = {}
        self.poi_index = len(self.pois) - 1
        pois_dict = {}
        for key, value in self.pois.items():
            pois_dict[int(key)] = np.array(value["position"])
        self.pois = pois_dict

        self.poi_pub = self.create_publisher(Odometry, "/mapping/poi", 10)
        self.poi_change_pub = self.create_publisher(Odometry, "/mapping/poi_change", 10)

        self.current_pose_pub = self.create_publisher(Odometry, "/mapping/current_pose", 10)
        self.global_plan_pub = self.create_publisher(Path, '/mapping/global_plan', 10)
        self.target_pose_pub = self.create_publisher(Odometry, "/control/target_pose", 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self._save_completed = False

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

    def localization_stop_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Received benchmark stop signal, starting save process...")
            try:
                self.save_relocalization_poses()
                self.get_logger().info("Localization save completed successfully")

                # Publish save finished signal
                save_finished_msg = Bool()
                save_finished_msg.data = True
                self.localization_data_saved_pub.publish(save_finished_msg)
                self.get_logger().info("Published data save finished signal")

            except Exception as e:
                self.get_logger().error(f"Error during localization save: {e}")
                # Still publish completion signal even if there was an error
                save_finished_msg = Bool()
                save_finished_msg.data = False
                self.localization_data_saved_pub.publish(save_finished_msg)

    def keyframe_callback(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, depth_msg:Image):
        self.keyframe_mapping(keyframe_image_msg, keyframe_odom_msg, depth_msg)
        image = self.bridge.imgmsg_to_cv2(keyframe_image_msg, desired_encoding="mono8")

        keyframe_image_timestamp_ns = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(keyframe_image_msg.header.stamp.nanosec)
        success, _ = self.keyframe_relocalization(keyframe_image_msg.header.stamp, image)
        if success:
            self.compute_transform_from_map_to_odom()

        with Timer(name = "nav path", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            self.try_publish_nav_path(keyframe_image_timestamp_ns)
            # timer or queue for publish the nav path
            # and record the map pose
            # compute the coordinate transform from the map pose to the keyframe pose
            # publish the nav path from the map pose to the keyframe pose with the cost map

    def keyframe_mapping_with_timer(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, depth_msg:Image):
        with Timer(name="Mapping Loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            self.keyframe_mapping(keyframe_image_msg, keyframe_odom_msg, depth_msg)

    def keyframe_mapping(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, depth_msg:Image):
        if self.K is None:
            return
        keyframe_image_timestamp = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(keyframe_image_msg.header.stamp.nanosec)
        keyframe_odom_timestamp = int(keyframe_odom_msg.header.stamp.sec * 1e9) + int(keyframe_odom_msg.header.stamp.nanosec)
        depth_timestamp = int(depth_msg.header.stamp.sec * 1e9) + int(depth_msg.header.stamp.nanosec)
        assert keyframe_image_timestamp == keyframe_odom_timestamp
        assert keyframe_image_timestamp == depth_timestamp
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        odom = msg2np(keyframe_odom_msg)
        image = self.bridge.imgmsg_to_cv2(keyframe_image_msg, desired_encoding="mono8")
        rgb_image_place_holder = einops.repeat(image, "h w -> h w c", c = 3)

        self.nav_temp_db.set_entry(keyframe_image_timestamp, depth = depth, infra1_image = image, rgb_image = rgb_image_place_holder)
        embedding = self.get_embeddings(image)
        self.nav_temp_db.set_entry(keyframe_image_timestamp, embedding = embedding)
        features = asyncio.run(self.super_point_extractor.infer(image))
        self.nav_temp_db.set_entry(keyframe_image_timestamp, features = features)

        if len(self.odom) == 0 and self.last_keyframe_timestamp is None:
            self.odom[keyframe_odom_timestamp] = odom
            self.pose_graph_used_pose[keyframe_odom_timestamp] = odom
        else:
            last_keyframe_odom_pose = self.odom[self.last_keyframe_timestamp]
            T_prev_curr = np.linalg.inv(last_keyframe_odom_pose) @ odom
            self.relative_pose_constraint.append((keyframe_image_timestamp, self.last_keyframe_timestamp, T_prev_curr))
            self.pose_graph_used_pose[keyframe_image_timestamp] = odom
            self.odom[keyframe_image_timestamp] = odom
            def find_loop_and_pose_graph(timestamp):
                    target_embedding = self.nav_temp_db.get_embedding(timestamp)
                    valid_timestamp = [t for t in self.pose_graph_used_pose.keys() if t + 10 * 1e9 < timestamp]
                    valid_embeddings = np.array([self.nav_temp_db.get_embedding(t) for t in valid_timestamp])

                    idx_to_timestamp = {i:t for i, t in enumerate(valid_timestamp)}
                    with Timer(name = "find loop", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        loop_list = find_loop(target_embedding, valid_embeddings, self.loop_similarity_threshold, self.loop_top_k)
                    with Timer(name = "Relative pose estimation", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        for idx, similarity in loop_list:
                            prev_timestamp = idx_to_timestamp[idx]
                            curr_timestamp = timestamp
                            prev_depth, _, prev_features, _, _ = self.nav_temp_db.get_depth_embedding_features_images(prev_timestamp)
                            curr_depth, _, curr_features, _, _ = self.nav_temp_db.get_depth_embedding_features_images(curr_timestamp)
                            prev_matched_keypoints, curr_matched_keypoints, matches = self.match_keypoints(prev_features, curr_features)
                            success, T_prev_curr, _, _, inliers = estimate_pose(prev_matched_keypoints, curr_matched_keypoints, curr_depth, self.K)
                            if success and len(inliers) >= 100:
                                self.relative_pose_constraint.append((curr_timestamp, prev_timestamp, T_prev_curr))
                                print(f"Added loop relative pose constraint: {curr_timestamp} -> {prev_timestamp}")
                    with Timer(name = "solve pose graph", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        self.pose_graph_used_pose = solve_pose_graph(self.pose_graph_used_pose, self.relative_pose_constraint, max_iteration_num = 5)
            find_loop_and_pose_graph(keyframe_image_timestamp)
            self.pose_graph_trajectory_publish(keyframe_image_timestamp)
        self.last_keyframe_timestamp = keyframe_odom_timestamp
        self.last_keyframe_image = image


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

    def relocalize_with_depth(self, keyframe: np.ndarray, keyframe_features: dict, K: np.ndarray | None) -> tuple[bool, np.ndarray, float]:
        if K is None:
            return False, np.eye(4), -np.inf
        query_embedding = self.get_embeddings(keyframe)
        query_embedding_normed = query_embedding / np.linalg.norm(query_embedding)

        idx_and_similarity_array = find_loop(query_embedding_normed, self.map_embeddings, self.relocalization_threshold, self.relocalization_loop_top_k)
        max_similarity = np.max([similarity for _, similarity in idx_and_similarity_array]) if len(idx_and_similarity_array) > 0 else 0
        if len(idx_and_similarity_array) > 0:
            point_3d_in_world_list = []
            point_2d_in_keyframe_list = []
            for idx_in_map, similarity in idx_and_similarity_array:
                timestamp_in_map = self.map_embeddings_idx_to_timestamp[idx_in_map]
                reference_keyframe_pose = self.map_poses[timestamp_in_map]
                reference_depth, _, reference_features, _, _ = self.db.get_depth_embedding_features_images(timestamp_in_map)
                reference_matched_keypoints, keyframe_matched_keypoints, matches = self.match_keypoints(reference_features, keyframe_features)
                if len(matches) >= 50:
                    point_3d_in_world, inliers = self.keypoint_with_depth_to_3d(reference_matched_keypoints, reference_depth, reference_keyframe_pose, self.map_K)
                    point_3d_in_world_list = point_3d_in_world[inliers]
                    point_2d_in_keyframe_list = keyframe_matched_keypoints[inliers]
                else:
                    print(f"not enough matched features to relocalize, {len(matches)} < 50")
                    return False, np.eye(4), -np.inf

            if len(point_3d_in_world_list) > 80:
                point_3d_in_world_list = np.array(point_3d_in_world_list)
                point_2d_in_keyframe_list = np.array(point_2d_in_keyframe_list)

                success, rvec, tvec, inliers = cv2.solvePnPRansac(point_3d_in_world_list, point_2d_in_keyframe_list, self.map_K, None)
                if success and len(inliers) >= 50:
                    R, _ = cv2.Rodrigues(rvec)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = tvec.reshape(3)
                    print(f"relocalization pose : {T}")
                    return True, T, len(inliers) / len(point_2d_in_keyframe_list)
            else:
                print(f"not enough landmarks to relocalize, {len(point_3d_in_world_list)}")
                return False, np.eye(4), -np.inf
        else:
            print(f"not enough similar embeddings to relocalize, {len(idx_and_similarity_array)}, max_similarity : {max_similarity}")
        return False, np.eye(4), -np.inf

    def keypoint_with_depth_to_3d(self, keypoints:np.ndarray, depth:np.ndarray, pose_from_camera_to_world:np.ndarray, K:np.ndarray):
        point_in_camera = []
        inliers = []
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        for kp in keypoints:
            u = int(kp[0])
            v = int(kp[1])
            Z = depth[v, u]
            if Z > 0 and Z < 50:
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                inliers.append(True)
            else:
                X = 0
                Y = 0
                inliers.append(False)
            point_in_camera.append(np.array([X, Y, Z]))
        # shape: (N, 3)
        point_in_camera = np.array(point_in_camera)
        inliers = np.array(inliers)
        rotation = pose_from_camera_to_world[:3, :3]
        translation = pose_from_camera_to_world[:3,3]

        point_in_world = (rotation @ point_in_camera.T).T + translation
        return point_in_world, inliers

    @Timer(name="Relocalization loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms")
    def keyframe_relocalization(self, timestamp, image:np.ndarray) -> tuple[bool, np.ndarray]:
        features = asyncio.run(self.super_point_extractor.infer(image))
        res, pose_in_camera, pose_cov_weight = self.relocalize_with_depth(image, features, self.K)
        if res:
            # publish the relocalization pose for debug
            pose_in_world = np.linalg.inv(pose_in_camera)
            timestamp_ns = int(timestamp.sec * 1e9) + int(timestamp.nanosec)
            self.relocation_pub.publish(np2msg(pose_in_world, timestamp, "world", "camera"))
            self.relocalization_poses[timestamp_ns] = pose_in_world
            self.relocalization_pose_weights[timestamp_ns] = pose_cov_weight
            return True, pose_in_world
        else:
            self.failed_relocalizations.append(timestamp)
            return False, np.eye(4)

    def save_relocalization_poses(self):
        if self._save_completed:
            self.get_logger().info("Relocalization data already saved, skipping duplicate save")
            return

        print("saving localization data...")
        self.continuous_odom_recorder.save_to_disk()

        if len(self.relocalization_poses) == 0:
            self.get_logger().warning("No relocalization poses found - not saving")
            return

        np.save(f"{self.tinynav_db_path}/relocalization_poses.npy", self.relocalization_poses, allow_pickle=True)
        np.save(f"{self.tinynav_db_path}/relocalization_pose_weights.npy", self.relocalization_pose_weights, allow_pickle=True)
        np.save(f"{self.tinynav_db_path}/failed_relocalizations.npy", self.failed_relocalizations, allow_pickle=True)
        np.save(f"{self.tinynav_db_path}/poses.npy", self.pose_graph_used_pose, allow_pickle=True)

        logging.info(f"Saved {len(self.relocalization_poses)} relocalization poses to {self.tinynav_db_path}")
        logging.info(f"Failed relocalizations count: {len(self.failed_relocalizations)}")

        self._save_completed = True

    def destroy_node(self):
        try:
            self.save_relocalization_poses()
            self.nav_temp_db.close()
            self.db.close()
            super().destroy_node()
        except Exception:
            # Ignore errors during destruction as resources may already be freed
            pass


    def compute_transform_from_map_to_odom(self):
        """
        Solve the optmization problem.
        """
        relative_pose_constraint = []
        optimized_parameters = {
            0 : np.eye(4),
            1 : np.eye(4),
        }
        constant_pose_index_dict = { 1: True }
        for timestamp, pose in self.relocalization_poses.items():
            if timestamp in self.pose_graph_used_pose:
                camera_in_map_world = pose
                camera_in_odom_world = self.pose_graph_used_pose[timestamp]
                observation_T_from_map_to_odom =  camera_in_odom_world @ np.linalg.inv(camera_in_map_world)
                weight = self.relocalization_pose_weights[timestamp]

                relative_pose_constraint.append((0, 1, observation_T_from_map_to_odom, weight * np.array([10.0, 10.0, 10.0]), weight * np.array([10.0, 10.0, 10.0])))
        relative_pose_constraint = relative_pose_constraint[-3:]
        optimized_parameters = pose_graph_solve(optimized_parameters, relative_pose_constraint, constant_pose_index_dict, max_iteration_num = 10)
        self.T_from_map_to_odom = optimized_parameters[0]


    def try_publish_nav_path(self, timestamp: int):
        self.get_logger().info(f"try_publish_nav_path, timestamp: {timestamp}")
        if self.T_from_map_to_odom is None:
            self.get_logger().info("Relocalization not successful yet, skip publishing nav path")
            return

        if self.poi_index == -1:
            self.get_logger().info("No POI found, skip publishing nav path")
            return

        if self.poi_index >= len(self.pois):
            self.get_logger().info("All POIs have been visited, skip publishing nav path")
            return

        poi = self.pois[self.poi_index]
        print(f"poi: {poi}")
        poi_pose = np.eye(4)
        poi_pose[:3, 3] = poi
        self.poi_pub.publish(np2msg(poi_pose, self.get_clock().now().to_msg(), "world", "map"))
        # get the pose from the map to the odom
        pose_in_map = np.linalg.inv(self.T_from_map_to_odom) @ self.pose_graph_used_pose[timestamp]
        self.current_pose_in_map_pub.publish(np2msg(pose_in_map, self.get_clock().now().to_msg(), "world", "map"))

        pose_in_map_position = pose_in_map[:3, 3]

        while self.poi_index < len(self.pois):
            poi = self.pois[self.poi_index]
            diff_position_norm = np.linalg.norm(poi[:2] - pose_in_map_position[:2])
            if diff_position_norm < 1.5:
                self.poi_index += 1
                dummy_pose = np.eye(4)
                self.poi_change_pub.publish(np2msg(dummy_pose, self.get_clock().now().to_msg(), "world", "map"))
                continue
            else:
                break

        if self.poi_index >= len(self.pois):
            self.get_logger().info("All POIs have been visited, skip publishing nav path")
            return

        target_poi = self.pois[self.poi_index]
        print(f"current_target_poi_index : {self.poi_index}")
        paths_in_map = self.generate_nav_path_in_map(pose_in_map = pose_in_map, target_poi = target_poi)

        if paths_in_map is not None:
            # use the max_speed to publish the position the robot should be after 5 seconds
            max_speed = 1.0
            if len(paths_in_map) > 1:
                accumulated_distance = 0.0
                start_point = pose_in_map_position[:2]
                target_position = paths_in_map[-1]
                for i in range(len(paths_in_map) - 1):
                    accumulated_distance += np.linalg.norm(paths_in_map[i] - start_point)
                    if accumulated_distance > max_speed * 5:
                        target_position = paths_in_map[i]
                        break
                    else:
                        start_point = paths_in_map[i]
            else:
                target_position = paths_in_map[0]

            target_position_in_map = np.array([target_position[0], target_position[1], 0.0])
            pose_in_origin_odom = self.odom[timestamp]
            T = pose_in_origin_odom @ np.linalg.inv(pose_in_map)
            target_position_in_odom = T[:3, :3] @ target_position_in_map + T[:3, 3]
            dummy_pose = np.eye(4)
            dummy_pose[:3, 3] = target_position_in_odom
            logging.info(f"target_position_in_odom: {target_position_in_odom}")
            self.target_pose_pub.publish(np2msg(dummy_pose, self.get_clock().now().to_msg(), "world", "camera"))
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = "map"
            for x, y in paths_in_map:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0
                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = 0.0
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
                self.global_plan_pub.publish(path_msg)
            self.tf_broadcaster.sendTransform(np2tf(T, self.get_clock().now().to_msg(), "world", "map"))
        else:
            logging.info("No path found in map")

    def generate_nav_path_in_map(self, pose_in_map: np.ndarray, target_poi: np.ndarray) -> np.ndarray|None:
        dummy_poi_pose = np.eye(4)
        dummy_poi_pose[:3, 3] = target_poi
        self.poi_pub.publish(np2msg(dummy_poi_pose, self.get_clock().now().to_msg(), "world", "map"))
        cost_map_origin = self.occupancy_map_meta[:2]
        resolution = self.occupancy_map_meta[3]
        start_idx = np.array([int((pose_in_map[0, 3] - cost_map_origin[0]) / resolution), int((pose_in_map[1, 3] - cost_map_origin[1]) / resolution)], dtype=np.int32)
        goal_idx = np.array([int((target_poi[0] - cost_map_origin[0]) / resolution), int((target_poi[1] - cost_map_origin[1]) / resolution)], dtype=np.int32)
        if start_idx[0] < 0 or start_idx[0] >= self.cost_map.shape[0] or start_idx[1] < 0 or start_idx[1] >= self.cost_map.shape[1] or goal_idx[0] < 0 or goal_idx[0] >= self.cost_map.shape[0] or goal_idx[1] < 0 or goal_idx[1] >= self.cost_map.shape[1]:
            return None
        path = theta_star(self.cost_map, start_idx, goal_idx, obstacles_cost = 10.0)
        if len(path) > 0:
            converted_path = path * resolution + cost_map_origin
            return converted_path
        return None

def main(args=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinynav_db_path", type=str, default="tinynav_temp")
    parser.add_argument("--tinynav_map_path", type=str, required=True)
    parser.add_argument("--verbose_timer", action="store_true", default=True, help="Enable verbose timer output")
    parser.add_argument("--no_verbose_timer", dest="verbose_timer", action="store_false", help="Disable verbose timer output")
    parsed_args, unknown_args = parser.parse_known_args(sys.argv[1:])
    node = MapNode(tinynav_db_path=parsed_args.tinynav_db_path,
                   tinynav_map_path=parsed_args.tinynav_map_path,
                   verbose_timer=parsed_args.verbose_timer)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, map node is shut down")
    except Exception as e:
        logging.exception(f"Error occurred: {e}")
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            logging.exception(f"Error occurred during cleanup: {e}")


if __name__ == '__main__':
    main()
