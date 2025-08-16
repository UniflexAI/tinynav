import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
import numpy as np
import sys

from math_utils import matrix_to_quat, msg2np, np2msg, estimate_pose
from sensor_msgs.msg import Image, CameraInfo
from message_filters import TimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
from codetiming import Timer
import os
import argparse

from tinynav.tinynav_cpp_bind import pose_graph_solve
from models_trt import LightGlueTRT, Dinov2TRT, SuperPointTRT
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from planning_node import run_raycasting_loopy
import logging
from scipy.ndimage import gaussian_filter
import asyncio

logger = logging.getLogger(__name__)


TINYNAV_DB = "tinynav_db"
TINYNAV_TEMP = "tinynav_temp"

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

class MapNode(Node):
    def __init__(self, mapping_mode):
        super().__init__('map_node')
        self.super_point_extractor = SuperPointTRT()
        self.light_glue_matcher = LightGlueTRT()
        self.dinov2_model = Dinov2TRT()

        self.bridge = CvBridge()

        self.disp_sub = Subscriber(self, Image, '/slam/keyframe_disparity')
        self.keyframe_image_sub = Subscriber(self, Image, '/slam/keyframe_image')
        self.keyframe_odom_sub = Subscriber(self, Odometry, '/slam/keyframe_odom')

        self.pose_graph_trajectory_pub = self.create_publisher(Path, "/mapping/pose_graph_trajectory", 10)

        self.relocation_pub = self.create_publisher(Odometry, '/map/relocalization', 10)

        self.project_3d_to_2d_pub = self.create_publisher(Image, "/mapping/project_3d_to_2d", 10)
        self.matches_image_pub = self.create_publisher(Image, "/mapping/keyframe_matches_images", 10)
        self.loop_matches_image_pub = self.create_publisher(Image, "/mapping/loop_matches_images", 10)
        self.relocation_image_pub = self.create_publisher(Image, "/mapping/relocation_images", 10)
        self.current_pose_in_map_pub = self.create_publisher(Odometry, "/mapping/current_pose_in_map", 10)

        self.global_map_pub = self.create_publisher(PointCloud2, "/mapping/global_map", 10)
        self.global_map_timer = self.create_timer(10.0, self.global_map_callback)
        self.point_cloud = None

        self.ts = TimeSynchronizer([self.keyframe_image_sub, self.keyframe_odom_sub, self.disp_sub], 1000)
        self.ts.registerCallback(self.keyframe_callback)

        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)
        self.K = None
        self.baseline = None
        self.last_keyframe_image = None

        self.odom = {}
        self.pose_graph_used_pose = {}
        self.relative_pose_constraint = []
        self.last_keyframe_timestamp = None
        self.features = {}
        self.embeddings = {}

        self.loop_similarity_threshold = 0.90
        self.relocalization_threshold = 0.85
        self.loop_top_k = 1


        self.image_paths = {}
        self.depth_paths = {}
        self.mapping_mode = mapping_mode

        os.makedirs(f"{TINYNAV_DB}/images", exist_ok=True)
        os.makedirs(f"{TINYNAV_DB}/depths", exist_ok=True)
        os.makedirs(f"{TINYNAV_TEMP}/images", exist_ok=True)
        os.makedirs(f"{TINYNAV_TEMP}/depths", exist_ok=True)

        self.map_image_paths = {}
        self.map_depth_paths = {}
        self.map_embeddings = {}
        self.map_features = {}
        self.map_poses = {}
        self.map_K = None
        self.relocalization_poses = {}
        self.relocalization_pose_weights = {}

        self.T_from_map_to_odom = None
        self.pois = []
        self.poi_index = -1

        self.poi_pub = self.create_publisher(Odometry, "/mapping/poi", 10)
        self.poi_change_pub = self.create_publisher(Odometry, "/mapping/poi_change", 10)

        self.current_pose_pub = self.create_publisher(Odometry, "/mapping/current_pose", 10)
        self.global_plan_pub = self.create_publisher(Path, '/mapping/global_plan', 10)
        self.target_pose_pub = self.create_publisher(Odometry, "/control/target_pose", 10)

        self.cost_map = None
        self.occupancy_map = None
        self.occuancy_map_meta = None
        

    def info_callback(self, msg:CameraInfo):
        if self.K is None:
            logger.info("Camera intrinsics received.")
            self.K = np.array(msg.k).reshape(3, 3)
            fx = self.K[0, 0]
            Tx = msg.p[3]
            self.baseline = -Tx / fx
            self.destroy_subscription(self.camera_info_sub)


    def keyframe_callback(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, disp_msg:Image):
        self.keyframe_mapping(keyframe_image_msg, keyframe_odom_msg, disp_msg)
        if not self.mapping_mode:
            image = self.bridge.imgmsg_to_cv2(keyframe_image_msg, desired_encoding="mono8")

            keyframe_image_timestamp_ns = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(keyframe_image_msg.header.stamp.nanosec)
            success, _ = self.keyframe_relocalization(keyframe_image_msg.header.stamp, image)
            if success:
                self.compute_transform_from_map_to_odom()

            with Timer(name = "nav path", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
                self.try_publish_nav_path(keyframe_image_timestamp_ns)

            # timer or queue for publish the nav path

            # and record the map pose 
            # compute the coordinate transform from the map pose to the keyframe pose
            # publish the nav path from the map pose to the keyframe pose with the cost map



    @Timer(name="Mapping Loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms")
    def keyframe_mapping(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, disp_msg:Image):
        if self.K is None:
            return
        keyframe_image_timestamp = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(keyframe_image_msg.header.stamp.nanosec)
        keyframe_odom_timestamp = int(keyframe_odom_msg.header.stamp.sec * 1e9) + int(keyframe_odom_msg.header.stamp.nanosec)
        disp_timestamp = int(disp_msg.header.stamp.sec * 1e9) + int(disp_msg.header.stamp.nanosec)
        assert keyframe_image_timestamp == keyframe_odom_timestamp
        assert keyframe_image_timestamp == disp_timestamp
        disp = self.bridge.imgmsg_to_cv2(disp_msg, desired_encoding="32FC1")
        depth = self.compute_depth_from_disparity(disp, self.K, self.baseline)

        odom = msg2np(keyframe_odom_msg)

        image = self.bridge.imgmsg_to_cv2(keyframe_image_msg, desired_encoding="mono8")

        image_path = f"{TINYNAV_TEMP}/images/{keyframe_image_timestamp}.npy"
        np.save(image_path, image)
        self.image_paths[keyframe_image_timestamp] = image_path

        depth_path = f"{TINYNAV_TEMP}/depths/depth_{keyframe_image_timestamp}.npy"
        np.save(depth_path, depth)
        self.depth_paths[keyframe_image_timestamp] = depth_path

        self.embeddings[keyframe_image_timestamp] = self.get_embeddings(image)
        self.features[keyframe_image_timestamp]  = asyncio.run(self.super_point_extractor.infer(image))

        if len(self.odom) == 0 and self.last_keyframe_timestamp is None:
            self.odom[keyframe_odom_timestamp] = odom
            self.pose_graph_used_pose[keyframe_odom_timestamp] = odom
        else:
            last_keyframe_odom_pose = self.odom[self.last_keyframe_timestamp]
            T_prev_curr = np.linalg.inv(last_keyframe_odom_pose) @ odom
            self.relative_pose_constraint.append((keyframe_image_timestamp, self.last_keyframe_timestamp, T_prev_curr, np.array([10.0, 10.0, 10.0]), np.array([10.0, 10.0, 10.0])))
            self.pose_graph_used_pose[keyframe_image_timestamp] = odom
            self.odom[keyframe_image_timestamp] = odom
            with Timer(name = "pose graph", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
                self.pose_graph_solve()
            with Timer(name = "loop", text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
                self.find_loop(keyframe_image_timestamp, depth)
            self.pose_graph_trajectory_publish(keyframe_image_timestamp)
        self.last_keyframe_timestamp = keyframe_odom_timestamp
        self.last_keyframe_image = image


    def pose_graph_solve(self):
        """
        Solve the bundle adjustment problem.
        """
        min_timestamp = min(self.pose_graph_used_pose.keys())
        constant_pose_index_dict = { min_timestamp : True }
        for curr_timestamp, prev_timestamp, _, _, _ in self.relative_pose_constraint:
            assert curr_timestamp != prev_timestamp, "current timestamp and previous timestamp should be different"
        optimized_camera_poses = pose_graph_solve(self.pose_graph_used_pose, self.relative_pose_constraint, constant_pose_index_dict)
        for timestamp, pose in optimized_camera_poses.items():
            self.pose_graph_used_pose[timestamp] = pose
        return optimized_camera_poses

    def compute_depth_from_disparity(self, disparity:np.ndarray, K:np.ndarray, baseline:float) -> np.ndarray:
        depth = np.zeros_like(disparity)
        depth[disparity > 0] = K[0, 0] * baseline / (disparity[disparity > 0])
        return depth

    def get_embeddings(self, image: np.ndarray) -> np.ndarray:
        processed_image = self.dinov2_model.preprocess_image(image)
        # shape: (1, 768)
        return self.dinov2_model.infer(processed_image)["last_hidden_state"][:, 0, :].squeeze(0)

    def match_keypoints(self, feats0:dict, feats1:dict, image_shape = np.array([640, 360], dtype = np.int64)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        match_result = asyncio.run(self.light_glue_matcher.infer(feats0["kpts"], feats1["kpts"], feats0['descps'], feats1['descps'], image_shape, image_shape))
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

    def publish_projected_3d_to_2d(self, timestamp:int, projected_2d:np.ndarray):
        msg = self.bridge.cv2_to_imgmsg(projected_2d, encoding="bgr8")
        msg.header.stamp.sec = int(timestamp / 1e9)
        msg.header.stamp.nanosec = int(timestamp % 1e9)
        msg.header.frame_id = "camera_frame"
        self.project_3d_to_2d_pub.publish(msg)

    def publish_matches_image(self, timestamp: int, matches_image:np.ndarray):
        msg = self.bridge.cv2_to_imgmsg(matches_image, encoding="bgr8")
        msg.header.stamp.sec = int(timestamp / 1e9)
        msg.header.stamp.nanosec = int(timestamp % 1e9)
        msg.header.frame_id = "camera_frame"
        self.matches_image_pub.publish(msg)

    def publish_loop_matches_image(self, timestamp: int, matches_image:np.ndarray):
        msg = self.bridge.cv2_to_imgmsg(matches_image, encoding="bgr8")
        msg.header.stamp.sec = int(timestamp / 1e9)
        msg.header.stamp.nanosec = int(timestamp % 1e9)
        msg.header.frame_id = "camera_frame"
        self.loop_matches_image_pub.publish(msg)

    def find_loop(self, timestamp: int, depth_image:np.ndarray):
        all_timestamps_ = list(self.pose_graph_used_pose.keys())
        all_timestamps_.sort()
        valid_timestamps = set([t for t in all_timestamps_ if t + 10 * 1e9 < timestamp])
        query_embedding = self.embeddings[timestamp]
        query_embedding_normed = query_embedding / np.linalg.norm(query_embedding)
        max_similarity = 0
        similarity_array = []
        for prev_timestamp in valid_timestamps:
            prev_embedding = self.embeddings[prev_timestamp]
            prev_embedding_normed = prev_embedding / np.linalg.norm(prev_embedding)
            similarity = np.dot(query_embedding_normed, prev_embedding_normed.T).item()
            if similarity > max_similarity:
                max_similarity = similarity
            if similarity > self.loop_similarity_threshold:
                similarity_array.append((prev_timestamp, similarity))
        # sort similarity_array by similarity, larger first
        similarity_array.sort(key=lambda x: x[1], reverse=True)
        if len(similarity_array) > 0:
                query_features = self.features[timestamp]
                query_keypoints_origin = query_features['kpts'].squeeze()
                _ = query_features['descps'].squeeze()
                for prev_timestamp, similarity in similarity_array[:self.loop_top_k]:
                    reference_features = self.features[prev_timestamp]
                    reference_keypoints_origin = reference_features['kpts'].squeeze()
                    _ = reference_features['descps'].squeeze()
                    reference_matched_keypoints, query_matched_keypoints, matches = self.match_keypoints(reference_features, query_features)

                    query_image = np.load(self.image_paths[timestamp])
                    reference_image = np.load(self.image_paths[prev_timestamp])
                    query_image_rgb = cv2.cvtColor(query_image, cv2.COLOR_GRAY2RGB)
                    reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2RGB)
                    image_match_image = draw_image_match_origin(reference_image_rgb, query_image_rgb, reference_keypoints_origin, query_keypoints_origin, matches)
                    cv2.putText(image_match_image, f"similarity: {similarity:.2f}, query_keypoints: {len(query_keypoints_origin)}, reference_keypoints: {len(reference_keypoints_origin)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # cv2.imwrite(f"map/loop_matches_{timestamp}_{prev_timestamp}.png", image_match_image)
                    self.publish_loop_matches_image(timestamp, image_match_image)
                    success, T_prev_curr_from_features, inliers_2d, inliers_3d, inliers = estimate_pose(reference_matched_keypoints, query_matched_keypoints, depth_image, self.K, self.baseline)
                    if success:
                        if len(inliers) >= 100:
                            self.relative_pose_constraint.append((timestamp, prev_timestamp, T_prev_curr_from_features, np.array([10.0, 10.0, 10.0]), np.array([10.0, 10.0, 10.0])))
                            logger.info(f"loop detected with similarity {similarity:.2f}, inliers: {len(inliers)}")
                        else:
                            logger.info(f"similarity {similarity:.2f} not enough inliers to relocalize, {len(inliers)} / {len(matches)}")
                    else:
                        logger.info(f"similarity {similarity:.2f} failed to compute relative pose")
        else:
            logger.info(f"not enough similar embeddings to find loop, {len(similarity_array)}, max_similarity : {max_similarity}")


    def relocalize_with_depth(self, keyframe: np.ndarray, keyframe_features: dict, K: np.ndarray) -> tuple[bool, np.ndarray, float]:
        if K is None:
            return False, np.eye(4), -np.inf
        query_embedding = self.get_embeddings(keyframe)
        query_embedding_normed = query_embedding / np.linalg.norm(query_embedding)

        total_timestamps = list(self.map_embeddings.keys())
        total_timestamps.sort()

        similarity_array = []
        max_similarity = 0
        for prev_timestamp in total_timestamps:
            prev_embedding = self.map_embeddings[prev_timestamp]
            prev_embedding_normed = prev_embedding / np.linalg.norm(prev_embedding)
            similarity = np.dot(query_embedding_normed, prev_embedding_normed.T).item()
            if similarity > max_similarity:
                max_similarity = similarity
            if similarity > self.relocalization_threshold:
                similarity_array.append((prev_timestamp, similarity))
        # sort similarity_array by similarity, larger first
        similarity_array.sort(key=lambda x: x[1], reverse=True)

        if len(similarity_array) > 0:
            point_3d_in_world_list = []
            point_2d_in_keyframe_list = []

            for prev_timestamp, similarity in similarity_array[:self.loop_top_k]:
                reference_keyframe_pose = self.map_poses[prev_timestamp]
                reference_features = self.map_features[prev_timestamp]
                reference_depth = np.load(self.map_depth_paths[prev_timestamp])
                reference_origin_keypoints = reference_features['kpts'].squeeze()
                keyframe_origin_keypoints = keyframe_features['kpts'].squeeze()

                reference_matched_keypoints, keyframe_matched_keypoints, matches = self.match_keypoints(reference_features, keyframe_features)
                if len(matches) > 20:
                    point_3d_in_world, inliers = self.keypoint_with_depth_to_3d(reference_matched_keypoints, reference_depth, reference_keyframe_pose, self.K)
                    point_3d_in_world_list = point_3d_in_world[inliers]
                    point_2d_in_keyframe_list = keyframe_matched_keypoints[inliers]

                    reference_image = np.load(self.map_image_paths[prev_timestamp])
                    reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2RGB)
                    query_image = keyframe
                    query_image_rgb = cv2.cvtColor(query_image, cv2.COLOR_GRAY2RGB)
                    image_match_image = draw_image_match_origin(reference_image_rgb, query_image_rgb, reference_origin_keypoints, keyframe_origin_keypoints, matches)
                    # os.makedirs("map_temp", exist_ok=True)
                    cv2.putText(image_match_image, f"similarity: {similarity:.2f}, query_keypoints: {len(keyframe_origin_keypoints)}, reference_keypoints: {len(reference_origin_keypoints)}, matches: {len(matches)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # cv2.imwrite(f"map_temp/relocation_image_{prev_timestamp}.png", image_match_image)
                    self.relocation_image_pub.publish(self.bridge.cv2_to_imgmsg(image_match_image, encoding="bgr8"))
                else:
                    print(f"not enough matched features to relocalize, {len(point_3d_in_world_list)}")
                    return False, np.eye(4), -np.inf

            if len(point_3d_in_world_list) > 80:
                point_3d_in_world_list = np.array(point_3d_in_world_list)
                point_2d_in_keyframe_list = np.array(point_2d_in_keyframe_list)

                success, rvec, tvec, inliers = cv2.solvePnPRansac(point_3d_in_world_list, point_2d_in_keyframe_list, K, None)
                if success and len(inliers) >= 50:
                    R, _ = cv2.Rodrigues(rvec)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = tvec.reshape(3)
                    print(f"relocalization pose : {T}")
                    return True, T, len(inliers) / len(keyframe_origin_keypoints)
            else:
                print(f"not enough landmarks to relocalize, {len(point_3d_in_world_list)}")
                return False, np.eye(4), -np.inf
        else:
            print(f"not enough similar embeddings to relocalize, {len(similarity_array)}, max_similarity : {max_similarity}")
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
            return False, np.eye(4)

    def load_mapping(self):
        self.map_embeddings = np.load(f"{TINYNAV_DB}/embedding.npy", allow_pickle=True).item()
        self.map_features = np.load(f"{TINYNAV_DB}/features.npy", allow_pickle=True).item()
        print(f"len of features: {len(self.map_features)}")
        self.map_depth_paths = np.load(f"{TINYNAV_DB}/depth_paths.npy", allow_pickle=True).item()
        self.map_image_paths = np.load(f"{TINYNAV_DB}/image_paths.npy", allow_pickle=True).item()
        print(f"len of map_image_paths: {len(self.map_image_paths)}")

        for timestamp, features in self.map_features.items():
            assert timestamp in self.map_image_paths, f"timestamp {timestamp} not in map_image_paths"

        self.map_poses = np.load(f"{TINYNAV_DB}/poses.npy", allow_pickle=True).item()
        self.map_K = np.load(f"{TINYNAV_DB}/intrinsics.npy", allow_pickle=True)
        self.pois = np.loadtxt(f"{TINYNAV_DB}/pois.txt").reshape(-1, 3)
        self.poi_index = 0 if len(self.pois) > 0 else -1

        # genereate point cloud from the depths
        self.point_cloud = self.genereate_point_cloud_from_depths()

        # genereate the cost map and occupancy map
        self.occupancy_map = np.load(f"{TINYNAV_DB}/occupancy_map.npy")
        self.occupancy_map_meta = np.load(f"{TINYNAV_DB}/occupancy_map_meta.npy")
        x_y_plane = np.max(self.occupancy_map, axis = 2)
        self.cost_map = x_y_plane.copy().astype(np.float32)
        sigma = 5.0
        self.cost_map[x_y_plane == 2] = 10.0
        self.cost_map[x_y_plane == 1] = 0.0
        self.cost_map[x_y_plane == 0] = 15.0
        self.cost_map = gaussian_filter(self.cost_map, sigma=sigma)

    def save_mapping(self):
        if not os.path.exists(TINYNAV_DB):
            os.mkdir(TINYNAV_DB)
        # self.landmark_tracker.save_to_dir("mapping")
        np.save(f"{TINYNAV_DB}/embedding.npy", self.embeddings)
        np.save(f"{TINYNAV_DB}/features.npy", self.features)
        for timestamp, depth_path in self.depth_paths.items():
            new_depth_path = f"{TINYNAV_DB}/depths/{timestamp}.npy"
            np.save(new_depth_path, np.load(depth_path))
            self.depth_paths[timestamp] = new_depth_path
        np.save(f"{TINYNAV_DB}/depth_paths.npy", self.depth_paths, allow_pickle=True)
        for timestamp, rgb_path in self.image_paths.items():
            new_rgb_path = f"{TINYNAV_DB}/images/{timestamp}.npy"
            np.save(new_rgb_path, np.load(rgb_path))
            self.image_paths[timestamp] = new_rgb_path
        np.save(f"{TINYNAV_DB}/image_paths.npy", self.image_paths, allow_pickle=True)
        np.save(f"{TINYNAV_DB}/poses.npy", self.pose_graph_used_pose)
        np.save(f"{TINYNAV_DB}/intrinsics.npy", self.K)

        logging.info("save map into mapping directory")
        self.generate_occupancy_map()

    def on_shutdown(self):
        if self.mapping_mode:
            self.save_mapping()

    def genereate_point_cloud_from_depths(self):

        point_clouds = []
        depth_paths = self.depth_paths
        K = self.K
        if not self.mapping_mode:
            depth_paths = self.map_depth_paths
            K = self.map_K
        if K is None:
            return None
        for timestamp, depth_path in depth_paths.items():
            depth = np.load(depth_path)
            point_cloud_in_camera = depth_to_cloud(depth, K)
            # downsample the point cloud to reduce the number of points
            point_cloud_in_camera = point_cloud_in_camera[::24]
            # filter point_cloud distance larger then 3m
            dist = np.linalg.norm(point_cloud_in_camera, axis=1)
            valid_indices = dist < 3.0
            point_cloud_in_camera_filtered = point_cloud_in_camera[valid_indices]

            if not self.mapping_mode:
                pose = self.map_poses[timestamp]
            else:
                pose = self.pose_graph_used_pose[timestamp]
            # transform point cloud to world frame
            point_cloud_in_world = transform_point_cloud(point_cloud_in_camera_filtered, pose)
            point_clouds.append(point_cloud_in_world)

        if len(point_clouds):
            return np.vstack(point_clouds)
        else:
            print("No point clouds found in depth paths")
            return None

    def global_map_callback(self):
        if not self.mapping_mode and self.point_cloud is not None:
            self.publish_global_map(self.point_cloud)
        else:
            point_clouds = self.genereate_point_cloud_from_depths()
            if point_clouds is not None:
                self.publish_global_map(point_clouds)

    def publish_global_map(self, point_cloud:np.ndarray):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'world'
        cloud = pc2.create_cloud_xyz32(header, point_cloud.tolist())
        self.global_map_pub.publish(cloud)
        print(f"Published global map with {len(point_cloud)} points")

    def generate_occupancy_map(self):
        """
            Genereate a occupancy grid map from the depth images.
            The occupancy grid map is a 3D grid with the following values:
             0 : Unknown
             1 : Free
             2 : Occupied
        """
        raycast_shape = (100, 100, 20)
        resolution = 0.05
        step = 10
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        global_grid = None
        global_origin = None
        for timestamp, odom_pose in self.pose_graph_used_pose.items():
            depth = np.load(self.depth_paths[timestamp])
            odom_translation = odom_pose[:3, 3]
            local_origin = odom_translation - 0.5 * np.array(raycast_shape) * resolution
            local_grid = run_raycasting_loopy(depth, odom_pose, raycast_shape, fx, fy, cx, cy, local_origin, step, resolution, True)
            if global_grid is None:
                global_grid = local_grid
                global_origin = local_origin
            else:
                global_grid, global_origin = merge_grids(global_grid, global_origin, local_grid, local_origin, resolution)
        grid_type = np.zeros_like(global_grid, dtype=np.uint8)
        grid_type[global_grid > 0] = 2  # Occupied
        grid_type[global_grid < 0] = 1  # Free
        x_y_plane = np.max(grid_type, axis=2)
        print(f"x_y_plane type counts - Occupied: {np.sum(x_y_plane == 2)}, Free: {np.sum(x_y_plane == 1)}, Unknown: {np.sum(x_y_plane == 0)}")
        # remapping type to 0 - 255 for display
        x_y_plane_image = np.zeros_like(x_y_plane, dtype=np.float32)
        x_y_plane_image[x_y_plane == 2] = 1.0
        x_y_plane_image[x_y_plane == 1] = 0.5
        x_y_plane_image = (x_y_plane_image * 255).astype(np.uint8)
        cv2.imwrite(f"{TINYNAV_DB}/occupancy_map.png", x_y_plane_image)
        meta_info = np.array([global_origin[0], global_origin[1], global_origin[2], resolution], dtype=np.float32)
        np.save(f"{TINYNAV_DB}/occupancy_map_meta.npy", meta_info)
        np.save(f"{TINYNAV_DB}/occupancy_map.npy", grid_type)
        print("occupancy map generated")

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
        optimized_parameters = pose_graph_solve(optimized_parameters, relative_pose_constraint, constant_pose_index_dict)
        self.T_from_map_to_odom = optimized_parameters[0]


    def try_publish_nav_path(self, timestamp: int):
        logger.info(f"try_publish_nav_path, timestamp: {timestamp}")
        if self.T_from_map_to_odom is None:
            logger.info("Relocalization not successful yet, skip publishing nav path")
            return

        if self.poi_index == -1:
            logger.info("No POI found, skip publishing nav path")
            return

        if self.poi_index >= len(self.pois):
            logger.info("All POIs have been visited, skip publishing nav path")
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
            logger.info("All POIs have been visited, skip publishing nav path")
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
            path_msg.header.frame_id = "world"
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
        else:
            logging.info("No path found in map")

    def generate_nav_path_in_map(self, pose_in_map: np.ndarray, target_poi: np.ndarray) -> np.ndarray:
        dummy_poi_pose = np.eye(4)
        dummy_poi_pose[:3, 3] = target_poi
        self.poi_pub.publish(np2msg(dummy_poi_pose, self.get_clock().now().to_msg(), "world", "map"))
        cost_map_origin = self.occupancy_map_meta[:2]
        resolution = self.occupancy_map_meta[3]
        start_idx = np.array([int((pose_in_map[0, 3] - cost_map_origin[0]) / resolution), int((pose_in_map[1, 3] - cost_map_origin[1]) / resolution)], dtype=np.int32)
        goal_idx = np.array([int((target_poi[0] - cost_map_origin[0]) / resolution), int((target_poi[1] - cost_map_origin[1]) / resolution)], dtype=np.int32)

        if start_idx[0] < 0 or start_idx[0] >= self.cost_map.shape[0] or start_idx[1] < 0 or start_idx[1] >= self.cost_map.shape[1] or goal_idx[0] < 0 or goal_idx[0] >= self.cost_map.shape[0] or goal_idx[1] < 0 or goal_idx[1] >= self.cost_map.shape[1]:
            return None
        path = A_star(self.cost_map, start_idx, goal_idx, obstacles_cost = 10.0)
        if len(path) > 0:
            converted_path = path * resolution + cost_map_origin
            return converted_path
        return None

 


def reconstruct_path(came_from: dict, current:np.ndarray) -> np.ndarray:    
    """
    Reconstructs the path from the start to the goal.
    :param came_from: dict, mapping of nodes to their predecessors
    :param current: tuple, the current node
    :return: list of tuples representing the path
    """
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    return np.array(path[::-1])


def A_star(cost_map:np.ndarray, start:np.ndarray, goal:np.ndarray, obstacles_cost: float) -> np.ndarray:
    """
    A* algorithm to find the path from start to goal in the cost map.
    parameters: 
        cost_map: np.ndarray (H, W)
        start: tuple[int, int], x_idx, y_idx
        goal: tuple[int, int], x_idx, y_idx
    returns: list of tuples representing the path from start to goal
    If no path is found, returns an empty list.
    0 - unknown, 0.5 - free, 1.0 - occupied
    """

    from queue import PriorityQueue
    import numpy as np
    start = tuple(start.flatten()) if isinstance(start, np.ndarray) else start
    goal = tuple(goal.flatten()) if isinstance(goal, np.ndarray) else goal

    open_set = PriorityQueue()
    open_set.put((cost_map[start], start))

    def heuristic(start, goal):
        return np.linalg.norm(np.array(start) - np.array(goal))
    came_from = {}
    g_score = {start: cost_map[start]}
    f_score = {start: heuristic(start, goal)}
    visited = set([start])

    while not open_set.empty():
        current = open_set.get()[1]
        visited.remove(current)

        if current == goal:
            return reconstruct_path(came_from, current)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < cost_map.shape[0] and
                        0 <= neighbor[1] < cost_map.shape[1] and cost_map[neighbor] < obstacles_cost and neighbor not in visited):
                    visited.add(neighbor)
                    tentative_g_score = g_score[current] + cost_map[neighbor]
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        open_set.put((f_score[neighbor], neighbor))
    return []

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def main(args=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", default=False, type=str2bool)
    parsed_args, unknown_args = parser.parse_known_args(sys.argv[1:])
    print(f"Map Node Mode : {parsed_args.mapping}")
    node = MapNode(mapping_mode=parsed_args.mapping)
    if not parsed_args.mapping:
        node.load_mapping()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if parsed_args.mapping:
            node.on_shutdown()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
