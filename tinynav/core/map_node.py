import rclpy
import os
import time
from datetime import datetime
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool, String
import numpy as np
import sys
import json

import heapq
from tinynav.core.math_utils import matrix_to_quat, msg2np, np2msg, estimate_pose, np2tf
from sensor_msgs.msg import Image, CameraInfo
from message_filters import TimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
from codetiming import Timer
import argparse

from tinynav.tinynav_cpp_bind import pose_graph_solve
from tinynav.core.models_trt import LightGlueTRT, Dinov2TRT, SuperPointTRT, YOLO11TRT
import logging
import asyncio
from tf2_ros import TransformBroadcaster
from tinynav.core.build_map_node import TinyNavDB
from tinynav.core.build_map_node import find_loop, solve_pose_graph
import einops
from tinynav.core.build_map_node import OdomPoseRecorder
logger = logging.getLogger(__name__)

def _pose_delta(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Return translation delta (m) and rotation delta (deg) between two 4x4 poses."""
    dT = np.linalg.inv(T_a) @ T_b
    trans = float(np.linalg.norm(dT[:3, 3]))
    trace = float(np.clip((np.trace(dT[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    rot_deg = float(np.degrees(np.arccos(trace)))
    return trans, rot_deg


class RetrievalDebugRecorder:
    def __init__(self, root_dir: str, enabled: bool, save_images: bool, save_every_n_keyframes: int, max_samples: int):
        self.enabled = enabled
        self.save_images = save_images
        self.save_every_n_keyframes = max(1, int(save_every_n_keyframes))
        self.max_samples = max(1, int(max_samples))
        self.sample_count = 0
        self.keyframe_counter = 0
        self.session_dir = None
        self.index_jsonl_path = None
        if enabled:
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(root_dir, session_name)
            os.makedirs(self.session_dir, exist_ok=True)
            self.index_jsonl_path = os.path.join(self.session_dir, "index.jsonl")
            logger.info(f"[retrieval-debug] session dir: {self.session_dir}")

    def should_save(self) -> bool:
        if not self.enabled:
            return False
        if self.sample_count >= self.max_samples:
            return False
        self.keyframe_counter += 1
        return (self.keyframe_counter % self.save_every_n_keyframes) == 0

    def save_sample(self, timestamp_ns: int, query_image: np.ndarray, retrieved_images: list, sample_meta: dict):
        if not self.enabled:
            return
        sample_name = f"sample_{self.sample_count:06d}_{timestamp_ns}"
        sample_dir = os.path.join(self.session_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)
        if self.save_images:
            cv2.imwrite(os.path.join(sample_dir, "query.png"), query_image)
            for i, image in enumerate(retrieved_images):
                if image is not None:
                    cv2.imwrite(os.path.join(sample_dir, f"retrieved_{i:03d}.png"), image)
        sample_json_path = os.path.join(sample_dir, "sample.json")
        with open(sample_json_path, "w", encoding="utf-8") as f:
            json.dump(sample_meta, f, ensure_ascii=True, indent=2)
        with open(self.index_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sample_id": sample_name,
                "query_timestamp_ns": int(timestamp_ns),
                "sample_json": sample_json_path,
            }, ensure_ascii=True) + "\n")
        self.sample_count += 1



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

def heuristic(start, goal, resolution):
    vec_start = np.array(start)
    vec_goal = np.array(goal)
    return np.linalg.norm((vec_start - vec_goal) * resolution) + 20 * np.abs(vec_start[2] - vec_goal[2]) * resolution

def reconstruct_path_sdf(parent:dict, current:tuple):
    path = []
    while current in parent:
        path.append(current)
        if current == parent[current]:
            break
        current = parent[current]
    return path[::-1]

def search_close_to_sdf_map(start_index:tuple, sdf_map:np.ndarray, occupancy_map:np.ndarray, stop_distance:np.ndarray):
    start_index = tuple(start_index.flatten()) if isinstance(start_index, np.ndarray) else start_index
    open_heap = [(sdf_map[start_index], start_index)]
    open_heap_set = set()
    open_heap_set.add(start_index)
    parent = {start_index: start_index}
    visited = set()
    while len(open_heap) > 0:
        current_sdf, current = heapq.heappop(open_heap)
        open_heap_set.remove(current)
        visited.add(current)
        if current_sdf < stop_distance:
            return reconstruct_path_sdf(parent, current)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if (0 <= neighbor[0] < sdf_map.shape[0] and
                            0 <= neighbor[1] < sdf_map.shape[1] and
                            0 <= neighbor[2] < sdf_map.shape[2]):
                        if neighbor not in open_heap_set and neighbor not in visited and occupancy_map[neighbor] != 2:
                            open_heap_set.add(neighbor)
                            heapq.heappush(open_heap, (sdf_map[neighbor], neighbor))
                            parent[neighbor] = current
    return []

def search_within_sdf_map( start:tuple, goal:tuple, sdf_map:np.ndarray, occupancy_map:np.ndarray, resolution: float):
    start = tuple(start.flatten()) if isinstance(start, np.ndarray) else start
    goal = tuple(goal.flatten()) if isinstance(goal, np.ndarray) else goal
    sdf_bins = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    def get_queue_index(sdf_value: float) -> int:
        for idx, threshold in enumerate(sdf_bins):
            if sdf_value < threshold:
                return idx
        return len(sdf_bins)

    open_heaps = [[] for _ in range(len(sdf_bins) + 1)]
    open_sets = [set() for _ in range(len(sdf_bins) + 1)]
    start_queue_idx = get_queue_index(float(sdf_map[start]))
    heapq.heappush(open_heaps[start_queue_idx], (heuristic(start, goal, resolution), start))
    open_sets[start_queue_idx].add(start)
    parent = {start: start}
    visited = set()

    while True:
        queue_idx = -1
        for i, q in enumerate(open_heaps):
            if len(q) > 0:
                queue_idx = i
                break
        if queue_idx == -1:
            break

        current_cost, current = heapq.heappop(open_heaps[queue_idx])
        open_sets[queue_idx].remove(current)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return reconstruct_path_sdf(parent, current)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if (0 <= neighbor[0] < sdf_map.shape[0] and
                            0 <= neighbor[1] < sdf_map.shape[1] and
                            0 <= neighbor[2] < sdf_map.shape[2]):
                        if neighbor in visited or occupancy_map[neighbor] == 2:
                            continue
                        neighbor_sdf = float(sdf_map[neighbor])
                        neighbor_queue_idx = get_queue_index(neighbor_sdf)
                        if neighbor in open_sets[neighbor_queue_idx]:
                            continue
                        open_sets[neighbor_queue_idx].add(neighbor)
                        heapq.heappush(
                            open_heaps[neighbor_queue_idx],
                            (heuristic(neighbor, goal, resolution), neighbor),
                        )
                        if neighbor not in parent:
                            parent[neighbor] = current
    return []

class MapNode(Node):
    def __init__(
        self,
        tinynav_db_path: str,
        tinynav_map_path: str,
        verbose_timer: bool = True,
        debug_retrieval_dataset: bool = False,
        debug_retrieval_dir: str = "tinynav_temp/retrieval_debug",
        debug_retrieval_topk: int = 3,
        debug_save_images: bool = True,
        debug_save_every_n_keyframes: int = 5,
        debug_max_samples: int = 10000,
    ):
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
        self.yolo11_model = YOLO11TRT()
        self.tinynav_db_path = tinynav_db_path

        self.bridge = CvBridge()

        # subs
        self.depth_sub = Subscriber(self, Image, '/slam/keyframe_depth')
        self.keyframe_image_sub = Subscriber(self, Image, '/slam/keyframe_image')
        self.keyframe_odom_sub = Subscriber(self, Odometry, '/slam/keyframe_odom')
        self.continuous_odom_sub = self.create_subscription(Odometry, '/slam/odometry', self.continuous_odom_callback, 100)
        self.pois_sub = self.create_subscription(String, '/mapping/cmd_pois', self.pois_callback, 10)

        # pubs
        self.pose_graph_trajectory_pub = self.create_publisher(Path, "/mapping/pose_graph_trajectory", 10)
        self.relocation_pub = self.create_publisher(Odometry, '/map/relocalization', 10)
        self.current_pose_in_map_pub = self.create_publisher(Odometry, "/mapping/current_pose_in_map", 10)

        # Add stop signal subscription and data saved publisher
        self.localization_stop_sub = self.create_subscription(Bool, '/benchmark/stop', self.localization_stop_callback, 10)
        self.localization_data_saved_pub = self.create_publisher(Bool, '/benchmark/data_saved', 10)
        self.ts = TimeSynchronizer([self.keyframe_image_sub, self.keyframe_odom_sub, self.depth_sub], 10)
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
        self.relocalization_loop_top_k = 3
        self.debug_retrieval_topk = max(1, int(debug_retrieval_topk))
        self.false_case_odom_trans_m = 0.5
        self.false_case_odom_rot_deg = 8.0
        self.false_case_map_trans_m = 2.0
        self.false_case_map_rot_deg = 25.0

        os.makedirs(f"{tinynav_db_path}/nav_temp", exist_ok=True)
        self.nav_temp_db = TinyNavDB(f"{tinynav_db_path}/nav_temp", is_scratch=True)
        self.map_poses = np.load(f"{tinynav_map_path}/poses.npy", allow_pickle=True).item()
        self.map_K = np.load(f"{tinynav_map_path}/intrinsics.npy")
        self.db = TinyNavDB(tinynav_map_path, is_scratch=False)
        self.map_embeddings_idx_to_timestamp = {idx: timestamp for idx, timestamp in enumerate(self.map_poses.keys())}
        self.map_embeddings = np.stack([self.db.get_embedding(timestamp) for idx, timestamp in self.map_embeddings_idx_to_timestamp.items()])
        self.occupancy_map = np.load(f"{tinynav_map_path}/occupancy_grid.npy")
        self.occupancy_map_meta = np.load(f"{tinynav_map_path}/occupancy_meta.npy")
        self.sdf_map = np.load(f"{tinynav_map_path}/sdf_map.npy")

        print(f"sdf_map.shape: {self.sdf_map.shape}")
        print(f"occupancy_map.shape: {self.occupancy_map.shape}")

        self.relocalization_poses = {}
        self.relocalization_pose_weights = {}
        self.failed_relocalizations = []

        self.T_from_map_to_odom = None

        self.pois = {}
        self.poi_index = -1
        self._nav_completed = False
        self._leg_initial_length: float | None = None
        self._leg_start_time: float | None = None
        self._speed_estimate: float | None = None

        self.poi_pub = self.create_publisher(Odometry, "/mapping/poi", 10)
        self.poi_change_pub = self.create_publisher(Odometry, "/mapping/poi_change", 10)
        self.nav_done_pub = self.create_publisher(Bool, '/mapping/nav_done', 10)
        self.nav_progress_pub = self.create_publisher(String, '/mapping/nav_progress', 10)

        self.current_pose_pub = self.create_publisher(Odometry, "/mapping/current_pose", 10)
        self.global_plan_pub = self.create_publisher(Path, '/mapping/global_plan', 10)
        self.target_pose_pub = self.create_publisher(Odometry, "/control/target_pose", 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self._save_completed = False
        self.retrieval_debug = RetrievalDebugRecorder(
            root_dir=debug_retrieval_dir,
            enabled=debug_retrieval_dataset,
            save_images=debug_save_images,
            save_every_n_keyframes=debug_save_every_n_keyframes,
            max_samples=debug_max_samples,
        )
        self._map_yolo_labels_cache = {}

    def _infer_yolo_labels(self, image: np.ndarray) -> tuple[set[int], dict[int, float]]:
        if callable(image):
            image = image()
        if image is None or not isinstance(image, np.ndarray):
            return set(), {}
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        det = asyncio.run(self.yolo11_model.infer(image))
        labels = set()
        max_scores = {}
        classes = det["classes"].tolist()
        scores = det["scores"].tolist()
        for cls, score in zip(classes, scores):
            cls_i = int(cls)
            score_f = float(score)
            labels.add(cls_i)
            prev = max_scores.get(cls_i, -1.0)
            if score_f > prev:
                max_scores[cls_i] = score_f
        return labels, max_scores

    def _filter_relocalization_candidates_by_label(
        self,
        query_labels: set[int],
        query_label_scores: dict[int, float],
        idx_and_similarity_array: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        if query_label_scores.get(6, 0.0) <= 0.7:
            return idx_and_similarity_array

        query_base_labels = {
            l for l in query_labels
            if 0 <= l <= 4 and query_label_scores.get(l, 0.0) > 0.7
        }
        filtered = []
        for idx_in_map, similarity in idx_and_similarity_array:
            timestamp_in_map = self.map_embeddings_idx_to_timestamp[idx_in_map]
            if timestamp_in_map in self._map_yolo_labels_cache:
                ref_labels, _ = self._map_yolo_labels_cache[timestamp_in_map]
            else:
                _, _, _, _, infra1_loader = self.db.get_depth_embedding_features_images(timestamp_in_map)
                ref_labels, ref_scores = self._infer_yolo_labels(infra1_loader)
                self._map_yolo_labels_cache[timestamp_in_map] = (ref_labels, ref_scores)

            if 6 not in ref_labels:
                continue
            if len(query_base_labels) > 0 and len(query_base_labels.intersection(ref_labels)) == 0:
                continue
            filtered.append((idx_in_map, similarity))
        return filtered

    def pois_callback(self, msg: String):
        self.get_logger().info("Received POIs from planner: " + msg.data)
        try:
            self.pois = json.loads(msg.data)

            pois_dict = {}
            keys = sorted([int (key) for key in self.pois.keys()])
            for index, key in enumerate(keys):
                pois_dict[index] = np.array(self.pois[str(key)]["position"])
            self.pois = pois_dict

            if not self.pois:
                self.poi_index = -1
                # Signal planning_node to clear target_pose so it stops publishing paths
                dummy_pose = np.eye(4)
                self.poi_change_pub.publish(np2msg(dummy_pose, self.get_clock().now().to_msg(), "world", "map"))
                self.get_logger().info("POIs cleared, navigation cancelled")
                return

            self.poi_index = min(0, len(self.pois) - 1)
            self._nav_completed = False
            self._leg_initial_length = None
            self._leg_start_time = None
            self._speed_estimate = None
            self.get_logger().info(f"Parsed POIs: {self.pois}")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse POIs JSON: {e}")
            self.pois = {}

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
        success, pose_in_world = self.keyframe_relocalization(keyframe_image_msg.header.stamp, image)
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
        odom, _ = msg2np(keyframe_odom_msg)
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
                                #print(f"Added loop relative pose constraint: {curr_timestamp} -> {prev_timestamp}")
                    with Timer(name = "solve pose graph", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        self.pose_graph_used_pose = solve_pose_graph(self.pose_graph_used_pose, self.relative_pose_constraint, max_iteration_num = 5)
            #find_loop_and_pose_graph(keyframe_image_timestamp)
            #self.pose_graph_trajectory_publish(keyframe_image_timestamp)
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

    def relocalize_with_depth(self, keyframe: np.ndarray, keyframe_features: dict, K: np.ndarray | None) -> tuple[bool, np.ndarray, float, dict]:
        if K is None:
            return False, np.eye(4), -np.inf, {"retrieved": [], "pnp_success": False, "inlier_count": 0, "inlier_ratio": 0.0}
        query_embedding = self.get_embeddings(keyframe)
        query_embedding_normed = query_embedding / np.linalg.norm(query_embedding)

        idx_and_similarity_array = find_loop(query_embedding_normed, self.map_embeddings, self.relocalization_threshold, self.relocalization_loop_top_k)
        query_labels, query_label_scores = self._infer_yolo_labels(keyframe)
        idx_and_similarity_array = self._filter_relocalization_candidates_by_label(query_labels, query_label_scores, idx_and_similarity_array)
        idx_and_similarity_array = idx_and_similarity_array[: self.debug_retrieval_topk]
        retrieved_debug = []
        for idx_in_map, similarity in idx_and_similarity_array:
            timestamp_in_map = int(self.map_embeddings_idx_to_timestamp[idx_in_map])
            retrieved_debug.append({
                "idx_in_map": int(idx_in_map),
                "timestamp_ns": timestamp_in_map,
                "similarity": float(similarity),
            })

        max_similarity = np.max([similarity for _, similarity in idx_and_similarity_array]) if len(idx_and_similarity_array) > 0 else 0
        if len(idx_and_similarity_array) > 0:
            point_3d_in_world_list = []
            point_2d_in_keyframe_list = []
            pnp_success = False
            inlier_count = 0
            inlier_ratio = 0.0
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

            if len(point_3d_in_world_list) > 80:
                point_3d_in_world_list = np.array(point_3d_in_world_list)
                point_2d_in_keyframe_list = np.array(point_2d_in_keyframe_list)

                success, rvec, tvec, inliers = cv2.solvePnPRansac(point_3d_in_world_list, point_2d_in_keyframe_list, self.map_K, None)
                pnp_success = bool(success)
                inlier_count = int(len(inliers)) if inliers is not None else 0
                inlier_ratio = float(inlier_count / max(1, len(point_2d_in_keyframe_list)))
                if success and len(inliers) >= 50:
                    R, _ = cv2.Rodrigues(rvec)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = tvec.reshape(3)
                    print(f"relocalization pose : {T}")
                    return True, T, inlier_ratio, {
                        "retrieved": retrieved_debug,
                        "pnp_success": pnp_success,
                        "inlier_count": inlier_count,
                        "inlier_ratio": inlier_ratio,
                    }
            else:
                print(f"not enough landmarks to relocalize, {len(point_3d_in_world_list)}")
                return False, np.eye(4), -np.inf, {
                    "retrieved": retrieved_debug,
                    "pnp_success": pnp_success,
                    "inlier_count": inlier_count,
                    "inlier_ratio": inlier_ratio,
                }
        else:
            print(f"not enough similar embeddings to relocalize, {len(idx_and_similarity_array)}, max_similarity : {max_similarity}")
        return False, np.eye(4), -np.inf, {"retrieved": retrieved_debug, "pnp_success": False, "inlier_count": 0, "inlier_ratio": 0.0}

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
        res, pose_in_camera, pose_cov_weight, debug_info = self.relocalize_with_depth(image, features, self.K)
        timestamp_ns = int(timestamp.sec * 1e9) + int(timestamp.nanosec)
        if res:
            # publish the relocalization pose for debug
            pose_in_world = np.linalg.inv(pose_in_camera)
            self.relocation_pub.publish(np2msg(pose_in_world, timestamp, "world", "camera"))
            self.relocalization_poses[timestamp_ns] = pose_in_world
            self.relocalization_pose_weights[timestamp_ns] = pose_cov_weight
            self._maybe_save_retrieval_debug_sample(timestamp_ns, image, pose_in_world, True, debug_info)
            return True, pose_in_world
        else:
            self.failed_relocalizations.append(timestamp)
            self._maybe_save_retrieval_debug_sample(timestamp_ns, image, np.eye(4), False, debug_info)
            return False, np.eye(4)

    def _maybe_save_retrieval_debug_sample(self, timestamp_ns: int, query_image: np.ndarray, map_pose_estimate: np.ndarray, relocalization_success: bool, debug_info: dict):
        if not self.retrieval_debug.should_save():
            return
        if timestamp_ns not in self.pose_graph_used_pose:
            self.get_logger().warning(f"[retrieval-debug] missing odom pose for timestamp={timestamp_ns}, skipping sample")
            return
        odom_pose = self.pose_graph_used_pose[timestamp_ns]
        odom_trans, odom_rot = _pose_delta(np.eye(4), odom_pose)
        map_trans, map_rot = _pose_delta(np.eye(4), map_pose_estimate)
        false_case = (
            odom_trans <= self.false_case_odom_trans_m
            and odom_rot <= self.false_case_odom_rot_deg
            and (map_trans >= self.false_case_map_trans_m or map_rot >= self.false_case_map_rot_deg)
        )

        retrieved_images = []
        retrieved_timestamps = []
        similarities = []
        for item in debug_info.get("retrieved", []):
            retrieved_timestamps.append(int(item["timestamp_ns"]))
            similarities.append(float(item["similarity"]))
            _, _, _, _, infra1_loader = self.db.get_depth_embedding_features_images(item["timestamp_ns"])
            ref_img = infra1_loader() if infra1_loader is not None else None
            retrieved_images.append(ref_img)

        sample_meta = {
            "query_timestamp_ns": int(timestamp_ns),
            "retrieved_timestamps_ns": retrieved_timestamps,
            "similarities": similarities,
            "selected_idx": -1,
            "pnp_success": bool(debug_info.get("pnp_success", False)),
            "inlier_count": int(debug_info.get("inlier_count", 0)),
            "inlier_ratio": float(debug_info.get("inlier_ratio", 0.0)),
            "relocalization_success": bool(relocalization_success),
            "false_case": bool(false_case),
            "false_case_reason": "pose_inconsistency" if false_case else "none",
            "odom_delta_trans_m": float(odom_trans),
            "odom_delta_rot_deg": float(odom_rot),
            "map_delta_trans_m": float(map_trans),
            "map_delta_rot_deg": float(map_rot),
            "odom_pose": odom_pose.tolist(),
            "map_pose_estimate": map_pose_estimate.tolist(),
            "review_label": "unreviewed",
            "review_note": "",
        }
        self.retrieval_debug.save_sample(timestamp_ns, query_image, retrieved_images, sample_meta)

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
            0 : np.eye(4) if self.T_from_map_to_odom is None else self.T_from_map_to_odom,
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
        relative_pose_constraint = relative_pose_constraint[-100:]
        optimized_parameters = pose_graph_solve(optimized_parameters, relative_pose_constraint, constant_pose_index_dict, max_iteration_num = 1000)
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
            diff_position_norm_xy = np.linalg.norm(poi[:2] - pose_in_map_position[:2])
            diff_position_norm_z = np.linalg.norm(poi[2] - pose_in_map_position[2])
            if diff_position_norm_xy < 0.5 and diff_position_norm_z < 2.0:
                if self._leg_initial_length is not None:
                    arrived_msg = String()
                    arrived_msg.data = json.dumps({
                        "poi_index": self.poi_index,
                        "percent": 100.0,
                        "path_remaining_m": 0.0,
                        "path_total_m": round(self._leg_initial_length, 2),
                        "estimated_remaining_s": 0.0,
                    })
                    self.nav_progress_pub.publish(arrived_msg)
                self.poi_index += 1
                self._leg_initial_length = None
                self._leg_start_time = None
                dummy_pose = np.eye(4)

                stamp_msg = self.get_clock().now().to_msg()
                stamp_msg.sec = int(timestamp / 1e9)
                stamp_msg.nanosec = int(timestamp % 1e9)
                self.poi_change_pub.publish(np2msg(dummy_pose, stamp_msg, "world", "map"))
                continue
            else:
                break

        if self.poi_index >= len(self.pois):
            if not self._nav_completed:
                self._nav_completed = True
                self.get_logger().info("All POIs have been visited, nav done")
                self.nav_done_pub.publish(Bool(data=True))
            return

        target_poi = self.pois[self.poi_index]
        with Timer(name = "generate nav path in map", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            paths_in_map = self.generate_nav_path_in_map(pose_in_map = pose_in_map, target_poi = target_poi)

        if paths_in_map is not None:
            remaining_length = sum(
                np.linalg.norm(paths_in_map[i + 1] - paths_in_map[i])
                for i in range(len(paths_in_map) - 1)
            ) if len(paths_in_map) > 1 else 0.0

            now = time.time()
            if self._leg_initial_length is None:
                self._leg_initial_length = remaining_length
                self._leg_start_time = now

            covered = self._leg_initial_length - remaining_length
            elapsed = now - self._leg_start_time
            if covered > 0.1 and elapsed > 1.0:
                self._speed_estimate = covered / elapsed

            initial = self._leg_initial_length
            percent = max(0.0, min(100.0, covered / initial * 100.0)) if initial > 0 else 0.0
            estimated_remaining_s = remaining_length / self._speed_estimate if self._speed_estimate else -1.0

            progress_msg = String()
            progress_msg.data = json.dumps({
                "poi_index": self.poi_index,
                "percent": round(percent, 1),
                "path_remaining_m": round(remaining_length, 2),
                "path_total_m": round(initial, 2),
                "estimated_remaining_s": round(estimated_remaining_s, 1),
            })
            self.nav_progress_pub.publish(progress_msg)

            # use the max_speed to publish the position the robot should be after 5 seconds
            with Timer(name = "Find target position", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                max_speed = 0.5
                if len(paths_in_map) > 1:
                    accumulated_distance = 0.0
                    start_point = pose_in_map_position[:3]
                    target_position = paths_in_map[-1]
                    for i in range(len(paths_in_map) - 1):
                        accumulated_distance += np.linalg.norm(paths_in_map[i] - start_point)
                        if accumulated_distance > max_speed * 5:
                            target_position = paths_in_map[i]
                            break
                        start_point = paths_in_map[i]
                else:
                    target_position = paths_in_map[0]
                target_position_in_map = np.array([target_position[0], target_position[1], target_position[2]])
                pose_in_origin_odom = self.odom[timestamp]
                T = pose_in_origin_odom @ np.linalg.inv(pose_in_map)
                target_position_in_odom = T[:3, :3] @ target_position_in_map + T[:3, 3]
                dummy_pose = np.eye(4)
                dummy_pose[:3, 3] = target_position_in_odom
                #logging.info(f"target_position_in_odom: {target_position_in_odom}")
                print(f"target_position_in_odom: {target_position_in_odom}")

                self.target_pose_pub.publish(np2msg(dummy_pose, self.get_clock().now().to_msg(), "world", "camera"))
                path_msg = Path()
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.header.frame_id = "map"
                for x, y, z in paths_in_map:
                    pose = PoseStamped()
                    pose.header = path_msg.header
                    pose.pose.position.x = x
                    pose.pose.position.y = y
                    pose.pose.position.z = z
                    pose.pose.orientation.x = 0.0
                    pose.pose.orientation.y = 0.0
                    pose.pose.orientation.z = 0.0
                    pose.pose.orientation.w = 1.0
                    path_msg.poses.append(pose)
                self.global_plan_pub.publish(path_msg)
                self.tf_broadcaster.sendTransform(np2tf(T, self.get_clock().now().to_msg(), "world", "map"))
        else:
            logging.info("No path found in map")

    def generate_nav_path_in_map(self, pose_in_map: np.ndarray, target_poi: np.ndarray) -> np.ndarray:
        dummy_poi_pose = np.eye(4)
        dummy_poi_pose[:3, 3] = target_poi
        self.poi_pub.publish(np2msg(dummy_poi_pose, self.get_clock().now().to_msg(), "world", "map"))
        occupancy_map_origin = self.occupancy_map_meta[:3]
        resolution = self.occupancy_map_meta[3]
        start_idx = np.array([
            int((pose_in_map[0, 3] - occupancy_map_origin[0]) / resolution),
            int((pose_in_map[1, 3] - occupancy_map_origin[1]) / resolution),
            int((pose_in_map[2, 3] - occupancy_map_origin[2]) / resolution)
        ], dtype=np.int32)
        poi_goal_idx = np.array([
            int((target_poi[0] - occupancy_map_origin[0]) / resolution),
            int((target_poi[1] - occupancy_map_origin[1]) / resolution),
            int((target_poi[2] - occupancy_map_origin[2]) / resolution)
        ], dtype=np.int32)

        if (
            start_idx[0] < 0
            or start_idx[0] >= self.occupancy_map.shape[0]
            or start_idx[1] < 0
            or start_idx[1] >= self.occupancy_map.shape[1]
            or start_idx[2] < 0
            or start_idx[2] >= self.occupancy_map.shape[2]
            or poi_goal_idx[0] < 0
            or poi_goal_idx[0] >= self.occupancy_map.shape[0]
            or poi_goal_idx[1] < 0
            or poi_goal_idx[1] >= self.occupancy_map.shape[1]
            or poi_goal_idx[2] < 0
            or poi_goal_idx[2] >= self.occupancy_map.shape[2]
        ):
            return None 
        sdf_start_path = search_close_to_sdf_map(start_idx, self.sdf_map, self.occupancy_map, 0.2)
        sdf_goal_path = search_close_to_sdf_map(poi_goal_idx, self.sdf_map, self.occupancy_map, 0.2)

        sdf_start_sdf = sdf_start_path[-1]
        sdf_goal_sdf = sdf_goal_path[-1]
        path_sdf = search_within_sdf_map(sdf_start_sdf, sdf_goal_sdf, self.sdf_map, self.occupancy_map, resolution)
        if len(path_sdf) == 0:
            self.get_logger().warning(
                f"search_within_sdf_map returned empty path: start_idx={tuple(sdf_start_sdf)}, goal_idx={tuple(sdf_goal_sdf)}"
            )
        path = sdf_start_path + path_sdf + sdf_goal_path[::-1]
        if len(path) > 0:
            converted_path = np.array(path) * resolution + occupancy_map_origin
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
    parser.add_argument("--debug_retrieval_dataset", action="store_true", default=False, help="Enable retrieval debug dataset capture")
    parser.add_argument("--debug_retrieval_dir", type=str, default="tinynav_temp/retrieval_debug")
    parser.add_argument("--debug_retrieval_topk", type=int, default=3)
    parser.add_argument("--debug_save_images", action="store_true", default=True, help="Save query/retrieved images")
    parser.add_argument("--no_debug_save_images", dest="debug_save_images", action="store_false")
    parser.add_argument("--debug_save_every_n_keyframes", type=int, default=5)
    parser.add_argument("--debug_max_samples", type=int, default=10000)
    parsed_args, unknown_args = parser.parse_known_args(sys.argv[1:])
    node = MapNode(tinynav_db_path=parsed_args.tinynav_db_path,
                   tinynav_map_path=parsed_args.tinynav_map_path,
                   verbose_timer=parsed_args.verbose_timer,
                   debug_retrieval_dataset=parsed_args.debug_retrieval_dataset,
                   debug_retrieval_dir=parsed_args.debug_retrieval_dir,
                   debug_retrieval_topk=parsed_args.debug_retrieval_topk,
                   debug_save_images=parsed_args.debug_save_images,
                   debug_save_every_n_keyframes=parsed_args.debug_save_every_n_keyframes,
                   debug_max_samples=parsed_args.debug_max_samples)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
