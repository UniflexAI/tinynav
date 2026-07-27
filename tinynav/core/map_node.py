import rclpy
import os
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool, String, Float32
import numpy as np
import sys
import json

import heapq
from tinynav.core.math_utils import matrix_to_quat, msg2np, np2msg, estimate_pose, np2tf, rerank_by_pnp_inliers
from sensor_msgs.msg import Image, CameraInfo
from message_filters import TimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
from codetiming import Timer
import argparse

from tinynav.tinynav_cpp_bind import pose_graph_solve
from tinynav.core.models_trt import LightGlueTRT, Dinov2TRT, SuperPointTRT
import logging
import asyncio
from tf2_ros import TransformBroadcaster
from tinynav.core.build_map_node import TinyNavDB
from tinynav.core.build_map_node import find_loop, solve_pose_graph
from tinynav.core.vlad import compute_vlad
import einops
from tinynav.core.build_map_node import OdomPoseRecorder
from tinynav.core.planning_node import GO2_CONFIG
from tinynav.core.stair_hint import PathClimbIndex
from tinynav.core.path_speed import PathSpeedIndex
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
        # Stair hint: label the robot's pose-in-map climbing/flat off the offline
        # capture-path labels (path_climb.npy). Folded in here (not a separate node)
        # because map_node already owns pose-in-map and the map dir. Consumers:
        # planning_node (relax z-span) and the app backend (frontend indicator).
        self.on_stairs_pub = self.create_publisher(Bool, "/planning/on_stairs", 10)
        # Capture-speed prior: the operator's local speed (path_speed.npy) at the
        # robot's pose-in-map. planning_node caps peak forward speed by it.
        self.speed_cap_pub = self.create_publisher(Float32, "/planning/speed_cap", 10)

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
        self.latest_odom_pose = None
        self.pose_graph_used_pose = {}
        self.relative_pose_constraint = []
        self.last_keyframe_timestamp = None

        self.loop_similarity_threshold = 0.90
        self.loop_top_k = 1

        self.relocalization_loop_top_k = 3

        os.makedirs(f"{tinynav_db_path}/nav_temp", exist_ok=True)
        self.nav_temp_db = TinyNavDB(f"{tinynav_db_path}/nav_temp", is_scratch=True)
        self.map_poses = np.load(f"{tinynav_map_path}/poses.npy", allow_pickle=True).item()
        stair_path = f"{tinynav_map_path}/path_climb.npy"
        self.stair_index = PathClimbIndex.load(stair_path) if os.path.exists(stair_path) else None
        speed_path = f"{tinynav_map_path}/path_speed.npy"
        self.speed_index = PathSpeedIndex.load(speed_path) if os.path.exists(speed_path) else None
        self.map_K = np.load(f"{tinynav_map_path}/intrinsics.npy")
        self.db = TinyNavDB(tinynav_map_path, is_scratch=False)
        self.vlad_timestamps = list(self.map_poses.keys())
        try:
            self.vlad_centres = self.db.metadata["vlad_centres"]
            self.map_vlad_descriptors = np.stack([
                self.db.vlad_descriptors[timestamp]
                for timestamp in self.vlad_timestamps
            ])
        except KeyError as exc:
            raise RuntimeError(
                "This map does not contain a complete DINOv2 patch VLAD "
                "relocalization index. Rebuild the map with this branch before "
                f"running map_node. map_path={tinynav_map_path}, missing_key={exc}"
            ) from exc
        self.get_logger().info(
            "Using DINOv2 patch VLAD relocalization index: "
            f"vocab={self.vlad_centres.shape}, "
            f"descriptors={self.map_vlad_descriptors.shape}, "
            f"keyframes={len(self.vlad_timestamps)}"
        )
        self.occupancy_map = np.load(f"{tinynav_map_path}/occupancy_grid.npy")
        self.occupancy_map_meta = np.load(f"{tinynav_map_path}/occupancy_meta.npy")
        self.sdf_map = np.load(f"{tinynav_map_path}/sdf_map.npy")

        print(f"sdf_map.shape: {self.sdf_map.shape}")
        print(f"occupancy_map.shape: {self.occupancy_map.shape}")

        self.relocalization_poses = {}
        self.relocalization_pose_weights = {}
        self.failed_relocalizations = []

        # Continuous relocalization. Every successful relocalization is one observation
        # of map->odom; they are fused by compute_transform_from_map_to_odom() as a
        # weighted pose graph, so the estimate keeps tracking the black-box VIO's drift
        # instead of being frozen at the first fix.
        #
        # Retrieval can still match a wrong-but-similar place, and one confident-but-
        # wrong observation would drag the whole graph, so observations are gated before
        # they are fused:
        #   bootstrap (no estimate yet) -- require a burst of `reloc_burst_window` recent
        #     observations agreeing within `reloc_burst_tol` before trusting any of them.
        #   steady state -- accept an observation within `reloc_accept_tol` of the current
        #     estimate. The tolerance is deliberately loose: a *correct* observation is
        #     supposed to disagree with a drifted estimate, and that disagreement is the
        #     signal we want to fuse, not reject.
        #   kidnap -- if a full burst mutually agrees yet all of it disagrees with the
        #     current estimate, we were wrong (or the robot was moved): re-bootstrap onto
        #     the burst rather than reject it forever.
        # Sliding window (not fill-then-clear) so a stray bad observation can't keep
        # resetting the count.
        self.reloc_burst_window = 3         # recent observations that must agree
        self.reloc_burst_tol = 0.3          # meters; max pairwise translation spread
        self.reloc_accept_tol = 1.5         # meters; max |t| delta vs the current estimate
        self._reloc_obs_window = []         # recent observation_T_from_map_to_odom (4x4)
        self._accepted_reloc_timestamps = []   # timestamps that passed the gate -> fused
        self._reloc_first_fix_published = False

        # Relocalization costs VLAD + SuperPoint + LightGlue x top_k + PnP per attempt,
        # while looper_bridge emits keyframes every 3 cm / 1 deg -- far faster than that
        # pipeline can run. Rate-limit attempts once we have a fix; before the first fix
        # go flat out, because relocalize.py's turn-in-place sweep is blocked waiting on
        # /map/relocalization.
        self.reloc_min_interval_s = 1.0
        self._last_reloc_attempt_s = None
        self._latest_keyframe_features = (None, None)

        self.T_from_map_to_odom = None

        self.pois = {}
        self.poi_index = -1
        self._nav_completed = False
        self._leg_initial_length: float | None = None
        self._leg_start_time: float | None = None
        self._speed_estimate: float | None = None
        self.cached_nav_path_in_map = None
        self.cached_nav_path_poi_index = -1

        self.poi_pub = self.create_publisher(Odometry, "/mapping/poi", 10)
        self.poi_change_pub = self.create_publisher(Odometry, "/mapping/poi_change", 10)
        self.nav_done_pub = self.create_publisher(Bool, '/mapping/nav_done', 10)
        self.nav_progress_pub = self.create_publisher(String, '/mapping/nav_progress', 10)

        self.current_pose_pub = self.create_publisher(Odometry, "/mapping/current_pose", 10)
        self.global_plan_pub = self.create_publisher(Path, '/mapping/global_plan', 10)
        self.target_pose_pub = self.create_publisher(Odometry, "/control/target_pose", 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self._save_completed = False
        self.nav_target_timer = self.create_timer(0.5, self.nav_target_timer_callback)

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
                self.cached_nav_path_in_map = None
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
            self.cached_nav_path_in_map = None
            self.cached_nav_path_poi_index = -1
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
        self.latest_odom_pose, _ = msg2np(odom_msg)

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

        if not self._should_attempt_relocalization():
            return
        timestamp_ns = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(keyframe_image_msg.header.stamp.nanosec)
        success, pose_in_world = self.keyframe_relocalization(keyframe_image_msg.header.stamp, image)
        if success:
            self.update_transform_from_map_to_odom(timestamp_ns, keyframe_image_msg.header.stamp)

    def _should_attempt_relocalization(self) -> bool:
        """Flat out until the first fix (the nav sweep is blocked on it), then capped at
        one attempt per reloc_min_interval_s -- keyframes arrive far faster than the
        VLAD/match/PnP pipeline can run."""
        now = time.monotonic()
        if self.T_from_map_to_odom is None:
            self._last_reloc_attempt_s = now
            return True
        if (self._last_reloc_attempt_s is not None
                and now - self._last_reloc_attempt_s < self.reloc_min_interval_s):
            return False
        self._last_reloc_attempt_s = now
        return True


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
        # Relocalization runs on this same keyframe right after; hand it these features
        # rather than paying for a second SuperPoint pass over the same image.
        self._latest_keyframe_features = (keyframe_image_timestamp, features)

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
            find_loop_and_pose_graph(keyframe_image_timestamp)
            self.pose_graph_trajectory_publish(keyframe_image_timestamp)
        self.last_keyframe_timestamp = keyframe_odom_timestamp
        self.last_keyframe_image = image


    def get_embeddings(self, image: np.ndarray) -> np.ndarray:
        # shape: (1, 768)
        return asyncio.run(self.dinov2_model.infer(image))

    def get_vlad_descriptor(self, image: np.ndarray) -> np.ndarray:
        patch_tokens = asyncio.run(self.dinov2_model.infer_patch_tokens(image))
        return compute_vlad(patch_tokens, self.vlad_centres)

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

        query_vlad = self.get_vlad_descriptor(keyframe)
        idx_and_similarity_array = find_loop(
            query_vlad,
            self.map_vlad_descriptors,
            -1.0,
            self.relocalization_loop_top_k,
        )
        if len(idx_and_similarity_array) == 0:
            print("VLAD: no relocalization candidates")
            return False, np.eye(4), -np.inf
        candidate_timestamps = [
            int(self.vlad_timestamps[idx_in_map])
            for idx_in_map, _similarity in idx_and_similarity_array
        ]

        pnp_candidates = []
        for timestamp_in_map in candidate_timestamps:
            reference_keyframe_pose = self.map_poses[timestamp_in_map]
            reference_depth, _, reference_features, _, _ = self.db.get_depth_embedding_features_images(timestamp_in_map)
            reference_matched_keypoints, keyframe_matched_keypoints, matches = self.match_keypoints(reference_features, keyframe_features)
            if len(matches) < 50:
                print(f"not enough matched features to relocalize, {len(matches)} < 50")
                continue

            point_3d_in_world, inliers = self.keypoint_with_depth_to_3d(reference_matched_keypoints, reference_depth, reference_keyframe_pose, self.map_K)
            point_3d_in_world_list = point_3d_in_world[inliers]
            point_2d_in_keyframe_list = keyframe_matched_keypoints[inliers]
            point_count = len(point_2d_in_keyframe_list)
            if point_count <= 80:
                print(f"not enough landmarks to relocalize, {point_count}")
                continue
            pnp_candidates.append((point_3d_in_world_list, point_2d_in_keyframe_list))

        success, best_pose_in_camera, pose_cov_weight, _, _, _ = rerank_by_pnp_inliers(pnp_candidates, self.map_K)
        if success:
            print(f"relocalization pose : {best_pose_in_camera}")
            return True, best_pose_in_camera, pose_cov_weight

        print("no valid PnP relocalization candidate found")
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
        timestamp_ns = int(timestamp.sec * 1e9) + int(timestamp.nanosec)
        cached_timestamp, cached_features = self._latest_keyframe_features
        if cached_timestamp == timestamp_ns:
            features = cached_features          # extracted by keyframe_mapping already
        else:
            features = asyncio.run(self.super_point_extractor.infer(image))
        res, pose_in_camera, pose_cov_weight = self.relocalize_with_depth(image, features, self.K)
        if res:
            pose_in_world = np.linalg.inv(pose_in_camera)
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


    def _burst_spread(self) -> float | None:
        """Max pairwise translation spread of the recent observation window, or None
        until the window is full."""
        if len(self._reloc_obs_window) < self.reloc_burst_window:
            return None
        translations = np.array([T[:3, 3] for T in self._reloc_obs_window])
        return float(np.max([
            np.linalg.norm(translations[i] - translations[j])
            for i in range(len(translations))
            for j in range(i + 1, len(translations))
        ]))

    def update_transform_from_map_to_odom(self, timestamp: int, stamp):
        """Gate one relocalization observation, then refuse or fuse it.

        Each successful relocalization gives one observation of map->odom. Retrieval can
        match a wrong-but-similar place, and a confident-but-wrong observation would drag
        the pose graph, so an observation is fused only if it survives the gate described
        where reloc_burst_window is defined.

        /map/relocalization fires once, on the first accepted fix -- consumers
        (relocalize.py's sweep, node_manager's localized latch) want to know the instant
        nav can plan a path, not every subsequent refinement.
        """
        if timestamp not in self.pose_graph_used_pose:
            return
        camera_in_map_world = self.relocalization_poses[timestamp]
        camera_in_odom_world = self.pose_graph_used_pose[timestamp]
        observation_T_from_map_to_odom = camera_in_odom_world @ np.linalg.inv(camera_in_map_world)

        self._reloc_obs_window.append(observation_T_from_map_to_odom)
        self._reloc_obs_window = self._reloc_obs_window[-self.reloc_burst_window:]
        spread = self._burst_spread()
        burst_agrees = spread is not None and spread <= self.reloc_burst_tol

        if self.T_from_map_to_odom is None:
            # Bootstrap: trust nothing until a full burst agrees with itself.
            if not burst_agrees:
                self.get_logger().info(
                    f"[reloc] bootstrapping, {len(self._reloc_obs_window)} obs not consistent yet"
                    + (f", spread={spread:.2f}m > tol={self.reloc_burst_tol}m" if spread is not None else ""))
                return
            self._accept(timestamp, camera_in_map_world, stamp)
            self.get_logger().info(
                f"[reloc] first fix (spread={spread:.2f}m over {self.reloc_burst_window} obs)")
            return

        delta = float(np.linalg.norm(
            observation_T_from_map_to_odom[:3, 3] - self.T_from_map_to_odom[:3, 3]))
        if delta <= self.reloc_accept_tol:
            self._accept(timestamp, camera_in_map_world, stamp)
            return

        # Disagrees with the current estimate. If the whole burst mutually agrees, the
        # estimate is what's wrong (bad lock, or the robot was picked up) -- re-bootstrap
        # onto the burst instead of rejecting correct observations forever.
        if burst_agrees:
            self.get_logger().warning(
                f"[reloc] {self.reloc_burst_window} consistent obs disagree with the current "
                f"estimate by {delta:.2f}m -- re-bootstrapping onto them")
            self._accepted_reloc_timestamps.clear()
            self.T_from_map_to_odom = None
            self._accept(timestamp, camera_in_map_world, stamp)
            return

        self.get_logger().info(
            f"[reloc] rejected outlier: {delta:.2f}m > tol={self.reloc_accept_tol}m from estimate")

    def _accept(self, timestamp: int, camera_in_map_world: np.ndarray, stamp):
        """Admit an observation into the fused set and refresh the estimate."""
        self._accepted_reloc_timestamps.append(timestamp)
        self.compute_transform_from_map_to_odom()
        if not self._reloc_first_fix_published:
            self.relocation_pub.publish(np2msg(camera_in_map_world, stamp, "world", "camera"))
            self._reloc_first_fix_published = True

    def compute_transform_from_map_to_odom(self):
        """Fuse the accepted relocalization observations into T_from_map_to_odom.

        Each accepted relocalization is one observation of the map->odom transform,
        weighted by its PnP inlier ratio. Solving over a sliding window (rather than
        taking the newest) averages out per-fix noise; keeping the window short is what
        lets the estimate follow the black-box VIO's drift instead of being anchored to
        stale observations.
        """
        relative_pose_constraint = []
        optimized_parameters = {
            0: np.eye(4) if self.T_from_map_to_odom is None else self.T_from_map_to_odom,
            1: np.eye(4),
        }
        constant_pose_index_dict = {1: True}
        for timestamp in self._accepted_reloc_timestamps:
            if timestamp not in self.pose_graph_used_pose:
                continue
            camera_in_map_world = self.relocalization_poses[timestamp]
            camera_in_odom_world = self.pose_graph_used_pose[timestamp]
            observation_T_from_map_to_odom = camera_in_odom_world @ np.linalg.inv(camera_in_map_world)
            weight = self.relocalization_pose_weights[timestamp]
            relative_pose_constraint.append((
                0, 1, observation_T_from_map_to_odom,
                weight * np.array([10.0, 10.0, 10.0]),
                weight * np.array([10.0, 10.0, 10.0]),
            ))
        if len(relative_pose_constraint) == 0:
            return
        relative_pose_constraint = relative_pose_constraint[-100:]
        self._accepted_reloc_timestamps = self._accepted_reloc_timestamps[-100:]
        optimized_parameters = pose_graph_solve(
            optimized_parameters, relative_pose_constraint, constant_pose_index_dict,
            max_iteration_num=1000)
        self.T_from_map_to_odom = optimized_parameters[0]

    def _publish_global_plan(self, paths_in_map: np.ndarray):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for x, y, z in paths_in_map:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.global_plan_pub.publish(path_msg)

    def nav_target_timer_callback(self):
        if (
            self.poi_index < 0
            or self.poi_index >= len(self.pois)
            or self.T_from_map_to_odom is None
            or self.latest_odom_pose is None
        ):
            self.get_logger().info(
                f"[nav_timer] skip: poi_index={self.poi_index}, "
                f"n_pois={len(self.pois)}, "
                f"T={'set' if self.T_from_map_to_odom is not None else 'None'}, "
                f"odom={'set' if self.latest_odom_pose is not None else 'None'}"
            )
            return

        pose_in_map = np.linalg.inv(self.T_from_map_to_odom) @ self.latest_odom_pose
        self.current_pose_in_map_pub.publish(np2msg(pose_in_map, self.get_clock().now().to_msg(), "world", "map"))
        on_stairs = bool(self.stair_index.on_stairs(pose_in_map[:3, 3])) if self.stair_index else False
        self.on_stairs_pub.publish(Bool(data=on_stairs))
        # Capture speed (m/s) near the robot; +inf when off-path/unknown (speed_cap's
        # own "no cap" sentinel) -> planning's isfinite guard treats it as no data and
        # falls back to vx_max, so publish it straight through.
        cap = self.speed_index.speed_cap(pose_in_map[:3, 3]) if self.speed_index else float('inf')
        self.speed_cap_pub.publish(Float32(data=float(cap)))

        poi = self.pois[self.poi_index]
        pos = pose_in_map[:3, 3]
        # Arrival is a "did the body reach the POI" test, but the odom pose is the
        # CAMERA pose — measuring from it leaves a POI beside/under the robot (e.g.
        # near the rear footprint) reading a full body-offset too far, so it never
        # registers as reached and nav sits there forever. Shift camera -> control
        # center using the shared robot config (same transform as planning_node's
        # camera_to_robot_center) so the offset can't drift from a hardcoded copy.
        # `pos` (camera) is still used for path following below.
        arrival_pos = pos - pose_in_map[:3, :3] @ GO2_CONFIG.cam_offset_3d

        if np.linalg.norm(poi[:2] - arrival_pos[:2]) < 0.5 and abs(poi[2] - arrival_pos[2]) < 2.0:
            if self._leg_initial_length is not None:
                self.nav_progress_pub.publish(String(data=json.dumps({
                    "poi_index": self.poi_index,
                    "percent": 100.0,
                    "path_remaining_m": 0.0,
                    "path_total_m": round(self._leg_initial_length, 2),
                    "estimated_remaining_s": 0.0,
                })))
            self.poi_index += 1
            self._leg_initial_length = None
            self._leg_start_time = None
            self._speed_estimate = None
            self.cached_nav_path_in_map = None
            self.cached_nav_path_poi_index = -1
            self.poi_change_pub.publish(np2msg(np.eye(4), self.get_clock().now().to_msg(), "world", "map"))
            if self.poi_index >= len(self.pois) and not self._nav_completed:
                self._nav_completed = True
                self.get_logger().info("All POIs have been visited, nav done")
                self.nav_done_pub.publish(Bool(data=True))
            return

        needs_replan = (
            self.cached_nav_path_in_map is None
            or self.cached_nav_path_poi_index != self.poi_index
        )
        if not needs_replan:
            paths = self.cached_nav_path_in_map
            closest_idx = int(np.argmin(np.linalg.norm(paths[:, :2] - pos[:2], axis=1)))
            if np.linalg.norm(paths[closest_idx, :2] - pos[:2]) > 0.5:
                needs_replan = True

        if needs_replan:
            paths = self.generate_nav_path_in_map(pose_in_map=pose_in_map, target_poi=poi)
            if paths is not None:
                self.cached_nav_path_in_map = paths
                self.cached_nav_path_poi_index = self.poi_index
            else:
                self.cached_nav_path_in_map = None
                self.cached_nav_path_poi_index = -1
                return

        paths = self.cached_nav_path_in_map
        self._publish_global_plan(paths)
        closest_idx = int(np.argmin(np.linalg.norm(paths[:, :2] - pos[:2], axis=1)))

        remaining_length = sum(
            np.linalg.norm(paths[i + 1] - paths[i])
            for i in range(closest_idx, len(paths) - 1)
        ) if closest_idx < len(paths) - 1 else 0.0

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

        self.nav_progress_pub.publish(String(data=json.dumps({
            "poi_index": self.poi_index,
            "percent": round(percent, 1),
            "path_remaining_m": round(remaining_length, 2),
            "path_total_m": round(initial, 2),
            "estimated_remaining_s": round(estimated_remaining_s, 1),
        })))

        max_speed = 0.5

        # local target = furthest point on the path reachable from the robot before
        # the heading turns past turn_thresh (a corner) or lookahead_max arc length is
        # reached. Drives to corners instead of slicing across them.
        local_i = self._local_target_index(paths, closest_idx, lookahead_max=max_speed * 5)
        target_position = paths[local_i]

        T = self.latest_odom_pose @ np.linalg.inv(pose_in_map)
        target_position_in_odom = T[:3, :3] @ target_position + T[:3, 3]
        dummy_pose = np.eye(4)
        dummy_pose[:3, 3] = target_position_in_odom
        self.target_pose_pub.publish(np2msg(dummy_pose, self.get_clock().now().to_msg(), "world", "camera"))

        self.tf_broadcaster.sendTransform(np2tf(T, self.get_clock().now().to_msg(), "world", "map"))

    def _local_target_index(self, path, start_i, lookahead_max, min_lookahead=1.0,
                            turn_thresh=np.deg2rad(60.0), smooth_m=0.4):
        """Index of the local target: walk forward from start_i, stop at the first
        point (beyond min_lookahead) whose smoothed heading has turned >= turn_thresh
        from the entry heading (a corner), or when lookahead_max arc length is reached.
        Never returns a point closer than min_lookahead in arc length."""
        pxy = [np.asarray(p[:2], dtype=np.float64) for p in path]
        n = len(pxy)
        cum = [0.0] * n
        for i in range(start_i + 1, n):
            cum[i] = cum[i - 1] + float(np.linalg.norm(pxy[i] - pxy[i - 1]))

        def sdir(i):
            j = i
            while j < n - 1 and (cum[j] - cum[i]) < smooth_m:
                j += 1
            d = pxy[j] - pxy[i]
            L = float(np.linalg.norm(d))
            return d / L if L > 1e-6 else None

        entry = sdir(start_i)
        li = start_i
        for k in range(start_i + 1, n):
            if cum[k] - cum[start_i] >= lookahead_max:
                li = k
                break
            dk = sdir(k)
            if (cum[k] - cum[start_i]) >= min_lookahead and entry is not None and dk is not None:
                turn = abs(np.arctan2(dk[0] * entry[1] - dk[1] * entry[0], float(dk @ entry)))
                if turn >= turn_thresh:
                    li = k
                    break
            li = k
        return li

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
    parsed_args, unknown_args = parser.parse_known_args(sys.argv[1:])
    node = MapNode(tinynav_db_path=parsed_args.tinynav_db_path,
                   tinynav_map_path=parsed_args.tinynav_map_path,
                   verbose_timer=parsed_args.verbose_timer)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
