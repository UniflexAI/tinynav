import rclpy
import os
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point32
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool, String, Float32
import numpy as np
import sys
import json

import heapq
from tinynav.core.math_utils import matrix_to_quat, msg2np, np2msg, estimate_pose, np2tf, rerank_by_pnp_inliers
from sensor_msgs.msg import Image, CameraInfo, PointCloud
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
from tinynav.core.path_speed import PathSpeedIndex, bake as bake_path_speed
from tinynav.core.path_climb import PathClimbIndex, bake as bake_path_climb, n_climbing
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

# Arrival radius, measured from the CAMERA (see nav_target_timer_callback).
_ARRIVE_M = 0.5
# How many consecutive ticks must agree that the robot is inside that radius.
# A relocalization can move the pose metres in one step -- that is what an
# accepted correction IS -- and a single tick taken right after one has ended
# legs at places the robot was nowhere near. Two ticks is 1s at the nav timer's
# 2Hz: long enough that a jump has to be confirmed by the pose that follows it,
# short enough to cost nothing on a real arrival.
_ARRIVE_TICKS = int(os.environ.get('TINYNAV_ARRIVE_TICKS', '2'))
# ...and for a POI that carries an arrival heading, where being 0.5m out matters.
_ARRIVE_HEADING_M = 0.2

# The target pose is a carrot at a TIME horizon, so how far along the path it sits has
# to ride the speed actually being driven -- which is the capture-speed prior this node
# already publishes on /planning/speed_cap. It used to be a flat 2.5m (0.5 m/s x 5s), so
# where the operator crept at 0.2 m/s the carrot sat 12.5s ahead, aiming past the tight
# stretch the slow capture speed was warning about.
_LOOKAHEAD_S = 5.0
# No prior (off-path, or a map with no path_speed.npy): planning falls back to vx_max for
# the speed, so the carrot falls back to the same number.
_NO_CAP_SPEED_MPS = 0.6
# Bounds on the resulting distance. These are planning's [vx_min, vx_hard_max] x
# _LOOKAHEAD_S -- the span of speeds it can actually command -- expressed as metres so
# this node needs none of planning's parameters to stay consistent with it.
_LOOKAHEAD_MIN_M = 1.0
_LOOKAHEAD_MAX_M = 5.0


def lookahead_distance_m(speed_cap_mps: float) -> float:
    """How far along the path the target pose sits, given the capture-speed prior.

    `speed_cap_mps` is what /planning/speed_cap carries: +inf (or NaN) means off-path or
    no prior, the same sentinel planning treats as "no data"."""
    speed = (speed_cap_mps if np.isfinite(speed_cap_mps) else _NO_CAP_SPEED_MPS)
    return float(np.clip(speed * _LOOKAHEAD_S, _LOOKAHEAD_MIN_M, _LOOKAHEAD_MAX_M))


# ── climb prior ─────────────────────────────────────────────────────────────── #
# Where the capture path climbs, as geometry planning_node applies per cell (it relaxes
# the obstacle z-span filter near these points -- see climb_region_radius_m there).
CLIMB_REGION_TOPIC = '/planning/climb_region'
# The same prior collapsed to "is the robot itself on it", which is all the app's
# indicator wanted. Kept separate so neither has to answer the other's question.
ON_STAIRS_TOPIC = '/planning/on_stairs'
# Timer rate for the map-frame priors published here.
MAP_PRIOR_HZ = 2.0
# Send-side cull only -- NOT the region's width, which the planner owns
# (climb_region_radius_m).
CLIMB_REGION_CULL_M = 3.5
# False = don't derive the region from the capture path at all: no labels are loaded or
# baked, the region publishes empty, and the planner falls back to its strict span filter
# everywhere -- i.e. unmodified upstream behaviour. The prior is only worth having while
# the labels are trustworthy; per-map VIO z is noisy enough that a false band RELAXES the
# obstacle filter on flat ground, so being able to switch it off is a safety valve, not a
# debug flag. A ROS parameter (`climb_prior`) like planning_node's own climb knobs, so a
# site can disable it from the launch without a rebuild.
CLIMB_PRIOR_DEFAULT = True


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
        # Capture-speed prior: the operator's local speed (path_speed.npy) at the
        # robot's pose-in-map. planning_node caps peak forward speed by it.
        self.speed_cap_pub = self.create_publisher(Float32, "/planning/speed_cap", 10)
        # Climb prior: the capture samples labelled climbing, as geometry planning_node
        # applies per cell, plus the collapsed "am I on it" flag for the app.
        self.climb_region_pub = self.create_publisher(PointCloud, CLIMB_REGION_TOPIC, 10)
        self.on_stairs_pub = self.create_publisher(Bool, ON_STAIRS_TOPIC, 10)

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
        self.speed_index = None
        self.climb_index = None
        self.declare_parameter('climb_prior', CLIMB_PRIOR_DEFAULT)
        self._climb_prior = bool(self.get_parameter('climb_prior').value)
        self.load_map_priors(tinynav_map_path)
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

        # SuperPoint features for the current keyframe, extracted once by
        # keyframe_mapping and reused by the relocalization that follows on the same
        # image rather than paying for a second pass.
        self._latest_keyframe_features = (None, None)

        self.T_from_map_to_odom = None

        self.pois = {}
        self.poi_index = -1
        # Consecutive nav ticks that have seen the robot inside the arrival radius.
        self._arrive_ticks = 0
        # Queue indices whose POI carries an arrival heading (tighter arrival radius).
        self.poi_has_heading = set()
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
        # Its own timer, not nav_target_timer_callback: that one returns early without
        # POIs, and the planner needs the climb region whenever it is planning at all.
        self.map_prior_timer = self.create_timer(1.0 / MAP_PRIOR_HZ, self.tick_map_priors)

    def load_map_priors(self, tinynav_map_path: str) -> None:
        """Load this map's capture-path priors — speed and climb. One method so a node
        that swaps maps at runtime refreshes both in one call.

        Never raises: a bad or missing prior degrades to "no data" (planning falls back
        to vx_max for speed, and to its strict span filter everywhere for climb), which
        is the safe direction and must not stop nav."""
        speed_path = f"{tinynav_map_path}/path_speed.npy"
        try:
            # Bakes when missing OR stale -- a reloop rewrites poses.npy and every prior
            # derived from it, so the check is on the mtime, not just existence.
            self.get_logger().info(f"[speed] {bake_path_speed(tinynav_map_path)}")
            self.speed_index = (PathSpeedIndex.load(speed_path)
                                if os.path.exists(speed_path) else None)
        except Exception as exc:
            self.get_logger().error(f"[speed] {speed_path} unusable: {exc}")
            self.speed_index = None

        climb_path = f"{tinynav_map_path}/path_climb.npy"
        self.climb_index = None
        if not self._climb_prior:
            # Before the bake, so switching off also stops writing labels nothing reads.
            self.get_logger().info(
                "[climb] climb_prior=false — no climb prior, strict everywhere")
            return
        try:
            self.get_logger().info(f"[climb] {bake_path_climb(tinynav_map_path)}")
            if os.path.exists(climb_path):
                self.climb_index = PathClimbIndex.load(climb_path)
        except Exception as exc:
            self.get_logger().error(f"[climb] {climb_path} unusable: {exc}")
            return
        if self.climb_index is None:
            self.get_logger().warning(
                f"[climb] no labels for {tinynav_map_path} — strict everywhere")
            return
        self.get_logger().info(
            f"[climb] {n_climbing(self.climb_index.pts)}/"
            f"{len(self.climb_index.pts)} capture samples labelled climbing")

    def tick_map_priors(self) -> None:
        """Publish the climbing samples around the robot, in the odom frame planning
        works in. Empty (but still published) when there is no prior or no fix, so a
        stale region never outlives the map it came from."""
        T = self.T_from_map_to_odom
        msg = PointCloud()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        on_stairs = False
        if self.climb_index is not None and T is not None and self.latest_odom_pose is not None:
            here = (np.linalg.inv(T) @ self.latest_odom_pose)[:3, 3]
            on_stairs = self.climb_index.on_stairs(here)
            pts_in_map = self.climb_index.climbing_within(here, CLIMB_REGION_CULL_M)
            pts_in_odom = (T[:3, :3] @ pts_in_map.T + T[:3, 3:4]).T
            msg.points = [Point32(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                          for p in pts_in_odom]
        self.climb_region_pub.publish(msg)
        self.on_stairs_pub.publish(Bool(data=bool(on_stairs)))

    def pois_callback(self, msg: String):
        self.get_logger().info("Received POIs from planner: " + msg.data)
        try:
            self.pois = json.loads(msg.data)

            pois_dict = {}
            poi_has_heading = set()
            keys = sorted([int (key) for key in self.pois.keys()])
            for index, key in enumerate(keys):
                entry = self.pois[str(key)]
                pois_dict[index] = np.array(entry["position"])
                if entry.get("yaw_deg") is not None:
                    poi_has_heading.add(index)
            self.pois = pois_dict
            self.poi_has_heading = poi_has_heading

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

        success, pose_in_world = self.keyframe_relocalization(keyframe_image_msg.header.stamp, image)
        if success:
            self.compute_transform_from_map_to_odom()


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

    def select_relocalization_candidates(self, query_vlad: np.ndarray) -> list[tuple[int, float]]:
        """(map keyframe timestamp, VLAD similarity) to try PnP against, best last.

        A hook so a subclass can drop candidates retrieval alone cannot tell apart —
        the whole-map top-k here is blind to where the robot actually is."""
        return [
            (int(self.vlad_timestamps[idx_in_map]), float(similarity))
            for idx_in_map, similarity in find_loop(
                query_vlad,
                self.map_vlad_descriptors,
                -1.0,
                self.relocalization_loop_top_k,
            )
        ]

    def rank_relocalization_candidates(self, pnp_candidates: list, candidate_timestamps: list[int]) -> tuple[bool, np.ndarray, float]:
        """Pick the pose among the surviving candidates. `candidate_timestamps` is
        parallel to `pnp_candidates` so an override can see where each one sits in the
        map (the batch call below only ever reports the winner)."""
        success, best_pose_in_camera, pose_cov_weight, _, _, _ = rerank_by_pnp_inliers(pnp_candidates, self.map_K)
        return success, best_pose_in_camera, pose_cov_weight

    def relocalize_with_depth(self, keyframe: np.ndarray, keyframe_features: dict, K: np.ndarray | None) -> tuple[bool, np.ndarray, float]:
        if K is None:
            return False, np.eye(4), -np.inf

        query_vlad = self.get_vlad_descriptor(keyframe)
        candidates = self.select_relocalization_candidates(query_vlad)
        if len(candidates) == 0:
            print("VLAD: no relocalization candidates")
            return False, np.eye(4), -np.inf

        pnp_candidates = []
        pnp_timestamps = []
        for timestamp_in_map, _similarity in candidates:
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
            pnp_timestamps.append(timestamp_in_map)

        success, best_pose_in_camera, pose_cov_weight = self.rank_relocalization_candidates(pnp_candidates, pnp_timestamps)
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
            # publish the relocalization pose for debug
            pose_in_world = np.linalg.inv(pose_in_camera)
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


    #: How many constraints the solve may use, newest first. Upstream's 100; kept as a
    #: knob rather than a constant because it is the number the travel bound below
    #: falls back to when the robot is standing still.
    FUSE_WINDOW = int(os.environ.get("TINYNAV_FUSE_WINDOW", "100"))
    #: How far the robot may have driven since an observation before that observation's
    #: implied map->odom is treated as stale. The units are metres of odom travel and
    #: not seconds, because what invalidates a constraint is drift, and drift comes
    #: with travel: a robot standing still holds its constraints for ever, which is
    #: what a stationary robot needs.
    FUSE_MAX_M = float(os.environ.get("TINYNAV_FUSE_MAX_M", "3.0"))

    def _fresh_constraints(self, constraints):
        """The constraints the solve should see: recent in travel, capped in count.

        Each entry carries the odom pose it was taken at (the last element), which is
        what the distance is measured against -- the newest constraint's odom is
        "here", so this asks how far back each one sits along the ride.
        """
        if not constraints:
            return []
        here = constraints[-1][-1][:3, 3]
        fresh = [c for c in constraints
                 if float(np.linalg.norm(c[-1][:3, 3] - here)) <= self.FUSE_MAX_M]
        # Never fewer than a handful: a solve with one constraint is that observation
        # alone, which is the aliasing case with no averaging left to damp it.
        if len(fresh) < 5:
            fresh = constraints[-5:]
        return [c[:5] for c in fresh[-self.FUSE_WINDOW:]]

    def compute_transform_from_map_to_odom(self):
        """
        Solve the optmization problem.

        Each constraint is one observation's implied map->odom, `camera_in_odom @
        inv(camera_in_map)` -- true at the moment that observation was taken, and only
        still true while odom has not drifted since. Upstream keeps the last 100
        regardless of when they were taken and weights them equally, which measured on
        118 (2026-08-31) is why a wrong estimate cannot be corrected: standing still,
        20 consecutive PnP answers all agreed on (42.44, -4.47) while the fused
        estimate sat 0.75m away at (43.06, -4.90) and its yaw wandered from 89 to 95
        degrees. At ~0.8 relocalizations a second, 100 constraints reach back over two
        minutes, so every good new observation was one vote against ninety-nine older
        ones -- including the wrong ones from a stretch the robot has already left.

        So the window is bounded by **travel** as well as by count: a constraint whose
        odom is more than FUSE_MAX_M of driving old describes a transform that has
        since drifted, and is dropped. Standing still expires nothing, which is what
        keeps a stationary robot from ending up with no constraints at all.
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

                relative_pose_constraint.append((0, 1, observation_T_from_map_to_odom, weight * np.array([10.0, 10.0, 10.0]), weight * np.array([10.0, 10.0, 10.0]), camera_in_odom_world))
        relative_pose_constraint = self._fresh_constraints(relative_pose_constraint)
        optimized_parameters = pose_graph_solve(optimized_parameters, relative_pose_constraint, constant_pose_index_dict, max_iteration_num = 1000)
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
        # Capture speed (m/s) near the robot; +inf when off-path/unknown (speed_cap's
        # own "no cap" sentinel) -> planning's isfinite guard treats it as no data and
        # falls back to vx_max, so publish it straight through.
        cap = self.speed_index.speed_cap(pose_in_map[:3, 3]) if self.speed_index else float('inf')
        self.speed_cap_pub.publish(Float32(data=float(cap)))

        poi = self.pois[self.poi_index]
        pos = pose_in_map[:3, 3]
        # Arrival is measured from the CAMERA pose, not the control center. Shifting it
        # back to the control center is more literally correct ("did the body reach the
        # POI") but it costs cam_offset (0.30 m) of extra approach before arrival fires,
        # and that margin is what keeps the robot out of the planner's near-goal dead
        # zone: with a camera reference, arrival triggers while the control center is
        # still ~0.8 m out, well before the trajectory lattice starts selecting vx=0.
        # Measuring from the camera declares arrival early; that is the point.
        # A POI with an authored heading is parked ON, not near: the mission turns to
        # that heading where it stops, so stopping 0.5m out puts the turn in the wrong
        # place. Everything else keeps the loose radius and the margin it buys (above).
        arrive_m = (_ARRIVE_HEADING_M if self.poi_index in self.poi_has_heading
                    else _ARRIVE_M)
        inside = (np.linalg.norm(poi[:2] - pos[:2]) < arrive_m
                  and abs(poi[2] - pos[2]) < 2.0)
        # Confirmed, not sampled: see _ARRIVE_TICKS.
        self._arrive_ticks = (self._arrive_ticks + 1) if inside else 0
        if inside and self._arrive_ticks >= _ARRIVE_TICKS:
            # Unconditional: this message is the only arrival edge consumers get, so
            # gating it on _leg_initial_length (i.e. "this leg published progress at
            # least once") silently loses the arrival for a POI the robot is ALREADY
            # standing at when the batch lands — no path is ever planned, so the
            # length stays None. The agent-side handoff/mission then waits out its
            # whole leg timeout on a leg that is already done.
            self.nav_progress_pub.publish(String(data=json.dumps({
                "poi_index": self.poi_index,
                # The arrival edge, said in a word. `percent` is path progress and
                # reaches 100 whenever the robot is at the END OF THE PATH -- a
                # replan, a path that doubles back near the robot, or a pose
                # correction that snaps the projection forward all get there
                # without the robot being anywhere near the POI. A consumer that
                # keyed on percent ended legs mid-route (pilot did).
                "arrived": True,
                "percent": 100.0,
                "path_remaining_m": 0.0,
                "path_total_m": round(self._leg_initial_length or 0.0, 2),
                "estimated_remaining_s": 0.0,
            })))
            self.poi_index += 1
            self._arrive_ticks = 0
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

        # False, not absent: a consumer can then tell "this build says when it
        # arrives" from "this build never says", and only fall back for the latter.
        self.nav_progress_pub.publish(String(data=json.dumps({
            "arrived": False,
            "poi_index": self.poi_index,
            "percent": round(percent, 1),
            "path_remaining_m": round(remaining_length, 2),
            "path_total_m": round(initial, 2),
            "estimated_remaining_s": round(estimated_remaining_s, 1),
        })))

        lookahead_m = lookahead_distance_m(cap)
        accumulated_distance = 0.0
        start_point = pos[:3]
        target_position = paths[-1]
        for i in range(closest_idx, len(paths) - 1):
            accumulated_distance += np.linalg.norm(paths[i][:2] - start_point[:2])
            if accumulated_distance > lookahead_m:
                target_position = paths[i]
                break
            start_point = paths[i]

        T = self.latest_odom_pose @ np.linalg.inv(pose_in_map)
        target_position_in_odom = T[:3, :3] @ target_position + T[:3, 3]
        dummy_pose = np.eye(4)
        dummy_pose[:3, 3] = target_position_in_odom
        self.target_pose_pub.publish(np2msg(dummy_pose, self.get_clock().now().to_msg(), "world", "camera"))

        self.tf_broadcaster.sendTransform(np2tf(T, self.get_clock().now().to_msg(), "world", "map"))

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
