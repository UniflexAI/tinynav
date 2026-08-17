import rclpy
import os
import time
from datetime import datetime
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool, String
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
from tinynav.core.vlad import compute_vlad, find_loop_vlad
import einops
from tinynav.core.build_map_node import OdomPoseRecorder
from tinynav.core.superpoint_bow import SUPERPOINT_BOW_INDEX_FILENAME, SuperPointBoWRetriever
logger = logging.getLogger(__name__)

_RTK_MAP_POSE_MAX_AGE_S = float(os.environ.get("TINYNAV_RTK_MAP_POSE_MAX_AGE_S", "1.0"))

# RTK antenna (tail) forward to camera (head) along the body: 0.30 + 0.35 m. In
# MAP units, ~6% larger than metres under the calibration Sim3, hence 0.70.
_RTK_ANTENNA_TO_CAMERA = float(os.environ.get("TINYNAV_RTK_ANTENNA_TO_CAMERA", "0.70"))


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



def _xy_segment_lengths(nodes: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(nodes, axis=0)[:, :2], axis=1)


def _interpolate_polyline(nodes: np.ndarray, cumulative: np.ndarray, distance: float) -> np.ndarray:
    distance = float(np.clip(distance, 0.0, cumulative[-1]))
    idx = int(np.searchsorted(cumulative, distance, side="right") - 1)
    idx = min(max(idx, 0), len(nodes) - 2)
    seg_len = cumulative[idx + 1] - cumulative[idx]
    if seg_len < 1e-6:
        return nodes[idx].copy()
    ratio = (distance - cumulative[idx]) / seg_len
    return nodes[idx] + ratio * (nodes[idx + 1] - nodes[idx])


def _drop_reversal_cusps(nodes: np.ndarray, reversal_angle_threshold_rad: float) -> np.ndarray:
    """Remove obvious backtrack-then-forward cusps from the target-selection path.

    These cusps can appear in the global SDF path as a short reverse step followed by
    a hard turn. They are bad local targets: stopping before them makes the robot aim
    at the backtrack, while following them can induce a sharp correction. For target
    pose selection we skip over them; the published global plan is left unchanged for
    debugging.
    """
    pruned = np.asarray(nodes, dtype=np.float64).copy()
    changed = True
    while changed and len(pruned) > 2:
        changed = False
        keep = np.ones(len(pruned), dtype=bool)
        for i in range(1, len(pruned) - 1):
            v_prev = pruned[i, :2] - pruned[i - 1, :2]
            v_next = pruned[i + 1, :2] - pruned[i, :2]
            prev_norm = float(np.linalg.norm(v_prev))
            next_norm = float(np.linalg.norm(v_next))
            if prev_norm < 1e-6 or next_norm < 1e-6:
                keep[i] = False
                changed = True
                continue
            cos_angle = float(np.clip(np.dot(v_prev, v_next) / (prev_norm * next_norm), -1.0, 1.0))
            if float(np.arccos(cos_angle)) >= reversal_angle_threshold_rad:
                keep[i] = False
                changed = True
        pruned = pruned[keep]
    return pruned


def select_target_position_on_path(
    path_points: np.ndarray,
    current_position: np.ndarray,
    lookahead_distance: float,
    turn_angle_threshold_rad: float = np.deg2rad(40.0),
    reversal_angle_threshold_rad: float = np.deg2rad(120.0),
    turn_stop_margin: float = 0.25,
    min_turn_distance: float = 0.3,
    turn_window_distance: float = 0.4,
    outside_offset_m: float = 0.0,
) -> np.ndarray:
    """Pick a local-planner target on the global path without looking past sharp turns.

    The local planner only receives a point target, not the whole global plan. If the
    target is placed beyond a stair landing / corridor corner, it can cut the corner
    toward unseen rails or walls. This helper caps the target before the first strong
    direction change, while keeping the old distance-based lookahead on straight
    segments. Turn detection uses a short polyline window so voxel-level zigzags do
    not look like real corners.

    When a turn is found and outside_offset_m > 0, the stop point is nudged laterally
    to the outside of the bend (left turn -> offset right) so constant-curvature
    local trajectories are pulled away from the inner corner.
    """
    pts = np.asarray(path_points, dtype=np.float64)
    curr = np.asarray(current_position, dtype=np.float64)
    if len(pts) == 0:
        return curr
    if len(pts) == 1:
        return pts[0]

    nodes = _drop_reversal_cusps(
        np.vstack([curr, pts]),
        reversal_angle_threshold_rad=reversal_angle_threshold_rad,
    )
    seg_lengths = _xy_segment_lengths(nodes)
    keep = np.concatenate([[True], seg_lengths > 1e-6])
    nodes = nodes[keep]
    if len(nodes) == 1:
        return nodes[0]
    seg_lengths = _xy_segment_lengths(nodes)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = float(cumulative[-1])
    target_distance = min(float(lookahead_distance), total_length)

    for i in range(1, len(nodes) - 1):
        d = float(cumulative[i])
        if d < min_turn_distance or d >= target_distance:
            continue
        prev_point = _interpolate_polyline(nodes, cumulative, max(0.0, d - turn_window_distance))
        next_point = _interpolate_polyline(nodes, cumulative, min(total_length, d + turn_window_distance))
        v_prev = nodes[i, :2] - prev_point[:2]
        v_next = next_point[:2] - nodes[i, :2]
        prev_norm = float(np.linalg.norm(v_prev))
        next_norm = float(np.linalg.norm(v_next))
        if prev_norm < turn_window_distance * 0.5 or next_norm < turn_window_distance * 0.5:
            continue
        cos_angle = float(np.clip(np.dot(v_prev, v_next) / (prev_norm * next_norm), -1.0, 1.0))
        angle = float(np.arccos(cos_angle))
        if angle >= turn_angle_threshold_rad:
            stop_distance = max(0.0, d - turn_stop_margin)
            stop = _interpolate_polyline(nodes, cumulative, stop_distance)
            if outside_offset_m <= 0.0:
                return stop
            # Left-handed normal of approach tangent; turn_sign > 0 means left turn.
            tangent = v_prev / prev_norm
            n_left = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            turn_sign = float(np.sign(v_prev[0] * v_next[1] - v_prev[1] * v_next[0]))
            if turn_sign == 0.0:
                return stop
            n_out = -turn_sign * n_left
            out = stop.copy()
            out[:2] = stop[:2] + n_out * float(outside_offset_m)
            return out

    return _interpolate_polyline(nodes, cumulative, target_distance)

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

# SDF bands that bucket the A* open set. search_within_sdf_map drains bucket 0
# before it looks at bucket 1, so the endpoint snap below must land inside
# bucket 0 -- these two numbers have to stay tied together, hence one constant.
SDF_BINS = [0.15, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


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

def _segment_is_shortcut_safe(
    start: tuple,
    goal: tuple,
    sdf_map: np.ndarray,
    occupancy_map: np.ndarray,
    resolution: float,
    max_segment_m: float = 1.0,
    sdf_margin_m: float = 0.2,
) -> bool:
    """Return whether a straight shortcut stays in known-safe map cells.

    Keep this deliberately conservative: bound segment length so DWA does not see
    huge sparse path jumps, reject occupied voxels, and avoid shortcuts that cut
    far away from the demonstrated/odom corridor encoded by sdf_map.
    """
    start_np = np.asarray(start, dtype=np.float32)
    goal_np = np.asarray(goal, dtype=np.float32)
    delta = goal_np - start_np
    distance_m = float(np.linalg.norm(delta) * resolution)
    if distance_m <= 1e-6:
        return True
    if distance_m > max_segment_m:
        return False

    steps = max(1, int(np.ceil(float(np.max(np.abs(delta))))))
    max_allowed_sdf = max(float(sdf_map[start]), float(sdf_map[goal]), 0.5) + sdf_margin_m
    for t in np.linspace(0.0, 1.0, steps + 1):
        idx = tuple(np.rint(start_np + delta * t).astype(np.int32).tolist())
        if (
            idx[0] < 0
            or idx[0] >= occupancy_map.shape[0]
            or idx[1] < 0
            or idx[1] >= occupancy_map.shape[1]
            or idx[2] < 0
            or idx[2] >= occupancy_map.shape[2]
        ):
            return False
        if occupancy_map[idx] == 2:
            return False
        if not np.isfinite(sdf_map[idx]) or float(sdf_map[idx]) > max_allowed_sdf:
            return False
    return True


def shortcut_prune_path(
    path: list,
    sdf_map: np.ndarray,
    occupancy_map: np.ndarray,
    resolution: float,
    max_segment_m: float = 1.0,
    max_skip_nodes: int = 30,
    max_prune_nodes: int = 100,
) -> list:
    """Greedily remove small zig-zags near the robot using local line-of-sight.

    The global path can be very long, so keep the shortcut work bounded: prune
    only the first max_prune_nodes points and append the untouched tail. From
    each kept point, look ahead at most max_skip_nodes / max_segment_m and keep
    the farthest safe shortcut.
    """
    if len(path) <= 2:
        return path

    prune_end = min(len(path), max_prune_nodes)
    prune_path = path[:prune_end]
    tail = path[prune_end:]

    pruned = [prune_path[0]]
    i = 0
    while i < len(prune_path) - 1:
        farthest = i + 1
        upper = min(len(prune_path) - 1, i + max_skip_nodes)
        for j in range(upper, i, -1):
            if _segment_is_shortcut_safe(
                prune_path[i],
                prune_path[j],
                sdf_map,
                occupancy_map,
                resolution,
                max_segment_m=max_segment_m,
            ):
                farthest = j
                break
        pruned.append(prune_path[farthest])
        i = farthest

    return pruned + tail


def sdf_queue_index(sdf_value: float) -> int:
    for idx, threshold in enumerate(SDF_BINS):
        if sdf_value < threshold:
            return idx
    return len(SDF_BINS)


def search_within_sdf_map( start:tuple, goal:tuple, sdf_map:np.ndarray, occupancy_map:np.ndarray, resolution: float, stats: dict | None = None):
    """A* over the SDF grid.

    The open set is bucketed by SDF band and the loop always drains the lowest
    non-empty bucket first, so the goal is only reachable once its own bucket is
    being drained. Keep both endpoints in bucket 0 (see SDF_BINS) or the search
    has to exhaust every bucket-0 cell on the map before it can pop the goal.

    `stats`, when given, is filled with how much of the grid the search actually
    had to touch: `expanded` nodes and the z-layer span they covered. That is the
    number to look at when a replan takes seconds -- it says whether the search is
    crawling the 3D volume rather than the route.
    """
    start = tuple(start.flatten()) if isinstance(start, np.ndarray) else start
    goal = tuple(goal.flatten()) if isinstance(goal, np.ndarray) else goal
    sdf_bins = SDF_BINS

    def _record(visited_set, found):
        if stats is None:
            return
        zs = [v[2] for v in visited_set] or [start[2]]
        stats.update(expanded=len(visited_set), z_lo=min(zs), z_hi=max(zs),
                     z_span=max(zs) - min(zs), found=found)

    get_queue_index = sdf_queue_index

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
            _record(visited, True)
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
    _record(visited, False)
    return []

class MapNode(Node):
    def __init__(
        self,
        tinynav_db_path: str,
        tinynav_map_path: str,
        verbose_timer: bool = True,
        enable_first_done: bool = False,
        initial_map_to_odom_transform_path: str | None = None,
    ):
        """Initialization

        Args:
            tinynav_db_path (str): Directory to store output data.
            tinynav_map_path (str): Directory to load the pre-built map.
            verbose_timer (bool): Whether to use verbose timer output.
            enable_first_done (bool): If true, stop keyframe relocalization after the first success.
            initial_map_to_odom_transform_path (str | None): Path to a .npy 4x4
                T_from_map_to_odom to seed this session with (map handoff), instead of
                waiting for a fresh cold relocalization/RTK fix.
        """
        super().__init__('map_node')
        self.logger = logging.getLogger(__name__)
        self.timer_logger = self.logger.info if verbose_timer else self.logger.debug
        self.enable_first_done = enable_first_done
        self.super_point_extractor = SuperPointTRT()
        self.light_glue_matcher = LightGlueTRT()
        self.dinov2_model = Dinov2TRT()
        self.tinynav_db_path = tinynav_db_path

        self.bridge = CvBridge()
        self.first_done = False

        # subs
        self.depth_sub = Subscriber(self, Image, '/slam/keyframe_depth')
        self.keyframe_image_sub = Subscriber(self, Image, '/slam/keyframe_image')
        self.keyframe_odom_sub = Subscriber(self, Odometry, '/slam/keyframe_odom')
        # 'vio' (default) or 'ekf', live-toggled via /localization/config -- see
        # _continuous_odom_vio_callback / _continuous_odom_ekf_callback below.
        self.odom_source = 'vio'
        self.continuous_odom_sub = self.create_subscription(
            Odometry, '/slam/odometry', self._continuous_odom_vio_callback, 100)
        self.continuous_odom_ekf_sub = self.create_subscription(
            Odometry, '/slam/odometry_fused', self._continuous_odom_ekf_callback, 100)
        _localization_config_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.localization_config_sub = self.create_subscription(
            String, '/localization/config', self.localization_config_callback, _localization_config_qos)
        self.rtk_map_pose_sub = self.create_subscription(Odometry, '/rtk/map_pose', self.rtk_map_pose_callback, 20)
        self.rtk_init_status_sub = self.create_subscription(String, '/rtk/init_status', self.rtk_init_status_callback, 10)
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
        self.enable_runtime_pose_graph = False

        self.loop_similarity_threshold = 0.90
        self.loop_top_k = 1

        self.relocalization_threshold = 0.70
        self.relocalization_loop_top_k = 3
        self.relocalization_min_inlier_count = 50
        self.night_relocalization_min_match_count = 30
        self.night_relocalization_min_landmark_count = 50
        self.night_relocalization_min_inlier_count = 30
        self.relocalization_odom_prior_threshold = 3.0  # meters, skip candidates too far from odom prediction
        self.target_pose_dist_factor = self._load_target_pose_dist_factor(tinynav_map_path)
        self.select_target_position_on_path_on = self._load_select_target_position_on_path_on(tinynav_map_path)
        self.poi_distance = self._load_poi_distance(tinynav_map_path)
        self.z_disable = self._load_z_disable(tinynav_map_path)
        self.planning_dilation_cells = self._load_planning_dilation_cells(tinynav_map_path)
        self.planning_comfort_radius = self._load_planning_comfort_radius(tinynav_map_path)
        self.planning_reverse_enter_threshold = self._load_planning_reverse_enter_threshold(tinynav_map_path)
        self.planning_terrain_mode = self._load_planning_terrain_mode(tinynav_map_path)
        self.planning_only_straight_back = self._load_planning_bool(tinynav_map_path, "only_straight_back", False)
        self.planning_max_linear_speed = self._load_planning_max_linear_speed(tinynav_map_path)
        self.planning_lidar_min_votes = self._load_planning_int(tinynav_map_path, "lidar_min_votes", None, minimum=1, maximum=50)
        self.planning_lidar_min_obstacle_area_cells = self._load_planning_int(
            tinynav_map_path, "lidar_min_obstacle_area_cells", None, minimum=0, maximum=200
        )
        self.planning_lidar_score_percentile = self._load_planning_float(
            tinynav_map_path, "lidar_score_percentile", None, minimum=0.0, maximum=50.0
        )
        self.planning_lidar_collision_tolerance = self._load_planning_int(
            tinynav_map_path, "lidar_collision_tolerance", None, minimum=0, maximum=50
        )
        self.planning_trajectory_smooth_weight = self._load_planning_float(
            tinynav_map_path, "trajectory_smooth_weight", None, minimum=0.0, maximum=100.0
        )
        self.rtk_mode = self._load_rtk_mode(tinynav_map_path)

        # VLAD: load vocabulary and descriptors if available.
        self.vlad_centres = None
        self.map_vlad_descriptors = None
        self.vlad_timestamps = None
        vlad_vocab_path = f"{tinynav_map_path}/vlad_vocab.npy"
        vlad_desc_path = f"{tinynav_map_path}/vlad_descriptors.npy"
        vlad_ts_path = f"{tinynav_map_path}/vlad_timestamps.npy"
        if os.path.exists(vlad_vocab_path) and os.path.exists(vlad_desc_path) and os.path.exists(vlad_ts_path):
            self.vlad_centres = np.load(vlad_vocab_path)
            self.map_vlad_descriptors = np.load(vlad_desc_path)
            self.vlad_timestamps = np.load(vlad_ts_path)
            print(f"VLAD loaded: vocab={self.vlad_centres.shape}, "
                  f"descriptors={self.map_vlad_descriptors.shape}, "
                  f"keyframes={len(self.vlad_timestamps)}")
        else:
            print("VLAD files not found")

        os.makedirs(f"{tinynav_db_path}/nav_temp", exist_ok=True)
        self.nav_temp_db = TinyNavDB(f"{tinynav_db_path}/nav_temp", is_scratch=True)
        self.map_poses = np.load(f"{tinynav_map_path}/poses.npy", allow_pickle=True).item()
        self.map_K = np.load(f"{tinynav_map_path}/intrinsics.npy")
        self.db = TinyNavDB(tinynav_map_path, is_scratch=False)
        map_embeddings_list = []
        self.map_embeddings_idx_to_timestamp = {}
        for timestamp in self.map_poses.keys():
            embedding = self.db.get_embedding(timestamp)
            if embedding is None:
                self.get_logger().warning(f'Missing embedding for keyframe {timestamp}, skipping')
                continue
            self.map_embeddings_idx_to_timestamp[len(map_embeddings_list)] = timestamp
            map_embeddings_list.append(embedding)
        self.map_embeddings = np.stack(map_embeddings_list)
        self.relocalization_bow: SuperPointBoWRetriever | None = None
        bow_index_path = os.path.join(tinynav_map_path, SUPERPOINT_BOW_INDEX_FILENAME)
        if os.path.exists(bow_index_path):
            try:
                self.relocalization_bow = SuperPointBoWRetriever.load(bow_index_path)
                missing_timestamps = [
                    timestamp
                    for timestamp in self.relocalization_bow.timestamps
                    if timestamp not in self.map_poses
                ]
                if missing_timestamps:
                    raise RuntimeError(
                        f"BoW index contains {len(missing_timestamps)} timestamps not present in poses.npy"
                    )
                self.get_logger().info(
                    f"Using SuperPoint BoW relocalization index: "
                    f"{len(self.relocalization_bow.timestamps)} keyframes, "
                    f"{len(self.relocalization_bow.vocab) if self.relocalization_bow.vocab is not None else 0} words"
                )
            except Exception as exc:
                self.relocalization_bow = None
                self.get_logger().warning(
                    f"Failed to load SuperPoint BoW index from {bow_index_path}: {exc}. "
                    "Falling back to DINO relocalization retrieval."
                )
        else:
            if self.vlad_centres is not None and self.map_vlad_descriptors is not None and self.vlad_timestamps is not None:
                self.get_logger().info(
                    f"No {SUPERPOINT_BOW_INDEX_FILENAME} found in map. "
                    "Using VLAD relocalization retrieval."
                )
            else:
                self.get_logger().info(
                    f"No {SUPERPOINT_BOW_INDEX_FILENAME} found in map. "
                    "Using DINO relocalization retrieval."
                )
        self.occupancy_map = np.load(f"{tinynav_map_path}/occupancy_grid.npy")
        self.occupancy_map_meta = np.load(f"{tinynav_map_path}/occupancy_meta.npy")
        self.sdf_map = np.load(f"{tinynav_map_path}/sdf_map.npy")

        print(f"sdf_map.shape: {self.sdf_map.shape}")
        print(f"occupancy_map.shape: {self.occupancy_map.shape}")

        self.relocalization_poses = {}
        self.relocalization_pose_weights = {}
        self.failed_relocalizations = []

        self.T_from_map_to_odom = None
        # Map-handoff seeding: publish one /map/relocalization the moment we have an odom
        # pose to pair it with (see continuous_odom_callback), instead of waiting for a real
        # relocalization -- node_manager's _on_relocalization sets its own _localized=True
        # off that same topic, so this needs no other changes to how "localized" is tracked.
        self._seed_map_to_odom_pending = False
        if initial_map_to_odom_transform_path is not None:
            self.T_from_map_to_odom = np.load(initial_map_to_odom_transform_path)
            self._seed_map_to_odom_pending = True
            self.get_logger().info(
                f"Seeded T_from_map_to_odom from {initial_map_to_odom_transform_path} for map handoff"
            )
        self._rtk_yaw_offset = None   # map<-odom yaw offset, locked once on first RTK fix
        # RTK replace mode only: stateless xy -> map-frame z lookup over the map keyframes.
        self._rtk_kf_xy = None
        self._rtk_kf_z = None
        if self.rtk_mode == "replace":
            self._build_rtk_ground_z_lookup()
        self.latest_odom_pose = None
        self.latest_odom_stamp_msg = None
        self.latest_rtk_map_pose = None
        self.latest_rtk_map_pose_stamp_msg = None
        self.latest_rtk_map_pose_received_at = None
        self.latest_rtk_state = None
        self.latest_rtk_transform_received_at = None
        self._last_rtk_relocalization_pub_at = 0.0
        self._last_rtk_log_at = 0.0
        self.nav_refresh_timer = None
        self.target_pose_timer = self.create_timer(1.0, self.target_pose_timer_callback)

        self.pois = {}
        self.poi_meta = {}
        self.poi_index = -1
        self._nav_completed = False
        self._leg_initial_length: float | None = None
        self._leg_start_time: float | None = None
        self._speed_estimate: float | None = None
        self._current_nav_path_in_map: np.ndarray | None = None
        self._cached_global_path: np.ndarray | None = None
        self._cached_global_path_poi_index: int | None = None
        self._cached_global_path_goal: np.ndarray | None = None
        self._cached_global_path_T: np.ndarray | None = None
        self._replan_tf_translation_threshold = 0.3
        self._replan_tf_yaw_threshold = np.deg2rad(10.0)
        self._replan_path_deviation_threshold = 0.5
        self._nav_subgoals_in_map: list[np.ndarray] = []
        self._nav_subgoals_poi_index: int | None = None
        self._nav_subgoal_index = 0
        self._nav_subgoal_segment_length_m = 20.0
        self._nav_subgoal_arrival_xy_threshold = 3.0
        self._nav_subgoal_arrival_z_threshold = 2.0
        self._nav_z_clamp_max_diff_m = 2.0
        self._nav_z_clamp_sdf_xy_radius_m = 1.5
        self._nav_z_clamp_sdf_z_window_m = 1.0
        self._last_nav_z_clamp_log_time = 0.0

        self.poi_pub = self.create_publisher(Odometry, "/mapping/poi", 10)
        self.poi_change_pub = self.create_publisher(Odometry, "/mapping/poi_change", 10)
        self.nav_done_pub = self.create_publisher(Bool, '/mapping/nav_done', 10)
        self.nav_progress_pub = self.create_publisher(String, '/mapping/nav_progress', 10)

        self.current_pose_pub = self.create_publisher(Odometry, "/mapping/current_pose", 10)
        self.global_plan_pub = self.create_publisher(Path, '/mapping/global_plan', 10)
        self.final_global_plan_pub = self.create_publisher(Path, '/mapping/final_global_plan', 10)
        self.target_pose_pub = self.create_publisher(Odometry, "/control/target_pose", 10)
        planning_config_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.planning_config_pub = self.create_publisher(String, "/planning/config", planning_config_qos)

        self.tf_broadcaster = TransformBroadcaster(self)

        self._save_completed = False
        self._publish_planning_config()

    def _load_target_pose_dist_factor(self, tinynav_map_path: str) -> float:
        default_factor = 4.0
        config_path = os.path.join(tinynav_map_path, "nav_flow.json")
        if not os.path.exists(config_path):
            return default_factor
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as exc:
            self.get_logger().warning(f"Failed to read nav_flow.json: {exc}; using target_pose_dist_factor={default_factor}")
            return default_factor
        if not isinstance(config, dict):
            return default_factor
        value = config.get("target_pose_dist_factor", default_factor)
        try:
            factor = float(value)
        except (TypeError, ValueError):
            self.get_logger().warning(f"Invalid target_pose_dist_factor={value!r}; using {default_factor}")
            return default_factor
        if factor <= 0:
            self.get_logger().warning(f"Invalid target_pose_dist_factor={value!r}; using {default_factor}")
            return default_factor
        self.get_logger().info(f"Using target_pose_dist_factor={factor}")
        return factor

    def _load_select_target_position_on_path_on(self, tinynav_map_path: str) -> bool:
        default_value = False
        config_path = os.path.join(tinynav_map_path, "nav_flow.json")
        if not os.path.exists(config_path):
            return default_value
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as exc:
            self.get_logger().warning(f"Failed to read nav_flow.json: {exc}; using select_target_position_on_path_on={default_value}")
            return default_value
        if not isinstance(config, dict):
            return default_value
        value = config.get("select_target_position_on_path_on", default_value)
        if isinstance(value, bool):
            enabled = value
        elif isinstance(value, str):
            enabled = value.strip().lower() in {"1", "true", "yes", "on"}
        elif isinstance(value, (int, float)):
            enabled = bool(value)
        else:
            self.get_logger().warning(f"Invalid select_target_position_on_path_on={value!r}; using {default_value}")
            return default_value
        self.get_logger().info(f"Using select_target_position_on_path_on={enabled}")
        return enabled

    def _load_poi_distance(self, tinynav_map_path: str) -> float:
        default_value = 0.5
        config_path = os.path.join(tinynav_map_path, "nav_flow.json")
        if not os.path.exists(config_path):
            return default_value
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as exc:
            self.get_logger().warning(f"Failed to read nav_flow.json: {exc}; using poi_distance={default_value}")
            return default_value
        if not isinstance(config, dict):
            return default_value
        if "poi_distance" not in config:
            return default_value
        value = config.get("poi_distance", default_value)
        try:
            poi_distance = float(value)
        except (TypeError, ValueError):
            self.get_logger().warning(f"Invalid poi_distance={value!r}; using {default_value}")
            return default_value
        if poi_distance <= 0:
            self.get_logger().warning(f"Invalid poi_distance={value!r}; using {default_value}")
            return default_value
        self.get_logger().info(f"Using poi_distance={poi_distance}")
        return poi_distance

    def _load_z_disable(self, tinynav_map_path: str) -> bool:
        default_value = False
        config_path = os.path.join(tinynav_map_path, "nav_flow.json")
        if not os.path.exists(config_path):
            return default_value
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as exc:
            self.get_logger().warning(f"Failed to read nav_flow.json: {exc}; using z_disable={default_value}")
            return default_value
        if not isinstance(config, dict):
            return default_value
        value = config.get("z_disable", default_value)
        if isinstance(value, bool):
            enabled = value
        elif isinstance(value, str):
            enabled = value.strip().lower() in {"1", "true", "yes", "on"}
        elif isinstance(value, (int, float)):
            enabled = bool(value)
        else:
            self.get_logger().warning(f"Invalid z_disable={value!r}; using {default_value}")
            return default_value
        self.get_logger().info(f"Using z_disable={enabled}")
        return enabled

    def _load_planning_dilation_cells(self, tinynav_map_path: str) -> int:
        default_value = 0
        config_path = os.path.join(tinynav_map_path, "nav_flow.json")
        if not os.path.exists(config_path):
            return default_value
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as exc:
            self.get_logger().warning(f"Failed to read nav_flow.json: {exc}; using planning.dilation_cells={default_value}")
            return default_value
        if not isinstance(config, dict):
            return default_value
        planning_config = config.get("planning", {})
        if not isinstance(planning_config, dict):
            return default_value
        value = planning_config.get("dilation_cells", default_value)
        try:
            dilation_cells = int(value)
        except (TypeError, ValueError):
            self.get_logger().warning(f"Invalid planning.dilation_cells={value!r}; using {default_value}")
            return default_value
        dilation_cells = max(0, min(20, dilation_cells))
        self.get_logger().info(f"Using planning.dilation_cells={dilation_cells}")
        return dilation_cells

    def _load_planning_comfort_radius(self, tinynav_map_path: str) -> float | None:
        config_path = os.path.join(tinynav_map_path, "nav_flow.json")
        if not os.path.exists(config_path):
            return None
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as exc:
            self.get_logger().warning(f"Failed to read nav_flow.json: {exc}; not overriding planning.comfort_radius")
            return None
        if not isinstance(config, dict):
            return None
        planning_config = config.get("planning", {})
        if not isinstance(planning_config, dict) or "comfort_radius" not in planning_config:
            return None
        value = planning_config.get("comfort_radius")
        try:
            comfort_radius = float(value)
        except (TypeError, ValueError):
            self.get_logger().warning(f"Invalid planning.comfort_radius={value!r}; not overriding")
            return None
        comfort_radius = max(0.0, min(3.0, comfort_radius))
        self.get_logger().info(f"Using planning.comfort_radius={comfort_radius:.2f}")
        return comfort_radius

    def _load_planning_reverse_enter_threshold(self, tinynav_map_path: str) -> float | None:
        config_path = os.path.join(tinynav_map_path, "nav_flow.json")
        if not os.path.exists(config_path):
            return None
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as exc:
            self.get_logger().warning(
                f"Failed to read nav_flow.json: {exc}; not overriding planning.reverse_enter_threshold"
            )
            return None
        if not isinstance(config, dict):
            return None
        planning_config = config.get("planning", {})
        if not isinstance(planning_config, dict) or "reverse_enter_threshold" not in planning_config:
            return None
        value = planning_config.get("reverse_enter_threshold")
        try:
            reverse_enter_threshold = float(value)
        except (TypeError, ValueError):
            self.get_logger().warning(f"Invalid planning.reverse_enter_threshold={value!r}; not overriding")
            return None
        reverse_enter_threshold = max(0.0, min(2.0, reverse_enter_threshold))
        self.get_logger().info(f"Using planning.reverse_enter_threshold={reverse_enter_threshold:.2f}")
        return reverse_enter_threshold

    def _load_planning_terrain_mode(self, tinynav_map_path: str) -> str:
        default_value = "normal"
        config_path = os.path.join(tinynav_map_path, "nav_flow.json")
        if not os.path.exists(config_path):
            return default_value
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as exc:
            self.get_logger().warning(f"Failed to read nav_flow.json: {exc}; using planning.terrain_mode={default_value}")
            return default_value
        if not isinstance(config, dict):
            return default_value
        planning_config = config.get("planning", {})
        if not isinstance(planning_config, dict):
            return default_value
        terrain_mode = str(planning_config.get("terrain_mode", default_value)).strip().lower()
        if terrain_mode not in {"normal", "stairs"}:
            self.get_logger().warning(f"Invalid planning.terrain_mode={terrain_mode!r}; using {default_value}")
            return default_value
        self.get_logger().info(f"Using planning.terrain_mode={terrain_mode}")
        return terrain_mode

    def _load_planning_max_linear_speed(self, tinynav_map_path: str) -> float:
        default_value = 0.5
        config_path = os.path.join(tinynav_map_path, "nav_flow.json")
        if not os.path.exists(config_path):
            return default_value
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as exc:
            self.get_logger().warning(f"Failed to read nav_flow.json: {exc}; using planning.max_linear_speed={default_value}")
            return default_value
        if not isinstance(config, dict):
            return default_value
        planning_config = config.get("planning", {})
        if not isinstance(planning_config, dict):
            return default_value
        value = planning_config.get("max_linear_speed", default_value)
        try:
            max_linear_speed = float(value)
        except (TypeError, ValueError):
            self.get_logger().warning(f"Invalid planning.max_linear_speed={value!r}; using {default_value}")
            return default_value
        max_linear_speed = max(0.05, min(1.5, max_linear_speed))
        self.get_logger().info(f"Using planning.max_linear_speed={max_linear_speed:.2f}")
        return max_linear_speed

    def _load_planning_config(self, tinynav_map_path: str) -> dict:
        config_path = os.path.join(tinynav_map_path, "nav_flow.json")
        if not os.path.exists(config_path):
            return {}
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as exc:
            self.get_logger().warning(f"Failed to read nav_flow.json: {exc}; ignoring optional planning config")
            return {}
        if not isinstance(config, dict):
            return {}
        planning_config = config.get("planning", {})
        return planning_config if isinstance(planning_config, dict) else {}

    def _load_planning_float(
        self,
        tinynav_map_path: str,
        key: str,
        default_value: float | None,
        *,
        minimum: float,
        maximum: float,
    ) -> float | None:
        planning_config = self._load_planning_config(tinynav_map_path)
        if key not in planning_config:
            return default_value
        value = planning_config.get(key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            self.get_logger().warning(f"Invalid planning.{key}={value!r}; using {default_value}")
            return default_value
        parsed = max(minimum, min(maximum, parsed))
        self.get_logger().info(f"Using planning.{key}={parsed}")
        return parsed

    def _load_planning_int(
        self,
        tinynav_map_path: str,
        key: str,
        default_value: int | None,
        *,
        minimum: int,
        maximum: int,
    ) -> int | None:
        value = self._load_planning_float(
            tinynav_map_path,
            key,
            None if default_value is None else float(default_value),
            minimum=float(minimum),
            maximum=float(maximum),
        )
        return None if value is None else int(value)

    def _load_planning_bool(self, tinynav_map_path: str, key: str, default_value: bool) -> bool:
        planning_config = self._load_planning_config(tinynav_map_path)
        if key not in planning_config:
            return default_value
        value = planning_config.get(key)
        if isinstance(value, bool):
            parsed = value
        elif isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                parsed = True
            elif normalized in {"false", "0", "no", "off"}:
                parsed = False
            else:
                self.get_logger().warning(f"Invalid planning.{key}={value!r}; using {default_value}")
                return default_value
        else:
            self.get_logger().warning(f"Invalid planning.{key}={value!r}; using {default_value}")
            return default_value
        self.get_logger().info(f"Using planning.{key}={parsed}")
        return parsed

    def _load_rtk_mode(self, tinynav_map_path: str) -> str:
        config_path = os.path.join(tinynav_map_path, "nav_flow.json")
        if not os.path.exists(config_path):
            return "off"
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as exc:
            self.get_logger().warning(f"Failed to read nav_flow.json: {exc}; using rtk.mode=off")
            return "off"
        if not isinstance(config, dict):
            return "off"
        rtk_config = config.get("rtk", {})
        if isinstance(rtk_config, bool):
            mode = "replace" if rtk_config else "off"
        elif isinstance(rtk_config, str):
            mode = rtk_config.strip().lower()
        elif isinstance(rtk_config, dict):
            mode = str(rtk_config.get("mode", "off")).strip().lower()
        else:
            self.get_logger().warning(f"Invalid nav_flow rtk config: {rtk_config!r}; using off")
            return "off"
        if mode in {"replace", "on", "true", "1", "yes"}:
            # rtk comment out
            # if not self._rtk_time_gate_open(datetime.now()):
            #     self.get_logger().info("RTK configured on but outside the time window at startup; using off")
            #     return "off"
            self.get_logger().info("Using RTK map pose replacement")
            return "replace"
        return "off"

    @staticmethod
    def _rtk_time_gate_open(now: datetime) -> bool:
        # Oct-Mar: effective after 18:00. Apr-Sep: effective after 19:00.
        if now.month in (10, 11, 12, 1, 2, 3):
            return now.hour >= 18
        return now.hour >= 19

    def _publish_planning_config(self):
        msg = String()
        config = {
            "dilation_cells": self.planning_dilation_cells,
            "terrain_mode": self.planning_terrain_mode,
            "only_straight_back": self.planning_only_straight_back,
            "max_linear_speed": self.planning_max_linear_speed,
        }
        if self.planning_comfort_radius is not None:
            config["comfort_radius"] = self.planning_comfort_radius
        if self.planning_reverse_enter_threshold is not None:
            config["reverse_enter_threshold"] = self.planning_reverse_enter_threshold
        if self.planning_lidar_min_votes is not None:
            config["lidar_min_votes"] = self.planning_lidar_min_votes
        if self.planning_lidar_min_obstacle_area_cells is not None:
            config["lidar_min_obstacle_area_cells"] = self.planning_lidar_min_obstacle_area_cells
        if self.planning_lidar_score_percentile is not None:
            config["lidar_score_percentile"] = self.planning_lidar_score_percentile
        if self.planning_lidar_collision_tolerance is not None:
            config["lidar_collision_tolerance"] = self.planning_lidar_collision_tolerance
        if self.planning_trajectory_smooth_weight is not None:
            config["trajectory_smooth_weight"] = self.planning_trajectory_smooth_weight
        msg.data = json.dumps(config)
        self.planning_config_pub.publish(msg)
        self.get_logger().info(f"Published /planning/config: {msg.data}")

    def pois_callback(self, msg: String):
        self.get_logger().info("Received POIs from planner: " + msg.data)
        try:
            raw_pois = json.loads(msg.data)

            pois_dict = {}
            poi_meta = {}
            keys = sorted([int(key) for key in raw_pois.keys()])
            for index, key in enumerate(keys):
                raw_poi = raw_pois[str(key)]
                pois_dict[index] = np.array(raw_poi["position"])
                poi_meta[index] = {
                    "id": raw_poi.get("id", key),
                    "name": raw_poi.get("name"),
                }
            self.pois = pois_dict
            self.poi_meta = poi_meta

            if not self.pois:
                self.poi_index = -1
                self._clear_global_path_cache()
                # Signal planning_node to clear target_pose so it stops publishing paths
                dummy_pose = np.eye(4)
                self.poi_change_pub.publish(np2msg(dummy_pose, self.get_clock().now().to_msg(), "world", "map"))
                self.poi_meta = {}
                self.get_logger().info("POIs cleared, navigation cancelled")
                return

            self.poi_index = min(0, len(self.pois) - 1)
            self._nav_completed = False
            self._leg_initial_length = None
            self._leg_start_time = None
            self._speed_estimate = None
            self._clear_global_path_cache()
            self.get_logger().info(f"Parsed POIs: {self.pois}")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse POIs JSON: {e}")
            self.pois = {}
            self._clear_global_path_cache()
            self.poi_meta = {}

    def _clear_global_path_cache(self):
        self._current_nav_path_in_map = None
        self._cached_global_path = None
        self._cached_global_path_poi_index = None
        self._cached_global_path_goal = None
        self._cached_global_path_T = None
        self._nav_subgoals_in_map = []
        self._nav_subgoals_poi_index = None
        self._nav_subgoal_index = 0
        self._publish_path_in_map(self.global_plan_pub, [])
        self._publish_path_in_map(self.final_global_plan_pub, [])

    def _publish_path_in_map(self, publisher, path_in_map) -> None:
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for x, y, z in path_in_map:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        publisher.publish(path_msg)

    def _nav_progress_payload(self, *, percent: float, path_remaining_m: float,
                              path_total_m: float, estimated_remaining_s: float) -> dict:
        meta = self.poi_meta.get(self.poi_index, {})
        return {
            "poi_index": self.poi_index,  # route index in the current command queue
            "poi_id": meta.get("id"),
            "poi_name": meta.get("name"),
            "percent": percent,
            "path_remaining_m": path_remaining_m,
            "path_total_m": path_total_m,
            "estimated_remaining_s": estimated_remaining_s,
        }

    def info_callback(self, msg:CameraInfo):
        if self.K is None:
            self.get_logger().info("Camera intrinsics received.")
            self.K = np.array(msg.k).reshape(3, 3)
            fx = self.K[0, 0]
            Tx = msg.p[3]
            self.baseline = -Tx / fx
            self.destroy_subscription(self.camera_info_sub)

    def _continuous_odom_vio_callback(self, odom_msg: Odometry):
        if self.odom_source != 'vio':
            return
        self.continuous_odom_callback(odom_msg)

    def _continuous_odom_ekf_callback(self, odom_msg: Odometry):
        if self.odom_source != 'ekf':
            return
        self.continuous_odom_callback(odom_msg)

    def continuous_odom_callback(self, odom_msg: Odometry):
        self.continuous_odom_recorder.record_odometry_msg(odom_msg)
        self.latest_odom_pose, _ = msg2np(odom_msg)
        self.latest_odom_stamp_msg = odom_msg.header.stamp
        if self._seed_map_to_odom_pending:
            self._seed_map_to_odom_pending = False
            pose_in_world = np.linalg.inv(self.T_from_map_to_odom) @ self.latest_odom_pose
            self.relocation_pub.publish(np2msg(pose_in_world, self.latest_odom_stamp_msg, "world", "camera"))
            self.first_done = True
            self.get_logger().info(f"Published seeded map-handoff pose: xyz={pose_in_world[:3, 3]}")

    def localization_config_callback(self, msg: String):
        try:
            config = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Failed to parse /localization/config: {exc}")
            return
        if not isinstance(config, dict):
            return
        if "odom_source" in config:
            odom_source = config["odom_source"]
            if odom_source not in ("vio", "ekf"):
                self.get_logger().warning(f"Invalid localization odom_source: {odom_source!r} (must be 'vio' or 'ekf')")
            else:
                old = self.odom_source
                self.odom_source = odom_source
                if old != odom_source:
                    self.get_logger().info(f"Updated localization odom_source: {old} -> {odom_source}")

    def rtk_init_status_callback(self, msg: String):
        if self.rtk_mode != "replace":
            return
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(status, dict):
            self.latest_rtk_state = status.get("state")

    def rtk_map_pose_callback(self, msg: Odometry):
        if self.rtk_mode != "replace":
            return
        # /rtk/map_pose arrives per fix (~10 Hz). A gap much larger than that means
        # this callback was queued behind a slow replan, i.e. localization went
        # blind for that long -- not that RTK stopped.
        self._log_callback_gap("rtk_map_pose", "_last_rtk_cb_at", 0.2)
        self.latest_rtk_map_pose, _ = msg2np(msg)
        self.latest_rtk_map_pose_stamp_msg = msg.header.stamp
        self.latest_rtk_map_pose_received_at = time.monotonic()
        self._update_transform_from_rtk_map_pose()

    def _has_fresh_rtk_map_pose(self) -> bool:
        if self.rtk_mode != "replace":
            return False
        if self.latest_rtk_map_pose is None or self.latest_rtk_map_pose_received_at is None:
            return False
        if time.monotonic() - self.latest_rtk_map_pose_received_at > _RTK_MAP_POSE_MAX_AGE_S:
            return False
        # /rtk/map_pose is only published after heading is ready, but honor the
        # handshake status when it is available so WAIT_FIX/NEED_YAW_INIT cannot
        # accidentally reuse a stale pose.
        if self.latest_rtk_state is not None and self.latest_rtk_state != "ACTIVE":
            return False
        return True

    def _has_active_rtk_transform(self) -> bool:
        if self.latest_rtk_transform_received_at is None:
            return False
        return time.monotonic() - self.latest_rtk_transform_received_at <= _RTK_MAP_POSE_MAX_AGE_S

    def _update_transform_from_rtk_map_pose(self) -> bool:
        if not self._has_fresh_rtk_map_pose():
            return False
        if self.latest_odom_pose is None:
            return False

        # T_from_map_to_odom is the map-world -> odom-world transform. Both worlds
        # are gravity-aligned (Z-up), so it MUST be a pure yaw about Z plus a
        # planar translation. Do NOT build it as odom @ inv(rtk_pose): the SLAM
        # odom rotation is camera-optical (forward = optical +Z) while rtk_map_pose
        # carries a planar-yaw quaternion -- multiplying the two mixes conventions
        # and contaminates the rotation, which skews every map->odom target
        # transform so the robot veers. Instead take the yaw offset from the two
        # headings and anchor the translation to the current position match.
        odom_pose = self.latest_odom_pose
        rtk_pose = self.latest_rtk_map_pose
        psi_map = np.arctan2(rtk_pose[1, 0], rtk_pose[0, 0])
        if self._rtk_yaw_offset is None:
            # The map-world -> odom-world yaw offset is a physical constant: File
            # B's heading (psi_map) and this heading_odom both derive from the SAME
            # /slam/odometry, so their difference does not change over time. Fix it
            # ONCE on the first fix. Recomputing it every fix let the ~0.1 s
            # sampling skew between File B's psi_map and this heading_odom (worst
            # while turning) perturb the yaw each fix -> the whole map frame
            # rotated and the global yaw/path jittered at the fix rate.
            fwd = odom_pose[:3, :3] @ np.array([0.0, 0.0, 1.0])   # camera forward, odom world
            heading_odom = np.arctan2(fwd[1], fwd[0])
            self._rtk_yaw_offset = heading_odom - psi_map
        phi = self._rtk_yaw_offset
        c, s = np.cos(phi), np.sin(phi)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        # odom_pose is the CAMERA, rtk_pose the ANTENNA: equating them shifted
        # every target 0.65 m along the heading, so near a waypoint the goal
        # orbited the real one. Does not fix the separate psi_map heading error.
        rtk_xyz = rtk_pose[:3, 3].copy()
        rtk_xyz[0] += _RTK_ANTENNA_TO_CAMERA * np.cos(psi_map)
        rtk_xyz[1] += _RTK_ANTENNA_TO_CAMERA * np.sin(psi_map)
        T = np.eye(4)
        T[:3, :3] = Rz
        T[:3, 3] = odom_pose[:3, 3] - Rz @ rtk_xyz   # translation still tracks RTK each fix
        self.T_from_map_to_odom = T
        self.latest_rtk_transform_received_at = time.monotonic()
        self.first_done = True

        now = time.monotonic()
        if now - self._last_rtk_relocalization_pub_at >= 0.5:
            stamp = self.latest_rtk_map_pose_stamp_msg or self.get_clock().now().to_msg()
            # Publish inv(T) @ odom, NOT latest_rtk_map_pose. The translation is the
            # same either way (T was just anchored so that inv(T) @ odom reproduces
            # the RTK position exactly), but the rotation convention differs and the
            # consumers assume the camera one: File B sends a pure yaw-about-Z
            # quaternion (qx = qy = 0), while the web arrow derives heading by
            # projecting the body +Z axis onto the map XY plane
            # (node_manager._odom_to_dict, matching SLAM's camera-optical poses).
            # For a pure yaw quaternion that projection is identically (0, 0), so the
            # arrow fell back to 0 deg and flickered against the correct heading that
            # /mapping/current_pose_in_map publishes at 1 Hz -- and sat at 0 deg
            # outright before navigation starts, when that topic is not published yet.
            self.relocation_pub.publish(
                np2msg(np.linalg.inv(T) @ odom_pose, stamp, "world", "camera")
            )
            self._last_rtk_relocalization_pub_at = now
        if now - self._last_rtk_log_at >= 5.0:
            xy = self.latest_rtk_map_pose[:2, 3]
            self.get_logger().info(
                f"Using /rtk/map_pose for localization: x={xy[0]:.2f}, y={xy[1]:.2f}"
            )
            self._last_rtk_log_at = now
        return True

    def nav_refresh_timer_callback(self):
        if self.latest_odom_pose is None:
            return
        self.try_publish_nav_path(
            timestamp=None,
            pose_in_origin_odom=self.latest_odom_pose,
            stamp_msg=self.latest_odom_stamp_msg,
        )

    def _log_callback_gap(self, name: str, attr: str, expected_s: float) -> None:
        """Report when a callback fires far later than its period.

        map_node runs on the default single-threaded executor, so a slow replan
        stalls every other callback in this node -- RTK pose updates included.
        That stall is invisible in the per-stage timings; it only shows up as the
        gap between consecutive callbacks, which is what this records.
        """
        now = time.monotonic()
        previous = getattr(self, attr, None)
        setattr(self, attr, now)
        if previous is None:
            return
        gap = now - previous
        if gap > expected_s * 1.5:
            self.get_logger().warning(
                f"cb_gap name={name} gap_s={gap:.2f} expected_s={expected_s:.2f} "
                f"late_s={gap - expected_s:.2f}"
            )

    def target_pose_timer_callback(self):
        self._log_callback_gap("target_pose_timer", "_last_target_tick_at", 1.0)
        if self.latest_odom_pose is None:
            return
        if self.T_from_map_to_odom is None:
            return
        self.try_publish_nav_path(
            timestamp=None,
            pose_in_origin_odom=self.latest_odom_pose,
            stamp_msg=self.latest_odom_stamp_msg,
            force_replan=True,
        )
        if self._current_nav_path_in_map is None:
            return
        self._publish_target_pose_from_path(
            self._current_nav_path_in_map,
            self.latest_odom_pose,
            self.latest_odom_stamp_msg,
        )

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
        # RTK replace in nav_flow only means RTK is allowed. While RTK is not ACTIVE
        # (WAIT_FIX / NEED_YAW_INIT), visual relocalization must still be able to
        # localize the map and let navigation start. Once /rtk/map_pose is ACTIVE,
        # RTK owns T_from_map_to_odom and visual relocalization is suppressed.
        if self._has_active_rtk_transform():
            return
        image = self.bridge.imgmsg_to_cv2(keyframe_image_msg, desired_encoding="mono8")

        if not (self.enable_first_done and self.first_done):
            success, pose_in_world = self.keyframe_relocalization(keyframe_image_msg.header.stamp, image)
            if success:
                self.compute_transform_from_map_to_odom()
                self.first_done = True

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
        odom, _ = msg2np(keyframe_odom_msg)

        if not self.enable_runtime_pose_graph:
            self.odom[keyframe_odom_timestamp] = odom
            self.pose_graph_used_pose[keyframe_image_timestamp] = odom
            self.last_keyframe_timestamp = keyframe_odom_timestamp
            return

        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
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
            find_loop_and_pose_graph(keyframe_image_timestamp)
            self.pose_graph_trajectory_publish(keyframe_image_timestamp)
        self.last_keyframe_timestamp = keyframe_odom_timestamp
        self.last_keyframe_image = image


    def get_embeddings(self, image: np.ndarray) -> np.ndarray:
        # shape: (1, 768)
        return asyncio.run(self.dinov2_model.infer(image))

    def get_vlad_descriptor(self, image: np.ndarray) -> np.ndarray | None:
        """Compute VLAD descriptor for a query image."""
        if self.vlad_centres is None:
            return None
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

    def relocalize_with_depth(self, keyframe: np.ndarray, keyframe_features: dict, K: np.ndarray | None, current_odom_pose: np.ndarray | None = None) -> tuple[bool, np.ndarray, float]:
        if K is None:
            return False, np.eye(4), -np.inf
        min_match_count, min_landmark_count, min_inlier_count = self._get_relocalization_pnp_thresholds()

        # Prefer DINOv2 patch VLAD if the map has a VLAD vocabulary/index.
        # Fall back to SuperPoint BoW if available, then finally to DINO global embedding.
        if self.vlad_centres is not None and self.map_vlad_descriptors is not None and self.vlad_timestamps is not None:
            query_vlad = self.get_vlad_descriptor(keyframe)
            if query_vlad is None:
                return False, np.eye(4), -np.inf
            idx_and_similarity_array = find_loop_vlad(
                query_vlad,
                self.map_vlad_descriptors,
                -1.0,
                self.relocalization_loop_top_k,
            )
            max_similarity = max((s for _, s in idx_and_similarity_array), default=0.0)
            if len(idx_and_similarity_array) == 0:
                print(f"VLAD: not enough similar embeddings, max_similarity: {max_similarity}")
                return False, np.eye(4), -np.inf
            candidate_timestamps = [
                int(self.vlad_timestamps[idx_in_map])
                for idx_in_map, _similarity in idx_and_similarity_array
            ]
        elif self.relocalization_bow is not None:
            candidate_timestamps = [
                self.relocalization_bow.timestamps[idx_in_map]
                for idx_in_map, _bow_score in self.relocalization_bow.query(
                    keyframe_features,
                    self.relocalization_loop_top_k,
                )
            ]
            if len(candidate_timestamps) == 0:
                print("not enough SuperPoint BoW candidates to relocalize")
                return False, np.eye(4), -np.inf
        else:
            query_embedding = self.get_embeddings(keyframe)
            query_embedding_normed = query_embedding / np.linalg.norm(query_embedding)
            idx_and_similarity_array = find_loop(
                query_embedding_normed,
                self.map_embeddings,
                self.relocalization_threshold,
                self.relocalization_loop_top_k,
            )
            max_similarity = np.max([similarity for _, similarity in idx_and_similarity_array]) if len(idx_and_similarity_array) > 0 else 0
            if len(idx_and_similarity_array) == 0:
                print(f"not enough similar embeddings to relocalize, {len(idx_and_similarity_array)}, max_similarity : {max_similarity}")
                return False, np.eye(4), -np.inf
            candidate_timestamps = [
                self.map_embeddings_idx_to_timestamp[idx_in_map]
                for idx_in_map, _similarity in idx_and_similarity_array
            ]

        pnp_candidates = []
        for timestamp_in_map in candidate_timestamps:
            reference_keyframe_pose = self.map_poses[timestamp_in_map]

            # Odom position prior filter: skip candidates too far from predicted position
            if self.T_from_map_to_odom is not None and current_odom_pose is not None:
                current_pose_in_map = np.linalg.inv(self.T_from_map_to_odom) @ current_odom_pose
                xy_dist = np.linalg.norm(reference_keyframe_pose[:3, 3][:2] - current_pose_in_map[:2, 3])
                if xy_dist > self.relocalization_odom_prior_threshold:
                    print(f"candidate too far from odom prediction: {xy_dist:.2f}m > {self.relocalization_odom_prior_threshold}m, skipping")
                    continue

            reference_depth, _, reference_features, _, _ = self.db.get_depth_embedding_features_images(timestamp_in_map)
            reference_matched_keypoints, keyframe_matched_keypoints, matches = self.match_keypoints(reference_features, keyframe_features)
            if len(matches) < min_match_count:
                print(f"not enough matched features to relocalize, {len(matches)} < {min_match_count}")
                continue

            point_3d_in_world, inliers = self.keypoint_with_depth_to_3d(reference_matched_keypoints, reference_depth, reference_keyframe_pose, self.map_K)
            point_3d_in_world_list = point_3d_in_world[inliers]
            point_2d_in_keyframe_list = keyframe_matched_keypoints[inliers]
            point_count = len(point_2d_in_keyframe_list)
            if point_count <= min_landmark_count:
                print(f"not enough landmarks to relocalize, {point_count} <= {min_landmark_count}")
                continue
            pnp_candidates.append((point_3d_in_world_list, point_2d_in_keyframe_list))

        success, best_pose_in_camera, pose_cov_weight, best_candidate_index, best_inlier_count, best_point_count = rerank_by_pnp_inliers(
            pnp_candidates,
            self.map_K,
            min_inlier_count=min_inlier_count,
        )
        if success:
            print(
                f"relocalization pose : {best_pose_in_camera}, "
                f"best_candidate_index: {best_candidate_index}, "
                f"pnp_inliers: {best_inlier_count}/{best_point_count}"
            )
            return True, best_pose_in_camera, pose_cov_weight

        print("no valid PnP relocalization candidate found")
        return False, np.eye(4), -np.inf

    def _get_relocalization_pnp_thresholds(self) -> tuple[int, int, int]:
        now_hour = datetime.now().hour
        is_night = now_hour >= 18 or now_hour < 6
        if is_night:
            return (
                self.night_relocalization_min_match_count,
                self.night_relocalization_min_landmark_count,
                self.night_relocalization_min_inlier_count,
            )
        return 50, 80, self.relocalization_min_inlier_count

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
        timestamp_ns = int(timestamp.sec * 1e9) + int(timestamp.nanosec)
        current_odom_pose = self.pose_graph_used_pose.get(timestamp_ns)
        res, pose_in_camera, pose_cov_weight = self.relocalize_with_depth(image, features, self.K, current_odom_pose=current_odom_pose)
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
        if self._has_active_rtk_transform():
            return  # RTK owns T_from_map_to_odom while ACTIVE; never let visual reloc overwrite it
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

    def try_publish_nav_path(
        self,
        timestamp: int | None,
        pose_in_origin_odom: np.ndarray | None = None,
        stamp_msg=None,
        force_replan: bool = False,
    ):
        self.get_logger().debug(f"try_publish_nav_path, timestamp: {timestamp}")
        if self.T_from_map_to_odom is None:
            self.get_logger().debug("Relocalization not successful yet, skip publishing nav path")
            return

        if self.poi_index == -1:
            self.get_logger().debug("No POI found, skip publishing nav path")
            return

        if self.poi_index >= len(self.pois):
            self.get_logger().debug("All POIs have been visited, skip publishing nav path")
            return

        poi = self.pois[self.poi_index]
        self.get_logger().debug(f"poi: {poi}")
        poi_pose = np.eye(4)
        poi_pose[:3, 3] = poi
        self.poi_pub.publish(np2msg(poi_pose, self.get_clock().now().to_msg(), "world", "map"))
        # get the pose from the map to the odom
        if pose_in_origin_odom is None:
            if timestamp not in self.pose_graph_used_pose:
                return
            pose_in_origin_odom = self.pose_graph_used_pose[timestamp]
        pose_in_map = np.linalg.inv(self.T_from_map_to_odom) @ pose_in_origin_odom
        if self._has_active_rtk_transform():
            pose_in_map[2, 3] = self._rtk_map_ground_z(pose_in_map[:3, 3])
        publish_stamp = stamp_msg if stamp_msg is not None else self.get_clock().now().to_msg()
        self.current_pose_in_map_pub.publish(np2msg(pose_in_map, publish_stamp, "world", "map"))

        pose_in_map_position = pose_in_map[:3, 3]

        while self.poi_index < len(self.pois):
            poi = self.pois[self.poi_index]
            diff_position_norm_xy = np.linalg.norm(poi[:2] - pose_in_map_position[:2])
            diff_position_norm_z = np.linalg.norm(poi[2] - pose_in_map_position[2])
            if diff_position_norm_xy < self.poi_distance and (self.z_disable or diff_position_norm_z < 2.0):
                arrived_msg = String()
                arrived_msg.data = json.dumps(self._nav_progress_payload(
                    percent=100.0,
                    path_remaining_m=0.0,
                    path_total_m=round(self._leg_initial_length or 0.0, 2),
                    estimated_remaining_s=0.0,
                ))
                self.nav_progress_pub.publish(arrived_msg)
                self.poi_index += 1
                self._leg_initial_length = None
                self._leg_start_time = None
                self._clear_global_path_cache()
                dummy_pose = np.eye(4)

                poi_change_stamp = stamp_msg if stamp_msg is not None else self.get_clock().now().to_msg()
                if timestamp is not None:
                    poi_change_stamp.sec = int(timestamp / 1e9)
                    poi_change_stamp.nanosec = int(timestamp % 1e9)
                self.poi_change_pub.publish(np2msg(dummy_pose, poi_change_stamp, "world", "map"))
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
        nav_goal = self._get_current_nav_goal_in_map(pose_in_map, target_poi)
        if force_replan:
            pose_for_planning = self._pose_with_nav_clamped_z(pose_in_map, path=self._current_nav_path_in_map)
            nav_goal_for_planning = self._nav_goal_for_planning(nav_goal, pose_for_planning)
            with Timer(name = "generate nav path in map", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                generated_path = self.generate_nav_path_in_map(pose_in_map=pose_for_planning, target_poi=nav_goal_for_planning)
            if generated_path is not None and len(generated_path) > 0:
                paths_in_map = generated_path
                self._cached_global_path = generated_path
                self._cached_global_path_poi_index = self.poi_index
                self._cached_global_path_goal = np.array(nav_goal_for_planning, dtype=np.float64)
            else:
                paths_in_map = self._current_nav_path_in_map
                self.get_logger().warning("Failed to regenerate global path; reusing previous path")
        else:
            paths_in_map = self._get_or_replan_global_path(pose_in_map, nav_goal)

        if paths_in_map is not None:
            self._current_nav_path_in_map = paths_in_map
            closest_position, remaining_length = self._path_progress(paths_in_map, pose_in_map_position[:3])

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
            progress_msg.data = json.dumps(self._nav_progress_payload(
                percent=round(percent, 1),
                path_remaining_m=round(remaining_length, 2),
                path_total_m=round(initial, 2),
                estimated_remaining_s=round(estimated_remaining_s, 1),
            ))
            self.nav_progress_pub.publish(progress_msg)

            T = pose_in_origin_odom @ np.linalg.inv(pose_in_map)
            self._publish_path_in_map(self.global_plan_pub, paths_in_map)
            self.tf_broadcaster.sendTransform(np2tf(T, self.get_clock().now().to_msg(), "world", "map"))
        else:
            self.get_logger().debug("No path found in map")

    def _get_current_nav_goal_in_map(self, pose_in_map: np.ndarray, target_poi: np.ndarray) -> np.ndarray:
        if self._nav_subgoals_poi_index != self.poi_index or not self._nav_subgoals_in_map:
            self._build_nav_subgoals_in_map(pose_in_map, target_poi)

        if not self._nav_subgoals_in_map:
            return target_poi

        pose_position = pose_in_map[:3, 3]
        while self._nav_subgoal_index < len(self._nav_subgoals_in_map) - 1:
            subgoal = self._nav_subgoals_in_map[self._nav_subgoal_index]
            diff_xy = np.linalg.norm(subgoal[:2] - pose_position[:2])
            diff_z = np.linalg.norm(subgoal[2] - pose_position[2])
            if diff_xy >= self._nav_subgoal_arrival_xy_threshold or (
                not self.z_disable and diff_z >= self._nav_subgoal_arrival_z_threshold
            ):
                break
            self._nav_subgoal_index += 1
            self._current_nav_path_in_map = None
            self._cached_global_path = None
            self._cached_global_path_poi_index = None
            self._cached_global_path_goal = None
            self._cached_global_path_T = None
            self._leg_initial_length = None
            self._leg_start_time = None
            self._speed_estimate = None
            self.get_logger().info(
                f"Advanced nav subgoal: {self._nav_subgoal_index + 1}/{len(self._nav_subgoals_in_map)}"
            )

        return self._nav_subgoals_in_map[self._nav_subgoal_index]

    def _build_nav_subgoals_in_map(self, pose_in_map: np.ndarray, target_poi: np.ndarray) -> None:
        self._nav_subgoals_in_map = []
        self._nav_subgoals_poi_index = self.poi_index
        self._nav_subgoal_index = 0
        pose_for_planning = self._pose_with_nav_clamped_z(pose_in_map, path=None)
        target_for_planning = self._nav_goal_for_planning(target_poi, pose_for_planning)

        with Timer(name = "generate nav subgoals", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            full_path = self.generate_nav_path_in_map(pose_in_map=pose_for_planning, target_poi=target_for_planning)

        if full_path is None or len(full_path) == 0:
            self.get_logger().warning("Failed to generate full path for nav subgoals; falling back to POI")
            self._nav_subgoals_in_map = [np.array(target_for_planning, dtype=np.float64)]
            self._publish_path_in_map(self.final_global_plan_pub, [])
            return

        self._publish_path_in_map(self.final_global_plan_pub, full_path)

        self._nav_subgoals_in_map = self._split_path_into_subgoals(
            full_path,
            self._nav_subgoal_segment_length_m,
            target_for_planning,
        )
        self.get_logger().info(
            f"Generated {len(self._nav_subgoals_in_map)} nav subgoals for POI {self.poi_index}"
        )

    def _split_path_into_subgoals(self, path: np.ndarray, segment_length_m: float, target_poi: np.ndarray) -> list[np.ndarray]:
        if len(path) == 0:
            return [np.array(target_poi, dtype=np.float64)]

        subgoals: list[np.ndarray] = []
        next_distance = float(segment_length_m)
        traveled = 0.0

        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            segment = end - start
            segment_length = float(np.linalg.norm(segment))
            if segment_length <= 1e-9:
                continue

            while traveled + segment_length >= next_distance:
                ratio = (next_distance - traveled) / segment_length
                subgoals.append(np.array(start + ratio * segment, dtype=np.float64))
                next_distance += float(segment_length_m)

            traveled += segment_length

        if not subgoals or np.linalg.norm(subgoals[-1] - target_poi) > 1e-6:
            subgoals.append(np.array(target_poi, dtype=np.float64))

        return subgoals

    def _get_or_replan_global_path(self, pose_in_map: np.ndarray, target_poi: np.ndarray) -> np.ndarray | None:
        initial_pose_for_planning = self._pose_with_nav_clamped_z(pose_in_map, path=None)
        target_for_planning = self._nav_goal_for_planning(target_poi, initial_pose_for_planning)
        reusable_cached_path = (
            self._cached_global_path is not None
            and self._cached_global_path_poi_index == self.poi_index
            and self._cached_global_path_goal is not None
            and np.linalg.norm(self._cached_global_path_goal - target_for_planning) < 1e-6
        )
        pose_for_planning = self._pose_with_nav_clamped_z(
            pose_in_map,
            path=self._cached_global_path if reusable_cached_path else None,
        )
        target_for_planning = self._nav_goal_for_planning(target_poi, pose_for_planning)
        pose_position = pose_for_planning[:3, 3]
        if reusable_cached_path:
            _, _, deviation = self._closest_point_on_path(self._cached_global_path, pose_position)
            if deviation <= self._replan_path_deviation_threshold:
                return self._cached_global_path
            self.get_logger().info(
                f"Replanning global path: deviation {deviation:.2f}m > "
                f"{self._replan_path_deviation_threshold:.2f}m"
            )

        with Timer(name = "generate nav path in map", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            path = self.generate_nav_path_in_map(pose_in_map=pose_for_planning, target_poi=target_for_planning)
        if path is not None:
            self._cached_global_path = path
            self._cached_global_path_poi_index = self.poi_index
            self._cached_global_path_goal = np.array(target_for_planning, dtype=np.float64)
        return path

    def _build_rtk_ground_z_lookup(self) -> None:
        """RTK replace mode only: build the xy -> map-frame z table from the map keyframes.

        poses.npy is the saved `pose_graph_used_pose`, i.e. exactly the same
        quantity as the `pose_in_map` this node computes at runtime, and the same
        frame the POIs and nav subgoals live in. Every keyframe is a place the
        robot physically stood, so its z is always a valid A* start height.
        The map is loaded once per node lifetime (a map switch restarts the node),
        so this table is built once and never goes stale.
        """
        try:
            positions = np.array([pose[:3, 3] for pose in self.map_poses.values()], dtype=np.float64)
        except Exception as exc:
            self.get_logger().warning(f"RTK ground-z lookup unavailable: {exc}")
            return
        if positions.size == 0:
            self.get_logger().warning("RTK ground-z lookup unavailable: map has no keyframe poses")
            return
        self._rtk_kf_xy = np.ascontiguousarray(positions[:, :2])
        self._rtk_kf_z = np.ascontiguousarray(positions[:, 2])
        self.get_logger().info(
            f"RTK ground-z lookup built from {len(self._rtk_kf_z)} keyframes "
            f"(map z {self._rtk_kf_z.min():.2f}..{self._rtk_kf_z.max():.2f})"
        )

    def _rtk_map_ground_z(self, position: np.ndarray) -> float:
        """Map-frame z at the robot's xy, for RTK replace mode only.

        RTK / File B publish z=0, and RTK altitude cannot be used either: the
        map's z is warped by VIO vertical drift (map_go_1 spans ~7 m in map z on
        one physical floor), so RTK's near-flat altitude does not match it. z must
        come from the map side, and it must be available *unconditionally* -- z is
        not merely a comparison term, it indexes the 3D occupancy/SDF grid, so a
        wrong z puts `start_idx[2]` outside the grid (map_back_1 is only 36 cells
        / 3.6 m tall), A* returns None, and the robot stalls or circles.

        Interpolate over the nearest map keyframes in xy. This is stateless and
        always available, unlike deriving z from the current nav path -- the path
        needs a start z to be planned in the first place, and it is cleared on
        every subgoal advance. Keyframe z is locally unambiguous: within 1.5 m in
        xy the keyframe z spread is 0.12 m median / 0.52 m max on map_go_1 and
        0.04 / 0.18 m on map_back_1, with no multi-level overlap on either map.

        This is an override, not the small-diff `_nav_clamped_z` snap (which would
        reject the large RTK-vs-map z gap), so the arrival / subgoal / z-clamp
        gates downstream then compare map-vs-map.
        """
        if self._rtk_kf_xy is not None:
            dx = self._rtk_kf_xy[:, 0] - float(position[0])
            dy = self._rtk_kf_xy[:, 1] - float(position[1])
            d2 = dx * dx + dy * dy
            k = min(3, d2.shape[0])
            nearest = np.argpartition(d2, k - 1)[:k] if d2.shape[0] > k else np.arange(d2.shape[0])
            weights = 1.0 / (np.sqrt(d2[nearest]) + 0.1)
            return float(np.dot(self._rtk_kf_z[nearest], weights) / weights.sum())
        # Degenerate map (no keyframe poses): fall back to the nav path, then SDF
        # ground, then the POI z. Do NOT reach the raw position z -- it is RTK's 0,
        # which is outside the grid entirely.
        path = self._current_nav_path_in_map
        if path is not None and len(path) > 0:
            _, closest, _ = self._closest_point_on_path_for_z_clamp(path, position)
            return float(closest[2])
        sdf_z, _ = self._nav_clamped_z_from_sdf(position)
        if sdf_z is not None:
            return float(sdf_z)
        if 0 <= self.poi_index < len(self.pois) and len(self.pois[self.poi_index]) >= 3:
            return float(self.pois[self.poi_index][2])
        return float(position[2])
    def _nav_goal_for_planning(self, target: np.ndarray, pose_for_planning: np.ndarray) -> np.ndarray:
        goal = np.array(target, dtype=np.float64).copy()
        if self.z_disable:
            goal[2] = float(pose_for_planning[2, 3])
        return goal

    def _pose_with_nav_clamped_z(self, pose_in_map: np.ndarray, path: np.ndarray | None = None) -> np.ndarray:
        pose_for_nav = pose_in_map.copy()
        position = pose_for_nav[:3, 3]
        clamped_z, source = self._nav_clamped_z(position, path)
        if clamped_z is not None:
            raw_z = float(position[2])
            position[2] = clamped_z
            self._log_nav_z_clamp(raw_z, clamped_z, source)
        return pose_for_nav

    def _nav_position_with_clamped_z(self, position: np.ndarray, path: np.ndarray | None = None) -> np.ndarray:
        nav_position = np.array(position, dtype=np.float64)
        clamped_z, source = self._nav_clamped_z(nav_position, path)
        if clamped_z is not None:
            raw_z = float(nav_position[2])
            nav_position[2] = clamped_z
            self._log_nav_z_clamp(raw_z, clamped_z, source)
        return nav_position

    def _nav_clamped_z(self, position: np.ndarray, path: np.ndarray | None = None) -> tuple[float | None, str]:
        if path is not None and len(path) > 0:
            _, closest_point, _ = self._closest_point_on_path_for_z_clamp(path, position)
            z_diff = abs(float(closest_point[2] - position[2]))
            if z_diff <= self._nav_z_clamp_max_diff_m:
                return float(closest_point[2]), "path"
            return None, "path_rejected"

        return self._nav_clamped_z_from_sdf(position)

    def _closest_point_on_path_for_z_clamp(self, path: np.ndarray, position: np.ndarray) -> tuple[int, np.ndarray, float]:
        if len(path) == 1:
            point = np.array(path[0], dtype=np.float64)
            xy_dist = float(np.linalg.norm(point[:2] - position[:2]))
            z_dist = abs(float(point[2] - position[2]))
            return 0, point, xy_dist + 0.3 * z_dist

        best_index = 0
        best_point = np.array(path[0], dtype=np.float64)
        best_score = np.inf
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            ab_xy = b[:2] - a[:2]
            denom = float(np.dot(ab_xy, ab_xy))
            ratio = 0.0 if denom <= 1e-9 else np.clip(np.dot(position[:2] - a[:2], ab_xy) / denom, 0.0, 1.0)
            point = a + ratio * (b - a)
            xy_dist = float(np.linalg.norm(point[:2] - position[:2]))
            z_dist = abs(float(point[2] - position[2]))
            score = xy_dist + 0.3 * z_dist
            if score < best_score:
                best_index = i
                best_point = point
                best_score = score
        return best_index, best_point, best_score

    def _nav_clamped_z_from_sdf(self, position: np.ndarray) -> tuple[float | None, str]:
        occupancy_map_origin = self.occupancy_map_meta[:3]
        resolution = float(self.occupancy_map_meta[3])
        center_idx = np.array([
            int((position[0] - occupancy_map_origin[0]) / resolution),
            int((position[1] - occupancy_map_origin[1]) / resolution),
            int((position[2] - occupancy_map_origin[2]) / resolution),
        ], dtype=np.int32)
        xy_radius = max(1, int(np.ceil(self._nav_z_clamp_sdf_xy_radius_m / resolution)))
        z_window = max(1, int(np.ceil(self._nav_z_clamp_sdf_z_window_m / resolution)))
        x0 = max(0, int(center_idx[0]) - xy_radius)
        x1 = min(self.sdf_map.shape[0], int(center_idx[0]) + xy_radius + 1)
        y0 = max(0, int(center_idx[1]) - xy_radius)
        y1 = min(self.sdf_map.shape[1], int(center_idx[1]) + xy_radius + 1)
        z0 = max(0, int(center_idx[2]) - z_window)
        z1 = min(self.sdf_map.shape[2], int(center_idx[2]) + z_window + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            return None, "sdf_out_of_bounds"

        sdf_crop = self.sdf_map[x0:x1, y0:y1, z0:z1]
        occ_crop = self.occupancy_map[x0:x1, y0:y1, z0:z1]
        valid = np.isfinite(sdf_crop) & (occ_crop != 2) & (sdf_crop < 0.2)
        if not np.any(valid):
            return None, "sdf_not_found"

        xs, ys, zs = np.nonzero(valid)
        world_x = (xs + x0) * resolution + occupancy_map_origin[0]
        world_y = (ys + y0) * resolution + occupancy_map_origin[1]
        world_z = (zs + z0) * resolution + occupancy_map_origin[2]
        xy_dist = np.hypot(world_x - position[0], world_y - position[1])
        z_dist = np.abs(world_z - position[2])
        sdf_values = sdf_crop[xs, ys, zs]
        score = xy_dist + 0.3 * z_dist + 0.5 * sdf_values
        best = int(np.argmin(score))
        clamped_z = float(world_z[best])
        if abs(clamped_z - float(position[2])) > self._nav_z_clamp_max_diff_m:
            return None, "sdf_rejected"
        return clamped_z, "sdf"

    def _log_nav_z_clamp(self, raw_z: float, clamped_z: float, source: str) -> None:
        if abs(raw_z - clamped_z) < 0.05:
            return
        now = time.monotonic()
        if now - self._last_nav_z_clamp_log_time < 2.0:
            return
        self._last_nav_z_clamp_log_time = now
        self.get_logger().info(
            f"nav_z_clamp source={source} raw_z={raw_z:.2f} clamped_z={clamped_z:.2f} "
            f"diff={abs(raw_z - clamped_z):.2f}"
        )

    def _closest_point_on_path(self, path: np.ndarray, position: np.ndarray) -> tuple[int, np.ndarray, float]:
        if len(path) == 1:
            return 0, path[0], np.linalg.norm(path[0] - position)

        best_index = 0
        best_point = path[0]
        best_distance = np.inf
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            ab = b - a
            denom = float(np.dot(ab, ab))
            ratio = 0.0 if denom <= 1e-9 else np.clip(np.dot(position - a, ab) / denom, 0.0, 1.0)
            point = a + ratio * ab
            distance = np.linalg.norm(point - position)
            if distance < best_distance:
                best_index = i
                best_point = point
                best_distance = distance
        return best_index, best_point, best_distance

    def _path_progress(self, path: np.ndarray, position: np.ndarray) -> tuple[np.ndarray, float]:
        nav_position = self._nav_position_with_clamped_z(position, path=path)
        closest_index, closest_position, _ = self._closest_point_on_path(path, nav_position)
        remaining = np.linalg.norm(path[closest_index + 1] - closest_position) if closest_index + 1 < len(path) else 0.0
        for i in range(closest_index + 1, len(path) - 1):
            remaining += np.linalg.norm(path[i + 1] - path[i])
        return closest_position, remaining

    def _select_target_position_in_map(
        self,
        paths_in_map: np.ndarray,
        pose_in_map_position: np.ndarray,
        lookahead_distance: float,
    ) -> np.ndarray:
        if len(paths_in_map) == 0:
            return pose_in_map_position
        closest_idx = int(np.argmin(np.linalg.norm(paths_in_map[:, :2] - pose_in_map_position[:2], axis=1)))
        closest_position = paths_in_map[closest_idx]
        if self.select_target_position_on_path_on:
            remaining_path = paths_in_map[closest_idx + 1 :]
            if len(remaining_path) == 0:
                return closest_position
            return select_target_position_on_path(
                remaining_path,
                pose_in_map_position,  #closest_position,
                lookahead_distance=lookahead_distance,
                turn_angle_threshold_rad=np.deg2rad(65.0),
                reversal_angle_threshold_rad=np.deg2rad(120.0),
                turn_stop_margin=0.2,
                min_turn_distance=0.5,
                turn_window_distance=0.4,
                outside_offset_m=0.2,
            )

        target_position = paths_in_map[-1]
        start_point = closest_position
        accumulated_distance = 0.0
        for i in range(closest_idx, len(paths_in_map) - 1):
            accumulated_distance += np.linalg.norm(paths_in_map[i][:2] - start_point[:2])
            if accumulated_distance > lookahead_distance:
                target_position = paths_in_map[i]
                break
            start_point = paths_in_map[i]
        return target_position

    def _publish_target_pose_from_path(self, paths_in_map: np.ndarray, pose_in_origin_odom: np.ndarray, stamp_msg=None):
        pose_in_map = np.linalg.inv(self.T_from_map_to_odom) @ pose_in_origin_odom
        if self._has_active_rtk_transform():
            pose_in_map[2, 3] = self._rtk_map_ground_z(pose_in_map[:3, 3])
        pose_in_map_position = self._nav_position_with_clamped_z(pose_in_map[:3, 3], path=paths_in_map)
        with Timer(name = "Find target position", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            max_speed = self.planning_max_linear_speed
            lookahead_distance = max_speed * self.target_pose_dist_factor
            target_position = self._select_target_position_in_map(
                paths_in_map,
                pose_in_map_position[:3],
                lookahead_distance,
            )
            target_position_in_map = np.array([target_position[0], target_position[1], target_position[2]])
            if self.z_disable:
                target_position_in_map[2] = pose_in_map[:3, 3][2]
            T = pose_in_origin_odom @ np.linalg.inv(pose_in_map)
            target_position_in_odom = T[:3, :3] @ target_position_in_map + T[:3, 3]
            dummy_pose = np.eye(4)
            dummy_pose[:3, 3] = target_position_in_odom
            self.target_pose_pub.publish(np2msg(
                dummy_pose,
                stamp_msg if stamp_msg is not None else self.get_clock().now().to_msg(),
                "world",
                "camera",
            ))

    def _point_ahead_on_path(self, path: np.ndarray, start_position: np.ndarray, distance_ahead: float) -> np.ndarray:
        closest_index, current, _ = self._closest_point_on_path(path, start_position)
        for i in range(closest_index + 1, len(path)):
            next_point = path[i]
            segment_length = np.linalg.norm(next_point - current)
            if segment_length >= distance_ahead:
                ratio = distance_ahead / segment_length if segment_length > 1e-9 else 0.0
                return current + ratio * (next_point - current)
            distance_ahead -= segment_length
            current = next_point
        return path[-1]

    def generate_nav_path_in_map(self, pose_in_map: np.ndarray, target_poi: np.ndarray) -> np.ndarray:
        profile_start_time = time.perf_counter()
        occupancy_map_origin = self.occupancy_map_meta[:3]
        resolution = self.occupancy_map_meta[3]
        pose_position = pose_in_map[:3, 3]
        start_idx = np.array([
            int((pose_position[0] - occupancy_map_origin[0]) / resolution),
            int((pose_position[1] - occupancy_map_origin[1]) / resolution),
            int((pose_position[2] - occupancy_map_origin[2]) / resolution)
        ], dtype=np.int32)
        poi_goal_idx = np.array([
            int((target_poi[0] - occupancy_map_origin[0]) / resolution),
            int((target_poi[1] - occupancy_map_origin[1]) / resolution),
            int((target_poi[2] - occupancy_map_origin[2]) / resolution)
        ], dtype=np.int32)
        subgoal_label = f"{self._nav_subgoal_index + 1}/{len(self._nav_subgoals_in_map)}" if self._nav_subgoals_in_map else "none"

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
            self.get_logger().warning(
                "nav_path_profile failed=out_of_bounds "
                f"subgoal={subgoal_label} "
                f"pose=({pose_position[0]:.2f},{pose_position[1]:.2f},{pose_position[2]:.2f}) "
                f"target=({target_poi[0]:.2f},{target_poi[1]:.2f},{target_poi[2]:.2f}) "
                f"start_idx={tuple(start_idx)} goal_idx={tuple(poi_goal_idx)} "
                f"map_shape={self.occupancy_map.shape}"
            )
            return None 
        start_snap_start_time = time.perf_counter()
        # Snap into SDF bucket 0, not merely "near a surface". The A* drains bucket
        # 0 across the whole map before it touches bucket 1, so an endpoint landing
        # in bucket 1 (sdf 0.15-0.20) costs a full sweep of every bucket-0 cell --
        # measured at 21986 nodes / 8.6 s on map_go_1, versus ~102 nodes when both
        # endpoints share bucket 0.
        sdf_start_path = search_close_to_sdf_map(start_idx, self.sdf_map, self.occupancy_map, SDF_BINS[0])
        start_snap_ms = (time.perf_counter() - start_snap_start_time) * 1000.0
        goal_snap_start_time = time.perf_counter()
        sdf_goal_path = search_close_to_sdf_map(poi_goal_idx, self.sdf_map, self.occupancy_map, SDF_BINS[0])
        goal_snap_ms = (time.perf_counter() - goal_snap_start_time) * 1000.0

        if len(sdf_start_path) == 0 or len(sdf_goal_path) == 0:
            self.get_logger().warning(
                "nav_path_profile failed=empty_sdf_snap "
                f"subgoal={subgoal_label} "
                f"pose=({pose_position[0]:.2f},{pose_position[1]:.2f},{pose_position[2]:.2f}) "
                f"target=({target_poi[0]:.2f},{target_poi[1]:.2f},{target_poi[2]:.2f}) "
                f"start_idx={tuple(start_idx)} goal_idx={tuple(poi_goal_idx)} "
                f"start_snap_len={len(sdf_start_path)} goal_snap_len={len(sdf_goal_path)} "
                f"start_snap_ms={start_snap_ms:.0f} goal_snap_ms={goal_snap_ms:.0f}"
            )
            return None

        sdf_start_sdf = sdf_start_path[-1]
        sdf_goal_sdf = sdf_goal_path[-1]
        start_sdf_val = float(self.sdf_map[tuple(sdf_start_sdf)])
        goal_sdf_val = float(self.sdf_map[tuple(sdf_goal_sdf)])
        endpoint_profile = (
            f"start_sdf={start_sdf_val:.3f} start_bin={sdf_queue_index(start_sdf_val)} "
            f"goal_sdf={goal_sdf_val:.3f} goal_bin={sdf_queue_index(goal_sdf_val)} "
        )
        sdf_search_start_time = time.perf_counter()
        search_stats: dict = {}
        path_sdf = search_within_sdf_map(sdf_start_sdf, sdf_goal_sdf, self.sdf_map, self.occupancy_map, resolution, search_stats)
        sdf_search_ms = (time.perf_counter() - sdf_search_start_time) * 1000.0
        search_profile = (
            f"expanded={search_stats.get('expanded', -1)} "
            f"z_span={search_stats.get('z_span', -1)} "
            f"z_range={search_stats.get('z_lo', -1)}..{search_stats.get('z_hi', -1)} "
        )
        if len(path_sdf) == 0:
            self.get_logger().warning(
                "nav_path_profile failed=empty_sdf_path "
                f"subgoal={subgoal_label} "
                f"pose=({pose_position[0]:.2f},{pose_position[1]:.2f},{pose_position[2]:.2f}) "
                f"target=({target_poi[0]:.2f},{target_poi[1]:.2f},{target_poi[2]:.2f}) "
                f"start_idx={tuple(start_idx)} goal_idx={tuple(poi_goal_idx)} "
                f"sdf_start_idx={tuple(sdf_start_sdf)} sdf_goal_idx={tuple(sdf_goal_sdf)} "
                f"start_snap_len={len(sdf_start_path)} goal_snap_len={len(sdf_goal_path)} "
                f"start_snap_ms={start_snap_ms:.0f} goal_snap_ms={goal_snap_ms:.0f} "
                f"sdf_search_ms={sdf_search_ms:.0f} {search_profile}{endpoint_profile}"
            )
        path = sdf_start_path + path_sdf + sdf_goal_path[::-1]
        if len(path) > 0:
            shortcut_start_time = time.perf_counter()
            pruned_path = shortcut_prune_path(path, self.sdf_map, self.occupancy_map, resolution)
            shortcut_ms = (time.perf_counter() - shortcut_start_time) * 1000.0
            if len(pruned_path) < len(path):
                self.get_logger().info(f"shortcut pruned nav path: {len(path)} -> {len(pruned_path)} points")
            converted_path = np.array(pruned_path) * resolution + occupancy_map_origin
            total_ms = (time.perf_counter() - profile_start_time) * 1000.0
            self.get_logger().info(
                "nav_path_profile ok "
                f"subgoal={subgoal_label} "
                f"pose=({pose_position[0]:.2f},{pose_position[1]:.2f},{pose_position[2]:.2f}) "
                f"target=({target_poi[0]:.2f},{target_poi[1]:.2f},{target_poi[2]:.2f}) "
                f"start_idx={tuple(start_idx)} goal_idx={tuple(poi_goal_idx)} "
                f"sdf_start_idx={tuple(sdf_start_sdf)} sdf_goal_idx={tuple(sdf_goal_sdf)} "
                f"start_snap_len={len(sdf_start_path)} goal_snap_len={len(sdf_goal_path)} "
                f"sdf_path_len={len(path_sdf)} raw_path_len={len(path)} pruned_path_len={len(pruned_path)} "
                f"start_snap_ms={start_snap_ms:.0f} goal_snap_ms={goal_snap_ms:.0f} "
                f"sdf_search_ms={sdf_search_ms:.0f} shortcut_ms={shortcut_ms:.0f} total_ms={total_ms:.0f} "
                f"{search_profile}{endpoint_profile}"
            )
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
    parser.add_argument(
        "--enable_first_done",
        action="store_true",
        default=False,
        help="Skip keyframe relocalization after the first successful relocalization",
    )
    parser.add_argument(
        "--initial_map_to_odom_transform",
        type=str,
        default=None,
        help=(
            "Path to a .npy 4x4 T_from_map_to_odom to seed this session with (map handoff "
            "between two calibrated-adjacent maps), instead of waiting for a fresh cold "
            "relocalization/RTK fix. See app/backend/node_manager.py's "
            "_maybe_seed_map_handoff."
        ),
    )
    parsed_args, unknown_args = parser.parse_known_args(sys.argv[1:])
    node = MapNode(tinynav_db_path=parsed_args.tinynav_db_path,
                   tinynav_map_path=parsed_args.tinynav_map_path,
                   verbose_timer=parsed_args.verbose_timer,
                   enable_first_done=parsed_args.enable_first_done,
                   initial_map_to_odom_transform_path=parsed_args.initial_map_to_odom_transform)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
