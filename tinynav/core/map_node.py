import rclpy
import os
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool, String
import numpy as np
import sys
import json

import heapq
from numba import njit
from tinynav.core.math_utils import matrix_to_quat, msg2np, np2msg, estimate_pose, np2tf, se3_inv
from sensor_msgs.msg import Image, CameraInfo
from message_filters import TimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
from codetiming import Timer
import argparse

from tinynav.tinynav_cpp_bind import pose_graph_solve
from tinynav.core.models_trt import (
    Dinov2TRT,
    LightGlueTRT,
    ORBFeatureTRTCompatible,
    ORBMatcher,
    SuperPointTRT,
)
import logging
import asyncio
import time
from tf2_ros import TransformBroadcaster
from tinynav.core.build_map_node import TinyNavDB
from tinynav.core.build_map_node import solve_pose_graph
import einops
from tinynav.core.build_map_node import OdomPoseRecorder, LoopClosure
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

@njit(cache=True)
def _nav_flat_index(x: int, y: int, z: int, ny: int, nz: int) -> int:
    return (x * ny + y) * nz + z

@njit(cache=True)
def _nav_unflat_index(index: int, ny: int, nz: int):
    x = index // (ny * nz)
    remainder = index - x * ny * nz
    y = remainder // nz
    z = remainder - y * nz
    return x, y, z

@njit(cache=True)
def _nav_heap_less(bucket_a: int, cost_a: float, node_a: int, bucket_b: int, cost_b: float, node_b: int) -> bool:
    if bucket_a != bucket_b:
        return bucket_a < bucket_b
    if cost_a != cost_b:
        return cost_a < cost_b
    return node_a < node_b

@njit(cache=True)
def _nav_heap_push(heap_nodes: np.ndarray, heap_buckets: np.ndarray, heap_costs: np.ndarray, heap_size: int, node: int, bucket: int, cost: float) -> int:
    i = heap_size
    heap_nodes[i] = node
    heap_buckets[i] = bucket
    heap_costs[i] = cost
    while i > 0:
        parent = (i - 1) // 2
        if not _nav_heap_less(heap_buckets[i], heap_costs[i], heap_nodes[i], heap_buckets[parent], heap_costs[parent], heap_nodes[parent]):
            break
        tmp_node = heap_nodes[parent]
        tmp_bucket = heap_buckets[parent]
        tmp_cost = heap_costs[parent]
        heap_nodes[parent] = heap_nodes[i]
        heap_buckets[parent] = heap_buckets[i]
        heap_costs[parent] = heap_costs[i]
        heap_nodes[i] = tmp_node
        heap_buckets[i] = tmp_bucket
        heap_costs[i] = tmp_cost
        i = parent
    return heap_size + 1

@njit(cache=True)
def _nav_heap_pop(heap_nodes: np.ndarray, heap_buckets: np.ndarray, heap_costs: np.ndarray, heap_size: int):
    node = heap_nodes[0]
    bucket = heap_buckets[0]
    cost = heap_costs[0]
    heap_size -= 1
    if heap_size > 0:
        heap_nodes[0] = heap_nodes[heap_size]
        heap_buckets[0] = heap_buckets[heap_size]
        heap_costs[0] = heap_costs[heap_size]
        i = 0
        while True:
            left = 2 * i + 1
            right = left + 1
            smallest = i
            if left < heap_size and _nav_heap_less(
                heap_buckets[left], heap_costs[left], heap_nodes[left],
                heap_buckets[smallest], heap_costs[smallest], heap_nodes[smallest],
            ):
                smallest = left
            if right < heap_size and _nav_heap_less(
                heap_buckets[right], heap_costs[right], heap_nodes[right],
                heap_buckets[smallest], heap_costs[smallest], heap_nodes[smallest],
            ):
                smallest = right
            if smallest == i:
                break
            tmp_node = heap_nodes[smallest]
            tmp_bucket = heap_buckets[smallest]
            tmp_cost = heap_costs[smallest]
            heap_nodes[smallest] = heap_nodes[i]
            heap_buckets[smallest] = heap_buckets[i]
            heap_costs[smallest] = heap_costs[i]
            heap_nodes[i] = tmp_node
            heap_buckets[i] = tmp_bucket
            heap_costs[i] = tmp_cost
            i = smallest
    return node, bucket, cost, heap_size

@njit(cache=True)
def _nav_reconstruct_path(parent: np.ndarray, current: int, ny: int, nz: int) -> np.ndarray:
    count = 1
    node = current
    while parent[node] != node and parent[node] >= 0:
        node = parent[node]
        count += 1

    path = np.empty((count, 3), dtype=np.int32)
    node = current
    for i in range(count - 1, -1, -1):
        x, y, z = _nav_unflat_index(node, ny, nz)
        path[i, 0] = x
        path[i, 1] = y
        path[i, 2] = z
        if parent[node] == node or parent[node] < 0:
            break
        node = parent[node]
    return path

@njit(cache=True)
def _nav_sdf_bucket(sdf_value: float) -> int:
    if sdf_value < 0.2:
        return 0
    if sdf_value < 0.5:
        return 1
    if sdf_value < 1.0:
        return 2
    if sdf_value < 2.0:
        return 3
    if sdf_value < 5.0:
        return 4
    if sdf_value < 10.0:
        return 5
    return 6

@njit(cache=True)
def _nav_heuristic_idx(x: int, y: int, z: int, gx: int, gy: int, gz: int, resolution: float) -> float:
    dx = (x - gx) * resolution
    dy = (y - gy) * resolution
    dz = (z - gz) * resolution
    return np.sqrt(dx * dx + dy * dy + dz * dz) + 20.0 * np.abs(dz)

@njit(cache=True)
def search_close_to_sdf_map_numba(start_index: np.ndarray, sdf_map: np.ndarray, occupancy_map: np.ndarray, stop_distance: float) -> np.ndarray:
    nx, ny, nz = sdf_map.shape
    total = nx * ny * nz
    sx = int(start_index[0])
    sy = int(start_index[1])
    sz = int(start_index[2])
    start_node = _nav_flat_index(sx, sy, sz, ny, nz)

    parent = np.full(total, -1, dtype=np.int64)
    visited = np.zeros(total, dtype=np.bool_)
    in_open = np.zeros(total, dtype=np.bool_)
    heap_nodes = np.empty(total, dtype=np.int64)
    heap_buckets = np.zeros(total, dtype=np.int32)
    heap_costs = np.empty(total, dtype=np.float64)

    parent[start_node] = start_node
    in_open[start_node] = True
    heap_size = _nav_heap_push(heap_nodes, heap_buckets, heap_costs, 0, start_node, 0, float(sdf_map[sx, sy, sz]))

    while heap_size > 0:
        current, _, _, heap_size = _nav_heap_pop(heap_nodes, heap_buckets, heap_costs, heap_size)
        in_open[current] = False
        if visited[current]:
            continue
        visited[current] = True
        cx, cy, cz = _nav_unflat_index(current, ny, nz)
        current_sdf = float(sdf_map[cx, cy, cz])
        if current_sdf < stop_distance:
            return _nav_reconstruct_path(parent, current, ny, nz)

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nbx = cx + dx
                    nby = cy + dy
                    nbz = cz + dz
                    if nbx < 0 or nbx >= nx or nby < 0 or nby >= ny or nbz < 0 or nbz >= nz:
                        continue
                    if occupancy_map[nbx, nby, nbz] == 2:
                        continue
                    neighbor = _nav_flat_index(nbx, nby, nbz, ny, nz)
                    if visited[neighbor] or in_open[neighbor]:
                        continue
                    parent[neighbor] = current
                    in_open[neighbor] = True
                    heap_size = _nav_heap_push(heap_nodes, heap_buckets, heap_costs, heap_size, neighbor, 0, float(sdf_map[nbx, nby, nbz]))

    return np.empty((0, 3), dtype=np.int32)

@njit(cache=True)
def search_within_sdf_map_numba(start: np.ndarray, goal: np.ndarray, sdf_map: np.ndarray, occupancy_map: np.ndarray, resolution: float) -> np.ndarray:
    nx, ny, nz = sdf_map.shape
    total = nx * ny * nz
    sx = int(start[0])
    sy = int(start[1])
    sz = int(start[2])
    gx = int(goal[0])
    gy = int(goal[1])
    gz = int(goal[2])
    start_node = _nav_flat_index(sx, sy, sz, ny, nz)
    goal_node = _nav_flat_index(gx, gy, gz, ny, nz)

    parent = np.full(total, -1, dtype=np.int64)
    visited = np.zeros(total, dtype=np.bool_)
    in_open = np.zeros(total, dtype=np.bool_)
    heap_nodes = np.empty(total, dtype=np.int64)
    heap_buckets = np.empty(total, dtype=np.int32)
    heap_costs = np.empty(total, dtype=np.float64)

    parent[start_node] = start_node
    in_open[start_node] = True
    start_bucket = _nav_sdf_bucket(float(sdf_map[sx, sy, sz]))
    start_cost = _nav_heuristic_idx(sx, sy, sz, gx, gy, gz, resolution)
    heap_size = _nav_heap_push(heap_nodes, heap_buckets, heap_costs, 0, start_node, start_bucket, start_cost)

    while heap_size > 0:
        current, _, _, heap_size = _nav_heap_pop(heap_nodes, heap_buckets, heap_costs, heap_size)
        in_open[current] = False
        if visited[current]:
            continue
        visited[current] = True
        if current == goal_node:
            return _nav_reconstruct_path(parent, current, ny, nz)

        cx, cy, cz = _nav_unflat_index(current, ny, nz)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nbx = cx + dx
                    nby = cy + dy
                    nbz = cz + dz
                    if nbx < 0 or nbx >= nx or nby < 0 or nby >= ny or nbz < 0 or nbz >= nz:
                        continue
                    if occupancy_map[nbx, nby, nbz] == 2:
                        continue
                    neighbor = _nav_flat_index(nbx, nby, nbz, ny, nz)
                    if visited[neighbor] or in_open[neighbor]:
                        continue
                    parent[neighbor] = current
                    in_open[neighbor] = True
                    bucket = _nav_sdf_bucket(float(sdf_map[nbx, nby, nbz]))
                    cost = _nav_heuristic_idx(nbx, nby, nbz, gx, gy, gz, resolution)
                    heap_size = _nav_heap_push(heap_nodes, heap_buckets, heap_costs, heap_size, neighbor, bucket, cost)

    return np.empty((0, 3), dtype=np.int32)

class DummyEmbeddingEngine:
    async def infer(self, _image: np.ndarray) -> np.ndarray:
        return np.zeros((1, 768), dtype=np.float32)


class MapNode(Node):
    def __init__(
        self,
        tinynav_db_path: str,
        tinynav_map_path: str,
        extractor,
        matcher,
        embedding_extractor,
        loop_closure_mode: str = "embedding",
        loop_closure_use_bow: bool = False,
        dbow3_vocabulary_path: str | None = None,
        verbose_timer: bool = True,
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
        self.extractor = extractor
        self.matcher = matcher
        self.embedding_extractor = embedding_extractor
        self.loop_closure_use_bow = bool(loop_closure_use_bow)
        self.loop_closure_mode = "bow" if self.loop_closure_use_bow else loop_closure_mode
        self.dbow3_vocabulary_path = dbow3_vocabulary_path
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

        os.makedirs(f"{tinynav_db_path}/nav_temp", exist_ok=True)
        self.nav_temp_db = TinyNavDB(f"{tinynav_db_path}/nav_temp", is_scratch=True)
        self.nav_loop_closure = LoopClosure(
            db=self.nav_temp_db,
            timestamps=[],
            mode=self.loop_closure_mode,
            dbow3_vocabulary_path=self.dbow3_vocabulary_path,
            embedding_similarity_threshold=self.loop_similarity_threshold,
            embedding_top_k=self.loop_top_k,
        )
        self.map_poses = np.load(f"{tinynav_map_path}/poses.npy", allow_pickle=True).item()
        self.map_K = np.load(f"{tinynav_map_path}/intrinsics.npy")
        self.db = TinyNavDB(tinynav_map_path, is_scratch=False)
        self.map_loop_closure = LoopClosure(
            db=self.db,
            timestamps=list(self.map_poses.keys()),
            mode=self.loop_closure_mode,
            dbow3_vocabulary_path=self.dbow3_vocabulary_path,
            embedding_similarity_threshold=self.relocalization_threshold,
            embedding_top_k=self.relocalization_loop_top_k,
        )
        self.occupancy_map = np.load(f"{tinynav_map_path}/occupancy_grid.npy")
        self.occupancy_map_meta = np.load(f"{tinynav_map_path}/occupancy_meta.npy")
        self.sdf_map = np.load(f"{tinynav_map_path}/sdf_map.npy")
        self._warmup_nav_path_search()

        print(f"sdf_map.shape: {self.sdf_map.shape}")
        print(f"occupancy_map.shape: {self.occupancy_map.shape}")

        self.relocalization_poses = {}
        self.relocalization_pose_weights = {}
        self.failed_relocalizations = []
        self.last_relocalization_failure_reason = ""
        self.last_relocalization_timing = {}

        self.T_from_map_to_odom = None

        self.pois = {}
        self.poi_index = -1

        self.poi_pub = self.create_publisher(Odometry, "/mapping/poi", 10)
        self.poi_change_pub = self.create_publisher(Odometry, "/mapping/poi_change", 10)

        self.current_pose_pub = self.create_publisher(Odometry, "/mapping/current_pose", 10)
        self.global_plan_pub = self.create_publisher(Path, '/mapping/global_plan', 10)
        self.target_pose_pub = self.create_publisher(Odometry, "/control/target_pose", 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self._save_completed = False

    def _warmup_nav_path_search(self):
        small_sdf = np.ones((3, 3, 3), dtype=self.sdf_map.dtype)
        small_sdf[1, 1, 1] = 0.0
        small_occupancy = np.zeros((3, 3, 3), dtype=self.occupancy_map.dtype)
        search_close_to_sdf_map_numba(np.array([0, 0, 0], dtype=np.int32), small_sdf, small_occupancy, 0.2)
        search_within_sdf_map_numba(
            np.array([0, 0, 0], dtype=np.int32),
            np.array([2, 2, 2], dtype=np.int32),
            small_sdf,
            small_occupancy,
            float(self.occupancy_map_meta[3]),
        )

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
        t_start = time.perf_counter()
        stage_timings = {}

        def mark_stage(name: str, previous_t: float) -> float:
            now = time.perf_counter()
            stage_timings[name] = (now - previous_t) * 1000.0
            return now

        self.get_logger().info("keyframe_mapping is temporarily disabled.")
        image = self.bridge.imgmsg_to_cv2(keyframe_image_msg, desired_encoding="mono8")
        t_stage = mark_stage("image_decode", t_start)

        keyframe_image_timestamp_ns = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(keyframe_image_msg.header.stamp.nanosec)
        keyframe_odom_timestamp_ns = int(keyframe_odom_msg.header.stamp.sec * 1e9) + int(keyframe_odom_msg.header.stamp.nanosec)
        depth_timestamp_ns = int(depth_msg.header.stamp.sec * 1e9) + int(depth_msg.header.stamp.nanosec)
        sync_skew_ms = (max(keyframe_image_timestamp_ns, keyframe_odom_timestamp_ns, depth_timestamp_ns) - min(keyframe_image_timestamp_ns, keyframe_odom_timestamp_ns, depth_timestamp_ns)) / 1e6
        t_stage = mark_stage("timestamp_parse", t_stage)

        success, pose_in_world = self.keyframe_relocalization(keyframe_image_msg.header.stamp, image)
        t_stage = mark_stage("relocalization", t_stage)

        odom, _ = msg2np(keyframe_odom_msg)
        t_stage = mark_stage("odom_msg_decode", t_stage)

        self.pose_graph_used_pose[keyframe_image_timestamp_ns] = odom
        self.odom[keyframe_image_timestamp_ns] = odom
        t_stage = mark_stage("pose_cache_update", t_stage)

        if success:
            self.compute_transform_from_map_to_odom()
        t_stage = mark_stage("tf_update", t_stage)

        with Timer(name = "nav path", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            self.try_publish_nav_path(keyframe_image_timestamp_ns)
            # timer or queue for publish the nav path
            # and record the map pose
            # compute the coordinate transform from the map pose to the keyframe pose
            # publish the nav path from the map pose to the keyframe pose with the cost map
        t_stage = mark_stage("nav_path", t_stage)

        total_ms = (t_stage - t_start) * 1000.0
        stage_parts = ", ".join(f"{name}={ms:.1f}" for name, ms in stage_timings.items())
        relocal_parts = ", ".join(f"relocal_{name}={ms:.1f}" for name, ms in self.last_relocalization_timing.items())
        msg = (
            f"Keyframe callback benchmark ms: timestamp={keyframe_image_timestamp_ns}, "
            f"total={total_ms:.1f}, sync_skew={sync_skew_ms:.1f}, {stage_parts}, success={success}"
        )
        if relocal_parts:
            msg += f", {relocal_parts}"
        if total_ms > 500.0:
            self.get_logger().warning(msg)
        else:
            self.get_logger().info(msg)

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
        features = asyncio.run(self.extractor.infer(image))
        self.nav_temp_db.set_entry(keyframe_image_timestamp, features = features)

        if len(self.odom) == 0 and self.last_keyframe_timestamp is None:
            self.odom[keyframe_odom_timestamp] = odom
            self.pose_graph_used_pose[keyframe_odom_timestamp] = odom
            self.nav_loop_closure.add_timestamp(keyframe_odom_timestamp)
        else:
            last_keyframe_odom_pose = self.odom[self.last_keyframe_timestamp]
            T_prev_curr = se3_inv(last_keyframe_odom_pose) @ odom
            self.relative_pose_constraint.append((keyframe_image_timestamp, self.last_keyframe_timestamp, T_prev_curr))
            self.pose_graph_used_pose[keyframe_image_timestamp] = odom
            self.odom[keyframe_image_timestamp] = odom
            def find_loop_and_pose_graph(timestamp):
                    valid_timestamp = [t for t in self.pose_graph_used_pose.keys() if t + 10 * 1e9 < timestamp]
                    if len(valid_timestamp) == 0:
                        return
                    target_embedding = self.nav_temp_db.get_embedding(timestamp)
                    _, _, curr_features, _, _ = self.nav_temp_db.get_depth_embedding_features_images(timestamp)
                    curr_kp = curr_features["kpts"][0] if curr_features["kpts"].ndim == 3 else curr_features["kpts"]
                    curr_desc = curr_features["descps"][0] if curr_features["descps"].ndim == 3 else curr_features["descps"]
                    with Timer(name = "find loop", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        loop_list = self.nav_loop_closure.find_candidate_timestamps(
                            curr_kp,
                            curr_desc,
                            target_embedding,
                            top_k=self.loop_top_k,
                            allowed_timestamps=set(valid_timestamp),
                        )
                    with Timer(name = "Relative pose estimation", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        for candidate in loop_list:
                            prev_timestamp = candidate["timestamp"]
                            curr_timestamp = timestamp
                            similarity = float(candidate["similarity"])
                            self.logger.info(
                                f"Loop candidate curr={curr_timestamp} prev={prev_timestamp} similarity={float(similarity):.4f}"
                            )
                            prev_depth, _, prev_features, _, _ = self.nav_temp_db.get_depth_embedding_features_images(prev_timestamp)
                            curr_depth, _, curr_features, _, _ = self.nav_temp_db.get_depth_embedding_features_images(curr_timestamp)
                            prev_matched_keypoints, curr_matched_keypoints, matches = self.match_keypoints(prev_features, curr_features)
                            success, T_prev_curr, _, _, inliers = estimate_pose(prev_matched_keypoints, curr_matched_keypoints, curr_depth, self.K)
                            if success and len(inliers) >= 100:
                                self.relative_pose_constraint.append((curr_timestamp, prev_timestamp, T_prev_curr))
                                self.logger.info(
                                    f"Loop accepted curr={curr_timestamp} prev={prev_timestamp} similarity={float(similarity):.4f} inliers={len(inliers)}"
                                )
                    with Timer(name = "solve pose graph", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        self.pose_graph_used_pose = solve_pose_graph(self.pose_graph_used_pose, self.relative_pose_constraint, max_iteration_num = 5)
            find_loop_and_pose_graph(keyframe_image_timestamp)
            self.nav_loop_closure.add_timestamp(keyframe_image_timestamp)
            self.pose_graph_trajectory_publish(keyframe_image_timestamp)
        self.last_keyframe_timestamp = keyframe_odom_timestamp
        self.last_keyframe_image = image


    def get_embeddings(self, image: np.ndarray) -> np.ndarray:
        # shape: (1, 768)
        return asyncio.run(self.embedding_extractor.infer(image))

    def match_keypoints(self, feats0:dict, feats1:dict, image_shape = np.array([848, 480], dtype = np.int64)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        match_result = asyncio.run(self.matcher.infer(feats0["kpts"], feats1["kpts"], feats0['descps'], feats1['descps'], feats0['mask'], feats1['mask'], image_shape, image_shape))
        match_indices = match_result["match_indices"][0]
        if feats0["kpts"].ndim != 3 or feats1["kpts"].ndim != 3:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.int64)
        if feats0["kpts"].shape[0] == 0 or feats1["kpts"].shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.int64)
        # Guard against invalid indices returned by matcher.
        match_indices = match_indices.copy()
        invalid = (match_indices < 0) | (match_indices >= feats1["kpts"][0].shape[0]) | (np.arange(match_indices.shape[0]) >= feats0["kpts"][0].shape[0])
        match_indices[invalid] = -1
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

    def _relocalization_failed(self, reason: str) -> tuple[bool, np.ndarray, float]:
        self.last_relocalization_failure_reason = reason
        return False, np.eye(4), -np.inf

    def _log_relocalization_timing(self, timestamp_ns: int, success: bool, timings: dict[str, float], extra: str = ""):
        total_ms = sum(timings.values())
        parts = ", ".join(f"{name}={ms:.1f}" for name, ms in timings.items())
        msg = f"Relocalization stage timing ms: timestamp={timestamp_ns}, total={total_ms:.1f}, {parts}, success={success}"
        if extra:
            msg += f", {extra}"
        if total_ms > 500.0:
            self.get_logger().warning(msg)
        else:
            self.get_logger().info(msg)

    def relocalize_with_depth(self, keyframe: np.ndarray, keyframe_features: dict, K: np.ndarray | None, timings: dict[str, float] | None = None) -> tuple[bool, np.ndarray, float]:
        if timings is None:
            timings = {}
        self.last_relocalization_failure_reason = ""
        if K is None:
            return self._relocalization_failed("camera intrinsics unavailable")
        t0 = time.perf_counter()
        query_embedding = self.get_embeddings(keyframe)
        query_embedding_norm = np.linalg.norm(query_embedding)
        if query_embedding_norm > 0:
            query_embedding = query_embedding / query_embedding_norm
        timings["embedding"] = timings.get("embedding", 0.0) + (time.perf_counter() - t0) * 1000.0

        query_kp = keyframe_features["kpts"][0] if keyframe_features["kpts"].ndim == 3 else keyframe_features["kpts"]
        query_desc = keyframe_features["descps"][0] if keyframe_features["descps"].ndim == 3 else keyframe_features["descps"]
        t0 = time.perf_counter()
        candidates = self.map_loop_closure.find_candidate_timestamps(
            query_kp,
            query_desc,
            query_embedding,
            top_k=self.relocalization_loop_top_k,
        )
        timings["candidate_search"] = timings.get("candidate_search", 0.0) + (time.perf_counter() - t0) * 1000.0
        max_similarity = max([c["similarity"] for c in candidates]) if len(candidates) > 0 else 0
        if len(candidates) > 0:
            point_3d_in_world_arrays = []
            point_2d_in_keyframe_arrays = []
            candidate_summaries = []
            candidate_timing_summaries = []
            for candidate in candidates:
                timestamp_in_map = int(candidate["timestamp"])
                similarity = float(candidate["similarity"])
                reference_keyframe_pose = self.map_poses[timestamp_in_map]
                t_db = time.perf_counter()
                reference_depth, _, reference_features, _, _ = self.db.get_depth_embedding_features_images(timestamp_in_map)
                db_ms = (time.perf_counter() - t_db) * 1000.0
                timings["db_load"] = timings.get("db_load", 0.0) + db_ms
                t_match = time.perf_counter()
                reference_matched_keypoints, keyframe_matched_keypoints, matches = self.match_keypoints(reference_features, keyframe_features)
                match_ms = (time.perf_counter() - t_match) * 1000.0
                timings["match"] = timings.get("match", 0.0) + match_ms
                depth_ms = 0.0
                if len(matches) >= 20:
                    t_depth = time.perf_counter()
                    point_3d_in_world, inliers = self.keypoint_with_depth_to_3d(reference_matched_keypoints, reference_depth, reference_keyframe_pose, self.map_K)
                    depth_ms = (time.perf_counter() - t_depth) * 1000.0
                    timings["depth3d"] = timings.get("depth3d", 0.0) + depth_ms
                    point_3d_in_world_arrays.append(point_3d_in_world[inliers])
                    point_2d_in_keyframe_arrays.append(keyframe_matched_keypoints[inliers])
                    candidate_summaries.append(
                        f"{timestamp_in_map}:sim={similarity:.3f},matches={len(matches)},valid_depth={int(np.count_nonzero(inliers))}"
                    )
                else:
                    candidate_summaries.append(
                        f"{timestamp_in_map}:sim={similarity:.3f},matches={len(matches)}<20"
                    )
                candidate_timing_summaries.append(
                    f"{timestamp_in_map}:db={db_ms:.1f},match={match_ms:.1f},depth3d={depth_ms:.1f}"
                )

            landmark_count = int(sum(points.shape[0] for points in point_3d_in_world_arrays))
            if landmark_count > 40:
                point_3d_in_world_list = np.concatenate(point_3d_in_world_arrays, axis=0)
                point_2d_in_keyframe_list = np.concatenate(point_2d_in_keyframe_arrays, axis=0)

                t_pnp = time.perf_counter()
                success, rvec, tvec, inliers = cv2.solvePnPRansac(point_3d_in_world_list, point_2d_in_keyframe_list, self.map_K, None)
                timings["pnp"] = timings.get("pnp", 0.0) + (time.perf_counter() - t_pnp) * 1000.0
                if success and len(inliers) >= 20:
                    R, _ = cv2.Rodrigues(rvec)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = tvec.reshape(3)
                    print(f"relocalization pose : {T}")
                    self.get_logger().info(f"Relocalization candidate timing ms: {'; '.join(candidate_timing_summaries)}")
                    return True, T, len(inliers) / len(point_2d_in_keyframe_list)
                inlier_count = 0 if inliers is None else len(inliers)
                self.get_logger().info(f"Relocalization candidate timing ms: {'; '.join(candidate_timing_summaries)}")
                return self._relocalization_failed(
                    f"solvePnPRansac failed or insufficient inliers: success={success}, "
                    f"inliers={inlier_count}<20, landmarks={len(point_3d_in_world_list)}, "
                    f"candidates=[{'; '.join(candidate_summaries)}]"
                )
            else:
                self.get_logger().info(f"Relocalization candidate timing ms: {'; '.join(candidate_timing_summaries)}")
                return self._relocalization_failed(
                    f"not enough valid depth landmarks: {landmark_count}<=40, "
                    f"candidates=[{'; '.join(candidate_summaries)}]"
                )
        else:
            return self._relocalization_failed(
                f"no loop candidates above threshold: candidates={len(candidates)}, max_similarity={max_similarity:.3f}"
            )
        return self._relocalization_failed("unknown relocalization failure")

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
        loop_t0 = time.perf_counter()
        timings = {}
        t0 = time.perf_counter()
        features = asyncio.run(self.extractor.infer(image))
        timings["feature_extract"] = (time.perf_counter() - t0) * 1000.0
        res, pose_in_camera, pose_cov_weight = self.relocalize_with_depth(image, features, self.K, timings)
        if res:
            t0 = time.perf_counter()
            # publish the relocalization pose for debug
            pose_in_world = se3_inv(pose_in_camera)
            self.relocation_pub.publish(np2msg(pose_in_world, timestamp, "world", "camera"))
            self.relocalization_poses[timestamp_ns] = pose_in_world
            self.relocalization_pose_weights[timestamp_ns] = pose_cov_weight
            timings["publish"] = (time.perf_counter() - t0) * 1000.0
            timings["unaccounted"] = max(0.0, (time.perf_counter() - loop_t0) * 1000.0 - sum(timings.values()))
            self.last_relocalization_timing = dict(timings)
            self._log_relocalization_timing(
                timestamp_ns,
                True,
                timings,
                extra=f"pose_weight={pose_cov_weight:.3f}",
            )
            return True, pose_in_world
        else:
            self.failed_relocalizations.append(timestamp)
            reason = self.last_relocalization_failure_reason or "unknown relocalization failure"
            timings["unaccounted"] = max(0.0, (time.perf_counter() - loop_t0) * 1000.0 - sum(timings.values()))
            self.last_relocalization_timing = dict(timings)
            self._log_relocalization_timing(timestamp_ns, False, timings, extra=f"reason={reason}")
            self.get_logger().warning(
                f"Relocalization failed at timestamp={timestamp_ns}: {reason}"
            )
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
            0 : np.eye(4) if self.T_from_map_to_odom is None else self.T_from_map_to_odom,
            1 : np.eye(4),
        }
        constant_pose_index_dict = { 1: True }
        for timestamp, pose in self.relocalization_poses.items():
            if timestamp in self.pose_graph_used_pose:
                camera_in_map_world = pose
                camera_in_odom_world = self.pose_graph_used_pose[timestamp]
                observation_T_from_map_to_odom =  camera_in_odom_world @ se3_inv(camera_in_map_world)
                weight = self.relocalization_pose_weights[timestamp]

                relative_pose_constraint.append((0, 1, observation_T_from_map_to_odom, weight * np.array([10.0, 10.0, 10.0]), weight * np.array([10.0, 10.0, 10.0])))
        relative_pose_constraint = relative_pose_constraint[-100:]
        optimized_parameters = pose_graph_solve(optimized_parameters, relative_pose_constraint, constant_pose_index_dict, max_iteration_num = 1000)
        self.T_from_map_to_odom = optimized_parameters[0]

    def try_publish_nav_path(self, timestamp: int):
        t_start = time.perf_counter()
        stage_timings = {}

        def mark_stage(name: str, previous_t: float) -> float:
            now = time.perf_counter()
            stage_timings[name] = (now - previous_t) * 1000.0
            return now

        def log_nav_timing(result: str, path_count: int | None = None):
            total_ms = (time.perf_counter() - t_start) * 1000.0
            parts = ", ".join(f"{name}={ms:.1f}" for name, ms in stage_timings.items())
            msg = f"Nav path timing ms: timestamp={timestamp}, total={total_ms:.1f}, result={result}"
            if path_count is not None:
                msg += f", path_count={path_count}"
            if parts:
                msg += f", {parts}"
            if total_ms > 500.0:
                self.get_logger().warning(msg)
            else:
                self.get_logger().info(msg)

        t_stage = t_start
        self.get_logger().info(f"try_publish_nav_path, timestamp: {timestamp}")
        t_stage = mark_stage("start_log", t_stage)
        if self.T_from_map_to_odom is None:
            self.get_logger().info("Relocalization not successful yet, skip publishing nav path")
            log_nav_timing("skip_no_relocalization")
            return

        pose_in_map = se3_inv(self.T_from_map_to_odom) @ self.pose_graph_used_pose[timestamp]
        self.current_pose_in_map_pub.publish(np2msg(pose_in_map, self.get_clock().now().to_msg(), "world", "map"))
        pose_in_map_position = pose_in_map[:3, 3]
        t_stage = mark_stage("pose_in_map_publish", t_stage)

        if self.poi_index == -1:
            self.get_logger().info("No POI found, skip publishing nav path")
            log_nav_timing("skip_no_poi")
            return

        if self.poi_index >= len(self.pois):
            self.get_logger().info("All POIs have been visited, skip publishing nav path")
            log_nav_timing("skip_all_pois_visited")
            return

        poi = self.pois[self.poi_index]
        print(f"poi: {poi}")
        poi_pose = np.eye(4)
        poi_pose[:3, 3] = poi
        self.poi_pub.publish(np2msg(poi_pose, self.get_clock().now().to_msg(), "world", "map"))
        t_stage = mark_stage("poi_publish", t_stage)

        advanced_poi_count = 0
        while self.poi_index < len(self.pois):
            poi = self.pois[self.poi_index]
            diff_position_norm_xy = np.linalg.norm(poi[:2] - pose_in_map_position[:2])
            diff_position_norm_z = np.linalg.norm(poi[2] - pose_in_map_position[2])
            if diff_position_norm_xy < 0.5 and diff_position_norm_z < 2.0:
                self.poi_index += 1
                advanced_poi_count += 1
                dummy_pose = np.eye(4)

                stamp_msg = self.get_clock().now().to_msg()
                stamp_msg.sec = int(timestamp / 1e9)
                stamp_msg.nanosec = int(timestamp % 1e9)
                self.poi_change_pub.publish(np2msg(dummy_pose, stamp_msg, "world", "map"))
                continue
            else:
                break
        t_stage = mark_stage("poi_advance", t_stage)

        if self.poi_index >= len(self.pois):
            self.get_logger().info("All POIs have been visited, skip publishing nav path")
            log_nav_timing(f"skip_all_pois_visited_after_advance:{advanced_poi_count}")
            return

        target_poi = self.pois[self.poi_index]
        with Timer(name = "generate nav path in map", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            paths_in_map = self.generate_nav_path_in_map(pose_in_map = pose_in_map, target_poi = target_poi)
        t_stage = mark_stage("generate_path", t_stage)

        if paths_in_map is not None:
            # use the max_speed to publish the position the robot should be after 10 seconds
            with Timer(name = "Find target position", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                max_speed = 0.5
                lookahead_seconds = 10.0
                if len(paths_in_map) > 1:
                    accumulated_distance = 0.0
                    start_point = pose_in_map_position[:3]
                    target_position = paths_in_map[-1]
                    for i in range(len(paths_in_map) - 1):
                        accumulated_distance += np.linalg.norm(paths_in_map[i] - start_point)
                        if accumulated_distance > max_speed * lookahead_seconds:
                            target_position = paths_in_map[i]
                            break
                        start_point = paths_in_map[i]
                else:
                    target_position = paths_in_map[0]
                target_position_in_map = np.array([target_position[0], target_position[1], target_position[2]])
                pose_in_origin_odom = self.odom[timestamp]
                T = pose_in_origin_odom @ se3_inv(pose_in_map)
                target_position_in_odom = T[:3, :3] @ target_position_in_map + T[:3, 3]
                dummy_pose = np.eye(4)
                dummy_pose[:3, 3] = target_position_in_odom
                #logging.info(f"target_position_in_odom: {target_position_in_odom}")
                print(f"target_position_in_odom: {target_position_in_odom}")
            t_stage = mark_stage("target_select", t_stage)

            self.target_pose_pub.publish(np2msg(dummy_pose, self.get_clock().now().to_msg(), "world", "camera"))
            t_stage = mark_stage("target_pose_publish", t_stage)

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
            t_stage = mark_stage("global_path_msg_build", t_stage)

            self.global_plan_pub.publish(path_msg)
            t_stage = mark_stage("global_path_publish", t_stage)

            self.tf_broadcaster.sendTransform(np2tf(T, self.get_clock().now().to_msg(), "world", "map"))
            mark_stage("tf_publish", t_stage)
            log_nav_timing("published", path_count=len(paths_in_map))
        else:
            logging.info("No path found in map")
            log_nav_timing("skip_no_path")

    def generate_nav_path_in_map(self, pose_in_map: np.ndarray, target_poi: np.ndarray) -> np.ndarray:
        t_start = time.perf_counter()
        stage_timings = {}

        def mark_stage(name: str, previous_t: float) -> float:
            now = time.perf_counter()
            stage_timings[name] = (now - previous_t) * 1000.0
            return now

        def log_generate_timing(
            result: str,
            start_path_count: int = 0,
            goal_path_count: int = 0,
            path_sdf_count: int = 0,
            path_count: int = 0,
        ):
            total_ms = (time.perf_counter() - t_start) * 1000.0
            parts = ", ".join(f"{name}={ms:.1f}" for name, ms in stage_timings.items())
            msg = (
                f"Generate nav path timing ms: total={total_ms:.1f}, result={result}, "
                f"start_path_count={start_path_count}, goal_path_count={goal_path_count}, "
                f"path_sdf_count={path_sdf_count}, path_count={path_count}"
            )
            if parts:
                msg += f", {parts}"
            if total_ms > 500.0:
                self.get_logger().warning(msg)
            else:
                self.get_logger().info(msg)

        t_stage = t_start
        dummy_poi_pose = np.eye(4)
        dummy_poi_pose[:3, 3] = target_poi
        self.poi_pub.publish(np2msg(dummy_poi_pose, self.get_clock().now().to_msg(), "world", "map"))
        t_stage = mark_stage("poi_publish", t_stage)

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
        t_stage = mark_stage("index_convert", t_stage)

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
            print("here")
            log_generate_timing("out_of_bounds")
            return None 

        sdf_start_path = search_close_to_sdf_map_numba(start_idx, self.sdf_map, self.occupancy_map, 0.2)
        t_stage = mark_stage("search_start_close", t_stage)
        sdf_goal_path = search_close_to_sdf_map_numba(poi_goal_idx, self.sdf_map, self.occupancy_map, 0.2)
        t_stage = mark_stage("search_goal_close", t_stage)

        if len(sdf_start_path) == 0 or len(sdf_goal_path) == 0:
            self.get_logger().warning(
                f"search_close_to_sdf_map returned empty path: start_count={len(sdf_start_path)}, goal_count={len(sdf_goal_path)}"
            )
            log_generate_timing("empty_close_path", start_path_count=len(sdf_start_path), goal_path_count=len(sdf_goal_path))
            return None

        sdf_start_sdf = sdf_start_path[-1]
        sdf_goal_sdf = sdf_goal_path[-1]
        path_sdf = search_within_sdf_map_numba(sdf_start_sdf, sdf_goal_sdf, self.sdf_map, self.occupancy_map, resolution)
        t_stage = mark_stage("search_within", t_stage)
        if len(path_sdf) == 0:
            self.get_logger().warning(
                f"search_within_sdf_map returned empty path: start_idx={tuple(sdf_start_sdf)}, goal_idx={tuple(sdf_goal_sdf)}"
            )
        path = np.vstack((sdf_start_path, path_sdf, sdf_goal_path[::-1]))
        t_stage = mark_stage("path_stack", t_stage)
        if len(path) > 0:
            converted_path = path.astype(np.float64) * resolution + occupancy_map_origin
            mark_stage("path_convert", t_stage)
            log_generate_timing(
                "ok",
                start_path_count=len(sdf_start_path),
                goal_path_count=len(sdf_goal_path),
                path_sdf_count=len(path_sdf),
                path_count=len(path),
            )
            return converted_path
        log_generate_timing(
            "empty_path",
            start_path_count=len(sdf_start_path),
            goal_path_count=len(sdf_goal_path),
            path_sdf_count=len(path_sdf),
        )
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
    parser.add_argument("--loop-closure-mode", type=str, default="embedding", choices=["embedding", "bow"])
    parser.add_argument("--loop-closure-use-bow", action="store_true", help="Use ORB+BF and DBoW3 for loop closure")
    parser.add_argument(
        "--dbow3-vocabulary-path",
        type=str,
        default="/tinynav/docs/Vocabulary/ORBvoc.txt",
        help="DBoW3 vocabulary path for bow mode",
    )
    parsed_args, unknown_args = parser.parse_known_args(sys.argv[1:])

    use_bow = parsed_args.loop_closure_use_bow or parsed_args.loop_closure_mode == "bow"
    if use_bow:
        parsed_args.loop_closure_mode = "bow"
        if not os.path.exists(parsed_args.dbow3_vocabulary_path):
            raise FileNotFoundError(
                f"DBoW3 vocabulary file not found: {parsed_args.dbow3_vocabulary_path}"
            )
        extractor = ORBFeatureTRTCompatible()
        matcher = ORBMatcher()
        embedding_extractor = DummyEmbeddingEngine()
    else:
        extractor = SuperPointTRT()
        matcher = LightGlueTRT()
        embedding_extractor = Dinov2TRT()

    node = MapNode(
        tinynav_db_path=parsed_args.tinynav_db_path,
        tinynav_map_path=parsed_args.tinynav_map_path,
        extractor=extractor,
        matcher=matcher,
        embedding_extractor=embedding_extractor,
        loop_closure_mode=parsed_args.loop_closure_mode,
        loop_closure_use_bow=use_bow,
        dbow3_vocabulary_path=parsed_args.dbow3_vocabulary_path,
        verbose_timer=parsed_args.verbose_timer,
    )

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
