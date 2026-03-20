import argparse
import matplotlib
matplotlib.use('Agg')
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointField
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point32
from cv_bridge import CvBridge
import numpy as np
from scipy.ndimage import maximum_filter, distance_transform_edt
from numba import njit
import message_filters
import matplotlib.pyplot as plt
from rclpy.time import Time
import io
from PIL import Image as PIL_Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointCloud
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from codetiming import Timer
import cv2
import heapq
from math_utils import rotvec_to_matrix, quat_to_matrix, matrix_to_quat, msg2np
from geometry_msgs.msg import Twist


# Footprint params (meters)
SAFETY_RADIUS = 0.3
HALF_SAFETY_LENGTH = 0.4
HALF_SAFETY_WIDTH = 0.15
CAMERA_FORWARD_OFFSET = 2.0 * HALF_SAFETY_LENGTH  # camera(front-center) -> rear reference point

# === Helper functions ===
@njit(cache=True)
def run_raycasting_loopy(depth_image, T_cam_to_world, grid_shape, fx, fy, cx, cy, origin, step, resolution, filter_ground = False):
    """
    A "C-style" version of run_raycasting that uses explicit loops instead of
    NumPy vector operations, designed for optimal Numba performance.
    Reference: https://numba.readthedocs.io/en/stable/user/performance-tips.html#loops
    """
    occupancy_grid = np.zeros(grid_shape)
    depth_height, depth_width = depth_image.shape

    grid_shape_x, grid_shape_y, grid_shape_z = grid_shape
    origin_x, origin_y, origin_z = origin

    cam_orig_x = T_cam_to_world[0, 3]
    cam_orig_y = T_cam_to_world[1, 3]
    cam_orig_z = T_cam_to_world[2, 3]

    start_voxel_x = int(np.floor((cam_orig_x - origin_x) / resolution))
    start_voxel_y = int(np.floor((cam_orig_y - origin_y) / resolution))
    start_voxel_z = int(np.floor((cam_orig_z - origin_z) / resolution))

    for v in range(0, depth_height, step):
        for u in range(0, depth_width, step):
            d = depth_image[v, u]
            if d <= 0:
                continue

            # Project to camera coordinates
            px = (u - cx) * d / fx
            py = (v - cy) * d / fy
            pz = d
            is_ground = py > 0

            # Transform to world coordinates (manual matrix multiplication)
            pw_x = T_cam_to_world[0, 0] * px + T_cam_to_world[0, 1] * py + T_cam_to_world[0, 2] * pz + T_cam_to_world[0, 3]
            pw_y = T_cam_to_world[1, 0] * px + T_cam_to_world[1, 1] * py + T_cam_to_world[1, 2] * pz + T_cam_to_world[1, 3]
            pw_z = T_cam_to_world[2, 0] * px + T_cam_to_world[2, 1] * py + T_cam_to_world[2, 2] * pz + T_cam_to_world[2, 3]

            # Calculate end voxel
            end_voxel_x = int(np.floor((pw_x - origin_x) / resolution))
            end_voxel_y = int(np.floor((pw_y - origin_y) / resolution))
            end_voxel_z = int(np.floor((pw_z - origin_z) / resolution))

            # Bresenham's line algorithm (simplified)
            diff_x = end_voxel_x - start_voxel_x
            diff_y = end_voxel_y - start_voxel_y
            diff_z = end_voxel_z - start_voxel_z

            steps = max(abs(diff_x), abs(diff_y), abs(diff_z))
            if steps == 0:
                continue

            for i in range(steps + 1):
                t = i / steps
                interp_x = int(round(start_voxel_x + t * diff_x))
                interp_y = int(round(start_voxel_y + t * diff_y))
                interp_z = int(round(start_voxel_z + t * diff_z))

                if (0 <= interp_x < grid_shape_x and
                    0 <= interp_y < grid_shape_y and
                    0 <= interp_z < grid_shape_z):
                    occupancy_grid[interp_x, interp_y, interp_z] -= 0.05

            if (0 <= end_voxel_x < grid_shape_x and
                0 <= end_voxel_y < grid_shape_y and
                0 <= end_voxel_z < grid_shape_z):
                if filter_ground and is_ground:
                    pass
                else:
                    occupancy_grid[end_voxel_x, end_voxel_y, end_voxel_z] += 0.2

    # Explicit clipping loop
    for i in range(grid_shape_x):
        for j in range(grid_shape_y):
            for k in range(grid_shape_z):
                if occupancy_grid[i, j, k] < -0.1:
                    occupancy_grid[i, j, k] = -0.1
                elif occupancy_grid[i, j, k] > 0.1:
                    occupancy_grid[i, j, k] = 0.1

    return occupancy_grid


@njit(cache=True)
def occupancy_grid_to_height_map(occupancy_grid, origin, resolution, threshold=0.1, method='max'):
    X, Y, Z = occupancy_grid.shape
    height_map = np.full((X, Y), -np.nan, dtype=np.float32)
    for x in range(X):
        for y in range(Y):
            zs = []
            for z in range(Z):
                if occupancy_grid[x, y, z] >= threshold:
                    world_z = origin[2] + (z + 0.5) * resolution
                    zs.append(world_z)
            if zs:
                if method == 'max':
                    height_map[x, y] = max(zs)
                elif method == 'min':
                    height_map[x, y] = min(zs)
    return height_map

def max_pool_height_map(height_map, kernel_size=1):
    nan_mask = np.isnan(height_map)
    filled = np.copy(height_map)
    filled[nan_mask] = -np.inf
    pooled = maximum_filter(filled, size=kernel_size, mode='nearest')
    return pooled

def height_map_to_ESDF(height_map, height_threshold, resolution, method='max'):
    if method == 'max':
        occupancy = (height_map > height_threshold).astype(np.float32)
    elif method == 'min':
        occupancy = (height_map < height_threshold).astype(np.float32)
    else:
        raise ValueError(f"Invalid method: {method}. Use 'max' or 'min'.")
    
    if np.any(occupancy):
        esdf = distance_transform_edt(occupancy == 0)
    else:
        esdf = np.full_like(occupancy, 100.0, dtype=np.float32)

    return resolution * esdf

@njit(cache=True)
def generate_trajectory_library_3d(
    num_samples=11, duration=2.0, dt=0.1,
    acc_std=0.00001, omega_y_std_deg=20.0,
    init_p=np.zeros(3), init_v=np.zeros(3), init_q=np.array([0, 0, 0, 1])
):
    num_steps = int(duration / dt) + 1
    max_velocity = 0.5
    velocity_samples = np.linspace(-max_velocity * 0.4, max_velocity, num_samples)
    max_omega = np.pi / 6
    omega_y_samples = np.linspace(-max_omega, max_omega, num_samples * 2)
    num_samples = len(velocity_samples) * len(omega_y_samples)

    trajectories = np.empty((num_samples, num_steps, 7))
    params = np.empty((num_samples, 2))

    k = -1
    for i_velocity in range(len(velocity_samples)):
        for i_omega in range(len(omega_y_samples)):
            k += 1
            dv = velocity_samples[i_velocity]
            omega_y = omega_y_samples[i_omega]
            p = init_p.copy()
            v_world = init_v.copy()
            q = quat_to_matrix(init_q)
            traj = np.empty((num_steps, 7))
            for i in range(num_steps):
                dq = rotvec_to_matrix(np.array([0.0, omega_y * dt, 0.0]))
                v_world = (q @ dq) @ q.T @ v_world
                q = q @ dq
                dv_camera = np.array([0.0, 0.0, dv])
                dv_world = q @ dv_camera
                v_world += dv_world * dt
                v_world = np.clip(v_world, -0.5, 0.5)
                p += v_world * dt
                traj[i, :3] = p
                traj[i, 3:] = matrix_to_quat(q)
            #hack
            for i in range(num_steps):
                traj[i, 2] = traj[0, 2]
            trajectories[k] = traj
            params[k, 0] = dv
            params[k, 1] = omega_y
    trajectories = trajectories[:k+1]
    params = params[:k+1]
    return trajectories, params

@njit(cache=True)
def score_trajectories_by_height_map(trajectories, height_map, origin, resolution):
    scores = []
    height_map_rows, height_map_cols = height_map.shape
    height_values = []
    for t in range(len(trajectories)):
        traj = trajectories[t]
        cost = 0.0
        height_value = []
        cum_distance = 2 * HALF_SAFETY_LENGTH
        for i in range(len(traj)):
            x_world, y_world, z_world = traj[i, 0], traj[i, 1], traj[i, 2]
            if cum_distance >= HALF_SAFETY_LENGTH or i == len(traj) - 1:
                cum_distance = 0.0
            else:
                cum_distance += np.linalg.norm(traj[i, :3] - traj[i-1, :3])
                continue

            quat = traj[i, 3:]
            rotation = quat_to_matrix(quat)
            delta_x_index_max = int(HALF_SAFETY_WIDTH / resolution)
            delta_z_index_max = int(HALF_SAFETY_LENGTH / resolution)
            grid_in_camera = np.zeros((2 * delta_x_index_max + 1, 2 * delta_z_index_max + 1))
            for delta_x_index in range(-delta_x_index_max, delta_x_index_max + 1):
                for delta_z_index in range(-delta_z_index_max, delta_z_index_max + 1):
                    x_in_camera = delta_x_index * resolution
                    z_in_camera = delta_z_index * resolution - HALF_SAFETY_LENGTH
                    y_in_camera = 0
                    point_in_camera = np.array([x_in_camera, y_in_camera, z_in_camera])
                    point_in_world = rotation @ point_in_camera + np.array([x_world, y_world, z_world])
                    x_img = int((point_in_world[0] - origin[0]) / resolution)
                    y_img = int((point_in_world[1] - origin[1]) / resolution)
                    if 0 <= x_img < height_map_rows and 0 <= y_img < height_map_cols:
                        delta_height = point_in_world[2] - height_map[x_img, y_img]
                        # magic number
                        if (delta_height > 0.05):
                            cost += 0.0  # Trajectory is above the height map, no collision
                        else:
                            # Trajectory is at or below the height map, add collision cost
                            bounding_distance = np.sqrt(x_in_camera**2 + z_in_camera**2)
                            cost += 1.0 + (2.0 - bounding_distance)
                        height_value.append(height_map[x_img, y_img])
                        grid_in_camera[delta_x_index + delta_x_index_max, delta_z_index + delta_z_index_max] = height_map[x_img, y_img]
                    else:
                        height_value.append(-np.inf)
                        grid_in_camera[delta_x_index + delta_x_index_max, delta_z_index + delta_z_index_max] = -np.inf
        height_values.append(height_value)
        scores.append(cost)

    return scores, height_values

@njit(cache=True)
def score_trajectories_by_ESDF(trajectories, ESDF_map, origin, resolution):
    scores = []
    occ_points = []
    ESDF_rows, ESDF_cols = ESDF_map.shape

    for t in range(len(trajectories)):
        traj = trajectories[t]
        min_dist_for_traj = float('inf')
        closest_step_for_traj = -1  # Initialize as -1 to indicate no step found

        for i in range(len(traj)):
            x_world, y_world, _ = traj[i, 0], traj[i, 1], traj[i, 2]
            x_img = int((x_world - origin[0]) / resolution)
            y_img = int((y_world - origin[1]) / resolution)
            if 0 <= x_img < ESDF_rows and 0 <= y_img < ESDF_cols:
                dist = ESDF_map[x_img, y_img]
                if dist < min_dist_for_traj:
                    min_dist_for_traj = dist
                    closest_step_for_traj = i
        # Scoring based on the found minimum distance and the step it occurred
        if min_dist_for_traj < 1e-3: # Consider it a collision
            scores.append(float('inf'))
        elif min_dist_for_traj != float('inf'):
            if min_dist_for_traj > SAFETY_RADIUS:
                scores.append(0.0)
            else:
                max_steps = len(traj)
                decay_factor = (max_steps - closest_step_for_traj) / max_steps
                base_score = 1.0 / (min_dist_for_traj+1e-3)
                scores.append(decay_factor * base_score)
        else:
            # If no obstacle is near, score is 0, closest_step_for_traj is the last step
            scores.append(0.0)
        occ_points.append(closest_step_for_traj)
    return scores, occ_points

def roll_occupancy_grid(occupancy_grid, old_origin, new_origin, resolution):
    shift_m = new_origin - old_origin
    shift_voxels = np.round(shift_m / resolution).astype(int)
    if np.all(shift_voxels == 0):
        return occupancy_grid, old_origin
    rolled = np.roll(occupancy_grid, shift=tuple(-shift_voxels), axis=(0, 1, 2))
    x, y, z = occupancy_grid.shape
    if shift_voxels[0] > 0:
        rolled[-shift_voxels[0]:, :, :] = 0
    elif shift_voxels[0] < 0:
        rolled[:-shift_voxels[0], :, :] = 0
    if shift_voxels[1] > 0:
        rolled[:, -shift_voxels[1]:, :] = 0
    elif shift_voxels[1] < 0:
        rolled[:, :-shift_voxels[1], :] = 0
    if shift_voxels[2] > 0:
        rolled[:, :, -shift_voxels[2]:] = 0
    elif shift_voxels[2] < 0:
        rolled[:, :, :-shift_voxels[2]] = 0
    updated_origin = old_origin + shift_voxels * resolution
    return rolled, updated_origin


def world_to_grid_2d(point_xy, origin, resolution):
    gx = int((point_xy[0] - origin[0]) / resolution)
    gy = int((point_xy[1] - origin[1]) / resolution)
    return (gx, gy)


def grid_to_world_2d(grid_xy, origin, resolution, z):
    return np.array([
        origin[0] + (grid_xy[0] + 0.5) * resolution,
        origin[1] + (grid_xy[1] + 0.5) * resolution,
        z,
    ])


def sample_height_from_map(point_xy, height_map, origin, resolution, default_z):
    gx = int((point_xy[0] - origin[0]) / resolution)
    gy = int((point_xy[1] - origin[1]) / resolution)
    h, w = height_map.shape
    if 0 <= gx < h and 0 <= gy < w:
        z = height_map[gx, gy]
        if np.isfinite(z):
            return float(z)
    return float(default_z)


def clip_grid_2d(grid_xy, shape):
    x = min(max(grid_xy[0], 0), shape[0] - 1)
    y = min(max(grid_xy[1], 0), shape[1] - 1)
    return (x, y)


def find_nearest_free(start, obstacle_mask, search_radius=8):
    sx, sy = start
    h, w = obstacle_mask.shape
    if 0 <= sx < h and 0 <= sy < w and (not obstacle_mask[sx, sy]):
        return (sx, sy)
    for r in range(1, search_radius + 1):
        x0, x1 = max(0, sx - r), min(h - 1, sx + r)
        y0, y1 = max(0, sy - r), min(w - 1, sy + r)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                if not obstacle_mask[x, y]:
                    return (x, y)
    return None


def astar_2d(cost_map, obstacle_mask, start, goal, search_mask=None, footprint_check_fn=None):
    if start is None or goal is None:
        return []
    if obstacle_mask[start] or obstacle_mask[goal]:
        return []
    if search_mask is not None:
        if (not search_mask[start]) or (not search_mask[goal]):
            return []

    def heuristic(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142),
    ]

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
    g_score = {start: 0.0}
    parent = {start: start}
    closed = set()

    h, w = obstacle_mask.shape
    while open_heap:
        _, g_cur, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)

        if cur == goal:
            path = [cur]
            while path[-1] != parent[path[-1]]:
                path.append(parent[path[-1]])
            path.reverse()
            return path

        cx, cy = cur
        for dx, dy, step_cost in neighbors:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= h or ny < 0 or ny >= w:
                continue
            if obstacle_mask[nx, ny]:
                continue
            # no corner cutting on diagonals
            if dx != 0 and dy != 0:
                if obstacle_mask[cx + dx, cy] or obstacle_mask[cx, cy + dy]:
                    continue
            if search_mask is not None and (not search_mask[nx, ny]):
                continue
            n = (nx, ny)
            if footprint_check_fn is not None and footprint_check_fn(n):
                continue
            trans_cost = step_cost * (0.5 * (cost_map[cx, cy] + cost_map[nx, ny]))
            new_g = g_cur + trans_cost
            if new_g < g_score.get(n, float('inf')):
                g_score[n] = new_g
                parent[n] = cur
                f = new_g + heuristic(n, goal)
                heapq.heappush(open_heap, (f, new_g, n))
    return []


def build_astar_cost_map(height_map, esdf_map, resolution):
    finite = np.isfinite(height_map)
    filled = np.nan_to_num(height_map, nan=0.0, neginf=0.0, posinf=0.0)
    gx, gy = np.gradient(filled)
    slope = np.sqrt(gx * gx + gy * gy)

    # Keep slope debug maps for RViz.
    slope_block = slope > 0.6
    slope_feasible = (~slope_block) & finite

    # ESDF obstacle source (using robot-relative height derived ESDF).
    stair_like = slope > 0.4
    # Slightly stricter ESDF hard-obstacle threshold to better reject wall-embedded targets.
    hard_esdf = np.where(stair_like, 0.02, 0.05)
    esdf_obstacle = (esdf_map < hard_esdf)

    # Combine slope and ESDF.
    obstacle = esdf_obstacle | slope_block

    inflation_radius_m = max(HALF_SAFETY_WIDTH, SAFETY_RADIUS)
    inflation_radius_px = max(1.0, inflation_radius_m / max(1e-6, resolution))
    dist_to_obstacle_px = distance_transform_edt(~obstacle)
    obstacle_inflated = dist_to_obstacle_px <= inflation_radius_px

    slope_penalty = np.clip(slope - 0.08, 0.0, 2.0)
    esdf_clamped = np.clip(esdf_map, 0.012, 2.0)
    clearance_cost = 1.0 / esdf_clamped
    unknown_penalty = np.where(finite, 0.0, 1.6)

    slope_w = np.where(stair_like, 0.08, 0.14)
    clear_w = np.where(stair_like, 0.05, 0.09)
    cost = 1.0 + slope_w * slope_penalty + clear_w * np.clip(clearance_cost, 0.0, 25.0) + unknown_penalty
    cost[obstacle_inflated] = np.inf
    return obstacle_inflated, cost, slope_feasible, slope_block


def smooth_path_world(path_world, window=5, passes=2):
    n = len(path_world)
    if n < 3:
        return path_world

    arr = np.array(path_world, dtype=np.float32)
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    half = window // 2

    smoothed = arr.copy()
    for _ in range(max(1, int(passes))):
        prev = smoothed.copy()
        for i in range(1, n - 1):
            l = max(0, i - half)
            r = min(n, i + half + 1)
            smoothed[i, :2] = np.mean(prev[l:r, :2], axis=0)
            smoothed[i, 2] = prev[i, 2]

        # keep endpoints fixed
        smoothed[0] = arr[0]
        smoothed[-1] = arr[-1]

    return [smoothed[i] for i in range(n)]


def build_global_path_corridor_mask(global_path_xy, origin, resolution, shape, radius_m):
    """Build a boolean corridor mask around global path polyline in local grid."""
    h, w = shape
    if global_path_xy is None or len(global_path_xy) < 1:
        return np.zeros((h, w), dtype=bool)

    radius_px = max(1, int(np.ceil(radius_m / max(1e-6, resolution))))
    seeds = np.zeros((h, w), dtype=bool)

    pts = np.asarray(global_path_xy, dtype=np.float32)
    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        seg_len = np.linalg.norm(p1 - p0)
        n = max(1, int(np.ceil(seg_len / max(1e-6, resolution * 0.5))))
        for t in np.linspace(0.0, 1.0, n + 1):
            p = (1.0 - t) * p0 + t * p1
            gx, gy = world_to_grid_2d(p, origin, resolution)
            if 0 <= gx < h and 0 <= gy < w:
                seeds[gx, gy] = True

    if len(pts) == 1:
        gx, gy = world_to_grid_2d(pts[0], origin, resolution)
        if 0 <= gx < h and 0 <= gy < w:
            seeds[gx, gy] = True

    dist = distance_transform_edt(~seeds)
    return dist <= radius_px


def pick_lookahead_point(path_world, robot_xy, lookahead_dist=0.6):
    if len(path_world) == 0:
        return None
    d_best = float('inf')
    i_best = 0
    for i, p in enumerate(path_world):
        d = np.linalg.norm(p[:2] - robot_xy)
        if d < d_best:
            d_best = d
            i_best = i
    for i in range(i_best, len(path_world)):
        if np.linalg.norm(path_world[i][:2] - robot_xy) >= lookahead_dist:
            return path_world[i]
    return path_world[-1]


def signed_angle_between(v_from, v_to):
    cross = v_from[0] * v_to[1] - v_from[1] * v_to[0]
    dot = v_from[0] * v_to[0] + v_from[1] * v_to[1]
    return np.arctan2(cross, dot)

# === PlanningNode class ===
class PlanningNode(Node):
    def __init__(self, sensor_source: str = 'auto'):
        super().__init__('planning_node')
        self.bridge = CvBridge()
        self.path_pub = self.create_publisher(Path, '/planning/trajectory_path', 10)
        self.height_map_pub = self.create_publisher(Image, "/planning/height_map", 10)
        self.traj_scores_pub = self.create_publisher(Image, "/planning/score_traj", 10)
        self.occupancy_cloud_pub = self.create_publisher(PointCloud2, '/planning/occupied_voxels', 10)
        self.occupancy_cloud_esdf_pub = self.create_publisher(PointCloud2, '/planning/occupied_voxels_with_esdf', 10)
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/planning/occupancy_grid', 10)
        self.slope_feasible_pub = self.create_publisher(OccupancyGrid, '/planning/slope_feasible', 10)
        self.slope_blocked_pub = self.create_publisher(OccupancyGrid, '/planning/slope_blocked', 10)
        self.global_corridor_pub = self.create_publisher(OccupancyGrid, '/planning/global_corridor', 10)
        self.footprint_corners_pub = self.create_publisher(PointCloud, '/planning/footprint_corners', 10)
        self.depth_sub = message_filters.Subscriber(self, Image, '/slam/depth')
        self.pose_sub = message_filters.Subscriber(self, Odometry, '/slam/odometry')
        self.planning_cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.ts = message_filters.TimeSynchronizer([self.depth_sub, self.pose_sub], queue_size=10)
        self.ts.registerCallback(self.sync_callback)

        if sensor_source == 'realsense':
            camera_info_topic = '/camera/camera/infra2/camera_info'
        elif sensor_source == 'insight':
            camera_info_topic = '/insight/camera_right_info'
        else:
            camera_info_topic = None

        start_wait = time.time()
        last_log = 0.0
        while True:
            active_topics = [t[0] for t in self.get_topic_names_and_types()]
            if camera_info_topic is None:
                if '/camera/camera/infra2/camera_info' in active_topics:
                    camera_info_topic = '/camera/camera/infra2/camera_info'
                elif '/insight/camera_right_info' in active_topics:
                    camera_info_topic = '/insight/camera_right_info'

            if camera_info_topic in active_topics:
                self.camera_info_sub = self.create_subscription(
                    CameraInfo, camera_info_topic, self.info_callback, 10
                )
                break

            now = time.time()
            if now - last_log > 2.0:
                self.get_logger().warning(
                    f"Waiting camera info topic for source={sensor_source}. Active topics count={len(active_topics)}"
                )
                last_log = now
            if now - start_wait > 60.0:
                raise RuntimeError(
                    f"Timeout waiting camera info topic for source={sensor_source}."
                )
            time.sleep(0.2)


        self.grid_shape = (100, 100, 15)
        self.resolution = 0.1
        self.origin = np.array(self.grid_shape) * self.resolution / -2.
        self.step = 5
        self.occupancy_grid = np.zeros(self.grid_shape)
        self.K = None
        self.baseline = None
        self.last_T = None
        self.last_param = (0.0, 0.0) # acc and gyro
        self.stamp = None
        self.current_pose = None  # Store the latest pose from odometry

        self.smoothed_velocity = 0.0

        self.create_subscription(Odometry, '/control/target_pose', self.target_pose_callback, 10)
        self.create_subscription(Path, '/mapping/global_plan', self.global_plan_callback, 10)
        self.target_pose = None
        self.global_path_xy = None

        # Keep planning event-driven (sync depth+odom), but publish cmd at a fixed rate.
        self.cmd_rate_hz = 20.0
        self.path_stale_slow_s = 0.3
        self.path_stale_stop_s = 0.6
        self.max_linear_speed = 0.5  # m/s
        self.max_linear_acc = 0.3   # m/s^2
        self.max_angular_acc = 1.0   # rad/s^2
        self.recovery_fast_speed = 0.18
        self.recovery_slow_speed = 0.08
        self.path_smooth_window = 5
        self.path_smooth_passes = 2

        # Local A* corridor around global path (meters). Fallback to unconstrained if failed.
        self.corridor_radius_m = 0.3
        self.corridor_fallback_scale = 1.6

        self.latest_cmd = Twist()
        self.prev_cmd = Twist()
        self.last_cmd_pub_time = time.monotonic()
        self.last_path_update_time = None

        self.cmd_timer = self.create_timer(1.0 / self.cmd_rate_hz, self.cmd_timer_callback)

        # Do not clear target_pose here: planner should keep chasing latest /control/target_pose.

    def target_pose_callback(self, msg):
        self.target_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def global_plan_callback(self, msg: Path):
        if len(msg.poses) == 0:
            self.global_path_xy = None
            return
        self.global_path_xy = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float32,
        )

    def info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            # P[0,3] = -fx * baseline
            fx = self.K[0, 0]
            Tx = msg.p[3] # From the right camera's projection matrix
            self.baseline = -Tx / fx
            self.get_logger().info(f"Camera intrinsics and baseline received. Baseline: {self.baseline:.4f}m")
            self.destroy_subscription(self.camera_info_sub)

    def publish_height_map_traj(self, pooled_map, trajectories, occ_points, top_indices, scores, params, origin, resolution):
        fig, ax = plt.subplots(figsize=(8, 6))
        height_normalized = (np.nan_to_num(pooled_map, nan=0.0) + 5) * 30
        height_uint8 = height_normalized.astype(np.uint8)
        ax.imshow(height_uint8, cmap='jet', vmin=0, vmax=255, origin='upper', interpolation='nearest')
        for idx in top_indices:
            if scores[idx] > -1:
                traj = trajectories[idx]
                occ_idx = occ_points[idx]
                x = (traj[:, 0] - origin[0]) / resolution
                y = (traj[:, 1] - origin[1]) / resolution
                ax.plot(y, x, label=f"score:{scores[idx]:.1f}, gyro:{params[idx][1]:.1f}", alpha=0.8)
                ax.plot(y[occ_idx], x[occ_idx], 'r*', markersize=8, label=None)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)


        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        img = np.array(PIL_Image.open(buf))[:, :, :3]  # Convert to RGB NumPy array
        buf.close()

        bridge = CvBridge()
        img_msg = bridge.cv2_to_imgmsg(img, encoding='rgb8')
        self.traj_scores_pub.publish(img_msg)

    def publish_height_map(self, origin, pooled_map, header):
        height_normalized = (np.nan_to_num(pooled_map, nan=0.0) + 5) * 30
        height_uint8 = height_normalized.astype(np.uint8)
        color_image = cv2.applyColorMap(height_uint8, cv2.COLORMAP_JET)
        img_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        img_msg.header = header
        self.height_map_pub.publish(img_msg)

    def publish_2d_occupancy_grid(self, ESDF_map, origin, resolution, stamp, z_offset=0.0):
        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header = Header()
        occupancy_grid_msg.header.stamp = stamp
        occupancy_grid_msg.header.frame_id = "world"
        occupancy_grid_msg.info.resolution = resolution
        occupancy_grid_msg.info.width = ESDF_map.shape[1]
        occupancy_grid_msg.info.height = ESDF_map.shape[0]
        occupancy_grid_msg.info.origin.position.x = origin[0]
        occupancy_grid_msg.info.origin.position.y = origin[1]
        occupancy_grid_msg.info.origin.position.z = origin[2] + z_offset
        occupancy_grid_msg.info.origin.orientation.w = 1.0
        flat_data = np.where(ESDF_map <= 0.00, 100, np.clip(((1-ESDF_map/0.5) * 120).astype(int), 0, 120)).ravel(order="F").tolist()
        occupancy_grid_msg.data = flat_data
        self.occupancy_grid_pub.publish(occupancy_grid_msg)

    def publish_binary_grid(self, mask, pub, origin, resolution, stamp, z_offset=0.0):
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.info.resolution = resolution
        msg.info.width = mask.shape[1]
        msg.info.height = mask.shape[0]
        msg.info.origin.position.x = origin[0]
        msg.info.origin.position.y = origin[1]
        msg.info.origin.position.z = origin[2] + z_offset
        msg.info.origin.orientation.w = 1.0
        msg.data = np.where(mask, 100, 0).astype(np.int8).ravel(order="F").tolist()
        pub.publish(msg)

    def publish_3d_occupancy_cloud(self, grid3d, resolution=0.1, origin=(0, 0, 0)):
        occupied = np.argwhere(grid3d > 0.1)
        # vectorized operation to avoid for loop
        if len(occupied) == 0:
            points = []
        else:
            origin_np = np.array(origin)
            world_coords = origin_np + occupied * resolution
            points = world_coords.tolist()

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "world"
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.occupancy_cloud_pub.publish(pc2_msg)

    def _camera_to_base_xy(self, camera_xy, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        forward_xy = np.array([c, s], dtype=np.float32)
        return np.asarray(camera_xy, dtype=np.float32) - forward_xy * CAMERA_FORWARD_OFFSET

    def _footprint_corners_xy_from_camera_pose(self, camera_xy, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        forward_xy = np.array([c, s], dtype=np.float32)
        left_xy = np.array([-s, c], dtype=np.float32)
        base_xy = self._camera_to_base_xy(camera_xy, yaw)
        return [
            base_xy + forward_xy * HALF_SAFETY_LENGTH + left_xy * HALF_SAFETY_WIDTH,
            base_xy + forward_xy * HALF_SAFETY_LENGTH - left_xy * HALF_SAFETY_WIDTH,
            base_xy - forward_xy * HALF_SAFETY_LENGTH - left_xy * HALF_SAFETY_WIDTH,
            base_xy - forward_xy * HALF_SAFETY_LENGTH + left_xy * HALF_SAFETY_WIDTH,
        ]

    def _footprint_corners_collide(self, grid_xy, yaw, obstacle_mask, origin, resolution):
        world_xy = grid_to_world_2d(grid_xy, origin, resolution, 0.0)[:2]
        corners_xy = self._footprint_corners_xy_from_camera_pose(world_xy, yaw)
        h, w = obstacle_mask.shape
        for p in corners_xy:
            gx = int((p[0] - origin[0]) / resolution)
            gy = int((p[1] - origin[1]) / resolution)
            if gx < 0 or gx >= h or gy < 0 or gy >= w:
                return True
            if obstacle_mask[gx, gy]:
                return True
        return False

    def publish_footprint_corners(self, T, stamp):
        """Publish 4 footprint corners for RViz debugging."""
        forward_world = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        yaw = float(np.arctan2(forward_world[1], forward_world[0]))
        camera_xy = T[:2, 3]
        z = float(T[2, 3])
        corners_xy = self._footprint_corners_xy_from_camera_pose(camera_xy, yaw)

        msg = PointCloud()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.points = [Point32(x=float(p[0]), y=float(p[1]), z=z) for p in corners_xy]
        self.footprint_corners_pub.publish(msg)

    def publish_3d_occupancy_cloud_with_esdf(self, grid3d, ESDF_map, resolution=0.1, origin=(0, 0, 0), max_dist=1.0):
        X, Y, Z = grid3d.shape
        # ground
        gx, gy = np.meshgrid(np.arange(X), np.arange(Y), indexing='ij')
        ground = np.stack([gx.ravel(), gy.ravel(), np.zeros_like(gx).ravel()+2], axis=-1)
        coords = ground * resolution + np.asarray(origin)
        # query ESDF
        ix, iy = ground[:, 0].astype(int), ground[:, 1].astype(int)
        valid = (0 <= ix) & (ix < ESDF_map.shape[0]) & (0 <= iy) & (iy < ESDF_map.shape[1])
        dist = np.full(len(ground), max_dist, dtype=np.float32)
        dist[valid] = np.clip(ESDF_map[ix[valid], iy[valid]], 0, max_dist)
        # map color
        v = np.uint8((1 - dist / max_dist) * 255)
        colors = cv2.applyColorMap(v.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
        rgb = (colors[:, 2].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8) | colors[:, 0].astype(np.uint32)
        # build point cloud
        dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)])
        points = np.zeros(coords.shape[0], dtype=dtype)
        points['x'], points['y'], points['z'] = coords[:, 0], coords[:, 1], coords[:, 2]
        points['rgb'] = rgb
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id="world")
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        self.occupancy_cloud_esdf_pub.publish(pc2.create_cloud(header, fields, points))

    def _clamp_step(self, target: float, current: float, max_delta: float) -> float:
        return float(np.clip(target - current, -max_delta, max_delta) + current)

    def cmd_timer_callback(self):
        now = time.monotonic()
        dt = max(1e-3, now - self.last_cmd_pub_time)
        self.last_cmd_pub_time = now

        # Stale-path protection: slow down, then stop if planner has not refreshed.
        age = float('inf') if self.last_path_update_time is None else (now - self.last_path_update_time)
        target_cmd = Twist()
        target_cmd.linear.x = self.latest_cmd.linear.x
        target_cmd.angular.z = self.latest_cmd.angular.z
        if age > self.path_stale_stop_s:
            target_cmd.linear.x = 0.0
            target_cmd.angular.z = 0.0
        elif age > self.path_stale_slow_s:
            target_cmd.linear.x *= 0.3
            target_cmd.angular.z *= 0.5

        # Acceleration limiting for smoother control.
        max_dv = self.max_linear_acc * dt
        max_dw = self.max_angular_acc * dt
        out = Twist()
        out.linear.x = self._clamp_step(target_cmd.linear.x, self.prev_cmd.linear.x, max_dv)
        out.angular.z = self._clamp_step(target_cmd.angular.z, self.prev_cmd.angular.z, max_dw)
        out.linear.y = 0.0

        self.planning_cmd_pub.publish(out)
        self.prev_cmd = out

    @Timer(name="Planning Loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms")
    def sync_callback(self, depth_msg, odom_msg):
        if self.K is None:
            return
        with Timer(name='preprocess', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            stamp = Time.from_msg(odom_msg.header.stamp).nanoseconds / 1e9
            T, velocity = msg2np(odom_msg)
            velocity_in_camera = T[:3, :3].T @ velocity
            sign = np.sign(velocity_in_camera[2])
            alpha = 0.9
            if self.last_T is None:
                self.last_T = T.copy()
                self.smoothed_velocity = sign * np.linalg.norm(velocity_in_camera)
            self.smoothed_velocity = alpha * self.smoothed_velocity + (1 - alpha) * sign * np.linalg.norm(velocity_in_camera)

            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]

        # 统一参考：XY 使用 base；Z 仍使用 camera
        forward_world = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        yaw = float(np.arctan2(forward_world[1], forward_world[0]))
        camera_xy = T[:2, 3]
        base_xy = self._camera_to_base_xy(camera_xy, yaw)

        with Timer(name='raycasting', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            center = self.origin + np.array(self.grid_shape) * self.resolution / 2
            robot_pos = np.array([base_xy[0], base_xy[1], T[2, 3]], dtype=np.float32)
            delta = robot_pos - center
            if np.linalg.norm(delta) > .1:
                new_center = robot_pos
                new_origin = new_center - np.array(self.grid_shape) * self.resolution / 2
                self.occupancy_grid, self.origin = roll_occupancy_grid(self.occupancy_grid, self.origin, new_origin, self.resolution)
            new_occ = run_raycasting_loopy(depth, T, self.grid_shape, fx, fy, cx, cy, self.origin, self.step, self.resolution)

            # seconds = log(0.5) / log(0.998) = 347.22 timestamp / 10 hz = around 35 seconds
            self.occupancy_grid *= 0.998
            self.occupancy_grid += new_occ
            self.occupancy_grid = np.clip(self.occupancy_grid, -0.2, 0.2)

            self.publish_3d_occupancy_cloud(self.occupancy_grid, self.resolution, self.origin)

        with Timer(name='heightmap', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            height_map = occupancy_grid_to_height_map(self.occupancy_grid, self.origin, self.resolution)
            pooled_map = max_pool_height_map(height_map)
            robot_z = T[2, 3]
            # Robot-relative height map for threshold-based ESDF decisions.
            pooled_map_rel = pooled_map - robot_z
            ESDF_map = height_map_to_ESDF(pooled_map_rel, 0.4, self.resolution)


        with Timer(name='vis heighmap and esdf', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            self.publish_3d_occupancy_cloud_with_esdf(self.occupancy_grid, ESDF_map, self.resolution, self.origin)
            self.publish_height_map(T[:3,3], ESDF_map, depth_msg.header)
            self.publish_2d_occupancy_grid(ESDF_map, self.origin, self.resolution, depth_msg.header.stamp, z_offset=self.grid_shape[2]*self.resolution/2)
            self.publish_footprint_corners(T, depth_msg.header.stamp)

        with Timer(name='local astar plan', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            local_path_world = []
            if self.target_pose is not None:
                obstacle_mask, astar_cost, slope_feasible, slope_block = build_astar_cost_map(pooled_map_rel, ESDF_map, self.resolution)
                self.publish_binary_grid(slope_feasible, self.slope_feasible_pub, self.origin, self.resolution, depth_msg.header.stamp, z_offset=self.grid_shape[2]*self.resolution/2)
                self.publish_binary_grid(slope_block, self.slope_blocked_pub, self.origin, self.resolution, depth_msg.header.stamp, z_offset=self.grid_shape[2]*self.resolution/2)

                # 使用前面统一计算出的 yaw/base_xy

                start_idx = clip_grid_2d(world_to_grid_2d(base_xy, self.origin, self.resolution), obstacle_mask.shape)
                goal_idx = clip_grid_2d(world_to_grid_2d(self.target_pose[:2], self.origin, self.resolution), obstacle_mask.shape)

                start_free = find_nearest_free(start_idx, obstacle_mask)
                goal_free = find_nearest_free(goal_idx, obstacle_mask)

                def footprint_check(grid_xy):
                    return self._footprint_corners_collide(
                        grid_xy, yaw, obstacle_mask, self.origin, self.resolution
                    )

                grid_path = []
                if self.global_path_xy is not None and len(self.global_path_xy) >= 2:
                    corridor_mask = build_global_path_corridor_mask(
                        self.global_path_xy,
                        self.origin,
                        self.resolution,
                        obstacle_mask.shape,
                        self.corridor_radius_m,
                    )
                    corridor_mask &= ~obstacle_mask
                    self.publish_binary_grid(
                        corridor_mask,
                        self.global_corridor_pub,
                        self.origin,
                        self.resolution,
                        depth_msg.header.stamp,
                        z_offset=self.grid_shape[2] * self.resolution / 2,
                    )

                    # # Pass 1: strict corridor-constrained A*.
                    # grid_path = astar_2d(
                    #     astar_cost,
                    #     obstacle_mask,
                    #     start_free,
                    #     goal_free,
                    #     search_mask=corridor_mask,
                    #     footprint_check_fn=footprint_check,
                    # )

                    # # Pass 2: relaxed corridor if strict one fails.
                    # if len(grid_path) == 0:
                    #     relaxed_mask = build_global_path_corridor_mask(
                    #         self.global_path_xy,
                    #         self.origin,
                    #         self.resolution,
                    #         obstacle_mask.shape,
                    #         self.corridor_radius_m * self.corridor_fallback_scale,
                    #     )
                    #     relaxed_mask &= ~obstacle_mask
                    #     grid_path = astar_2d(
                    #         astar_cost,
                    #         obstacle_mask,
                    #         start_free,
                    #         goal_free,
                    #         search_mask=relaxed_mask,
                    #         footprint_check_fn=footprint_check,
                    #     )

                # Pass 3 fallback: original unconstrained local A*.
                if len(grid_path) == 0:
                    grid_path = astar_2d(
                        astar_cost,
                        obstacle_mask,
                        start_free,
                        goal_free,
                        footprint_check_fn=footprint_check,
                    )

                if (len(grid_path) == 0):
                    self.get_logger().warning("No local path with footprint check found")
                    return
                
                self.get_logger().info(f"Path found: {len(grid_path)} points")

                robot_z = T[2, 3]
                local_path_world = []
                for g in grid_path:
                    p = grid_to_world_2d(g, self.origin, self.resolution, robot_z)
                    # Keep path Z fixed at current robot height for flatter trajectories.
                    p[2] = robot_z
                    local_path_world.append(p)

                if len(local_path_world) >= 3:
                    local_path_world = smooth_path_world(
                        local_path_world,
                        window=self.path_smooth_window,
                        passes=self.path_smooth_passes,
                    )

        with Timer(name='pub', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            path_msg = Path()
            path_msg.header = depth_msg.header
            path_msg.header.frame_id = "world"

            for p in local_path_world[::2]:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = float(p[0])
                pose.pose.position.y = float(p[1])
                pose.pose.position.z = float(p[2])
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self.path_pub.publish(path_msg)

            cmd = Twist()
            if self.target_pose is None:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            else:
                robot_xy = base_xy
                target_dist = np.linalg.norm(self.target_pose[:2] - robot_xy)
                if target_dist < 0.25:
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                elif len(local_path_world) < 2:
                    # Recovery behavior: cautiously probe toward target instead of pure spinning.
                    forward_world = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
                    forward_xy = forward_world[:2]
                    to_target = self.target_pose[:2] - robot_xy
                    norm_f = np.linalg.norm(forward_xy)
                    norm_t = np.linalg.norm(to_target)
                    if norm_f < 1e-6 or norm_t < 1e-6:
                        cmd.linear.x = 0.0
                        cmd.angular.z = 0.5
                    else:
                        forward_xy = forward_xy / norm_f
                        to_target = to_target / norm_t
                        heading_err = signed_angle_between(forward_xy, to_target)
                        cmd.angular.z = float(np.clip(1.6 * heading_err, -1.0, 1.0))
                        cmd.linear.x = self.recovery_fast_speed if abs(heading_err) < 0.6 else self.recovery_slow_speed
                else:
                    lookahead = pick_lookahead_point(local_path_world, robot_xy, lookahead_dist=0.6)
                    forward_world = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
                    forward_xy = forward_world[:2]
                    norm_f = np.linalg.norm(forward_xy)
                    to_wp = lookahead[:2] - robot_xy
                    norm_t = np.linalg.norm(to_wp)

                    if norm_f < 1e-6 or norm_t < 1e-6:
                        cmd.linear.x = 0.0
                        cmd.angular.z = 0.0
                    else:
                        forward_xy = forward_xy / norm_f
                        to_wp = to_wp / norm_t
                        heading_err = signed_angle_between(forward_xy, to_wp)

                        cmd.angular.z = float(np.clip(1.8 * heading_err, -1.2, 1.2))
                        heading_scale = max(0.0, np.cos(heading_err))
                        dist_scale = np.clip(target_dist / 1.0, 0.2, 1.0)
                        cmd.linear.x = float(np.clip(self.max_linear_speed * heading_scale * dist_scale, 0.0, self.max_linear_speed))
                        if abs(heading_err) > 1.0:
                            cmd.linear.x *= 0.15

            cmd.linear.y = 0.0
            self.latest_cmd = cmd
            self.last_path_update_time = time.monotonic()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sensor_source',
        type=str,
        choices=['auto', 'realsense', 'insight'],
        default='auto',
        help='Sensor topic source selection',
    )
    parsed_args, unknown_args = parser.parse_known_args(args)

    rclpy.init(args=unknown_args)
    node = PlanningNode(sensor_source=parsed_args.sensor_source)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

