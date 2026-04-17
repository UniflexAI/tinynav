import argparse
import time
import os
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point32
from cv_bridge import CvBridge
import numpy as np
from scipy.ndimage import maximum_filter, binary_dilation
from numba import njit
import message_filters
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointCloud
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from std_msgs.msg import Float64
from codetiming import Timer
import cv2
import heapq
from dataclasses import dataclass
from math_utils import quat_to_matrix, msg2np
from geometry_msgs.msg import Twist


@dataclass
class RobotConfig:
    """Robot geometry from YAML. Body frame: +x forward, +y left, (0,0) = shape center."""
    name: str = 'go2'
    shape: str = 'square'
    length: float = 0.7
    width: float = 0.3
    radius: float = 0.3
    camera_x: float = 0.35
    camera_y: float = 0.0
    control_x: float = 0.0
    control_y: float = 0.0
    safety_radius: float = 0.3

    @property
    def cam_offset_3d(self):
        """[left, up, forward] from control center to camera in Tinynav 3D body frame."""
        return np.array([self.camera_y - self.control_y, 0.0, self.camera_x - self.control_x], dtype=np.float32)

    @property
    def half_size(self):
        if self.shape == 'circle':
            return (self.radius, self.radius)
        return (self.length / 2.0, self.width / 2.0)

    def footprint_from_control(self):
        """front_len, rear_len, half_w relative to control center."""
        hl, hw = self.half_size
        return float(hl + self.control_x), float(hl - self.control_x), float(hw)


def load_robot_config(path: str) -> RobotConfig:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    r = raw.get('robot', {})
    return RobotConfig(
        name=r.get('name', 'go2'), shape=r.get('shape', 'square'),
        length=float(r.get('length', 0.7)), width=float(r.get('width', 0.3)),
        radius=float(r.get('radius', 0.3)),
        camera_x=float(r.get('camera', {}).get('x', 0.35)),
        camera_y=float(r.get('camera', {}).get('y', 0.0)),
        control_x=float(r.get('control_center', {}).get('x', 0.0)),
        control_y=float(r.get('control_center', {}).get('y', 0.0)),
        safety_radius=float(r.get('safety_radius', 0.3)),
    )


DEFAULT_ROBOT_CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'robot_b2.yaml')


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

@dataclass
class ObstacleConfig:
    robot_z_bottom: float = -0.3
    robot_z_top: float = 0.8
    occ_threshold: float = 0.1
    min_wall_span_m: float = 0.6
    dilation_cells: int = 2


def build_obstacle_map(occupancy_grid, origin, resolution, robot_z, config=None):
    """Obstacle = cells where occupied voxels span >= min_wall_span_m in z.
    Walls have large z-span (tall vertical surface); stair risers have small span."""
    config = config or ObstacleConfig()
    h, w, z_dim = occupancy_grid.shape
    z_world = origin[2] + (np.arange(z_dim) + 0.5) * resolution
    z_rel = z_world - robot_z
    z_mask = (z_rel >= config.robot_z_bottom) & (z_rel <= config.robot_z_top)

    obstacle = np.zeros((h, w), dtype=bool)
    if np.any(z_mask):
        band_occ = occupancy_grid[:, :, z_mask] > config.occ_threshold
        has_occ = np.any(band_occ, axis=2)
        n_z = band_occ.shape[2]
        z_idx = np.arange(n_z, dtype=np.float32)
        occ_high = np.where(band_occ, z_idx[np.newaxis, np.newaxis, :], -1).max(axis=2)
        occ_low = np.where(band_occ, z_idx[np.newaxis, np.newaxis, :], n_z).min(axis=2)
        z_span = (occ_high - occ_low) * resolution
        obstacle = has_occ & (z_span >= config.min_wall_span_m)

    if config.dilation_cells > 0 and np.any(obstacle):
        obstacle = binary_dilation(obstacle, iterations=config.dilation_cells)
    return obstacle


def build_static_wall_mask(occupancy_grid, resolution, occ_threshold=0.1, min_wall_span_m=0.7):
    """Robot-z independent wall mask from full-z occupancy span."""
    h, w, z_dim = occupancy_grid.shape
    band_occ = occupancy_grid > occ_threshold
    has_occ = np.any(band_occ, axis=2)
    z_idx = np.arange(z_dim, dtype=np.float32)
    occ_high = np.where(band_occ, z_idx[np.newaxis, np.newaxis, :], -1).max(axis=2)
    occ_low = np.where(band_occ, z_idx[np.newaxis, np.newaxis, :], z_dim).min(axis=2)
    z_span = (occ_high - occ_low) * resolution
    wall_mask = has_occ & (z_span >= float(min_wall_span_m))
    if np.any(wall_mask):
        wall_mask = binary_dilation(wall_mask, iterations=1)
    return wall_mask

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


def clip_grid_2d(grid_xy, shape):
    x = min(max(grid_xy[0], 0), shape[0] - 1)
    y = min(max(grid_xy[1], 0), shape[1] - 1)
    return (x, y)


def find_nearest_free(start, obstacle_mask, search_radius=8):
    sx, sy = start
    h, w = obstacle_mask.shape
    if 0 <= sx < h and 0 <= sy < w and not obstacle_mask[sx, sy]:
        return (sx, sy)
    for r in range(1, search_radius + 1):
        x0, x1 = max(0, sx - r), min(h - 1, sx + r)
        y0, y1 = max(0, sy - r), min(w - 1, sy + r)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                if not obstacle_mask[x, y]:
                    return (x, y)
    return None


def snap_target_to_free(target_xy, robot_xy, forward_xy, obstacle_mask, origin, resolution,
                        search_radius=10, heading_weight=0.3, global_path_xy=None):
    """Snap target to free cell: global-path first, then near-path, then near target; None if failed."""
    h, w = obstacle_mask.shape
    tg = world_to_grid_2d(target_xy, origin, resolution)
    tx, ty = tg
    if 0 <= tx < h and 0 <= ty < w and not obstacle_mask[tx, ty]:
        return target_xy

    fwd = np.asarray(forward_xy, dtype=np.float32)
    nf = np.linalg.norm(fwd)
    if nf > 1e-6:
        fwd = fwd / nf

    def _cell_world(x, y):
        return np.array([
            origin[0] + (x + 0.5) * resolution,
            origin[1] + (y + 0.5) * resolution,
        ], dtype=np.float32)

    def _score_cell(x, y, ref_x, ref_y):
        dist = np.hypot(x - ref_x, y - ref_y)
        cell_world = _cell_world(x, y)
        to_cell = cell_world - robot_xy
        nt = np.linalg.norm(to_cell)
        heading_penalty = 0.0
        if nt > 1e-6 and nf > 1e-6:
            cos_angle = np.dot(fwd, to_cell / nt)
            heading_penalty = 1.0 - max(0.0, cos_angle)
        return dist + heading_weight * search_radius * heading_penalty, cell_world

    # 1) Prefer a free point directly on global path (toward goal end first).
    if global_path_xy is not None and len(global_path_xy) > 0:
        for pt in global_path_xy[::-1]:
            gx, gy = world_to_grid_2d(pt, origin, resolution)
            if 0 <= gx < h and 0 <= gy < w and not obstacle_mask[gx, gy]:
                return _cell_world(gx, gy)

        # 2) If none on path, search around path cells.
        path_search_radius = max(2, search_radius // 2)
        best_path = None
        best_path_score = float('inf')
        for r in range(1, path_search_radius + 1):
            found_this_r = False
            for pt in global_path_xy[::-1]:
                gx, gy = world_to_grid_2d(pt, origin, resolution)
                if gx < 0 or gx >= h or gy < 0 or gy >= w:
                    continue
                x0, x1 = max(0, gx - r), min(h - 1, gx + r)
                y0, y1 = max(0, gy - r), min(w - 1, gy + r)
                for x in range(x0, x1 + 1):
                    for y in range(y0, y1 + 1):
                        if obstacle_mask[x, y]:
                            continue
                        score, cell_world = _score_cell(x, y, gx, gy)
                        if score < best_path_score:
                            best_path_score = score
                            best_path = cell_world
                            found_this_r = True
            if found_this_r:
                return best_path

    # 3) Fallback: search around original target.
    best = None
    best_score = float('inf')
    for r in range(1, search_radius + 1):
        x0, x1 = max(0, tx - r), min(h - 1, tx + r)
        y0, y1 = max(0, ty - r), min(w - 1, ty + r)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                if obstacle_mask[x, y]:
                    continue
                score, cell_world = _score_cell(x, y, tx, ty)
                if score < best_score:
                    best_score = score
                    best = cell_world
        if best is not None:
            break
    return best


def astar_2d(cost_map, obstacle_mask, start, goal):
    if start is None or goal is None:
        return []
    if obstacle_mask[start] or obstacle_mask[goal]:
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
            n = (nx, ny)
            trans_cost = step_cost * (0.5 * (cost_map[cx, cy] + cost_map[nx, ny]))
            new_g = g_cur + trans_cost
            if new_g < g_score.get(n, float('inf')):
                g_score[n] = new_g
                parent[n] = cur
                f = new_g + heuristic(n, goal)
                heapq.heappush(open_heap, (f, new_g, n))
    return []


def build_astar_cost_map(unknown_mask):
    unknown_penalty = np.where(unknown_mask, 3.0, 0.0)
    return 1.0 + unknown_penalty


def _dilate_one_numpy(mask):
    """4-connected single-step dilation using numpy shifts — no scipy, no OpenMP."""
    r = mask.copy()
    r[1:,  :] |= mask[:-1, :]
    r[:-1, :] |= mask[1:,  :]
    r[:,  1:] |= mask[:, :-1]
    r[:, :-1] |= mask[:,  1:]
    return r


def build_proximity_cost(obstacle_mask, max_cells=2, weight=2.0):
    """
    Obstacle proximity penalty for DWA/A* cost map.

    Adds a cost that linearly decays from `weight` at the obstacle boundary
    to 0 at `max_cells` distance. Uses pure numpy shifts (no scipy, no OpenMP
    threads) so SIGINT / Ctrl+C is never blocked.

    Keeps radius small (default 2 cells = 0.2 m beyond the already-dilated
    obstacle mask) so it doesn't squeeze the robot in narrow passages.
    Blind-spot areas have no detected obstacles → no repulsion added there.
    """
    cost = np.zeros(obstacle_mask.shape, dtype=np.float32)
    cost[obstacle_mask] = weight
    frontier = obstacle_mask.copy()
    for ring in range(1, max_cells + 1):
        expanded = _dilate_one_numpy(frontier)
        new_cells = expanded & ~frontier
        cost[new_cells] = weight * (max_cells - ring) / max_cells
        frontier = expanded
    return cost


def apply_global_path_bonus(cost_map, global_path_xy, origin, resolution,
                             bonus=4.0, dilation_cells=1):
    """
    Reduce cost along the global path cells so A*/DWA naturally follows the
    pre-planned route instead of hugging known-but-tight walls.

    Args:
        cost_map:       float32 2D cost array (modified in-place copy returned)
        global_path_xy: (N,2) world-frame XY waypoints from /mapping/global_plan
        origin:         grid origin (world coords)
        resolution:     metres per cell
        bonus:          cost reduction applied to path cells (and neighbours)
        dilation_cells: number of cells around each path point to apply bonus
    Returns:
        cost_map with bonus applied (new array, original unchanged)
    """
    if global_path_xy is None or len(global_path_xy) == 0:
        return cost_map

    h, w = cost_map.shape
    bonus_map = np.zeros((h, w), dtype=np.float32)

    for pt in global_path_xy:
        gx = int((pt[0] - origin[0]) / resolution)
        gy = int((pt[1] - origin[1]) / resolution)
        for dx in range(-dilation_cells, dilation_cells + 1):
            for dy in range(-dilation_cells, dilation_cells + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < h and 0 <= ny < w:
                    bonus_map[nx, ny] = bonus

    return np.maximum(1.0, cost_map - bonus_map)


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


def footprint_collision(center_xy, forward_xy, obstacle_mask, origin, resolution,
                        front_len=0.35, rear_len=0.35, half_w=0.15):
    h, w = obstacle_mask.shape
    n = np.linalg.norm(forward_xy)
    if n < 1e-6:
        fwd = np.array([1.0, 0.0], dtype=np.float32)
    else:
        fwd = forward_xy / n
    left = np.array([-fwd[1], fwd[0]], dtype=np.float32)
    c = np.asarray(center_xy, dtype=np.float32)
    for fl, ll in [(front_len, half_w), (front_len, -half_w),
                   (-rear_len, -half_w), (-rear_len, half_w)]:
        corner = c + fwd * fl + left * ll
        gx, gy = world_to_grid_2d(corner, origin, resolution)
        if gx < 0 or gx >= h or gy < 0 or gy >= w:
            return True
        if obstacle_mask[gx, gy]:
            return True
    return False


def plan_dwa_minimal(robot_xy, robot_forward_xy, target_xy, obstacle_mask, cost_map, origin, resolution,
                     horizon_s=1.2, dt=0.2, v_samples=(), w_samples=(),
                     front_len=0.35, rear_len=0.35, half_w=0.15):
    if len(v_samples) == 0 or len(w_samples) == 0:
        raise ValueError("plan_dwa_minimal requires non-empty v_samples and w_samples")

    rf = np.asarray(robot_forward_xy, dtype=np.float32)
    nrf = np.linalg.norm(rf)
    if nrf < 1e-6:
        rf = np.array([1.0, 0.0], dtype=np.float32)
    else:
        rf = rf / nrf
    theta0 = float(np.arctan2(rf[1], rf[0]))

    best_score = float('inf')
    best_path = []
    steps = max(1, int(np.ceil(horizon_s / dt)))

    for v in v_samples:
        for w in w_samples:
            x, y, theta = float(robot_xy[0]), float(robot_xy[1]), theta0
            traj = []
            score = 0.0
            collided = False
            for _ in range(steps):
                theta += float(w) * dt
                x += float(v) * np.cos(theta) * dt
                y += float(v) * np.sin(theta) * dt
                traj.append(np.array([x, y], dtype=np.float32))

                gx, gy = world_to_grid_2d((x, y), origin, resolution)
                if gx < 0 or gy < 0 or gx >= obstacle_mask.shape[0] or gy >= obstacle_mask.shape[1]:
                    collided = True
                    score += 1e6
                    break
                if obstacle_mask[gx, gy]:
                    collided = True
                    score += 1e6
                    break
                fwd = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                if footprint_collision(
                    np.array([x, y], dtype=np.float32), fwd,
                    obstacle_mask, origin, resolution,
                    front_len, rear_len, half_w,
                ):
                    collided = True
                    score += 1e6
                    break

                score += float(cost_map[gx, gy]) * dt
                score += 0.5 * np.linalg.norm(np.array([x, y], dtype=np.float32) - target_xy) * dt
                score += 0.12 * abs(float(w)) * dt

            if not collided and len(traj) > 0:
                score += 2.0 * np.linalg.norm(traj[-1] - target_xy)

            if score < best_score:
                best_score = score
                best_path = traj

    return best_path


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


def min_clearance_in_direction(robot_xy, dir_xy, obstacle_mask, origin, resolution,
                               max_dist=0.6, step=0.05, lateral_half=0.15):
    d = np.asarray(dir_xy, dtype=np.float32)
    nd = np.linalg.norm(d)
    if nd < 1e-6:
        return 0.0
    d = d / nd
    left = np.array([-d[1], d[0]], dtype=np.float32)
    h, w = obstacle_mask.shape
    n_long = max(1, int(np.ceil(max_dist / step)))
    n_lat = max(0, int(np.ceil(lateral_half / step)))
    for i in range(1, n_long + 1):
        s = i * step
        for j in range(-n_lat, n_lat + 1):
            off = j * step
            p = robot_xy + d * s + left * off
            gx, gy = world_to_grid_2d(p, origin, resolution)
            if gx < 0 or gx >= h or gy < 0 or gy >= w:
                return 0.0
            if obstacle_mask[gx, gy]:
                return max(0.0, s - step)
    return float(max_dist)


def apply_shared_safety_gate(cmd, robot_xy, forward_xy, obstacle_mask, origin, resolution):
    out = Twist()
    out.linear.x = cmd.linear.x
    out.angular.z = cmd.angular.z

    front_clear = min_clearance_in_direction(robot_xy, forward_xy, obstacle_mask, origin, resolution,
                                             max_dist=0.55, step=0.05, lateral_half=0.18)
    back_clear = min_clearance_in_direction(robot_xy, -np.asarray(forward_xy, dtype=np.float32), obstacle_mask, origin, resolution,
                                            max_dist=0.35, step=0.05, lateral_half=0.18)

    front_danger = front_clear < 0.18
    back_safe = back_clear > 0.16

    if out.linear.x >= 0.0 and front_danger:
        if back_safe:
            out.linear.x = -0.08
            if abs(out.angular.z) < 0.4:
                out.angular.z = 0.4
        else:
            out.linear.x = 0.0
            if abs(out.angular.z) < 0.5:
                out.angular.z = 0.5

    if out.linear.x < 0.0 and not back_safe:
        out.linear.x = 0.0

    out.linear.y = 0.0
    return out

# === PlanningNode class ===
class PlanningNode(Node):
    def __init__(self, sensor_source: str = 'auto', robot_config_path: str = None):
        super().__init__('planning_node')
        cfg_path = robot_config_path or DEFAULT_ROBOT_CONFIG
        self.robot = load_robot_config(cfg_path)
        self.get_logger().info(
            f"Robot: {self.robot.name} ({self.robot.shape} {self.robot.length}x{self.robot.width}m, "
            f"cam=({self.robot.camera_x},{self.robot.camera_y}), "
            f"ctrl=({self.robot.control_x},{self.robot.control_y}), "
            f"safety_r={self.robot.safety_radius}m)"
        )
        self.bridge = CvBridge()
        self.path_pub = self.create_publisher(Path, '/planning/trajectory_path', 10)
        self.safety_path_pub = self.create_publisher(Path, '/planning/safety_gate_path', 10)
        self.height_map_pub = self.create_publisher(Image, "/planning/height_map", 10)
        self.occupancy_cloud_pub = self.create_publisher(PointCloud2, '/planning/occupied_voxels', 10)
        self.obstacle_mask_pub = self.create_publisher(OccupancyGrid, '/planning/obstacle_mask', 10)
        self.astar_cost_pub = self.create_publisher(Image, '/planning/astar_cost', 10)
        self.footprint_pub = self.create_publisher(PointCloud, '/planning/footprint', 10)
        self.raycast_time_pub = self.create_publisher(
            Float64, '/planning/raycast_time_sec', 10
        )
        self.raycast_roll_time_pub = self.create_publisher(
            Float64, '/planning/raycast_roll_time_sec', 10
        )
        self.raycast_kernel_time_pub = self.create_publisher(
            Float64, '/planning/raycast_kernel_time_sec', 10
        )
        self.raycast_fuse_time_pub = self.create_publisher(
            Float64, '/planning/raycast_fuse_time_sec', 10
        )
        self.raycast_cloud_pub_time_pub = self.create_publisher(
            Float64, '/planning/raycast_cloud_pub_time_sec', 10
        )
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


        self.grid_shape = (100, 100, 40)
        self.resolution = 0.1
        self.origin = np.array(self.grid_shape) * self.resolution / -2.
        self.step = 5
        self.occupancy_grid = np.zeros(self.grid_shape)
        self.obstacle_config = ObstacleConfig()
        self.K = None
        self.baseline = None
        self.last_T = None
        self.last_param = (0.0, 0.0) # acc and gyro
        self.stamp = None
        self.current_pose = None  # Store the latest pose from odometry

        self.smoothed_velocity = 0.0

        self.create_subscription(Odometry, '/control/target_pose', self.target_pose_callback, 10)
        self.create_subscription(Path, '/mapping/global_plan', self.global_plan_callback, 10)
        # Backward compatibility: map_node stop currently reuses /mapping/poi_change.
        self.create_subscription(Odometry, '/mapping/poi_change', self.poi_change_callback, 10)
        self.target_pose = None
        self.global_path_xy = None

        # Keep planning event-driven (sync depth+odom), but publish cmd at a fixed rate.
        self.cmd_rate_hz = 30.0
        self.path_stale_slow_s = 0.3
        self.path_stale_stop_s = 0.6
        self.max_linear_speed = 0.8  # m/s
        self.max_reverse_speed = 0.1  # m/s
        self.max_angular_speed = 0.5  # rad/s
        self.max_linear_acc = 2.0   # m/s^2
        self.max_angular_acc = 2.5   # rad/s^2
        # Keep DWA sampling consistent with cmd speed limits.
        self.dwa_v_samples = np.array(
            [
                -self.max_reverse_speed,
                0.0,
                0.1 * self.max_linear_speed,
                0.25 * self.max_linear_speed,
                0.4 * self.max_linear_speed,
                0.6 * self.max_linear_speed,
                0.8 * self.max_linear_speed,
                self.max_linear_speed,
            ],
            dtype=np.float32,
        )
        self.dwa_w_samples = np.linspace(
            -self.max_angular_speed,
            self.max_angular_speed,
            num=9,
            dtype=np.float32,
        )
        self.recovery_fast_speed = 0.24
        self.recovery_slow_speed = 0.12
        self.path_smooth_window = 20
        self.path_smooth_passes = 3
        self.planner_mode = 'astar'  # 'astar' | 'dwa'

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
            # map_node stop publishes an empty global plan; clear target so cmd goes to zero.
            self.target_pose = None
            return
        self.global_path_xy = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float32,
        )

    def poi_change_callback(self, _msg: Odometry):
        # Keep old stop path working: clear target when POI sequence changes/stops.
        self.target_pose = None

    def info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            # P[0,3] = -fx * baseline
            fx = self.K[0, 0]
            Tx = msg.p[3] # From the right camera's projection matrix
            self.baseline = -Tx / fx
            self.get_logger().info(f"Camera intrinsics and baseline received. Baseline: {self.baseline:.4f}m")
            self.destroy_subscription(self.camera_info_sub)

    def camera_to_robot_center(self, T):
        """World control-center position from T_cam->world."""
        return T[:3, 3] - T[:3, :3] @ self.robot.cam_offset_3d

    def publish_height_map(self, origin, pooled_map, header):
        height_normalized = (np.nan_to_num(pooled_map, nan=0.0) + 5) * 30
        height_uint8 = height_normalized.astype(np.uint8)
        color_image = cv2.applyColorMap(height_uint8, cv2.COLORMAP_JET)
        img_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        img_msg.header = header
        self.height_map_pub.publish(img_msg)

    def publish_obstacle_mask(self, mask, stamp):
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.info.resolution = self.resolution
        msg.info.width = mask.shape[1]
        msg.info.height = mask.shape[0]
        msg.info.origin.position.x = self.origin[0]
        msg.info.origin.position.y = self.origin[1]
        msg.info.origin.position.z = self.origin[2] + self.grid_shape[2] * self.resolution / 2
        msg.info.origin.orientation.w = 1.0
        msg.data = np.where(mask, 100, 0).astype(np.int8).ravel(order="F").tolist()
        self.obstacle_mask_pub.publish(msg)

    def publish_cost_heatmap(self, cost, header):
        finite = np.isfinite(cost)
        c = np.nan_to_num(cost, nan=0.0, posinf=1e6, neginf=0.0)
        if np.any(finite):
            lo = float(np.min(c[finite]))
            hi = float(np.max(c[finite]))
            if hi > lo + 1e-6:
                norm = (c - lo) / (hi - lo)
            else:
                norm = np.zeros_like(c, dtype=np.float32)
        else:
            norm = np.zeros_like(c, dtype=np.float32)

        # 0(可行/低代价)->绿, 1(高代价)->红, 中间黄
        norm = np.clip(norm, 0.0, 1.0)
        h, w = norm.shape
        heat = np.zeros((h, w, 3), dtype=np.uint8)  # BGR
        heat[:, :, 2] = (norm * 255.0).astype(np.uint8)           # R
        heat[:, :, 1] = ((1.0 - norm) * 255.0).astype(np.uint8)   # G
        heat[:, :, 0] = 0                                          # B

        # unknown/invalid 置黑，防止误导
        heat[~finite] = (0, 0, 0)

        img_msg = self.bridge.cv2_to_imgmsg(heat, encoding='bgr8')
        img_msg.header = header
        self.astar_cost_pub.publish(img_msg)

    def publish_3d_occupancy_cloud(self, grid3d, resolution=0.1, origin=(0, 0, 0), stamp=None):
        occ = grid3d > 0.1
        has_occ = np.any(occ, axis=2)
        if not np.any(has_occ):
            points = []
        else:
            origin_np = np.asarray(origin, dtype=np.float64)
            # Keep only the top occupied voxel for each (x, y) column.
            z_from_top = np.argmax(occ[:, :, ::-1], axis=2)
            top_z = (occ.shape[2] - 1) - z_from_top
            xy = np.argwhere(has_occ)
            z = top_z[has_occ].reshape(-1, 1)
            top_voxels = np.hstack((xy, z)).astype(np.float64)
            world_coords = origin_np + top_voxels * float(resolution)
            points = world_coords.tolist()

        header = Header()
        header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        header.frame_id = "world"
        pc2_msg = pc2.create_cloud_xyz32(header, points)
        self.occupancy_cloud_pub.publish(pc2_msg)

    def publish_footprint(self, T, stamp):
        """Publish robot footprint rectangle as a point cloud for RViz."""
        forward = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        left = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
        center = self.camera_to_robot_center(T)

        fl, rl, hw = self.robot.footprint_from_control()
        corners = [
            center + forward * fl + left * hw,
            center + forward * fl - left * hw,
            center - forward * rl - left * hw,
            center - forward * rl + left * hw,
        ]

        # Draw rectangle edges by sampling points.
        edge_samples = 20
        points = []
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            for k in range(edge_samples + 1):
                t = k / edge_samples
                p = (1.0 - t) * a + t * b
                points.append(Point32(x=float(p[0]), y=float(p[1]), z=float(p[2])))

        msg = PointCloud()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.points = points
        self.footprint_pub.publish(msg)

    def publish_safety_gate_path(self, robot_xy, forward_xy, cmd, stamp, robot_z):
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = 'world'

        p0 = PoseStamped()
        p0.header = path_msg.header
        p0.pose.position.x = float(robot_xy[0])
        p0.pose.position.y = float(robot_xy[1])
        p0.pose.position.z = float(robot_z)
        p0.pose.orientation.w = 1.0
        path_msg.poses.append(p0)

        d = np.asarray(forward_xy, dtype=np.float32)
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            d = np.array([1.0, 0.0], dtype=np.float32)
        else:
            d = d / nd
        if cmd.linear.x < -1e-3:
            d = -d
        if abs(cmd.linear.x) < 1e-3:
            d = np.array([0.0, 0.0], dtype=np.float32)

        p1 = PoseStamped()
        p1.header = path_msg.header
        p1.pose.position.x = float(robot_xy[0] + d[0] * 0.35)
        p1.pose.position.y = float(robot_xy[1] + d[1] * 0.35)
        p1.pose.position.z = float(robot_z)
        p1.pose.orientation.w = 1.0
        path_msg.poses.append(p1)

        self.safety_path_pub.publish(path_msg)

    def _clamp_step(self, target: float, current: float, max_delta: float) -> float:
        return float(np.clip(target - current, -max_delta, max_delta) + current)

    def _clamp_cmd_limits(self, cmd: Twist) -> Twist:
        out = Twist()
        out.linear.x = float(np.clip(cmd.linear.x, -self.max_reverse_speed, self.max_linear_speed))
        out.angular.z = float(np.clip(cmd.angular.z, -self.max_angular_speed, self.max_angular_speed))
        out.linear.y = 0.0
        return out

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
        target_cmd = self._clamp_cmd_limits(target_cmd)

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

        _t_raycast0 = time.perf_counter()
        with Timer(name='raycasting', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            roll_dt = 0.0
            kernel_dt = 0.0
            fuse_dt = 0.0
            cloud_pub_dt = 0.0

            _t_roll0 = time.perf_counter()
            center = self.origin + np.array(self.grid_shape) * self.resolution / 2
            robot_pos = self.camera_to_robot_center(T)
            delta = robot_pos - center
            if np.linalg.norm(delta) > .1:
                new_center = robot_pos
                new_origin = new_center - np.array(self.grid_shape) * self.resolution / 2
                self.occupancy_grid, self.origin = roll_occupancy_grid(self.occupancy_grid, self.origin, new_origin, self.resolution)
            roll_dt = time.perf_counter() - _t_roll0

            _t_kernel0 = time.perf_counter()
            new_occ = run_raycasting_loopy(depth, T, self.grid_shape, fx, fy, cx, cy, self.origin, self.step, self.resolution)
            kernel_dt = time.perf_counter() - _t_kernel0

            _t_fuse0 = time.perf_counter()
            # seconds = log(0.5) / log(0.998) = 347.22 timestamp / 10 hz = around 35 seconds
            self.occupancy_grid *= 0.995
            self.occupancy_grid += new_occ
            self.occupancy_grid = np.clip(self.occupancy_grid, -0.2, 0.2)
            fuse_dt = time.perf_counter() - _t_fuse0

            _t_cloud0 = time.perf_counter()
            self.publish_3d_occupancy_cloud(
                self.occupancy_grid, self.resolution, self.origin, stamp=depth_msg.header.stamp
            )
            cloud_pub_dt = time.perf_counter() - _t_cloud0
        self.raycast_time_pub.publish(
            Float64(data=time.perf_counter() - _t_raycast0)
        )
        self.raycast_roll_time_pub.publish(Float64(data=roll_dt))
        self.raycast_kernel_time_pub.publish(Float64(data=kernel_dt))
        self.raycast_fuse_time_pub.publish(Float64(data=fuse_dt))
        self.raycast_cloud_pub_time_pub.publish(Float64(data=cloud_pub_dt))

        with Timer(name='heightmap+obstacle', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            height_map = occupancy_grid_to_height_map(self.occupancy_grid, self.origin, self.resolution)
            pooled_map = max_pool_height_map(height_map)
            robot_z = T[2, 3]
            pooled_map_rel = pooled_map - robot_z
            unknown_mask = ~np.isfinite(pooled_map_rel)
            obstacle_mask = build_obstacle_map(
                self.occupancy_grid, self.origin, self.resolution, robot_z,
                config=self.obstacle_config,
            )

        with Timer(name='vis heighmap', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            self.publish_height_map(T[:3,3], pooled_map_rel, depth_msg.header)
            self.publish_obstacle_mask(obstacle_mask, depth_msg.header.stamp)
            self.publish_footprint(T, depth_msg.header.stamp)


        with Timer(name='local plan', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            local_path_world = []
            effective_target = None
            if self.target_pose is not None:
                robot_center = self.camera_to_robot_center(T)
                robot_z = T[2, 3]
                forward_world = T[:3, :3] @ np.array([0.0, 0.0, 1.0])

                # Range-limited obstacle mask for planning: ignore obstacles beyond
                # v*4 s (min 1 m, max 6 m) to suppress far-range depth noise (e.g.
                # stair risers misclassified as walls at distance). Visualisation
                # keeps the full obstacle_mask unchanged.
                concern_m = float(np.clip(abs(self.smoothed_velocity) * 4.0, 3.0, 10.0))
                concern_cells = concern_m / self.resolution
                robot_gx, robot_gy = world_to_grid_2d(robot_center[:2], self.origin, self.resolution)
                h_om, w_om = obstacle_mask.shape
                xs = np.arange(h_om, dtype=np.float32) - robot_gx
                ys = np.arange(w_om, dtype=np.float32) - robot_gy
                dist_grid = np.sqrt(xs[:, np.newaxis] ** 2 + ys[np.newaxis, :] ** 2)
                planning_obstacle_mask = obstacle_mask.copy()
                planning_obstacle_mask[dist_grid > concern_cells] = False

                astar_cost = build_astar_cost_map(unknown_mask)
                astar_cost = astar_cost + build_proximity_cost(planning_obstacle_mask, max_cells=2, weight=2.0)
                # Apply global path bonus: reduce cost along the pre-planned route so
                # A*/DWA follows it rather than hugging tight-but-known walls.
                # astar_cost = apply_global_path_bonus(
                #     astar_cost, self.global_path_xy, self.origin, self.resolution,
                #     bonus=0.5, dilation_cells=2,
                # )
                self.publish_cost_heatmap(astar_cost, depth_msg.header)

                effective_target = snap_target_to_free(
                    self.target_pose[:2], robot_center[:2], forward_world[:2],
                    planning_obstacle_mask, self.origin, self.resolution,
                    global_path_xy=self.global_path_xy,
                )
                if effective_target is None:
                    self.get_logger().warning("No free target found on/near global path; stop this cycle.")
                else:

                    if self.planner_mode == 'dwa':
                        fl, rl, hw = self.robot.footprint_from_control()
                        dwa_path_xy = plan_dwa_minimal(
                            robot_center[:2],
                            forward_world[:2],
                            effective_target,
                            planning_obstacle_mask,
                            astar_cost,
                            self.origin,
                            self.resolution,
                            v_samples=self.dwa_v_samples,
                            w_samples=self.dwa_w_samples,
                            front_len=fl, rear_len=rl, half_w=hw,
                        )
                        for pxy in dwa_path_xy:
                            local_path_world.append(np.array([pxy[0], pxy[1], robot_z], dtype=np.float32))
                    else:
                        start_idx = clip_grid_2d(world_to_grid_2d(robot_center[:2], self.origin, self.resolution), planning_obstacle_mask.shape)
                        goal_idx = clip_grid_2d(world_to_grid_2d(effective_target, self.origin, self.resolution), planning_obstacle_mask.shape)

                        start_free = find_nearest_free(start_idx, planning_obstacle_mask)
                        goal_free = find_nearest_free(goal_idx, planning_obstacle_mask)

                        grid_path = astar_2d(astar_cost, planning_obstacle_mask, start_free, goal_free)

                        if len(grid_path) >= 2:
                            relax_scales = (1.0, 0.9, 0.8, 0.7)
                            min_keep = max(3, int(0.25 * len(grid_path)))
                            best_filtered = None

                            for s in relax_scales:
                                filtered = [grid_path[0]]
                                for i in range(1, len(grid_path)):
                                    gx, gy = grid_path[i]
                                    if planning_obstacle_mask[gx, gy]:
                                        break
                                    filtered.append(grid_path[i])

                                if best_filtered is None or len(filtered) > len(best_filtered):
                                    best_filtered = filtered
                                if len(filtered) >= min_keep:
                                    best_filtered = filtered
                                    break

                            if best_filtered is not None and len(best_filtered) >= 2:
                                grid_path = best_filtered

                        for g in grid_path:
                            p = grid_to_world_2d(g, self.origin, self.resolution, robot_z)
                            p[2] = robot_z
                            local_path_world.append(p)

                        if len(local_path_world) >= 3 and self.path_smooth_passes > 0:
                            smoothed = smooth_path_world(
                                local_path_world,
                                window=self.path_smooth_window,
                                passes=self.path_smooth_passes,
                            )
                            # Safety check: discard smooth if any point lands in obstacle
                            smooth_safe = True
                            for pt in smoothed:
                                gx, gy = world_to_grid_2d(pt[:2], self.origin, self.resolution)
                                if 0 <= gx < planning_obstacle_mask.shape[0] and 0 <= gy < planning_obstacle_mask.shape[1]:
                                    if planning_obstacle_mask[gx, gy]:
                                        smooth_safe = False
                                        break
                            if smooth_safe:
                                local_path_world = smoothed
                            # else: keep original A* path unmodified

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
            elif effective_target is None:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            else:
                robot_xy = self.camera_to_robot_center(T)[:2]
                forward_world = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
                forward_xy = forward_world[:2]
                target_dist = np.linalg.norm(effective_target - robot_xy)
                if target_dist < 0.25:
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                elif len(local_path_world) < 2:
                    # Recovery behavior: cautiously probe toward target instead of pure spinning.
                    # This keeps the robot moving even when footprint filtering truncates the path.
                    to_target = effective_target - robot_xy
                    norm_f = np.linalg.norm(forward_xy)
                    norm_t = np.linalg.norm(to_target)
                    if norm_f < 1e-6 or norm_t < 1e-6:
                        cmd.linear.x = 0.0
                        cmd.angular.z = 0.0
                    else:
                        forward_xy = forward_xy / norm_f
                        to_target = to_target / norm_t
                        heading_err = signed_angle_between(forward_xy, to_target)
                        cmd.angular.z = float(np.clip(1.6 * heading_err, -self.max_angular_speed, self.max_angular_speed))
                        cmd.linear.x = self.recovery_fast_speed if abs(heading_err) < 0.6 else self.recovery_slow_speed
                else:
                    lookahead = pick_lookahead_point(local_path_world, robot_xy, lookahead_dist=1.5)
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

                        cmd.angular.z = float(np.clip(1.8 * heading_err, -self.max_angular_speed, self.max_angular_speed))
                        heading_scale = max(0.0, np.cos(heading_err))
                        dist_scale = np.clip(target_dist / 1.0, 0.2, 1.0)
                        cmd.linear.x = float(np.clip(self.max_linear_speed * heading_scale * dist_scale, 0.0, self.max_linear_speed))
                        if abs(heading_err) > 1.0:
                            cmd.linear.x *= 0.40

            if self.target_pose is not None:
                forward_world = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
                robot_xy_gate = self.camera_to_robot_center(T)[:2]
                self.publish_safety_gate_path(robot_xy_gate, forward_world[:2], cmd, depth_msg.header.stamp, T[2, 3])
            self.latest_cmd = self._clamp_cmd_limits(cmd)
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
    parser.add_argument(
        '--robot_config',
        type=str,
        default=None,
        help='Path to robot YAML config (default: config/robot_go2.yaml)',
    )
    parsed_args, unknown_args = parser.parse_known_args(args)

    rclpy.init(args=unknown_args)
    node = PlanningNode(
        sensor_source=parsed_args.sensor_source,
        robot_config_path=parsed_args.robot_config,
    )

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
