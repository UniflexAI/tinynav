import heapq
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointField
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from cv_bridge import CvBridge
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from dataclasses import dataclass
from numba import njit
import message_filters
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointCloud
from geometry_msgs.msg import PoseStamped, Point32
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from codetiming import Timer
import cv2
from tinynav.core.math_utils import rotvec_to_matrix, quat_to_matrix, matrix_to_quat, msg2np


@dataclass
class RobotConfig:
    """Robot geometry. Body frame: +x forward, +y left."""
    name: str = 'go2'
    shape: str = 'square'
    length: float = 0.7
    width: float = 0.3
    radius: float = 0.3
    camera_x: float = 0.35
    camera_y: float = 0.0
    control_x: float = 0.0
    control_y: float = 0.0
    safety_radius: float = 0.1

    @property
    def cam_offset_3d(self):
        """Offset [left, up, forward] from control center to camera in body frame."""
        return np.array([self.camera_y - self.control_y, 0.0, self.camera_x - self.control_x], dtype=np.float32)

    @property
    def half_size(self):
        if self.shape == 'circle':
            return (self.radius, self.radius)
        return (self.length / 2.0, self.width / 2.0)

    def footprint_from_control(self):
        """Returns (front_len, rear_len, half_w) relative to control center."""
        hl, hw = self.half_size
        return float(hl - self.control_x), float(hl + self.control_x), float(hw)


GO2_CONFIG = RobotConfig(
    name='go2', shape='square',
    length=0.4, width=0.3,
    camera_x=0.2, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.05,
)

B2_CONFIG = RobotConfig(
    name='b2', shape='square',
    length=1.0, width=0.5,
    camera_x=0.5, camera_y=0.0,
    control_x=-0.5, control_y=0.0,
    safety_radius=0.1,
)

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
            if (not np.isfinite(d)) or d <= 0:
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


# Cost for cells with no A* route, or trajectories leaving the local grid.
OFF_ROUTE_COST = 1.0e3


@dataclass
class ObstacleConfig:
    robot_z_bottom: float = -0.4
    robot_z_top: float = 0.4
    occ_threshold: float = 0.1
    # Schmitt-trigger release level, keeps cells near occ_threshold from flipping every frame
    occ_release_threshold: float = 0.05
    min_wall_span_m: float = 0.2
    dilation_cells: int = 2


def _span_filtered_occupancy(band, threshold, min_span_m, resolution):
    """
    Cells whose occupied voxels span >= min_span_m vertically within the band.
    Walls have a large z-span, stair risers and ground bumps have a small one.
    """
    band_occ = band > threshold
    has_occ = np.any(band_occ, axis=2)
    n_z = band_occ.shape[2]
    z_idx = np.arange(n_z, dtype=np.float32)
    occ_high = np.where(band_occ, z_idx[np.newaxis, np.newaxis, :], -1).max(axis=2)
    occ_low = np.where(band_occ, z_idx[np.newaxis, np.newaxis, :], n_z).min(axis=2)
    z_span = (occ_high - occ_low) * resolution
    return has_occ & (z_span >= min_span_m)


def build_obstacle_map(occupancy_grid, origin, resolution, robot_z, config=None,
                       prev_core=None):
    """
    Mark cells whose occupied voxels span >= min_wall_span_m in z as obstacles.
    :return: (obstacle, core), dilated mask for the ESDF and the raw mask, which the
             caller feeds back as prev_core for hysteresis.
    """
    config = config or ObstacleConfig()
    h, w, z_dim = occupancy_grid.shape
    z_world = origin[2] + (np.arange(z_dim) + 0.5) * resolution
    z_rel = z_world - robot_z
    z_mask = (z_rel >= config.robot_z_bottom) & (z_rel <= config.robot_z_top)

    core = np.zeros((h, w), dtype=bool)
    if np.any(z_mask):
        band = occupancy_grid[:, :, z_mask]
        core = _span_filtered_occupancy(band, config.occ_threshold,
                                        config.min_wall_span_m, resolution)
        if prev_core is not None and config.occ_release_threshold < config.occ_threshold:
            # cells already marked as obstacle survive on the looser threshold
            held = _span_filtered_occupancy(band, config.occ_release_threshold,
                                            config.min_wall_span_m, resolution)
            core = core | (prev_core & held)

    obstacle = core
    if config.dilation_cells > 0 and np.any(core):
        obstacle = binary_dilation(core, iterations=config.dilation_cells)
    return obstacle, core

@njit(cache=True)
def generate_trajectory_library_3d(
    num_samples=15, duration=3.0, dt=0.1,
    init_p=np.zeros(3), init_q=np.array([0, 0, 0, 1])
):
    """Regular sampled lattice (forward-only)."""
    num_steps = int(duration / dt) + 1

    vx_max = 0.5
    n_vx = max(3, int(num_samples / 2))
    vx_samples = np.linspace(0.0, vx_max, n_vx)
    omega_y_samples = np.linspace(-np.pi / 3, np.pi / 3, num_samples)

    num_samples = len(vx_samples) * len(omega_y_samples)

    trajectories = np.empty((num_samples, num_steps, 7))
    params = np.empty((num_samples, 2))

    k = -1
    for i_vx in range(len(vx_samples)):
        for i_omega in range(len(omega_y_samples)):
            k += 1
            vx = vx_samples[i_vx]
            omega_y = omega_y_samples[i_omega]
            p = init_p.copy()
            q = quat_to_matrix(init_q)
            traj = np.empty((num_steps, 7))
            for i in range(num_steps):
                dq = rotvec_to_matrix(np.array([0.0, omega_y * dt, 0.0]))
                q = q @ dq
                v_world = q @ np.array([0.0, 0.0, vx])
                p += v_world * dt
                traj[i, :3] = p
                traj[i, 3:] = matrix_to_quat(q)
            #hack
            for i in range(num_steps):
                traj[i, 2] = traj[0, 2]
            trajectories[k] = traj
            params[k, 0] = vx
            params[k, 1] = omega_y
    return trajectories, params


def generate_predefined_trajectory_vocabularies(
    duration=3.0, dt=0.1,
    init_p=np.zeros(3), init_q=np.array([0, 0, 0, 1])
):
    """
    Predefined trajectory vocabularies.
    """
    num_steps = int(duration / dt) + 1
    trajectories = []
    params = []

    # constant reverse trajectory
    # vx = -0.2 m/s, omega = 0
    reverse_speed = 0.2
    p = init_p.copy()
    q = quat_to_matrix(init_q)
    traj = np.empty((num_steps, 7), dtype=np.float64)
    for i in range(num_steps):
        v_world = q @ np.array([0.0, 0.0, -reverse_speed])
        p += v_world * dt
        traj[i, :3] = p
        traj[i, 3:] = matrix_to_quat(q)
    for i in range(num_steps):
        traj[i, 2] = traj[0, 2]
    trajectories.append(traj)
    params.append(np.array([-reverse_speed, 0.0], dtype=np.float64))

    return np.asarray(trajectories), np.asarray(params)

@njit(cache=True)
def score_trajectories_by_ESDF(trajectories, ESDF_map, path_dist_map, remaining_map,
                                origin, resolution, safety_radius=0.1,
                                front_len=0.35, rear_len=0.35, half_w=0.15):
    """
    Score trajectories by ESDF clearance over the footprint (center + 4 corners) and
    against the A* route maps, which share the shape, origin and resolution of ESDF_map.
    :return: per trajectory, the obstacle score (inf on collision), the step index of the
             minimum clearance, the mean distance from the trajectory center to the route,
             and the route arc length still ahead of its end cell.
    """
    scores = []
    occ_points = []
    path_costs = []
    end_remainings = []
    ESDF_rows, ESDF_cols = ESDF_map.shape

    for t in range(len(trajectories)):
        traj = trajectories[t]
        min_dist_for_traj = float('inf')
        closest_step_for_traj = -1
        path_cost_sum = 0.0
        path_cost_n = 0

        for i in range(len(traj)):
            x_world, y_world = traj[i, 0], traj[i, 1]
            qx, qy, qz, qw = traj[i, 3], traj[i, 4], traj[i, 5], traj[i, 6]

            # world XY forward from quaternion (body +Z forward)
            fwd_x = 2.0 * (qx * qz + qw * qy)
            fwd_y = 2.0 * (qy * qz - qw * qx)
            n = (fwd_x * fwd_x + fwd_y * fwd_y) ** 0.5
            if n > 1e-6:
                fwd_x /= n
                fwd_y /= n
            else:
                fwd_x, fwd_y = 1.0, 0.0
            left_x = -fwd_y
            left_y = fwd_x

            # center + 4 corners, unrolled for numba
            check_xs = (
                x_world,
                x_world + fwd_x * front_len + left_x * half_w,
                x_world + fwd_x * front_len - left_x * half_w,
                x_world - fwd_x * rear_len  + left_x * half_w,
                x_world - fwd_x * rear_len  - left_x * half_w,
            )
            check_ys = (
                y_world,
                y_world + fwd_y * front_len + left_y * half_w,
                y_world + fwd_y * front_len - left_y * half_w,
                y_world - fwd_y * rear_len  + left_y * half_w,
                y_world - fwd_y * rear_len  - left_y * half_w,
            )

            for k in range(5):
                x_img = int((check_xs[k] - origin[0]) / resolution)
                y_img = int((check_ys[k] - origin[1]) / resolution)
                if 0 <= x_img < ESDF_rows and 0 <= y_img < ESDF_cols:
                    dist = ESDF_map[x_img, y_img]
                    if dist < min_dist_for_traj:
                        min_dist_for_traj = dist
                        closest_step_for_traj = i
                    if k == 0:  # route adherence is measured on the center only
                        path_cost_sum += float(path_dist_map[x_img, y_img])
                        path_cost_n += 1

        if path_cost_n > 0:
            path_costs.append(path_cost_sum / path_cost_n)
        else:
            path_costs.append(OFF_ROUTE_COST)  # the whole trajectory left the grid

        end_x_img = int((traj[-1, 0] - origin[0]) / resolution)
        end_y_img = int((traj[-1, 1] - origin[1]) / resolution)
        if 0 <= end_x_img < ESDF_rows and 0 <= end_y_img < ESDF_cols:
            end_remainings.append(float(remaining_map[end_x_img, end_y_img]))
        else:
            end_remainings.append(OFF_ROUTE_COST)

        if min_dist_for_traj < 1e-3:  # collision
            scores.append(float('inf'))
        elif min_dist_for_traj != float('inf'):
            if min_dist_for_traj > safety_radius:
                scores.append(0.0)
            else:
                max_steps = len(traj)
                decay_factor = (max_steps - closest_step_for_traj) / max_steps
                base_score = 1.0 / (min_dist_for_traj + 1e-3)
                scores.append(decay_factor * base_score)
        else:
            scores.append(0.0)
        occ_points.append(closest_step_for_traj)
    return scores, occ_points, path_costs, end_remainings

def roll_and_clear(array, shift_cells, fill):
    """
    Roll an array by whole cells along its leading axes and clear the newly exposed band,
    which has not been observed at its new coordinates.
    """
    rolled = np.roll(array, shift=tuple(-shift_cells), axis=tuple(range(len(shift_cells))))
    for axis, shift in enumerate(shift_cells):
        if shift == 0:
            continue
        band = [slice(None)] * array.ndim
        band[axis] = slice(-shift, None) if shift > 0 else slice(None, -shift)
        rolled[tuple(band)] = fill
    return rolled

def roll_occupancy_grid(occupancy_grid, old_origin, new_origin, resolution):
    shift_m = new_origin - old_origin
    shift_voxels = np.round(shift_m / resolution).astype(int)
    if np.all(shift_voxels == 0):
        return occupancy_grid, old_origin
    rolled = roll_and_clear(occupancy_grid, shift_voxels, 0)
    updated_origin = old_origin + shift_voxels * resolution
    return rolled, updated_origin


def _astar_grid_path(start_idx, goal_idx, blocked, resolution, hist_cost=None):
    """
    8-connected A* over a 2D blocked-cell grid.
    :param hist_cost: optional per-cell surcharge in m on entering a cell, see build_history_cost.
    :return: grid indices from start_idx to goal_idx, or to the closest visited cell when
             goal_idx is unreachable.
    """
    rows, cols = blocked.shape
    neighbor_offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    def heuristic(a, b):
        return resolution * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    open_heap = [(heuristic(start_idx, goal_idx), start_idx)]
    came_from = {}
    g_score = {start_idx: 0.0}
    visited = set()
    best_reached = start_idx
    best_reached_h = heuristic(start_idx, goal_idx)

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        h = heuristic(current, goal_idx)
        if h < best_reached_h:
            best_reached_h = h
            best_reached = current
        if current == goal_idx:
            break

        for dr, dc in neighbor_offsets:
            neighbor = (current[0] + dr, current[1] + dc)
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            if blocked[neighbor]:
                continue
            step_cost = resolution * (1.41421356 if dr != 0 and dc != 0 else 1.0)
            if hist_cost is not None:
                step_cost += float(hist_cost[neighbor])
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative_g + heuristic(neighbor, goal_idx), neighbor))

    path = [best_reached]
    node = best_reached
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def smooth_route(route_xy, blocked, origin, resolution,
                 weight_data=0.4, weight_smooth=0.3, iterations=40):
    """
    Gradient-descent smoothing of an A* route, endpoints pinned.
    Each interior point is pulled weight_data toward where A* put it and weight_smooth
    toward the average of its neighbors, all updated at once (Jacobi). Moves into blocked
    space are rejected, so the route is never smoothed through an obstacle.
    :return: (N, 2) world xy, same point count and endpoints as the input.
    """
    if len(route_xy) < 3 or iterations <= 0:
        return list(route_xy)

    rows, cols = blocked.shape
    original = np.asarray(route_xy, dtype=float)
    smoothed = original.copy()

    for _ in range(iterations):
        interior = smoothed[1:-1]
        laplacian = smoothed[:-2] + smoothed[2:] - 2.0 * interior
        candidate = (interior
                     + weight_data * (original[1:-1] - interior)
                     + weight_smooth * laplacian)
        r = ((candidate[:, 0] - origin[0]) / resolution).astype(np.int64)
        c = ((candidate[:, 1] - origin[1]) / resolution).astype(np.int64)
        inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        accept = inside.copy()
        accept[inside] &= ~blocked[r[inside], c[inside]]
        interior[accept] = candidate[accept]
    return list(smoothed)


def build_history_cost(prev_route_xy, blocked, origin, resolution, weight, cap,
                       min_valid_frac=0.5):
    """
    Per-cell surcharge in m that biases A* toward the route of the last cycle, so that
    near-tied choices around an obstacle do not flip sides on noise alone.
    prev_route_xy is in world coordinates and re-rasterized against the current origin.
    :return: (H, W) cost map, or None when less than min_valid_frac of the route survives.
    """
    if prev_route_xy is None or len(prev_route_xy) < 2 or weight <= 0.0:
        return None
    rows, cols = blocked.shape
    route = np.asarray(prev_route_xy, dtype=float)
    r = ((route[:, 0] - origin[0]) / resolution).astype(np.int64)
    c = ((route[:, 1] - origin[1]) / resolution).astype(np.int64)
    inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)  # rolled-away cells drop out
    in_grid = int(inside.sum())
    if in_grid == 0:
        return None
    prev_mask = np.zeros((rows, cols), dtype=bool)
    prev_mask[r[inside], c[inside]] = True
    prev_mask &= ~blocked
    if prev_mask.sum() < min_valid_frac * in_grid:
        return None
    dist = distance_transform_edt(~prev_mask) * resolution
    return (weight * np.minimum(dist, cap)).astype(np.float32)


def plan_local_astar_route(start_xy, goal_xy, ESDF_map, origin, resolution,
                            safety_radius=0.1, robot_half_width=0.0,
                            prev_route_xy=None, history_weight=0.0, history_cap=0.5,
                            smooth_weight=0.3, smooth_data_weight=0.4,
                            smooth_iterations=40):
    """
    Plan a local, obstacle-aware route from start_xy to goal_xy over the freshly observed
    occupancy grid, smooth it, and return it as world-xy points.
    Cells are blocked wherever ESDF_map < safety_radius + robot_half_width. The search
    carries no heading state, so the inflation keeps the footprint corners clear at any yaw.
    :return: (route, start_margin), route is empty when no useful route exists. start_margin
             is the clearance of the robot cell minus the inflation, negative when the robot
             sits inside the inflated band and the caller has to reverse out.
    """
    rows, cols = ESDF_map.shape
    inflation = safety_radius + robot_half_width
    blocked = ESDF_map < inflation

    def to_idx(xy):
        r = int((xy[0] - origin[0]) / resolution)
        c = int((xy[1] - origin[1]) / resolution)
        return (min(max(r, 0), rows - 1), min(max(c, 0), cols - 1))

    def to_world(idx):
        return np.array([
            origin[0] + (idx[0] + 0.5) * resolution,
            origin[1] + (idx[1] + 0.5) * resolution,
        ])

    start_idx = to_idx(start_xy)
    goal_idx = to_idx(goal_xy)

    start_margin = float(ESDF_map[start_idx]) - inflation
    if start_margin < 0.0:
        return [], start_margin  # inside the inflated band, the caller reverses out

    hist_cost = build_history_cost(prev_route_xy, blocked, origin, resolution,
                                   history_weight, history_cap)
    path_idx = _astar_grid_path(start_idx, goal_idx, blocked, resolution, hist_cost)
    if len(path_idx) < 2:
        return [], start_margin  # no reachable progress from start

    # smooth before anything consumes it, the route feeds the DWA lookup maps,
    # the history bias and the published path alike
    route = smooth_route([to_world(idx) for idx in path_idx], blocked,
                         origin, resolution, smooth_data_weight,
                         smooth_weight, smooth_iterations)
    return route, start_margin


def build_route_fields(route_xy, shape, origin, resolution):
    """
    Rasterize an A* route into the lookup maps read by the DWA scoring, so that scoring
    costs one array lookup per trajectory point instead of a search over route points.
    The route is resampled at half-cell steps so the rasterized line has no gaps for the EDT.
    :param route_xy: (N, 2) route in world xy, smooth_route moves points off cell centers.
    :return: (path_dist_map, remaining_map, has_route). path_dist_map is the distance in m
             from each cell to the nearest route cell, remaining_map is the route arc length
             still ahead, with off-route cells inheriting the arc length of their nearest
             route cell.
    """
    path_dist_map = np.full(shape, OFF_ROUTE_COST, dtype=np.float32)
    remaining_map = np.full(shape, OFF_ROUTE_COST, dtype=np.float32)
    if len(route_xy) < 2:
        return path_dist_map, remaining_map, False

    rows, cols = shape
    route = np.asarray(route_xy, dtype=float)
    node_arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))))
    arc = float(node_arc[-1])
    if arc < 1e-9:
        return path_dist_map, remaining_map, False

    sample_arc = np.linspace(0.0, arc, int(np.ceil(arc / (0.5 * resolution))) + 1)
    r = ((np.interp(sample_arc, node_arc, route[:, 0]) - origin[0]) / resolution).astype(np.int64)
    c = ((np.interp(sample_arc, node_arc, route[:, 1]) - origin[1]) / resolution).astype(np.int64)
    inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    if not np.any(inside):
        return path_dist_map, remaining_map, False

    route_mask = np.zeros(shape, dtype=bool)
    arc_map = np.zeros(shape, dtype=np.float32)
    route_mask[r[inside], c[inside]] = True
    # last write wins, a cell crossed twice reports the later arc length
    arc_map[r[inside], c[inside]] = sample_arc[inside]

    dist_cells, (near_r, near_c) = distance_transform_edt(~route_mask, return_indices=True)
    path_dist_map = (dist_cells * resolution).astype(np.float32)
    remaining_map = (arc - arc_map[near_r, near_c]).astype(np.float32)
    return path_dist_map, remaining_map, True


# === PlanningNode class ===
class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')
        self.robot = GO2_CONFIG
        self.get_logger().info(
            f"Robot: {self.robot.name} ({self.robot.shape} {self.robot.length}x{self.robot.width}m, "
            f"cam=({self.robot.camera_x},{self.robot.camera_y}), "
            f"ctrl=({self.robot.control_x},{self.robot.control_y}), "
            f"safety_r={self.robot.safety_radius}m)"
        )
        self.bridge = CvBridge()
        self.path_pub = self.create_publisher(Path, '/planning/trajectory_path', 10)
        self.astar_path_pub = self.create_publisher(Path, '/planning/astar_path', 10)
        self.height_map_pub = self.create_publisher(Image, "/planning/height_map", 10)
        self.obstacle_mask_pub = self.create_publisher(OccupancyGrid, '/planning/obstacle_mask', 10)
        self.footprint_pub = self.create_publisher(PointCloud, '/planning/footprint', 10)
        self.occupancy_cloud_pub = self.create_publisher(PointCloud2, '/planning/occupied_voxels', 10)
        self.occupancy_cloud_esdf_pub = self.create_publisher(PointCloud2, '/planning/occupied_voxels_with_esdf', 10)
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/planning/occupancy_grid', 10)
        self.depth_sub = message_filters.Subscriber(self, Image, '/slam/depth')
        self.pose_sub = message_filters.Subscriber(self, Odometry, '/slam/odometry_visual')

        self.ts = message_filters.TimeSynchronizer([self.depth_sub, self.pose_sub], queue_size=10)
        self.ts.registerCallback(self.sync_callback)
        self.camerainfo_sub = self.create_subscription(CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)

        self.grid_shape = (100, 100, 10)
        self.resolution = 0.1
        self.origin = np.array(self.grid_shape) * self.resolution / -2.
        self.step = 10
        self.occupancy_grid = np.zeros(self.grid_shape)
        self.K = None
        self.baseline = None
        self.last_T = None
        self.last_param = (0.0, 0.0) # acc and gyro
        self.obstacle_config = ObstacleConfig()
        # DWA cost weights, the obstacle term dominates with its 1e5 multiplier in
        # cost_function. Among the survivors, progress drives the speed, follow keeps
        # the robot on the side of an obstacle chosen by A*, and terminal breaks the
        # tie once remaining_map saturates at 0 past the route end.
        self.w_route_progress = 100.0
        self.w_path_follow = 80.0
        self.w_goal_terminal = 100.0
        self.route_terminal_band = 0.5  # m of remaining route that arms w_goal_terminal

        # surcharge for straying from the route of the last cycle, capped so a better
        # route can still win, a fully deviating route pays about +50% length here
        self.astar_history_weight = 0.1
        self.astar_history_cap = 0.5  # m
        self.astar_smooth_weight = 0.3
        self.astar_smooth_data_weight = 0.4
        self.astar_smooth_iterations = 15  # converges by ~10 sweeps at any route length
        # extra clearance required before forward motion resumes, so the robot does
        # not re-enter the inflated band and chatter at its boundary
        self.blocked_reverse_exit_margin = 0.10  # m
        self._reversing_from_blocked = False
        self.prev_route_world = None  # in world xy, self.origin rolls so indices would not survive
        self.prev_obstacle_core = None  # pre-dilation mask feeding the occupancy hysteresis
        self.stamp = None
        self.current_pose = None  # Store the latest pose from odometry

        self.smoothed_velocity = 0.0

        self.create_subscription(Odometry, '/control/target_pose', self.target_pose_callback, 10)
        self.target_pose = None

        self.poi_change_sub = self.create_subscription(Odometry, "/mapping/poi_change", self.poi_change_callback, 10)

    def poi_change_callback(self, msg):
        self.target_pose = None
        self.prev_route_world = None  # the remembered route leads to the old target

    def target_pose_callback(self, msg):
        self.target_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            # P[0,3] = -fx * baseline
            fx = self.K[0, 0]
            Tx = msg.p[3] # From the right camera's projection matrix
            self.baseline = -Tx / fx
            self.get_logger().info(f"Camera intrinsics and baseline received. Baseline: {self.baseline:.4f}m")
            self.destroy_subscription(self.camerainfo_sub)

    def camera_to_robot_center(self, T):
        """World control-center position derived from camera pose T_cam->world."""
        return T[:3, 3] - T[:3, :3] @ self.robot.cam_offset_3d

    def publish_footprint(self, T, stamp):
        """Publish robot footprint rectangle as a PointCloud for RViz."""
        forward = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        left    = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
        center  = self.camera_to_robot_center(T)
        fl, rl, hw = self.robot.footprint_from_control()
        corners = [
            center + forward * fl + left * hw,
            center + forward * fl - left * hw,
            center - forward * rl - left * hw,
            center - forward * rl + left * hw,
        ]
        points = []
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            for k in range(21):
                t = k / 20
                p = (1.0 - t) * a + t * b
                points.append(Point32(x=float(p[0]), y=float(p[1]), z=float(p[2])))
        msg = PointCloud()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.points = points
        self.footprint_pub.publish(msg)

    def publish_astar_path(self, route_xy, z, stamp):
        """
        Publish the A* route (world xy) as a Path, separately from the sampled arc
        on /planning/trajectory_path.
        """
        path = Path()
        path.header = Header()
        path.header.stamp = stamp
        path.header.frame_id = "world"
        for p in route_xy:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(p[0])
            pose.pose.position.y = float(p[1])
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.astar_path_pub.publish(path)

    def _front_obstacle_dist(self, T, obstacle_mask, max_dist=0.5):
        """
        Distance from the front face of the robot to the nearest obstacle in the forward
        corridor. The scan starts at the front face, so the value is the physical clearance.
        """
        center = self.camera_to_robot_center(T)
        fwd = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        n = (fwd[0] ** 2 + fwd[1] ** 2) ** 0.5
        fx, fy = (fwd[0] / n, fwd[1] / n) if n > 1e-6 else (1.0, 0.0)
        lx, ly = -fy, fx
        fl, _, hw = self.robot.footprint_from_control()
        rows, cols = obstacle_mask.shape
        steps = int(max_dist / self.resolution) + 1
        for step in range(steps):
            d_from_face = step * self.resolution
            d_from_center = fl + d_from_face
            for w in (-hw, 0.0, hw):
                xi = int((center[0] + fx * d_from_center + lx * w - self.origin[0]) / self.resolution)
                yi = int((center[1] + fy * d_from_center + ly * w - self.origin[1]) / self.resolution)
                if 0 <= xi < rows and 0 <= yi < cols and obstacle_mask[xi, yi]:
                    return d_from_face
        return max_dist + 1.0

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

    def publish_height_map(self, origin, esdf_map, header):
        height_normalized = np.clip(esdf_map / 2.0 * 255, 0, 255).astype(np.uint8)
        color_image = cv2.applyColorMap(height_normalized, cv2.COLORMAP_JET)
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

    @Timer(name="Planning Loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms")
    def sync_callback(self, depth_msg, odom_msg):
        if self.K is None:
            return
        with Timer(name='preprocess', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            stamp = Time.from_msg(odom_msg.header.stamp).nanoseconds / 1e9
            T,_ = msg2np(odom_msg)
            if self.last_T is None:
                self.last_T = T.copy()
                self.smoothed_velocity = 0.0
                self.last_stamp = 0
                self.smoothed_velocity = 0.0
            velocity_estimated = np.linalg.norm(T[:3, 3] - self.last_T[:3, 3]) / (stamp - self.last_stamp)
            self.smoothed_velocity = 0.9 * self.smoothed_velocity + 0.1 * velocity_estimated
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]

        with Timer(name='raycasting', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            center = self.origin + np.array(self.grid_shape) * self.resolution / 2
            robot_pos = T[:3, 3]
            delta = robot_pos - center
            if np.linalg.norm(delta) > .1:
                new_center = robot_pos
                new_origin = new_center - np.array(self.grid_shape) * self.resolution / 2
                old_origin = self.origin.copy()
                self.occupancy_grid, self.origin = roll_occupancy_grid(self.occupancy_grid, self.origin, new_origin, self.resolution)
                # the hysteresis mask is indexed by grid cell, so it rolls with the grid
                shift_cells = np.round((self.origin - old_origin) / self.resolution).astype(int)[:2]
                if self.prev_obstacle_core is not None and np.any(shift_cells):
                    self.prev_obstacle_core = roll_and_clear(self.prev_obstacle_core, shift_cells, False)
            new_occ = run_raycasting_loopy(depth, T, self.grid_shape, fx, fy, cx, cy, self.origin, self.step, self.resolution)
            self.occupancy_grid *= 0.99
            self.occupancy_grid += new_occ
            self.occupancy_grid = np.clip(self.occupancy_grid, -0.2, 0.2)

            self.publish_3d_occupancy_cloud(self.occupancy_grid, self.resolution, self.origin)

        with Timer(name='obstacle map', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            obstacle_mask, self.prev_obstacle_core = build_obstacle_map(
                self.occupancy_grid, self.origin, self.resolution,
                robot_z=T[2, 3], config=self.obstacle_config,
                prev_core=self.prev_obstacle_core,
            )
            ESDF_map = distance_transform_edt(~obstacle_mask).astype(np.float32) * self.resolution

        with Timer(name='vis', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            self.publish_3d_occupancy_cloud_with_esdf(self.occupancy_grid, ESDF_map, self.resolution, self.origin)
            self.publish_height_map(T[:3,3], ESDF_map, depth_msg.header)
            self.publish_2d_occupancy_grid(ESDF_map, self.origin, self.resolution, depth_msg.header.stamp, z_offset=self.grid_shape[2]*self.resolution/2)
            self.publish_obstacle_mask(obstacle_mask, depth_msg.header.stamp)
            self.publish_footprint(T, depth_msg.header.stamp)

        with Timer(name='traj gen', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            init_p = self.camera_to_robot_center(T)
            init_q = np.array([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w])
            trajectories, params = generate_trajectory_library_3d(init_p=init_p, init_q=init_q)
            vocab_trajs, vocab_params = generate_predefined_trajectory_vocabularies(init_p=init_p, init_q=init_q)
            trajectories = np.concatenate([trajectories, vocab_trajs], axis=0)
            params = np.concatenate([params, vocab_params], axis=0)
            self.last_T = T
            self.last_stamp = stamp

        with Timer(name='astar', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            astar_route = []
            start_margin = float('inf')
            if self.target_pose is not None:
                astar_route, start_margin = plan_local_astar_route(
                    init_p[:2], self.target_pose[:2], ESDF_map,
                    self.origin, self.resolution, self.robot.safety_radius,
                    self.robot.width / 2.0,
                    self.prev_route_world, self.astar_history_weight, self.astar_history_cap,
                    self.astar_smooth_weight, self.astar_smooth_data_weight,
                    self.astar_smooth_iterations,
                )
                self.prev_route_world = astar_route  # build_history_cost drops a short one
                if astar_route:
                    self.publish_astar_path(astar_route, init_p[2], depth_msg.header.stamp)
            path_dist_map, remaining_map, has_route = build_route_fields(
                astar_route, ESDF_map.shape, self.origin, self.resolution,
            )

        with Timer(name='traj score', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            front_len, rear_len, half_w = self.robot.footprint_from_control()
            scores, occ_points, path_costs, end_remainings = score_trajectories_by_ESDF(
                trajectories, ESDF_map, path_dist_map, remaining_map,
                self.origin, self.resolution, self.robot.safety_radius,
                front_len, rear_len, half_w,
            )

        with Timer(name='pub', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            front_clearance = self._front_obstacle_dist(T, obstacle_mask)
            enter_threshold = 0.2

            # latched, releasing at margin 0 would put the robot back on the
            # boundary it just backed off and oscillate there
            if start_margin < 0.0:
                self._reversing_from_blocked = True
            elif start_margin > self.blocked_reverse_exit_margin:
                self._reversing_from_blocked = False
            if self._reversing_from_blocked:
                self.get_logger().warning(
                    f'Robot inside inflated band (margin {start_margin:+.2f} m), reversing out'
                )

            def cost_function(traj, param, score, path_cost, end_remaining):
                # predefined backward trajectory penalty
                is_backward_traj = param[0] < 0.0
                should_reverse = (front_clearance <= enter_threshold
                                  or self._reversing_from_blocked)
                reverse_gate_penalty = 0.0
                if should_reverse != is_backward_traj:
                    reverse_gate_penalty = 1e9

                smoothness = (10 * abs(self.last_param[0] - param[0])
                              + 10 * abs(self.last_param[1] - param[1]))

                if not has_route:
                    # no route this cycle, both route maps are flat, so fall back
                    # to steering at the raw target
                    traj_end = np.array(traj[-1, :3])
                    target_end = self.target_pose if self.target_pose is not None else traj_end
                    return (score * 100000 + 100 * np.linalg.norm(traj_end - target_end)
                            + smoothness + reverse_gate_penalty)

                # xy only, the trajectory z is pinned to the height of the robot
                terminal = 0.0
                if end_remaining < self.route_terminal_band and self.target_pose is not None:
                    terminal = self.w_goal_terminal * float(
                        np.linalg.norm(traj[-1, :2] - self.target_pose[:2])
                    )

                return (score * 100000
                        + self.w_route_progress * end_remaining
                        + self.w_path_follow * path_cost
                        + terminal
                        + smoothness
                        + reverse_gate_penalty)

            top_k = 1
            costs = np.array([
                cost_function(trajectories[i], params[i], scores[i], path_costs[i],
                              end_remainings[i])
                for i in range(len(trajectories))
            ])
            top_indices = np.argsort(costs, kind='stable')[:top_k]
            self.last_param = params[top_indices[0]]

            # path
            path = Path()
            path.header = depth_msg.header
            path.header.frame_id = "world"

            if self.target_pose is None:
                return

            if all(s == float('inf') for s in scores):
                self.get_logger().info('All trajectories in collision, stopping path.')
                return

            for i in top_indices:
                for j in range(0, len(trajectories[i]), 10):
                    x,y,z,qx,qy,qz,qw = trajectories[i][j]
                    pose = PoseStamped()
                    pose.header = depth_msg.header
                    pose.pose.position.x = x
                    pose.pose.position.y = y
                    pose.pose.position.z = z
                    pose.pose.orientation.x = qx
                    pose.pose.orientation.y = qy
                    pose.pose.orientation.z = qz
                    pose.pose.orientation.w = qw
                    path.poses.append(pose)
            self.path_pub.publish(path)

def main(args=None):
    rclpy.init(args=args)
    node = PlanningNode()

    try:
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
