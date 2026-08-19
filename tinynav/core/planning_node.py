import os
import rclpy
import json
import threading
import multiprocessing as mp
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointField
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from cv_bridge import CvBridge
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, label as connected_components
from dataclasses import dataclass
from numba import njit
import message_filters
import time
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointCloud
from geometry_msgs.msg import PoseStamped, Point32
from tf2_ros import Buffer, TransformException, TransformListener
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Float32, Header, String
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
    safety_radius: float = 0.5
    comfort_radius: float = 0.5

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
    safety_radius=0.2,
    comfort_radius=0.2,
)

B2_CONFIG = RobotConfig(
    name='b2', shape='square',
    length=0.5, width=0.35,
    camera_x=0.3, camera_y=0.0,
    control_x=0.0, control_y=0.0,
    safety_radius=0.2,
    comfort_radius=0.2,
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


@njit(cache=True)
def run_raycasting_points_loopy(points, T_sensor_to_world, grid_shape, origin, resolution, min_votes=2):
    """
    Same free-space carving / hit accumulation scheme as run_raycasting_loopy, but the
    input is already a set of sensor-frame 3D points (e.g. radar returns) instead of a
    depth image, so there is no per-pixel intrinsics projection step.

    A voxel only gets marked occupied if at least min_votes points in this frame land in
    it -- indoor lidar returns include isolated "fly points" (edge/reflection artifacts)
    that don't recur at the same voxel, unlike a real surface which gets hit by many
    points at once. Free-space carving along each ray is applied regardless of the vote
    count: treating a noisy ray's path as clear is the safer failure mode than treating a
    real obstacle's path as clear, so it isn't gated the same way.
    """
    occupancy_grid = np.zeros(grid_shape)
    grid_shape_x, grid_shape_y, grid_shape_z = grid_shape
    origin_x, origin_y, origin_z = origin

    sensor_orig_x = T_sensor_to_world[0, 3]
    sensor_orig_y = T_sensor_to_world[1, 3]
    sensor_orig_z = T_sensor_to_world[2, 3]

    start_voxel_x = int(np.floor((sensor_orig_x - origin_x) / resolution))
    start_voxel_y = int(np.floor((sensor_orig_y - origin_y) / resolution))
    start_voxel_z = int(np.floor((sensor_orig_z - origin_z) / resolution))

    n_points = points.shape[0]
    end_voxels = np.empty((n_points, 3), dtype=np.int64)
    hit_count = np.zeros(grid_shape, dtype=np.int32)

    # Pass 1: transform each point to world, record its end voxel, and tally per-voxel votes.
    for n in range(n_points):
        px, py, pz = points[n, 0], points[n, 1], points[n, 2]

        pw_x = T_sensor_to_world[0, 0] * px + T_sensor_to_world[0, 1] * py + T_sensor_to_world[0, 2] * pz + T_sensor_to_world[0, 3]
        pw_y = T_sensor_to_world[1, 0] * px + T_sensor_to_world[1, 1] * py + T_sensor_to_world[1, 2] * pz + T_sensor_to_world[1, 3]
        pw_z = T_sensor_to_world[2, 0] * px + T_sensor_to_world[2, 1] * py + T_sensor_to_world[2, 2] * pz + T_sensor_to_world[2, 3]

        end_voxel_x = int(np.floor((pw_x - origin_x) / resolution))
        end_voxel_y = int(np.floor((pw_y - origin_y) / resolution))
        end_voxel_z = int(np.floor((pw_z - origin_z) / resolution))
        end_voxels[n, 0] = end_voxel_x
        end_voxels[n, 1] = end_voxel_y
        end_voxels[n, 2] = end_voxel_z

        if (0 <= end_voxel_x < grid_shape_x and
            0 <= end_voxel_y < grid_shape_y and
            0 <= end_voxel_z < grid_shape_z):
            hit_count[end_voxel_x, end_voxel_y, end_voxel_z] += 1

    # Pass 2: free-space carving (always) + occupied vote (gated on hit_count).
    for n in range(n_points):
        end_voxel_x = end_voxels[n, 0]
        end_voxel_y = end_voxels[n, 1]
        end_voxel_z = end_voxels[n, 2]

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
            if hit_count[end_voxel_x, end_voxel_y, end_voxel_z] >= min_votes:
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


@dataclass
class ObstacleConfig:
    robot_z_bottom: float = -0.6
    robot_z_top: float = 0.2
    occ_threshold: float = 0.1
    min_wall_span_m: float = 0.4
    dilation_cells: int = 0


def build_obstacle_map(occupancy_grid, origin, resolution, robot_z, config=None):
    """Obstacle = cells where occupied voxels span >= min_wall_span_m in z.
    Walls have large z-span; stair risers / ground bumps have small span."""
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


def remove_small_obstacle_components(obstacle_mask, min_area_cells):
    """Remove tiny 2D obstacle islands, mainly for lidar fly points."""
    if min_area_cells <= 1 or not np.any(obstacle_mask):
        return obstacle_mask
    labeled, component_count = connected_components(obstacle_mask)
    if component_count == 0:
        return obstacle_mask
    counts = np.bincount(labeled.ravel())
    keep = counts >= min_area_cells
    keep[0] = False
    return keep[labeled]

@njit(cache=True)
def generate_trajectory_library_3d(
    num_samples=15, duration=3.0, dt=0.1,
    init_p=np.zeros(3), init_q=np.array([0, 0, 0, 1]),
    vx_max=0.5,
):
    """Regular sampled lattice (forward-only)."""
    num_steps = int(duration / dt) + 1

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
    init_p=np.zeros(3), init_q=np.array([0, 0, 0, 1]),
    reverse_speed=0.3,
    reverse_omegas=(0.0, -0.5, 0.5),
):
    """
    Predefined trajectory vocabularies.
    """
    num_steps = int(duration / dt) + 1
    trajectories = []
    params = []

    # Recovery reverse trajectories.
    # Keep the speed conservative here; speed tuning is handled separately.
    for omega_y in reverse_omegas:
        p = init_p.copy()
        q = quat_to_matrix(init_q)
        traj = np.empty((num_steps, 7), dtype=np.float64)
        for i in range(num_steps):
            dq = rotvec_to_matrix(np.array([0.0, omega_y * dt, 0.0]))
            q = q @ dq
            v_world = q @ np.array([0.0, 0.0, -reverse_speed])
            p += v_world * dt
            traj[i, :3] = p
            traj[i, 3:] = matrix_to_quat(q)
        for i in range(num_steps):
            traj[i, 2] = traj[0, 2]
        trajectories.append(traj)
        params.append(np.array([-reverse_speed, omega_y], dtype=np.float64))

    return np.asarray(trajectories), np.asarray(params)

@njit(cache=True)
def score_trajectories_by_ESDF(trajectories, ESDF_map, origin, resolution, safety_radius=0.1, comfort_radius=0.8,
                                front_len=0.35, rear_len=0.35, half_w=0.15,
                                score_percentile=0.0, collision_tolerance=0):
    """Score trajectories by footprint clearance.

    Default score_percentile=0 and collision_tolerance=0 preserves the old
    min-clearance behavior. Lidar mode can raise these slightly so isolated
    occupancy spikes do not kill an otherwise good trajectory.
    """
    scores = []
    occ_points = []
    ESDF_rows, ESDF_cols = ESDF_map.shape

    for t in range(len(trajectories)):
        traj = trajectories[t]
        min_dist_for_traj = float('inf')
        closest_step_for_traj = -1
        clearance_values = np.empty(len(traj) * 5, dtype=np.float64)
        clearance_count = 0
        collision_count = 0

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
                    clearance_values[clearance_count] = dist
                    clearance_count += 1
                    if dist < 1e-3:
                        collision_count += 1
                    if dist < min_dist_for_traj:
                        min_dist_for_traj = dist
                        closest_step_for_traj = i

        effective_dist_for_traj = min_dist_for_traj
        if clearance_count > 0 and score_percentile > 0.0:
            sorted_clearance = np.sort(clearance_values[:clearance_count])
            percentile_index = int((score_percentile / 100.0) * (clearance_count - 1))
            effective_dist_for_traj = sorted_clearance[percentile_index]

        if collision_count > collision_tolerance:  # collision
            scores.append(float('inf'))
        elif effective_dist_for_traj != float('inf'):
            if effective_dist_for_traj > comfort_radius:
                scores.append(0.0)
            else:
                max_steps = len(traj)
                decay_factor = (max_steps - closest_step_for_traj) / max_steps
                if effective_dist_for_traj <= safety_radius:
                    base_score = 1.0 / (effective_dist_for_traj + 1e-3)
                else:
                    comfort_span = max(comfort_radius - safety_radius, 1e-3)
                    comfort_ratio = (comfort_radius - effective_dist_for_traj) / comfort_span
                    base_score = 0.05 * comfort_ratio * comfort_ratio
                scores.append(decay_factor * base_score)
        else:
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


# Fixed camera-optical <-> body-frame axis mapping. Assumes the camera is mounted with its
# bore-sight aligned with the robot's forward direction (no additional tilt/roll):
#   body +x (forward) = camera +z (forward)
#   body +y (left)    = camera -x (camera +x is right)
#   body +z (up)      = camera -y (camera +y is down)
# Validated against the bag-derived lidar->camera calibration this replaces: composing this
# with Unitree's factory lidar->base_link extrinsic reproduces that calibration's rotation
# to ~1.5 degrees (its translation was for a different rig/mount and is not used).
R_BODY_TO_CAM_OPTICAL = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])

# Front camera looks +Z (optical). Rear camera is mounted at the butt, same height,
# yawed 180 deg: optical Ry(pi) and translated along -Z by the front/rear baseline.
# Default baseline = 2 * camera_x (front at +camera_x, rear at -camera_x in body +x).
def _make_T_front_cam_to_rear_cam(baseline_m: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    T[:3, 3] = [0.0, 0.0, -float(baseline_m)]
    return T


# === PlanningNode class ===

def _rear_depth_worker_main(baseline_m, stop_event, t_shared, k_shared, origin_shared, ready_shared, occ_queue):
    """Own process + own RMW participant. Kill/join fully drops camera1 DataReader."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    rclpy.init()
    node = Node('planning_rear_depth', use_global_arguments=False)
    bridge = CvBridge()
    T_ext = _make_T_front_cam_to_rear_cam(float(baseline_m))

    def cb(depth_msg):
        if stop_event.is_set():
            return
        if int(ready_shared[0]) == 0 or int(ready_shared[1]) == 0:
            return
        try:
            if depth_msg.encoding in ('16UC1', 'mono16'):
                depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32) / 1000.0
            else:
                depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        except Exception:
            return
        T = np.array(t_shared[:], dtype=np.float64).reshape(4, 4)
        K = np.array(k_shared[:], dtype=np.float64).reshape(3, 3)
        T_rear = T @ T_ext
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        # grid params must match PlanningNode defaults
        grid_shape = (80, 80, 40)
        resolution = 0.1
        origin = np.array(origin_shared[:], dtype=np.float64)
        step = 4
        new_occ = run_raycasting_loopy(
            depth, T_rear, grid_shape, fx, fy, cx, cy, origin, step, resolution,
        )
        try:
            while not occ_queue.empty():
                try:
                    occ_queue.get_nowait()
                except Exception:
                    break
            occ_queue.put_nowait(new_occ.astype(np.float32))
        except Exception:
            pass

    node.create_subscription(Image, '/camera1/camera/depth/image_rect_raw', cb, 10)
    try:
        while not stop_event.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')
        self.robot = B2_CONFIG
        self.get_logger().info(
            f"Robot: {self.robot.name} ({self.robot.shape} {self.robot.length}x{self.robot.width}m, "
            f"cam=({self.robot.camera_x},{self.robot.camera_y}), "
            f"ctrl=({self.robot.control_x},{self.robot.control_y}), "
            f"safety_r={self.robot.safety_radius}m, comfort_r={self.robot.comfort_radius}m)"
        )
        self.bridge = CvBridge()
        self.path_pub = self.create_publisher(Path, '/planning/trajectory_path', 10)
        self.height_map_pub = self.create_publisher(Image, "/planning/height_map", 10)
        self.obstacle_mask_pub = self.create_publisher(OccupancyGrid, '/planning/obstacle_mask', 10)
        self.front_clearance_pub = self.create_publisher(Float32, '/planning/front_clearance', 10)
        self.footprint_pub = self.create_publisher(PointCloud, '/planning/footprint', 10)
        self.occupancy_cloud_pub = self.create_publisher(PointCloud2, '/planning/occupied_voxels', 10)
        self.occupancy_cloud_esdf_pub = self.create_publisher(PointCloud2, '/planning/occupied_voxels_with_esdf', 10)
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/planning/occupancy_grid', 10)
        self.depth_sub = message_filters.Subscriber(self, Image, '/slam/depth')
        self.pose_sub = message_filters.Subscriber(self, Odometry, '/slam/odometry_visual')

        # 'vio' (default) or 'ekf', live-toggled via the same /localization/config topic
        # map_node listens on -- one button switches both nodes together. /slam/odometry_fused
        # (ekf_odom_node) is tracked unconditionally so a switch can substitute it into
        # sync_callback/lidar_sync_callback without touching the depth/lidar TimeSynchronizer
        # wiring above, which stays on /slam/odometry_visual's exact-stamp matching either way.
        # Was previously VIO-only regardless of map_node's own toggle: map_node's published
        # /control/target_pose is in whichever frame is active there, and comparing that
        # against a vio-frame T here silently steered toward the wrong target once map_node
        # switched to ekf (robot spun in place, never getting closer to the goal).
        self.odom_source = 'vio'
        self._latest_ekf_pose = None
        self.ekf_pose_sub = self.create_subscription(
            Odometry, '/slam/odometry_fused', self._ekf_pose_callback, 100)
        _localization_config_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.localization_config_sub = self.create_subscription(
            String, '/localization/config', self.localization_config_callback, _localization_config_qos)

        self.ts = message_filters.TimeSynchronizer([self.depth_sub, self.pose_sub], queue_size=10)
        self.ts.registerCallback(self.sync_callback)
        self.camerainfo_sub = self.create_subscription(CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)

        # Rear Looper (camera1): same model, butt-mounted, looks backward. Occupancy uses the
        # latest front /slam pose composed with a fixed 180-deg optical extrinsic -- not camera1's
        # independent VIO world. Enable/disable and baseline via /planning/config or env.
        self.rear_depth_enabled = os.environ.get('TINYNAV_REAR_DEPTH_OCC', '1') not in ('0', 'false', 'False')
        self.rear_depth_baseline_m = float(
            os.environ.get('TINYNAV_REAR_CAM_BASELINE_M', str(max(0.2, 2.0 * abs(self.robot.camera_x))))
        )
        self.T_front_cam_to_rear_cam = _make_T_front_cam_to_rear_cam(self.rear_depth_baseline_m)
        self._executor = None
        self._occ_lock = threading.Lock()
        # Rear camera1 runs in a child process so mode switches can kill the whole
        # DDS participant (in-process destroy_node/context leave zombie readers).
        self._mp_ctx = mp.get_context('spawn')
        self._rear_t_shared = self._mp_ctx.Array('d', 16, lock=False)
        self._rear_k_shared = self._mp_ctx.Array('d', 9, lock=False)
        self._rear_origin_shared = self._mp_ctx.Array('d', 3, lock=False)
        self._rear_ready_shared = self._mp_ctx.Array('i', 2, lock=False)  # [has_T, has_K]
        self._rear_occ_queue = self._mp_ctx.Queue(maxsize=1)
        self._rear_stop_event = None
        self._rear_proc = None
        self._rear_drain_timer = None
        # Lidar stays in-process on a helper node; exclusive with rear process.
        self._lidar_node = None

        # Lidar-driven occupancy grid input (feeds the same rolling grid as depth, see
        # lidar_sync_callback / _plan_and_publish). Lidar scans arrive at a much lower rate
        # than depth (~10Hz vs 30Hz) and are paired with an independent pose source, so this
        # uses an approximate (slop-tolerant) sync instead of an exact one.
        # T_lidar_to_body (front lidar -> base_link), from Unitree's A2 SDK factory
        # calibration (support.unitree.com A2_SDK_Development_Guide). /slam/odometry_visual's
        # pose is in the camera-optical frame, not base_link, so this composes lidar->body
        # with a camera->body transform (built from the robot's camera_x/camera_y mount
        # offset plus the fixed camera<->body rotation above) to get lidar->camera, letting
        # lidar_sync_callback keep using the same T_cam_to_world @ T_lidar_to_cam chain as
        # the depth path.
        T_lidar_to_body = np.array([
            [0.0, 0.0, 1.0, 0.33767],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.08134],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        T_cam_to_body = np.eye(4)
        T_cam_to_body[:3, :3] = R_BODY_TO_CAM_OPTICAL.T
        T_cam_to_body[:3, 3] = [
            self.robot.camera_x - self.robot.control_x,
            self.robot.camera_y - self.robot.control_y,
            0.0,
        ]
        T_body_to_cam = np.linalg.inv(T_cam_to_body)
        self.T_lidar_to_cam = T_body_to_cam @ T_lidar_to_body
        # Translation replaced with a direct on-robot measurement (lidar relative to
        # camera, in the camera-optical frame: x=right, y=down, z=forward) -- the
        # rotation above is kept as derived/validated, only the offset is overridden.
        self.T_lidar_to_cam[:3, 3] = [0.0, 0.07, -0.02]
        self.lidar_step = 4  # subsample stride; dense Hesai scans are ~115k points/msg
        self.lidar_min_range = 0.2  # metres; drop closer returns as sensor blind-zone noise
        self.lidar_min_votes = 3  # a voxel needs this many same-frame point hits to count as occupied
        self.lidar_min_obstacle_area_cells = 0
        self.lidar_score_percentile = 0.0
        self.lidar_collision_tolerance = 0
        self.lidar_sub = None
        self.lidar_pose_sub = None
        self.lidar_ts = None

        self.grid_shape = (80, 80, 40)
        self.resolution = 0.1
        self.origin = np.array(self.grid_shape) * self.resolution / -2.
        self.step = 4
        self.occupancy_grid = np.zeros(self.grid_shape)
        self._rear_origin_shared[:] = np.asarray(self.origin, dtype=np.float64).tolist()
        self.occupancy_source = 'depth'  # 'depth' or 'lidar' -- exclusive, set live via /planning/config
        self.get_logger().info(f"Planning occupancy_source current: {self.occupancy_source}")
        self.K = None
        self.baseline = None
        self.last_T = None
        self.last_param = (0.0, 0.0) # acc and gyro
        self.obstacle_config = ObstacleConfig()
        planning_config_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.planning_config_sub = self.create_subscription(
            String,
            "/planning/config",
            self.planning_config_callback,
            planning_config_qos,
        )
        self.stamp = None
        self.current_pose = None  # Store the latest pose from odometry
        self.reverse_enter_threshold = 0.30
        self.terrain_mode = "normal"
        self.max_linear_speed = 0.5
        self.only_straight_back = False
        self.reverse_speed = 0.3
        self.reverse_omegas = (0.0, -0.5, 0.5)
        self.trajectory_smooth_weight = 10.0

        self.smoothed_velocity = 0.0
        self._last_avoidance_debug_log_time = 0.0
        self._last_lidar_filter_log_time = 0.0

        self.create_subscription(Odometry, '/control/target_pose', self.target_pose_callback, 10)
        self.target_pose_pub = self.create_publisher(Odometry, '/control/target_pose', 10)
        self.create_subscription(Path, '/mapping/global_plan', self.global_plan_callback, 1)
        self.target_pose = None
        self._global_plan_in_map = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.target_pose_avoid_obstable = False
        self._fallback_target_pose = None
        self._last_override_target_pose = None
        self._last_override_publish_time = 0.0
        self._last_fallback_log_time = 0.0
        self._stuck_anchor_position = None
        self._stuck_anchor_time = None
        self._stuck_timeout_s = 3.0
        self._stuck_move_threshold_m = 0.08
        self._fallback_min_clearance_m = max(self.robot.safety_radius + 0.05, self.robot.comfort_radius + 0.05)
        # Match map_node target lookahead: max_linear_speed * target_pose_dist_factor (~2.0).
        self._fallback_lookahead_factor = 2.0

        self.poi_change_sub = self.create_subscription(Odometry, "/mapping/poi_change", self.poi_change_callback, 10)


    def attach_executor(self, executor):
        self._executor = executor
        self._rear_drain_timer = self.create_timer(0.05, self._drain_rear_occ_queue)
        self._apply_occupancy_subscriptions()

    def _publish_last_T_to_shared(self, T):
        flat = np.asarray(T, dtype=np.float64).reshape(-1)
        self._rear_t_shared[:] = flat.tolist()
        self._rear_ready_shared[0] = 1

    def _publish_K_to_shared(self):
        if self.K is None:
            return
        self._rear_k_shared[:] = np.asarray(self.K, dtype=np.float64).reshape(-1).tolist()
        self._rear_ready_shared[1] = 1

    def _drain_rear_occ_queue(self):
        if self._rear_proc is None or self.occupancy_source != 'depth':
            return
        got = False
        new_occ = None
        try:
            while True:
                new_occ = self._rear_occ_queue.get_nowait()
                got = True
        except Exception:
            pass
        if got and new_occ is not None:
            self._integrate_occupancy(np.asarray(new_occ, dtype=np.float64))

    def _apply_occupancy_subscriptions(self):
        """Exclusive occupancy sensors: lidar XOR (front /slam/depth + rear camera1)."""
        want_lidar = self.occupancy_source == 'lidar'
        if want_lidar:
            self._set_front_depth_listening(False)
            self._set_rear_depth_listening(False)
            self._set_lidar_listening(True)
            return
        self._set_lidar_listening(False)
        self._set_front_depth_listening(True)
        self._set_rear_depth_listening(self.rear_depth_enabled)

    def _set_front_depth_listening(self, enabled):
        """Subscribe /slam/depth only in depth mode so lidar does not pull front camera DDS."""
        if enabled:
            if self.depth_sub is not None:
                return
            self.depth_sub = message_filters.Subscriber(self, Image, '/slam/depth')
            self.pose_sub = message_filters.Subscriber(self, Odometry, '/slam/odometry_visual')
            self.ts = message_filters.TimeSynchronizer([self.depth_sub, self.pose_sub], queue_size=10)
            self.ts.registerCallback(self.sync_callback)
            self.get_logger().info('Subscribed /slam/depth for occupancy_source=depth')
            return
        if self.depth_sub is None and self.pose_sub is None:
            return
        for sub in (self.depth_sub, self.pose_sub):
            if sub is None:
                continue
            try:
                sub.unregister()
            except Exception:
                pass
        self.depth_sub = None
        self.pose_sub = None
        self.ts = None
        self.get_logger().info('Unsubscribed /slam/depth (occupancy_source!=depth)')

    def _set_lidar_listening(self, enabled):
        if enabled:
            if self._lidar_node is not None:
                return
            if self._executor is None:
                return
            self._lidar_node = Node('planning_lidar', use_global_arguments=False)
            self.lidar_sub = message_filters.Subscriber(self._lidar_node, PointCloud2, '/lidar/points')
            self.lidar_pose_sub = message_filters.Subscriber(self._lidar_node, Odometry, '/slam/odometry_visual')
            self.lidar_ts = message_filters.ApproximateTimeSynchronizer(
                [self.lidar_sub, self.lidar_pose_sub], queue_size=10, slop=0.2)
            self.lidar_ts.registerCallback(self.lidar_sync_callback)
            self._executor.add_node(self._lidar_node)
            self.get_logger().info('Subscribed /lidar/points on planning_lidar node')
            return
        if self._lidar_node is None:
            return
        if self.lidar_sub is not None:
            try:
                self.lidar_sub.unregister()
            except Exception:
                pass
            try:
                self.lidar_pose_sub.unregister()
            except Exception:
                pass
        try:
            self._executor.remove_node(self._lidar_node)
        except Exception:
            pass
        try:
            self._lidar_node.destroy_node()
        except Exception:
            pass
        self._lidar_node = None
        self.lidar_sub = None
        self.lidar_pose_sub = None
        self.lidar_ts = None
        self.get_logger().info('Destroyed planning_lidar node (occupancy_source!=lidar)')

    def _set_rear_depth_listening(self, enabled):
        if enabled:
            if self._rear_proc is not None and self._rear_proc.is_alive():
                return
            self._publish_K_to_shared()
            stop_event = self._mp_ctx.Event()
            proc = self._mp_ctx.Process(
                target=_rear_depth_worker_main,
                args=(
                    self.rear_depth_baseline_m,
                    stop_event,
                    self._rear_t_shared,
                    self._rear_k_shared,
                    self._rear_origin_shared,
                    self._rear_ready_shared,
                    self._rear_occ_queue,
                ),
                name='planning_rear_depth',
                daemon=True,
            )
            self._rear_stop_event = stop_event
            self._rear_proc = proc
            proc.start()
            self.get_logger().info(
                'Started planning_rear_depth process for /camera1/camera/depth/image_rect_raw '
                f'pid={proc.pid} baseline={self.rear_depth_baseline_m:.3f}m'
            )
            return
        proc = self._rear_proc
        stop_event = self._rear_stop_event
        if proc is None and stop_event is None:
            return
        if stop_event is not None:
            try:
                stop_event.set()
            except Exception:
                pass
        # Drain + cancel_join_thread first. Otherwise Process.join can hang forever
        # when the child exited with items still buffered on the multiprocessing Queue.
        try:
            while True:
                self._rear_occ_queue.get_nowait()
        except Exception:
            pass
        try:
            self._rear_occ_queue.cancel_join_thread()
        except Exception:
            pass
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
            try:
                proc.join(timeout=1.0)
            except Exception:
                pass
            try:
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1.0)
            except Exception:
                pass
        self._rear_proc = None
        self._rear_stop_event = None
        # Recreate queue so the next depth session does not reuse a cancelled feeder.
        self._rear_occ_queue = self._mp_ctx.Queue(maxsize=1)
        self.get_logger().info('Killed planning_rear_depth process (rear depth disabled)')

    def _ekf_pose_callback(self, msg: Odometry):
        self._latest_ekf_pose = msg

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
                    self.get_logger().info(f"Updated planning odom_source: {old} -> {odom_source}")

    def _active_pose_msg(self, vio_pose_msg: Odometry) -> Odometry:
        """Pose to actually drive local planning/control from: vio_pose_msg (the
        TimeSynchronizer-matched /slam/odometry_visual sample) unless odom_source is
        'ekf' and a fused pose has arrived, matching whichever source map_node is
        using for /control/target_pose right now."""
        if self.odom_source == 'ekf' and self._latest_ekf_pose is not None:
            return self._latest_ekf_pose
        return vio_pose_msg

    def planning_config_callback(self, msg: String):
        try:
            config = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Failed to parse /planning/config: {exc}")
            return
        if not isinstance(config, dict):
            return
        if "dilation_cells" in config:
            try:
                dilation_cells = int(config["dilation_cells"])
            except (TypeError, ValueError):
                self.get_logger().warning(f"Invalid planning dilation_cells: {config.get('dilation_cells')!r}")
            else:
                dilation_cells = max(0, min(20, dilation_cells))
                old = self.obstacle_config.dilation_cells
                self.obstacle_config.dilation_cells = dilation_cells
                if old != dilation_cells:
                    self.get_logger().info(f"Updated planning dilation_cells: {old} -> {dilation_cells}")

        # ros2 topic pub --once /planning/config std_msgs/msg/String "data: '{\"occupancy_source\": \"lidar\"}'"
        if "occupancy_source" in config:
            occupancy_source = config["occupancy_source"]
            if occupancy_source not in ("depth", "lidar"):
                self.get_logger().warning(f"Invalid planning occupancy_source: {occupancy_source!r} (must be 'depth' or 'lidar')")
            else:
                old = self.occupancy_source
                self.occupancy_source = occupancy_source
                if old != occupancy_source:
                    self.get_logger().info(f"Updated planning occupancy_source: {old} -> {occupancy_source}")
                    self._apply_occupancy_subscriptions()

        if "comfort_radius" in config:
            try:
                comfort_radius = float(config["comfort_radius"])
            except (TypeError, ValueError):
                self.get_logger().warning(f"Invalid planning comfort_radius: {config.get('comfort_radius')!r}")
            else:
                comfort_radius = max(self.robot.safety_radius, min(3.0, comfort_radius))
                old = self.robot.comfort_radius
                self.robot.comfort_radius = comfort_radius
                if abs(old - comfort_radius) > 1e-6:
                    self.get_logger().info(f"Updated planning comfort_radius: {old:.2f} -> {comfort_radius:.2f}")

        if "reverse_enter_threshold" in config:
            try:
                reverse_enter_threshold = float(config["reverse_enter_threshold"])
            except (TypeError, ValueError):
                self.get_logger().warning(
                    f"Invalid planning reverse_enter_threshold: {config.get('reverse_enter_threshold')!r}"
                )
            else:
                reverse_enter_threshold = max(0.0, min(2.0, reverse_enter_threshold))
                old = self.reverse_enter_threshold
                self.reverse_enter_threshold = reverse_enter_threshold
                if abs(old - reverse_enter_threshold) > 1e-6:
                    self.get_logger().info(
                        f"Updated planning reverse_enter_threshold: {old:.2f} -> {reverse_enter_threshold:.2f}"
                    )

        if "terrain_mode" in config:
            terrain_mode = str(config["terrain_mode"]).strip().lower()
            if terrain_mode not in ("normal", "stairs"):
                self.get_logger().warning(
                    f"Invalid planning terrain_mode: {config.get('terrain_mode')!r} (must be 'normal' or 'stairs')"
                )
            else:
                old = self.terrain_mode
                self.terrain_mode = terrain_mode
                self._update_reverse_behavior()
                if old != terrain_mode:
                    self.get_logger().info(
                        f"Updated planning terrain_mode: {old} -> {terrain_mode} "
                        f"(only_straight_back={self.only_straight_back}, "
                        f"reverse_speed={self.reverse_speed:.2f}, reverse_omegas={list(self.reverse_omegas)})"
                    )

        if "target_pose_avoid_obstable" in config:
            value = config["target_pose_avoid_obstable"]
            if isinstance(value, bool):
                enabled = value
            elif isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in ("true", "1", "yes", "on"):
                    enabled = True
                elif normalized in ("false", "0", "no", "off"):
                    enabled = False
                else:
                    self.get_logger().warning(
                        f"Invalid planning target_pose_avoid_obstable: {config.get('target_pose_avoid_obstable')!r}"
                    )
                    enabled = self.target_pose_avoid_obstable
            else:
                self.get_logger().warning(
                    f"Invalid planning target_pose_avoid_obstable: {config.get('target_pose_avoid_obstable')!r}"
                )
                enabled = self.target_pose_avoid_obstable
            old = self.target_pose_avoid_obstable
            self.target_pose_avoid_obstable = enabled
            if old != enabled:
                self.get_logger().info(
                    f"Updated planning target_pose_avoid_obstable: {old} -> {enabled}"
                )

        if "only_straight_back" in config:
            value = config["only_straight_back"]
            if isinstance(value, bool):
                only_straight_back = value
            elif isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in ("true", "1", "yes", "on"):
                    only_straight_back = True
                elif normalized in ("false", "0", "no", "off"):
                    only_straight_back = False
                else:
                    self.get_logger().warning(
                        f"Invalid planning only_straight_back: {config.get('only_straight_back')!r}"
                    )
                    only_straight_back = self.only_straight_back
            else:
                self.get_logger().warning(
                    f"Invalid planning only_straight_back: {config.get('only_straight_back')!r}"
                )
                only_straight_back = self.only_straight_back
            old = self.only_straight_back
            self.only_straight_back = only_straight_back
            self._update_reverse_behavior()
            if old != only_straight_back:
                self.get_logger().info(
                    f"Updated planning only_straight_back: {old} -> {only_straight_back} "
                    f"(reverse_speed={self.reverse_speed:.2f}, reverse_omegas={list(self.reverse_omegas)})"
                )

        if "max_linear_speed" in config:
            try:
                max_linear_speed = float(config["max_linear_speed"])
            except (TypeError, ValueError):
                self.get_logger().warning(f"Invalid planning max_linear_speed: {config.get('max_linear_speed')!r}")
            else:
                max_linear_speed = max(0.05, min(1.5, max_linear_speed))
                old = self.max_linear_speed
                self.max_linear_speed = max_linear_speed
                if abs(old - max_linear_speed) > 1e-6:
                    self.get_logger().info(
                        f"Updated planning max_linear_speed: {old:.2f} -> {max_linear_speed:.2f}"
                    )

        if "rear_depth_enabled" in config:
            value = config["rear_depth_enabled"]
            if isinstance(value, bool):
                enabled = value
            elif isinstance(value, str):
                enabled = value.strip().lower() in ("1", "true", "yes", "on")
            else:
                enabled = bool(value)
            old = self.rear_depth_enabled
            self.rear_depth_enabled = enabled
            if old != enabled:
                self.get_logger().info(f"Updated planning rear_depth_enabled: {old} -> {enabled}")
            self._apply_occupancy_subscriptions()

        if "rear_depth_baseline_m" in config:
            try:
                baseline = float(config["rear_depth_baseline_m"])
            except (TypeError, ValueError):
                self.get_logger().warning(
                    f"Invalid planning rear_depth_baseline_m: {config.get('rear_depth_baseline_m')!r}"
                )
            else:
                baseline = max(0.05, min(3.0, baseline))
                old = self.rear_depth_baseline_m
                self.rear_depth_baseline_m = baseline
                self.T_front_cam_to_rear_cam = _make_T_front_cam_to_rear_cam(baseline)
                if abs(old - baseline) > 1e-6:
                    self.get_logger().info(
                        f"Updated planning rear_depth_baseline_m: {old:.3f} -> {baseline:.3f}"
                    )

        if "lidar_min_votes" in config:
            try:
                lidar_min_votes = int(config["lidar_min_votes"])
            except (TypeError, ValueError):
                self.get_logger().warning(f"Invalid planning lidar_min_votes: {config.get('lidar_min_votes')!r}")
            else:
                lidar_min_votes = max(1, min(50, lidar_min_votes))
                old = self.lidar_min_votes
                self.lidar_min_votes = lidar_min_votes
                if old != lidar_min_votes:
                    self.get_logger().info(f"Updated planning lidar_min_votes: {old} -> {lidar_min_votes}")

        if "lidar_min_obstacle_area_cells" in config:
            try:
                min_area = int(config["lidar_min_obstacle_area_cells"])
            except (TypeError, ValueError):
                self.get_logger().warning(
                    f"Invalid planning lidar_min_obstacle_area_cells: {config.get('lidar_min_obstacle_area_cells')!r}"
                )
            else:
                min_area = max(0, min(200, min_area))
                old = self.lidar_min_obstacle_area_cells
                self.lidar_min_obstacle_area_cells = min_area
                if old != min_area:
                    self.get_logger().info(
                        f"Updated planning lidar_min_obstacle_area_cells: {old} -> {min_area}"
                    )

        if "lidar_score_percentile" in config:
            try:
                score_percentile = float(config["lidar_score_percentile"])
            except (TypeError, ValueError):
                self.get_logger().warning(
                    f"Invalid planning lidar_score_percentile: {config.get('lidar_score_percentile')!r}"
                )
            else:
                score_percentile = max(0.0, min(50.0, score_percentile))
                old = self.lidar_score_percentile
                self.lidar_score_percentile = score_percentile
                if abs(old - score_percentile) > 1e-6:
                    self.get_logger().info(
                        f"Updated planning lidar_score_percentile: {old:.1f} -> {score_percentile:.1f}"
                    )

        if "lidar_collision_tolerance" in config:
            try:
                collision_tolerance = int(config["lidar_collision_tolerance"])
            except (TypeError, ValueError):
                self.get_logger().warning(
                    f"Invalid planning lidar_collision_tolerance: {config.get('lidar_collision_tolerance')!r}"
                )
            else:
                collision_tolerance = max(0, min(50, collision_tolerance))
                old = self.lidar_collision_tolerance
                self.lidar_collision_tolerance = collision_tolerance
                if old != collision_tolerance:
                    self.get_logger().info(
                        f"Updated planning lidar_collision_tolerance: {old} -> {collision_tolerance}"
                    )

        if "trajectory_smooth_weight" in config:
            try:
                smooth_weight = float(config["trajectory_smooth_weight"])
            except (TypeError, ValueError):
                self.get_logger().warning(
                    f"Invalid planning trajectory_smooth_weight: {config.get('trajectory_smooth_weight')!r}"
                )
            else:
                smooth_weight = max(0.0, min(100.0, smooth_weight))
                old = self.trajectory_smooth_weight
                self.trajectory_smooth_weight = smooth_weight
                if abs(old - smooth_weight) > 1e-6:
                    self.get_logger().info(
                        f"Updated planning trajectory_smooth_weight: {old:.1f} -> {smooth_weight:.1f}"
                    )

    def _update_reverse_behavior(self):
        if self.terrain_mode == "stairs":
            self.reverse_speed = 0.15
        else:
            self.reverse_speed = 0.3

        if self.terrain_mode == "stairs" or self.only_straight_back:
            self.reverse_omegas = (0.0,)
        else:
            self.reverse_omegas = (0.0, -0.5, 0.5)

    def poi_change_callback(self, msg):
        self.target_pose = None
        self._fallback_target_pose = None
        self._stuck_anchor_position = None
        self._stuck_anchor_time = None

    def global_plan_callback(self, msg):
        if not msg.poses:
            self._global_plan_in_map = None
            return
        self._global_plan_in_map = np.array([
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            for pose in msg.poses
        ], dtype=np.float64)

    def target_pose_callback(self, msg):
        self.target_pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=np.float64)

    def _lookup_global_plan_in_world(self):
        if self._global_plan_in_map is None or len(self._global_plan_in_map) == 0:
            return None
        try:
            tf_msg = self._tf_buffer.lookup_transform('world', 'map', Time())
        except TransformException:
            return None
        q = tf_msg.transform.rotation
        R = quat_to_matrix(np.array([q.x, q.y, q.z, q.w], dtype=np.float64))
        t = np.array([
            tf_msg.transform.translation.x,
            tf_msg.transform.translation.y,
            tf_msg.transform.translation.z,
        ], dtype=np.float64)
        return (R @ self._global_plan_in_map.T).T + t

    def _esdf_clearance_at(self, point, esdf_map):
        point_arr = np.asarray(point, dtype=np.float64)
        origin_arr = np.asarray(self.origin, dtype=np.float64)
        if esdf_map.ndim >= 3:
            idx = np.floor((point_arr[:3] - origin_arr[:3]) / self.resolution).astype(np.int32)
            shape = np.array(esdf_map.shape[:3], dtype=np.int32)
            if np.any(idx < 0) or np.any(idx >= shape):
                return -1.0
            return float(esdf_map[tuple(idx)])
        idx2 = np.floor((point_arr[:2] - origin_arr[:2]) / self.resolution).astype(np.int32)
        shape2 = np.array(esdf_map.shape[:2], dtype=np.int32)
        if np.any(idx2 < 0) or np.any(idx2 >= shape2):
            return -1.0
        return float(esdf_map[idx2[0], idx2[1]])

    def _is_stuck(self, robot_position):
        now = time.monotonic()
        if self._stuck_anchor_position is None:
            self._stuck_anchor_position = robot_position.copy()
            self._stuck_anchor_time = now
            return False
        moved = float(np.linalg.norm(robot_position[:2] - self._stuck_anchor_position[:2]))
        if moved >= self._stuck_move_threshold_m:
            self._stuck_anchor_position = robot_position.copy()
            self._stuck_anchor_time = now
            return False
        return self._stuck_anchor_time is not None and (now - self._stuck_anchor_time) >= self._stuck_timeout_s

    def _closest_point_on_path_xy(self, path, position):
        if len(path) == 1:
            pt = path[0].copy()
            return 0, pt, float(np.linalg.norm(pt[:2] - position[:2]))
        best_index = 0
        best_point = path[0].copy()
        best_distance = np.inf
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            ab = b[:2] - a[:2]
            denom = float(np.dot(ab, ab))
            ratio = 0.0 if denom <= 1e-9 else np.clip(np.dot(position[:2] - a[:2], ab) / denom, 0.0, 1.0)
            point = a + ratio * (b - a)
            distance = float(np.linalg.norm(point[:2] - position[:2]))
            if distance < best_distance:
                best_index = i
                best_point = point.copy()
                best_distance = distance
        return best_index, best_point, best_distance

    def _point_ahead_on_path(self, path, start_position, distance_ahead):
        """Point distance_ahead along global plan from closest projection onto the path."""
        if path is None or len(path) == 0:
            return None
        closest_index, current, _ = self._closest_point_on_path_xy(path, start_position)
        remaining = float(distance_ahead)
        for i in range(closest_index + 1, len(path)):
            nxt = path[i]
            seg = float(np.linalg.norm(nxt[:2] - current[:2]))
            if seg < 1e-9:
                current = nxt
                continue
            if seg >= remaining:
                ratio = remaining / seg
                return current + ratio * (nxt - current)
            remaining -= seg
            current = nxt
        return path[-1].copy()

    def _select_fallback_target(self, robot_position, esdf_map, stuck=False):
        path_world = self._lookup_global_plan_in_world()
        if path_world is None or len(path_world) == 0:
            return None
        lookahead = max(0.5, float(self.max_linear_speed) * float(self._fallback_lookahead_factor))
        return self._point_ahead_on_path(path_world, robot_position, lookahead)

    def _publish_override_target_pose(self, target_pose, stamp):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'world'
        msg.child_frame_id = 'camera'
        msg.pose.pose.position.x = float(target_pose[0])
        msg.pose.pose.position.y = float(target_pose[1])
        msg.pose.pose.position.z = float(target_pose[2])
        msg.pose.pose.orientation.w = 1.0
        self.target_pose_pub.publish(msg)
        self._last_override_target_pose = target_pose.copy()
        self._last_override_publish_time = time.monotonic()

    def _resolve_effective_target_pose(self, T, esdf_map, stamp):
        if self.target_pose is None:
            return None
        if not self.target_pose_avoid_obstable:
            return self.target_pose
        target_clearance = self._esdf_clearance_at(self.target_pose, esdf_map)
        stuck = self._is_stuck(T[:3, 3])
        need_fallback = (target_clearance >= 0.0 and target_clearance < self._fallback_min_clearance_m) or stuck
        if not need_fallback:
            self._fallback_target_pose = None
            return self.target_pose
        fallback = self._select_fallback_target(T[:3, 3], esdf_map, stuck=stuck)
        if fallback is None:
            return self.target_pose
        self._fallback_target_pose = fallback
        now = time.monotonic()
        should_publish = (
            self._last_override_target_pose is None
            or float(np.linalg.norm(fallback[:2] - self._last_override_target_pose[:2])) > 0.05
            or now - self._last_override_publish_time > 1.0
        )
        if should_publish:
            self._publish_override_target_pose(fallback, stamp)
        if now - self._last_fallback_log_time > 0.5:
            self._last_fallback_log_time = now
            self.get_logger().warning(
                f"Fallback target_pose applied: stuck={stuck} target_clearance={target_clearance:.2f} "
                f"lookahead={max(0.5, float(self.max_linear_speed) * float(self._fallback_lookahead_factor)):.2f} "
                f"fallback={fallback.tolist()}"
            )
        return fallback

    def info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self._publish_K_to_shared()
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

    def _front_obstacle_dist(self, T, obstacle_mask, max_dist=0.5):
        """Distance from the robot's front face to the nearest obstacle in the forward corridor.
        Scans start at the front face so the returned value matches physical clearance."""
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

    def _roll_grid_if_needed(self, T):
        with self._occ_lock:
            center = self.origin + np.array(self.grid_shape) * self.resolution / 2
            robot_pos = T[:3, 3]
            delta = robot_pos - center
            if np.linalg.norm(delta) > .1:
                new_center = robot_pos
                new_origin = new_center - np.array(self.grid_shape) * self.resolution / 2
                self.occupancy_grid, self.origin = roll_occupancy_grid(self.occupancy_grid, self.origin, new_origin, self.resolution)
                self._rear_origin_shared[:] = np.asarray(self.origin, dtype=np.float64).tolist()

    def _integrate_occupancy(self, new_occ):
        with self._occ_lock:
            self.occupancy_grid *= 0.994
            self.occupancy_grid += new_occ
            self.occupancy_grid = np.clip(self.occupancy_grid, -0.2, 0.2)
            self.publish_3d_occupancy_cloud(self.occupancy_grid, self.resolution, self.origin)

    @Timer(name="Planning Loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms")
    def sync_callback(self, depth_msg, odom_msg):
        if self.occupancy_source != 'depth':
            return
        if self.K is None:
            return
        odom_msg = self._active_pose_msg(odom_msg)
        with Timer(name='preprocess', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            # The raycaster wants metres. /slam/depth is 32FC1 already in metres,
            # but the camera's own /camera/camera/depth/image_rect_raw is mono16 in
            # millimetres -- cv_bridge converts the dtype and not the unit, so a
            # straight 32FC1 request would hand the raycaster values 1000x too big.
            # Accept either topic by scaling on the declared encoding.
            if depth_msg.encoding in ('16UC1', 'mono16'):
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32) / 1000.0
            else:
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
            self._roll_grid_if_needed(T)
            new_occ = run_raycasting_loopy(depth, T, self.grid_shape, fx, fy, cx, cy, self.origin, self.step, self.resolution)
            self._integrate_occupancy(new_occ)

        self.last_T = T
        self._publish_last_T_to_shared(T)
        self.last_stamp = stamp

        init_q = np.array([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
                            odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w])
        # Keep legacy sensor timestamp in VIO/depth mode; in EKF mode publish with EKF odom stamp
        # so footprint/path follow the same pose-time domain as odom_fused.
        publish_header = odom_msg.header if self.odom_source == 'ekf' else depth_msg.header
        self._plan_and_publish(T, init_q, publish_header)

    @Timer(name="Lidar Planning Loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms")
    def rear_depth_callback(self, depth_msg):
        # Kept for compatibility; camera1 depth is handled in planning_rear_depth process.
        return

    def lidar_sync_callback(self, lidar_msg, pose_msg):
        if self.occupancy_source != 'lidar':
            return
        pose_msg = self._active_pose_msg(pose_msg)
        with Timer(name='lidar preprocess', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            cloud = pc2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)
            if cloud.shape[0] == 0:
                return
            points_lidar = np.stack([cloud['x'], cloud['y'], cloud['z']], axis=-1).astype(np.float64)
            points_lidar = points_lidar[::self.lidar_step]
            # Drop near-range returns inside the sensor's blind zone -- these are noisy
            # self-occlusion hits off the mount/chassis rather than real obstacles.
            points_lidar = points_lidar[np.linalg.norm(points_lidar, axis=1) >= self.lidar_min_range]
            if points_lidar.shape[0] == 0:
                return
            T, _ = msg2np(pose_msg)
            self.last_T = T
            self._publish_last_T_to_shared(T)
            T_lidar_to_world = T @ self.T_lidar_to_cam

        with Timer(name='lidar raycasting', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            self._roll_grid_if_needed(T)
            new_occ = run_raycasting_points_loopy(points_lidar, T_lidar_to_world, self.grid_shape, self.origin, self.resolution, self.lidar_min_votes)
            self._integrate_occupancy(new_occ)

        init_q = np.array([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y,
                            pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w])
        publish_header = pose_msg.header if self.odom_source == 'ekf' else lidar_msg.header
        self._plan_and_publish(T, init_q, publish_header)

    def _plan_and_publish(self, T, init_q, header):
        with Timer(name='obstacle map', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            obstacle_mask = build_obstacle_map(
                self.occupancy_grid, self.origin, self.resolution,
                robot_z=T[2, 3], config=self.obstacle_config,
            )
            if self.occupancy_source == 'lidar' and self.lidar_min_obstacle_area_cells > 1:
                obstacle_count_before = int(np.sum(obstacle_mask))
                obstacle_mask = remove_small_obstacle_components(obstacle_mask, self.lidar_min_obstacle_area_cells)
                obstacle_count_after = int(np.sum(obstacle_mask))
                now_filter_log = time.monotonic()
                if obstacle_count_before != obstacle_count_after and now_filter_log - self._last_lidar_filter_log_time > 1.0:
                    self._last_lidar_filter_log_time = now_filter_log
                    self.get_logger().info(
                        "lidar_obstacle_filter "
                        f"min_area_cells={self.lidar_min_obstacle_area_cells} "
                        f"obstacle_cells={obstacle_count_before}->{obstacle_count_after}"
                    )
            ESDF_map = distance_transform_edt(~obstacle_mask).astype(np.float32) * self.resolution

        with Timer(name='vis', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            self.publish_3d_occupancy_cloud_with_esdf(self.occupancy_grid, ESDF_map, self.resolution, self.origin)
            self.publish_height_map(T[:3,3], ESDF_map, header)
            self.publish_2d_occupancy_grid(ESDF_map, self.origin, self.resolution, header.stamp, z_offset=self.grid_shape[2]*self.resolution/2)
            self.publish_obstacle_mask(obstacle_mask, header.stamp)
            self.publish_footprint(T, header.stamp)

        with Timer(name='traj gen', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            init_p = self.camera_to_robot_center(T)
            trajectories, params = generate_trajectory_library_3d(
                init_p=init_p,
                init_q=init_q,
                vx_max=self.max_linear_speed,
            )
            vocab_trajs, vocab_params = generate_predefined_trajectory_vocabularies(
                init_p=init_p,
                init_q=init_q,
                reverse_speed=self.reverse_speed,
                reverse_omegas=self.reverse_omegas,
            )
            trajectories = np.concatenate([trajectories, vocab_trajs], axis=0)
            params = np.concatenate([params, vocab_params], axis=0)

        with Timer(name='traj score', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            front_len, rear_len, half_w = self.robot.footprint_from_control()
            score_percentile = self.lidar_score_percentile if self.occupancy_source == 'lidar' else 0.0
            collision_tolerance = self.lidar_collision_tolerance if self.occupancy_source == 'lidar' else 0
            scores, occ_points = score_trajectories_by_ESDF(
                trajectories,
                ESDF_map,
                self.origin,
                self.resolution,
                self.robot.safety_radius,
                self.robot.comfort_radius,
                front_len,
                rear_len,
                half_w,
                score_percentile,
                collision_tolerance,
            )
            scores = np.asarray(scores, dtype=np.float64)
            top_k = 100
            top_indices = np.argsort(scores, kind='stable')[:top_k]

        with Timer(name='cc', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            front_clearance = self._front_obstacle_dist(T, obstacle_mask, max_dist=2.0)
            self.front_clearance_pub.publish(Float32(data=float(front_clearance)))
            effective_target_pose = self._resolve_effective_target_pose(T, ESDF_map, header.stamp)
            enter_threshold = self.reverse_enter_threshold
            should_reverse = front_clearance <= enter_threshold
            valid_traj_count = int(np.sum(np.isfinite(scores)))
            all_collision = valid_traj_count == 0
            recovery_indices = np.flatnonzero(params[:, 0] < 0.0)

            def choose_recovery_index():
                if len(recovery_indices) == 0:
                    return 0, "none"
                recovery_scores = scores[recovery_indices]
                finite_mask = np.isfinite(recovery_scores)
                if np.any(finite_mask):
                    finite_indices = recovery_indices[finite_mask]
                    finite_scores = recovery_scores[finite_mask]
                    return int(finite_indices[int(np.argmin(finite_scores))]), "finite"

                ignore_steps = min(3, trajectories.shape[1] - 1)
                delayed_scores, _ = score_trajectories_by_ESDF(
                    trajectories[recovery_indices, ignore_steps:, :],
                    ESDF_map,
                    self.origin,
                    self.resolution,
                    self.robot.safety_radius,
                    self.robot.comfort_radius,
                    front_len,
                    rear_len,
                    half_w,
                    score_percentile,
                    collision_tolerance,
                )
                delayed_scores = np.asarray(delayed_scores, dtype=np.float64)
                delayed_finite_mask = np.isfinite(delayed_scores)
                if np.any(delayed_finite_mask):
                    finite_indices = recovery_indices[delayed_finite_mask]
                    finite_scores = delayed_scores[delayed_finite_mask]
                    return int(finite_indices[int(np.argmin(finite_scores))]), f"delayed_{ignore_steps}"

                straight_reverse = recovery_indices[int(np.argmin(np.abs(params[recovery_indices, 1])))]
                return int(straight_reverse), "fallback_straight"

            def cost_function(traj, param, score, target_pose):
                # predefined backward trajectory penalty
                is_backward_traj = param[0] < 0.0
                reverse_gate_penalty = 0.0
                if should_reverse and not is_backward_traj:
                        reverse_gate_penalty = 1e9
                elif not should_reverse and is_backward_traj:
                        reverse_gate_penalty = 1e9

                # regular trajectory penalty
                traj_end = np.array(traj[-1,:3])
                target_end = target_pose if target_pose is not None else traj_end
                dist = np.linalg.norm(traj_end - target_end)

                return (
                    score * 100
                    + 100 * dist
                    + self.trajectory_smooth_weight * abs(self.last_param[0] - param[0])
                    + self.trajectory_smooth_weight * abs(self.last_param[1] - param[1])
                    + reverse_gate_penalty
                )

            top_k = 1
            recovery_reason = "normal"
            if all_collision:
                selected_index, recovery_reason = choose_recovery_index()
                top_indices = np.array([selected_index], dtype=np.int64)
                costs = np.full(len(trajectories), float("inf"), dtype=np.float64)
                costs[selected_index] = 0.0
            else:
                costs = np.array([cost_function(trajectories[i], params[i], scores[i], effective_target_pose) for i in range(len(trajectories))])
                top_indices = np.argsort(costs, kind='stable')[:top_k]
            selected_index = int(top_indices[0])
            selected_param = params[selected_index]
            selected_score = float(scores[selected_index])
            selected_cost = float(costs[selected_index])
            selected_is_reverse = bool(selected_param[0] < 0.0)
            target_dist = float("nan")
            if effective_target_pose is not None:
                target_dist = float(np.linalg.norm(trajectories[selected_index][-1, :3] - effective_target_pose))
            now_debug = time.monotonic()
            should_log_debug = (
                now_debug - self._last_avoidance_debug_log_time > 0.5
                or all_collision
                or should_reverse
                or selected_is_reverse
            )
            if should_log_debug:
                self._last_avoidance_debug_log_time = now_debug
                self.get_logger().info(
                    "planning_avoidance_debug "
                    f"front_clearance={front_clearance:.2f} enter_threshold={enter_threshold:.2f} "
                    f"should_reverse={should_reverse} all_collision={all_collision} "
                    f"valid_traj_count={valid_traj_count}/{len(trajectories)} "
                    f"recovery_reason={recovery_reason} "
                    f"selected_idx={selected_index} selected_vx={selected_param[0]:.2f} "
                    f"selected_omega={selected_param[1]:.2f} selected_reverse={selected_is_reverse} "
                    f"selected_score={selected_score:.3f} selected_cost={selected_cost:.1f} "
                    f"target_dist={target_dist:.2f} last_vx={self.last_param[0]:.2f} "
                    f"last_omega={self.last_param[1]:.2f} source={self.occupancy_source} "
                    f"dilation_cells={self.obstacle_config.dilation_cells} "
                    f"lidar_min_votes={self.lidar_min_votes} "
                    f"lidar_min_obstacle_area_cells={self.lidar_min_obstacle_area_cells} "
                    f"lidar_score_percentile={self.lidar_score_percentile:.1f} "
                    f"lidar_collision_tolerance={self.lidar_collision_tolerance} "
                    f"trajectory_smooth_weight={self.trajectory_smooth_weight:.1f}"
                )
            self.last_param = selected_param

            # path
            path = Path()
            path.header = header
            path.header.frame_id = "world"

            if effective_target_pose is None:
                return

            if all_collision:
                self.get_logger().warning(
                    "All trajectories in collision; publishing recovery path "
                    f"idx={selected_index} vx={selected_param[0]:.2f} "
                    f"omega={selected_param[1]:.2f} reason={recovery_reason}"
                )

            for i in top_indices:
                for j in range(0, len(trajectories[i]), 10):
                    x,y,z,qx,qy,qz,qw = trajectories[i][j]
                    pose = PoseStamped()
                    pose.header = header
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
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    node.attach_executor(executor)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._set_lidar_listening(False)
        node._set_rear_depth_listening(False)
        try:
            executor.remove_node(node)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
