"""Occupancy-grid + ESDF + trajectory-library planner.

The planner builds a rolling 3D occupancy grid from depth, derives a 2D
obstacle map + ESDF, samples a trajectory library, and selects the trajectory
that minimizes a cost of clearance + distance-to-goal (+ smoothness, + goal
heading for turn-in-place candidates), subject to a hard collision filter and a
reverse gate.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, maximum_filter
from dataclasses import dataclass
from numba import njit
import cv2
import rclpy
import message_filters
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointField, PointCloud2, PointCloud
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point32, Twist
from std_msgs.msg import Header, Float32
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
from codetiming import Timer
from tinynav.core.math_utils import rotvec_to_matrix, quat_to_matrix, matrix_to_quat, msg2np
from tinynav.core.robot_specs import ROBOT_CONFIG

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


@dataclass
class ObstacleConfig:
    robot_z_bottom: float = -0.45
    robot_z_top: float = 0.2
    occ_threshold: float = 0.05
    min_wall_span_m: float = 0.05
    ground_band_m: float = 0.3
    dilation_cells: int = 0


def build_obstacle_map(occupancy_grid, origin, resolution, robot_z, config=None,
                       min_span_map=None):
    """Obstacle = cells where occupied voxels span >= min_wall_span_m in z.
    The span filter only applies to cells whose lowest occupied voxel sits near
    the ground (within ground_band_m of robot_z_bottom): walls have large z-span
    while stair risers / ground bumps have small span. Cells whose occupancy
    starts above that ground band (floating / mid-height obstacles) use a
    single-voxel span threshold (resolution) just to reject single-voxel
    noise, so real low-profile obstacles are still kept.

    `min_span_map` is an optional (h,w) array overriding min_wall_span_m per cell,
    which is how a caller marks somewhere a riser should read as a step rather than
    a wall. Cells it relaxes skip the ground-band gate -- on a staircase the ground
    is the staircase, and the steps ahead sit above the band. Per cell, not a global
    switch, so low obstacles beside a staircase keep blocking while it is being
    climbed; the caller owns what the regions mean."""
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
        z_rel_band = z_rel[z_mask]
        z_idx = np.arange(n_z, dtype=np.float32)
        occ_high = np.where(band_occ, z_idx[np.newaxis, np.newaxis, :], -1).max(axis=2)
        occ_low = np.where(band_occ, z_idx[np.newaxis, np.newaxis, :], n_z).min(axis=2)
        z_span = (occ_high - occ_low) * resolution
        # relative height of the lowest occupied voxel in each cell
        low_z_rel = z_rel_band[np.clip(occ_low, 0, n_z - 1).astype(np.int64)]
        near_ground = low_z_rel <= config.robot_z_bottom + config.ground_band_m
        min_span = config.min_wall_span_m if min_span_map is None else min_span_map
        # A relaxed cell keeps its threshold at any height: mid-climb the steps ahead
        # start above the ground band, so gating on height would read every riser
        # above it as a floating obstacle.
        relaxed = False if min_span_map is None else min_span_map > config.min_wall_span_m
        # ground-anchored (or caller-relaxed) cells: full span filter (wall vs
        # stair/bump); floating cells: single-voxel span filter just to reject noise
        span_ok = np.where(near_ground | relaxed,
                           z_span >= min_span,
                           z_span >= resolution)
        obstacle = has_occ & span_ok

    if config.dilation_cells > 0 and np.any(obstacle):
        obstacle = binary_dilation(obstacle, iterations=config.dilation_cells)
    return obstacle


@njit(cache=True)
def generate_trajectory_library_3d(
    num_samples=15, duration=3.0, dt=0.1,
    init_p=np.zeros(3), init_q=np.array([0, 0, 0, 1]),
    max_linear_vel=0.5, max_angular_vel=np.pi / 3,
    max_path_len_m=1e9, max_lat_acc=1e9,
):
    """Regular sampled lattice (forward-only).

    Two caps shape the lattice beyond the velocity bounds, because `duration` alone
    ties both the planning horizon and the turn rate to whatever speed is allowed:

    `max_path_len_m` caps a trajectory's ARC LENGTH, not its speed. Without it the
    lattice reaches vx*duration -- at 1.34 m/s over 3 s that is a 4 m arc, committing
    the robot to a shape further out than the obstacle map is worth trusting. A
    trajectory that hits the cap freezes in place for its remaining steps, so vx (the
    commanded speed, and the feedforward) is untouched: it still travels fast, it is
    just drawn less far. vx=0 rows never accumulate length, which is what leaves the
    turn-in-place vocabulary at its full `duration` of rotation -- the cost function's
    heading term depends on those rows swinging a real angle.

    `max_lat_acc` caps vx*omega, so the same steering that is fine at a crawl is not
    offered at speed. It binds only above max_lat_acc/max_angular_vel; below that the
    omega range is unchanged, and at a standstill it does not bind at all.
    """
    num_steps = int(duration / dt) + 1

    vx_max = max_linear_vel
    n_vx = max(3, int(num_samples / 2))
    n_omega = num_samples
    vx_samples = np.linspace(0.0, vx_max, n_vx)

    num_samples = n_vx * n_omega

    trajectories = np.empty((num_samples, num_steps, 7))
    params = np.empty((num_samples, 2))

    k = -1
    for i_vx in range(n_vx):
        vx = vx_samples[i_vx]
        # Per-speed omega range: the lattice stays rectangular (n_vx * n_omega rows),
        # only the span of each row's omega shrinks as vx rises.
        omega_lim = max_angular_vel
        if vx > 1e-6 and max_lat_acc / vx < omega_lim:
            omega_lim = max_lat_acc / vx
        omega_y_samples = np.linspace(-omega_lim, omega_lim, n_omega)
        # Nominal arc length per step; 0 for the stationary rows, which is why the
        # length cap cannot touch them.
        step_len = vx * dt
        for i_omega in range(n_omega):
            k += 1
            omega_y = omega_y_samples[i_omega]
            p = init_p.copy()
            q = quat_to_matrix(init_q)
            traj = np.empty((num_steps, 7))
            path_len = 0.0
            for i in range(num_steps):
                if path_len + step_len <= max_path_len_m:
                    dq = rotvec_to_matrix(np.array([0.0, omega_y * dt, 0.0]))
                    q = q @ dq
                    v_world = q @ np.array([0.0, 0.0, vx])
                    p += v_world * dt
                    path_len += step_len
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
def score_trajectories_by_ESDF(trajectories, ESDF_map, origin, resolution, safety_radius=0.1,
                                front_len=0.35, rear_len=0.35, half_w=0.15):
    """Score trajectories by minimum ESDF clearance across the robot footprint (center + 4 corners)."""
    scores = []
    occ_points = []
    ESDF_rows, ESDF_cols = ESDF_map.shape

    for t in range(len(trajectories)):
        traj = trajectories[t]
        min_dist_for_traj = float('inf')
        closest_step_for_traj = -1

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


# === PlanningNode class ===
class PlanningNode(Node):
    """Occupancy-grid + ESDF + trajectory-library planner.

    Cost = clearance + distance-to-goal + smoothness + goal heading (stationary
    candidates only), subject to a hard collision filter and a reverse gate.
    """

    def __init__(self, node_name='planning_node'):
        super().__init__(node_name)
        self.get_logger().info(
            f"Robot: {ROBOT_CONFIG.name} ({ROBOT_CONFIG.shape} {ROBOT_CONFIG.length}x{ROBOT_CONFIG.width}m, "
            f"cam=({ROBOT_CONFIG.camera_x},{ROBOT_CONFIG.camera_y}), "
            f"ctrl=({ROBOT_CONFIG.control_x},{ROBOT_CONFIG.control_y}), "
            f"safety_r={ROBOT_CONFIG.safety_radius}m)"
        )
        self.bridge = CvBridge()
        self.path_pub = self.create_publisher(Path, '/planning/trajectory_path', 10)
        # Instantaneous (vx, omega) feedforward of the selected trajectory. cmd_vel_control
        # consumes this directly instead of reverse-engineering it from path poses.
        # angular.x is a backward-segment flag (fixed-speed reverse vocabulary).
        self.velocity_ff_pub = self.create_publisher(Twist, '/planning/velocity_ff', 10)
        # Open-space forward-speed target (capture-speed prior or vx_max fallback), so
        # cmd_vel_control caps to the same prior-driven ceiling instead of a static one.
        self.forward_speed_cap_pub = self.create_publisher(Float32, '/planning/forward_speed_cap', 10)
        self.height_map_pub = self.create_publisher(Image, "/planning/height_map", 10)
        self.obstacle_mask_pub = self.create_publisher(OccupancyGrid, '/planning/obstacle_mask', 10)
        self.footprint_pub = self.create_publisher(PointCloud, '/planning/footprint', 10)
        self.occupancy_cloud_pub = self.create_publisher(PointCloud2, '/planning/occupied_voxels', 10)
        self.occupancy_cloud_esdf_pub = self.create_publisher(PointCloud2, '/planning/occupied_voxels_with_esdf', 10)
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/planning/occupancy_grid', 10)
        latest_depth_only = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE
        )
        poses_covering_one_depth_frame = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE
        )
        self.depth_sub = message_filters.Subscriber(self, Image, '/slam/depth',
                                                    qos_profile=latest_depth_only)
        self.pose_sub = message_filters.Subscriber(self, Odometry, '/slam/odometry_visual',
                                                   qos_profile=poses_covering_one_depth_frame)

        self.ts = message_filters.TimeSynchronizer([self.depth_sub, self.pose_sub], queue_size=10)
        self.ts.registerCallback(self.sync_callback)
        self.camerainfo_sub = self.create_subscription(CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)

        self.resolution = 0.05
        # Ground-anchored span filter (walls/bumps vs stair risers) now applies
        # everywhere when map_node's `climb_prior` parameter is false (no climb-region
        # relaxation at all) -- so its default is the same single-voxel noise floor as the
        # floating-obstacle check, to avoid filtering out real low/thin obstacles.
        # A ROS parameter, like the climb knobs below, so a site can retune from the
        # launch without a code edit.
        self.declare_parameter('min_wall_span_m', ObstacleConfig.min_wall_span_m)
        self.obstacle_config = ObstacleConfig(
            min_wall_span_m=float(self.get_parameter('min_wall_span_m').value))
        # Derive the grid's z extent and vertical offset from the obstacle band so
        # the grid covers exactly [robot_z_bottom, robot_z_top] relative to the camera.
        z_layers = int(round((self.obstacle_config.robot_z_top - self.obstacle_config.robot_z_bottom) / self.resolution))
        self.grid_shape = (100, 100, z_layers)
        self.z_grid_drop = -(self.obstacle_config.robot_z_top + self.obstacle_config.robot_z_bottom) / 2
        self.origin = np.array(self.grid_shape) * self.resolution / -2.
        self.step = 4
        self._traj_dt = 0.1  # matches generate_trajectory_library_3d / vocab dt

        # --- Speed scaling by forward clearance (with reaction-latency compensation) ---
        # Peak forward speed is modulated per cycle by the free space ahead: the open-space
        # TARGET in tight spots creeps to vx_min (this also subsumes the old openness prior).
        # The open target itself is the capture-speed prior (see _open_target_speed) when
        # available, else vx_max -- so vx_max is the no-capture fallback, NOT the ceiling;
        # the prior may raise the target above it, up to vx_hard_max (hardware absolute).
        # Depth latency (~100ms) + raycast makes the effective clearance smaller than
        # measured, so the schedule discounts it by v*t_react (see _speed_from_clearance).
        self.declare_parameter('vx_max', 0.6)        # open-space target when NO capture prior (fallback)
        self.declare_parameter('vx_hard_max', 1.0)   # absolute forward-speed ceiling (hardware)
        self.declare_parameter('vx_min', 0.2)        # creep speed in tight space (m/s)
        self.declare_parameter('clear_c0_m', 0.35)   # net clearance <= this -> only vx_min
        self.declare_parameter('clear_open_m', 1.0)  # net clearance >= this -> full open target
        self.declare_parameter('clear_scan_m', 2.0)  # forward clearance scan cap (m)
        self.declare_parameter('t_react_s', 0.2)     # perception+plan latency (s)
        # Arc-length cap on a lattice trajectory (m). Bounds how far ahead a plan
        # commits, independently of vx -- see generate_trajectory_library_3d.
        self.declare_parameter('traj_max_len_m', 2.5)
        # Lateral-acceleration cap, vx*omega (m/s^2). Binds only above
        # max_lat_acc/max_angular_vel; below that the omega range is unchanged.
        self.declare_parameter('traj_max_lat_acc', 0.5)
        self._vx_max = float(self.get_parameter('vx_max').value)
        self._vx_hard_max = float(self.get_parameter('vx_hard_max').value)
        self._vx_min = float(self.get_parameter('vx_min').value)
        self._clear_c0_m = float(self.get_parameter('clear_c0_m').value)
        self._clear_open_m = float(self.get_parameter('clear_open_m').value)
        self._clear_scan_m = float(self.get_parameter('clear_scan_m').value)
        self._t_react_s = float(self.get_parameter('t_react_s').value)
        self._traj_max_len_m = float(self.get_parameter('traj_max_len_m').value)
        self._traj_max_lat_acc = float(self.get_parameter('traj_max_lat_acc').value)

        # Collision is checked over the WHOLE 3 s rollout, as upstream does. A
        # receding-horizon "commit" window was tried here -- checking only ~0.8 m ahead
        # so a distant wall could not veto a trajectory whose near segment is clear --
        # but it let the planner commit to trajectories it had not fully vetted, and the
        # freeze it was meant to prevent turned out to have other causes.

        # Fixed-speed reverse fallback: driven when every trajectory is in collision
        # but the blockage is ahead (not already under the footprint).
        self.declare_parameter('reverse_speed_fallback', 0.2)
        self._reverse_speed_fallback = float(self.get_parameter('reverse_speed_fallback').value)

        self.occupancy_grid = np.zeros(self.grid_shape)
        self.K = None
        self.baseline = None
        self.last_param = (0.0, 0.0)  # (vx, omega) of the last selected trajectory

        self.create_subscription(Odometry, '/control/target_pose', self.target_pose_callback, 10)
        self.target_pose = None

        self.poi_change_sub = self.create_subscription(Odometry, "/mapping/poi_change", self.poi_change_callback, 10)

        # Climb region: the capture-path points, in this grid's frame, that the map
        # says were climbed through. Cells near them relax the obstacle z-span filter
        # so a riser reads as a step, and only those cells do -- everything beside the
        # staircase keeps the strict default. Producer: core_runtime's PilotMapNode.
        # No message, or a stale stream, means no region, i.e. strict everywhere.
        # Both radius and span are deliberately small: a relaxed cell cannot see an
        # obstacle shorter than the span, and the labels that open these regions are
        # inferred from path z, which VIO drift fakes -- a generous radius relaxed most
        # of a route on the strength of a handful of real runs. A ROS parameter, so a
        # site can be retuned from the launch without a code edit.
        self.declare_parameter('climb_region_radius_m', 0.75)
        self._climb_region_cells = int(round(
            float(self.get_parameter('climb_region_radius_m').value) / self.resolution))
        self.declare_parameter('climb_region_ttl_s', 3.0)
        self._climb_region_ttl_ns = int(float(self.get_parameter('climb_region_ttl_s').value) * 1e9)
        # 0.2 keeps a ~0.15m riser reading as a step while a 0.2m+ obstacle survives.
        self.declare_parameter('climb_min_wall_span_m', 0.2)
        self._climb_min_wall_span_m = float(self.get_parameter('climb_min_wall_span_m').value)
        self._climb_points = np.empty((0, 2))
        self._climb_stamp_ns = None
        self.create_subscription(PointCloud, '/planning/climb_region',
                                 self.climb_region_callback, 10)

        # Capture-speed prior from map_node's /planning/speed_cap: the operator's
        # local speed (m/s) near the robot. It IS the open-space target speed (scaled
        # by capture_speed_gain), clamped to [vx_min, vx_hard_max] -- so it may raise
        # the target above vx_max where the operator went fast, never past the hardware
        # ceiling. NaN (off-path / unknown) or a stale stream -> fall back to vx_max.
        # Gain 1.0: replay the capture speed as driven. It was >1 on the theory that
        # capture is deliberately slow for mapping stability and replay can afford to
        # be quicker, but the operator's speed is already the best evidence of what
        # this stretch tolerates, so scaling it up just overdrives the tight parts.
        self.declare_parameter('capture_speed_gain', 1.0)
        self._capture_speed_gain = float(self.get_parameter('capture_speed_gain').value)
        self.declare_parameter('speed_cap_ttl_s', 2.0)
        self._speed_cap_ttl_ns = int(float(self.get_parameter('speed_cap_ttl_s').value) * 1e9)
        self._speed_cap = None
        self._speed_cap_stamp_ns = None
        self.create_subscription(Float32, '/planning/speed_cap', self.speed_cap_callback, 10)

    # --- callbacks ---------------------------------------------------------
    def climb_region_callback(self, msg):
        # An empty cloud is a real answer ("no region here"), not a missed message:
        # only the stamp decides freshness.
        self._climb_points = np.array(
            [[p.x, p.y] for p in msg.points], dtype=np.float64).reshape(-1, 2)
        self._climb_stamp_ns = self.get_clock().now().nanoseconds

    def _min_span_map(self, origin, resolution, shape):
        """Per-cell min_wall_span_m for build_obstacle_map, or None for the strict
        default everywhere -- which is also what a missing or stale region gives.

        Cells within climb_region_radius_m of a climb point get the relaxed
        threshold. That radius is the whole of the look-ahead: the labelling window
        already extends the region ~1m back along the path."""
        if not self._signal_fresh(self._climb_stamp_ns, self._climb_region_ttl_ns):
            return None
        pts = self._climb_points
        if not len(pts):
            return None
        seeds = np.zeros(shape, dtype=bool)
        idx = np.floor((pts - origin[:2]) / resolution).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
        seeds[idx[inside, 0], idx[inside, 1]] = True
        if not seeds.any():
            return None
        # Grow each seed into a square of the radius. Cost is independent of how
        # many points came in, unlike a per-point distance test.
        region = maximum_filter(seeds, size=2 * self._climb_region_cells + 1,
                                mode='constant')
        return np.where(region, self._climb_min_wall_span_m,
                        self.obstacle_config.min_wall_span_m)

    def speed_cap_callback(self, msg):
        self._speed_cap = float(msg.data)
        self._speed_cap_stamp_ns = self.get_clock().now().nanoseconds

    def _open_target_speed(self):
        """Open-space target forward speed: the capture-speed prior (scaled by
        capture_speed_gain) when a fresh, finite value is available, else vx_max
        (the no-capture fallback). Clamped to [vx_min, vx_hard_max] -- the prior may
        raise the target above vx_max but never past the hardware ceiling."""
        # _signal_fresh short-circuits on a never-received (None) stamp, so a fresh
        # stamp implies _speed_cap was set -> the isfinite guard is safe.
        if (self._signal_fresh(self._speed_cap_stamp_ns, self._speed_cap_ttl_ns)
                and np.isfinite(self._speed_cap)):
            return float(np.clip(self._speed_cap * self._capture_speed_gain,
                                 self._vx_min, self._vx_hard_max))
        return self._vx_max

    def poi_change_callback(self, msg):
        self.target_pose = None

    def target_pose_callback(self, msg):
        self.target_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def _signal_fresh(self, stamp_ns, window_ns):
        """True if a signal last stamped at stamp_ns is still within window_ns of
        now. Never-received (stamp_ns None) -> stale, the safe default."""
        if stamp_ns is None:
            return False
        return self.get_clock().now().nanoseconds - stamp_ns <= window_ns

    def _speed_from_clearance(self, clearance_m, v_prev, v_open):
        """Linear peak-speed schedule from forward clearance, discounted for reaction
        latency (the robot travels ~v_prev*t_react before a new command takes effect).
        net clearance <= clear_c0_m -> vx_min; >= clear_open_m -> v_open; linear between.
        v_open is the open-space target (capture-speed prior or vx_max fallback)."""
        c_eff = max(0.0, clearance_m - v_prev * self._t_react_s)
        # np.interp saturates to the endpoints outside [clear_c0_m, clear_open_m].
        return float(np.interp(c_eff, [self._clear_c0_m, self._clear_open_m],
                               [self._vx_min, v_open]))

    def _publish_velocity_ff(self, vx, omega):
        """Publish a (vx, omega) feedforward on /planning/velocity_ff. angular.x
        carries the fixed-speed reverse flag, derived from the sign of vx here so the
        reverse-vocabulary convention cmd_vel_control consumes lives in one place."""
        ff = Twist()
        ff.linear.x = vx
        ff.angular.z = omega
        ff.angular.x = 1.0 if vx < 0.0 else 0.0
        self.velocity_ff_pub.publish(ff)

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
        return T[:3, 3] - T[:3, :3] @ ROBOT_CONFIG.cam_offset_3d

    def publish_footprint(self, T, stamp):
        """Publish robot footprint rectangle as a PointCloud for RViz."""
        forward = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        left    = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
        center  = self.camera_to_robot_center(T)
        fl, rl, hw = ROBOT_CONFIG.footprint_from_control()
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
        fl, _, hw = ROBOT_CONFIG.footprint_from_control()
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
            T,_ = msg2np(odom_msg)
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]

        with Timer(name='raycasting', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            center = self.origin + np.array(self.grid_shape) * self.resolution / 2
            robot_pos = T[:3, 3]
            target_center = robot_pos - np.array([0.0, 0.0, self.z_grid_drop])
            delta = target_center - center
            if np.linalg.norm(delta) > .1:
                new_center = target_center
                new_origin = new_center - np.array(self.grid_shape) * self.resolution / 2
                self.occupancy_grid, self.origin = roll_occupancy_grid(self.occupancy_grid, self.origin, new_origin, self.resolution)
            new_occ = run_raycasting_loopy(depth, T, self.grid_shape, fx, fy, cx, cy, self.origin, self.step, self.resolution)
            self.occupancy_grid *= 0.995
            self.occupancy_grid += new_occ
            self.occupancy_grid = np.clip(self.occupancy_grid, -0.2, 0.2)

            self.publish_3d_occupancy_cloud(self.occupancy_grid, self.resolution, self.origin)

        with Timer(name='obstacle map', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            min_span_map = self._min_span_map(self.origin, self.resolution,
                                              self.occupancy_grid.shape[:2])
            obstacle_mask = build_obstacle_map(
                self.occupancy_grid, self.origin, self.resolution,
                robot_z=T[2, 3], config=self.obstacle_config, min_span_map=min_span_map,
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
            # Forward clearance drives both the peak-speed schedule and the reverse gate.
            front_clearance = self._front_obstacle_dist(T, obstacle_mask, max_dist=self._clear_scan_m)
            v_open = self._open_target_speed()
            v_allow = self._speed_from_clearance(front_clearance, abs(float(self.last_param[0])), v_open)
            # Publish the prior-driven open target so cmd_vel caps to the same ceiling.
            self.forward_speed_cap_pub.publish(Float32(data=float(v_open)))
            trajectories, params = generate_trajectory_library_3d(
                init_p=init_p, init_q=init_q,
                max_linear_vel=v_allow,
                max_angular_vel=ROBOT_CONFIG.max_angular_vel,
                max_path_len_m=self._traj_max_len_m,
                max_lat_acc=self._traj_max_lat_acc,
            )
            vocab_trajs, vocab_params = generate_predefined_trajectory_vocabularies(init_p=init_p, init_q=init_q)
            trajectories = np.concatenate([trajectories, vocab_trajs], axis=0)
            params = np.concatenate([params, vocab_params], axis=0)

        with Timer(name='traj score', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            front_len, rear_len, half_w = ROBOT_CONFIG.footprint_from_control()
            safety_radius = ROBOT_CONFIG.safety_radius
            scores, occ_points = score_trajectories_by_ESDF(trajectories, ESDF_map, self.origin, self.resolution, safety_radius, front_len, rear_len, half_w)

        with Timer(name='pub', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            enter_threshold = 0.30
            should_reverse = front_clearance <= enter_threshold

            if self.target_pose is None:
                return

            target = self.target_pose

            # World heading (rad) of a trajectory pose, body +Z-forward convention
            # (same as score_trajectories_by_ESDF and the published Path). Used by the
            # velocity feedforward to derive omega from the trajectory's own poses.
            def _world_heading(pose7):
                qx, qy, qz, qw = pose7[3], pose7[4], pose7[5], pose7[6]
                return np.arctan2(2.0 * (qy * qz - qw * qx), 2.0 * (qx * qz + qw * qy))

            def _end_heading_error(pose7, goal):
                """|wrapped angle| between the pose's heading and the bearing from that
                pose to the goal."""
                err = np.arctan2(goal[1] - pose7[1], goal[0] - pose7[0]) - _world_heading(pose7)
                return abs(np.arctan2(np.sin(err), np.cos(err)))

            if all(s == float('inf') for s in scores):
                # Diagnose WHERE it collides: is the robot's own footprint cell already
                # an obstacle (phantom ground/self), or is it genuinely walled in?
                center = self.camera_to_robot_center(T)
                cxi = int((center[0] - self.origin[0]) / self.resolution)
                cyi = int((center[1] - self.origin[1]) / self.resolution)
                rows, cols = obstacle_mask.shape
                center_obst = (0 <= cxi < rows and 0 <= cyi < cols and obstacle_mask[cxi, cyi])
                self.get_logger().warn(
                    f'All trajectories in collision. obst_cells={int(obstacle_mask.sum())} '
                    f'center_cell_obstacle={center_obst} front_clearance={front_clearance:.2f} '
                    f'ESDF@center={ESDF_map[cxi, cyi] if (0<=cxi<rows and 0<=cyi<cols) else -1:.2f} '
                    f'should_reverse={should_reverse}'
                )
                # Fallback: if the blockage is ahead (not already under the footprint),
                # back out slowly instead of freezing so the next cycle can re-plan.
                # When the footprint cell is itself an obstacle (phantom ground/self)
                # we stay put rather than reversing blindly into noise.
                if should_reverse and not center_obst:
                    self._publish_velocity_ff(-self._reverse_speed_fallback, 0.0)
                return

            # Single cost, as upstream: clearance + distance-to-goal + smoothness, with
            # the reverse gate as a large additive penalty rather than a hard filter.
            # The penalty degrades gracefully on its own -- a colliding trajectory costs
            # scores[i]*100000 == inf, which loses to any non-colliding gate violator --
            # so it already gives the "never stall outright" fallback that a two-stage
            # filter had to spell out, in one term. Clearance stays soft on purpose:
            # safety_radius is a margin, not a collision boundary, and a corridor
            # narrower than the band forces every forward trajectory to intrude into it.
            # The heading term exists because the lattice's vx=0 rows all END where they
            # started: distance-to-goal cannot tell them apart, so smoothness picked
            # whichever rotation matched last cycle -- omega=0 from a standstill. With a
            # goal further behind than a 3 s arc can swing around (past ~165 deg, since a
            # half-turn at the tightest radius only moves the endpoint sideways), standing
            # still was the cost minimum and stayed it: a permanent freeze. Scoring the
            # stationary rows by their end heading vs the goal bearing ranks the rotations
            # and prices standing-still-while-misaligned above turning to face the goal;
            # the forward arcs take over on their own once the robot has come around. Only
            # the stationary rows get it -- a moving trajectory's endpoint already says
            # where the arc went, and penalizing its heading would fight the distance term.
            def cost_function(i):
                traj, param = trajectories[i], params[i]
                reverse_gate_penalty = 0.0 if (param[0] < 0.0) == should_reverse else 1e9
                traj_end = np.array(traj[-1, :3])
                target_end = target if target is not None else traj_end
                dist = np.linalg.norm(traj_end - target_end)
                smooth = abs(self.last_param[0] - param[0]) + abs(self.last_param[1] - param[1])
                # Stationary candidates only: skipped once within 0.3 m of the goal, where
                # the bearing is noise and rotating achieves nothing.
                if abs(param[0]) <= 1e-3 and dist > 0.3:
                    heading_penalty = 60 * _end_heading_error(traj[-1], target_end)
                else:
                    heading_penalty = 0.0
                return (scores[i] * 100000
                        + 100 * dist
                        + 10 * smooth
                        + heading_penalty
                        + reverse_gate_penalty)

            top_indices = [min(range(len(trajectories)), key=cost_function)]
            self.last_param = params[top_indices[0]]

            # Confirm what actually got selected: if vx≈0 while non-colliding forward
            # trajectories exist, the robot is "stuck by cost", not by collision.
            n_fwd_ok = sum(1 for i in range(len(trajectories))
                           if params[i][0] > 1e-3 and scores[i] != float('inf'))
            self.get_logger().info(
                f'sel vx={params[top_indices[0]][0]:.2f} omega={params[top_indices[0]][1]:.2f} '
                f'fwd_ok={n_fwd_ok} '
                f'goal_err={np.rad2deg(_end_heading_error(trajectories[top_indices[0]][0], target)):.0f}deg '
                f'v_allow={v_allow:.2f} front_clr={front_clearance:.2f} '
                f'should_reverse={should_reverse} '
                f'climb_cells={0 if min_span_map is None else int((min_span_map > self.obstacle_config.min_wall_span_m).sum())}'
            )

            # velocity feedforward for cmd_vel_control: (vx, omega) of the selected
            # trajectory. vx is the commanded body-forward speed (lattice param; its
            # sign flags the fixed-speed reverse vocabulary via angular.x). omega is
            # NOT taken from the lattice param -- that omega is about the camera optical
            # axis and would need a hand-maintained sign/frame correction. Instead we
            # derive the yaw rate straight from the trajectory's own world poses, using
            # the same body-+z-forward convention as score_trajectories_by_ESDF and the
            # published Path: angular.z = d(world heading)/dt over the first step. This
            # stays consistent with the path by construction and is correct even if the
            # camera pitches (where -omega_y would be subtly wrong).
            sel_traj = trajectories[top_indices[0]]
            sel_vx = float(params[top_indices[0]][0])

            dh = _world_heading(sel_traj[1]) - _world_heading(sel_traj[0])
            sel_omega = float(np.arctan2(np.sin(dh), np.cos(dh)) / self._traj_dt)

            self._publish_velocity_ff(sel_vx, sel_omega)

            # path
            path = Path()
            path.header = depth_msg.header
            path.header.frame_id = "world"

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
