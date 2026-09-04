import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointField, CompressedImage
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from cv_bridge import CvBridge
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.spatial.transform import Rotation as R
from numba import njit
import message_filters
from rclpy.time import Time
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointCloud
from geometry_msgs.msg import PoseStamped, Point32
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from codetiming import Timer
from collections import deque
import cv2
import asyncio

from tinynav.core.math_utils import rotvec_to_matrix, quat_to_matrix, matrix_to_quat, msg2np, tf2np
from tinynav.core.robot_specs import ROBOT_CONFIG, ObstacleConfig
# YoloDetectorTRT (tinynav.core.models_trt) is imported lazily inside
# PlanningNode._get_detector, not here: models_trt imports tensorrt/cuda,
# which aren't available on every machine that needs to import this module
# (e.g. offline unit tests for the pure functions below).

# Fixed BGR palette so a given COCO class id always renders the same color in RViz.
_OBJECT_CLASS_PALETTE = np.array([
    [(37 * i) % 256, (17 + 91 * i) % 256, (53 + 149 * i) % 256] for i in range(80)
], dtype=np.uint8)

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

@njit(cache=True)
def generate_trajectory_library_3d(
    num_samples=15, duration=3.0, dt=0.1,
    init_p=np.zeros(3), init_q=np.array([0, 0, 0, 1]),
    max_linear_vel=0.5, max_angular_vel=np.pi / 3,
):
    """Regular sampled lattice (forward-only)."""
    num_steps = int(duration / dt) + 1

    vx_max = max_linear_vel
    n_vx = max(3, int(num_samples / 2))
    vx_samples = np.linspace(0.0, vx_max, n_vx)
    omega_y_samples = np.linspace(-max_angular_vel, max_angular_vel, num_samples)

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

def goal_heading_error(traj_end, target):
    """Absolute yaw error between a trajectory's end heading and the bearing to the target."""
    dx = target[0] - traj_end[0]
    dy = target[1] - traj_end[1]
    yaw1 = R.from_quat(traj_end[3:7]).as_euler("xyz")[2] + np.pi / 2
    yaw2 = np.arctan2(dy, dx)
    return abs(np.arctan2(np.sin(yaw2 - yaw1), np.cos(yaw2 - yaw1)))

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


def roll_object_grids(class_grid, ttl_grid, old_origin, new_origin, resolution):
    """Roll the object-class/TTL grids in lockstep with the occupancy grid.

    Same shift math as roll_occupancy_grid, but the newly-exposed slab is
    filled with "no object" (-1 for class, 0 for TTL) instead of 0/0.
    """
    shift_m = new_origin - old_origin
    shift_voxels = np.round(shift_m / resolution).astype(int)
    if np.all(shift_voxels == 0):
        return class_grid, ttl_grid, old_origin

    rolled_class = np.roll(class_grid, shift=tuple(-shift_voxels), axis=(0, 1, 2))
    rolled_ttl = np.roll(ttl_grid, shift=tuple(-shift_voxels), axis=(0, 1, 2))
    for axis, shift in enumerate(shift_voxels):
        if shift == 0:
            continue
        idx = [slice(None)] * 3
        idx[axis] = slice(-shift, None) if shift > 0 else slice(None, -shift)
        rolled_class[tuple(idx)] = -1
        rolled_ttl[tuple(idx)] = 0

    updated_origin = old_origin + shift_voxels * resolution
    return rolled_class, rolled_ttl, updated_origin


def label_occupied_column(occupancy_grid, class_id, x, y, origin, resolution, occ_threshold):
    """Tag whichever voxels are ALREADY marked occupied in occupancy_grid's
    (x, y) column with class_id, instead of inventing new geometry.

    Returns an (N, 4) int array of (vx, vy, vz, class_id) hits; empty if the
    (x, y) column falls outside the grid or has no occupied cells.
    """
    grid_shape = occupancy_grid.shape
    vx = int(np.floor((x - origin[0]) / resolution))
    vy = int(np.floor((y - origin[1]) / resolution))
    if not (0 <= vx < grid_shape[0] and 0 <= vy < grid_shape[1]):
        return np.empty((0, 4), dtype=np.int32)

    z_indices = np.nonzero(occupancy_grid[vx, vy, :] > occ_threshold)[0]
    if z_indices.size == 0:
        return np.empty((0, 4), dtype=np.int32)

    hits = np.zeros((z_indices.size, 4), dtype=np.int32)
    hits[:, 0] = vx
    hits[:, 1] = vy
    hits[:, 2] = z_indices
    hits[:, 3] = int(class_id)
    return hits


def store_transform(tf_edges, parent_frame, child_frame, T):
    """Record a TF edge (and its inverse) in an adjacency dict of world-fixed transforms."""
    tf_edges.setdefault(parent_frame, {})[child_frame] = T.astype(np.float32, copy=False)
    tf_edges.setdefault(child_frame, {})[parent_frame] = np.linalg.inv(T).astype(np.float32, copy=False)


def lookup_transform(tf_edges, source_frame, target_frame):
    """BFS a chain of stored TF edges from source_frame to target_frame. None if unresolved."""
    if source_frame == target_frame:
        return np.eye(4, dtype=np.float32)
    if source_frame not in tf_edges or target_frame not in tf_edges:
        return None

    queue = deque([(source_frame, np.eye(4, dtype=np.float32))])
    visited = {source_frame}
    while queue:
        frame, T_source_frame = queue.popleft()
        for next_frame, T_frame_next in tf_edges.get(frame, {}).items():
            if next_frame in visited:
                continue
            T_source_next = T_source_frame @ T_frame_next
            if next_frame == target_frame:
                return T_source_next.astype(np.float32, copy=False)
            visited.add(next_frame)
            queue.append((next_frame, T_source_next))
    return None


def project_color_detections_to_voxels(
    detections, depth, T_cam_to_world, depth_K, color_K, T_depth_color, color_shape,
    occupancy_grid, origin, resolution, step=10, occ_threshold=0.1,
):
    """Project detections made on the COLOR image into world-frame voxels.

    detections are 2D boxes in COLOR-image pixel coordinates. Every depth
    pixel is backprojected and reprojected into the color image via
    T_depth_color (the mirror of global_pointcloud_publisher.depth_to_color_cloud's
    depth->color coloring step) to find which detection box, if any, it falls
    inside. Each detection's matched depth pixels then anchor a lookup into
    occupancy_grid's real occupied column (label_occupied_column), exactly as
    for the non-color path.

    Returns an (N, 4) int array of (vx, vy, vz, class_id) hits.
    """
    if not detections:
        return np.empty((0, 4), dtype=np.int32)

    depth_h, depth_w = depth.shape
    fx, fy = depth_K[0, 0], depth_K[1, 1]
    cx, cy = depth_K[0, 2], depth_K[1, 2]
    color_h, color_w = color_shape[:2]

    v_grid, u_grid = np.mgrid[0:depth_h:step, 0:depth_w:step]
    d = depth[v_grid, u_grid]
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.empty((0, 4), dtype=np.int32)
    u = u_grid[valid]
    v = v_grid[valid]
    d = d[valid]

    px = (u - cx) * d / fx
    py = (v - cy) * d / fy
    points_depth = np.stack([px, py, d], axis=1)

    color_points = points_depth @ T_depth_color[:3, :3].T + T_depth_color[:3, 3]
    in_front = color_points[:, 2] > 1e-6
    if not np.any(in_front):
        return np.empty((0, 4), dtype=np.int32)
    points_depth = points_depth[in_front]
    color_points = color_points[in_front]

    color_u = color_K[0, 0] * color_points[:, 0] / color_points[:, 2] + color_K[0, 2]
    color_v = color_K[1, 1] * color_points[:, 1] / color_points[:, 2] + color_K[1, 2]
    in_bounds = (color_u >= 0) & (color_u < color_w) & (color_v >= 0) & (color_v < color_h)
    if not np.any(in_bounds):
        return np.empty((0, 4), dtype=np.int32)
    points_depth = points_depth[in_bounds]
    color_u = color_u[in_bounds]
    color_v = color_v[in_bounds]

    points_cam = np.concatenate([points_depth, np.ones((points_depth.shape[0], 1))], axis=1)
    points_world = points_cam @ T_cam_to_world.T

    hits = []
    for class_id, _score, x1, y1, x2, y2 in detections:
        in_box = (color_u >= x1) & (color_u < x2) & (color_v >= y1) & (color_v < y2)
        if not np.any(in_box):
            continue
        anchor_xy = np.median(points_world[in_box, :2], axis=0)
        column_hits = label_occupied_column(occupancy_grid, class_id, anchor_xy[0], anchor_xy[1], origin, resolution, occ_threshold)
        if column_hits.shape[0] == 0:
            continue
        hits.append(column_hits)

    if not hits:
        return np.empty((0, 4), dtype=np.int32)
    return np.concatenate(hits, axis=0)


def apply_object_hits(class_grid, ttl_grid, hits, ttl_frames):
    """Write class_id + refresh TTL at each hit voxel. Mutates grids in place."""
    if hits.shape[0] == 0:
        return
    vx, vy, vz, class_id = hits[:, 0], hits[:, 1], hits[:, 2], hits[:, 3]
    class_grid[vx, vy, vz] = class_id
    ttl_grid[vx, vy, vz] = ttl_frames


def decay_object_grids(class_grid, ttl_grid):
    """Count TTL down by one frame; clear the class label where TTL expires. Mutates in place."""
    ttl_grid -= 1
    expired = ttl_grid <= 0
    class_grid[expired] = -1
    ttl_grid[expired] = 0


# === PlanningNode class ===
class PlanningNode(Node):
    def __init__(self):
        super().__init__('planning_node')
        self.get_logger().info(
            f"Robot: {ROBOT_CONFIG.name} ({ROBOT_CONFIG.shape} {ROBOT_CONFIG.length}x{ROBOT_CONFIG.width}m, "
            f"cam=({ROBOT_CONFIG.camera_x},{ROBOT_CONFIG.camera_y}), "
            f"ctrl=({ROBOT_CONFIG.control_x},{ROBOT_CONFIG.control_y}), "
            f"safety_r={ROBOT_CONFIG.safety_radius}m, "
            f"z_band=[{ROBOT_CONFIG.obstacle.robot_z_bottom}, {ROBOT_CONFIG.obstacle.robot_z_top}]m)"
        )
        self.bridge = CvBridge()
        self.path_pub = self.create_publisher(Path, '/planning/trajectory_path', 10)
        self.height_map_pub = self.create_publisher(Image, "/planning/height_map", 10)
        self.obstacle_mask_pub = self.create_publisher(OccupancyGrid, '/planning/obstacle_mask', 10)
        self.footprint_pub = self.create_publisher(PointCloud, '/planning/footprint', 10)
        self.occupancy_cloud_pub = self.create_publisher(PointCloud2, '/planning/occupied_voxels', 10)
        self.occupancy_cloud_esdf_pub = self.create_publisher(PointCloud2, '/planning/occupied_voxels_with_esdf', 10)
        self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/planning/occupancy_grid', 10)
        self.object_voxel_pub = self.create_publisher(PointCloud2, '/planning/object_voxels', 10)
        self.depth_sub = message_filters.Subscriber(self, Image, '/slam/depth')
        self.pose_sub = message_filters.Subscriber(self, Odometry, '/slam/odometry_visual')

        self.ts = message_filters.TimeSynchronizer([self.depth_sub, self.pose_sub], queue_size=10)
        self.ts.registerCallback(self.sync_callback)
        self.camerainfo_sub = self.create_subscription(CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)

        # Object detection runs on the color image (RGB, matches what the COCO
        # detector was trained on) instead of infra1 grayscale/IR, which was
        # found to misclassify real objects (e.g. a person as "oven"). Detected
        # boxes are then reprojected from color-image space into the depth
        # (infra1) frame via the TF-derived depth->color extrinsic — the mirror
        # of tool/global_pointcloud_publisher.py's depth_to_color_cloud, which
        # goes the other way (depth pixel -> color pixel) to sample color.
        self.color_K = None
        self.color_frame_id = None
        self.depth_frame_id = None
        self.T_depth_color = None
        self.tf_edges = {}
        self.latest_color_image = None
        self._logged_color_tf_wait = False
        self.depth_frame_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/infra1/camera_info', self.depth_frame_info_callback, 10,
        )
        self.color_camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.color_camera_info_callback, 10,
        )
        tf_static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.tf_static_sub = self.create_subscription(TFMessage, '/tf_static', self.tf_static_callback, tf_static_qos)
        # Subscribe to both known color topic conventions (RealSense raw vs.
        # Looper compressed, see app/backend/node_manager.py's _COLOR_TOPIC_*)
        # and just use whichever one is actually publishing.
        self.color_image_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_image_callback, 10,
        )
        self.color_image_compressed_sub = self.create_subscription(
            CompressedImage, '/camera/camera/color/image_rect_raw/compressed', self.color_image_compressed_callback, 10,
        )

        self.grid_shape = (100, 100, 10)
        self.resolution = 0.05
        self.origin = np.array(self.grid_shape) * self.resolution / -2.
        self.step = 10
        self.occupancy_grid = np.zeros(self.grid_shape)
        self.object_class_grid = np.full(self.grid_shape, -1, dtype=np.int16)
        self.object_ttl_grid = np.zeros(self.grid_shape, dtype=np.int16)
        self.object_detection_config = ROBOT_CONFIG.object_detection
        self.detector = None
        self.kept_class_ids = None  # resolved lazily in _get_detector, () means "keep all"
        self.frame_index = 0
        self.K = None
        self.baseline = None
        self.last_T = None
        self.last_param = (0.0, 0.0) # acc and gyro
        self.obstacle_config = ROBOT_CONFIG.obstacle
        self.stamp = None
        self.current_pose = None  # Store the latest pose from odometry

        self.smoothed_velocity = 0.0

        self.create_subscription(Odometry, '/control/target_pose', self.target_pose_callback, 10)
        self.target_pose = None

        self.poi_change_sub = self.create_subscription(Odometry, "/mapping/poi_change", self.poi_change_callback, 10)

    def poi_change_callback(self, msg):
        self.target_pose = None

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

    def depth_frame_info_callback(self, msg):
        if self.depth_frame_id is None:
            self.depth_frame_id = msg.header.frame_id
            self.get_logger().info(f"Depth frame id resolved to '{self.depth_frame_id}' (from infra1 camera_info).")
            self._try_resolve_depth_color_extrinsic()

    def color_camera_info_callback(self, msg):
        if self.color_K is None:
            self.color_K = np.array(msg.k).reshape(3, 3)
            self.color_frame_id = msg.header.frame_id
            self.get_logger().info(f"Color camera intrinsics received, frame '{self.color_frame_id}'.")
            self._try_resolve_depth_color_extrinsic()

    def tf_static_callback(self, msg):
        for transform in msg.transforms:
            frame_id, child_frame_id, T = tf2np(transform)
            store_transform(self.tf_edges, frame_id, child_frame_id, T)
        self._try_resolve_depth_color_extrinsic()

    def _try_resolve_depth_color_extrinsic(self):
        if self.T_depth_color is not None or self.depth_frame_id is None or self.color_frame_id is None:
            return
        T = lookup_transform(self.tf_edges, self.depth_frame_id, self.color_frame_id)
        if T is not None:
            self.T_depth_color = T
            self.get_logger().info(f"Resolved {self.depth_frame_id} -> {self.color_frame_id} extrinsic for color-based detection.")

    def color_image_callback(self, msg):
        self.latest_color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def color_image_compressed_callback(self, msg):
        self.latest_color_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')

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

    def publish_object_voxel_cloud(self, class_grid, resolution, origin):
        occupied = np.argwhere(class_grid >= 0)
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id="world")
        if len(occupied) == 0:
            self.object_voxel_pub.publish(pc2.create_cloud(header, [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
            ], np.zeros(0, dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)])))
            return

        coords = np.asarray(origin) + occupied * resolution
        class_ids = class_grid[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
        colors = _OBJECT_CLASS_PALETTE[class_ids % len(_OBJECT_CLASS_PALETTE)]
        rgb = (colors[:, 2].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8) | colors[:, 0].astype(np.uint32)

        dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)])
        points = np.zeros(coords.shape[0], dtype=dtype)
        points['x'], points['y'], points['z'] = coords[:, 0], coords[:, 1], coords[:, 2]
        points['rgb'] = rgb
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        self.object_voxel_pub.publish(pc2.create_cloud(header, fields, points))

    def _get_detector(self):
        """Lazily construct the TensorRT detector (and resolve the class-name
        allowlist) so nodes/tests that never exercise object detection don't
        need tensorrt/cuda importable."""
        if self.detector is None:
            from tinynav.core.models_trt import YoloDetectorTRT, coco_class_ids
            self.detector = YoloDetectorTRT(
                confidence_threshold=self.object_detection_config.confidence_threshold,
                iou_threshold=self.object_detection_config.iou_threshold,
            )
            self.kept_class_ids = coco_class_ids(self.object_detection_config.kept_class_names)
        return self.detector

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
                self.object_class_grid, self.object_ttl_grid, _ = roll_object_grids(
                    self.object_class_grid, self.object_ttl_grid, self.origin, new_origin, self.resolution,
                )
                self.occupancy_grid, self.origin = roll_occupancy_grid(self.occupancy_grid, self.origin, new_origin, self.resolution)
            new_occ = run_raycasting_loopy(depth, T, self.grid_shape, fx, fy, cx, cy, self.origin, self.step, self.resolution)
            self.occupancy_grid *= 0.99
            self.occupancy_grid += new_occ
            self.occupancy_grid = np.clip(self.occupancy_grid, -0.2, 0.2)

            self.publish_3d_occupancy_cloud(self.occupancy_grid, self.resolution, self.origin)

        with Timer(name='object detection', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            det_config = self.object_detection_config
            self.frame_index += 1
            color_ready = (
                self.latest_color_image is not None
                and self.color_K is not None
                and self.T_depth_color is not None
            )
            if not color_ready and not self._logged_color_tf_wait:
                self._logged_color_tf_wait = True
                self.get_logger().info(
                    "Waiting for color image + color camera_info + depth->color TF before object detection can start."
                )
            if det_config.enabled and color_ready and self.frame_index % max(1, det_config.detect_every_n_frames) == 0:
                detector = self._get_detector()
                detections = asyncio.run(detector.infer(self.latest_color_image))
                if self.kept_class_ids:
                    kept = set(self.kept_class_ids)
                    detections = [d for d in detections if d[0] in kept]
                hits = project_color_detections_to_voxels(
                    detections, depth, T, self.K, self.color_K, self.T_depth_color, self.latest_color_image.shape,
                    self.occupancy_grid, self.origin, self.resolution,
                    step=self.step, occ_threshold=self.obstacle_config.occ_threshold,
                )
                apply_object_hits(self.object_class_grid, self.object_ttl_grid, hits, det_config.ttl_frames)
            decay_object_grids(self.object_class_grid, self.object_ttl_grid)
            self.publish_object_voxel_cloud(self.object_class_grid, self.resolution, self.origin)

        with Timer(name='obstacle map', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            obstacle_mask = build_obstacle_map(
                self.occupancy_grid, self.origin, self.resolution,
                robot_z=T[2, 3], config=self.obstacle_config,
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
            trajectories, params = generate_trajectory_library_3d(
                init_p=init_p, init_q=init_q,
                max_linear_vel=ROBOT_CONFIG.max_linear_vel,
                max_angular_vel=ROBOT_CONFIG.max_angular_vel,
            )
            vocab_trajs, vocab_params = generate_predefined_trajectory_vocabularies(init_p=init_p, init_q=init_q)
            trajectories = np.concatenate([trajectories, vocab_trajs], axis=0)
            params = np.concatenate([params, vocab_params], axis=0)
            self.last_T = T
            self.last_stamp = stamp

        with Timer(name='traj score', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            front_len, rear_len, half_w = ROBOT_CONFIG.footprint_from_control()
            scores, occ_points = score_trajectories_by_ESDF(trajectories, ESDF_map, self.origin, self.resolution, ROBOT_CONFIG.safety_radius, front_len, rear_len, half_w)
            top_k = 100
            top_indices = np.argsort(scores, kind='stable')[:top_k]

        with Timer(name='pub', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            front_clearance = self._front_obstacle_dist(T, obstacle_mask)
            enter_threshold = 0.30

            def cost_function(traj, param, score, target_pose):
                # predefined backward trajectory penalty
                is_backward_traj = param[0] < 0.0
                should_reverse = front_clearance <= enter_threshold
                reverse_gate_penalty = 0.0
                if should_reverse and not is_backward_traj:
                        reverse_gate_penalty = 1e9
                elif not should_reverse and is_backward_traj:
                        reverse_gate_penalty = 1e9

                # regular trajectory penalty
                traj_end = np.array(traj[-1,:3])
                target_end = target_pose if target_pose is not None else traj_end
                dist = np.linalg.norm(traj_end - target_end)
                # heading error weighted like distance (1 rad ~ 1 m) far from the goal, faded out
                # linearly inside 2 m so bearing noise cannot dominate the distance term on arrival
                heading = goal_heading_error(traj[-1], target_end) * min(1.0, dist / 2.0)

                return (
                    score * 100000
                    + 100 * dist
                    + 100 * heading
                    + 10 * abs(self.last_param[0] - param[0])
                    + 10 * abs(self.last_param[1] - param[1])
                    + reverse_gate_penalty
                )

            top_k = 1
            top_indices = np.argsort(np.array([cost_function(trajectories[i], params[i], scores[i], self.target_pose) for i in range(len(trajectories))]), kind='stable')[:top_k]
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
