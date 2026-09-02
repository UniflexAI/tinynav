import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointField
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from cv_bridge import CvBridge
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.spatial.transform import Rotation as R
from numba import njit
import message_filters
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointCloud
from geometry_msgs.msg import PoseStamped, Point32
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from codetiming import Timer
import cv2
from tinynav.core.math_utils import rotvec_to_matrix, quat_to_matrix, matrix_to_quat, msg2np, tf2np
from tinynav.core.robot_specs import ROBOT_CONFIG, ObstacleConfig
from tf2_ros import Buffer, TransformListener

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

def _camera_quat_from_xy_tangent(tx, ty, fallback_q):
    tangent = np.array([tx, ty, 0.0], dtype=np.float64)
    n = np.linalg.norm(tangent)
    if n < 1e-6:
        return fallback_q
    cam_z = tangent / n
    cam_y = np.array([0.0, 0.0, -1.0])
    cam_x = np.cross(cam_y, cam_z)
    xn = np.linalg.norm(cam_x)
    if xn < 1e-6:
        return fallback_q
    cam_x = cam_x / xn
    cam_y = np.cross(cam_z, cam_x)
    return matrix_to_quat(np.column_stack((cam_x, cam_y, cam_z)))

def _heading_change_along_path(path_xy: np.ndarray, start_idx: int, distance: float) -> float:
    pts = path_xy[start_idx:]
    if len(pts) < 2 or distance < 1e-3:
        return 0.0
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    d = min(float(distance), float(cum[-1]))
    if d < 1e-3:
        return 0.0
    k = int(np.searchsorted(cum, d, side='right') - 1)
    k = min(max(k, 0), len(seg_len) - 1)
    d0 = pts[1] - pts[0]
    d1 = pts[min(k + 1, len(pts) - 1)] - pts[k]
    n0 = np.linalg.norm(d0)
    n1 = np.linalg.norm(d1)
    if n0 < 1e-6 or n1 < 1e-6:
        return 0.0
    dot = float(np.clip((d0 / n0) @ (d1 / n1), -1.0, 1.0))
    return float(np.arccos(dot))

def generate_path_follow_trajectory(
    path_xy, init_p, init_q, speed, duration=3.0, dt=0.1,
):
    """Timed trajectory that walks the global plan. Captures L/U turns the lattice cannot."""
    if path_xy is None or len(path_xy) < 2:
        return None, None, 0.0
    robot_xy = np.asarray(init_p[:2], dtype=np.float64)
    closest = int(np.argmin(np.linalg.norm(path_xy - robot_xy[None, :], axis=1)))
    pts = np.vstack((robot_xy, path_xy[closest:]))
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-4:
            keep.append(i)
    pts = pts[np.asarray(keep)]
    if len(pts) < 2:
        return None, None, 0.0

    num_steps = int(duration / dt) + 1
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cum[-1])
    if total < 1e-3:
        return None, None, 0.0

    traj = np.empty((num_steps, 7), dtype=np.float64)
    z = float(init_p[2])
    fallback_q = np.asarray(init_q, dtype=np.float64)
    for i in range(num_steps):
        s = min(speed * i * dt, total)
        k = int(np.searchsorted(cum, s, side='right') - 1)
        k = min(max(k, 0), len(seg_len) - 1)
        t = (s - cum[k]) / max(seg_len[k], 1e-12)
        xy = pts[k] + t * (pts[k + 1] - pts[k])
        traj[i, 0] = xy[0]
        traj[i, 1] = xy[1]
        traj[i, 2] = z
        traj[i, 3:] = _camera_quat_from_xy_tangent(
            pts[k + 1, 0] - pts[k, 0],
            pts[k + 1, 1] - pts[k, 1],
            fallback_q,
        )

    i_omega = min(5, num_steps - 1)
    z0 = quat_to_matrix(traj[0, 3:]) @ np.array([0.0, 0.0, 1.0])
    z1 = quat_to_matrix(traj[i_omega, 3:]) @ np.array([0.0, 0.0, 1.0])
    yaw0 = np.arctan2(z0[1], z0[0])
    yaw1 = np.arctan2(z1[1], z1[0])
    omega = float(np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0)) / max(i_omega * dt, 1e-6))
    param = np.array([speed, omega], dtype=np.float64)
    return traj, param, _heading_change_along_path(path_xy, closest, speed * duration)

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
        self.depth_sub = message_filters.Subscriber(self, Image, '/slam/depth')
        self.pose_sub = message_filters.Subscriber(self, Odometry, '/slam/odometry_visual')

        self.ts = message_filters.TimeSynchronizer([self.depth_sub, self.pose_sub], queue_size=10)
        self.ts.registerCallback(self.sync_callback)
        self.camerainfo_sub = self.create_subscription(CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)

        self.grid_shape = (100, 100, 10)
        self.resolution = 0.05
        self.origin = np.array(self.grid_shape) * self.resolution / -2.
        self.step = 10
        self.occupancy_grid = np.zeros(self.grid_shape)
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
        self.create_subscription(Path, '/mapping/global_plan', self.global_plan_callback, 10)
        self._map_plan_xyz = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

    def poi_change_callback(self, msg):
        self.target_pose = None
        self._map_plan_xyz = None

    def global_plan_callback(self, msg: Path):
        if not msg.poses:
            self._map_plan_xyz = None
            return
        self._map_plan_xyz = np.array(
            [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses],
            dtype=np.float64,
        )

    def _global_plan_xy_in_world(self):
        if self._map_plan_xyz is None or len(self._map_plan_xyz) < 2:
            return None
        try:
            tf_msg = self._tf_buffer.lookup_transform('world', 'map', Time())
        except Exception:
            return None
        _, _, T = tf2np(tf_msg)
        pts = (T[:3, :3] @ self._map_plan_xyz.T).T + T[:3, 3]
        return pts[:, :2]

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
                self.occupancy_grid, self.origin = roll_occupancy_grid(self.occupancy_grid, self.origin, new_origin, self.resolution)
            new_occ = run_raycasting_loopy(depth, T, self.grid_shape, fx, fy, cx, cy, self.origin, self.step, self.resolution)
            self.occupancy_grid *= 0.99
            self.occupancy_grid += new_occ
            self.occupancy_grid = np.clip(self.occupancy_grid, -0.2, 0.2)

            self.publish_3d_occupancy_cloud(self.occupancy_grid, self.resolution, self.origin)

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
            path_follow_turn = np.zeros(len(trajectories), dtype=bool)
            world_plan_xy = self._global_plan_xy_in_world()
            if world_plan_xy is not None:
                pf_trajs = []
                pf_params = []
                pf_turns = []
                for speed in (0.4, 0.7):
                    traj, param, turn = generate_path_follow_trajectory(
                        world_plan_xy, init_p, init_q, speed,
                    )
                    if traj is None:
                        continue
                    pf_trajs.append(traj)
                    pf_params.append(param)
                    pf_turns.append(turn)
                if pf_trajs:
                    trajectories = np.concatenate([trajectories, np.stack(pf_trajs)], axis=0)
                    params = np.concatenate([params, np.stack(pf_params)], axis=0)
                    path_follow_turn = np.concatenate([
                        path_follow_turn,
                        np.array(pf_turns, dtype=np.float64) > np.deg2rad(60.0),
                    ])
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

            def cost_function(traj, param, score, target_pose, follow_turn=False):
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
                # Prefer walking the global plan around a sharp corner instead of cutting it.
                follow_bonus = -400.0 if follow_turn and np.isfinite(score) else 0.0

                return (
                    score * 100000
                    + 100 * dist
                    + 100 * heading
                    + 10 * abs(self.last_param[0] - param[0])
                    + 10 * abs(self.last_param[1] - param[1])
                    + reverse_gate_penalty
                    + follow_bonus
                )

            top_k = 1
            top_indices = np.argsort(np.array([
                cost_function(
                    trajectories[i], params[i], scores[i], self.target_pose,
                    follow_turn=bool(path_follow_turn[i]),
                )
                for i in range(len(trajectories))
            ]), kind='stable')[:top_k]
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
