import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import Image, CameraInfo, PointField
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from cv_bridge import CvBridge
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
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
from tinynav.core.robot_specs import ROBOT_CONFIG, ObstacleConfig

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
def score_trajectories_by_ESDF(trajectories, ESDF_map, path_dist_map, remaining_map,
                                origin, resolution, safety_radius=0.1,
                                front_len=0.35, rear_len=0.35, half_w=0.15):
    """
    Score trajectories by ESDF clearance over the footprint (center + 4 corners), plus
    two lookups against the global route (path_dist_map, remaining_map -- see
    build_route_fields), which share ESDF_map's shape, origin and resolution.
    :return: per trajectory, the obstacle score (inf on collision), the step index of the
             minimum clearance, the worst (max) distance from the trajectory center to the
             route over the whole trajectory, and the route arc length still ahead of its
             end cell.
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
        path_cost_max = 0.0
        path_cost_n = 0
        traveled_arc = 0.0

        for i in range(len(traj)):
            x_world, y_world = traj[i, 0], traj[i, 1]
            if i > 0:
                dx = x_world - traj[i - 1, 0]
                dy = y_world - traj[i - 1, 1]
                traveled_arc += (dx * dx + dy * dy) ** 0.5
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
                        center_path_dist = float(path_dist_map[x_img, y_img])
                        if center_path_dist > path_cost_max:
                            path_cost_max = center_path_dist
                        path_cost_n += 1

        if path_cost_n > 0:
            path_costs.append(path_cost_max)
        else:
            path_costs.append(1e3)  # the whole trajectory left the grid

        end_x_img = int((traj[-1, 0] - origin[0]) / resolution)
        end_y_img = int((traj[-1, 1] - origin[1]) / resolution)
        if 0 <= end_x_img < ESDF_rows and 0 <= end_y_img < ESDF_cols:
            end_remaining = float(remaining_map[end_x_img, end_y_img])
        else:
            end_remaining = 1e3

        start_x_img = int((traj[0, 0] - origin[0]) / resolution)
        start_y_img = int((traj[0, 1] - origin[1]) / resolution)
        if 0 <= start_x_img < ESDF_rows and 0 <= start_y_img < ESDF_cols:
            start_remaining = float(remaining_map[start_x_img, start_y_img])
            if start_remaining < 1e3:
                end_remaining = max(end_remaining, start_remaining - traveled_arc)
        end_remainings.append(end_remaining)

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


def build_route_fields(route_xy, shape, origin, resolution):
    """
    Rasterize a route into the lookup maps read by the DWA scoring, so scoring costs
    one array lookup per trajectory point instead of a search over route points.
    :param route_xy: (N, 2) route in world xy, e.g. map_node's global plan transformed
                     into this node's world/odom frame -- the path is already clean of
                     static obstacles by construction, this just makes it queryable.
    :return: (path_dist_map, remaining_map, has_route). path_dist_map is the distance in m
             from each cell to the nearest route cell, remaining_map is the route arc length
             still ahead, with off-route cells inheriting the arc length of their nearest
             route cell.
    """
    path_dist_map = np.full(shape, 1e3, dtype=np.float32)
    remaining_map = np.full(shape, 1e3, dtype=np.float32)
    if len(route_xy) < 2:
        return path_dist_map, remaining_map, False

    rows, cols = shape
    route = np.asarray(route_xy, dtype=float)
    node_arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))))
    arc = float(node_arc[-1])
    if arc < 1e-9:
        return path_dist_map, remaining_map, False

    # resample at half-cell steps so the rasterized line has no gaps for the EDT
    sample_arc = np.linspace(0.0, arc, int(np.ceil(arc / (0.5 * resolution))) + 1)
    r = ((np.interp(sample_arc, node_arc, route[:, 0]) - origin[0]) / resolution).astype(np.int64)
    c = ((np.interp(sample_arc, node_arc, route[:, 1]) - origin[1]) / resolution).astype(np.int64)
    inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    if not np.any(inside):
        return path_dist_map, remaining_map, False

    route_mask = np.zeros(shape, dtype=bool)
    arc_map = np.zeros(shape, dtype=np.float32)
    route_mask[r[inside], c[inside]] = True
    arc_map[r[inside], c[inside]] = sample_arc[inside]  # a cell crossed twice keeps the later arc

    dist_cells, (near_r, near_c) = distance_transform_edt(~route_mask, return_indices=True)
    path_dist_map = (dist_cells * resolution).astype(np.float32)
    remaining_map = (arc - arc_map[near_r, near_c]).astype(np.float32)
    return path_dist_map, remaining_map, True


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
        self.resolution = 0.1
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

        # map_node's global plan already avoids static obstacles by construction --
        # DWA scoring below just needs it as two queryable maps (build_route_fields).
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.global_route_sub = self.create_subscription(
            Path, '/mapping/global_plan', self._on_global_route, 1
        )
        self._global_route_map_xy = None
        # obstacle score dominates (1e5 multiplier); among survivors, progress drives
        # speed and follow keeps the robot from cutting across to a closer route point
        self.w_route_progress = 100.0
        self.w_path_follow = 80.0
        # pulls the last stretch onto the exact goal, since remaining_map alone
        # saturates at 0 before reaching it
        self.w_goal_terminal = 100.0
        self.route_terminal_band = 0.5  # m of remaining route that arms w_goal_terminal

    def poi_change_callback(self, msg):
        self.target_pose = None
        self._global_route_map_xy = None  # the cached route led to the old target

    def target_pose_callback(self, msg):
        self.target_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def _on_global_route(self, msg: Path):
        self._global_route_map_xy = (
            np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
            if len(msg.poses) >= 2 else None
        )

    def _route_in_world(self):
        """The cached global route (map frame) transformed into this node's world/odom
        frame via TF, or None if there is no route yet or the transform isn't up."""
        if self._global_route_map_xy is None:
            return None
        try:
            t = self._tf_buffer.lookup_transform('world', 'map', rclpy.time.Time())
        except Exception:
            return None
        tr = t.transform.translation
        rot = t.transform.rotation
        R = quat_to_matrix([rot.x, rot.y, rot.z, rot.w])
        return self._global_route_map_xy @ R[:2, :2].T + np.array([tr.x, tr.y])

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
            self.last_T = T
            self.last_stamp = stamp

        with Timer(name='route fields', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            route_xy = self._route_in_world()
            path_dist_map, remaining_map, has_route = build_route_fields(
                route_xy if route_xy is not None else [], ESDF_map.shape, self.origin, self.resolution,
            )

        with Timer(name='traj score', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            front_len, rear_len, half_w = ROBOT_CONFIG.footprint_from_control()
            scores, occ_points, path_costs, end_remainings = score_trajectories_by_ESDF(
                trajectories, ESDF_map, path_dist_map, remaining_map,
                self.origin, self.resolution, ROBOT_CONFIG.safety_radius,
                front_len, rear_len, half_w,
            )
            top_k = 100
            top_indices = np.argsort(scores, kind='stable')[:top_k]

        with Timer(name='pub', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            front_clearance = self._front_obstacle_dist(T, obstacle_mask)
            enter_threshold = 0.30

            def cost_function(traj, param, score, path_cost, end_remaining):
                # predefined backward trajectory penalty
                is_backward_traj = param[0] < 0.0
                should_reverse = front_clearance <= enter_threshold
                reverse_gate_penalty = 0.0
                if should_reverse and not is_backward_traj:
                        reverse_gate_penalty = 1e9
                elif not should_reverse and is_backward_traj:
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
                cost_function(trajectories[i], params[i], scores[i], path_costs[i], end_remainings[i])
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
