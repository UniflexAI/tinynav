import argparse
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point32
from cv_bridge import CvBridge
import numpy as np
from scipy.ndimage import maximum_filter, distance_transform_edt
from numba import njit
import message_filters
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointCloud
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from codetiming import Timer
import cv2
import heapq
from math_utils import quat_to_matrix, msg2np
from geometry_msgs.msg import Twist


# half width of go2 is around 0.25 meters.
# set to 3x of resolution.
SAFETY_RADIUS=0.3

# Robot body safety footprint in planning frame (camera/world aligned forward).
# Reference point is robot control-center (near rear), so front/rear are asymmetric.
FRONT_SAFETY_LENGTH = 0.8
REAR_SAFETY_LENGTH = 0.0
HALF_SAFETY_WIDTH = 0.3

# Camera extrinsics wrt robot control center frame (meters)
# +x: robot left, +y: robot up, +z: robot forward (Tinynav convention in this node)
# Keep CAM_TO_BOTTOM as the single tuning knob for forward camera placement.
CAM_TO_BOTTOM = 0.6
CAM_TO_CENTER = 0.0 # left (negative) or right (positive)
CAM_TO_ABOVE = 0.0
ROBOT_TO_CAMERA_XYZ = np.array([CAM_TO_CENTER, CAM_TO_ABOVE, CAM_TO_BOTTOM], dtype=np.float32)


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

def build_fused_esdf_from_height(height_map_rel, occupancy_grid, origin, resolution, robot_z):
    """Fuse step/cliff edges from height map with wall/solid obstacles from 3D occupancy."""
    finite = np.isfinite(height_map_rel)
    filled = np.nan_to_num(height_map_rel, nan=0.0, neginf=0.0, posinf=0.0)

    # Pairwise neighbor height jump (meters), only where both cells are valid.
    h, w = filled.shape
    step_x = np.zeros((h, w), dtype=np.float32)
    step_y = np.zeros((h, w), dtype=np.float32)

    if w > 1:
        valid_x = finite[:, 1:] & finite[:, :-1]
        diff_x = np.abs(filled[:, 1:] - filled[:, :-1])
        diff_x = np.where(valid_x, diff_x, 0.0)
        step_x[:, 1:] = np.maximum(step_x[:, 1:], diff_x)
        step_x[:, :-1] = np.maximum(step_x[:, :-1], diff_x)

    if h > 1:
        valid_y = finite[1:, :] & finite[:-1, :]
        diff_y = np.abs(filled[1:, :] - filled[:-1, :])
        diff_y = np.where(valid_y, diff_y, 0.0)
        step_y[1:, :] = np.maximum(step_y[1:, :], diff_y)
        step_y[:-1, :] = np.maximum(step_y[:-1, :], diff_y)

    step_height = np.maximum(step_x, step_y)

    # Step/cliff obstacle source from 2D height discontinuity.
    max_step_height = 0.30
    step_obstacle = (step_height > max_step_height)

    # Wall/solid obstacle source from 3D occupancy projection around robot body height band.
    occ_threshold = 0.1
    wall_band_bottom = robot_z - 0.2
    wall_band_top = robot_z + 0.8
    z_world = origin[2] + (np.arange(occupancy_grid.shape[2]) + 0.5) * resolution
    z_mask = (z_world >= wall_band_bottom) & (z_world <= wall_band_top)
    if np.any(z_mask):
        wall_obstacle = np.any(occupancy_grid[:, :, z_mask] > occ_threshold, axis=2)
    else:
        wall_obstacle = np.zeros((h, w), dtype=bool)

    obstacle = step_obstacle | wall_obstacle

    # Light denoise for step speckles, but keep projected wall obstacles.
    step_u8 = (step_obstacle.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    step_obstacle_denoised = cv2.morphologyEx(step_u8, cv2.MORPH_OPEN, kernel) > 0
    obstacle = step_obstacle_denoised | wall_obstacle

    if np.any(obstacle):
        fused_esdf = distance_transform_edt(~obstacle) * resolution
    else:
        fused_esdf = np.full_like(filled, 100.0, dtype=np.float32)

    # For downstream cost, keep a normalized slope-like magnitude (m/m).
    slope = step_height / max(1e-6, resolution)
    unknown_mask = (~finite) & (~wall_obstacle)

    return fused_esdf, slope, unknown_mask

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


def astar_2d(cost_map, obstacle_mask, start, goal, search_mask=None):
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
            if search_mask is not None and (not search_mask[nx, ny]):
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


def build_astar_cost_map_from_fused_esdf(fused_esdf, unknown_mask, resolution):
    # Pure ESDF mode: keep hard obstacle narrow, but add a stronger clearance barrier
    # so A* naturally prefers center-of-free-space instead of edge-hugging.
    hard_esdf = 0.15
    obstacle = fused_esdf < hard_esdf

    d = np.clip(fused_esdf, hard_esdf, 2.0)

    # Preferred clearance band (meters). Inside this band, cost rises rapidly.
    safe_clearance = 0.45
    barrier = np.where(
        d >= safe_clearance,
        0.0,
        ((safe_clearance - d) / max(1e-6, safe_clearance)) ** 2,
    )

    unknown_penalty = np.where(unknown_mask, 15.0, 0.0)

    # Do not inflate obstacles geometrically here; let ESDF barrier shape the route.
    cost = 1.0 + 10.0 * barrier + unknown_penalty
    return obstacle, cost


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


def footprint_corners_xy(center_xy, forward_xy, front_length, rear_length, half_width):
    n = np.linalg.norm(forward_xy)
    if n < 1e-6:
        forward_xy = np.array([1.0, 0.0], dtype=np.float32)
    else:
        forward_xy = forward_xy / n
    left_xy = np.array([-forward_xy[1], forward_xy[0]], dtype=np.float32)

    c = np.asarray(center_xy, dtype=np.float32)
    return [
        c + forward_xy * front_length + left_xy * half_width,
        c + forward_xy * front_length - left_xy * half_width,
        c - forward_xy * rear_length - left_xy * half_width,
        c - forward_xy * rear_length + left_xy * half_width,
    ]


def footprint_collision_4corners(center_xy, forward_xy, obstacle_mask, origin, resolution, front_length, rear_length, half_width):
    h, w = obstacle_mask.shape
    corners = footprint_corners_xy(center_xy, forward_xy, front_length, rear_length, half_width)
    for corner in corners:
        gx, gy = world_to_grid_2d(corner, origin, resolution)
        if gx < 0 or gx >= h or gy < 0 or gy >= w:
            return True
        if obstacle_mask[gx, gy]:
            return True
    return False

# === PlanningNode class ===
class PlanningNode(Node):
    def __init__(self, sensor_source: str = 'auto'):
        super().__init__('planning_node')
        self.bridge = CvBridge()
        self.path_pub = self.create_publisher(Path, '/planning/trajectory_path', 10)
        self.height_map_pub = self.create_publisher(Image, "/planning/height_map", 10)
        self.occupancy_cloud_pub = self.create_publisher(PointCloud2, '/planning/occupied_voxels', 10)
        self.fused_esdf_pub = self.create_publisher(OccupancyGrid, '/planning/fused_esdf', 10)
        self.astar_cost_pub = self.create_publisher(Image, '/planning/astar_cost', 10)
        self.footprint_pub = self.create_publisher(PointCloud, '/planning/footprint', 10)
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


        self.grid_shape = (100, 100, 30)
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
        self.max_angular_acc = 2.5   # rad/s^2 (unlock turning response)
        self.recovery_fast_speed = 0.18
        self.recovery_slow_speed = 0.08
        self.path_smooth_window = 5
        self.path_smooth_passes = 0


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

    def camera_to_robot_center(self, T):
        """Convert camera pose (T_cam->world) to robot control-center position in world."""
        return T[:3, 3] - T[:3, :3] @ ROBOT_TO_CAMERA_XYZ

    def publish_height_map(self, origin, pooled_map, header):
        height_normalized = (np.nan_to_num(pooled_map, nan=0.0) + 5) * 30
        height_uint8 = height_normalized.astype(np.uint8)
        color_image = cv2.applyColorMap(height_uint8, cv2.COLORMAP_JET)
        img_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        img_msg.header = header
        self.height_map_pub.publish(img_msg)

    def publish_2d_occupancy_grid(self, ESDF_map, origin, resolution, stamp, z_offset=0.0, pub=None):
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
        if pub is not None:
            pub.publish(occupancy_grid_msg)

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

    def publish_footprint(self, T, stamp):
        """Publish robot footprint rectangle as a point cloud for RViz."""
        forward = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        left = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
        center = self.camera_to_robot_center(T)

        corners = [
            center + forward * FRONT_SAFETY_LENGTH + left * HALF_SAFETY_WIDTH,
            center + forward * FRONT_SAFETY_LENGTH - left * HALF_SAFETY_WIDTH,
            center - forward * REAR_SAFETY_LENGTH - left * HALF_SAFETY_WIDTH,
            center - forward * REAR_SAFETY_LENGTH + left * HALF_SAFETY_WIDTH,
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

        with Timer(name='raycasting', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            center = self.origin + np.array(self.grid_shape) * self.resolution / 2
            robot_pos = self.camera_to_robot_center(T)
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
            pooled_map_rel = pooled_map - robot_z
            fused_esdf, _, unknown_mask = build_fused_esdf_from_height(
                pooled_map_rel,
                self.occupancy_grid,
                self.origin,
                self.resolution,
                robot_z,
            )


        with Timer(name='vis heighmap and esdf', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            self.publish_height_map(T[:3,3], pooled_map_rel, depth_msg.header)
            self.publish_2d_occupancy_grid(
                fused_esdf,
                self.origin,
                self.resolution,
                depth_msg.header.stamp,
                z_offset=self.grid_shape[2]*self.resolution/2,
                pub=self.fused_esdf_pub,
            )
            self.publish_footprint(T, depth_msg.header.stamp)

        with Timer(name='local astar plan', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            local_path_world = []
            if self.target_pose is not None:
                obstacle_mask, astar_cost = build_astar_cost_map_from_fused_esdf(
                    fused_esdf, unknown_mask, self.resolution
                )
                self.publish_cost_heatmap(astar_cost, depth_msg.header)

                robot_center = self.camera_to_robot_center(T)
                start_idx = clip_grid_2d(world_to_grid_2d(robot_center[:2], self.origin, self.resolution), obstacle_mask.shape)
                goal_idx = clip_grid_2d(world_to_grid_2d(self.target_pose[:2], self.origin, self.resolution), obstacle_mask.shape)

                start_free = find_nearest_free(start_idx, obstacle_mask)
                goal_free = find_nearest_free(goal_idx, obstacle_mask)

                # 单通道策略：只保留 pass3（基于 ESDF+slope 生成的 obstacle/cost）
                grid_path = astar_2d(astar_cost, obstacle_mask, start_free, goal_free)

                # 4-corner footprint collision filtering (robust mode):
                # progressively relax footprint size to avoid over-pruning and "freeze" behavior.
                if len(grid_path) >= 2:
                    relax_scales = (1.0, 0.9, 0.8, 0.7)
                    min_keep = max(3, int(0.25 * len(grid_path)))
                    best_filtered = None

                    for s in relax_scales:
                        front_len = FRONT_SAFETY_LENGTH * s
                        rear_len = REAR_SAFETY_LENGTH * s
                        half_w = HALF_SAFETY_WIDTH * s

                        filtered = [grid_path[0]]
                        for i in range(1, len(grid_path)):
                            p_prev = grid_to_world_2d(filtered[-1], self.origin, self.resolution, 0.0)[:2]
                            p_cur = grid_to_world_2d(grid_path[i], self.origin, self.resolution, 0.0)[:2]
                            fwd = p_cur - p_prev
                            if np.linalg.norm(fwd) < 1e-6:
                                fwd = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
                                fwd = fwd[:2]
                            if footprint_collision_4corners(
                                p_cur,
                                fwd,
                                obstacle_mask,
                                self.origin,
                                self.resolution,
                                front_len,
                                rear_len,
                                half_w,
                            ):
                                break
                            filtered.append(grid_path[i])

                        if best_filtered is None or len(filtered) > len(best_filtered):
                            best_filtered = filtered
                        if len(filtered) >= min_keep:
                            best_filtered = filtered
                            break

                    # Fail-open fallback: avoid losing tracking completely when footprint is too strict.
                    if best_filtered is not None and len(best_filtered) >= 2:
                        grid_path = best_filtered

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
                robot_xy = self.camera_to_robot_center(T)[:2]
                target_dist = np.linalg.norm(self.target_pose[:2] - robot_xy)
                if target_dist < 0.25:
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                elif len(local_path_world) < 2:
                    # Recovery behavior: cautiously probe toward target instead of pure spinning.
                    # This keeps the robot moving even when footprint filtering truncates the path.
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
                        cmd.angular.z = float(np.clip(1.6 * heading_err, -2.0, 2.0))
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

                        cmd.angular.z = float(np.clip(1.8 * heading_err, -2.0, 2.0))
                        heading_scale = max(0.0, np.cos(heading_err))
                        dist_scale = np.clip(target_dist / 1.0, 0.2, 1.0)
                        cmd.linear.x = float(np.clip(self.max_linear_speed * heading_scale * dist_scale, 0.0, self.max_linear_speed))
                        if abs(heading_err) > 1.0:
                            cmd.linear.x *= 0.40

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
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

