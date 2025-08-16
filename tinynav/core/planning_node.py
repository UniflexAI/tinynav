import matplotlib
matplotlib.use('Agg')
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
from scipy.ndimage import maximum_filter
from numba import njit
import message_filters
import matplotlib.pyplot as plt
from rclpy.time import Time
import io
from PIL import Image as PIL_Image
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from codetiming import Timer
import cv2
from math_utils import rotvec_to_matrix, quat_to_matrix, matrix_to_quat, msg2np

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

def max_pool_height_map(height_map, kernel_size=5):
    nan_mask = np.isnan(height_map)
    filled = np.copy(height_map)
    filled[nan_mask] = -np.inf
    pooled = maximum_filter(filled, size=kernel_size, mode='nearest')
    return pooled

@njit(cache=True)
def generate_trajectory_library_3d(
    num_samples=11, duration=5.0, dt=0.1,
    acc_std=0.00001, omega_y_std_deg=20.0,
    init_p=np.zeros(3), init_v=np.zeros(3), init_q=np.array([0, 0, 0, 1])
):
    num_steps = int(duration / dt) + 1

    max_acc = 0.2
    acc_samples = np.linspace(-max_acc, max_acc, int(num_samples / 2))
    max_omega = np.pi / 8
    omega_y_samples = np.linspace(-max_omega, max_omega, num_samples)

    num_samples = len(acc_samples) * len(omega_y_samples)

    trajectories = np.empty((num_samples, num_steps, 7))
    params = np.empty((num_samples, 2))

    k = -1
    for i_acc in range(len(acc_samples)):
        for i_omega in range(len(omega_y_samples)):
            k += 1
            dv = acc_samples[i_acc]
            omega_y = omega_y_samples[i_omega]
            p = init_p.copy()
            v_world = init_v.copy()
            q = quat_to_matrix(init_q)
            traj = np.empty((num_steps, 7))
            for i in range(num_steps):
                dq = rotvec_to_matrix(np.array([0.0, omega_y * dt, 0.0]))
                v_world = q @ dq.T @ q.T @ v_world
                q = q @ dq

                acc_body = q.T @ v_world
                acc_body = acc_body / np.linalg.norm(acc_body)  # normalize
                acc_body = acc_body * dv

                acc_world = q @ acc_body
                v_world += acc_world * dt
                p += v_world * dt
                traj[i, :3] = p
                traj[i, 3:] = matrix_to_quat(q)
            #hack
            for i in range(num_steps):
                traj[i, 2] = traj[0, 2]
            trajectories[k] = traj
            params[k, 0] = dv
            params[k, 1] = omega_y
    return trajectories, params

@njit(cache=True)
def score_trajectories_by_heightmap(trajectories, height_map, origin, resolution):
    scores = []
    occ_points = []
    for t in range(len(trajectories)):
        traj = trajectories[t]
        valid_steps = 0
        valid_step_idx = []
        for i in range(len(traj)):
            x_world, y_world, z_world = traj[i, 0],traj[i, 1],traj[i, 2]
            x_img = int((x_world - origin[0]) / resolution)
            y_img = int((y_world - origin[1]) / resolution)
            if 0 <= x_img < height_map.shape[0] and 0 <= y_img < height_map.shape[1]:
                val = height_map[x_img, y_img]
                if not np.isnan(val) and val > z_world:
                    valid_steps += 1
                    valid_step_idx.append(i)
        scores.append(valid_steps)
        occ_points.append(valid_step_idx)
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
    def __init__(self):
        super().__init__('planning_node')
        self.bridge = CvBridge()
        self.path_pub = self.create_publisher(Path, '/planning/trajectory_path', 10)
        self.height_map_pub = self.create_publisher(Image, "/planning/height_map", 10)
        self.traj_scores_pub = self.create_publisher(Image, "/planning/score_traj", 10)
        self.occupancy_cloud_pub = self.create_publisher(PointCloud2, '/planning/occupied_voxels', 10)
        self.disp_sub = message_filters.Subscriber(self, Image, '/slam/disparity')
        self.pose_sub = message_filters.Subscriber(self, Odometry, '/slam/odometry')

        self.ts = message_filters.TimeSynchronizer([self.disp_sub, self.pose_sub], queue_size=10)
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
        self.stamp = None
        self.current_pose = None  # Store the latest pose from odometry

        self.smoothed_T = None

        self.create_subscription(Odometry, '/control/target_pose', self.target_pose_callback, 10)
        self.target_pose = np.array([0.0, 0.0, 0.0])

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
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "map"
        self.height_map_pub.publish(img_msg)
    def publish_3d_occupancy_cloud(self, grid3d, resolution=0.1, origin=(0, 0, 0)):
        occupied = np.argwhere(grid3d > 0)
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

    @Timer(name="Planning Loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms")
    def sync_callback(self, disp_msg, odom_msg):
        if self.K is None:
            return
        with Timer(name='preprocess', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            disparity = self.bridge.imgmsg_to_cv2(disp_msg, desired_encoding='32FC1')
            stamp = Time.from_msg(odom_msg.header.stamp).nanoseconds / 1e9
            T = msg2np(odom_msg)
            self.smoothed_T = T if self.smoothed_T is None else 0.9 * self.smoothed_T + 0.1 * T
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            depth = np.zeros_like(disparity)
            depth[disparity > 0] = fx * self.baseline / (disparity[disparity > 0])

        with Timer(name='raycasting', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            center = self.origin + np.array(self.grid_shape) * self.resolution / 2
            robot_pos = T[:3, 3]
            delta = robot_pos - center
            if np.linalg.norm(delta) > .1:
                new_center = robot_pos
                new_origin = new_center - np.array(self.grid_shape) * self.resolution / 2
                self.occupancy_grid, self.origin = roll_occupancy_grid(self.occupancy_grid, self.origin, new_origin, self.resolution)
            new_occ = run_raycasting_loopy(depth, T, self.grid_shape, fx, fy, cx, cy, self.origin, self.step, self.resolution)
            self.occupancy_grid += new_occ
            self.occupancy_grid = np.clip(self.occupancy_grid, -0.2, 0.2)

            self.publish_3d_occupancy_cloud(self.occupancy_grid, self.resolution, self.origin)

        with Timer(name='heightmap', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            height_map = occupancy_grid_to_height_map(self.occupancy_grid, self.origin, self.resolution)
            pooled_map = max_pool_height_map(height_map)

        with Timer(name='traj gen', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            if self.last_T is None:
                self.last_T = self.smoothed_T
                self.last_stamp = 0
            init_v = (T[:3, 3] - self.last_T[:3, 3]) / (stamp - self.last_stamp)
            trajectories, params = generate_trajectory_library_3d(
                init_p = T[:3, 3],
                init_v = init_v if np.linalg.norm(init_v) > 0.01 else np.array([0, 0.1, 0]),
                init_q = np.array([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w])
            )
            self.last_T = self.smoothed_T
            self.last_stamp = stamp

        with Timer(name='traj score', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            scores, occ_points = score_trajectories_by_heightmap(trajectories, pooled_map, self.origin, self.resolution)
            top_k = 100
            top_indices = np.argsort(scores, kind='stable')[:top_k]

        #with Timer(name='vis height map', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
        #    self.publish_height_map(T[:3,3], pooled_map, disp_msg.header)
        #with Timer(name='vis traj scores', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
        #    self.publish_height_map_traj(pooled_map, trajectories, occ_points, top_indices, scores, params, self.origin, self.resolution)

        with Timer(name='pub', text="[{name}] Elapsed time: {milliseconds:.0f} ms"):
            def cost_function(traj, param, score, target_pose):
                traj_end = np.array(traj[-1,:3])
                target_end = target_pose
                dist = np.linalg.norm(traj_end - target_end)
                return score * 100000 + 10 * dist + 10 * abs(self.last_param[0] - param[0]) + 10 * abs(self.last_param[1] - param[1])

            top_k = 1
            top_indices = np.argsort(np.array([cost_function(trajectories[i], params[i], scores[i], self.target_pose) for i in range(len(trajectories))]), kind='stable')[:top_k]
            self.last_param = params[top_indices[0]]

            # path
            path = Path()
            path.header = disp_msg.header
            path.header.frame_id = "world"
            for i in top_indices:
                for j in range(0, len(trajectories[i]), 10):
                    x,y,z,qx,qy,qz,qw = trajectories[i][j]
                    pose = PoseStamped()
                    pose.header = disp_msg.header
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
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
