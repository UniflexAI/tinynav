import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import cv2
from func import lru_cache_numpy

@njit(cache=True)
def rotvec_to_matrix(rv):
    """Convert a rotation vector to a rotation matrix using Rodrigues' formula."""
    theta = np.linalg.norm(rv)
    if theta < 1e-8:
        return np.eye(3)
    axis = rv / theta
    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1 - c
    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,     y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]
    ])
    return R

@njit(cache=True)
def quat_to_matrix(q):
    """Convert a quaternion [x, y, z, w] to a rotation matrix."""
    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w
    R = np.empty((3, 3))
    R[0, 0] = 1 - 2 * (yy + zz)
    R[0, 1] = 2 * (xy - zw)
    R[0, 2] = 2 * (xz + yw)
    R[1, 0] = 2 * (xy + zw)
    R[1, 1] = 1 - 2 * (xx + zz)
    R[1, 2] = 2 * (yz - xw)
    R[2, 0] = 2 * (xz - yw)
    R[2, 1] = 2 * (yz + xw)
    R[2, 2] = 1 - 2 * (xx + yy)
    return R

@njit(cache=True)
def matrix_to_quat(R):
    """Convert a rotation matrix to a quaternion [x, y, z, w]."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw]) 

# get rotation matrix from two vectors, so that R @ a = b
def rot_from_two_vector(a, b):
    """Get rotation matrix that rotates vector a to vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.linalg.norm(v) < 1e-8 and abs(c - 1.0) < 1e-8:
        return np.eye(3)  # No rotation needed

    s = np.linalg.norm(v)
    v /= s
    vx, vy, vz = v
    R = np.array([
        [c + vx*vx*(1-c), vx*vy*(1-c) - vz*s, vx*vz*(1-c) + vy*s],
        [vy*vx*(1-c) + vz*s, c + vy*vy*(1-c), vy*vz*(1-c) - vx*s],
        [vz*vx*(1-c) - vy*s, vz*vy*(1-c) + vx*s, c + vz*vz*(1-c)]
    ])
    return R

def np2msg(odom_np, timestamp, frame_id, child_frame_id):
    R_odom = odom_np[:3, :3]
    t_odom = odom_np[:3, 3]
    quat = R.from_matrix(R_odom).as_quat()
    odom_msg = Odometry()
    odom_msg.header.stamp = timestamp
    odom_msg.header.frame_id = frame_id
    odom_msg.child_frame_id = child_frame_id
    odom_msg.pose.pose.position.x = t_odom[0]
    odom_msg.pose.pose.position.y = t_odom[1]
    odom_msg.pose.pose.position.z = t_odom[2]
    odom_msg.pose.pose.orientation.x = quat[0]
    odom_msg.pose.pose.orientation.y = quat[1]
    odom_msg.pose.pose.orientation.z = quat[2]
    odom_msg.pose.pose.orientation.w = quat[3]
    return odom_msg

def np2tf(odom_np, timestamp, frame_id, child_frame_id):
    odom_msg = np2msg(odom_np, timestamp, frame_id, child_frame_id)
    tf_msg = TransformStamped()
    tf_msg.header.stamp = timestamp
    tf_msg.header.frame_id = frame_id
    tf_msg.child_frame_id = child_frame_id
    tf_msg.transform.translation.x = odom_msg.pose.pose.position.x
    tf_msg.transform.translation.y = odom_msg.pose.pose.position.y
    tf_msg.transform.translation.z = odom_msg.pose.pose.position.z
    tf_msg.transform.rotation.x = odom_msg.pose.pose.orientation.x
    tf_msg.transform.rotation.y = odom_msg.pose.pose.orientation.y
    tf_msg.transform.rotation.z = odom_msg.pose.pose.orientation.z
    tf_msg.transform.rotation.w = odom_msg.pose.pose.orientation.w
    return tf_msg

def tf2np(tf_msg:TransformStamped):
    T = np.eye(4)
    position = tf_msg.transform.translation
    rot = tf_msg.transform.rotation
    quat = [rot.x, rot.y, rot.z, rot.w]
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = np.array([position.x, position.y, position.z]).ravel()
    return tf_msg.header.frame_id, tf_msg.child_frame_id, T

def msg2np(msg):
    T = np.eye(4)
    position = msg.pose.pose.position
    rot = msg.pose.pose.orientation
    quat = [rot.x, rot.y, rot.z, rot.w]
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = np.array([position.x, position.y, position.z]).ravel()
    return T

@njit(cache=True)
def depth_to_cloud(depth, K, step=10, max_dist=1e9):
    h, w = depth.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    pts = []  # numba-typed list

    for v in range(0, h, step):
        for u in range(0, w, step):
            z = depth[v, u]
            if z > 0.0 and z <= max_dist:
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                pts.append((x, y, z))   # tuples are allowed

    # convert typed list → ndarray
    return np.array(pts)

@njit(cache=True)
def process_keypoints(kpts_prev, kpts_curr, idx_valid, depth, K):
    points_3d = np.empty((len(kpts_prev), 3), dtype=np.float32)
    points_2d = np.empty((len(kpts_prev), 2), dtype=np.float32)
    valid_idx = np.empty(len(kpts_prev), dtype=np.int32)
    valid_count = 0
    
    for i in range(len(kpts_prev)):
        u, v = int(kpts_curr[i,0]), int(kpts_curr[i,1])
        if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
            Z = depth[v, u]
            if Z > 0.1 and Z < 10.0:
                X = (kpts_curr[i,0] - K[0,2]) * Z / K[0,0]
                Y = (kpts_curr[i,1] - K[1,2]) * Z / K[1,1]
                points_3d[valid_count] = (X, Y, Z)
                points_2d[valid_count] = kpts_prev[i]
                valid_idx[valid_count] = idx_valid[i]
                valid_count += 1
    
    return points_3d[:valid_count], points_2d[:valid_count], valid_idx[:valid_count]

@lru_cache_numpy(maxsize=128)
def estimate_pose(kpts_prev, kpts_curr, depth, K, idx_valid=None):
    """
    Unified pose estimation function with cache support.
    """
    if idx_valid is None:
        idx_valid = np.arange(len(kpts_prev), dtype=np.int32)
    
    # Core pose estimation logic
    points_3d, points_2d, idx_valid = process_keypoints(
        kpts_prev.astype(np.float32), 
        kpts_curr.astype(np.float32),
        idx_valid,
        depth, 
        K.astype(np.float32)
    )
    if len(points_3d) < 6:
        return False, np.eye(4), [], [], []
    points_3d = np.array(points_3d, dtype=np.float32)
    points_2d = np.array(points_2d, dtype=np.float32)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, K, None, reprojectionError=2.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        return False, np.eye(4), [], [], []
    R_mat, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = tvec.ravel()
    inliers = inliers.flatten()
    inliers_2d = points_2d[inliers]
    inliers_3d = points_3d[inliers]
    inlier_idx_original = idx_valid[inliers]
    return True, T, inliers_2d, inliers_3d, inlier_idx_original

# Disjoint Set (Union-Find) implementation with path compression and union by rank
@njit(cache=True)
def uf_init(n):
    parent = np.empty(n, np.int64)
    rank = np.zeros(n, np.int64)
    for i in range(n):
        parent[i] = i
    return parent, rank

@njit(cache=True)
def uf_find(i, parent):
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != i:
        p = parent[i]
        parent[i] = root
        i = p
    return root

@njit(cache=True)
def uf_union(a, b, parent, rank):
    ra = uf_find(a, parent)
    rb = uf_find(b, parent)
    if ra == rb:
        return ra
    if rank[ra] < rank[rb]:
        parent[ra] = rb
        return rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
        return ra
    else:
        parent[rb] = ra
        rank[ra] += 1
        return ra

def uf_all_sets_list(parent):
    root_to_members = {}
    for i in range(len(parent)):
        r = parent[i]
        root_to_members.setdefault(r, []).append(i)
    return list(root_to_members.values())

def reconstruct_path(parent: dict, current:np.ndarray) -> np.ndarray:
    """
    Reconstructs the path from the start to the goal.
    :param came_from: dict, mapping of nodes to their predecessors
    :param current: tuple, the current node
    :return: list of tuples representing the path
    """
    path = []
    while current in parent:
        path.append(current)
        if current == parent[current]:
            break
        current = parent[current]
    return np.array(path[::-1])


def heuristic(start, goal):
    return np.linalg.norm(np.array(start) - np.array(goal))


def theta_star(cost_map:np.ndarray, start:np.ndarray, goal:np.ndarray, obstacles_cost: float) -> np.ndarray:
    """
    theta* algorithm to find the path from start to goal in the cost map.
    parameters:
        cost_map: np.ndarray (H, W)
        start: tuple[int, int], x_idx, y_idx
        goal: tuple[int, int], x_idx, y_idx
    returns: list of tuples representing the path from start to goal
    If no path is found, returns an empty list.
    0 - unknown, 0.5 - free, 1.0 - occupied
    """
    start = tuple(start.flatten()) if isinstance(start, np.ndarray) else start
    goal = tuple(goal.flatten()) if isinstance(goal, np.ndarray) else goal
    open_set = set()

    open_set.add(start)

    g_score = {start: cost_map[start]}
    f_score = {start: heuristic(start, goal) + cost_map[start]}
    parent = {start: start}
    visited = set()
    while len(open_set) > 0:
        # there can be a better way to maintain a min heap to reduce the complexity
        current = sorted(open_set, key=lambda x: f_score[x])[0]
        open_set.remove(current)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return reconstruct_path(parent, current)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in visited:
                    continue
                if (0 <= neighbor[0] < cost_map.shape[0] and
                        0 <= neighbor[1] < cost_map.shape[1] and cost_map[neighbor] < obstacles_cost):
                    if neighbor not in open_set:
                        g_score[neighbor] = float('inf')
                        f_score[neighbor] = float('inf')
                        parent[neighbor] = None
                    update_node(cost_map, g_score, f_score, open_set, parent, current, neighbor, goal, obstacles_cost)
    return []


def line_of_sight(cost_map:np.ndarray, start:tuple, end:tuple, obstacles_cost: float):
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0 and dy == 0:
        return True, cost_map[start]
    sX = -1
    sY = -1
    if (dx > 0):
        sX = 1
    if (dy > 0):
        sY = 1
    max_step = max(abs(dx), abs(dy))

    dx = abs(dx) / max_step
    dy = abs(dy) / max_step
    accumulated_cost = 0.0
    for i in range(max_step):
        node = (int(x0 + (i + 1) * sX * dx), int(y0 + (i + 1) * sY * dy))
        if cost_map[node] >= obstacles_cost:
            return False,  accumulated_cost
        accumulated_cost += cost_map[node]
    return True, accumulated_cost

def update_node(cost_map:np.ndarray, g_score:dict, f_score:dict, open_set:set, parent:dict, current:tuple, neighbor:tuple, goal:tuple, obstacles_cost: float):
    status, cost = line_of_sight(cost_map, parent[current], neighbor, obstacles_cost)
    if status and g_score[parent[current]] + cost < g_score[neighbor]:
        g_score[neighbor] = g_score[parent[current]] + cost
        parent[neighbor] = parent[current]
        f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
        if neighbor not in open_set:
            open_set.add(neighbor)
    else:
       if g_score[current] + cost_map[neighbor] < g_score[neighbor]:
            g_score[neighbor] = g_score[current] + cost_map[neighbor]
            parent[neighbor] = current
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
            if neighbor not in open_set:
                open_set.add(neighbor)
def se3_inv(matrix_4x4:np.ndarray):
    rotation = matrix_4x4[:3, :3]
    translation = matrix_4x4[:3, 3]
    T = np.eye(4)
    T[:3, :3] = rotation.T
    T[:3, 3] = -rotation.T @ translation
    return T
