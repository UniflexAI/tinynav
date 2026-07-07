import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
import cv2
import fufpy
from tinynav.core.func import lru_cache_numpy

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

def np2msg(odom_np, timestamp, frame_id, child_frame_id, velocity=None):
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
    if velocity is not None:
        odom_msg.twist.twist.linear.x = velocity[0]
        odom_msg.twist.twist.linear.y = velocity[1]
        odom_msg.twist.twist.linear.z = velocity[2]
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
    if msg.twist.twist is not None:
        velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
    else:
        velocity = np.array([0.0, 0.0, 0.0])
    return T, velocity

def pose_msg2np(msg: PoseStamped):
    T = np.eye(4)
    position = msg.pose.position
    rot = msg.pose.orientation
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
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float64)
    # convert typed list → ndarray
    return np.array(pts)

def backproject_depth_with_normals(depth, K, step=6, max_dist=4.0, min_view_cos=0.2):
    """Grid-strided pinhole back-projection with a per-point surface normal
    (central-difference on the depth grid, oriented to face the camera).

    Border pixels (no full 4-neighbor set) and grazing-incidence points
    (normal more than ~acos(min_view_cos) off the viewing ray — the classic
    source of noisy/unreliable depth-derived geometry) are dropped.

    Returns flat (points[N,3], normals[N,3]) in the depth camera's own frame
    (not vectorized via numba: this runs once per keyframe pair for ICP, not
    per-pixel-per-frame at high rate, so plain numpy is fast enough).
    """
    h, w = depth.shape
    us = np.arange(0, w, step)
    vs = np.arange(0, h, step)
    grid_u, grid_v = np.meshgrid(us, vs)
    z = depth[grid_v, grid_u]
    valid = (z > 0.0) & (z <= max_dist)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (grid_u - cx) * z / fx
    y = (grid_v - cy) * z / fy
    pts_grid = np.stack([x, y, z], axis=-1)

    rows, cols = pts_grid.shape[0], pts_grid.shape[1]
    if rows < 3 or cols < 3:
        return np.empty((0, 3)), np.empty((0, 3))

    center = pts_grid[1:-1, 1:-1]
    du = pts_grid[1:-1, 2:] - pts_grid[1:-1, :-2]
    dv = pts_grid[2:, 1:-1] - pts_grid[:-2, 1:-1]
    n = np.cross(du, dv)
    n_norm = np.linalg.norm(n, axis=-1)
    finite = n_norm > 1e-9
    safe_norm = np.where(finite, n_norm, 1.0)
    unit_n = n / safe_norm[..., None]

    # Camera is at the origin of its own frame, so each point's own position
    # is exactly the viewing ray from camera to point.
    view_dist = np.linalg.norm(center, axis=-1)
    safe_view_dist = np.where(view_dist > 1e-9, view_dist, 1.0)
    view_cos = -np.sum(unit_n * center, axis=-1) / safe_view_dist
    flip = view_cos < 0
    unit_n[flip] *= -1
    view_cos = np.abs(view_cos)

    neighbors_valid = (
        valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
        & valid[2:, 1:-1] & valid[:-2, 1:-1]
    )
    keep = neighbors_valid & finite & (view_dist > 1e-9) & (view_cos >= min_view_cos)
    return center[keep], unit_n[keep]


def point_to_plane_icp(source_points, target_points, target_normals, T_init,
                        max_iters=20, max_corr_dist=0.15, tol=1e-6):
    """Point-to-plane ICP (Gauss-Newton). `target_points`/`target_normals`
    define the fixed reference surface; `source_points` are transformed by
    successive estimates of T (maps source frame -> target frame) until the
    update step shrinks below `tol` or `max_iters` is reached.

    Returns (T, info). info = {converged, n_iters, inlier_ratio,
    mean_residual, min_eigval}: `min_eigval` is the smallest eigenvalue of
    the last iteration's 6x6 normal-equations matrix (J^T J) — a low value
    means the alignment is poorly constrained along some direction (e.g.
    sliding along a flat wall or down a corridor), the standard ICP
    degeneracy signal. Callers should gate acceptance on all of these, not
    just `converged` (a degenerate problem can "converge" to a wrong answer).
    """
    T = np.array(T_init, dtype=np.float64, copy=True)
    n_src = len(source_points)
    info = {"converged": False, "n_iters": 0, "inlier_ratio": 0.0,
            "mean_residual": float("inf"), "min_eigval": 0.0}
    if n_src == 0 or len(target_points) == 0:
        return T, info

    tree = cKDTree(target_points)

    for it in range(max_iters):
        info["n_iters"] = it + 1
        src_t = (T[:3, :3] @ source_points.T).T + T[:3, 3]
        dists, idx = tree.query(src_t, k=1)
        mask = dists <= max_corr_dist
        n_corr = int(np.count_nonzero(mask))
        if n_corr < 6:
            break

        p = src_t[mask]
        q = target_points[idx[mask]]
        n = target_normals[idx[mask]]
        r0 = np.sum(n * (p - q), axis=1)

        # Point-to-plane linearization: for a small rigid perturbation
        # p' = p + dtheta x p + dt applied on the left (world/target frame),
        # residual' ~= dot(n, p-q) + dtheta.(p x n) + dt.n.
        J = np.concatenate([np.cross(p, n), n], axis=1)  # (n_corr, 6): [dtheta | dt]
        H = J.T @ J
        g = -J.T @ r0
        info["min_eigval"] = float(np.linalg.eigvalsh(H)[0])
        info["inlier_ratio"] = n_corr / n_src
        info["mean_residual"] = float(np.mean(np.abs(r0)))

        try:
            x = np.linalg.solve(H + 1e-9 * np.eye(6), g)
        except np.linalg.LinAlgError:
            break

        T_delta = np.eye(4)
        T_delta[:3, :3] = rotvec_to_matrix(x[:3])
        T_delta[:3, 3] = x[3:]
        T = T_delta @ T

        if np.linalg.norm(x) < tol:
            info["converged"] = True
            break

    return T, info


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

def rerank_by_pnp_inliers(
    pnp_candidates: list[tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    min_point_count: int = 80,
    min_inlier_count: int = 50,
) -> tuple[bool, np.ndarray, float, int, int, int]:
    """
    Estimate PnP for each candidate and return the pose with the most inliers.

    Args:
        pnp_candidates: list of (points_3d, points_2d) pairs.
        K: camera intrinsic matrix.
        min_point_count: minimum number of 3D/2D correspondences required.
        min_inlier_count: minimum number of PnP inliers required.

    Returns:
        success, pose, inlier_ratio, best_candidate_index, best_inlier_count, best_point_count.
    """
    best_pose = None
    best_candidate_index = -1
    best_inlier_count = 0
    best_point_count = 0

    for candidate_index, (points_3d, points_2d) in enumerate(pnp_candidates):
        point_count = len(points_2d)
        if point_count <= min_point_count:
            continue

        success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, K, None)
        inlier_count = 0 if inliers is None else len(inliers)
        if not success or inliers is None or inlier_count < min_inlier_count:
            continue

        if inlier_count > best_inlier_count:
            best_candidate_index = candidate_index
            best_inlier_count = inlier_count
            best_point_count = point_count
            best_pose = np.eye(4)
            R_mat, _ = cv2.Rodrigues(rvec)
            best_pose[:3, :3] = R_mat
            best_pose[:3, 3] = tvec.reshape(3)

    if best_pose is None:
        return False, np.eye(4), -np.inf, -1, 0, 0

    return True, best_pose, best_inlier_count / best_point_count, best_candidate_index, best_inlier_count, best_point_count

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

# Union–find via fufpy (https://github.com/LuisScoccola/fufpy)
def uf_init(n):
    return fufpy.dynamic_partition_create(int(n))


def uf_union(a, b, uf, _rank=None):
    return fufpy.dynamic_partition_union(uf, int(a), int(b))


def uf_all_sets_list(uf, min_component_size=1):
    out = []
    for part in fufpy.dynamic_partition_parts(uf):
        if part.size >= int(min_component_size):
            out.append(np.sort(part).tolist())
    return out
