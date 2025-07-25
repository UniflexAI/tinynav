import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

@njit
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

@njit
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

@njit
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

def msg2np(msg):
    T = np.eye(4)
    position = msg.pose.pose.position
    rot = msg.pose.pose.orientation
    quat = [rot.x, rot.y, rot.z, rot.w]
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = np.array([position.x, position.y, position.z]).ravel()
    return T
    
