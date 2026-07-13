#!/usr/bin/env python3
"""
tool/qr_odom/pose_utils.py

Shared SE3-averaging helpers used by every qr_odom node/script that collects
multiple pose samples of the same physical thing (a board, an anchor) and
needs a single averaged pose out.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def mean_T(samples: list[np.ndarray]) -> np.ndarray:
    """Average a list of 4x4 SE3 matrices: mean position + mean rotation."""
    p_mean = np.mean([T[:3, 3] for T in samples], axis=0)
    rots   = Rotation.concatenate([Rotation.from_matrix(T[:3, :3]) for T in samples])
    R_mean = rots.mean().as_matrix()
    T_mean = np.eye(4)
    T_mean[:3, :3] = R_mean
    T_mean[:3,  3] = p_mean
    return T_mean


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle (deg) of the relative rotation between two 3x3 rotation matrices."""
    relative = Rotation.from_matrix(R_a.T @ R_b)
    return float(np.degrees(relative.magnitude()))
