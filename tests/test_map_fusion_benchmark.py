import sys

import numpy as np
import pytest

sys.path.append(".")
sys.path.append("tool/benchmark")

from map_fusion_benchmark import (
    _estimate_rigid_transform,
    _estimate_se2_z_transform,
    _ransac_transform,
    _rotation_error_deg,
    _summary,
    _threshold_stats,
)


def _random_rigid_transform(rng, translation_scale=5.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-np.pi, np.pi)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = rng.uniform(-translation_scale, translation_scale, size=3)
    return transform


def test_estimate_rigid_transform_recovers_known_transform():
    rng = np.random.default_rng(0)
    transform = _random_rigid_transform(rng)
    points_src = rng.uniform(-10, 10, size=(8, 3))
    points_dst = (transform[:3, :3] @ points_src.T).T + transform[:3, 3]

    recovered = _estimate_rigid_transform(points_src, points_dst)

    np.testing.assert_allclose(recovered, transform, atol=1e-9)


def test_estimate_se2_z_transform_recovers_planar_transform():
    rng = np.random.default_rng(1)
    yaw = 0.7
    rot_2d = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    translation_xy = np.array([2.0, -3.0])
    z_offset = 1.5

    points_src = rng.uniform(-10, 10, size=(6, 3))
    points_dst = points_src.copy()
    points_dst[:, :2] = (rot_2d @ points_src[:, :2].T).T + translation_xy
    points_dst[:, 2] = points_src[:, 2] + z_offset

    transform = _estimate_se2_z_transform(points_src, points_dst)

    np.testing.assert_allclose(transform[:2, :2], rot_2d, atol=1e-9)
    np.testing.assert_allclose(transform[:2, 3], translation_xy, atol=1e-9)
    np.testing.assert_allclose(transform[2, 3], z_offset, atol=1e-9)


def test_ransac_transform_is_robust_to_outliers():
    rng = np.random.default_rng(2)
    transform = _random_rigid_transform(rng, translation_scale=3.0)
    timestamps = list(range(20))
    source_poses = {}
    target_poses = {}
    for i, ts in enumerate(timestamps):
        pose_src = np.eye(4)
        pose_src[:3, 3] = rng.uniform(-10, 10, size=3)
        pose_dst = np.eye(4)
        pose_dst[:3, 3] = (transform[:3, :3] @ pose_src[:3, 3]) + transform[:3, 3]
        if i < 4:
            # corrupt a minority of correspondences with large outlier noise
            pose_dst[:3, 3] += rng.uniform(5, 10, size=3)
        source_poses[ts] = pose_src
        target_poses[ts] = pose_dst

    recovered, inlier_timestamps, info = _ransac_transform(
        source_poses=source_poses,
        target_poses=target_poses,
        inlier_threshold_m=0.05,
        iterations=500,
        seed=7,
        alignment_mode="se3",
    )

    np.testing.assert_allclose(recovered[:3, :3], transform[:3, :3], atol=1e-6)
    np.testing.assert_allclose(recovered[:3, 3], transform[:3, 3], atol=1e-6)
    assert info["inlier_count"] == 16
    assert set(inlier_timestamps) == set(timestamps[4:])


def test_rotation_error_deg_identity_is_zero():
    identity = np.eye(3)
    assert _rotation_error_deg(identity, identity) == 0.0


def test_rotation_error_deg_matches_known_angle():
    angle_deg = 30.0
    theta = np.radians(angle_deg)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    error = _rotation_error_deg(np.eye(3), rot)
    assert error == pytest.approx(angle_deg)


def test_threshold_stats_counts_and_ratios():
    errors = [{"translation_error_m": v} for v in [0.01, 0.05, 0.15, 0.5]]
    stats = _threshold_stats(errors, [0.05, 0.20])
    assert stats["0.05m"] == {"count": 2, "ratio": 0.5}
    assert stats["0.20m"] == {"count": 3, "ratio": 0.75}


def test_threshold_stats_empty_errors():
    stats = _threshold_stats([], [0.05])
    assert stats["0.05m"] == {"count": 0, "ratio": 0.0}


def test_summary_basic_stats():
    result = _summary([1.0, 2.0, 3.0, 4.0])
    assert result["count"] == 4
    assert result["mean"] == 2.5
    assert result["median"] == 2.5
    assert result["max"] == 4.0


def test_summary_empty_returns_nones():
    result = _summary([])
    assert result["count"] == 0
    assert result["mean"] is None
