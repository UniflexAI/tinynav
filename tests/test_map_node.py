import sys
sys.path.append(".")
sys.path.append("/tinynav/tinynav/core")

from tinynav.tinynav_cpp_bind import pose_graph_solve
from tinynav.core.math_utils import theta_star
import numpy as np

def angle_diff_from_two_rotation_matrix(R1, R2):
    """Calculate the angle difference (in radians) between two rotation matrices."""
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    # Clamp the cosine value to the valid domain [-1, 1] to avoid NaNs due to numerical errors
    cos_theta = (trace - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle

def rad_to_deg(angle_rad):
    """Convert radians to degrees."""
    return angle_rad * (180.0 / np.pi)


def test_pose_graph_solve():
    camera_number = 64
    # generate a target pose for each camera
    target_pose = {0: np.eye(4)}
    for i in range(1, camera_number):
        target_pose[i] = np.eye(4)
        target_pose[i][:3, :3] = np.array([
            [np.cos(i * np.pi / camera_number), -np.sin(i * np.pi / camera_number), 0],
            [np.sin(i * np.pi / camera_number), np.cos(i * np.pi / camera_number), 0],
            [0, 0, 1]
        ])
        target_pose[i][:3, 3] = np.random.rand(3) * 0.1  # small random translation


    camera_poses = {k : np.eye(4) for k in range(camera_number)}

    relative_pose_constraints = []
    for i in range(camera_number):
        target_relative_pose = np.linalg.inv(target_pose[i]) @ target_pose[(i + 1) % camera_number]
        relative_pose_constraints.append(
            ((i + 1) % camera_number, i, target_relative_pose, np.array([1, 1, 1]), np.array([1, 1, 1]))
        )

    constant_camera_poses = {0: True}

    optimized_camera_pose = pose_graph_solve(
        camera_poses,
        relative_pose_constraints,
        constant_camera_poses,
        max_iteration_num = 100
    )

    # sort the optimized camera poses by their keys increased
    optimized_camera_pose = dict(sorted(optimized_camera_pose.items()))

    for camera_timestamp, pose in optimized_camera_pose.items():
        translation_error = np.linalg.norm(pose[:3, 3] - target_pose[camera_timestamp][:3, 3])
        rotation_error = rad_to_deg(angle_diff_from_two_rotation_matrix(
            pose[:3, :3], target_pose[camera_timestamp][:3, :3]
        ))
        assert translation_error < 1e-6, f"Translation error {translation_error} for camera {camera_timestamp} is too high."
        assert rotation_error < 1e-6, f"Rotation error {rotation_error} for camera {camera_timestamp} is too high."

def test_theta_star():
    cost_map = np.array([
        [1.0, 0.0, 0.5, 0.5, 1.0],
        [1.0, 1.0, 0.5, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5, 1.0],
        [0.5, 0.5, 0.5, 0.5, 1.0],
        [0.5, 0.5, 0.5, 0.5, 1.0],
        [0.5, 0.5, 0.5, 0.5, 1.0],
        [0.5, 0.5, 0.5, 0.5, 1.0],
        [0.5, 0.7, 0.6, 0.5, 1.0],
        [1.0, 1.0, 0.6, 0.5, 1.0],
        [0.0, 0.5, 0.6, 0.5, 1.0],
        [1.0, 1.0, 0, 0.5, 1.0],
    ])
    start = (0, 1)
    goal = (9, 0)
    path = theta_star(cost_map, start, goal, obstacles_cost=1.0)
    ground_truth_path = [(0, 1), (1, 2), (8, 2), (9, 1), (9, 0)]
    for i,point in enumerate(path):
        assert point[0] == ground_truth_path[i][0]
        assert point[1] == ground_truth_path[i][1]

def test_lookahead_rides_the_capture_speed():
    """The target pose is a carrot at a fixed TIME horizon, so its distance has to
    track the speed actually being driven — a flat 2.5m was 12.5s ahead wherever the
    operator crept at 0.2 m/s, aiming past the stretch that slowness was warning about.
    """
    from tinynav.core.map_node import lookahead_distance_m, _LOOKAHEAD_S

    # Constant horizon across the speeds planning can command ([vx_min, vx_hard_max]).
    for cap in (0.2, 0.5, 0.6, 1.0):
        assert abs(lookahead_distance_m(cap, gain=1.0) / cap - _LOOKAHEAD_S) < 1e-6

    # The old hardcoded 0.5 m/s is preserved exactly where it was right.
    assert abs(lookahead_distance_m(0.5, gain=1.0) - 2.5) < 1e-6

    # Faster than the hardware ceiling clamps rather than flinging the carrot away.
    assert lookahead_distance_m(3.0, gain=1.0) == 5.0
    assert lookahead_distance_m(0.01, gain=1.0) == 1.0

    # Off-path / no path_speed.npy: +inf is speed_cap's "no data" sentinel, and the
    # fallback matches planning's own (vx_max = 0.6).
    assert abs(lookahead_distance_m(float("inf"), gain=1.0) - 3.0) < 1e-6
    assert abs(lookahead_distance_m(float("nan"), gain=1.0) - 3.0) < 1e-6


def test_lookahead_rides_the_gained_speed():
    """The horizon is a TIME, and planning drives cap*gain -- so the carrot has to sit
    _LOOKAHEAD_S ahead at THAT speed. Reading the raw prior instead shortens the horizon
    by exactly 1/gain (at gain 1.5 the 5s carrot became 3.33s)."""
    from tinynav.core.map_node import lookahead_distance_m, _LOOKAHEAD_S, _NO_CAP_SPEED_MPS

    for gain in (1.0, 1.2, 1.5):
        for cap in (0.2, 0.4, 0.512, 0.6):
            driven = cap * gain
            horizon_s = lookahead_distance_m(cap, gain=gain) / driven
            assert abs(horizon_s - _LOOKAHEAD_S) < 1e-6, (cap, gain, horizon_s)

    # The no-prior fallback is planning's vx_max, which the gain does not touch, so it
    # must stay put as the gain moves.
    for gain in (1.0, 1.5, 3.0):
        assert abs(lookahead_distance_m(float("inf"), gain=gain)
                   - _NO_CAP_SPEED_MPS * _LOOKAHEAD_S) < 1e-6


if __name__ == "__main__":
    print("Running pose graph solve test...")
    test_pose_graph_solve()
    print("Pose graph solve test passed.")
    test_theta_star()
    print("A* test passed.")
    test_lookahead_rides_the_capture_speed()
    print("Lookahead test passed.")
