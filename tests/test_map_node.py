from tinynav.tinynav_cpp_bind import pose_graph_solve
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

if __name__ == "__main__":
    test_pose_graph_solve()
    print("Pose graph solve test passed.")
