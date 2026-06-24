from pathlib import Path

import cv2
import numpy as np

from tool.generate_stereo_vio_feature_dataset import (
    CameraModel,
    GeneratorConfig,
    LandmarkSet,
    _landmark_patch,
    generate_dataset,
    generate_trajectory,
    project_landmarks,
)


def test_stereo_projection_disparity_matches_baseline():
    camera = CameraModel(width=96, height=64, fx=80.0, fy=80.0, cx=48.0, cy=32.0, baseline=0.1)
    pose = np.eye(4, dtype=np.float64)
    landmarks = LandmarkSet(
        ids=np.array([0], dtype=np.int32),
        points_world=np.array([[0.0, 0.0, 4.0]], dtype=np.float64),
        texture_seeds=np.array([1], dtype=np.int64),
        radii=np.array([2], dtype=np.int16),
        intensities=np.array([180], dtype=np.uint8),
    )

    left_u, left_v, left_z = project_landmarks(landmarks, pose, camera)
    right_u, right_v, right_z = project_landmarks(
        landmarks,
        pose,
        camera,
        np.array([camera.baseline, 0.0, 0.0], dtype=np.float64),
    )

    assert np.allclose(left_z, right_z)
    assert np.allclose(left_v, right_v)
    assert np.allclose(left_u - right_u, camera.fx * camera.baseline / left_z)


def test_trajectory_timestamps_are_monotonic():
    samples = generate_trajectory(duration_sec=1.2, hz=100.0, start_timestamp_ns=123, gravity=9.8015)
    timestamps = np.array([sample.timestamp_ns for sample in samples])
    positions = np.stack([sample.pose_camera_to_world[:3, 3] for sample in samples], axis=0)
    assert len(samples) == 121
    assert np.all(np.diff(timestamps) > 0)
    assert np.allclose(positions[:, 2], 0.0)
    assert np.ptp(positions[:, 0]) > 0.01
    assert np.ptp(positions[:, 1]) > 0.01
    for sample in samples:
        quat = sample.pose_camera_to_world[:3, :3]
        assert np.allclose(quat.T @ quat, np.eye(3), atol=1e-6)


def test_planar_trajectory_holds_origin_for_static_start():
    samples = generate_trajectory(duration_sec=1.2, hz=100.0, start_timestamp_ns=123, gravity=9.8015)
    positions = np.stack([sample.pose_camera_to_world[:3, 3] for sample in samples], axis=0)
    static_mask = np.array([sample.t_sec <= 1.0 + 1e-12 for sample in samples])
    moving_mask = np.array([sample.t_sec > 1.0 + 1e-12 for sample in samples])

    assert np.allclose(positions[static_mask], 0.0)
    assert np.allclose([sample.pose_camera_to_world[:3, :3] for sample in np.array(samples, dtype=object)[static_mask]], np.eye(3))
    assert np.allclose([sample.velocity_world for sample in np.array(samples, dtype=object)[static_mask]], 0.0)
    assert np.allclose([sample.acceleration_world for sample in np.array(samples, dtype=object)[static_mask]], 0.0)
    assert np.allclose([sample.angular_velocity_camera for sample in np.array(samples, dtype=object)[static_mask]], 0.0)
    assert np.allclose(
        [sample.specific_force_camera for sample in np.array(samples, dtype=object)[static_mask]],
        np.array([0.0, 0.0, 9.8015]),
    )
    assert np.linalg.norm(positions[moving_mask][0, :2]) > 0.0


def test_trajectory_pose_maps_camera_frame_into_world_frame():
    samples = generate_trajectory(duration_sec=1.2, hz=100.0, start_timestamp_ns=123, gravity=9.8015)
    sample = next(sample for sample in samples if sample.t_sec > 1.0 and np.linalg.norm(sample.velocity_world[:2]) > 1e-6)
    r_world_camera = sample.pose_camera_to_world[:3, :3]
    camera_forward_world = r_world_camera @ np.array([0.0, 0.0, 1.0])
    camera_down_world = r_world_camera @ np.array([0.0, 1.0, 0.0])
    velocity_xy = sample.velocity_world.copy()
    velocity_xy[2] = 0.0
    velocity_xy /= np.linalg.norm(velocity_xy)

    assert np.dot(camera_forward_world, velocity_xy) > 0.99
    assert np.allclose(camera_down_world, np.array([0.0, 0.0, 1.0]), atol=1e-6)


def test_quadratic_patch_has_convex_anchor():
    radius = 7
    patch, alpha = _landmark_patch(texture_seed=17, radius=radius, intensity=180, patch_type="quadratic")
    image = patch.astype(np.float32)
    c = radius
    dx = 0.5 * (image[c, c + 1] - image[c, c - 1])
    dy = 0.5 * (image[c + 1, c] - image[c - 1, c])
    dxx = image[c, c + 1] - 2.0 * image[c, c] + image[c, c - 1]
    dyy = image[c + 1, c] - 2.0 * image[c, c] + image[c - 1, c]
    dxy = 0.25 * (image[c + 1, c + 1] - image[c + 1, c - 1] - image[c - 1, c + 1] + image[c - 1, c - 1])

    assert abs(dx) <= 1.0
    assert abs(dy) <= 1.0
    assert dxx > 0.0
    assert dyy > 0.0
    assert dxx * dyy - dxy * dxy > 0.0
    assert alpha[c, c] == 1.0


def test_x_junction_patch_has_center_anchor_and_no_square_boundary():
    radius = 7
    patch, alpha = _landmark_patch(texture_seed=17, radius=radius, intensity=180, patch_type="x_junction")
    c = radius

    assert patch[c, c] >= 120
    assert alpha[c, c] > 0.9
    assert np.max(alpha[0, :]) < 0.5
    assert np.max(alpha[-1, :]) < 0.5
    assert np.max(alpha[:, 0]) < 0.5
    assert np.max(alpha[:, -1]) < 0.5


def test_generate_dataset_sidecars_without_rosbag(tmp_path: Path):
    out = tmp_path / "synthetic_feature_dataset"
    cfg = GeneratorConfig(
        output_bag=str(out),
        duration_sec=1.2,
        camera_hz=5.0,
        imu_hz=50.0,
        width=96,
        height=64,
        fx=80.0,
        fy=80.0,
        cx=48.0,
        cy=32.0,
        baseline=0.1,
        num_points=80,
        min_depth=1.5,
        max_depth=4.0,
        patch_radius=2,
        landmark_surface="camera_rays",
        seed=7,
        skip_bag=True,
    )
    metadata = generate_dataset(cfg)

    assert (out / "metadata.json").exists()
    assert (out / "observations.npz").exists()
    assert (out / "matches_gt.npz").exists()
    assert (out / "landmarks.csv").exists()
    assert (out / "trajectory_tum.txt").exists()
    assert len(list((out / "depth_gt" / "infra1").glob("*.npz"))) > 0
    assert len(list((out / "depth_gt" / "infra2").glob("*.png"))) > 0
    assert metadata["counts"]["camera_frames"] > 0
    assert metadata["counts"]["visible_observations"] > 0
    assert metadata["config"]["background_type"] == "textured_cylinder"
    assert metadata["config"]["static_start_sec"] == 1.0
    assert metadata["static_start_sec"] == 1.0
    assert metadata["trajectory_origin_policy"] == "origin_static_then_planar_analytic"
    assert metadata["planar_motion_time_origin_sec"] == 1.0
    assert metadata["imu_accel_convention"] == "perception_linear_acceleration_camera = R_cw @ (gravity_world - acceleration_world)"
    assert metadata["pose_convention"]["name"] == "T_world_camera"
    assert metadata["pose_convention"]["odom_frame_id"] == "world"
    assert metadata["pose_convention"]["odom_child_frame_id"] == "camera"
    observations = np.load(out / "observations.npz")
    assert {"timestamp_ns", "camera_id", "landmark_id", "u", "v", "feature_u", "feature_v", "visible"}.issubset(observations.files)
    assert np.allclose(observations["feature_u"], np.rint(observations["u"]))
    assert np.allclose(observations["feature_v"], np.rint(observations["v"]))
    image = cv2.imread(str(next((out / "images" / "infra1").glob("*.png"))), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    assert float(np.std(image)) > 8.0
    depth = np.load(next((out / "depth_gt" / "infra1").glob("*.npz")))["depth_m"]
    assert depth.shape == image.shape
    assert np.count_nonzero(depth > 0.0) > 0


def test_cylinder_landmarks_lie_on_circular_wall(tmp_path: Path):
    out = tmp_path / "cylinder_feature_dataset"
    radius = 4.0
    cfg = GeneratorConfig(
        output_bag=str(out),
        duration_sec=0.5,
        camera_hz=5.0,
        imu_hz=50.0,
        width=160,
        height=120,
        fx=120.0,
        fy=120.0,
        cx=80.0,
        cy=60.0,
        baseline=0.1,
        num_points=120,
        min_depth=1.0,
        max_depth=6.0,
        patch_radius=2,
        background_type="textured_cylinder",
        background_cylinder_radius_m=radius,
        landmark_surface="cylinder",
        static_start_sec=0.0,
        wall_z_min_m=-2.0,
        wall_z_max_m=2.0,
        seed=13,
        skip_bag=True,
    )
    metadata = generate_dataset(cfg)
    landmarks = np.genfromtxt(out / "landmarks.csv", delimiter=",", names=True)
    radial_distance = np.sqrt(landmarks["x"] * landmarks["x"] + landmarks["y"] * landmarks["y"])

    assert metadata["resolved_landmark_surface"] == "cylinder"
    assert np.allclose(radial_distance, radius, atol=1e-6)
    assert np.all(landmarks["z"] >= cfg.wall_z_min_m - 1e-6)
    assert np.all(landmarks["z"] <= cfg.wall_z_max_m + 1e-6)
    assert metadata["counts"]["visible_observations"] > 0
    depth = np.load(next((out / "depth_gt" / "infra1").glob("*.npz")))["depth_m"]
    assert depth.shape == (cfg.height, cfg.width)
    assert float(np.median(depth[depth > 0.0])) > 1.0


def test_wall_landmarks_lie_on_walls(tmp_path: Path):
    out = tmp_path / "wall_feature_dataset"
    half_extent = 4.0
    cfg = GeneratorConfig(
        output_bag=str(out),
        duration_sec=0.5,
        camera_hz=5.0,
        imu_hz=50.0,
        width=160,
        height=120,
        fx=120.0,
        fy=120.0,
        cx=80.0,
        cy=60.0,
        baseline=0.1,
        num_points=120,
        min_depth=1.0,
        max_depth=6.0,
        patch_radius=2,
        background_type="textured_walls",
        landmark_surface="walls",
        static_start_sec=0.0,
        wall_half_extent_m=half_extent,
        wall_z_min_m=-2.0,
        wall_z_max_m=2.0,
        seed=11,
        skip_bag=True,
    )
    metadata = generate_dataset(cfg)
    landmarks = np.genfromtxt(out / "landmarks.csv", delimiter=",", names=True)
    on_x_wall = np.isclose(np.abs(landmarks["x"]), half_extent, atol=1e-6)
    on_y_wall = np.isclose(np.abs(landmarks["y"]), half_extent, atol=1e-6)

    assert metadata["resolved_landmark_surface"] == "walls"
    assert np.all(on_x_wall | on_y_wall)
    assert np.all(landmarks["z"] >= cfg.wall_z_min_m - 1e-6)
    assert np.all(landmarks["z"] <= cfg.wall_z_max_m + 1e-6)
    assert metadata["counts"]["visible_observations"] > 0
    depth = np.load(next((out / "depth_gt" / "infra1").glob("*.npz")))["depth_m"]
    assert depth.shape == (cfg.height, cfg.width)
    assert np.count_nonzero(depth > 0.0) > 0
