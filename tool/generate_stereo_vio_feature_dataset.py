#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


LEFT_TOPIC = "/camera/camera/infra1/image_rect_raw"
RIGHT_TOPIC = "/camera/camera/infra2/image_rect_raw"
LEFT_INFO_TOPIC = "/camera/camera/infra1/camera_info"
RIGHT_INFO_TOPIC = "/camera/camera/infra2/camera_info"
IMU_TOPIC = "/camera/camera/imu"
GT_ODOM_TOPIC = "/ground_truth/odometry"
POSE_CONVENTION = {
    "name": "T_world_camera",
    "description": "Camera pose in world frame; maps camera-frame points into world-frame points.",
    "equation": "p_world = R_world_camera @ p_camera + t_world_camera",
    "odom_frame_id": "world",
    "odom_child_frame_id": "camera",
}

MATCH_STEREO = 0
MATCH_TEMPORAL_LEFT = 1
MATCH_TEMPORAL_RIGHT = 2
PATCH_TYPES = ("corner", "junction", "l_junction", "t_junction", "x_junction", "quadratic", "checker", "texture")
TRAJECTORY_MODES = ("planar_xy", "spatial_3d")
BACKGROUND_TYPES = ("textured_cylinder", "textured_walls", "gradient")
LANDMARK_SURFACES = ("auto", "camera_rays", "cylinder", "walls")

CAMERA_PRESETS: dict[str, dict[str, float | int]] = {
    "looper": {
        "width": 544,
        "height": 640,
        "fx": 305.95681763,
        "fy": 305.95681763,
        "cx": 267.96417236,
        "cy": 314.43896484,
        "baseline": 0.10831039,
    },
    "realsense": {
        "width": 848,
        "height": 480,
        "fx": 423.9984,
        "fy": 423.9984,
        "cx": 424.0,
        "cy": 240.0,
        "baseline": 0.051,
    },
}


@dataclass(frozen=True)
class CameraModel:
    width: int = 544
    height: int = 640
    fx: float = 305.95681763
    fy: float = 305.95681763
    cx: float = 267.96417236
    cy: float = 314.43896484
    baseline: float = 0.10831039

    @property
    def k(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class GeneratorConfig:
    output_bag: str
    camera_preset: str = "looper"
    duration_sec: float = 18.0
    camera_hz: float = 30.0
    imu_hz: float = 400.0
    width: int = 544
    height: int = 640
    fx: float = 305.95681763
    fy: float = 305.95681763
    cx: float = 267.96417236
    cy: float = 314.43896484
    baseline: float = 0.10831039
    num_points: int = 512
    min_depth: float = 2.0
    max_depth: float = 8.0
    patch_radius: int = 7
    patch_type: str = "x_junction"
    trajectory_mode: str = "planar_xy"
    static_start_sec: float = 1.0
    seed: int = 0
    start_timestamp_ns: int = 100_000_000_000
    gravity: float = 9.8015
    accel_noise_sigma: float = 0.0
    gyro_noise_sigma: float = 0.0
    image_noise_sigma: float = 1.0
    brightness_jitter: float = 3.0
    background_type: str = "textured_cylinder"
    background_texture_strength: float = 52.0
    background_cylinder_radius_m: float = 7.0
    background_texture_scale: float = 9.0
    wall_half_extent_m: float = 5.0
    wall_z_min_m: float = -2.2
    wall_z_max_m: float = 2.2
    landmark_surface: str = "auto"
    skip_bag: bool = False
    overwrite: bool = False

    @property
    def camera(self) -> CameraModel:
        return CameraModel(
            width=self.width,
            height=self.height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            baseline=self.baseline,
        )


@dataclass(frozen=True)
class TrajectorySample:
    timestamp_ns: int
    t_sec: float
    pose_camera_to_world: np.ndarray
    velocity_world: np.ndarray
    acceleration_world: np.ndarray
    angular_velocity_camera: np.ndarray
    specific_force_camera: np.ndarray


@dataclass(frozen=True)
class LandmarkSet:
    ids: np.ndarray
    points_world: np.ndarray
    texture_seeds: np.ndarray
    radii: np.ndarray
    intensities: np.ndarray


def parse_args() -> GeneratorConfig:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--camera_preset", choices=sorted(CAMERA_PRESETS.keys()), default="looper")
    preset_args, _ = preset_parser.parse_known_args()
    camera_defaults = CAMERA_PRESETS[preset_args.camera_preset]

    parser = argparse.ArgumentParser(
        description="Generate a synthetic textured stereo/IMU ROS 2 bag with GT landmark projections.",
        parents=[preset_parser],
    )
    parser.add_argument("--output_bag", required=True, help="Output ROS 2 bag directory. Sidecars are written inside it.")
    parser.add_argument("--duration_sec", type=float, default=18.0)
    parser.add_argument("--camera_hz", type=float, default=30.0)
    parser.add_argument("--imu_hz", type=float, default=400.0)
    parser.add_argument("--width", type=int, default=int(camera_defaults["width"]))
    parser.add_argument("--height", type=int, default=int(camera_defaults["height"]))
    parser.add_argument("--fx", type=float, default=float(camera_defaults["fx"]))
    parser.add_argument("--fy", type=float, default=float(camera_defaults["fy"]))
    parser.add_argument("--cx", type=float, default=float(camera_defaults["cx"]))
    parser.add_argument("--cy", type=float, default=float(camera_defaults["cy"]))
    parser.add_argument("--baseline", type=float, default=float(camera_defaults["baseline"]))
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--min_depth", type=float, default=2.0)
    parser.add_argument("--max_depth", type=float, default=8.0)
    parser.add_argument("--patch_radius", type=int, default=7)
    parser.add_argument(
        "--patch_type",
        choices=PATCH_TYPES,
        default="x_junction",
        help="Landmark image patch renderer. corner/junction draw sharp L/T/X junctions; quadratic keeps the convex anchor; texture keeps the old random blob renderer.",
    )
    parser.add_argument(
        "--trajectory_mode",
        choices=TRAJECTORY_MODES,
        default="planar_xy",
        help="planar_xy keeps GT z=0 and moves in x/y; spatial_3d keeps the previous 3D path.",
    )
    parser.add_argument("--static_start_sec", type=float, default=1.0, help="planar_xy origin hold before motion starts.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_timestamp_ns", type=int, default=100_000_000_000)
    parser.add_argument("--gravity", type=float, default=9.8015)
    parser.add_argument("--accel_noise_sigma", type=float, default=0.0)
    parser.add_argument("--gyro_noise_sigma", type=float, default=0.0)
    parser.add_argument("--image_noise_sigma", type=float, default=1.0)
    parser.add_argument("--brightness_jitter", type=float, default=3.0)
    parser.add_argument(
        "--background_type",
        choices=BACKGROUND_TYPES,
        default="textured_cylinder",
        help="textured_cylinder and textured_walls render stereo-consistent procedural world surfaces; gradient keeps the old smooth background.",
    )
    parser.add_argument("--background_texture_strength", type=float, default=52.0)
    parser.add_argument("--background_cylinder_radius_m", type=float, default=7.0)
    parser.add_argument("--background_texture_scale", type=float, default=9.0)
    parser.add_argument("--wall_half_extent_m", type=float, default=5.0)
    parser.add_argument("--wall_z_min_m", type=float, default=-2.2)
    parser.add_argument("--wall_z_max_m", type=float, default=2.2)
    parser.add_argument(
        "--landmark_surface",
        choices=LANDMARK_SURFACES,
        default="auto",
        help="auto places landmarks on the rendered surface for textured_cylinder/textured_walls.",
    )
    parser.add_argument("--skip_bag", action="store_true", help="Write sidecars/images only; useful for geometry tests without ROS.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return GeneratorConfig(**vars(args))


def _spatial_rotation_from_time(t: np.ndarray) -> np.ndarray:
    roll = 0.08 * np.sin(0.7 * t)
    pitch = 0.06 * np.sin(0.45 * t + 0.3)
    yaw = 0.18 * np.sin(0.35 * t)
    return Rotation.from_euler("xyz", np.stack([roll, pitch, yaw], axis=-1)).as_matrix()


def _spatial_position_from_time(t: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            0.45 * np.sin(0.5 * t) + 0.12 * np.sin(1.7 * t),
            0.22 * np.sin(0.6 * t + 0.4),
            0.22 * t + 0.30 * np.sin(0.35 * t),
        ],
        axis=-1,
    )


def _planar_xy_kinematics_from_time(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = 1.10 * np.sin(0.38 * t) + 0.35 * np.sin(0.93 * t)
    y = 0.80 * np.sin(0.52 * t + 0.35) + 0.30 * np.sin(1.17 * t)
    vx = 1.10 * 0.38 * np.cos(0.38 * t) + 0.35 * 0.93 * np.cos(0.93 * t)
    vy = 0.80 * 0.52 * np.cos(0.52 * t + 0.35) + 0.30 * 1.17 * np.cos(1.17 * t)
    ax = -1.10 * 0.38 * 0.38 * np.sin(0.38 * t) - 0.35 * 0.93 * 0.93 * np.sin(0.93 * t)
    ay = -0.80 * 0.52 * 0.52 * np.sin(0.52 * t + 0.35) - 0.30 * 1.17 * 1.17 * np.sin(1.17 * t)
    return (
        np.stack([x, y, np.zeros_like(t)], axis=-1),
        np.stack([vx, vy, np.zeros_like(t)], axis=-1),
        np.stack([ax, ay, np.zeros_like(t)], axis=-1),
    )


def _planar_xy_position_from_time(t: np.ndarray) -> np.ndarray:
    positions, _, _ = _planar_xy_kinematics_from_time(t)
    return positions


def _planar_xy_rotation_from_velocity(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(vy, vx)
    c = np.cos(yaw)
    s = np.sin(yaw)
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], len(vx), axis=0)
    moving = (vx * vx + vy * vy) > 1e-12
    rotations[moving, :, 0] = np.stack([-s[moving], c[moving], np.zeros_like(s[moving])], axis=-1)
    rotations[moving, :, 1] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rotations[moving, :, 2] = np.stack([c[moving], s[moving], np.zeros_like(s[moving])], axis=-1)
    return rotations


def _planar_xy_motion_from_time(
    t: np.ndarray,
    static_start_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if static_start_sec < 0.0:
        raise ValueError(f"static_start_sec must be non-negative, got {static_start_sec}")

    positions = np.zeros((len(t), 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    angular_velocities = np.zeros_like(positions)
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], len(t), axis=0)

    moving = np.ones(len(t), dtype=bool) if static_start_sec <= 0.0 else t > static_start_sec
    if not np.any(moving):
        return positions, rotations, velocities, accelerations, angular_velocities

    motion_t = t[moving] - max(static_start_sec, 0.0)
    motion_positions, motion_velocities, motion_accelerations = _planar_xy_kinematics_from_time(motion_t)
    origin_position, _, _ = _planar_xy_kinematics_from_time(np.array([0.0], dtype=np.float64))
    motion_positions = motion_positions - origin_position[0]

    positions[moving] = motion_positions
    velocities[moving] = motion_velocities
    accelerations[moving] = motion_accelerations
    rotations[moving] = _planar_xy_rotation_from_velocity(motion_velocities[:, 0], motion_velocities[:, 1])

    vx = motion_velocities[:, 0]
    vy = motion_velocities[:, 1]
    ax = motion_accelerations[:, 0]
    ay = motion_accelerations[:, 1]
    denom = vx * vx + vy * vy
    valid = denom > 1e-12
    yaw_rate = np.zeros_like(vx)
    yaw_rate[valid] = (vx[valid] * ay[valid] - vy[valid] * ax[valid]) / denom[valid]
    angular_velocities[moving, 1] = yaw_rate

    return positions, rotations, velocities, accelerations, angular_velocities


def _planar_xy_rotation_from_time(t: np.ndarray) -> np.ndarray:
    _, velocities, _ = _planar_xy_kinematics_from_time(t)
    return _planar_xy_rotation_from_velocity(velocities[:, 0], velocities[:, 1])


def _trajectory_from_time(
    t: np.ndarray,
    mode: str,
    static_start_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if mode == "spatial_3d":
        return (
            _spatial_position_from_time(t),
            _spatial_rotation_from_time(t),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            None,
            None,
            None,
        )
    if mode == "planar_xy":
        positions, rotations, velocities, accelerations, angular_velocities = _planar_xy_motion_from_time(t, static_start_sec)
        return (
            positions,
            rotations,
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            velocities,
            accelerations,
            angular_velocities,
        )
    raise ValueError(f"Unsupported trajectory_mode: {mode}")


def _make_times(duration_sec: float, hz: float) -> np.ndarray:
    if duration_sec <= 0.0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}")
    if hz <= 0.0:
        raise ValueError(f"hz must be positive, got {hz}")
    count = int(math.floor(duration_sec * hz)) + 1
    return np.arange(count, dtype=np.float64) / float(hz)


def generate_trajectory(
    duration_sec: float,
    hz: float,
    start_timestamp_ns: int,
    gravity: float,
    trajectory_mode: str = "planar_xy",
    static_start_sec: float = 1.0,
) -> list[TrajectorySample]:
    times = _make_times(duration_sec, hz)
    positions, rotations, gravity_axis_world, velocities, accelerations, angular_velocities = _trajectory_from_time(
        times,
        trajectory_mode,
        static_start_sec,
    )
    dt = 1.0 / float(hz)
    if velocities is None:
        velocities = np.gradient(positions, dt, axis=0, edge_order=2)
    if accelerations is None:
        accelerations = np.gradient(velocities, dt, axis=0, edge_order=2)

    if angular_velocities is None:
        angular_velocities = np.zeros_like(positions)
        if len(times) > 1:
            rot_objs = Rotation.from_matrix(rotations)
            for i in range(len(times)):
                if i == 0:
                    delta = rot_objs[i].inv() * rot_objs[i + 1]
                    angular_velocities[i] = delta.as_rotvec() / dt
                elif i == len(times) - 1:
                    delta = rot_objs[i - 1].inv() * rot_objs[i]
                    angular_velocities[i] = delta.as_rotvec() / dt
                else:
                    delta = rot_objs[i - 1].inv() * rot_objs[i + 1]
                    angular_velocities[i] = delta.as_rotvec() / (2.0 * dt)

    gravity_world = float(gravity) * gravity_axis_world
    samples: list[TrajectorySample] = []
    for t, p, r, v, a, w in zip(times, positions, rotations, velocities, accelerations, angular_velocities):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = r
        pose[:3, 3] = p
        # Perception initializes gravity directly from linear_acceleration.
        # Static identity therefore needs +g on camera z, not physical specific force -g.
        specific_force = r.T @ (gravity_world - a)
        samples.append(
            TrajectorySample(
                timestamp_ns=int(start_timestamp_ns + round(t * 1e9)),
                t_sec=float(t),
                pose_camera_to_world=pose,
                velocity_world=v.astype(np.float64),
                acceleration_world=a.astype(np.float64),
                angular_velocity_camera=w.astype(np.float64),
                specific_force_camera=specific_force.astype(np.float64),
            )
        )
    return samples


def _camera_point_to_world(point_camera: np.ndarray, pose_camera_to_world: np.ndarray) -> np.ndarray:
    return pose_camera_to_world[:3, :3] @ point_camera + pose_camera_to_world[:3, 3]


def _intersect_room_walls(
    origin: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    wall_half_extent_m: float,
    wall_z_min_m: float,
    wall_z_max_m: float,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    half = float(wall_half_extent_m)
    z_min = float(wall_z_min_m)
    z_max = float(wall_z_max_m)
    best_t = np.full(np.shape(dx), np.inf, dtype=np.float64)
    best_wall = np.full(np.shape(dx), -1, dtype=np.int16)

    def update(candidate_t: np.ndarray, valid: np.ndarray, wall_id: int) -> None:
        nonlocal best_t, best_wall
        valid = valid & np.isfinite(candidate_t) & (candidate_t >= min_depth) & (candidate_t <= max_depth)
        take = valid & (candidate_t < best_t)
        best_t = np.where(take, candidate_t, best_t)
        best_wall = np.where(take, wall_id, best_wall)

    with np.errstate(divide="ignore", invalid="ignore"):
        for wall_id, x_plane in enumerate((-half, half)):
            t = (x_plane - origin[0]) / dx
            y = origin[1] + t * dy
            z = origin[2] + t * dz
            update(t, (np.abs(y) <= half) & (z >= z_min) & (z <= z_max), wall_id)

        for offset, y_plane in enumerate((-half, half), start=2):
            t = (y_plane - origin[1]) / dy
            x = origin[0] + t * dx
            z = origin[2] + t * dz
            update(t, (np.abs(x) <= half) & (z >= z_min) & (z <= z_max), offset)

    return best_t, best_wall


def _intersect_camera_rays_with_walls(
    pose_camera_to_world: np.ndarray,
    camera: CameraModel,
    u: np.ndarray,
    v: np.ndarray,
    min_depth: float,
    max_depth: float,
    wall_half_extent_m: float,
    wall_z_min_m: float,
    wall_z_max_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_cam = (u.astype(np.float64) - camera.cx) / camera.fx
    y_cam = (v.astype(np.float64) - camera.cy) / camera.fy
    r_wc = pose_camera_to_world[:3, :3]
    origin = pose_camera_to_world[:3, 3]
    dx = r_wc[0, 0] * x_cam + r_wc[0, 1] * y_cam + r_wc[0, 2]
    dy = r_wc[1, 0] * x_cam + r_wc[1, 1] * y_cam + r_wc[1, 2]
    dz = r_wc[2, 0] * x_cam + r_wc[2, 1] * y_cam + r_wc[2, 2]
    depth, wall_id = _intersect_room_walls(
        origin,
        dx,
        dy,
        dz,
        wall_half_extent_m,
        wall_z_min_m,
        wall_z_max_m,
        min_depth,
        max_depth,
    )
    points = np.stack([origin[0] + depth * dx, origin[1] + depth * dy, origin[2] + depth * dz], axis=-1)
    valid = wall_id >= 0
    return points, depth, valid


def _intersect_camera_rays_with_cylinder(
    pose_camera_to_world: np.ndarray,
    camera: CameraModel,
    u: np.ndarray,
    v: np.ndarray,
    min_depth: float,
    max_depth: float,
    cylinder_radius_m: float,
    wall_z_min_m: float,
    wall_z_max_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_cam = (u.astype(np.float64) - camera.cx) / camera.fx
    y_cam = (v.astype(np.float64) - camera.cy) / camera.fy
    r_wc = pose_camera_to_world[:3, :3]
    origin = pose_camera_to_world[:3, 3]
    dx = r_wc[0, 0] * x_cam + r_wc[0, 1] * y_cam + r_wc[0, 2]
    dy = r_wc[1, 0] * x_cam + r_wc[1, 1] * y_cam + r_wc[1, 2]
    dz = r_wc[2, 0] * x_cam + r_wc[2, 1] * y_cam + r_wc[2, 2]

    radius = max(float(cylinder_radius_m), 1.0)
    a = dx * dx + dy * dy
    b = 2.0 * (origin[0] * dx + origin[1] * dy)
    c = origin[0] * origin[0] + origin[1] * origin[1] - radius * radius
    disc = b * b - 4.0 * a * c
    valid_disc = disc >= 0.0
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    denom = np.maximum(2.0 * a, 1e-9)
    t0 = (-b - sqrt_disc) / denom
    t1 = (-b + sqrt_disc) / denom
    depth = np.where(t0 > 0.0, t0, t1)
    points = np.stack([origin[0] + depth * dx, origin[1] + depth * dy, origin[2] + depth * dz], axis=-1)
    valid = (
        valid_disc
        & np.isfinite(depth)
        & (depth >= float(min_depth))
        & (depth <= float(max_depth))
        & (points[:, 2] >= float(wall_z_min_m))
        & (points[:, 2] <= float(wall_z_max_m))
    )
    return points, depth, valid


def generate_landmarks(
    camera_samples: list[TrajectorySample],
    camera: CameraModel,
    num_points: int,
    min_depth: float,
    max_depth: float,
    patch_radius: int,
    seed: int,
    landmark_surface: str = "camera_rays",
    cylinder_radius_m: float = 7.0,
    wall_half_extent_m: float = 5.0,
    wall_z_min_m: float = -2.2,
    wall_z_max_m: float = 2.2,
) -> LandmarkSet:
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")
    if min_depth <= 0.0 or max_depth <= min_depth:
        raise ValueError(f"invalid depth range: [{min_depth}, {max_depth}]")
    rng = np.random.default_rng(seed)
    margin = max(8, int(patch_radius) + 4)
    if landmark_surface in {"cylinder", "walls"}:
        accepted: list[np.ndarray] = []
        attempts = 0
        batch_size = max(num_points * 3, 256)
        while sum(len(batch) for batch in accepted) < num_points and attempts < 80:
            attempts += 1
            frame_indices = rng.integers(0, len(camera_samples), size=batch_size)
            u = rng.uniform(margin, camera.width - margin, size=batch_size)
            v = rng.uniform(margin, camera.height - margin, size=batch_size)
            batch_points = []
            for frame_idx in np.unique(frame_indices):
                mask = frame_indices == int(frame_idx)
                if landmark_surface == "cylinder":
                    points_world, _, valid = _intersect_camera_rays_with_cylinder(
                        camera_samples[int(frame_idx)].pose_camera_to_world,
                        camera,
                        u[mask],
                        v[mask],
                        min_depth,
                        max_depth,
                        cylinder_radius_m,
                        wall_z_min_m,
                        wall_z_max_m,
                    )
                else:
                    points_world, _, valid = _intersect_camera_rays_with_walls(
                        camera_samples[int(frame_idx)].pose_camera_to_world,
                        camera,
                        u[mask],
                        v[mask],
                        min_depth,
                        max_depth,
                        wall_half_extent_m,
                        wall_z_min_m,
                        wall_z_max_m,
                    )
                if np.any(valid):
                    batch_points.append(points_world[valid])
            if batch_points:
                accepted.append(np.concatenate(batch_points, axis=0))
        if not accepted:
            raise RuntimeError(f"Could not sample any {landmark_surface} landmarks; check surface/depth/camera settings.")
        points = np.concatenate(accepted, axis=0)[:num_points].astype(np.float64)
        if len(points) < num_points:
            raise RuntimeError(f"Only sampled {len(points)} {landmark_surface} landmarks out of requested {num_points}")
    elif landmark_surface == "camera_rays":
        frame_indices = rng.integers(0, len(camera_samples), size=num_points)
        u = rng.uniform(margin, camera.width - margin, size=num_points)
        v = rng.uniform(margin, camera.height - margin, size=num_points)
        depth = rng.uniform(min_depth, max_depth, size=num_points)
        points = np.empty((num_points, 3), dtype=np.float64)
        for i, frame_idx in enumerate(frame_indices):
            point_camera = np.array(
                [(u[i] - camera.cx) * depth[i] / camera.fx, (v[i] - camera.cy) * depth[i] / camera.fy, depth[i]],
                dtype=np.float64,
            )
            points[i] = _camera_point_to_world(point_camera, camera_samples[int(frame_idx)].pose_camera_to_world)
    else:
        raise ValueError(f"Unsupported landmark_surface: {landmark_surface}")

    return LandmarkSet(
        ids=np.arange(num_points, dtype=np.int32),
        points_world=points,
        texture_seeds=rng.integers(0, 2**31 - 1, size=num_points, dtype=np.int64),
        radii=rng.integers(max(1, patch_radius - 1), patch_radius + 2, size=num_points, dtype=np.int16),
        intensities=rng.integers(90, 230, size=num_points, dtype=np.uint8),
    )


def project_landmarks(
    landmarks: LandmarkSet,
    pose_camera_to_world: np.ndarray,
    camera: CameraModel,
    camera_offset_left_frame: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offset = np.zeros(3, dtype=np.float64) if camera_offset_left_frame is None else camera_offset_left_frame
    r_wc = pose_camera_to_world[:3, :3]
    t_wc = pose_camera_to_world[:3, 3] + r_wc @ offset
    points_camera = (r_wc.T @ (landmarks.points_world - t_wc).T).T
    z = points_camera[:, 2]
    u = camera.fx * points_camera[:, 0] / z + camera.cx
    v = camera.fy * points_camera[:, 1] / z + camera.cy
    return u, v, z


def _junction_arm_alpha(xx: np.ndarray, yy: np.ndarray, direction: tuple[float, float], radius: int) -> np.ndarray:
    dx, dy = direction
    along = xx.astype(np.float32) * float(dx) + yy.astype(np.float32) * float(dy)
    cross = -xx.astype(np.float32) * float(dy) + yy.astype(np.float32) * float(dx)
    width = max(1.2, float(radius) * 0.24)
    arm = np.exp(-0.5 * (cross / width) ** 2)
    arm *= (along >= 0.0) & (along <= float(radius))

    # Fade away from the center and taper the far end. This keeps the junction
    # as the strongest response while smoothing arm edges into the background.
    radial_sigma = max(2.0, float(radius) * 0.58)
    arm *= np.exp(-0.5 * (along / radial_sigma) ** 2)
    fade_start = float(radius) * 0.58
    denom = max(1e-6, float(radius) - fade_start)
    end_fade = np.clip((float(radius) - along) / denom, 0.0, 1.0)
    arm *= np.where(along > fade_start, end_fade * end_fade * (3.0 - 2.0 * end_fade), 1.0)
    return arm.astype(np.float32)


def _junction_patch(texture_seed: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
    size = int(radius) * 2 + 1
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    variant = int(texture_seed) % 3
    rng = np.random.default_rng(int(texture_seed))

    if variant == 0:
        directions = [(1.0, 0.0), (0.0, 1.0)]
    elif variant == 1:
        directions = [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    else:
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        directions = [
            (inv_sqrt2, inv_sqrt2),
            (-inv_sqrt2, -inv_sqrt2),
            (inv_sqrt2, -inv_sqrt2),
            (-inv_sqrt2, inv_sqrt2),
        ]

    angle = rng.uniform(0.0, 2.0 * math.pi)
    c, s = math.cos(angle), math.sin(angle)
    rotated = [(c * dx - s * dy, s * dx + c * dy) for dx, dy in directions]
    alpha = np.zeros((size, size), dtype=np.float32)
    for direction in rotated:
        alpha = np.maximum(alpha, _junction_arm_alpha(xx, yy, direction, radius))

    # Pin the center to a strong junction. Ends are tapered by _junction_arm_alpha.
    center = radius
    alpha[max(0, center - 1) : center + 2, max(0, center - 1) : center + 2] = np.maximum(
        alpha[max(0, center - 1) : center + 2, max(0, center - 1) : center + 2],
        0.96,
    )
    alpha[alpha < 0.005] = 0.0
    gray = int(rng.integers(120, 256))
    return np.full((size, size), gray, dtype=np.uint8), np.clip(alpha, 0.0, 1.0)


def _landmark_patch(texture_seed: int, radius: int, intensity: int, patch_type: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(texture_seed))
    size = int(radius) * 2 + 1
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    if patch_type == "texture":
        mask = (xx * xx + yy * yy) <= (radius + 0.35) ** 2
        texture = rng.normal(loc=float(intensity), scale=38.0, size=(size, size))
        # Add a stable asymmetric stripe so descriptors can distinguish nearby patches.
        texture += 35.0 * np.sin(0.9 * xx + 0.37 * yy + (int(texture_seed) % 31))
        return np.clip(texture, 0, 255).astype(np.uint8), mask.astype(np.float32)

    if patch_type not in PATCH_TYPES:
        raise ValueError(f"Unsupported patch_type: {patch_type}")

    if patch_type in {"corner", "junction", "l_junction", "t_junction", "x_junction"}:
        if patch_type == "l_junction":
            seed = int(texture_seed) - int(texture_seed) % 3
        elif patch_type == "t_junction":
            seed = int(texture_seed) - int(texture_seed) % 3 + 1
        elif patch_type == "x_junction":
            seed = int(texture_seed) - int(texture_seed) % 3 + 2
        else:
            seed = int(texture_seed)
        return _junction_patch(seed, radius)

    low = 8.0 + float(texture_seed % 11)
    high = 236.0 - float((texture_seed // 11) % 17)
    if patch_type == "quadratic":
        theta = (float(texture_seed % 360) / 180.0) * math.pi
        c, s = math.cos(theta), math.sin(theta)
        xf = xx.astype(np.float32)
        yf = yy.astype(np.float32)
        xr = c * xf + s * yf
        yr = -s * xf + c * yf
        kx = 1.0 + 0.18 * float((texture_seed % 5) - 2)
        ky = 1.0 + 0.18 * float(((texture_seed // 5) % 5) - 2)
        q = (kx * xr * xr + ky * yr * yr) / float(max(1, radius * radius))
        q = np.clip(q, 0.0, 1.0)
        texture = low + (high - low) * q

        radial = (xf * xf + yf * yf) / float(max(1, radius * radius))
        outer = np.clip((radial - 0.35) / 0.65, 0.0, 1.0) ** 2
        code_freq = 3 + int(texture_seed % 5)
        code_phase = float((texture_seed // 13) % 31)
        code = np.sin(0.47 * code_freq * xf + 0.63 * (code_freq + 1) * yf + code_phase)
        texture += outer * (10.0 * code + rng.normal(0.0, 4.0, size=(size, size)))
    else:
        cell = max(2, radius // 2)
        pattern = ((np.floor_divide(xx + radius, cell) + np.floor_divide(yy + radius, cell)) % 2) == 0
        pattern ^= (xx >= 0) ^ (yy >= 0)
        if int(texture_seed) % 2:
            pattern = ~pattern
        texture = np.where(pattern, high, low).astype(np.float32)

        # Keep the anchor as the dominant feature while giving each patch a distinct descriptor.
        texture += rng.normal(0.0, 5.0, size=(size, size))
        texture += 8.0 * np.sin(0.7 * xx + 1.1 * yy + float(texture_seed % 37))
    sigma = max(2.0, radius * 0.62)
    alpha = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    alpha /= float(np.max(alpha))
    if patch_type == "quadratic":
        alpha[(xx * xx + yy * yy) <= 4] = 1.0
    alpha[alpha < 0.03] = 0.0
    return np.clip(texture, 0, 255).astype(np.uint8), alpha


def _value_noise_2d(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    xf = x - x0
    yf = y - y0
    sx = xf * xf * (3.0 - 2.0 * xf)
    sy = yf * yf * (3.0 - 2.0 * yf)

    def hashed(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        value = np.sin(ix.astype(np.float64) * 127.1 + iy.astype(np.float64) * 311.7 + float(seed) * 74.7) * 43758.5453
        return value - np.floor(value)

    n00 = hashed(x0, y0)
    n10 = hashed(x0 + 1, y0)
    n01 = hashed(x0, y0 + 1)
    n11 = hashed(x0 + 1, y0 + 1)
    nx0 = n00 * (1.0 - sx) + n10 * sx
    nx1 = n01 * (1.0 - sx) + n11 * sx
    return nx0 * (1.0 - sy) + nx1 * sy


def _gradient_background(
    camera: CameraModel,
    timestamp_ns: int,
    camera_id: int,
    seed: int,
    noise_sigma: float,
    brightness_jitter: float,
) -> np.ndarray:
    h, w = camera.height, camera.width
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    phase = (timestamp_ns % 1_000_000_000) * 1e-9
    base = 32.0 + 14.0 * x + 10.0 * y + brightness_jitter * math.sin(phase + camera_id)
    image = np.repeat(base, h, axis=0) if base.shape[0] == 1 else np.broadcast_to(base, (h, w)).copy()
    rng = np.random.default_rng(seed + camera_id * 1_000_003 + int(timestamp_ns % 1_000_003))
    if noise_sigma > 0.0:
        image += rng.normal(0.0, noise_sigma, size=(h, w))
    return np.clip(image, 0, 255).astype(np.uint8)


def _textured_cylinder_background(
    camera: CameraModel,
    pose_camera_to_world: np.ndarray,
    timestamp_ns: int,
    camera_id: int,
    seed: int,
    noise_sigma: float,
    brightness_jitter: float,
    texture_strength: float,
    cylinder_radius_m: float,
    texture_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = camera.height, camera.width
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x_cam = (uu - camera.cx) / camera.fx
    y_cam = (vv - camera.cy) / camera.fy

    offset = np.array([camera.baseline, 0.0, 0.0], dtype=np.float64) if camera_id == 1 else np.zeros(3, dtype=np.float64)
    r_wc = pose_camera_to_world[:3, :3]
    origin = pose_camera_to_world[:3, 3] + r_wc @ offset

    dx = r_wc[0, 0] * x_cam + r_wc[0, 1] * y_cam + r_wc[0, 2]
    dy = r_wc[1, 0] * x_cam + r_wc[1, 1] * y_cam + r_wc[1, 2]
    dz = r_wc[2, 0] * x_cam + r_wc[2, 1] * y_cam + r_wc[2, 2]

    radius = max(float(cylinder_radius_m), 1.0)
    a = dx * dx + dy * dy
    b = 2.0 * (origin[0] * dx + origin[1] * dy)
    c = origin[0] * origin[0] + origin[1] * origin[1] - radius * radius
    disc = np.maximum(b * b - 4.0 * a * c, 0.0)
    sqrt_disc = np.sqrt(disc)
    denom = np.maximum(2.0 * a, 1e-9)
    t0 = (-b - sqrt_disc) / denom
    t1 = (-b + sqrt_disc) / denom
    ray_t = np.where(t0 > 0.0, t0, t1)
    ray_t = np.where(ray_t > 0.0, ray_t, radius)
    depth = ray_t.astype(np.float32)

    px = origin[0] + ray_t * dx
    py = origin[1] + ray_t * dy
    pz = origin[2] + ray_t * dz

    scale = max(float(texture_scale), 0.5)
    u_tex = np.arctan2(py, px) * scale
    v_tex = pz * scale
    n1 = _value_noise_2d(u_tex, v_tex, seed + 31)
    n2 = _value_noise_2d(u_tex * 3.7 + 11.3, v_tex * 3.7 - 4.1, seed + 173)
    n3 = _value_noise_2d(px * scale * 0.75 + pz * 0.31, py * scale * 0.75 - pz * 0.27, seed + 911)
    n4 = _value_noise_2d(px * scale * 2.4 - pz * 1.1, py * scale * 2.4 + pz * 0.9, seed + 1543)
    speckle_cell_x = np.floor(px * scale * 5.2 + pz * 1.3)
    speckle_cell_y = np.floor(py * scale * 5.2 - pz * 0.8)
    speckle_hash = np.sin(speckle_cell_x * 269.5 + speckle_cell_y * 183.3 + float(seed) * 41.9) * 24634.6345
    speckle_hash = speckle_hash - np.floor(speckle_hash)
    sign_hash = np.sin(speckle_cell_x * 113.7 + speckle_cell_y * 271.9 + float(seed) * 19.3) * 19181.9123
    sign_hash = sign_hash - np.floor(sign_hash)
    speckle_sign = np.where(sign_hash > 0.5, 1.0, -1.0)
    speckles = np.maximum((speckle_hash - 0.78) / 0.22, 0.0) * speckle_sign
    waves = 0.5 * np.sin(2.7 * u_tex + 0.9 * v_tex) + 0.5 * np.sin(-1.8 * u_tex + 3.1 * v_tex + 0.4)
    phase = (timestamp_ns % 1_000_000_000) * 1e-9
    image = 112.0 + float(texture_strength) * (
        0.25 * (n1 - 0.5)
        + 0.25 * (n2 - 0.5)
        + 0.28 * (n3 - 0.5)
        + 0.13 * (n4 - 0.5)
        + 0.12 * speckles
        + 0.06 * waves
    )
    image += 10.0 * x_cam + 6.0 * y_cam + brightness_jitter * math.sin(phase)

    rng = np.random.default_rng(seed + camera_id * 1_000_003 + int(timestamp_ns % 1_000_003))
    if noise_sigma > 0.0:
        image += rng.normal(0.0, noise_sigma, size=(h, w))
    return np.clip(image, 0, 255).astype(np.uint8), depth


def _textured_walls_background(
    camera: CameraModel,
    pose_camera_to_world: np.ndarray,
    timestamp_ns: int,
    camera_id: int,
    seed: int,
    noise_sigma: float,
    brightness_jitter: float,
    texture_strength: float,
    texture_scale: float,
    wall_half_extent_m: float,
    wall_z_min_m: float,
    wall_z_max_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = camera.height, camera.width
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x_cam = (uu - camera.cx) / camera.fx
    y_cam = (vv - camera.cy) / camera.fy

    offset = np.array([camera.baseline, 0.0, 0.0], dtype=np.float64) if camera_id == 1 else np.zeros(3, dtype=np.float64)
    r_wc = pose_camera_to_world[:3, :3]
    origin = pose_camera_to_world[:3, 3] + r_wc @ offset
    dx = r_wc[0, 0] * x_cam + r_wc[0, 1] * y_cam + r_wc[0, 2]
    dy = r_wc[1, 0] * x_cam + r_wc[1, 1] * y_cam + r_wc[1, 2]
    dz = r_wc[2, 0] * x_cam + r_wc[2, 1] * y_cam + r_wc[2, 2]
    ray_t, wall_id = _intersect_room_walls(
        origin,
        dx,
        dy,
        dz,
        wall_half_extent_m,
        wall_z_min_m,
        wall_z_max_m,
        0.1,
        1e6,
    )

    valid = wall_id >= 0
    px = origin[0] + ray_t * dx
    py = origin[1] + ray_t * dy
    pz = origin[2] + ray_t * dz
    px = np.where(valid, px, 0.0)
    py = np.where(valid, py, 0.0)
    pz = np.where(valid, pz, 0.0)

    scale = max(float(texture_scale), 0.5)
    u_tex = np.where(wall_id < 2, py, px) * scale
    v_tex = pz * scale
    wall_phase = np.maximum(wall_id, 0).astype(np.float64) * 17.0
    n1 = _value_noise_2d(u_tex + wall_phase, v_tex, seed + 47)
    n2 = _value_noise_2d(u_tex * 3.1 - 5.0 + wall_phase, v_tex * 3.1 + 2.0, seed + 233)
    n3 = _value_noise_2d(u_tex * 8.0 + 1.7 * wall_phase, v_tex * 8.0 - 3.0, seed + 1549)
    mortar = np.minimum(np.abs((u_tex % 1.0) - 0.5), np.abs((v_tex % 0.6) - 0.3))
    mortar = np.clip((0.06 - mortar) / 0.06, 0.0, 1.0)
    speckle_hash = np.sin(np.floor(u_tex * 5.5) * 269.5 + np.floor(v_tex * 5.5) * 183.3 + wall_phase) * 24634.6345
    speckle_hash = speckle_hash - np.floor(speckle_hash)
    speckles = np.maximum((speckle_hash - 0.80) / 0.20, 0.0)
    phase = (timestamp_ns % 1_000_000_000) * 1e-9
    image = 112.0 + float(texture_strength) * (
        0.30 * (n1 - 0.5)
        + 0.28 * (n2 - 0.5)
        + 0.18 * (n3 - 0.5)
        - 0.16 * mortar
        + 0.08 * speckles
    )
    image += 8.0 * x_cam + 5.0 * y_cam + brightness_jitter * math.sin(phase)

    rng = np.random.default_rng(seed + camera_id * 1_000_003 + int(timestamp_ns % 1_000_003))
    if noise_sigma > 0.0:
        image += rng.normal(0.0, noise_sigma, size=(h, w))
    image = np.where(valid, image, 18.0)
    depth = np.where(valid, ray_t, 0.0).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8), depth


def _background(
    camera: CameraModel,
    pose_camera_to_world: np.ndarray,
    timestamp_ns: int,
    camera_id: int,
    seed: int,
    noise_sigma: float,
    brightness_jitter: float,
    background_type: str,
    texture_strength: float,
    cylinder_radius_m: float,
    texture_scale: float,
    wall_half_extent_m: float,
    wall_z_min_m: float,
    wall_z_max_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if background_type == "gradient":
        return _gradient_background(camera, timestamp_ns, camera_id, seed, noise_sigma, brightness_jitter), np.zeros(
            (camera.height, camera.width), dtype=np.float32
        )
    if background_type == "textured_cylinder":
        return _textured_cylinder_background(
            camera,
            pose_camera_to_world,
            timestamp_ns,
            camera_id,
            seed,
            noise_sigma,
            brightness_jitter,
            texture_strength,
            cylinder_radius_m,
            texture_scale,
        )
    if background_type == "textured_walls":
        return _textured_walls_background(
            camera,
            pose_camera_to_world,
            timestamp_ns,
            camera_id,
            seed,
            noise_sigma,
            brightness_jitter,
            texture_strength,
            texture_scale,
            wall_half_extent_m,
            wall_z_min_m,
            wall_z_max_m,
        )
    raise ValueError(f"Unsupported background_type: {background_type}")


def render_camera_image(
    landmarks: LandmarkSet,
    pose_camera_to_world: np.ndarray,
    camera: CameraModel,
    timestamp_ns: int,
    camera_id: int,
    seed: int,
    noise_sigma: float,
    brightness_jitter: float,
    patch_type: str,
    background_type: str,
    background_texture_strength: float,
    background_cylinder_radius_m: float,
    background_texture_scale: float,
    wall_half_extent_m: float,
    wall_z_min_m: float,
    wall_z_max_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    offset = np.array([camera.baseline, 0.0, 0.0], dtype=np.float64) if camera_id == 1 else np.zeros(3, dtype=np.float64)
    u, v, depth = project_landmarks(landmarks, pose_camera_to_world, camera, offset)
    radius_max = int(np.max(landmarks.radii)) if len(landmarks.radii) else 0
    in_fov = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(depth)
        & (depth > 0.1)
        & (u >= radius_max)
        & (u < camera.width - radius_max)
        & (v >= radius_max)
        & (v < camera.height - radius_max)
    )

    image, scene_depth = _background(
        camera,
        pose_camera_to_world,
        timestamp_ns,
        camera_id,
        seed,
        noise_sigma,
        brightness_jitter,
        background_type,
        background_texture_strength,
        background_cylinder_radius_m,
        background_texture_scale,
        wall_half_extent_m,
        wall_z_min_m,
        wall_z_max_m,
    )
    scene_depth = np.asarray(scene_depth, dtype=np.float32).copy()
    z_buffer = np.full((camera.height, camera.width), np.inf, dtype=np.float32)
    visible = np.zeros(len(landmarks.ids), dtype=bool)
    order = np.argsort(depth)
    feature_u = np.rint(u).astype(np.float32)
    feature_v = np.rint(v).astype(np.float32)
    for idx in order:
        if not in_fov[idx]:
            continue
        radius = int(landmarks.radii[idx])
        patch, alpha = _landmark_patch(int(landmarks.texture_seeds[idx]), radius, int(landmarks.intensities[idx]), patch_type)
        cu = int(round(float(u[idx])))
        cv = int(round(float(v[idx])))
        x0, x1 = cu - radius, cu + radius + 1
        y0, y1 = cv - radius, cv + radius + 1
        if x0 < 0 or y0 < 0 or x1 > camera.width or y1 > camera.height:
            continue
        roi_z = z_buffer[y0:y1, x0:x1]
        update = (alpha > 0.0) & (float(depth[idx]) < roi_z)
        if not np.any(update):
            continue
        roi = image[y0:y1, x0:x1]
        blended = (1.0 - alpha[update]) * roi[update].astype(np.float32) + alpha[update] * patch[update].astype(np.float32)
        roi[update] = np.clip(blended, 0, 255).astype(np.uint8)
        roi_z[update] = float(depth[idx])
        scene_depth[y0:y1, x0:x1][update] = float(depth[idx])
        visible[idx] = True

    fov_indices = np.flatnonzero(in_fov)
    obs = {
        "landmark_id": landmarks.ids[fov_indices].astype(np.int32),
        "u": u[fov_indices].astype(np.float32),
        "v": v[fov_indices].astype(np.float32),
        "feature_u": feature_u[fov_indices].astype(np.float32),
        "feature_v": feature_v[fov_indices].astype(np.float32),
        "depth_m": depth[fov_indices].astype(np.float32),
        "visible": visible[fov_indices],
        "occluded": (~visible[fov_indices]).astype(bool),
        "patch_radius_px": landmarks.radii[fov_indices].astype(np.int16),
    }
    return image, scene_depth, obs


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def _write_depth_sidecars(depth_dir: Path, timestamp_ns: int, depth_m: np.ndarray) -> None:
    depth_dir.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(depth_m, dtype=np.float32)
    np.savez_compressed(depth_dir / f"{timestamp_ns}.npz", depth_m=depth)
    depth_mm = np.clip(np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0, 0.0, 65535.0).astype(np.uint16)
    if not cv2.imwrite(str(depth_dir / f"{timestamp_ns}.png"), depth_mm):
        raise RuntimeError(f"Failed to write GT depth image: {depth_dir / f'{timestamp_ns}.png'}")


def _stamp_from_ns(timestamp_ns: int) -> tuple[int, int]:
    sec = int(timestamp_ns // 1_000_000_000)
    nsec = int(timestamp_ns % 1_000_000_000)
    return sec, nsec


def _write_landmarks_csv(path: Path, landmarks: LandmarkSet) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("landmark_id,x,y,z,texture_seed,patch_radius_px,intensity\n")
        for idx, point, texture_seed, radius, intensity in zip(
            landmarks.ids,
            landmarks.points_world,
            landmarks.texture_seeds,
            landmarks.radii,
            landmarks.intensities,
        ):
            f.write(
                f"{int(idx)},{point[0]:.9f},{point[1]:.9f},{point[2]:.9f},{int(texture_seed)},{int(radius)},{int(intensity)}\n"
            )


def _write_trajectory_files(output_dir: Path, samples: list[TrajectorySample]) -> None:
    with (output_dir / "trajectory_tum.txt").open("w", encoding="utf-8") as f_tum, (
        output_dir / "trajectory_15dof.csv"
    ).open("w", encoding="utf-8") as f_csv:
        f_csv.write(
            "timestamp,x,y,z,qx,qy,qz,qw,vel_x,vel_y,vel_z,ang_vel_x,ang_vel_y,ang_vel_z,acc_x,acc_y,acc_z\n"
        )
        for sample in samples:
            pose = sample.pose_camera_to_world
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
            xyz = pose[:3, 3]
            timestamp_sec = sample.timestamp_ns * 1e-9
            f_tum.write(
                f"{timestamp_sec:.9f} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} "
                f"{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}\n"
            )
            f_csv.write(
                f"{sample.timestamp_ns},{xyz[0]:.9f},{xyz[1]:.9f},{xyz[2]:.9f},"
                f"{quat[0]:.9f},{quat[1]:.9f},{quat[2]:.9f},{quat[3]:.9f},"
                f"{sample.velocity_world[0]:.9f},{sample.velocity_world[1]:.9f},{sample.velocity_world[2]:.9f},"
                f"{sample.angular_velocity_camera[0]:.9f},{sample.angular_velocity_camera[1]:.9f},{sample.angular_velocity_camera[2]:.9f},"
                f"{sample.specific_force_camera[0]:.9f},{sample.specific_force_camera[1]:.9f},{sample.specific_force_camera[2]:.9f}\n"
            )


def _append_observations(
    rows: dict[str, list[Any]],
    timestamp_ns: int,
    camera_id: int,
    obs: dict[str, np.ndarray],
) -> None:
    count = len(obs["landmark_id"])
    rows["timestamp_ns"].extend([timestamp_ns] * count)
    rows["camera_id"].extend([camera_id] * count)
    for name in ["landmark_id", "u", "v", "feature_u", "feature_v", "depth_m", "visible", "occluded", "patch_radius_px"]:
        rows[name].extend(obs[name].tolist())


def _rows_to_arrays(rows: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    return {
        "timestamp_ns": np.asarray(rows["timestamp_ns"], dtype=np.int64),
        "camera_id": np.asarray(rows["camera_id"], dtype=np.int8),
        "landmark_id": np.asarray(rows["landmark_id"], dtype=np.int32),
        "u": np.asarray(rows["u"], dtype=np.float32),
        "v": np.asarray(rows["v"], dtype=np.float32),
        "feature_u": np.asarray(rows["feature_u"], dtype=np.float32),
        "feature_v": np.asarray(rows["feature_v"], dtype=np.float32),
        "depth_m": np.asarray(rows["depth_m"], dtype=np.float32),
        "visible": np.asarray(rows["visible"], dtype=bool),
        "occluded": np.asarray(rows["occluded"], dtype=bool),
        "patch_radius_px": np.asarray(rows["patch_radius_px"], dtype=np.int16),
    }


def _build_visible_lookup(obs_arrays: dict[str, np.ndarray]) -> dict[tuple[int, int], dict[int, tuple[float, float]]]:
    lookup: dict[tuple[int, int], dict[int, tuple[float, float]]] = {}
    visible_indices = np.flatnonzero(obs_arrays["visible"])
    u_key = "feature_u" if "feature_u" in obs_arrays else "u"
    v_key = "feature_v" if "feature_v" in obs_arrays else "v"
    for idx in visible_indices:
        key = (int(obs_arrays["timestamp_ns"][idx]), int(obs_arrays["camera_id"][idx]))
        lookup.setdefault(key, {})[int(obs_arrays["landmark_id"][idx])] = (
            float(obs_arrays[u_key][idx]),
            float(obs_arrays[v_key][idx]),
        )
    return lookup


def _append_match_rows(
    rows: dict[str, list[Any]],
    ts_a: int,
    cam_a: int,
    ts_b: int,
    cam_b: int,
    match_type: int,
    obs_a: dict[int, tuple[float, float]],
    obs_b: dict[int, tuple[float, float]],
) -> None:
    for landmark_id in sorted(set(obs_a).intersection(obs_b)):
        u_a, v_a = obs_a[landmark_id]
        u_b, v_b = obs_b[landmark_id]
        rows["timestamp_ns_a"].append(ts_a)
        rows["camera_id_a"].append(cam_a)
        rows["landmark_id"].append(landmark_id)
        rows["u_a"].append(u_a)
        rows["v_a"].append(v_a)
        rows["timestamp_ns_b"].append(ts_b)
        rows["camera_id_b"].append(cam_b)
        rows["u_b"].append(u_b)
        rows["v_b"].append(v_b)
        rows["match_type"].append(match_type)


def build_gt_matches(obs_arrays: dict[str, np.ndarray], frame_timestamps_ns: np.ndarray) -> dict[str, np.ndarray]:
    lookup = _build_visible_lookup(obs_arrays)
    rows: dict[str, list[Any]] = {
        "timestamp_ns_a": [],
        "camera_id_a": [],
        "landmark_id": [],
        "u_a": [],
        "v_a": [],
        "timestamp_ns_b": [],
        "camera_id_b": [],
        "u_b": [],
        "v_b": [],
        "match_type": [],
    }
    timestamps = [int(t) for t in frame_timestamps_ns]
    for ts in timestamps:
        _append_match_rows(rows, ts, 0, ts, 1, MATCH_STEREO, lookup.get((ts, 0), {}), lookup.get((ts, 1), {}))
    for ts_a, ts_b in zip(timestamps[:-1], timestamps[1:]):
        _append_match_rows(
            rows,
            ts_a,
            0,
            ts_b,
            0,
            MATCH_TEMPORAL_LEFT,
            lookup.get((ts_a, 0), {}),
            lookup.get((ts_b, 0), {}),
        )
        _append_match_rows(
            rows,
            ts_a,
            1,
            ts_b,
            1,
            MATCH_TEMPORAL_RIGHT,
            lookup.get((ts_a, 1), {}),
            lookup.get((ts_b, 1), {}),
        )
    return {
        "timestamp_ns_a": np.asarray(rows["timestamp_ns_a"], dtype=np.int64),
        "camera_id_a": np.asarray(rows["camera_id_a"], dtype=np.int8),
        "landmark_id": np.asarray(rows["landmark_id"], dtype=np.int32),
        "u_a": np.asarray(rows["u_a"], dtype=np.float32),
        "v_a": np.asarray(rows["v_a"], dtype=np.float32),
        "timestamp_ns_b": np.asarray(rows["timestamp_ns_b"], dtype=np.int64),
        "camera_id_b": np.asarray(rows["camera_id_b"], dtype=np.int8),
        "u_b": np.asarray(rows["u_b"], dtype=np.float32),
        "v_b": np.asarray(rows["v_b"], dtype=np.float32),
        "match_type": np.asarray(rows["match_type"], dtype=np.int8),
    }


def _camera_sample_indices(imu_samples: list[TrajectorySample], camera_hz: float) -> list[int]:
    if not imu_samples:
        return []
    start_ns = imu_samples[0].timestamp_ns
    duration_ns = imu_samples[-1].timestamp_ns - start_ns
    camera_period_ns = int(round(1e9 / camera_hz))
    camera_timestamps = np.arange(start_ns, start_ns + duration_ns + 1, camera_period_ns, dtype=np.int64)
    imu_timestamps = np.asarray([s.timestamp_ns for s in imu_samples], dtype=np.int64)
    return [int(np.argmin(np.abs(imu_timestamps - ts))) for ts in camera_timestamps]


def _try_import_ros() -> dict[str, Any]:
    try:
        from builtin_interfaces.msg import Time
        from geometry_msgs.msg import Quaternion, Vector3
        from nav_msgs.msg import Odometry
        from rclpy.serialization import serialize_message
        from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata
        from sensor_msgs.msg import CameraInfo, Image, Imu
        from std_msgs.msg import Header
    except ImportError as exc:
        raise RuntimeError("ROS 2 Python packages are required to write the bag. Use --skip_bag for sidecars only.") from exc
    return locals()


def _make_header(ros: dict[str, Any], timestamp_ns: int, frame_id: str):
    Header = ros["Header"]
    Time = ros["Time"]
    sec, nsec = _stamp_from_ns(timestamp_ns)
    header = Header()
    header.stamp = Time(sec=sec, nanosec=nsec)
    header.frame_id = frame_id
    return header


def _make_image_msg(ros: dict[str, Any], image: np.ndarray, timestamp_ns: int, frame_id: str):
    Image = ros["Image"]
    msg = Image()
    msg.header = _make_header(ros, timestamp_ns, frame_id)
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = int(image.shape[1])
    msg.data = image.tobytes()
    return msg


def _make_camera_info_msg(ros: dict[str, Any], camera: CameraModel, timestamp_ns: int, frame_id: str, is_right: bool):
    CameraInfo = ros["CameraInfo"]
    msg = CameraInfo()
    msg.header = _make_header(ros, timestamp_ns, frame_id)
    msg.height = int(camera.height)
    msg.width = int(camera.width)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [camera.fx, 0.0, camera.cx, 0.0, camera.fy, camera.cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    tx = -camera.fx * camera.baseline if is_right else 0.0
    msg.p = [camera.fx, 0.0, camera.cx, tx, 0.0, camera.fy, camera.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def _make_imu_msg(ros: dict[str, Any], sample: TrajectorySample, rng: np.random.Generator, cfg: GeneratorConfig):
    Imu = ros["Imu"]
    msg = Imu()
    msg.header = _make_header(ros, sample.timestamp_ns, "camera")
    accel = sample.specific_force_camera.copy()
    gyro = sample.angular_velocity_camera.copy()
    if cfg.accel_noise_sigma > 0.0:
        accel += rng.normal(0.0, cfg.accel_noise_sigma, size=3)
    if cfg.gyro_noise_sigma > 0.0:
        gyro += rng.normal(0.0, cfg.gyro_noise_sigma, size=3)
    msg.linear_acceleration.x = float(accel[0])
    msg.linear_acceleration.y = float(accel[1])
    msg.linear_acceleration.z = float(accel[2])
    msg.angular_velocity.x = float(gyro[0])
    msg.angular_velocity.y = float(gyro[1])
    msg.angular_velocity.z = float(gyro[2])
    msg.linear_acceleration_covariance = [cfg.accel_noise_sigma**2, 0.0, 0.0, 0.0, cfg.accel_noise_sigma**2, 0.0, 0.0, 0.0, cfg.accel_noise_sigma**2]
    msg.angular_velocity_covariance = [cfg.gyro_noise_sigma**2, 0.0, 0.0, 0.0, cfg.gyro_noise_sigma**2, 0.0, 0.0, 0.0, cfg.gyro_noise_sigma**2]
    return msg


def _make_odom_msg(ros: dict[str, Any], sample: TrajectorySample):
    Odometry = ros["Odometry"]
    msg = Odometry()
    msg.header = _make_header(ros, sample.timestamp_ns, "world")
    msg.child_frame_id = "camera"
    pose = sample.pose_camera_to_world
    quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
    xyz = pose[:3, 3]
    msg.pose.pose.position.x = float(xyz[0])
    msg.pose.pose.position.y = float(xyz[1])
    msg.pose.pose.position.z = float(xyz[2])
    msg.pose.pose.orientation.x = float(quat[0])
    msg.pose.pose.orientation.y = float(quat[1])
    msg.pose.pose.orientation.z = float(quat[2])
    msg.pose.pose.orientation.w = float(quat[3])
    msg.twist.twist.linear.x = float(sample.velocity_world[0])
    msg.twist.twist.linear.y = float(sample.velocity_world[1])
    msg.twist.twist.linear.z = float(sample.velocity_world[2])
    msg.twist.twist.angular.x = float(sample.angular_velocity_camera[0])
    msg.twist.twist.angular.y = float(sample.angular_velocity_camera[1])
    msg.twist.twist.angular.z = float(sample.angular_velocity_camera[2])
    return msg


def _create_topic(writer: Any, ros: dict[str, Any], name: str, msg_type: str) -> None:
    TopicMetadata = ros["TopicMetadata"]
    writer.create_topic(TopicMetadata(name=name, type=msg_type, serialization_format="cdr"))


def _open_writer(output_dir: Path, ros: dict[str, Any]):
    SequentialWriter = ros["SequentialWriter"]
    StorageOptions = ros["StorageOptions"]
    ConverterOptions = ros["ConverterOptions"]
    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=str(output_dir), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    _create_topic(writer, ros, LEFT_TOPIC, "sensor_msgs/msg/Image")
    _create_topic(writer, ros, RIGHT_TOPIC, "sensor_msgs/msg/Image")
    _create_topic(writer, ros, LEFT_INFO_TOPIC, "sensor_msgs/msg/CameraInfo")
    _create_topic(writer, ros, RIGHT_INFO_TOPIC, "sensor_msgs/msg/CameraInfo")
    _create_topic(writer, ros, IMU_TOPIC, "sensor_msgs/msg/Imu")
    _create_topic(writer, ros, GT_ODOM_TOPIC, "nav_msgs/msg/Odometry")
    return writer


def generate_dataset(cfg: GeneratorConfig) -> dict[str, Any]:
    output_dir = Path(cfg.output_bag).resolve()
    if output_dir.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    if cfg.skip_bag:
        output_dir.mkdir(parents=True, exist_ok=True)

    camera = cfg.camera
    if cfg.landmark_surface == "auto":
        if cfg.background_type == "textured_cylinder":
            landmark_surface = "cylinder"
        elif cfg.background_type == "textured_walls":
            landmark_surface = "walls"
        else:
            landmark_surface = "camera_rays"
    else:
        landmark_surface = cfg.landmark_surface
    imu_samples = generate_trajectory(
        cfg.duration_sec,
        cfg.imu_hz,
        cfg.start_timestamp_ns,
        cfg.gravity,
        cfg.trajectory_mode,
        cfg.static_start_sec,
    )
    camera_indices = _camera_sample_indices(imu_samples, cfg.camera_hz)
    camera_samples = [imu_samples[i] for i in camera_indices]
    landmarks = generate_landmarks(
        camera_samples,
        camera,
        cfg.num_points,
        cfg.min_depth,
        cfg.max_depth,
        cfg.patch_radius,
        cfg.seed,
        landmark_surface=landmark_surface,
        cylinder_radius_m=cfg.background_cylinder_radius_m,
        wall_half_extent_m=cfg.wall_half_extent_m,
        wall_z_min_m=cfg.wall_z_min_m,
        wall_z_max_m=cfg.wall_z_max_m,
    )

    ros = None
    writer = None
    serialize_message = None
    if not cfg.skip_bag:
        ros = _try_import_ros()
        writer = _open_writer(output_dir, ros)
        serialize_message = ros["serialize_message"]

    images_dir = output_dir / "images"
    obs_rows: dict[str, list[Any]] = {
        "timestamp_ns": [],
        "camera_id": [],
        "landmark_id": [],
        "u": [],
        "v": [],
        "feature_u": [],
        "feature_v": [],
        "depth_m": [],
        "visible": [],
        "occluded": [],
        "patch_radius_px": [],
    }
    frame_timestamps_ns: list[int] = []
    rng = np.random.default_rng(cfg.seed + 11)

    for sample in camera_samples:
        timestamp_ns = sample.timestamp_ns
        frame_timestamps_ns.append(timestamp_ns)
        left_image, left_depth_gt, left_obs = render_camera_image(
            landmarks,
            sample.pose_camera_to_world,
            camera,
            timestamp_ns,
            camera_id=0,
            seed=cfg.seed,
            noise_sigma=cfg.image_noise_sigma,
            brightness_jitter=cfg.brightness_jitter,
            patch_type=cfg.patch_type,
            background_type=cfg.background_type,
            background_texture_strength=cfg.background_texture_strength,
            background_cylinder_radius_m=cfg.background_cylinder_radius_m,
            background_texture_scale=cfg.background_texture_scale,
            wall_half_extent_m=cfg.wall_half_extent_m,
            wall_z_min_m=cfg.wall_z_min_m,
            wall_z_max_m=cfg.wall_z_max_m,
        )
        right_image, right_depth_gt, right_obs = render_camera_image(
            landmarks,
            sample.pose_camera_to_world,
            camera,
            timestamp_ns,
            camera_id=1,
            seed=cfg.seed,
            noise_sigma=cfg.image_noise_sigma,
            brightness_jitter=cfg.brightness_jitter,
            patch_type=cfg.patch_type,
            background_type=cfg.background_type,
            background_texture_strength=cfg.background_texture_strength,
            background_cylinder_radius_m=cfg.background_cylinder_radius_m,
            background_texture_scale=cfg.background_texture_scale,
            wall_half_extent_m=cfg.wall_half_extent_m,
            wall_z_min_m=cfg.wall_z_min_m,
            wall_z_max_m=cfg.wall_z_max_m,
        )
        _append_observations(obs_rows, timestamp_ns, 0, left_obs)
        _append_observations(obs_rows, timestamp_ns, 1, right_obs)
        _write_png(images_dir / "infra1" / f"{timestamp_ns}.png", left_image)
        _write_png(images_dir / "infra2" / f"{timestamp_ns}.png", right_image)
        _write_depth_sidecars(output_dir / "depth_gt" / "infra1", timestamp_ns, left_depth_gt)
        _write_depth_sidecars(output_dir / "depth_gt" / "infra2", timestamp_ns, right_depth_gt)

        if writer is not None and ros is not None and serialize_message is not None:
            writer.write(LEFT_TOPIC, serialize_message(_make_image_msg(ros, left_image, timestamp_ns, "camera")), timestamp_ns)
            writer.write(RIGHT_TOPIC, serialize_message(_make_image_msg(ros, right_image, timestamp_ns, "camera")), timestamp_ns)
            writer.write(LEFT_INFO_TOPIC, serialize_message(_make_camera_info_msg(ros, camera, timestamp_ns, "camera", False)), timestamp_ns)
            writer.write(RIGHT_INFO_TOPIC, serialize_message(_make_camera_info_msg(ros, camera, timestamp_ns, "camera", True)), timestamp_ns)
            writer.write(GT_ODOM_TOPIC, serialize_message(_make_odom_msg(ros, sample)), timestamp_ns)

    if writer is not None and ros is not None and serialize_message is not None:
        for sample in imu_samples:
            writer.write(IMU_TOPIC, serialize_message(_make_imu_msg(ros, sample, rng, cfg)), sample.timestamp_ns)

    obs_arrays = _rows_to_arrays(obs_rows)
    frame_timestamps = np.asarray(frame_timestamps_ns, dtype=np.int64)
    matches = build_gt_matches(obs_arrays, frame_timestamps)

    np.savez_compressed(output_dir / "observations.npz", **obs_arrays)
    np.savez_compressed(output_dir / "matches_gt.npz", **matches)
    np.save(output_dir / "frame_timestamps_ns.npy", frame_timestamps)
    _write_landmarks_csv(output_dir / "landmarks.csv", landmarks)
    _write_trajectory_files(output_dir, imu_samples)

    metadata = {
        "format": "tinynav_synthetic_stereo_vio_feature_dataset_v1",
        "pose_convention": POSE_CONVENTION,
        "static_start_sec": float(cfg.static_start_sec) if cfg.trajectory_mode == "planar_xy" else 0.0,
        "trajectory_origin_policy": "origin_static_then_planar_analytic"
        if cfg.trajectory_mode == "planar_xy"
        else "spatial_3d_unchanged",
        "planar_motion_time_origin_sec": float(cfg.static_start_sec) if cfg.trajectory_mode == "planar_xy" else None,
        "imu_accel_convention": "perception_linear_acceleration_camera = R_cw @ (gravity_world - acceleration_world)",
        "topics": {
            "left_image": LEFT_TOPIC,
            "right_image": RIGHT_TOPIC,
            "left_camera_info": LEFT_INFO_TOPIC,
            "right_camera_info": RIGHT_INFO_TOPIC,
            "imu": IMU_TOPIC,
            "ground_truth_odometry": GT_ODOM_TOPIC,
        },
        "sidecars": {
            "left_depth_gt": "depth_gt/infra1/<timestamp_ns>.npz and .png",
            "right_depth_gt": "depth_gt/infra2/<timestamp_ns>.npz and .png",
            "depth_gt_png_units": "uint16 millimeters, 0 means invalid/no surface",
            "depth_gt_npz_units": "float32 meters under key depth_m",
        },
        "camera": asdict(camera),
        "config": asdict(cfg),
        "resolved_landmark_surface": landmark_surface,
        "counts": {
            "imu_samples": len(imu_samples),
            "camera_frames": len(camera_samples),
            "landmarks": int(len(landmarks.ids)),
            "observations": int(len(obs_arrays["timestamp_ns"])),
            "visible_observations": int(np.count_nonzero(obs_arrays["visible"])),
            "gt_matches": int(len(matches["landmark_id"])),
        },
        "match_type_ids": {
            "stereo_same_time": MATCH_STEREO,
            "temporal_left": MATCH_TEMPORAL_LEFT,
            "temporal_right": MATCH_TEMPORAL_RIGHT,
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def main() -> None:
    cfg = parse_args()
    metadata = generate_dataset(cfg)
    print(json.dumps({"output_bag": str(Path(cfg.output_bag).resolve()), "counts": metadata["counts"]}, indent=2))


if __name__ == "__main__":
    main()
