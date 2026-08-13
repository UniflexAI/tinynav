#!/usr/bin/env python3
"""Offline perception-to-planning MVP for tinynav.

This intentionally stays small and inspectable:
  scene objects -> synthetic depth -> tinynav raycasting -> obstacle/ESDF -> trajectory scoring

Example:
  uv run python tool/simulator/offline_planning_mvp.py
  uv run python tool/simulator/offline_planning_mvp.py --write-example /tmp/scene.json
  uv run python tool/simulator/offline_planning_mvp.py --config /tmp/scene.json
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R

from tinynav.core.planning_node import (
    B2_CONFIG,
    GO2_CONFIG,
    ObstacleConfig,
    build_obstacle_map,
    generate_predefined_trajectory_vocabularies,
    generate_trajectory_library_3d,
    score_trajectories_by_ESDF,
    select_trajectory_with_recovery,
)


@dataclass
class SimObject:
    name: str
    kind: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(self.center, dtype=np.float64)
        half = np.asarray(self.size, dtype=np.float64) / 2.0
        return center - half, center + half


def default_config() -> dict:
    return {
        "name": "narrow_passage",
        "output_dir": "outputs/offline_planning_mvp",
        "robot": {
            "preset": "b2",
            "length": 1.0,
            "width": 0.5,
            "camera_x": 0.5,
            "control_x": 0.5,
            "safety_radius": 0.1,
            "comfort_radius": 0.1,
        },
        "grid": {
            "shape": [120, 100, 40],
            "resolution": 0.1,
            "origin": [-2.0, -5.0, -1.0],
        },
        "camera": {
            "width": 160,
            "image_height": 100,
            "fx": 120.0,
            "fy": 120.0,
            "max_range": 6.0,
            "ray_step": 3,
            "mount_height": 0.45,
        },
        "start": {"xy": [0.0, 0.0], "yaw_deg": 0.0},
        "target": [4.0, 0.0, 0.0],
        "planner": {
            "num_samples": 21,
            "duration": 3.0,
            "dt": 0.1,
            "vx_max": 0.5,
            "reverse_speed": 0.3,
            "reverse_omegas": [0.0, -0.5, 0.5],
            "front_reverse_threshold": 0.3,
            "trajectory_smooth_weight": 10.0,
            "last_param": [0.0, 0.0],
        },
        "obstacle": {
            "robot_z_bottom": -0.5,
            "robot_z_top": 0.4,
            "occ_threshold": 0.09,
            "min_wall_span_m": 0.35,
            "dilation_cells": 1,
        },
        "objects": [
            {"name": "left_wall", "kind": "box", "center": [2.0, 1.0, 0.35], "size": [3.2, 0.25, 1.3]},
            {"name": "right_wall", "kind": "box", "center": [2.0, -1.0, 0.35], "size": [3.2, 0.25, 1.3]},
            {"name": "center_box", "kind": "box", "center": [1.65, 0.0, 0.25], "size": [0.45, 0.55, 0.5]},
        ],
    }


def deep_update(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | None) -> dict:
    config = default_config()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            deep_update(config, json.load(f))
    return config


def robot_from_config(config: dict):
    preset = config["robot"].get("preset", "b2").lower()
    robot = B2_CONFIG if preset == "b2" else GO2_CONFIG
    robot = type(robot)(**asdict(robot))
    for key, value in config["robot"].items():
        if key != "preset" and hasattr(robot, key):
            setattr(robot, key, value)
    if not hasattr(robot, "comfort_radius"):
        robot.comfort_radius = config["robot"].get("comfort_radius", 0.8)
    return robot


def make_camera_pose(start_xy: list[float], yaw_deg: float, camera_height: float,
                     forward_offset: float = 0.0, left_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    yaw = np.deg2rad(yaw_deg)
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float64)
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0], dtype=np.float64)
    down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    rot = np.column_stack([right, down, forward])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    camera_xy = np.asarray([start_xy[0], start_xy[1], 0.0], dtype=np.float64)
    camera_xy = camera_xy + forward * float(forward_offset) + left * float(left_offset)
    T[:3, 3] = [camera_xy[0], camera_xy[1], camera_height]
    return T, R.from_matrix(rot).as_quat()


def intersect_box(origin: np.ndarray, direction: np.ndarray, obj: SimObject, max_range: float) -> float:
    box_min, box_max = obj.bounds
    inv_d = np.divide(1.0, direction, out=np.full(3, np.inf), where=np.abs(direction) > 1e-9)
    t0 = (box_min - origin) * inv_d
    t1 = (box_max - origin) * inv_d
    t_near = np.maximum.reduce(np.minimum(t0, t1))
    t_far = np.minimum.reduce(np.maximum(t0, t1))
    if t_far < 0.0 or t_near > t_far:
        return np.inf
    hit = t_near if t_near > 0.0 else t_far
    return hit if 0.0 < hit <= max_range else np.inf


@njit(cache=True)
def render_depth_boxes(bounds_min: np.ndarray, bounds_max: np.ndarray, T_cam_to_world: np.ndarray,
                       width: int, height: int, fx: float, fy: float, cx: float, cy: float,
                       max_range: float) -> tuple[np.ndarray, np.ndarray]:
    depth = np.zeros((height, width), dtype=np.float32)
    hit_mask = np.zeros((height, width), dtype=np.bool_)
    cam_origin = T_cam_to_world[:3, 3]
    rot = T_cam_to_world[:3, :3]

    for v in range(height):
        for u in range(width):
            ray_cam_x = (u - cx) / fx
            ray_cam_y = (v - cy) / fy
            ray_cam_z = 1.0
            norm = np.sqrt(ray_cam_x * ray_cam_x + ray_cam_y * ray_cam_y + ray_cam_z * ray_cam_z)
            ray_cam_x /= norm
            ray_cam_y /= norm
            ray_cam_z /= norm

            ray_world_x = rot[0, 0] * ray_cam_x + rot[0, 1] * ray_cam_y + rot[0, 2] * ray_cam_z
            ray_world_y = rot[1, 0] * ray_cam_x + rot[1, 1] * ray_cam_y + rot[1, 2] * ray_cam_z
            ray_world_z = rot[2, 0] * ray_cam_x + rot[2, 1] * ray_cam_y + rot[2, 2] * ray_cam_z

            best = np.inf
            for obj_idx in range(bounds_min.shape[0]):
                t_near = -np.inf
                t_far = np.inf
                valid = True
                for axis in range(3):
                    if axis == 0:
                        ray = ray_world_x
                        origin = cam_origin[0]
                    elif axis == 1:
                        ray = ray_world_y
                        origin = cam_origin[1]
                    else:
                        ray = ray_world_z
                        origin = cam_origin[2]

                    if abs(ray) <= 1e-9:
                        if origin < bounds_min[obj_idx, axis] or origin > bounds_max[obj_idx, axis]:
                            valid = False
                            break
                        continue

                    inv_ray = 1.0 / ray
                    t0 = (bounds_min[obj_idx, axis] - origin) * inv_ray
                    t1 = (bounds_max[obj_idx, axis] - origin) * inv_ray
                    if t0 > t1:
                        tmp = t0
                        t0 = t1
                        t1 = tmp
                    if t0 > t_near:
                        t_near = t0
                    if t1 < t_far:
                        t_far = t1

                if not valid or t_far < 0.0 or t_near > t_far:
                    continue
                hit = t_near if t_near > 0.0 else t_far
                if 0.0 < hit <= max_range and hit < best:
                    best = hit

            if np.isfinite(best):
                depth[v, u] = best * ray_cam_z
                hit_mask[v, u] = True
    return depth, hit_mask


def render_depth(objects: list[SimObject], T_cam_to_world: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    width = int(cam["width"])
    height = int(cam.get("image_height", cam.get("height", 100)))
    fx = float(cam["fx"])
    fy = float(cam["fy"])
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    max_range = float(cam["max_range"])
    bounds_min = np.zeros((len(objects), 3), dtype=np.float64)
    bounds_max = np.zeros((len(objects), 3), dtype=np.float64)
    for idx, obj in enumerate(objects):
        bounds_min[idx], bounds_max[idx] = obj.bounds
    return render_depth_boxes(bounds_min, bounds_max, T_cam_to_world, width, height, fx, fy, cx, cy, max_range)


@njit(cache=True)
def run_synthetic_raycasting(depth_image, hit_mask, T_cam_to_world, grid_shape, fx, fy, cx, cy,
                             origin, step, resolution, max_range):
    """Raycast synthetic depth.

    All rays carve free space. Only rays that hit an object add occupied evidence
    at the endpoint. This matches real planning behavior better than skipping
    no-hit pixels, because moved-away objects become observed free space.
    """
    occupancy_grid = np.zeros(grid_shape, dtype=np.float64)
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
            hit = bool(hit_mask[v, u])
            d = float(depth_image[v, u]) if hit else float(max_range)
            if (not np.isfinite(d)) or d <= 0:
                continue

            px = (u - cx) * d / fx
            py = (v - cy) * d / fy
            pz = d

            pw_x = T_cam_to_world[0, 0] * px + T_cam_to_world[0, 1] * py + T_cam_to_world[0, 2] * pz + T_cam_to_world[0, 3]
            pw_y = T_cam_to_world[1, 0] * px + T_cam_to_world[1, 1] * py + T_cam_to_world[1, 2] * pz + T_cam_to_world[1, 3]
            pw_z = T_cam_to_world[2, 0] * px + T_cam_to_world[2, 1] * py + T_cam_to_world[2, 2] * pz + T_cam_to_world[2, 3]

            end_voxel_x = int(np.floor((pw_x - origin_x) / resolution))
            end_voxel_y = int(np.floor((pw_y - origin_y) / resolution))
            end_voxel_z = int(np.floor((pw_z - origin_z) / resolution))

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

            if hit and (0 <= end_voxel_x < grid_shape_x and
                        0 <= end_voxel_y < grid_shape_y and
                        0 <= end_voxel_z < grid_shape_z):
                occupancy_grid[end_voxel_x, end_voxel_y, end_voxel_z] += 0.2

    return np.clip(occupancy_grid, -0.1, 0.1)


def front_obstacle_dist(center: np.ndarray, yaw_deg: float, obstacle_mask: np.ndarray, origin: np.ndarray,
                        resolution: float, front_len: float, half_w: float, max_dist: float = 2.0) -> float:
    yaw = np.deg2rad(yaw_deg)
    fwd = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    left = np.array([-fwd[1], fwd[0]], dtype=np.float64)
    steps = int(max_dist / resolution) + 1
    rows, cols = obstacle_mask.shape
    for step in range(steps):
        d_from_face = step * resolution
        d_from_center = front_len + d_from_face
        for w in (-half_w, 0.0, half_w):
            p = center[:2] + fwd * d_from_center + left * w
            ix = int((p[0] - origin[0]) / resolution)
            iy = int((p[1] - origin[1]) / resolution)
            if 0 <= ix < rows and 0 <= iy < cols and obstacle_mask[ix, iy]:
                return d_from_face
    return max_dist + resolution


def generate_forward_trajectories(planner: dict, init_p: np.ndarray, init_q: np.ndarray):
    kwargs = {
        "num_samples": int(planner["num_samples"]),
        "duration": float(planner["duration"]),
        "dt": float(planner["dt"]),
        "init_p": init_p,
        "init_q": init_q,
    }
    try:
        return generate_trajectory_library_3d(**kwargs, vx_max=float(planner["vx_max"]))
    except TypeError as exc:
        if "vx_max" not in str(exc) and "too many arguments" not in str(exc):
            raise
        return generate_trajectory_library_3d(**kwargs)


def generate_reverse_trajectories(planner: dict, init_p: np.ndarray, init_q: np.ndarray):
    kwargs = {
        "duration": float(planner["duration"]),
        "dt": float(planner["dt"]),
        "init_p": init_p,
        "init_q": init_q,
    }
    try:
        return generate_predefined_trajectory_vocabularies(
            **kwargs,
            reverse_speed=float(planner["reverse_speed"]),
            reverse_omegas=tuple(float(v) for v in planner["reverse_omegas"]),
        )
    except TypeError as exc:
        if "reverse_speed" not in str(exc) and "reverse_omegas" not in str(exc):
            raise
        return generate_predefined_trajectory_vocabularies(**kwargs)


def score_trajectories_compat(trajectories: np.ndarray, esdf: np.ndarray, origin: np.ndarray, resolution: float,
                              robot, front_len: float, rear_len: float, half_w: float):
    try:
        return score_trajectories_by_ESDF(
            trajectories,
            esdf,
            origin,
            resolution,
            float(robot.safety_radius),
            float(robot.comfort_radius),
            front_len,
            rear_len,
            half_w,
        )
    except TypeError as exc:
        if "too many arguments" not in str(exc):
            raise
        return score_trajectories_by_ESDF(
            trajectories,
            esdf,
            origin,
            resolution,
            float(robot.safety_radius),
            front_len,
            rear_len,
            half_w,
        )


def footprint_polygon(pose: np.ndarray, front_len: float, rear_len: float, half_w: float) -> np.ndarray:
    x, y, _, qx, qy, qz, qw = pose
    fwd = np.array([2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx)])
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-6 else np.array([1.0, 0.0])
    left = np.array([-fwd[1], fwd[0]])
    center = np.array([x, y])
    return np.array([
        center + fwd * front_len + left * half_w,
        center + fwd * front_len - left * half_w,
        center - fwd * rear_len - left * half_w,
        center - fwd * rear_len + left * half_w,
        center + fwd * front_len + left * half_w,
    ])


def trajectory_xy(traj: np.ndarray) -> list[list[float]]:
    return [[float(p[0]), float(p[1])] for p in traj]


def trajectory_yaw_deg(traj: np.ndarray) -> list[float]:
    return [yaw_from_pose(p) for p in traj]


def footprint_xy_series(traj: np.ndarray, front_len: float, rear_len: float, half_w: float) -> list[list[list[float]]]:
    return [
        [[float(p[0]), float(p[1])] for p in footprint_polygon(pose, front_len, rear_len, half_w)]
        for pose in traj
    ]

def image_u8_payload(image: np.ndarray, vmin: float, vmax: float) -> dict:
    clipped = np.clip((image.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    u8 = np.round(clipped * 255.0).astype(np.uint8)
    return {
        "width": int(u8.shape[1]),
        "height": int(u8.shape[0]),
        "data": u8.ravel().tolist(),
    }


def mask_u8_payload(mask: np.ndarray) -> dict:
    u8 = np.where(mask, 255, 0).astype(np.uint8)
    return {
        "width": int(u8.shape[1]),
        "height": int(u8.shape[0]),
        "data": u8.ravel().tolist(),
    }


def plot_result(config: dict, objects: list[SimObject], depth: np.ndarray, obstacle_mask: np.ndarray,
                esdf: np.ndarray, trajectories: np.ndarray, scores: np.ndarray, costs: np.ndarray,
                selected_index: int, robot, output_png: Path) -> None:
    origin = np.asarray(config["grid"]["origin"], dtype=np.float64)
    resolution = float(config["grid"]["resolution"])
    extent = [
        origin[1],
        origin[1] + obstacle_mask.shape[1] * resolution,
        origin[0],
        origin[0] + obstacle_mask.shape[0] * resolution,
    ]
    target = np.asarray(config["target"], dtype=np.float64)
    front_len, rear_len, half_w = robot.footprint_from_control()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    ax_scene, ax_depth, ax_esdf, ax_traj = axes.ravel()

    ax_scene.set_title("Scene layout")
    for obj in objects:
        bmin, bmax = obj.bounds
        rect = plt.Rectangle((bmin[1], bmin[0]), bmax[1] - bmin[1], bmax[0] - bmin[0],
                             facecolor="#777777", alpha=0.55, edgecolor="black")
        ax_scene.add_patch(rect)
        ax_scene.text(obj.center[1], obj.center[0], obj.name, ha="center", va="center", fontsize=8)
    ax_scene.scatter([0.0], [0.0], c="tab:green", label="start")
    ax_scene.scatter([target[1]], [target[0]], c="tab:red", label="target")
    ax_scene.set_xlabel("world y")
    ax_scene.set_ylabel("world x")
    ax_scene.axis("equal")
    ax_scene.legend(loc="upper right")

    ax_depth.set_title("Synthetic depth")
    ax_depth.imshow(depth, cmap="magma")
    ax_depth.set_axis_off()

    ax_esdf.set_title("Perceived obstacles + ESDF")
    esdf_img = ax_esdf.imshow(esdf, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=1.5)
    ax_esdf.contour(obstacle_mask.T, levels=[0.5], origin="lower", extent=extent, colors="white", linewidths=0.8)
    fig.colorbar(esdf_img, ax=ax_esdf, label="clearance (m)")
    ax_esdf.set_xlabel("world y")
    ax_esdf.set_ylabel("world x")
    ax_esdf.axis("equal")

    ax_traj.set_title("Candidate trajectories")
    ax_traj.imshow(esdf, origin="lower", extent=extent, cmap="Greys", alpha=0.35, vmin=0.0, vmax=1.0)
    finite_scores = scores[np.isfinite(scores)]
    score_hi = np.percentile(finite_scores, 80) if len(finite_scores) else 1.0
    for i, traj in enumerate(trajectories):
        color = "0.75"
        alpha = 0.25
        if np.isfinite(scores[i]):
            color = plt.cm.plasma(1.0 - min(scores[i] / max(score_hi, 1e-6), 1.0))
            alpha = 0.45
        ax_traj.plot(traj[:, 1], traj[:, 0], color=color, alpha=alpha, linewidth=0.8)
    selected = trajectories[selected_index]
    ax_traj.plot(selected[:, 1], selected[:, 0], color="#00d5ff", linewidth=3.0, label="selected")
    poly = footprint_polygon(selected[0], front_len, rear_len, half_w)
    ax_traj.plot(poly[:, 1], poly[:, 0], color="#00d5ff", linewidth=1.6, label="footprint")
    ax_traj.scatter([target[1]], [target[0]], c="tab:red")
    ax_traj.text(
        0.02,
        0.98,
        f"selected={selected_index}\nscore={scores[selected_index]:.3g}\ncost={costs[selected_index]:.1f}",
        transform=ax_traj.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    ax_traj.set_xlabel("world y")
    ax_traj.set_ylabel("world x")
    ax_traj.axis("equal")
    ax_traj.legend(loc="upper right")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def yaw_from_pose(pose: np.ndarray) -> float:
    qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
    fwd_x = 2.0 * (qx * qz + qw * qy)
    fwd_y = 2.0 * (qy * qz - qw * qx)
    if abs(fwd_x) < 1e-9 and abs(fwd_y) < 1e-9:
        return 0.0
    return float(np.rad2deg(np.arctan2(fwd_y, fwd_x)))


def run(
    config: dict,
    render_plot: bool = True,
    write_summary: bool = True,
    include_images: bool = False,
    occupancy_grid_state: np.ndarray | None = None,
    occupancy_fade: float | None = None,
    return_occupancy_grid: bool = False,
) -> dict | tuple[dict, np.ndarray]:
    robot = robot_from_config(config)
    objects = [SimObject(**obj) for obj in config["objects"]]
    grid = config["grid"]
    cam = config["camera"]
    planner = config["planner"]
    origin = np.asarray(grid["origin"], dtype=np.float64)
    grid_shape = tuple(int(v) for v in grid["shape"])
    resolution = float(grid["resolution"])

    mount_height = float(cam.get("mount_height", cam.get("height", 0.45)))
    T, init_q = make_camera_pose(
        config["start"]["xy"],
        float(config["start"]["yaw_deg"]),
        mount_height,
        float(robot.camera_x - robot.control_x),
        float(getattr(robot, "camera_y", 0.0) - getattr(robot, "control_y", 0.0)),
    )
    depth, hit_mask = render_depth(objects, T, cam)
    new_occupancy = run_synthetic_raycasting(
        depth,
        hit_mask,
        T,
        grid_shape,
        float(cam["fx"]),
        float(cam["fy"]),
        (int(cam["width"]) - 1) / 2.0,
        (int(cam.get("image_height", cam.get("height", 100))) - 1) / 2.0,
        origin,
        int(cam["ray_step"]),
        resolution,
        float(cam["max_range"]),
    )
    if occupancy_grid_state is None:
        occupancy = new_occupancy
    else:
        occupancy = occupancy_grid_state
        occupancy *= 0.994 if occupancy_fade is None else float(occupancy_fade)
        occupancy += new_occupancy
        occupancy = np.clip(occupancy, -0.2, 0.2)

    obstacle_config = ObstacleConfig(**config["obstacle"])
    obstacle_mask = build_obstacle_map(occupancy, origin, resolution, T[2, 3], obstacle_config)
    esdf = distance_transform_edt(~obstacle_mask).astype(np.float32) * resolution

    init_p = np.array([config["start"]["xy"][0], config["start"]["xy"][1], 0.0], dtype=np.float64)
    trajectories, params = generate_forward_trajectories(planner, init_p, init_q)
    reverse_trajs, reverse_params = generate_reverse_trajectories(planner, init_p, init_q)
    trajectories = np.concatenate([trajectories, reverse_trajs], axis=0)
    params = np.concatenate([params, reverse_params], axis=0)

    front_len, rear_len, half_w = robot.footprint_from_control()
    scores, occ_points = score_trajectories_compat(
        trajectories, esdf, origin, resolution, robot, front_len, rear_len, half_w
    )
    scores = np.asarray(scores, dtype=np.float64)
    front_clearance = front_obstacle_dist(
        init_p,
        float(config["start"]["yaw_deg"]),
        obstacle_mask,
        origin,
        resolution,
        front_len,
        half_w,
    )
    selection = select_trajectory_with_recovery(
        trajectories,
        params,
        scores,
        esdf,
        origin,
        resolution,
        float(robot.safety_radius),
        float(robot.comfort_radius),
        front_len,
        rear_len,
        half_w,
        np.asarray(config["target"], dtype=np.float64),
        front_clearance,
        float(planner["front_reverse_threshold"]),
        float(planner["trajectory_smooth_weight"]),
        np.asarray(planner["last_param"], dtype=np.float64),
        0.0,
        0,
        top_k=1,
    )
    selected_index = selection["selected_index"]
    costs = selection["costs"]

    output_dir = Path(config["output_dir"])
    output_png = output_dir / f"{config['name']}.png"
    if render_plot:
        plot_result(config, objects, depth, obstacle_mask, esdf, trajectories, scores, costs, selected_index, robot, output_png)

    summary = {
        "name": config["name"],
        "output_png": str(output_png),
        "selected_index": int(selected_index),
        "selected_param": params[selected_index].round(4).tolist(),
        "selected_score": float(scores[selected_index]),
        "selected_cost": float(costs[selected_index]),
        "should_reverse": bool(selection["should_reverse"]),
        "all_collision": bool(selection["all_collision"]),
        "recovery_reason": selection["recovery_reason"],
        "selected_is_reverse": bool(selection["selected_is_reverse"]),
        "front_clearance": float(front_clearance),
        "valid_trajectories": int(np.sum(np.isfinite(scores))),
        "trajectory_count": int(len(trajectories)),
        "obstacle_cells": int(np.sum(obstacle_mask)),
        "selected_closest_step": int(occ_points[selected_index]),
        "selected_trajectory_xy": trajectory_xy(trajectories[selected_index]),
        "selected_trajectory_yaw_deg": trajectory_yaw_deg(trajectories[selected_index]),
        "selected_footprints_xy": footprint_xy_series(trajectories[selected_index], front_len, rear_len, half_w),
        "candidate_trajectories_xy": [
            trajectory_xy(trajectories[i])
            for i in np.argsort(costs, kind="stable")[: min(40, len(trajectories))]
            if np.isfinite(costs[i])
        ],
        "obstacle_outline_xy": [
            [float(obj.center[0]), float(obj.center[1]), float(obj.size[0]), float(obj.size[1])]
            for obj in objects
        ],
        "robot_footprint": {
            "front_len": float(front_len),
            "rear_len": float(rear_len),
            "half_w": float(half_w),
        },
    }
    if include_images:
        summary["depth_u8"] = image_u8_payload(depth, 0.0, float(cam["max_range"]))
        summary["esdf_u8"] = image_u8_payload(esdf, 0.0, 1.5)
        summary["obstacle_u8"] = mask_u8_payload(obstacle_mask)
    if write_summary:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / f"{config['name']}.summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary_json"] = str(summary_path)
    if return_occupancy_grid:
        return summary, occupancy
    return summary


def run_realtime_step(config: dict, advance_step: int = 3) -> dict:
    """Compute one closed-loop step from the current config pose and scene."""
    summary = run(config, render_plot=False, write_summary=False, include_images=True)
    traj = summary["selected_trajectory_xy"]
    yaws = summary["selected_trajectory_yaw_deg"]
    footprints = summary["selected_footprints_xy"]
    idx = min(max(1, int(advance_step)), len(traj) - 1)
    next_xy = traj[idx]
    next_yaw = float(yaws[min(idx, len(yaws) - 1)]) if yaws else float(config["start"]["yaw_deg"])
    frame = {
        "robot_xy": next_xy,
        "robot_yaw_deg": next_yaw,
        "robot_footprint_xy": footprints[idx],
        "selected_trajectory_xy": summary["selected_trajectory_xy"],
        "candidate_trajectories_xy": summary["candidate_trajectories_xy"],
        "obstacle_outline_xy": summary["obstacle_outline_xy"],
        "selected_param": summary["selected_param"],
        "should_reverse": summary["should_reverse"],
        "all_collision": summary["all_collision"],
        "recovery_reason": summary["recovery_reason"],
        "selected_is_reverse": summary["selected_is_reverse"],
        "front_clearance": summary["front_clearance"],
        "valid_trajectories": summary["valid_trajectories"],
        "obstacle_cells": summary["obstacle_cells"],
        "depth_u8": summary["depth_u8"],
        "esdf_u8": summary["esdf_u8"],
        "next_start": {
            "xy": next_xy,
            "yaw_deg": next_yaw,
        },
    }
    return frame


def run_realtime_step_with_grid(
    config: dict,
    occupancy_grid_state: np.ndarray | None = None,
    advance_step: int = 3,
    occupancy_fade: float = 0.994,
) -> tuple[dict, np.ndarray]:
    """Compute one realtime step while preserving planning_node-style occupancy fadeaway."""
    summary, updated_grid = run(
        config,
        render_plot=False,
        write_summary=False,
        include_images=True,
        occupancy_grid_state=occupancy_grid_state,
        occupancy_fade=occupancy_fade,
        return_occupancy_grid=True,
    )
    traj = summary["selected_trajectory_xy"]
    yaws = summary["selected_trajectory_yaw_deg"]
    footprints = summary["selected_footprints_xy"]
    idx = min(max(1, int(advance_step)), len(traj) - 1)
    next_xy = traj[idx]
    next_yaw = float(yaws[min(idx, len(yaws) - 1)]) if yaws else float(config["start"]["yaw_deg"])
    frame = {
        "robot_xy": next_xy,
        "robot_yaw_deg": next_yaw,
        "robot_footprint_xy": footprints[idx],
        "selected_trajectory_xy": summary["selected_trajectory_xy"],
        "candidate_trajectories_xy": summary["candidate_trajectories_xy"],
        "obstacle_outline_xy": summary["obstacle_outline_xy"],
        "selected_param": summary["selected_param"],
        "should_reverse": summary["should_reverse"],
        "all_collision": summary["all_collision"],
        "recovery_reason": summary["recovery_reason"],
        "selected_is_reverse": summary["selected_is_reverse"],
        "front_clearance": summary["front_clearance"],
        "valid_trajectories": summary["valid_trajectories"],
        "obstacle_cells": summary["obstacle_cells"],
        "depth_u8": summary["depth_u8"],
        "esdf_u8": summary["esdf_u8"],
        "obstacle_u8": summary["obstacle_u8"],
        "next_start": {
            "xy": next_xy,
            "yaw_deg": next_yaw,
        },
    }
    return frame, updated_grid


def run_rollout(config: dict, steps: int = 30, advance_step: int = 3, render_snapshots: bool = False) -> dict:
    """Closed-loop rollout: replan from the updated robot pose at every tick."""
    sim_config = copy.deepcopy(config)
    frames = []
    executed_path = []
    selected_index_history = []
    selected_param_history = []
    first_summary = None

    for step in range(max(1, int(steps))):
        sim_config["name"] = f"{config.get('name', 'web_scene')}_rollout_{step:03d}"
        summary = run(sim_config, render_plot=render_snapshots, write_summary=False)
        if first_summary is None:
            first_summary = summary

        traj = summary["selected_trajectory_xy"]
        yaws = summary["selected_trajectory_yaw_deg"]
        footprints = summary["selected_footprints_xy"]
        idx = min(max(1, int(advance_step)), len(traj) - 1)
        robot_xy = traj[idx]
        robot_fp = footprints[idx]
        executed_path.append(robot_xy)
        selected_index_history.append(summary["selected_index"])
        selected_param_history.append(summary["selected_param"])
        frames.append({
            "step": step,
            "robot_xy": robot_xy,
            "robot_yaw_deg": float(yaws[min(idx, len(yaws) - 1)]) if yaws else float(sim_config["start"]["yaw_deg"]),
            "robot_footprint_xy": robot_fp,
            "selected_trajectory_xy": summary["selected_trajectory_xy"],
            "candidate_trajectories_xy": summary["candidate_trajectories_xy"],
            "obstacle_outline_xy": summary["obstacle_outline_xy"],
            "selected_param": summary["selected_param"],
            "front_clearance": summary["front_clearance"],
            "valid_trajectories": summary["valid_trajectories"],
            "obstacle_cells": summary["obstacle_cells"],
            "diagnostic_png": summary.get("output_png"),
        })

        # Keep pose update from the trajectory orientation, not from displacement.
        # Reverse motion has displacement opposite to heading; deriving yaw from dx/dy
        # would make the simulated robot "turn around" instead of backing up.
        sim_config["start"]["xy"] = robot_xy
        if yaws:
            sim_config["start"]["yaw_deg"] = float(yaws[min(idx, len(yaws) - 1)])

        target = np.asarray(sim_config["target"][:2], dtype=np.float64)
        if np.linalg.norm(np.asarray(robot_xy, dtype=np.float64) - target) < 0.25:
            break

    output = {
        "name": config.get("name", "web_scene"),
        "frames": frames,
        "executed_path_xy": executed_path,
        "selected_index_history": selected_index_history,
        "selected_param_history": selected_param_history,
        "frame_count": len(frames),
    }
    if first_summary is not None:
        output["diagnostic_summary"] = first_summary
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional JSON scene config. Values override the built-in default.")
    parser.add_argument("--output-dir", help="Override output directory from config.")
    parser.add_argument("--write-example", help="Write the default editable JSON config and exit.")
    args = parser.parse_args()

    if args.write_example:
        path = Path(args.write_example)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(default_config(), indent=2), encoding="utf-8")
        print(f"Wrote example config: {path}")
        return

    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    summary = run(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
