"""Shared synthetic scene helpers for the planning lab.

Used by the ROS-backed web simulator. Keeps box geometry, camera pose and
depth rendering in one place so we do not grow a second copy of the same sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tool.simulator.map_volume import MapVolume, raycast_map


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


def box(name: str, center: list[float], size: list[float]) -> dict[str, Any]:
    return {"name": name, "kind": "box", "center": center, "size": size}


def cam_size(cam: dict[str, Any]) -> tuple[int, int]:
    return int(cam["width"]), int(cam.get("image_height", cam.get("height", 100)))


def make_camera_pose(
    control_xy: list[float],
    yaw_deg: float,
    *,
    mount_height: float,
    forward_offset: float = 0.0,
    left_offset: float = 0.0,
) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_deg))
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float64)
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0], dtype=np.float64)
    down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    pos = np.array([control_xy[0], control_xy[1], float(mount_height)], dtype=np.float64)
    pos[:2] += forward[:2] * float(forward_offset) + left[:2] * float(left_offset)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.column_stack([right, down, forward])
    T[:3, 3] = pos
    return T


def make_camera_pose_from_config(
    control_xy: list[float],
    yaw_deg: float,
    robot: dict[str, Any],
    cam: dict[str, Any],
) -> np.ndarray:
    return make_camera_pose(
        control_xy,
        yaw_deg,
        mount_height=float(cam.get("mount_height", 0.45)),
        forward_offset=float(robot.get("camera_x", 0.0)) - float(robot.get("control_x", 0.0)),
        left_offset=float(robot.get("camera_y", 0.0)) - float(robot.get("control_y", 0.0)),
    )


def render_depth(
    objects: list[SimObject],
    T_cam_to_world: np.ndarray,
    cam: dict[str, Any],
    map_volume: MapVolume | None = None,
) -> np.ndarray:
    """Pinhole depth with optional ground plane (default z=0) and map voxels."""
    width, height = cam_size(cam)
    fx, fy = float(cam["fx"]), float(cam["fy"])
    max_range = float(cam["max_range"])
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0

    us, vs = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    rays_cam = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], axis=-1)
    rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    rays = (rays_cam @ T_cam_to_world[:3, :3].T).reshape((-1, 3))
    z_cam = rays_cam[..., 2].reshape(-1)
    origin = T_cam_to_world[:3, 3]
    best = np.full(rays.shape[0], np.inf)

    ground_z = float(cam.get("ground_z", 0.0))
    dz = rays[:, 2]
    down = dz < -1e-9
    t_ground = np.full(rays.shape[0], np.inf)
    t_ground[down] = (ground_z - origin[2]) / dz[down]
    hit_ground = down & (t_ground > 1e-4) & (t_ground <= max_range)
    best = np.where(hit_ground, t_ground, best)

    if map_volume is not None:
        t_map = raycast_map(origin, rays, z_cam, max_range, map_volume)
        best = np.where(np.isfinite(t_map) & (t_map < best), t_map, best)

    for obj in objects:
        box_min, box_max = obj.bounds
        inv = np.divide(1.0, rays, out=np.full_like(rays, np.inf), where=np.abs(rays) > 1e-9)
        t0, t1 = (box_min - origin) * inv, (box_max - origin) * inv
        t_near = np.maximum.reduce(np.minimum(t0, t1), axis=1)
        t_far = np.minimum.reduce(np.maximum(t0, t1), axis=1)
        hit = np.where(t_near > 0.0, t_near, t_far)
        valid = (t_far >= 0.0) & (t_near <= t_far) & (hit > 0.0) & (hit <= max_range)
        best = np.where(valid & (hit < best), hit, best)

    depth = np.zeros(best.shape[0], dtype=np.float32)
    ok = np.isfinite(best)
    depth[ok] = (best[ok] * z_cam[ok]).astype(np.float32)
    return depth.reshape((height, width))


def footprint_polygon_xy(control_xy, yaw_rad, robot: dict[str, Any]) -> np.ndarray:
    """Control-frame rectangle in world XY, matching planning_node footprint_from_control."""
    xy = np.asarray(control_xy, dtype=np.float64)[:2]
    yaw = float(yaw_rad)
    fwd = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    left = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
    length = float(robot.get("length", 0.4))
    width = float(robot.get("width", 0.3))
    cx = float(robot.get("control_x", 0.0))
    hl, hw = 0.5 * length, 0.5 * width
    front, rear = hl - cx, hl + cx
    return np.stack([
        xy + fwd * front + left * hw,
        xy + fwd * front - left * hw,
        xy - fwd * rear - left * hw,
        xy - fwd * rear + left * hw,
    ])


def _project_xy(points: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    dots = points @ axis
    return float(dots.min()), float(dots.max())


def _obb_hits_aabb(poly: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
    half = 0.5 * (aabb_max - aabb_min)
    mid = 0.5 * (aabb_max + aabb_min)
    e0 = poly[1] - poly[0]
    e1 = poly[3] - poly[0]
    axes = (np.array([1.0, 0.0]), np.array([0.0, 1.0]),
            e0 / (np.linalg.norm(e0) + 1e-12), e1 / (np.linalg.norm(e1) + 1e-12))
    for axis in axes:
        a0, a1 = _project_xy(poly, axis)
        c = float(mid @ axis)
        r = abs(axis[0]) * half[0] + abs(axis[1]) * half[1]
        if a1 < c - r or c + r < a0:
            return False
    return True


def robot_hits_objects(control_xy, yaw_deg, robot: dict[str, Any], objects: list[SimObject]) -> bool:
    """XY footprint vs scene boxes. Ignores map voxels and occupancy/ESDF."""
    poly = footprint_polygon_xy(control_xy, np.deg2rad(float(yaw_deg)), robot)
    for obj in objects:
        if obj.kind != "box":
            continue
        lo, hi = obj.bounds
        if _obb_hits_aabb(poly, lo[:2], hi[:2]):
            return True
    return False


def image_u8_payload(image: np.ndarray, vmin: float, vmax: float) -> dict[str, Any]:
    u8 = np.clip((image.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    u8 = np.round(u8 * 255.0).astype(np.uint8)
    return {"width": int(u8.shape[1]), "height": int(u8.shape[0]), "data": u8.ravel().tolist()}
