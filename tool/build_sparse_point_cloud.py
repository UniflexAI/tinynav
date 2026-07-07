#!/usr/bin/env python3
"""
tool/build_sparse_point_cloud.py

Multi-view depth fusion: back-project a subsample of keyframe depth maps
(saved by build_map_node.py under a map dir) into one colored, world-frame
surfel-filtered point cloud, written as a .ply.

Each retained point carries a surface normal (estimated from local depth
gradients) and is rejected outright if its normal is too oblique to the
viewing ray — a standard surfel-fusion quality gate that throws out grazing,
depth-noisy pixels before they ever reach the point cloud. Normals are
written as extra ply vertex properties (nx/ny/nz) for inspection/reuse; they
are NOT read by nerfstudio's stock splatfacto (its gaussian init only reads
xyz/rgb from the ply — scale comes from nearest-neighbor spacing, rotation is
random), so this is a point-cloud *quality* improvement, not a splatfacto
initialization change.

Used by tool/convert_to_nerf_format.py to seed nerfstudio/splatfacto's
gaussian positions with real geometry instead of random init — without this,
splatfacto falls back to random gaussian initialization.

Usage (standalone):
    python tool/build_sparse_point_cloud.py --map-dir <map_dir>
"""

import argparse
import shelve
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from tool.video_db import VideoDB


def _backproject_depth_grid(
    depth: np.ndarray, K: np.ndarray, step: int, max_depth_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Pinhole back-project a strided pixel grid into the depth camera's own
    frame, keeping the (rows, cols, 3) grid shape (rather than flattening) so
    callers can do neighbor-based ops — e.g. surface normals — before masking
    down to a flat point list.

    Returns (points_cam_grid [R,C,3], valid [R,C])."""
    h, w = depth.shape
    us = np.arange(0, w, step)
    vs = np.arange(0, h, step)
    grid_u, grid_v = np.meshgrid(us, vs)
    z = depth[grid_v, grid_u]
    valid = (z > 0.0) & (z <= max_depth_m)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (grid_u - cx) * z / fx
    y = (grid_v - cy) * z / fy
    points_cam_grid = np.stack([x, y, z], axis=-1)
    return points_cam_grid, valid


def _estimate_surfel_normals(
    points_cam_grid: np.ndarray, valid: np.ndarray, min_view_cos: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference surface normals on the back-projected pixel grid,
    oriented to face the camera, plus a keep-mask that additionally rejects
    points whose normal is too oblique to the viewing ray (grazing incidence
    — the classic source of noisy/unreliable depth-derived surfels).

    Border pixels (no full set of 4 neighbors) are marked invalid — normal
    estimation needs a local neighborhood, unlike a bare position lookup.

    Returns (normals_cam_grid [R,C,3], keep [R,C])."""
    R, C, _ = points_cam_grid.shape
    normals = np.zeros((R, C, 3), dtype=np.float64)
    keep = np.zeros((R, C), dtype=bool)
    if R < 3 or C < 3:
        return normals, keep

    center = points_cam_grid[1:-1, 1:-1]
    du = points_cam_grid[1:-1, 2:] - points_cam_grid[1:-1, :-2]
    dv = points_cam_grid[2:, 1:-1] - points_cam_grid[:-2, 1:-1]
    n = np.cross(du, dv)
    n_norm = np.linalg.norm(n, axis=-1)
    finite = n_norm > 1e-9
    safe_norm = np.where(finite, n_norm, 1.0)
    unit_n = n / safe_norm[..., None]

    # Camera is at the origin of its own frame, so each point's own position
    # is exactly the viewing ray from camera to point. A front-facing normal
    # should point back toward the camera, i.e. opposite the viewing ray.
    view_dir = center
    view_dist = np.linalg.norm(view_dir, axis=-1)
    safe_view_dist = np.where(view_dist > 1e-9, view_dist, 1.0)
    view_cos = -np.sum(unit_n * view_dir, axis=-1) / safe_view_dist
    flip = view_cos < 0
    unit_n[flip] *= -1
    view_cos = np.abs(view_cos)

    neighbors_valid = (
        valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
        & valid[2:, 1:-1] & valid[:-2, 1:-1]
    )
    center_keep = neighbors_valid & finite & (view_dist > 1e-9) & (view_cos >= min_view_cos)

    normals[1:-1, 1:-1] = unit_n
    keep[1:-1, 1:-1] = center_keep
    return normals, keep


def export_sparse_point_cloud(
    map_dir: Path,
    poses: Dict[int, np.ndarray],
    depth_intrinsics: np.ndarray,
    max_points: int = 500_000,
    frame_stride: int = 10,
    pixel_step: int = 8,
    max_depth_m: float = 8.0,
    min_view_cos: float = 0.2,
) -> Optional[str]:
    """Fuse a subsample of keyframe depth maps (already available in map_dir
    from build_map_node.py) into a colored, world-frame, surfel-filtered
    point cloud, written to <map_dir>/sparse_pc.ply.

    Each keyframe's depth is back-projected in its own camera frame — with a
    per-point surface normal estimated from local depth gradients, and any
    point whose normal is more than ~acos(min_view_cos) off the viewing ray
    dropped as an unreliable grazing-incidence measurement — then transformed
    into the world frame using that keyframe's pose (poses[timestamp], already
    loop-closure-optimized). Multiple keyframes are fused into one point cloud,
    not a single-view snapshot.

    Color comes from the infra1 (grayscale) image, not RGB — it's exactly
    co-registered with the depth map (same camera), so no extra reprojection
    through T_rgb_to_infra1 (and its extra error) is needed just to seed a
    plausible init color; splatfacto will refine color during training anyway.

    Returns the ply filename (relative to map_dir) on success, or None if no
    depth was available (e.g. depths.db missing) — training then falls back
    to random init, exactly as before this function existed.
    """
    if not (map_dir / "depths.db").exists():
        print(f"No depths.db under {map_dir}; skipping point-cloud init "
              f"(splatfacto will use random init)")
        return None
    depths_db = shelve.open(str(map_dir / "depths"), flag="r")
    infra1_db = VideoDB(dir_path=str(map_dir / "infra1_images_db"), mode="read")

    timestamps = sorted(int(k) for k in poses.keys())[::frame_stride]
    points_world_chunks = []
    normals_world_chunks = []
    color_chunks = []
    n_rejected_oblique = 0
    for timestamp in tqdm(timestamps, desc="Backprojecting depth for point-cloud init", unit="frame"):
        key = str(timestamp)
        if key not in depths_db:
            continue
        depth = np.asarray(depths_db[key])
        gray = infra1_db.read(timestamp)
        if gray is None:
            continue

        points_cam_grid, valid = _backproject_depth_grid(depth, depth_intrinsics, pixel_step, max_depth_m)
        normals_cam_grid, keep = _estimate_surfel_normals(points_cam_grid, valid, min_view_cos)
        n_rejected_oblique += int(np.count_nonzero(valid[1:-1, 1:-1]) - np.count_nonzero(keep[1:-1, 1:-1]))
        if not np.any(keep):
            continue

        us = np.arange(0, depth.shape[1], pixel_step)
        vs = np.arange(0, depth.shape[0], pixel_step)
        grid_u, grid_v = np.meshgrid(us, vs)
        gu, gv = grid_u[keep], grid_v[keep]
        points_cam = points_cam_grid[keep]
        normals_cam = normals_cam_grid[keep]
        gray_vals = gray[gv, gu].astype(np.float64)

        T_world_infra1 = poses[timestamp]
        R_world_infra1 = T_world_infra1[:3, :3]
        points_h = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)
        points_world_chunks.append((T_world_infra1 @ points_h.T).T[:, :3])
        normals_world_chunks.append((R_world_infra1 @ normals_cam.T).T)
        color_chunks.append(np.stack([gray_vals, gray_vals, gray_vals], axis=-1))
    infra1_db.close()
    depths_db.close()

    if not points_world_chunks:
        print("No depth frames back-projected; skipping point-cloud init")
        return None
    print(f"Surfel filter rejected {n_rejected_oblique} points "
          f"(grazing incidence view_cos < {min_view_cos}, or a border/invalid-neighbor pixel)")

    points_world = np.concatenate(points_world_chunks, axis=0)
    normals_world = np.concatenate(normals_world_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    if points_world.shape[0] > max_points:
        keep = np.random.choice(points_world.shape[0], size=max_points, replace=False)
        points_world, normals_world, colors = points_world[keep], normals_world[keep], colors[keep]

    from plyfile import PlyData, PlyElement

    vertex = np.zeros(
        points_world.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = points_world[:, 0], points_world[:, 1], points_world[:, 2]
    vertex["nx"], vertex["ny"], vertex["nz"] = normals_world[:, 0], normals_world[:, 1], normals_world[:, 2]
    gray_u8 = np.clip(colors[:, 0], 0, 255).astype(np.uint8)
    vertex["red"], vertex["green"], vertex["blue"] = gray_u8, gray_u8, gray_u8

    ply_file_name = "sparse_pc.ply"
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(map_dir / ply_file_name))
    print(f"Wrote {points_world.shape[0]} point-cloud-init points to {map_dir / ply_file_name}")
    return ply_file_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse keyframe depth maps into a world-frame point cloud (.ply)"
    )
    parser.add_argument("--map-dir", required=True,
                        help="TinyNav map directory containing poses.npy/intrinsics.npy/depths.db")
    parser.add_argument("--max-points", type=int, default=500_000,
                        help="Cap on total points written (default: 500000)")
    parser.add_argument("--frame-stride", type=int, default=10,
                        help="Use every Nth keyframe's depth (default: 10)")
    parser.add_argument("--pixel-step", type=int, default=8,
                        help="Depth pixel grid stride per frame (default: 8)")
    parser.add_argument("--max-depth", type=float, default=8.0,
                        help="Ignore depth beyond this range in meters (default: 8.0)")
    parser.add_argument("--min-view-cos", type=float, default=0.2,
                        help="Reject surfels whose normal is more than "
                             "~acos(min_view_cos) off the viewing ray (default: 0.2, ~78deg)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_dir = Path(args.map_dir)
    poses = np.load(map_dir / "poses.npy", allow_pickle=True).item()
    depth_intrinsics = np.load(map_dir / "intrinsics.npy", allow_pickle=True)
    export_sparse_point_cloud(
        map_dir=map_dir,
        poses=poses,
        depth_intrinsics=depth_intrinsics,
        max_points=args.max_points,
        frame_stride=args.frame_stride,
        pixel_step=args.pixel_step,
        max_depth_m=args.max_depth,
        min_view_cos=args.min_view_cos,
    )


if __name__ == "__main__":
    main()
