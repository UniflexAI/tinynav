"""Load TinyNav map occupancy volumes for the planning web simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit


OCCUPIED = 2


@dataclass(frozen=True)
class MapInfo:
    map_path: str
    origin_x: float
    origin_y: float
    origin_z: float
    resolution: float
    width: int   # Nx (world X)
    height: int  # Ny (world Y)
    depth: int   # Nz (world Z)

    @property
    def x_min(self) -> float:
        return self.origin_x

    @property
    def x_max(self) -> float:
        return self.origin_x + self.width * self.resolution

    @property
    def y_min(self) -> float:
        return self.origin_y

    @property
    def y_max(self) -> float:
        return self.origin_y + self.height * self.resolution

    def as_dict(self) -> dict:
        return {
            "map_path": self.map_path,
            "origin": [self.origin_x, self.origin_y, self.origin_z],
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "bounds": {
                "x_min": self.x_min,
                "x_max": self.x_max,
                "y_min": self.y_min,
                "y_max": self.y_max,
            },
        }


@dataclass
class MapVolume:
    grid: np.ndarray
    origin: np.ndarray
    resolution: float
    map_path: str

    @classmethod
    def load(cls, map_path: str | Path) -> MapVolume:
        root = Path(map_path).expanduser().resolve()
        grid_file = root / "occupancy_grid.npy"
        meta_file = root / "occupancy_meta.npy"
        if not grid_file.is_file() or not meta_file.is_file():
            raise FileNotFoundError(
                f"Map needs occupancy_grid.npy and occupancy_meta.npy under {root}"
            )
        grid = np.load(grid_file)
        meta = np.load(meta_file).astype(np.float64)
        if grid.ndim != 3:
            raise ValueError(f"occupancy_grid must be 3D, got shape {grid.shape}")
        if meta.shape[0] < 4:
            raise ValueError(f"occupancy_meta must have 4 values, got {meta.shape}")
        return cls(
            grid=np.ascontiguousarray(grid, dtype=np.uint8),
            origin=np.array(meta[:3], dtype=np.float64),
            resolution=float(meta[3]),
            map_path=str(root),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.grid.shape)

    def info(self) -> MapInfo:
        nx, ny, nz = self.shape
        return MapInfo(
            map_path=self.map_path,
            origin_x=float(self.origin[0]),
            origin_y=float(self.origin[1]),
            origin_z=float(self.origin[2]),
            resolution=float(self.resolution),
            width=nx,
            height=ny,
            depth=nz,
        )

    def default_start_xy(self) -> tuple[float, float]:
        info = self.info()
        return (
            (info.x_min + info.x_max) * 0.5,
            (info.y_min + info.y_max) * 0.5,
        )

    def background_rgb(self) -> np.ndarray:
        """Top-down RGB preview for the planning web UI.

        Canvas convention (must match ``worldToCanvas`` in app.js):
        - image width  = world Y  (screen horizontal)
        - image height = world X  (screen vertical, +X points up)
        """
        occupied = np.any(self.grid == OCCUPIED, axis=2)
        free = np.any(self.grid == 1, axis=2)
        nx, ny = occupied.shape
        img = np.full((nx, ny, 3), 128, dtype=np.uint8)
        img[free] = (210, 210, 210)
        img[occupied] = (35, 35, 35)
        # Row 0 → max world X so +X is up on screen after worldToCanvas.
        return np.flipud(img)


@njit(cache=True)
def _raycast_map(origin, rays, z_cam, max_range, grid, origin_xyz, resolution):
    n_rays = rays.shape[0]
    nx, ny, nz = grid.shape[0], grid.shape[1], grid.shape[2]
    ox, oy, oz = origin_xyz[0], origin_xyz[1], origin_xyz[2]
    inv_res = 1.0 / resolution
    best = np.empty(n_rays, dtype=np.float64)
    for i in range(n_rays):
        best[i] = np.inf
        dx, dy, dz = rays[i, 0], rays[i, 1], rays[i, 2]
        zc = z_cam[i]
        if zc <= 1e-9:
            continue
        ix = int(np.floor((origin[0] - ox) * inv_res))
        iy = int(np.floor((origin[1] - oy) * inv_res))
        iz = int(np.floor((origin[2] - oz) * inv_res))
        if ix < 0 or iy < 0 or iz < 0 or ix >= nx or iy >= ny or iz >= nz:
            continue
        step_x = 1 if dx >= 0.0 else -1
        step_y = 1 if dy >= 0.0 else -1
        step_z = 1 if dz >= 0.0 else -1
        if abs(dx) < 1e-12:
            t_delta_x = np.inf
            t_max_x = np.inf
        else:
            next_x = ix + (1 if dx >= 0.0 else 0)
            t_delta_x = abs(resolution / dx)
            t_max_x = ((ox + next_x * resolution) - origin[0]) / dx
        if abs(dy) < 1e-12:
            t_delta_y = np.inf
            t_max_y = np.inf
        else:
            next_y = iy + (1 if dy >= 0.0 else 0)
            t_delta_y = abs(resolution / dy)
            t_max_y = ((oy + next_y * resolution) - origin[1]) / dy
        if abs(dz) < 1e-12:
            t_delta_z = np.inf
            t_max_z = np.inf
        else:
            next_z = iz + (1 if dz >= 0.0 else 0)
            t_delta_z = abs(resolution / dz)
            t_max_z = ((oz + next_z * resolution) - origin[2]) / dz
        dist = 0.0
        while dist <= max_range:
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                if grid[ix, iy, iz] == OCCUPIED:
                    if dist > 1e-4:
                        best[i] = dist
                    break
            else:
                break
            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    dist = t_max_x
                    t_max_x += t_delta_x
                    ix += step_x
                else:
                    dist = t_max_z
                    t_max_z += t_delta_z
                    iz += step_z
            elif t_max_y < t_max_z:
                dist = t_max_y
                t_max_y += t_delta_y
                iy += step_y
            else:
                dist = t_max_z
                t_max_z += t_delta_z
                iz += step_z
    return best


def raycast_map(
    origin: np.ndarray,
    rays: np.ndarray,
    z_cam: np.ndarray,
    max_range: float,
    volume: MapVolume,
) -> np.ndarray:
    return _raycast_map(
        origin.astype(np.float64),
        np.ascontiguousarray(rays, dtype=np.float64),
        np.ascontiguousarray(z_cam, dtype=np.float64),
        float(max_range),
        volume.grid,
        volume.origin.astype(np.float64),
        float(volume.resolution),
    )
