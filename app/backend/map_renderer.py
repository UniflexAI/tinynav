"""
Render occupancy_grid.npy → PNG (top-down 2D projection).
"""
from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image


def render_map(map_path: str) -> tuple[bytes, dict]:
    """
    Load occupancy_grid.npy / occupancy_meta.npy from *map_path* and
    return (png_bytes, metadata_dict).

    Grid values: 0 = unknown, 1 = free, 2 = occupied
    Meta format: [origin_x, origin_y, origin_z, resolution]
    Grid shape:  (Nx, Ny, Nz) where axis 0 = world-X, 1 = world-Y, 2 = world-Z
    """
    grid_file = os.path.join(map_path, 'occupancy_grid.npy')
    meta_file = os.path.join(map_path, 'occupancy_meta.npy')

    if not os.path.exists(grid_file) or not os.path.exists(meta_file):
        raise FileNotFoundError(f'Map files not found in {map_path}')

    grid = np.load(grid_file)   # (Nx, Ny, Nz)
    meta = np.load(meta_file)   # [ox, oy, oz, res]

    origin_x, origin_y, origin_z, resolution = (
        float(meta[0]), float(meta[1]), float(meta[2]), float(meta[3])
    )

    # Project along Z (axis=2) → (Nx, Ny) 2-D grid
    occupied_2d = np.any(grid == 2, axis=2)   # (Nx, Ny)
    free_2d = np.any(grid == 1, axis=2)        # (Nx, Ny)

    nx, ny = occupied_2d.shape

    # Colour convention:  unknown = medium gray, free = light gray, occupied = near-black
    img = np.full((nx, ny, 3), 128, dtype=np.uint8)
    img[free_2d] = [210, 210, 210]
    img[occupied_2d] = [35, 35, 35]

    # Grid is (Nx, Ny, Nz): axis-0=X, axis-1=Y.
    # Transpose (1,0,2) → (Ny, Nx, 3): rows=Y, cols=X.
    # flipud → row 0 = max Y, so Y increases upward (matches painter: X=right, Y=up).
    img = np.flipud(img.transpose(1, 0, 2))

    pil_img = Image.fromarray(img, mode='RGB')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG', optimize=True)

    return buf.getvalue(), {
        'origin_x': origin_x,
        'origin_y': origin_y,
        'resolution': resolution,
        'width': nx,    # image cols (Nx) ↔ world-X (horizontal)
        'height': ny,   # image rows (Ny) ↔ world-Y (vertical, inverted)
    }
