"""BEV grid definition and camera <-> BEV geometry shared by label generation, training and runtime.

BEV frame: gravity aligned, x forward along the camera heading, y left, z up, origin at the camera.
Grid rows run from far forward (row 0) to behind; columns from far left (col 0) to right.
"""
import numpy as np

RES = 0.05
BEV_X = (-1.6, 4.8)
BEV_Y = (-3.2, 3.2)
BEV_N = 128
# heights (m, relative to the camera) at which image features are sampled into the BEV
Z_SLICES = np.linspace(-1.0, 0.4, 8).astype(np.float32)


def heading_rotation(R_cam_to_world):
    """World -> BEV rotation: keeps the camera yaw, removes pitch and roll."""
    fwd = R_cam_to_world[:, 2].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd) + 1e-9
    return np.stack([fwd, [-fwd[1], fwd[0], 0.0], [0.0, 0.0, 1.0]])


def cam_to_bev_rotation(R_cam_to_world):
    """Camera optical frame -> BEV frame; depends only on camera pitch and roll."""
    return heading_rotation(R_cam_to_world) @ R_cam_to_world


def cell_centers():
    xs = BEV_X[1] - (np.arange(BEV_N) + 0.5) * RES
    ys = BEV_Y[1] - (np.arange(BEV_N) + 0.5) * RES
    return np.meshgrid(xs, ys, indexing='ij')


def to_grid(pb):
    """BEV points (N,3) -> row, col, inside mask."""
    r = np.floor((BEV_X[1] - pb[:, 0]) / RES).astype(np.int64)
    q = np.floor((BEV_Y[1] - pb[:, 1]) / RES).astype(np.int64)
    return r, q, (r >= 0) & (r < BEV_N) & (q >= 0) & (q < BEV_N)


def depth_to_points(depth_m, K, step=2, zmax=6.0):
    """Camera-frame points from a metric depth image."""
    d = depth_m[::step, ::step]
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]] * step
    ok = (d > 0.15) & (d < zmax)
    z = d[ok]
    return np.stack([(u[ok] - K[0, 2]) / K[0, 0] * z, (v[ok] - K[1, 2]) / K[1, 1] * z, z], 1)


def depth_raster(depth_m, K, R_cb, step=2):
    """Runtime-depth BEV channels: log point count, min z, max z (rel. camera), observed."""
    pb = depth_to_points(depth_m, K, step) @ R_cb.T
    r, q, ok = to_grid(pb)
    r, q, z = r[ok], q[ok], pb[ok, 2]
    cnt = np.zeros((BEV_N, BEV_N), np.float32)
    zmin = np.full((BEV_N, BEV_N), 9.0, np.float32)
    zmax = np.full((BEV_N, BEV_N), -9.0, np.float32)
    np.add.at(cnt, (r, q), 1)
    np.minimum.at(zmin, (r, q), z)
    np.maximum.at(zmax, (r, q), z)
    seen = cnt > 0
    return np.stack([np.log1p(cnt), np.where(seen, zmin, 0), np.where(seen, zmax, 0), seen.astype(np.float32)])


def sampling_grid(K, R_cb, hw):
    """Normalized image coords of every (z slice, BEV cell) center, for F.grid_sample.

    Returns grid (Z*N, N, 2) in [-1, 1] and valid (Z, N, N) = in front of the camera and inside the image.
    """
    X, Y = cell_centers()
    Z = Z_SLICES[:, None, None]
    pb = np.stack(np.broadcast_arrays(X[None], Y[None], Z), -1).reshape(-1, 3)
    pc = pb @ R_cb  # BEV -> camera (R_cb is orthonormal)
    zc = pc[:, 2]
    u = K[0, 0] * pc[:, 0] / np.maximum(zc, 1e-3) + K[0, 2]
    v = K[1, 1] * pc[:, 1] / np.maximum(zc, 1e-3) + K[1, 2]
    h, w = hw
    gx, gy = (u + 0.5) / w * 2 - 1, (v + 0.5) / h * 2 - 1
    valid = (zc > 0.1) & (np.abs(gx) <= 1) & (np.abs(gy) <= 1)
    gx, gy = np.where(valid, gx, -2), np.where(valid, gy, -2)  # outside -> zero padding
    nz = len(Z_SLICES)
    grid = np.stack([gx, gy], -1).reshape(nz * BEV_N, BEV_N, 2).astype(np.float32)
    return grid, valid.reshape(nz, BEV_N, BEV_N)
