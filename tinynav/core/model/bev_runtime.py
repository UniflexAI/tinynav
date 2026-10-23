"""BevNet at runtime: preprocessing, TensorRT inference and fusion of model obstacles into the planning grid.

The model may only add obstacles. They are trusted only inside the camera's view (outside it the model guesses)
and never under the robot (the robot stands there). Shared by planning_node and the offline replay.
"""
import numpy as np
import cv2
from scipy.ndimage import binary_dilation

from .train.bev_geometry import (BEV_N, BEV_X, BEV_Y, IMG_HW, RES, cam_to_bev_rotation, depth_raster,
                                 heading_rotation, sampling_grid, scale_K)
from .train.cost import PRESETS, obstacle_prob, split_channels

DEPTH_COLS = 136   # camera depth is subsampled to about this width before rasterizing (as in training)


def preprocess(gray, depth_m, K, R_cw):
    """Network inputs from an infra1 image, metric depth aligned with it (any resolution) and camera rotation."""
    K = np.asarray(K, np.float64).reshape(3, 3)
    h, w = gray.shape
    R_cb = cam_to_bev_rotation(R_cw)
    Ki = scale_K(K, IMG_HW[1] / w, IMG_HW[0] / h)
    img = cv2.resize(gray, IMG_HW[::-1], interpolation=cv2.INTER_AREA).astype(np.float32)[None, None] / 255
    Kd = scale_K(K, depth_m.shape[1] / w, depth_m.shape[0] / h)
    s = max(1, int(round(depth_m.shape[1] / DEPTH_COLS)))
    Kd[:2] /= s  # slicing keeps pixel centres at multiples of s
    raster = depth_raster(np.ascontiguousarray(depth_m[::s, ::s], np.float32), Kd, R_cb, step=1)[None]
    grid, valid = sampling_grid(Ki, R_cb, IMG_HW)
    hfov = 2 * np.arctan(w / 2 / K[0, 0])
    return dict(img=img, raster=raster.astype(np.float32), grid=grid[None], valid=valid[None].astype(np.float32)), hfov


def obstacle_bev(logits, preset='quadruped'):
    """(C, N, N) network output -> P(observed and obstacle) for a robot preset, (N, N)."""
    p = {k: 1 / (1 + np.exp(-v.astype(np.float32))) for k, v in split_channels(logits).items() if k != 'ground'}
    return obstacle_prob(p['span'], p['step'], p['rail'], PRESETS[preset])


class BevObstacleFuser:
    """Keeps model obstacle evidence in the planner's world-aligned 2D grid and returns the cells to add."""

    def __init__(self, grid_shape, resolution, thr=0.7, dilate=0, decay=0.98, self_clear=0.35, fov_only=True):
        self.shape, self.res = tuple(grid_shape[:2]), resolution
        self.thr, self.dilate, self.decay, self.self_clear, self.fov_only = thr, dilate, decay, self_clear, fov_only
        self.prob = np.zeros(self.shape, np.float32)

    def cell_xy(self, origin):
        i, j = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        return origin[0] + (i + 0.5) * self.res, origin[1] + (j + 0.5) * self.res

    def shift(self, old_origin, new_origin):
        """Follow the planner when it rolls its grid to a new origin."""
        d = np.round((np.asarray(new_origin[:2]) - np.asarray(old_origin[:2])) / self.res).astype(int)
        if not d.any():
            return
        out = np.zeros_like(self.prob)
        H, W = out.shape
        src = self.prob[max(d[0], 0):H + min(d[0], 0), max(d[1], 0):W + min(d[1], 0)]
        out[max(-d[0], 0):max(-d[0], 0) + src.shape[0], max(-d[1], 0):max(-d[1], 0) + src.shape[1]] = src
        self.prob = out

    def update(self, obs, T, origin, robot_xy, hfov):
        """Write the current prediction (N, N) seen from camera pose T; return the obstacle cells to add."""
        X, Y = self.cell_xy(origin)
        Rh = heading_rotation(T[:3, :3])
        dx, dy = X - T[0, 3], Y - T[1, 3]
        xb = Rh[0, 0] * dx + Rh[0, 1] * dy
        yb = Rh[1, 0] * dx + Rh[1, 1] * dy
        r = np.floor((BEV_X[1] - xb) / RES).astype(int)
        q = np.floor((BEV_Y[1] - yb) / RES).astype(int)
        inside = (r >= 0) & (r < BEV_N) & (q >= 0) & (q < BEV_N)
        if self.fov_only:
            inside &= (xb > 0.3) & (np.abs(np.arctan2(yb, xb)) < hfov / 2)
        self.prob *= self.decay
        self.prob[inside] = obs[r[inside], q[inside]]
        extra = self.prob >= self.thr
        if self.dilate > 0 and extra.any():
            extra = binary_dilation(extra, iterations=self.dilate)
        if self.self_clear > 0:
            extra &= (X - robot_xy[0]) ** 2 + (Y - robot_xy[1]) ** 2 > self.self_clear ** 2
        return extra


def load_trt(engine_path):
    """TensorRT engine built from export_onnx.py output (inputs img, raster, grid, valid; output out)."""
    from tinynav.core.models_trt import TRTBase

    class BevNetTRT(TRTBase):
        async def infer(self, inputs):
            for inp in self.inputs:
                np.copyto(inp['host'], inputs[inp['name']].astype(inp['host'].dtype, copy=False))
            return (await self.run_graph())['out'][0]

    return BevNetTRT(engine_path)
