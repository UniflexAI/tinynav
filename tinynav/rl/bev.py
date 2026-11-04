"""Robot-centric 2.5D BEV for the RL policy.

Shared by the simulator and the real robot so both feed the policy the exact same tensor.
Layout: [C, BEV_SIZE, BEV_SIZE] float32, robot control center at the image center,
row 0 = farthest ahead (heading points up), column 0 = robot's left.

Channel "height" is elevation above the ground under the robot, and 0 for cells seen with no
hit. Raw ground hits only cover rings (one per sampled pixel row) whose density depends on
speed and floor texture, so flat ground is mapped to 0 with or without a hit; only obstacles
(positive) and drops (negative) stand out. This also removes the camera mounting height.
"""
import numpy as np

from tinynav.core.robot_specs import ROBOT_CONFIG

GRID_SHAPE, GRID_RES, RAY_STEP = (100, 100, 40), 0.1, 10  # keep in sync with PlanningNode
BEV_SIZE = 64
BEV_RES = 0.1  # 6.4m square; its diagonal (9.05m) still fits in the 10m planning grid at any yaw
CHANNELS = ("height", "known")
OCC_THRESHOLD = 0.1  # same as ObstacleConfig.occ_threshold
KNOWN_EPS = 0.02  # |occupancy| below this has decayed back to "never seen"
GROUND_RING = (0.3, 1.5)  # ground hits cover 90%+ of cells this close, so its median is stable
GROUND_MAX = -0.15  # hits at least this far below the camera can be ground
GROUND_PRIOR = -0.45  # go2 camera height; used when too few ground hits are near the robot
ELEV_RANGE = (-0.5, 1.0)


def column_features(occupancy_grid, origin, resolution, ref_z, z_max):
    """Collapse the 3D grid to per-column (height, known).

    height: top of the highest occupied voxel at or below z_max, relative to ref_z (camera
    height), or the grid floor when nothing qualifies. z_max drops ceilings / overhangs the
    robot passes under, otherwise the space under a table would read as a wall.
    known: any voxel observed (hit or free).
    """
    z_top = origin[2] + (np.arange(occupancy_grid.shape[2]) + 1) * resolution - ref_z
    floor = origin[2] - ref_z
    occ = (occupancy_grid > OCC_THRESHOLD) & (z_top - resolution <= z_max)
    height = np.where(occ, z_top, floor).max(axis=2)
    known = (np.abs(occupancy_grid) > KNOWN_EPS).any(axis=2)
    return height.astype(np.float32), known.astype(np.float32), floor


def robot_frame_cells(size=BEV_SIZE, res=BEV_RES):
    """(forward, left) offsets in meters of every BEV cell center, each shape [size, size]."""
    c = (size / 2 - np.arange(size) - 0.5) * res
    return np.meshgrid(c, c, indexing="ij")


def make_bev(occupancy_grid, origin, resolution, center_xy, yaw, ref_z, z_max=ROBOT_CONFIG.obstacle.robot_z_top):
    """Sample the world-aligned grid into the robot frame. Returns [C, BEV_SIZE, BEV_SIZE]."""
    height, known, floor = column_features(occupancy_grid, origin, resolution, ref_z, z_max)
    fwd, left = robot_frame_cells()
    c, s = np.cos(yaw), np.sin(yaw)
    wx = center_xy[0] + c * fwd - s * left
    wy = center_xy[1] + s * fwd + c * left
    ix = np.floor((wx - origin[0]) / resolution).astype(int)
    iy = np.floor((wy - origin[1]) / resolution).astype(int)
    inside = (ix >= 0) & (ix < height.shape[0]) & (iy >= 0) & (iy < height.shape[1])
    ix, iy = np.clip(ix, 0, height.shape[0] - 1), np.clip(iy, 0, height.shape[1] - 1)
    h = np.where(inside, height[ix, iy], floor)
    hit = h > floor
    d = np.hypot(fwd, left)
    g = np.sort(h[hit & (d >= GROUND_RING[0]) & (d <= GROUND_RING[1]) & (h <= GROUND_MAX)])
    ground = g[(len(g) - 1) // 2] if len(g) >= 10 else GROUND_PRIOR  # lower median, same as torch
    elev = np.where(hit, np.clip(h - ground, *ELEV_RANGE), 0.0)
    return np.stack([elev, np.where(inside, known[ix, iy], 0.0)]).astype(np.float32)


def camera_yaw(T):
    """World yaw of the robot heading; the camera optical axis (+z) points forward."""
    return float(np.arctan2(T[1, 2], T[0, 2]))


def to_robot_frame(p_world, center_xy, yaw):
    """World xy point -> (forward, left) in the robot frame."""
    dx, dy = p_world[0] - center_xy[0], p_world[1] - center_xy[1]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float32)
