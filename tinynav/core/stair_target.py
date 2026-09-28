"""Stair mode local target generator.

In a stairwell the only information we need from outside is the direction (up/down).
Everything else comes from local geometry: build a short-memory 2.5D height map from
depth, find ground cells connected to the robot through steps no higher than max_step,
and pick the highest/lowest reachable cell. The geodesic path to it gives a lookahead
target. Short memory keeps the map robust to odometry jumps; no map/relocalization is used.
"""
import heapq
from collections import deque
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.ndimage import binary_dilation, distance_transform_edt


@dataclass
class StairConfig:
    resolution: float = 0.05
    half_size: float = 3.0          # local map spans robot +- half_size [m]
    memory_s: float = 3.0           # keep points seen within the last memory_s seconds
    camera_height: float = 0.66     # camera optical center above the ground it stands on
    band_above_cam: float = 0.3     # ignore points higher than this above camera (ceiling)
    max_depth: float = 4.0
    pixel_stride: int = 4
    max_step: float = 0.22          # max height change between neighbor cells
    max_drop_down: float = 0.45     # descending: the stair edge hides the first treads, so drops look bigger
    wall_span: float = 0.35         # z-span inside one cell above which it is wall/railing
    robot_radius: float = 0.20
    seed_radius: float = 0.6        # ground cells this close to the robot can seed the search
    blind_radius: float = 0.8       # camera cannot see the ground this close; bridge it from the feet
    max_slope: float = 0.75         # steepest staircase rise/run allowed when bridging the blind zone
    seed_height_tol: float = 0.35
    min_level_gain: float = 0.10    # target must be this much higher/lower than current ground
    lookahead: float = 1.2          # lookahead distance along the geodesic path [m]
    clearance_weight: float = 0.5   # extra cost near obstacles, keeps the path centered
    well_drop: float = 0.4          # cells this far beyond ground in the travel direction reveal the stairwell
    well_decay: float = 0.97        # memory of which side the stairwell is on
    turn_target_dist: float = 0.6   # in-place turn target distance when nothing is explored yet


def depth_to_world_points(depth_m, K, T_cam_to_world, cfg: StairConfig):
    s = cfg.pixel_stride
    z = depth_m[::s, ::s]
    v, u = np.nonzero((z > 0.1) & (z < cfg.max_depth))
    Z = z[v, u]
    P = np.stack([(u * s - K[0, 2]) * Z / K[0, 0], (v * s - K[1, 2]) * Z / K[1, 1], Z], axis=1)
    return P @ T_cam_to_world[:3, :3].T + T_cam_to_world[:3, 3]


def build_height_map(points, center_xy, cam_z, cfg: StairConfig):
    """Returns (height, obstacle, observed) grids of shape (n, n); index [ix, iy]."""
    n = int(round(2 * cfg.half_size / cfg.resolution))
    origin = center_xy - cfg.half_size
    keep = points[:, 2] < cam_z + cfg.band_above_cam
    idx = np.floor((points[keep, :2] - origin) / cfg.resolution).astype(np.int64)
    zs = points[keep, 2]
    inside = np.all((idx >= 0) & (idx < n), axis=1)
    flat = idx[inside, 0] * n + idx[inside, 1]
    zs = zs[inside]
    zmax = np.full(n * n, -np.inf)
    zmin = np.full(n * n, np.inf)
    np.maximum.at(zmax, flat, zs)
    np.minimum.at(zmin, flat, zs)
    observed = np.isfinite(zmax).reshape(n, n)
    span = np.where(observed.ravel(), zmax - zmin, 0.0).reshape(n, n)
    obstacle = observed & (span > cfg.wall_span)
    height = np.where(observed, zmax.reshape(n, n), np.nan)
    return height, obstacle, observed, origin


@njit(cache=True)
def _dijkstra(height, free, cost, seeds_x, seeds_y, max_rise, max_drop, resolution):
    n0, n1 = height.shape
    dist = np.full((n0, n1), np.inf)
    parent = np.full((n0, n1, 2), -1, dtype=np.int64)
    heap = [(0.0, np.int64(0), np.int64(0))]
    heap.pop()
    for k in range(len(seeds_x)):
        dist[seeds_x[k], seeds_y[k]] = 0.0
        heapq.heappush(heap, (0.0, seeds_x[k], seeds_y[k]))
    while len(heap) > 0:
        d, x, y = heapq.heappop(heap)
        if d > dist[x, y]:
            continue
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= n0 or ny >= n1 or not free[nx, ny]:
                    continue
                dh = height[nx, ny] - height[x, y]
                if dh > max_rise or -dh > max_drop:
                    continue
                nd = d + resolution * np.sqrt(dx * dx + dy * dy) * cost[nx, ny]
                if nd < dist[nx, ny]:
                    dist[nx, ny] = nd
                    parent[nx, ny, 0] = x
                    parent[nx, ny, 1] = y
                    heapq.heappush(heap, (nd, nx, ny))
    return dist, parent


class StairTargetGenerator:
    def __init__(self, cfg: StairConfig | None = None):
        self.cfg = cfg or StairConfig()
        self.frames = deque()  # (stamp, points_world)
        self.well_side = 0.0   # >0: stairwell on the robot's left, <0: right

    def add_depth(self, stamp, depth_m, K, T_cam_to_world):
        self.frames.append((stamp, depth_to_world_points(depth_m, K, T_cam_to_world, self.cfg)))
        while self.frames and stamp - self.frames[0][0] > self.cfg.memory_s:
            self.frames.popleft()

    def reset(self):
        self.frames.clear()
        self.well_side = 0.0

    def compute(self, T_cam_to_world, direction: str):
        """direction: 'up' or 'down'. Returns dict with status in {'ok', 'no_seed', 'search'}.
        'search' still carries a target (frontier or in-place turn) unless there is no seed."""
        cfg = self.cfg
        assert direction in ('up', 'down')
        cam = T_cam_to_world[:3, 3]
        points = np.concatenate([p for _, p in self.frames]) if self.frames else np.zeros((0, 3))
        height, obstacle, observed, origin = build_height_map(points, cam[:2], cam[2], cfg)
        inflated = binary_dilation(obstacle, iterations=max(1, int(round(cfg.robot_radius / cfg.resolution))))
        free = observed & ~inflated
        clearance = distance_transform_edt(~obstacle) * cfg.resolution
        cost = 1.0 + cfg.clearance_weight / np.maximum(clearance, cfg.resolution)
        out = dict(height=height, obstacle=obstacle, free=free, origin=origin, target=None, goal=None, path=None)

        ground_z = cam[2] - cfg.camera_height
        n = height.shape[0]
        gx, gy = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        cell_xy = origin + (np.stack([gx, gy], -1) + 0.5) * cfg.resolution
        rel_xy = cell_xy - cam[:2]
        # seed from the ground level closest to where the feet should be; the camera rarely sees right below itself
        near = free & (np.linalg.norm(rel_xy, axis=-1) < cfg.seed_radius)
        dz = np.abs(np.nan_to_num(height, nan=1e9) - ground_z)
        at_feet = near & (dz < cfg.seed_height_tol)
        foot_z = height[at_feet][np.argmin(dz[at_feet])] if np.any(at_feet) else ground_z
        # blind zone: ramp from the feet to the nearest seen ground, only if no steeper than a staircase
        r_robot = np.linalg.norm(rel_xy, axis=-1)
        blind = ~observed & ~inflated & (r_robot < cfg.blind_radius) & np.any(free)
        d_obs, (ix, iy) = distance_transform_edt(~free, return_indices=True)
        d_obs = d_obs * cfg.resolution
        h_near = height[ix, iy]
        ramp = foot_z + (h_near - foot_z) * r_robot / np.maximum(r_robot + d_obs, 1e-6)
        run = r_robot + d_obs
        blind &= np.abs(h_near - foot_z) <= cfg.max_step + cfg.max_slope * run
        height = np.where(blind, ramp, height)
        free = free | blind
        out.update(height=height, free=free)
        seeds = free & (r_robot < cfg.seed_radius) & (np.abs(np.nan_to_num(height, nan=1e9) - foot_z) < cfg.max_step)
        if not np.any(seeds):
            out['status'] = 'no_seed'
            return out
        sx, sy = np.nonzero(seeds)
        sign = 1.0 if direction == 'up' else -1.0
        max_drop = cfg.max_drop_down if direction == 'down' else cfg.max_step
        dist, parent = _dijkstra(np.nan_to_num(height), free, cost, sx.astype(np.int64), sy.astype(np.int64), cfg.max_step, max_drop, cfg.resolution)
        reachable = np.isfinite(dist)
        out['reachable'] = reachable
        fwd = T_cam_to_world[:2, :3] @ np.array([0.0, 0.0, 1.0])
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        lateral = fwd[0] * rel_xy[..., 1] - fwd[1] * rel_xy[..., 0]  # >0 on the left

        # the stairwell shows up as observed-but-unreachable cells far beyond ground level in the travel direction
        well = observed & ~reachable & (sign * (np.nan_to_num(height) - foot_z) > cfg.well_drop) & (np.abs(lateral) < 1.5)
        if np.any(well):
            self.well_side = cfg.well_decay * self.well_side + (1 - cfg.well_decay) * np.clip(np.mean(np.sign(lateral[well])), -1, 1) * 10
        out['well_side'] = self.well_side

        h = np.where(reachable, height, np.nan)
        best = np.nanmax(sign * h)
        if best >= sign * foot_z + cfg.min_level_gain:
            # among cells of the extreme level, take the nearest in path cost
            level = reachable & (sign * np.nan_to_num(h, nan=-np.inf) > best - 0.1)
            out['status'] = 'ok'
            goal = np.unravel_index(np.argmin(np.where(level, dist, np.inf)), dist.shape)
        else:
            # landing: nothing better is visible, explore the frontier on the stairwell side
            out['status'] = 'search'
            unobserved_nb = binary_dilation(~(observed | blind), structure=np.ones((3, 3), bool))
            frontier = reachable & unobserved_nb & (np.linalg.norm(rel_xy, axis=-1) > 0.4) & (rel_xy @ fwd > -0.3)
            side = np.sign(self.well_side)
            if side != 0:
                frontier &= side * lateral > 0
            if not np.any(frontier):
                left = np.array([-fwd[1], fwd[0]])
                turn = cam[:2] + cfg.turn_target_dist * (left if side >= 0 else -left)
                out['target'] = np.array([turn[0], turn[1], foot_z])
                return out
            # the next flight is beside the one we came from, across the stairwell
            score = np.where(frontier, side * lateral, -np.inf)
            goal = np.unravel_index(np.argmax(score), score.shape)
        path = [goal]
        while parent[path[-1][0], path[-1][1], 0] >= 0:
            path.append((parent[path[-1][0], path[-1][1], 0], parent[path[-1][0], path[-1][1], 1]))
        path = path[::-1]
        path_xyz = np.array([[*(origin + (np.array(c) + 0.5) * cfg.resolution), height[c]] for c in path])
        seg = np.linalg.norm(np.diff(path_xyz[:, :2], axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        k = min(np.searchsorted(arc, cfg.lookahead), len(path_xyz) - 1)
        out.update(goal=path_xyz[-1], target=path_xyz[k], path=path_xyz)
        return out
