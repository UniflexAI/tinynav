"""Stair mode local target generator.

In a stairwell the only information we need from outside is the direction (up/down).
Everything else comes from local geometry: build a short-memory 2.5D height map from
depth, find ground cells connected to the robot through steps no higher than max_step,
and pick the highest/lowest reachable cell. The geodesic path to it gives a lookahead
target. Short memory keeps the map robust to odometry jumps; no map/relocalization is used.
Odometry itself is checked on its raw high-rate stream: a jump between consecutive poses means
the VIO is failing, so the map is dropped and no target is given until it has been quiet for a while.
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
    camera_height: float = 0.66     # camera optical center above the ground it stands on (initial value) ...
    auto_camera_height: bool = True # ... re-measured on flat ground from the floor in the depth image: a wrong value
                                    # puts the feet on the step below and ends a flight early
    band_above_cam: float = 0.3     # ignore points higher than this above camera (ceiling)
    max_depth: float = 4.0
    pixel_stride: int = 4
    max_step: float = 0.22          # max height change between neighbor cells
    max_drop_down: float = 0.45     # descending: the stair edge hides the first treads, so drops look bigger
    wall_span: float = 0.35         # z-span inside one cell above which it is wall/railing ...
    wall_min_top: float = 0.3       # ... and only if it also rises this far above the feet: going down, step edges
                                    # collect points of several treads (depth edge pixels, many viewpoints) and
                                    # would otherwise read as walls across the flight
    robot_radius: float = 0.20
    seed_radius: float = 0.6        # ground cells this close to the robot can seed the search
    blind_radius: float = 0.8       # camera cannot see the ground this close; bridge it from the feet
    max_slope: float = 0.75         # steepest staircase rise/run allowed when bridging the blind zone
    seed_height_tol: float = 0.35
    min_level_gain: float = 0.20    # target must be more than a step higher/lower than the feet (landings have small bumps)
    min_level_gain_exit: float = 0.10  # hysteresis: once on a flight, stay 'ok' down to this gain
    ok_hold_s: float = 0.75         # the next level flickers out of view (occlusion, blind zone): keep the last
                                    # 'ok' goal this long before switching to landing search
    lookahead: float = 1.2          # lookahead distance along the geodesic path [m]
    goal_push_down: float = 0.8     # going down, aim this much farther (path cost) into the next level, at its deepest
                                    # cell keeping search_clearance from walls, instead of its near edge. Going up it
                                    # made landings worse on the upstairs bag, so it is down only.
    target_open_radius: float = 0.2 # target: move it up to this far from the path point, to the cell farthest from
                                    # walls (clearance counted up to target_open_cap), off walls and railings
    target_open_cap: float = 0.6
    clearance_weight: float = 0.5   # extra cost near obstacles, keeps the path centered
    side_view_min: float = 0.7      # cells seen this far to the side of a flight ...
    side_view_max: float = 2.5      # ... up to here tell which side is open (railing) and which is a wall
    well_decay: float = 0.97        # memory of which side the stairwell is on
    turn_target_dist: float = 0.6   # last-resort in-place turn target distance
    search_clearance: float = 0.35  # landing targets keep this far from walls (planning backs off near walls)
    search_commit_s: float = 4.0    # keep a landing target this long instead of re-picking every call
    search_reached: float = 0.4     # a landing target this close counts as reached
    search_unknown_margin: float = 0.2  # landing targets stay this far from unseen cells: turn to look, don't walk in
    turn_in_place_dist: float = 0.3     # with no room to the side, a target this close only turns the robot
    flight_pitch: float = 12.0      # body pitched more than this (deg, smoothed): we are on a flight
    level_pitch: float = 8.0        # back below this for level_min_s after >= flight_min_s on a flight: on the landing
    flight_min_s: float = 1.0
    level_min_s: float = 0.5
    landing_search: bool = False    # at the landing: False stops there ('landing'), True looks for the next flight
    max_target_bearing: float = 75.0    # never put the target further round than this from the camera heading;
                                        # a U-turn is done by turning toward the side instead of aiming behind
    guide_min_dist: float = 0.8     # with a remembered direction (stair_memory.py), aim at a reachable cell
    guide_max_dist: float = 2.0     # this far away ...
    guide_max_angle: float = 60.0   # ... within this many degrees of it, else ignore the memory
    jump_dist: float = 0.10         # consecutive odometry poses further apart than this are a jump ...
    jump_speed: float = 3.0         # ... if that is also faster than this (a gap in the stream is not a jump)
    flight_track_m: float = 1.0     # flight direction = horizontal displacement over this much recent travel
    flight_min_dz: float = 0.15     # ... counted only if the height changed this much over it (on a flight)
    jump_hold_s: float = 2.0        # odometry stays invalid this long after the last jump


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
        self.well_side = 0.0   # >0: next flight (railing side) on the left of the last flight, <0: right
        self.flight_dir = None  # horizontal travel direction on the last flight
        self.turn_side = 0      # U-turn side at landings if known: +1 left, -1 right, 0 estimate from the stairwell
        self.track = deque()    # recent odometry positions, for the flight direction
        self.z_log = deque()    # (stamp, z) of recent poses, to tell flat ground for the camera height
        self.last_pose_stamp = None
        self.pitch = 0.0        # smoothed camera pitch (deg, + up)
        self.pitched_since = self.level_since = None
        self.flight_done = False  # walked a flight and are level again
        self.was_on_flight = False
        self.camera_height = self.cfg.camera_height
        self.last_status = None
        self.search_goal = None  # (xy, stamp picked) of the current landing target
        self.ok_goal = None      # (xy, stamp last seen) of the last 'ok' goal
        self.last_position = None
        self.last_jump_stamp = -np.inf
        self.latest_stamp = -np.inf

    def add_pose(self, stamp, position):
        """Feed every raw odometry pose (e.g. 100 Hz), not only the ones matched to depth."""
        position = np.asarray(position, dtype=np.float64)
        step = np.linalg.norm(position - self.last_position) if self.last_position is not None else 0.0
        dt = stamp - self.last_pose_stamp if self.last_pose_stamp is not None else 0.0
        if step > self.cfg.jump_dist and step > self.cfg.jump_speed * max(dt, 1e-3):
            self.last_jump_stamp = stamp
            self.frames.clear()
            self.track.clear()
        self.last_position = position
        self.last_pose_stamp = stamp
        self.latest_stamp = max(self.latest_stamp, stamp)
        self.z_log.append((stamp, position[2]))
        while self.z_log and stamp - self.z_log[0][0] > 2.0:
            self.z_log.popleft()
        if not self.track or np.linalg.norm(position[:2] - self.track[-1][:2]) > 0.02:
            self.track.append(position)
            while len(self.track) > 200:
                self.track.popleft()
            self._update_flight_dir()

    def _update_flight_dir(self):
        now, arc = self.track[-1], 0.0
        for i in range(len(self.track) - 2, -1, -1):
            arc += np.linalg.norm(self.track[i + 1][:2] - self.track[i][:2])
            if arc >= self.cfg.flight_track_m:
                d = now[:2] - self.track[i][:2]
                if abs(now[2] - self.track[i][2]) > self.cfg.flight_min_dz and np.linalg.norm(d) > 0.5 * arc:
                    self.flight_dir = d / np.linalg.norm(d)
                return

    @property
    def odom_valid(self):
        return self.latest_stamp - self.last_jump_stamp >= self.cfg.jump_hold_s

    def add_depth(self, stamp, depth_m, K, T_cam_to_world):
        self.latest_stamp = max(self.latest_stamp, stamp)
        if not self.odom_valid:
            return
        self._update_pitch(stamp, T_cam_to_world)
        points = depth_to_world_points(depth_m, K, T_cam_to_world, self.cfg)
        self.frames.append((stamp, points))
        while self.frames and stamp - self.frames[0][0] > self.cfg.memory_s:
            self.frames.popleft()
        if self.cfg.auto_camera_height:
            self._measure_camera_height(points, T_cam_to_world[:3, 3])

    def _update_pitch(self, stamp, T_cam_to_world):
        """The body pitches ~20 deg on a flight and is level on a landing; the camera is fixed to it."""
        cfg = self.cfg
        pitch = np.degrees(np.arcsin(np.clip(T_cam_to_world[2, 2], -1.0, 1.0)))  # camera forward, z component
        self.pitch = 0.7 * self.pitch + 0.3 * pitch  # ~0.5 s at the depth rate, the body rocks on every step
        if abs(self.pitch) > cfg.flight_pitch:
            self.pitched_since = self.pitched_since if self.pitched_since is not None else stamp
            self.was_on_flight |= stamp - self.pitched_since >= cfg.flight_min_s
        else:
            self.pitched_since = None
        if abs(self.pitch) < cfg.level_pitch:
            self.level_since = self.level_since if self.level_since is not None else stamp
            self.flight_done |= self.was_on_flight and stamp - self.level_since >= cfg.level_min_s
        else:
            self.level_since = None

    @property
    def on_flight(self):
        return abs(self.pitch) > self.cfg.flight_pitch

    def _measure_camera_height(self, points, cam):
        """On flat ground (height steady for 2 s) the floor ahead is the strongest level below the camera."""
        zs = [z for _, z in self.z_log]
        if len(zs) < 50 or self.z_log[-1][0] - self.z_log[0][0] < 1.5 or max(zs) - min(zs) > 0.05:
            return
        rel = points[:, 2] - cam[2]
        r = np.linalg.norm(points[:, :2] - cam[:2], axis=1)
        below = rel[(rel < -0.15) & (rel > -1.2) & (r > 0.3) & (r < 1.5)]
        if len(below) < 300:
            return
        hist, edges = np.histogram(below, bins=np.arange(-1.2, -0.14, 0.01))
        if hist.max() < 0.2 * len(below):  # no clear single floor (stairs, clutter)
            return
        self.camera_height = 0.8 * self.camera_height + 0.2 * -(edges[np.argmax(hist)] + 0.005)

    def new_run(self):
        """Forget per-run state (last flight, stairwell side, landing target) but keep the height map."""
        self.well_side = 0.0
        self.flight_dir = None
        self.last_status = None
        self.search_goal = None
        self.ok_goal = None
        self.pitched_since = self.level_since = None
        self.flight_done = self.was_on_flight = False

    def reset(self):
        self.frames.clear()
        self.track.clear()
        self.z_log.clear()
        self.new_run()
        self.last_position = None
        self.last_jump_stamp = -np.inf
        self.latest_stamp = -np.inf

    def compute(self, T_cam_to_world, direction: str, prior_dir=None):
        """direction: 'up' or 'down'. Returns dict with status in {'ok', 'no_seed', 'search', 'odom_invalid', 'landing'}.
        'search' still carries a target (frontier or in-place turn); 'no_seed', 'odom_invalid' and 'landing'
        (walked a flight and level again, with landing_search off) mean stop.
        prior_dir: optional world-frame horizontal direction remembered for this place (stair_memory.py); it
        only picks among reachable, wall-clear cells and is ignored when none lies near it."""
        cfg = self.cfg
        assert direction in ('up', 'down')
        cam = T_cam_to_world[:3, 3]
        points = np.concatenate([p for _, p in self.frames]) if self.frames else np.zeros((0, 3))
        height, obstacle, observed, origin = build_height_map(points, cam[:2], cam[2], cfg)
        if not self.odom_valid:
            self.last_status, self.search_goal, self.ok_goal = 'odom_invalid', None, None
            return dict(status='odom_invalid', height=height, obstacle=obstacle, free=observed & ~obstacle, origin=origin, target=None, goal=None, path=None)
        obstacle = obstacle & (np.nan_to_num(height, nan=-np.inf) > cam[2] - self.camera_height + cfg.wall_min_top)
        inflated = binary_dilation(obstacle, iterations=max(1, int(round(cfg.robot_radius / cfg.resolution))))
        free = observed & ~inflated
        clearance = distance_transform_edt(~obstacle) * cfg.resolution
        cost = 1.0 + cfg.clearance_weight / np.maximum(clearance, cfg.resolution)
        out = dict(height=height, obstacle=obstacle, free=free, origin=origin, target=None, goal=None, path=None)

        ground_z = cam[2] - self.camera_height
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
            self.last_status, self.search_goal, self.ok_goal = 'no_seed', None, None
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
        heading = fwd

        def lateral_to(d):  # >0 on the left of direction d
            return d[0] * rel_xy[..., 1] - d[1] * rel_xy[..., 0]

        if self.flight_done and not cfg.landing_search:
            # walked a flight and are level again: stop on the landing (turning for the next flight is not done)
            self.last_status, self.search_goal, self.ok_goal = 'landing', None, None
            out['status'] = 'landing'
            return out
        h = np.where(reachable, height, np.nan)
        best = np.nanmax(sign * h)
        # hysteresis so the end of a flight does not flip between 'ok' and 'search' every call
        gain = cfg.min_level_gain_exit if self.last_status == 'ok' else cfg.min_level_gain
        held = None
        # still pitched = still on the flight: keep going for the last goal however long the next level is out of view
        hold_s = np.inf if self.on_flight else cfg.ok_hold_s
        if best < sign * foot_z + gain and self.ok_goal is not None and self.latest_stamp - self.ok_goal[1] < hold_s:
            gi = tuple(np.floor((self.ok_goal[0] - origin) / cfg.resolution).astype(int))
            if 0 <= gi[0] < n and 0 <= gi[1] < n and reachable[gi] and np.linalg.norm(self.ok_goal[0] - cam[:2]) > cfg.search_reached:
                held = gi
        if held is not None:
            out['status'] = 'ok'
            goal = held
        elif best >= sign * foot_z + gain:
            out['status'] = 'ok'
            self.search_goal = None
            # on a flight: the next flight is behind the railing, not behind the wall. A wall hides what is beyond
            # it while a railing lets us see the parallel flight (above or below), so the open side is the one
            # with more cells seen far to the side. This works both up and down.
            ref = self.flight_dir if self.flight_dir is not None else fwd
            lateral, along = lateral_to(ref), rel_xy @ ref
            beside = observed & (np.abs(along) < 1.5)
            seen_l = np.sum(beside & (lateral > cfg.side_view_min) & (lateral < cfg.side_view_max))
            seen_r = np.sum(beside & (lateral < -cfg.side_view_min) & (lateral > -cfg.side_view_max))
            if seen_l + seen_r > 0:
                self.well_side = cfg.well_decay * self.well_side + (1 - cfg.well_decay) * (seen_l - seen_r) / (seen_l + seen_r) * 10
            # among cells of the extreme level, take the nearest in path cost
            level = reachable & (sign * np.nan_to_num(h, nan=-np.inf) > best - 0.1)
            level_dist = np.where(level, dist, np.inf)
            push = cfg.goal_push_down if direction == 'down' else 0.0
            deep = (level_dist <= level_dist.min() + push) & (clearance >= cfg.search_clearance)
            if push > 0 and np.any(deep):
                goal = np.unravel_index(np.argmax(np.where(deep, level_dist, -np.inf)), dist.shape)
            else:
                goal = np.unravel_index(np.argmin(level_dist), dist.shape)
            self.ok_goal = (cell_xy[goal].copy(), self.latest_stamp)
        else:
            # landing: nothing better is visible, explore the frontier on the stairwell side. Sides are taken
            # w.r.t. the last flight, not the camera, which keeps turning on the landing. Targets keep clear of
            # walls, and a chosen one is kept for a while so the target does not jump around.
            out['status'] = 'search'
            self.ok_goal = None
            fwd = self.flight_dir if self.flight_dir is not None else fwd
            lateral = lateral_to(fwd)
            side = self.turn_side or np.sign(self.well_side)
            roomy = reachable & (clearance >= cfg.search_clearance)
            goal = None
            if self.search_goal is not None and self.latest_stamp - self.search_goal[1] < cfg.search_commit_s \
                    and np.linalg.norm(self.search_goal[0] - cam[:2]) > cfg.search_reached:
                gi = tuple(np.floor((self.search_goal[0] - origin) / cfg.resolution).astype(int))
                if 0 <= gi[0] < n and 0 <= gi[1] < n and reachable[gi]:
                    goal = gi
            if goal is None:
                # only seen, roomy cells away from anything unseen: we turn to look instead of walking into the
                # unknown, which is where railings and the stairwell edge hide
                near_unknown = binary_dilation(~(observed | blind), iterations=max(1, int(round(cfg.search_unknown_margin / cfg.resolution))))
                seen = roomy & observed & ~near_unknown & (r_robot > 0.4) & (r_robot < 1.5) & (rel_xy @ fwd > -0.3)
                if side != 0:
                    seen &= side * lateral > 0
                if np.any(seen):
                    # the next flight is beside the one we came from, across the stairwell
                    goal = np.unravel_index(np.argmax(np.where(seen, side * lateral, -np.inf)), seen.shape)
                else:
                    # nothing to explore in view: turn toward the stairwell side, aiming at the roomiest reachable
                    # spot there so the target never sits in a wall
                    near_cells = reachable & (r_robot > 0.3) & (r_robot < 1.2)
                    turn_cells = near_cells & (side * lateral > 0) if side != 0 else near_cells
                    cand = turn_cells if np.any(turn_cells) else near_cells
                    if not np.any(cand):
                        left = np.array([-fwd[1], fwd[0]])
                        turn = cam[:2] + cfg.turn_target_dist * (left if side >= 0 else -left)
                        self.last_status = 'search'
                        target = self._keep_in_front(np.array([turn[0], turn[1], foot_z]), cam, heading, reachable,
                                                     observed, clearance, cell_xy, height, rel_xy, r_robot, out)
                        out.update(target=target, well_side=self.well_side)
                        return out
                    goal = np.unravel_index(np.argmax(np.where(cand, clearance, -np.inf)), cand.shape)
                self.search_goal = (cell_xy[goal].copy(), self.latest_stamp)
        self.last_status = out['status']
        if prior_dir is not None:
            d = np.asarray(prior_dir, dtype=np.float64)
            d = d / (np.linalg.norm(d) + 1e-9)
            cos = (rel_xy @ d) / np.maximum(r_robot, 1e-6)
            cand = reachable & (clearance >= cfg.search_clearance) & (r_robot > cfg.guide_min_dist) \
                & (r_robot < cfg.guide_max_dist) & (cos > np.cos(np.radians(cfg.guide_max_angle)))
            if np.any(cand):
                # closest to the remembered direction, a little farther when tied
                goal = np.unravel_index(np.argmax(np.where(cand, cos + 0.05 * r_robot, -np.inf)), cand.shape)
                out['guided'] = True
        out['well_side'] = self.well_side
        path = [goal]
        while parent[path[-1][0], path[-1][1], 0] >= 0:
            path.append((parent[path[-1][0], path[-1][1], 0], parent[path[-1][0], path[-1][1], 1]))
        path = path[::-1]
        path_xyz = np.array([[*(origin + (np.array(c) + 0.5) * cfg.resolution), height[c]] for c in path])
        seg = np.linalg.norm(np.diff(path_xyz[:, :2], axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        k = min(np.searchsorted(arc, cfg.lookahead), len(path_xyz) - 1)
        target = path_xyz[k]
        if cfg.target_open_radius > 0:
            # nudge the target into open space so planning is not pulled along a wall or railing
            near = reachable & (np.linalg.norm(cell_xy - target[:2], axis=-1) <= cfg.target_open_radius)
            if np.any(near):
                c = np.unravel_index(np.argmax(np.where(near, np.minimum(clearance, cfg.target_open_cap), -np.inf)), near.shape)
                target = np.array([*cell_xy[c], height[c]])
        target = self._keep_in_front(target, cam, heading, reachable, observed, clearance, cell_xy, height, rel_xy, r_robot, out)
        out.update(goal=path_xyz[-1], target=target, path=path_xyz)
        return out

    def _keep_in_front(self, target, cam, heading, reachable, observed, clearance, cell_xy, height, rel_xy, r_robot, out):
        """A target behind the robot makes planning back up or swing round in a stairwell: keep it within
        max_target_bearing of the camera heading, on the same side, at a reachable open cell."""
        cfg = self.cfg
        d = target[:2] - cam[:2]
        bearing = np.degrees(np.arctan2(heading[0] * d[1] - heading[1] * d[0], heading @ d))
        if abs(bearing) <= cfg.max_target_bearing:
            return target
        side = np.sign(bearing)
        cell_bearing = np.degrees(np.arctan2(heading[0] * rel_xy[..., 1] - heading[1] * rel_xy[..., 0], rel_xy @ heading))
        roomy = reachable & observed & (r_robot > 0.3) & (r_robot < 1.2) & (side * cell_bearing > 0) \
            & (np.abs(cell_bearing) <= cfg.max_target_bearing) & (clearance >= cfg.search_clearance)
        out['turned'] = True
        if np.any(roomy):
            # as far round as allowed, i.e. turn toward where the target was, but off walls
            c = np.unravel_index(np.argmax(np.where(roomy, side * cell_bearing, -np.inf)), roomy.shape)
            return np.array([*cell_xy[c], height[c]])
        # no room on that side (a wall right there): a target just beside us makes planning turn in place
        # rather than walk anywhere; after turning there is something new to see
        a = np.radians(side * cfg.max_target_bearing)
        v = np.array([heading[0] * np.cos(a) - heading[1] * np.sin(a), heading[0] * np.sin(a) + heading[1] * np.cos(a)])
        out['turn_in_place'] = True
        return np.array([*(cam[:2] + cfg.turn_in_place_dist * v), target[2]])
