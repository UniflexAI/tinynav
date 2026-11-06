"""Batched GPU navigation sim that reproduces the real perception -> BEV pipeline.

World: flat ground (z=0) + axis-aligned boxes. Each step renders the depth pixels the planner
samples (every RAY_STEP-th pixel) analytically, fuses them with a torch port of
run_raycasting_loopy + update_occupancy_grid, and builds the BEV like bev.make_bev. So the
policy sees the same ring-shaped ground hits, free-space carving and decay as on the robot.

The rolling grid is stored toroidally (global voxel index mod GRID_SHAPE) so recentering is a
strip clear instead of a per-env roll.
"""
import math
from dataclasses import dataclass

import torch

from tinynav.core.robot_specs import RL_FOOTPRINT, ROBOT_CONFIG
from tinynav.rl.bev import (BEV_RES, BEV_SIZE, ELEV_RANGE, GRID_RES, GRID_SHAPE, GROUND_MAX, GROUND_PRIOR, GROUND_RING,
                            KNOWN_EPS, OCC_THRESHOLD, RAY_STEP, TRAIL_LEN, robot_frame_cells)
from tinynav.rl.rl_planner import ESCAPE_TOL, GUARD_MARGIN, V_MAX, V_MIN, W_MAX

GX, GY, GZ = GRID_SHAPE
MAX_RAY_SAMPLES = 56  # past this many voxel steps a ray from the grid center is outside the grid
FAR_DEPTH = 65.5  # what the stereo depth reports for sky / out-of-range pixels
N_TRAP = 3  # extra box slots for the wall / U-cup placed between start and goal
N_NARROW = 4  # wall with a doorway across start->goal, optionally extended into a corridor
NOVELTY_CELL = 0.5
GEO_RES = 0.2  # geodesic distance field, training-only critic input
GEO_INFLATE = RL_FOOTPRINT.half_width  # 0.8m doorways keep a free band wider than the 0.2m geodesic grid
GEO_UNREACHABLE = 30.0
PRIV_DIM = 9  # geodesic distance here + its slope toward 8 robot-frame directions


@dataclass
class SimConfig:
    num_envs: int = 256
    dt: float = 0.15  # planner runs at the depth rate, ~7Hz on the debug bags
    max_steps: int = 400  # room to back out of a dead end
    env_chunk: int = 32  # envs per raycast chunk, bounds GPU memory
    # world
    arena: float = 16.0
    n_boxes: int = 24
    box_size: tuple = (0.3, 2.0)
    box_height: tuple = (0.2, 1.5)
    wall_prob: float = 0.3
    density: tuple = (0.3, 1.0)  # per-env fraction of the n_boxes that are present
    # camera (RealSense infra1 on the debug bags)
    img_w: int = 544
    img_h: int = 640
    fx: float = 306.82
    fy: float = 306.82
    cx: float = 272.28
    cy: float = 312.66
    cam_height: tuple = (0.38, 0.48)
    cam_pitch_deg: tuple = (0.0, 5.0)
    cam_forward: tuple = (0.1, 0.5)  # camera ahead of the control center; varies across robots
    rear_prob: float = 0.5  # episodes that also have a rear camera, so one policy serves 1- and 2-camera robots
    rear_back: tuple = (0.1, 0.5)  # rear camera behind the control center, looking backwards
    pitch_jitter_deg: float = 0.7  # per-frame body sway of a walking robot
    height_jitter: float = 0.01
    depth_drop: float = 0.2  # fraction of invalid stereo pixels
    disparity_noise_px: float = 0.1
    baseline: float = 0.05
    # robot
    v_min: float = V_MIN
    v_max: float = V_MAX
    w_max: float = W_MAX
    lin_acc: float = 1.0
    ang_acc: float = 2.0
    latency_steps: int = 1
    act_noise: float = 0.1
    # task
    goal_dist: tuple = (2.0, 8.0)
    goal_tol: float = 0.3
    clearance: float = 0.3
    trap_prob: float = 0.3  # episodes with a wall (or U-cup) between start and goal
    trap_half: tuple = (1.5, 3.5)  # wall half-length range, grown by the curriculum
    cup_prob: float = 0.5  # fraction of traps closed into a U around the start
    narrow_prob: float = 0.35  # non-trap episodes whose only way through is a doorway / corridor
    door_width: tuple = (0.8, 1.2)  # 0.15-0.35m each side of the 0.5m wide RL footprint
    corridor_prob: float = 0.5
    novelty_reward: float = 0.05  # per newly visited NOVELTY_CELL, keeps exploring when stuck
    clearance_margin: float = 0.25  # on top of the footprint's own margin; penalty ramps in below this
    clearance_penalty: float = 0.15  # per step at zero clearance; the guard penalty below does the real work
    turn_penalty: float = 0.1  # per step at full yaw rate (scaled by (w / w_max)^2); stops weaving
    guard_margin: float = GUARD_MARGIN  # the runtime guard refuses commands whose 1s rollout comes closer
    # Per step at contact, only for commands the runtime guard would refuse. Off: both v8 runs that used
    # it (0.2) ended below v7, while v7 at the real 1 m/s already rarely triggers the guard.
    guard_penalty: float = 0.0
    time_penalty: float = 0.01  # per step; standing still must not be free
    reverse_penalty: float = 0.05  # per step at full reverse without a rear camera (x0.3 with one)
    efficiency_bonus: float = 5.0  # extra on arrival, scaled by geodesic start distance / path length


def _ray_voxels(start, end, valid, o3):
    """Storage indices + masks of every voxel run_raycasting_loopy touches. Compiled: pure elementwise."""
    shape = torch.tensor(GRID_SHAPE, device=start.device)
    diff = end - start[:, None]
    steps = diff.abs().amax(-1)
    valid = valid & (steps > 0)
    i = torch.arange(MAX_RAY_SAMPLES + 1, device=start.device, dtype=torch.float32)
    t = i / steps.clamp(min=1)[..., None].float()
    interp = torch.round(start[:, None, None].float() + t[..., None] * diff[:, :, None].float()).int()
    inb = ((interp >= 0) & (interp < shape)).all(-1) & (i <= steps[..., None]) & valid[..., None]
    e_inb = ((end >= 0) & (end < shape)).all(-1) & valid

    def flat(v, off):
        g = v + off
        return ((g[..., 0] % GX) * GY + (g[..., 1] % GY)) * GZ + g[..., 2]
    fi = torch.where(inb, flat(interp, o3[:, None, None]), 0)
    fe = torch.where(e_inb, flat(end, o3[:, None]), 0)
    return fi.flatten(1).long(), inb.flatten(1), fe.long(), e_inb


_ray_voxels_compiled = torch.compile(_ray_voxels, dynamic=True)


def _rot(yaw, pitch):
    """Camera-to-world rotation, columns = (right, down, forward) for heading yaw, pitched down."""
    cy, sy, cp, sp = torch.cos(yaw), torch.sin(yaw), torch.cos(pitch), torch.sin(pitch)
    f = torch.stack([cp * cy, cp * sy, -sp], -1)
    r = torch.stack([sy, -cy, torch.zeros_like(yaw)], -1)
    d = torch.cross(f, r, dim=-1)
    return torch.stack([r, d, f], -1)


class NavSim:
    def __init__(self, cfg: SimConfig = SimConfig(), device="cuda", seed=0):
        self.cfg, self.device, n = cfg, torch.device(device), cfg.num_envs
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        dev = self.device
        v, u = torch.meshgrid(torch.arange(0, cfg.img_h, RAY_STEP), torch.arange(0, cfg.img_w, RAY_STEP), indexing="ij")
        self.pix_dir = torch.stack([(u - cfg.cx) / cfg.fx, (v - cfg.cy) / cfg.fy, torch.ones_like(u, dtype=torch.float)], -1).reshape(-1, 3).float().to(dev)
        fwd, left = robot_frame_cells()
        self.bev_fwd, self.bev_left = torch.tensor(fwd, device=dev).float(), torch.tensor(left, device=dev).float()
        d = self.bev_left.hypot(self.bev_fwd)
        self.ground_ring = (d >= GROUND_RING[0]) & (d <= GROUND_RING[1])
        self.fp = (RL_FOOTPRINT.front, RL_FOOTPRINT.rear, RL_FOOTPRINT.half_width)
        self.cam_off = torch.zeros(n, device=dev)  # forward offset control -> camera, per episode
        self.rear_on = torch.zeros(n, dtype=torch.bool, device=dev)
        self.rear_off = torch.zeros(n, device=dev)

        self.grid = torch.zeros((n, GX, GY, GZ), device=dev)
        self.base = torch.zeros((n, 3), device=dev)  # world position of global voxel (0,0,0)
        self.o = torch.zeros((n, 2), dtype=torch.long, device=dev)  # grid window origin, global voxels
        self.n_slots = cfg.n_boxes + 4 + N_TRAP + N_NARROW  # random boxes, border, trap, doorway
        self.boxes = torch.zeros((n, self.n_slots, 5), device=dev)  # cx, cy, hx, hy, height
        self.pos = torch.zeros((n, 2), device=dev)
        self.yaw = torch.zeros(n, device=dev)
        self.vel = torch.zeros((n, 2), device=dev)
        self.cmd_queue = torch.zeros((n, cfg.latency_steps + 1, 2), device=dev)
        self.last_action = torch.zeros((n, 2), device=dev)
        self.goal = torch.zeros((n, 2), device=dev)
        self.cam_h = torch.zeros(n, device=dev)
        self.pitch = torch.zeros(n, device=dev)
        self.steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.best_dist = torch.zeros(n, device=dev)
        # per-episode accumulators: path length, shortest-path length at start, min clearance, sum |w|
        self.path_len, self.geo_start = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
        self.min_clear, self.abs_w = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
        fl, rl, hw = self.fp
        f = torch.arange(-rl, fl + 1e-6, 0.05, device=dev)
        sd = torch.arange(-hw, hw + 1e-6, 0.05, device=dev)
        self.fp_edge = torch.cat([torch.stack([f, torch.full_like(f, hw)], 1), torch.stack([f, torch.full_like(f, -hw)], 1),
                                  torch.stack([torch.full_like(sd, fl), sd], 1), torch.stack([torch.full_like(sd, -rl), sd], 1)])
        self.trap = torch.zeros(n, dtype=torch.bool, device=dev)
        self.narrow = torch.zeros(n, dtype=torch.bool, device=dev)
        self.hist = torch.full((n, TRAIL_LEN, 2), math.nan, device=dev)  # own path, index = age in ticks
        self.nv = int(math.ceil((cfg.arena + 2) / NOVELTY_CELL))
        self.visited = torch.zeros((n, self.nv, self.nv), dtype=torch.bool, device=dev)
        self.ng = int(math.ceil((cfg.arena + 2.4) / GEO_RES))
        self.geo = torch.zeros((n, self.ng, self.ng), device=dev)
        g = (torch.arange(self.ng, device=dev) + 0.5) * GEO_RES - self.ng * GEO_RES / 2
        self.geo_x, self.geo_y = torch.meshgrid(g, g, indexing="ij")
        a = torch.arange(8, device=dev) * math.pi / 4
        self.priv_dirs = torch.stack([torch.cos(a), torch.sin(a)], -1)  # robot frame

    # ---------- randomness ----------
    def _rand(self, *shape, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(shape, generator=self.gen, device=self.device)

    # ---------- world ----------
    def _make_boxes(self, idx):
        c, k = self.cfg, len(idx)
        b = torch.zeros((k, self.n_slots, 5), device=self.device)
        half = c.arena / 2
        b[:, :c.n_boxes, 0:2] = self._rand(k, c.n_boxes, 2, lo=-half, hi=half)
        b[:, :c.n_boxes, 2:4] = self._rand(k, c.n_boxes, 2, lo=c.box_size[0], hi=c.box_size[1]) / 2
        wall = self._rand(k, c.n_boxes) < c.wall_prob  # long thin boxes
        along = self._rand(k, c.n_boxes) < 0.5
        b[:, :c.n_boxes, 2] = torch.where(wall & along, self._rand(k, c.n_boxes, lo=1.0, hi=2.5), torch.where(wall, 0.08, b[:, :c.n_boxes, 2]))
        b[:, :c.n_boxes, 3] = torch.where(wall & ~along, self._rand(k, c.n_boxes, lo=1.0, hi=2.5), torch.where(wall, 0.08, b[:, :c.n_boxes, 3]))
        b[:, :c.n_boxes, 4] = self._rand(k, c.n_boxes, lo=c.box_height[0], hi=c.box_height[1])
        b[:, :c.n_boxes, 4] *= self._rand(k, c.n_boxes) < self._rand(k, 1, lo=c.density[0], hi=c.density[1])  # height 0 = absent
        border = torch.tensor([[half + 0.1, 0, 0.1, half + 0.2], [-half - 0.1, 0, 0.1, half + 0.2],
                               [0, half + 0.1, half + 0.2, 0.1], [0, -half - 0.1, half + 0.2, 0.1]], device=self.device)
        b[:, c.n_boxes:c.n_boxes + 4, :4] = border
        b[:, c.n_boxes:c.n_boxes + 4, 4] = 1.0
        self.boxes[idx] = b

    def _footprint_hits(self, boxes, pos, yaw, margin=0.0):
        """SAT test of the robot footprint (inflated by margin) against every box. Returns [k, B]."""
        fl, rl, hw = self.fp
        hl, hw = (fl + rl) / 2 + margin, hw + margin
        cy, sy = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        c = pos[:, None] + ((fl - rl) / 2) * torch.stack([cy, sy], -1)
        d = c - boxes[..., :2]
        bx, by, present = boxes[..., 2], boxes[..., 3], boxes[..., 4] > 0
        ax_x = (d[..., 0].abs() <= bx + hl * cy.abs() + hw * sy.abs())
        ax_y = (d[..., 1].abs() <= by + hl * sy.abs() + hw * cy.abs())
        ax_f = ((d[..., 0] * cy + d[..., 1] * sy).abs() <= hl + bx * cy.abs() + by * sy.abs())
        ax_l = ((-d[..., 0] * sy + d[..., 1] * cy).abs() <= hw + bx * sy.abs() + by * cy.abs())
        return ax_x & ax_y & ax_f & ax_l & present

    def _point_clear(self, boxes, p, margin):
        d = (p[:, None] - boxes[..., :2]).abs()
        inside = (d[..., 0] <= boxes[..., 2] + margin) & (d[..., 1] <= boxes[..., 3] + margin) & (boxes[..., 4] > 0)
        return ~inside.any(1)

    def _add_traps(self, idx):
        """Axis-aligned wall across the start->goal line, half of them closed into a U around the start."""
        c, k, dev = self.cfg, len(idx), self.device
        s, g = self.pos[idx], self.goal[idx]
        d = g - s
        ax = (d[:, 0].abs() >= d[:, 1].abs()).long()  # 1: dominant axis is x
        dom, lat = 1 - ax, ax  # column index of dominant / lateral coordinate
        ar = torch.arange(k, device=dev)
        dd, sign = d[ar, dom].abs(), d[ar, dom].sign()
        s_dom, s_lat, g_lat = s[ar, dom], s[ar, lat], g[ar, lat]
        dw = 0.8 + self._rand(k) * 0.5 * (dd - 0.8).clamp(min=0)
        half = self._rand(k, lo=c.trap_half[0], hi=c.trap_half[1])
        lat_c = s_lat + self._rand(k) * (g_lat - s_lat)
        back = dw + self._rand(k, lo=0.3, hi=1.0)
        walls = torch.zeros((k, N_TRAP, 5), device=dev)
        def put(w, along, lateral, h_along, h_lat):
            walls[ar, w, dom], walls[ar, w, lat] = along, lateral
            walls[ar, w, 2 + dom], walls[ar, w, 2 + lat] = h_along, h_lat
        put(0, s_dom + sign * dw, lat_c, torch.full_like(dw, 0.08), half)
        put(1, s_dom + sign * (dw - back / 2), lat_c + half, back / 2, torch.full_like(dw, 0.08))
        put(2, s_dom + sign * (dw - back / 2), lat_c - half, back / 2, torch.full_like(dw, 0.08))
        walls[:, 0, 4] = self._rand(k, lo=0.5, hi=1.5)
        walls[:, 1:, 4] = (self._rand(k, 1) < c.cup_prob) * walls[:, :1, 4]
        walls[:, :, 4] *= ((self._rand(k) < c.trap_prob) & (dd > 1.5))[:, None]
        ok = ~self._footprint_hits(walls, s, self.yaw[idx], c.clearance).any(1) & self._point_clear(walls, g, c.clearance + 0.2)
        walls[:, :, 4] *= ok[:, None]
        t0 = c.n_boxes + 4
        self.boxes[idx, t0:t0 + N_TRAP] = walls
        self.trap[idx] = walls[:, 0, 4] > 0

    def _add_narrow(self, idx):
        """Wall across the start->goal line whose only opening is a door_width doorway, half the time
        extended into a corridor. Teaches that a tight passage must be taken when it is the way."""
        c, k, dev = self.cfg, len(idx), self.device
        s, g = self.pos[idx], self.goal[idx]
        d = g - s
        ax = (d[:, 0].abs() >= d[:, 1].abs()).long()
        dom, lat = 1 - ax, ax
        ar = torch.arange(k, device=dev)
        dd, sign = d[ar, dom].abs(), d[ar, dom].sign()
        frac = self._rand(k, lo=0.3, hi=0.7)
        along = s[ar, dom] + sign * frac * dd
        door = s[ar, lat] + frac * d[ar, lat] + self._rand(k, lo=-1.0, hi=1.0)
        w = self._rand(k, lo=c.door_width[0], hi=c.door_width[1])
        span = self._rand(k, lo=4.0, hi=8.0)  # long enough that going round the wall end rarely pays
        clen = self._rand(k, lo=1.0, hi=3.0) * (self._rand(k) < c.corridor_prob)
        walls = torch.zeros((k, N_NARROW, 5), device=dev)
        def put(j, a, lt, h_a, h_l):
            walls[ar, j, dom], walls[ar, j, lat], walls[ar, j, 2 + dom], walls[ar, j, 2 + lat] = a, lt, h_a, h_l
        thin = torch.full_like(w, 0.08)
        put(0, along, door + w / 2 + span / 2, thin, span / 2)  # wall halves on either side of the door
        put(1, along, door - w / 2 - span / 2, thin, span / 2)
        put(2, along, door + w / 2 + 0.08, clen / 2, thin)  # corridor sides through the door
        put(3, along, door - w / 2 - 0.08, clen / 2, thin)
        on = ~self.trap[idx] & (self._rand(k) < c.narrow_prob) & (dd > 1.5)
        walls[:, :2, 4] = 1.0
        walls[:, 2:, 4] = (clen > 0)[:, None].float()
        walls[:, :, 4] *= on[:, None]
        ok = ~self._footprint_hits(walls, s, self.yaw[idx], c.clearance).any(1) & self._point_clear(walls, g, c.clearance + 0.2)
        walls[:, :, 4] *= ok[:, None]
        t0 = c.n_boxes + 4 + N_TRAP
        self.boxes[idx, t0:t0 + N_NARROW] = walls
        self.narrow[idx] = walls[:, 0, 4] > 0

    def _geodesic(self, idx):
        """Shortest-path distance to the goal over the inflated boxes (8-connected relaxation).
        Training-only: feeds the critic and rejects unsolvable scenes, never the policy."""
        k, dev = len(idx), self.device
        b = self.boxes[idx]
        occ = torch.zeros((k, self.ng, self.ng), dtype=torch.bool, device=dev)
        for j in range(b.shape[1]):
            bj = b[:, j, :, None, None]
            occ |= ((self.geo_x - bj[:, 0]).abs() <= bj[:, 2] + GEO_INFLATE) & ((self.geo_y - bj[:, 1]).abs() <= bj[:, 3] + GEO_INFLATE) & (bj[:, 4] > 0)
        gi = ((self.goal[idx] + self.ng * GEO_RES / 2) / GEO_RES).long().clamp(0, self.ng - 1)
        dist = torch.full((k, self.ng, self.ng), math.inf, device=dev)
        dist[torch.arange(k, device=dev), gi[:, 0], gi[:, 1]] = 0
        steps = [(dx, dy, GEO_RES * math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        for it in range(4 * self.ng):
            p = torch.nn.functional.pad(dist, (1, 1, 1, 1), value=math.inf)
            nd = dist
            for dx, dy, cost in steps:
                nd = torch.minimum(nd, p[:, 1 + dx:1 + dx + self.ng, 1 + dy:1 + dy + self.ng] + cost)
            nd = torch.where(occ, math.inf, nd)
            if it % 16 == 15 and torch.equal(nd, dist):
                break
            dist = nd
        self.geo[idx] = dist.clamp(max=GEO_UNREACHABLE)

    def _geo_at(self, idx, p):
        i = ((p + self.ng * GEO_RES / 2) / GEO_RES).long().clamp(0, self.ng - 1)
        return self.geo[idx[:, None], i[..., 0], i[..., 1]] if p.dim() == 3 else self.geo[idx, i[:, 0], i[:, 1]]

    def privileged(self):
        """Critic-only observation: geodesic distance here and its slope toward 8 directions 0.4m away."""
        idx = torch.arange(self.cfg.num_envs, device=self.device)
        c, s = torch.cos(self.yaw)[:, None], torch.sin(self.yaw)[:, None]
        d = self.priv_dirs
        pts = self.pos[:, None] + 0.4 * torch.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], -1)
        here = self._geo_at(idx, self.pos)
        slope = ((self._geo_at(idx, pts) - here[:, None]) / 0.4).clamp(-1.5, 1.5)
        return torch.cat([(here / 10).clamp(max=3)[:, None], slope], 1)

    def clearance(self, pos=None, yaw=None):
        """Distance from the robot outline to the nearest box (0 when touching). pos [n,(T,)2], yaw [n,(T)]."""
        pos = self.pos if pos is None else pos
        yaw = self.yaw if yaw is None else yaw
        c, s = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
        e = self.fp_edge
        pts = pos[..., None, :] + torch.stack([c * e[:, 0] - s * e[:, 1], s * e[:, 0] + c * e[:, 1]], -1)  # [n,(T,)P,2]
        b = self.boxes.view(self.boxes.shape[0], *([1] * (pts.dim() - 2)), *self.boxes.shape[1:])  # [n,(1,)1,B,5]
        d = ((pts[..., None, :] - b[..., :2]).abs() - b[..., 2:4]).clamp(min=0).norm(dim=-1)  # [n,(T,)P,B]
        d = torch.where(b[..., 4] > 0, d, math.inf)
        return d.amin((-2, -1))

    def rollout_clearance(self, cmd, horizon=1.0, dt=0.1):
        """Smallest clearance along a constant-command unicycle rollout, like the runtime guard's check."""
        k = int(horizon / dt)
        th = self.yaw[:, None] + cmd[:, 1:2] * dt * torch.arange(1, k + 1, device=self.device)
        step = cmd[:, 0, None, None] * dt * torch.stack([torch.cos(th), torch.sin(th)], -1)  # [n,T,2]
        return self.clearance(self.pos[:, None] + step.cumsum(1), th).amin(1)

    def _visit(self, idx):
        """Mark the current novelty cell; returns True where it was not visited before."""
        cell = ((self.pos[idx] + self.cfg.arena / 2 + 1) / NOVELTY_CELL).long().clamp(0, self.nv - 1)
        new = ~self.visited[idx, cell[:, 0], cell[:, 1]]
        self.visited[idx, cell[:, 0], cell[:, 1]] = True
        return new

    def reset(self, idx=None):
        c = self.cfg
        idx = torch.arange(c.num_envs, device=self.device) if idx is None else idx
        if len(idx) == 0:
            return
        self._make_boxes(idx)
        todo = idx
        for attempt in range(50):
            k, half = len(todo), c.arena / 2 - 1.0
            pos = self._rand(k, 2, lo=-half, hi=half)
            yaw = self._rand(k, lo=-math.pi, hi=math.pi)
            r = self._rand(k, lo=c.goal_dist[0], hi=c.goal_dist[1])
            a = self._rand(k, lo=-math.pi, hi=math.pi)
            goal = pos + r[:, None] * torch.stack([torch.cos(a), torch.sin(a)], -1)
            ok = ~self._footprint_hits(self.boxes[todo], pos, yaw, c.clearance).any(1)
            ok &= self._point_clear(self.boxes[todo], goal, c.clearance + 0.2) & (goal.abs() < half).all(1)
            self.pos[todo[ok]], self.yaw[todo[ok]], self.goal[todo[ok]] = pos[ok], yaw[ok], goal[ok]
            todo = todo[~ok]
            if len(todo) == 0:
                break
            if attempt % 10 == 9:
                self._make_boxes(todo)  # too crowded, regenerate
        assert len(todo) == 0, "could not place start/goal"
        self._add_traps(idx)
        self._add_narrow(idx)
        self._geodesic(idx)
        ok = self._geo_at(idx, self.pos[idx]) < GEO_UNREACHABLE
        if not ok.all():  # walls closed the start off from the goal; draw those scenes again
            self.reset(idx[~ok])
            idx = idx[ok]
            if len(idx) == 0:
                return
        k = len(idx)
        self.cam_h[idx] = self._rand(k, lo=c.cam_height[0], hi=c.cam_height[1])
        self.pitch[idx] = torch.deg2rad(self._rand(k, lo=c.cam_pitch_deg[0], hi=c.cam_pitch_deg[1]))
        self.cam_off[idx] = self._rand(k, lo=c.cam_forward[0], hi=c.cam_forward[1])
        self.rear_on[idx] = self._rand(k) < c.rear_prob
        self.rear_off[idx] = self._rand(k, lo=c.rear_back[0], hi=c.rear_back[1])
        self.vel[idx] = 0
        self.cmd_queue[idx] = 0
        self.last_action[idx] = 0
        self.steps[idx] = 0
        self.grid[idx] = 0
        cam = self.camera_pose(idx)[1]
        # real grid starts centered on the first camera pose: origin = cam - shape*res/2
        self.base[idx] = cam - torch.tensor(GRID_SHAPE, device=self.device) * GRID_RES / 2
        self.o[idx] = 0
        self.best_dist[idx] = (self.goal[idx] - self.pos[idx]).norm(dim=1)
        self.geo_start[idx] = self._geo_at(idx, self.pos[idx])
        self.path_len[idx], self.abs_w[idx] = 0, 0
        self.min_clear[idx] = math.inf
        self.hist[idx] = math.nan
        self.hist[idx, 0] = self.pos[idx]
        self.visited[idx] = False
        self._visit(idx)
        self._sense(idx)

    # ---------- perception ----------
    def camera_pose(self, idx, jitter=False, rear=False):
        """Front camera (or, with rear=True, the backward-looking rear camera) pose for envs idx."""
        pitch, h = self.pitch[idx], self.cam_h[idx]
        if jitter:
            pitch = pitch + math.radians(self.cfg.pitch_jitter_deg) * torch.randn(pitch.shape, generator=self.gen, device=self.device)
            h = h + self.cfg.height_jitter * torch.randn(h.shape, generator=self.gen, device=self.device)
        yaw = self.yaw[idx] + math.pi if rear else self.yaw[idx]
        R = _rot(yaw, pitch)
        off = self.rear_off[idx, None] if rear else self.cam_off[idx, None]
        return R, torch.cat([self.pos[idx], h[:, None]], 1) + R[..., 2] * off

    def _render(self, idx, R, cam):
        """Z-depth of the sampled pixels against ground + boxes. Returns [k, P] (nan = invalid)."""
        c, b = self.cfg, self.boxes[idx]
        dirs = torch.einsum("kij,pj->kpi", R, self.pix_dir)  # world ray per unit z-depth
        dz = dirs[..., 2]
        depth = torch.where(dz < -1e-6, -cam[:, None, 2] / dz, torch.full_like(dz, math.inf))
        o, d = cam[:, None, None], dirs[:, :, None]  # [k,1,1,3], [k,P,1,3]
        lo = torch.stack([b[..., 0] - b[..., 2], b[..., 1] - b[..., 3], torch.zeros_like(b[..., 0])], -1)[:, None]
        hi = torch.stack([b[..., 0] + b[..., 2], b[..., 1] + b[..., 3], b[..., 4]], -1)[:, None]
        inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        t1, t2 = (lo - o) * inv, (hi - o) * inv
        tn = torch.minimum(t1, t2).amax(-1)
        tf = torch.maximum(t1, t2).amin(-1)
        hit = (tn <= tf) & (tn > 0) & (b[:, None, :, 4] > 0)
        depth = torch.minimum(depth, torch.where(hit, tn, torch.full_like(tn, math.inf)).amin(-1))
        sigma = depth ** 2 * c.disparity_noise_px / (c.fx * c.baseline)
        depth = torch.where(depth < 20.0, depth + sigma.nan_to_num(0) * torch.randn(depth.shape, generator=self.gen, device=self.device), torch.full_like(depth, FAR_DEPTH))
        drop = torch.rand(depth.shape, generator=self.gen, device=self.device) < c.depth_drop
        return torch.where(drop | (depth <= 0), torch.full_like(depth, math.nan), depth)

    def _recenter(self, idx, cam):
        """Port of the node's recenter rule + roll_occupancy_grid on the toroidal grid."""
        center = self.base[idx, :2] + (self.o[idx] + torch.tensor([GX, GY], device=self.device) / 2) * GRID_RES
        move = (cam[:, :2] - center).norm(dim=1) > 0.1  # z never shifts: flat ground, fixed camera height
        if not move.any():
            return
        idx, cam = idx[move], cam[move]
        new_o = torch.round((cam[:, :2] - torch.tensor([GX, GY], device=self.device) * GRID_RES / 2 - self.base[idx, :2]) / GRID_RES).long()
        old_o = self.o[idx]
        ar = torch.arange(GX, device=self.device)
        g = new_o[:, :, None] + ar  # global indices of the new window, per axis
        enter = (g < old_o[:, :, None]) | (g >= old_o[:, :, None] + GX)
        mask = torch.zeros_like(enter).scatter_(2, g % GX, enter)  # storage slots that must be cleared
        keep = ~(mask[:, 0, :, None, None] | mask[:, 1, None, :, None])
        self.grid[idx] *= keep
        self.o[idx] = new_o

    def _fuse(self, idx, R, cam, depth, primary=True):
        """Port of run_raycasting_loopy + decay/clip, written in local window coordinates. The primary
        (front) frame decays the grid; a rear frame is only added, like the Pilot's dual-camera planner."""
        k, dev = len(idx), self.device
        origin = self.base[idx] + torch.cat([self.o[idx], torch.zeros((k, 1), dtype=torch.long, device=dev)], 1) * GRID_RES
        pts = cam[:, None] + torch.einsum("kij,kpj->kpi", R, self.pix_dir[None] * depth[..., None])
        # float64: the grid origin is cam - shape*res/2, so the camera sits exactly on a voxel edge
        start = torch.floor((cam.double() - origin.double()) / GRID_RES).int()  # [k,3]
        end = torch.floor((pts.double() - origin[:, None].double()) / GRID_RES).nan_to_num(0).clamp(-10**6, 10**6).int()
        o3 = torch.cat([self.o[idx], torch.zeros((k, 1), dtype=torch.long, device=dev)], 1).int()
        voxels = _ray_voxels_compiled if dev.type == "cuda" else _ray_voxels
        fi, inb, fe, e_inb = voxels(start, end, torch.isfinite(depth), o3)
        new = torch.zeros((k, GX * GY * GZ), device=dev)
        new.scatter_add_(1, fi, inb.float() * -0.05)
        new.scatter_add_(1, fe, e_inb.float() * 0.2)
        new = new.clamp(-0.1, 0.1).view(k, GX, GY, GZ)
        self.grid[idx] = ((self.grid[idx] * 0.99 if primary else self.grid[idx]) + new).clamp(-0.2, 0.2)

    def _sense(self, idx):
        for ch in torch.split(idx, self.cfg.env_chunk):
            R, cam = self.camera_pose(ch, jitter=True)
            self._recenter(ch, cam)
            self._fuse(ch, R, cam, self._render(ch, R, cam))
            rear = ch[self.rear_on[ch]]
            if len(rear):
                R, cam = self.camera_pose(rear, jitter=True, rear=True)
                self._fuse(rear, R, cam, self._render(rear, R, cam), primary=False)

    def bev(self, idx=None):
        """Port of bev.make_bev on the toroidal grid. Returns [k, 2, BEV_SIZE, BEV_SIZE]."""
        idx = torch.arange(self.cfg.num_envs, device=self.device) if idx is None else idx
        cam = self.camera_pose(idx)[1]
        k = len(idx)
        ref_z = cam[:, 2]
        z_top = self.base[idx, 2, None] + (torch.arange(GZ, device=self.device) + 1) * GRID_RES - ref_z[:, None]  # [k,Z]
        floor = self.base[idx, 2] - ref_z
        center = self.pos[idx]  # export_bev: cam - R @ cam_offset_3d == control center
        c, s = torch.cos(self.yaw[idx])[:, None, None], torch.sin(self.yaw[idx])[:, None, None]
        wx = center[:, 0, None, None] + c * self.bev_fwd - s * self.bev_left
        wy = center[:, 1, None, None] + s * self.bev_fwd + c * self.bev_left
        gx = torch.floor((wx - self.base[idx, 0, None, None]) / GRID_RES).long()
        gy = torch.floor((wy - self.base[idx, 1, None, None]) / GRID_RES).long()
        lx, ly = gx - self.o[idx, 0, None, None], gy - self.o[idx, 1, None, None]
        inside = (lx >= 0) & (lx < GX) & (ly >= 0) & (ly < GY)
        flat = ((gx % GX) * GY + gy % GY).view(k, -1)
        cols = self.grid.view(self.cfg.num_envs, GX * GY, GZ)[idx[:, None], flat]  # only the sampled columns, [k,64*64,Z]
        occ = (cols > OCC_THRESHOLD) & ((z_top - GRID_RES) <= ROBOT_CONFIG.obstacle.robot_z_top)[:, None]
        h = torch.where(occ, z_top[:, None], floor[:, None, None]).amax(-1).view(k, BEV_SIZE, BEV_SIZE)
        kn = (cols.abs() > KNOWN_EPS).any(-1).float().view(k, BEV_SIZE, BEV_SIZE)
        h = torch.where(inside, h, floor[:, None, None])
        hit = h > floor[:, None, None]
        cand = hit & self.ground_ring & (h <= GROUND_MAX)
        ground = torch.nanmedian(torch.where(cand, h, math.nan).view(k, -1), dim=1).values  # lower median
        ground = torch.where(cand.view(k, -1).sum(1) >= 10, ground, GROUND_PRIOR)
        elev = torch.where(hit, (h - ground[:, None, None]).clamp(*ELEV_RANGE), 0.0)
        return torch.stack([elev, torch.where(inside, kn, 0.0), self._trail(idx)], 1)

    def _trail(self, idx):
        """Port of bev.trail_channel."""
        k = len(idx)
        rel = self.hist[idx] - self.pos[idx, None]
        c, s = torch.cos(self.yaw[idx])[:, None], torch.sin(self.yaw[idx])[:, None]
        fwd, left = c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]
        i = torch.floor(BEV_SIZE / 2 - fwd / BEV_RES).nan_to_num(-1).long()
        j = torch.floor(BEV_SIZE / 2 - left / BEV_RES).nan_to_num(-1).long()
        ok = (i >= 0) & (i < BEV_SIZE) & (j >= 0) & (j < BEV_SIZE)
        val = (1 - torch.arange(TRAIL_LEN, device=self.device) / TRAIL_LEN).expand(k, -1) * ok
        trail = torch.zeros((k, BEV_SIZE * BEV_SIZE), device=self.device)
        trail.scatter_reduce_(1, torch.where(ok, i * BEV_SIZE + j, 0), val, reduce="amax")
        return torch.nn.functional.max_pool2d(trail.view(k, 1, BEV_SIZE, BEV_SIZE), 3, 1, 1)[:, 0]

    def unrolled_grid(self, i):
        """Env i's grid in the planner's local layout plus its world origin (for tests / numpy make_bev)."""
        g = torch.roll(self.grid[i], shifts=(-int(self.o[i, 0] % GX), -int(self.o[i, 1] % GY)), dims=(0, 1))
        origin = self.base[i].clone()
        origin[:2] += self.o[i].float() * GRID_RES
        return g, origin

    # ---------- RL interface ----------
    def goal_robot_frame(self):
        d = self.goal - self.pos
        c, s = torch.cos(self.yaw), torch.sin(self.yaw)
        return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)

    def observe(self):
        c = self.cfg
        vec = torch.cat([self.goal_robot_frame() / 5.0, self.vel / torch.tensor([c.v_max, c.w_max], device=self.device), self.last_action], 1)
        return {"bev": self.bev(), "vec": vec}

    def step(self, action):
        """action in [-1, 1]^2 -> (forward speed, yaw rate). Auto-resets finished envs."""
        c = self.cfg
        action = action.clamp(-1, 1)
        cmd = torch.stack([torch.where(action[:, 0] >= 0, action[:, 0] * c.v_max, -action[:, 0] * c.v_min), action[:, 1] * c.w_max], 1)
        self.cmd_queue = torch.cat([self.cmd_queue[:, 1:], cmd[:, None]], 1)
        target = self.cmd_queue[:, 0] * (1 + c.act_noise * torch.randn(cmd.shape, generator=self.gen, device=self.device))
        lim = torch.tensor([c.lin_acc, c.ang_acc], device=self.device) * c.dt
        self.vel = self.vel + (target - self.vel).clamp(-lim, lim)
        self.yaw = self.yaw + self.vel[:, 1] * c.dt
        move = self.vel[:, :1] * c.dt * torch.stack([torch.cos(self.yaw), torch.sin(self.yaw)], -1)
        self.pos = self.pos + move
        self.path_len += move.norm(dim=1)
        self.abs_w += self.vel[:, 1].abs()
        self.yaw = torch.atan2(torch.sin(self.yaw), torch.cos(self.yaw))
        self.steps += 1

        self.hist = torch.cat([self.pos[:, None], self.hist[:, :-1]], 1)

        dist = (self.goal - self.pos).norm(dim=1)
        collided = self._footprint_hits(self.boxes, self.pos, self.yaw).any(1)
        reached = dist < c.goal_tol
        timeout = self.steps >= c.max_steps
        # progress only counts new closest approaches, so backing out of a dead end costs nothing
        reward = (self.best_dist - dist).clamp(min=0) - c.time_penalty - 0.01 * ((action - self.last_action) ** 2).sum(1)
        reward += c.novelty_reward * self._visit(torch.arange(c.num_envs, device=self.device))
        clear = self.clearance()
        self.min_clear = torch.minimum(self.min_clear, clear)
        reward -= c.clearance_penalty * ((c.clearance_margin - clear) / c.clearance_margin).clamp(0, 1)
        reward -= c.turn_penalty * (self.vel[:, 1] / c.w_max) ** 2
        # Same rule as rl_planner.safe_command: a command is refused if its 1s rollout comes within the
        # margin, unless the robot is already that close and the move gets no closer (escape). Penalizing
        # every close rollout instead made standing still the cheapest option and the policy froze.
        roll = self.rollout_clearance(cmd)
        refused = (roll < c.guard_margin) & ~((clear < c.guard_margin) & (roll >= (clear - ESCAPE_TOL).clamp(min=0)))
        reward -= c.guard_penalty * refused * ((c.guard_margin - roll) / c.guard_margin).clamp(0, 1)
        reward -= c.reverse_penalty * (-cmd[:, 0] / -c.v_min).clamp(min=0) * torch.where(self.rear_on, 0.3, 1.0)
        efficiency = ((self.geo_start - c.goal_tol) / self.path_len.clamp(min=1e-3)).clamp(0, 1)
        reward += (10.0 + c.efficiency_bonus * efficiency) * reached - 10.0 * collided
        self.best_dist, self.last_action = torch.minimum(self.best_dist, dist), action
        done = collided | reached | timeout
        info = {"collided": collided, "reached": reached, "timeout": timeout & ~collided & ~reached, "trap": self.trap.clone(), "narrow": self.narrow.clone(),
                "min_clearance": self.min_clear.clone(), "efficiency": efficiency, "mean_abs_w": self.abs_w / self.steps.clamp(min=1)}
        self._sense(torch.nonzero(~done).squeeze(1))
        self.reset(torch.nonzero(done).squeeze(1))
        return self.observe(), reward, done, info
