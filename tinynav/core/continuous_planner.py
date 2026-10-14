"""Experimental ground-robot planner: grid guide, B-spline fit, feasible rollout."""
import heapq
import time
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def guide_path(start, target, esdf, origin, resolution, clearance, preferred=None):
    preferred = clearance + .15 if preferred is None else preferred
    shape = np.array(esdf.shape)
    cell = lambda p: tuple(np.clip(np.floor((p[:2] - origin[:2]) / resolution).astype(int), 0, shape - 1))
    source = cell(start)
    # Pick a reachable local frontier toward distant targets rather than treating
    # the clipped target cell as necessarily reachable.
    goal = cell(target)
    queue = [(0., source)]
    costs, parents = {source: 0.}, {}
    best = source
    best_dist = np.linalg.norm(np.array(source) - goal)
    moves = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
    closed = set()
    while queue:
        _, u = heapq.heappop(queue)
        if u in closed:
            continue
        closed.add(u)
        dist = np.linalg.norm(np.array(u) - goal)
        if dist < best_dist:
            best, best_dist = u, dist
        if u == goal:
            best = u
            break
        for dx, dy in moves:
            v = (u[0] + dx, u[1] + dy)
            if not (0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]):
                continue
            if esdf[v] < clearance:
                continue
            if dx and dy and (esdf[u[0] + dx, u[1]] < clearance or esdf[u[0], u[1] + dy] < clearance):
                continue
            g = costs[u] + np.hypot(dx, dy) * (1 + .08 / max(esdf[v], .01)
                + 8 * (max(preferred - esdf[v], 0.) / preferred) ** 2)
            if g < costs.get(v, float('inf')):
                costs[v], parents[v] = g, u
                heapq.heappush(queue, (g + np.linalg.norm(np.array(v) - goal), v))
    if best == source:
        return None
    cells = [best]
    while cells[-1] != source:
        cells.append(parents[cells[-1]])
    path = origin[:2] + (np.asarray(cells[::-1]) + .5) * resolution
    path[0] = start[:2]
    target_grid = (target[:2] - origin[:2]) / resolution
    if best == goal and np.all(target_grid >= 0) and np.all(target_grid < shape):
        path[-1] = target[:2]
    return path


def plan(start, quat, target, esdf, origin, resolution, robot, extra_clearance=0.):
    began = time.perf_counter()
    front, rear, half_w = robot.footprint_from_control()
    radius = half_w + robot.safety_radius
    preferred = max(radius, np.hypot(max(front, rear), half_w) + robot.safety_radius + .10 + extra_clearance)
    guide = guide_path(start, target, esdf, origin, resolution, radius, preferred)
    if guide is None:
        return None, {'reason': 'no_local_route'}
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(guide, axis=0), axis=1))]
    horizon = min(arc[-1], 1.5)
    knots = np.linspace(0, horizon, 8)
    seed = np.column_stack([np.interp(knots, arc, guide[:, d]) for d in range(2)])
    basis = make_interp_spline(np.linspace(0, 1, 8), np.eye(8), k=3)(np.linspace(0, 1, 61))
    def curve(x):
        return basis @ np.vstack([seed[0], x.reshape(-1, 2), seed[-1]])
    def objective(x):
        points = curve(x)
        grid = ((points - origin[:2]) / resolution - .5).T
        distances = map_coordinates(esdf, grid, order=1, mode='constant', cval=0.)
        return (20 * np.square(np.maximum(preferred - distances, 0)).sum()
                + 2 * np.square(np.diff(points, n=2, axis=0)).sum()
                + .05 * np.square(points - basis @ seed).sum())
    bounds = [(origin[d] + resolution, origin[d] + (esdf.shape[d] - 1) * resolution)
              for _ in range(6) for d in range(2)]
    result = minimize(objective, seed[1:-1].ravel(), method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 15, 'maxfun': 240, 'ftol': 1e-5})
    points = curve(result.x)
    fwd = Rotation.from_quat(quat).as_matrix()[:2, 2]
    initial_yaw = np.arctan2(fwd[1], fwd[0])
    body_x = np.linspace(-rear, front, max(2, int(np.ceil((front+rear)/(resolution*.5)))+1))
    body_y = np.linspace(-half_w, half_w, max(2, int(np.ceil(2*half_w/(resolution*.5)))+1))
    body = np.stack(np.meshgrid(body_x, body_y), axis=-1).reshape(-1, 2)
    if robot.shape == 'circle':
        body = body[np.linalg.norm(body, axis=1) <= robot.radius]
    def rollout(limit):
        p, yaw, nearest = start[:2].copy(), initial_yaw, 0
        poses = []
        for _ in range(31):
            nearest += int(np.argmin(np.linalg.norm(points[nearest:] - p, axis=1)))
            ahead = points[nearest:]
            idx = np.flatnonzero(np.linalg.norm(ahead - p, axis=1) >= .3)
            aim = ahead[idx[0]] if len(idx) else ahead[-1]
            delta = aim - p
            error = (np.arctan2(delta[1], delta[0]) - yaw + np.pi) % (2*np.pi) - np.pi
            omega = np.clip(2.5 * error, -robot.max_angular_vel, robot.max_angular_vel)
            speed = min(limit, robot.max_linear_vel, np.linalg.norm(target[:2] - p)) * max(0., np.cos(error)) ** 2
            if speed < robot.min_linear_vel:
                speed = robot.min_linear_vel if limit > 0 and abs(error) < .45 and np.linalg.norm(target[:2]-p) > .08 else 0.
            if abs(omega) < robot.min_angular_vel:
                omega = 0.
            q = Rotation.from_euler('z', yaw).as_matrix() @ np.array([[0,0,1],[-1,0,0],[0,-1,0]])
            poses.append(np.r_[p, start[2], Rotation.from_matrix(q).as_quat()])
            yaw += omega * .1
            p += speed * .1 * np.array([np.cos(yaw), np.sin(yaw)])
        return np.asarray(poses)
    def safe(poses):
        rotations = Rotation.from_quat(poses[:,3:]).as_matrix()
        forward = rotations[:,:2,2]
        left = np.column_stack([-forward[:,1],forward[:,0]])
        samples = poses[:,None,:2] + forward[:,None,:] * body[None,:,0,None] + left[:,None,:] * body[None,:,1,None]
        grid = ((samples.reshape(-1,2) - origin[:2]) / resolution - .5).T
        distances = map_coordinates(esdf, grid, order=1, mode='constant', cval=0.)
        return bool(np.all(distances >= max(robot.safety_radius, .01)))
    for limit in (.45, .25, .12, 0.):
        poses = rollout(limit)
        if safe(poses):
            return (poses, np.array([limit, 0.])), {
                'reason': 'ok', 'solve_ms': (time.perf_counter()-began)*1000,
                'speed_limit': limit, 'optimizer_success': bool(result.success),
                'guide_points': len(guide)}
    # The controller executes the first second only. A safe short prefix can
    # advance the receding horizon when a later predicted turn is infeasible.
    for limit in (.25, .12, 0.):
        prefix = rollout(limit)[:11]
        if safe(prefix):
            poses = np.vstack([prefix, np.tile(prefix[-1], (20, 1))])
            return (poses, np.array([limit, 0.])), {
                'reason': 'short_horizon', 'speed_limit': limit,
                'solve_ms': (time.perf_counter()-began)*1000}
    return None, {'reason': 'rollout_collision', 'solve_ms': (time.perf_counter()-began)*1000}


class ContinuousPlanner:
    """Progress watchdog and bounded recovery, using observed free space only."""
    def __init__(self):
        self.anchor = None
        self.anchor_time = None
        self.target = None
        self.recovery_until = 0.
        self.recovery_count = 0
        self.last_recovery = -float('inf')

    def step(self, start, quat, target, esdf, origin, resolution, robot, stamp, known_free):
        if self.target is None or np.linalg.norm(target - self.target) > .1:
            self.__init__()
            self.target = target.copy()
        if self.anchor is None or np.linalg.norm(start[:2] - self.anchor) > .08:
            self.anchor, self.anchor_time = start[:2].copy(), stamp
        if np.linalg.norm(start[:2] - target[:2]) < .08:
            return None, {'reason': 'arrived', 'recovery_count': self.recovery_count}
        stalled = stamp - self.anchor_time >= 2.
        if stamp < self.recovery_until:
            output = self.reverse(start, quat, esdf, origin, resolution, robot, known_free)
            if output is not None:
                return output, {'reason': 'recover_reverse', 'recovery_count': self.recovery_count}
            self.recovery_until = 0.
        output, info = plan(start, quat, target, esdf, origin, resolution, robot,
                            extra_clearance=.08 if stalled else 0.)
        info['recovery_count'] = self.recovery_count
        info['stalled'] = stalled
        blocked = output is None or output[1][0] <= 0.
        if stalled and blocked and self.recovery_count >= 3:
            return None, dict(info, reason="recovery_exhausted")
        if stalled and blocked and stamp - self.last_recovery >= 5.:
            reverse = self.reverse(start, quat, esdf, origin, resolution, robot, known_free)
            if reverse is not None:
                self.recovery_until, self.last_recovery = stamp + .6, stamp
                self.recovery_count += 1
                return reverse, {'reason': 'recover_reverse', 'recovery_count': self.recovery_count}
            info['recovery_blocked'] = 'rear_not_confirmed_clear'
        return output, info

    @staticmethod
    def reverse(start, quat, esdf, origin, resolution, robot, known_free):
        forward = Rotation.from_quat(quat).as_matrix()[:2, 2]
        left = np.array([-forward[1], forward[0]])
        poses = np.tile(np.r_[start, quat], (31, 1))
        poses[:, :2] -= (np.minimum(np.arange(31) * .1, .8) * robot.reverse_speed)[:, None] * forward
        front, rear, half_w = robot.footprint_from_control()
        xx = np.linspace(-rear, front, max(2, int(np.ceil((front+rear)/(resolution*.5)))+1))
        yy = np.linspace(-half_w, half_w, max(2, int(np.ceil(2*half_w/(resolution*.5)))+1))
        offsets = np.stack(np.meshgrid(xx, yy), axis=-1).reshape(-1, 2)
        if robot.shape == 'circle':
            offsets = offsets[np.linalg.norm(offsets, axis=1) <= robot.radius]
        samples = poses[:,None,:2] + offsets[None,:,0,None]*forward + offsets[None,:,1,None]*left
        grid = ((samples.reshape(-1,2) - origin[:2]) / resolution - .5).T
        clearance = map_coordinates(esdf, grid, order=1, mode='constant', cval=0.)
        observed = map_coordinates(known_free.astype(float), grid, order=0, mode='constant', cval=0.)
        if np.all(clearance >= max(robot.safety_radius,.01)) and np.all(observed > .5):
            return poses, np.array([-robot.reverse_speed, 0.])
        return None
