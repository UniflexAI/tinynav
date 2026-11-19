import numpy as np
from tinynav.core.navigation_safety import advance, executable_speed

"""Score Go2 route progress against recorded low/high actuation responses.

Motion candidates include a full controller-speed braking tail. The extra half
grid cell on moving commands reserves space for the next BEV update.
"""


def motion(rate, previous, profile, q):
    out = np.zeros_like(rate)
    for j, names in enumerate([("forward", "reverse"), ("left", "right")]):
        for sign, name in zip((1, -1), names):
            magnitude = np.maximum(0, rate[:, j] * sign)
            moving = previous[:, j] * sign > (0.02 if j == 0 else 0.04)
            for active, mode in [(False, "start"), (True, "sustain")]:
                x, y = profile[name, mode, q]
                speed = np.interp(magnitude, x, y) + np.maximum(0, magnitude - x[-1])
                out[:, j] += sign * np.where(active == moving, speed, 0.0)
    return out


def compile_profile(data):
    out = {}
    for name, modes in data.items():
        if name == "description":
            continue
        for mode, table in modes.items():
            for q in (0, 1):
                out[name, mode, q] = np.array(
                    [(0.0, 0.0)]
                    + sorted(
                        (float(k), max(0.0, [min, max][q](v))) for k, v in table.items()
                    )
                ).T
    return out


def path(g, commands, profile, q, prior, *, right_stalled=False):
    n = len(commands)
    dt = g.limits.period
    pose = np.tile(g.pose, (n, 1))
    cmd = np.tile(g.command, (n, 1))
    vel = np.tile(prior, (n, 1))
    poses = [pose.copy()]
    steps = max(1, int(np.ceil(g.delay / 0.025)))
    for _ in range(steps):
        pose = advance(pose, vel[:, 0], vel[:, 1], g.delay / steps)
        poses.append(pose)
    hold_ticks = 12
    brake_ticks = (
        int(
            np.ceil(
                max(g.limits.max_reverse, g.limits.max_forward, abs(g.command[0]))
                / (g.limits.linear_acceleration * dt)
            )
        )
        + 2
    )
    for tick in range(hold_ticks + brake_ticks):
        target = commands if tick < 12 else np.zeros_like(commands)
        cmd[:, 0] = executable_speed(cmd[:, 0], target[:, 0], dt, g.limits)
        cmd[:, 1] = target[:, 1]
        vel = motion(cmd, vel, profile, q)
        if right_stalled:
            vel[:, 1] = np.maximum(vel[:, 1], 0.0)
        for _ in range(2):
            pose = advance(pose, vel[:, 0], vel[:, 1], dt / 2)
            poses.append(pose)
        if tick >= 12 and not np.any(cmd):
            break
    return np.stack(poses, axis=1)


def select(g, goal, route, guide, profile, prior):
    from tinynav.rl.route_rejoin import route_coordinates

    if g.delay > 0.5 or not np.isfinite(prior).all() or route is None or len(route) < 2:
        return 0.0, 0.0, g.now
    projection = route_coordinates(route, g.pose[:2])
    if projection is None:
        return 0.0, 0.0, g.now
    pts, s, at, off = projection
    ahead = np.array([np.interp(min(at + 0.70, s[-1]), s, pts[:, k]) for k in (0, 1)])
    bearing = np.arctan2(*(ahead - g.pose[:2])[::-1])
    cmds = g.limits.executable(
        [guide]
        + [
            (v, w)
            for v in (0.0, 0.25, 0.35, 0.45, 0.55, -0.25)
            for w in (0.0, -0.25, 0.3, -0.5, 0.55, -0.75, 0.75)
        ]
    )
    paths = [path(g, cmds, profile, q, prior) for q in (0, 1)]
    scores = []
    for paths_q in paths:
        end = paths_q[:, -1]
        err = np.angle(np.exp(1j * (bearing - end[:, 2])))
        scores.append(-np.linalg.norm(end[:, :2] - ahead, axis=1) - 0.35 * np.abs(err))
    # Right-turn sustain was not measured. Also check translation with no right yaw.
    paths.extend(path(g, cmds, profile, q, prior, right_stalled=True) for q in (0, 1))
    order = np.argsort(-np.minimum(*scores))
    for idx in np.array_split(order, (len(cmds) + 3) // 4):
        ok, nominal, _ = g.evaluate(cmds[idx])
        room = nominal.copy()
        for p in paths:
            room = np.minimum(
                room,
                g.clearance(p[idx].reshape(-1, 3)).reshape(len(idx), -1).min(axis=1),
            )
        valid = ok & (room >= g.limits.margin + 0.5 * g.resolution)
        if np.any(valid):
            k = np.flatnonzero(valid)[0]
            return *cmds[idx[k]], float(room[k])
    stop = np.zeros((1, 2))
    room = float(g.evaluate(stop)[1][0])
    for q in (0, 1):
        room = min(room, float(g.clearance(path(g, stop, profile, q, prior)[0]).min()))
    return 0.0, 0.0, room
