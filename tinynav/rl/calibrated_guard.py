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
    poses = [pose.copy()]
    vel = np.tile(prior, (n, 1))
    applied = np.asarray(g.command, dtype=float).copy()
    steps = int(np.ceil(g.delay / .025))
    for _ in range(steps):
        dt_delay = g.delay / steps
        if g.pending_command is not g.command:
            applied[0] = executable_speed(applied[0], g.pending_command[0], dt_delay, g.limits)
            applied[1] = g.pending_command[1]
            vel = motion(np.broadcast_to(applied, (n, 2)), vel, profile, q)
        if right_stalled:
            vel[:, 1] = np.maximum(vel[:, 1], 0.)
        pose = advance(pose, vel[:, 0], vel[:, 1], dt_delay)
        poses.append(pose.copy())

    hold_ticks = 12
    brake_ticks = int(np.ceil(max(g.limits.max_forward, g.limits.max_reverse,
                                  abs(applied[0])) / (g.limits.linear_acceleration * dt))) + 2
    rates = []
    speed = np.full(n, applied[0])
    for tick in range(hold_ticks + brake_ticks):
        target = commands if tick < hold_ticks else np.zeros_like(commands)
        speed = executable_speed(speed, target[:, 0], dt, g.limits)
        rates.append(np.column_stack((speed, target[:, 1])))
        if tick >= hold_ticks and not speed.any():
            break
    # Batch the interpolation; only the choice of starting/sustaining response
    # depends on the previous tick.
    rates = np.asarray(rates)
    flat = rates.reshape(-1, 2)
    starting = motion(flat, np.zeros_like(flat), profile, q).reshape(rates.shape)
    sustaining = motion(flat, np.sign(flat), profile, q).reshape(rates.shape)
    for rate, low_speed, moving_speed in zip(rates, starting, sustaining):
        vel = np.where(vel * np.sign(rate) > (.02, .04), moving_speed, low_speed)
        if right_stalled:
            vel[:, 1] = np.maximum(vel[:, 1], 0.)
        for _ in range(2):
            pose = advance(pose, vel[:, 0], vel[:, 1], dt / 2)
            poses.append(pose)
    return np.stack(poses, axis=1)


# Upper bound from the final live-pose snapshot through command publication.
SELECTION_BUDGET_S = 0.20


def retreat(g, profile, prior):
    """One slow reverse step toward a wider section, rechecked on each new BEV."""
    distance = np.arange(0., .81, .025)
    corridor = np.tile(g.pose, (len(distance), 1))
    corridor[:, :2] -= distance[:, None] * np.array([np.cos(g.pose[2]), np.sin(g.pose[2])])
    width = g.clearance(corridor)
    wider = width >= g.limits.margin + .5 * g.resolution
    if not np.any(wider & (np.minimum.accumulate(width) >= g.limits.margin)):
        return None

    commands = g.limits.executable([(-.20, 0.)])
    ok, room, _ = g.evaluate(commands)
    if not ok[0]:
        return None
    for q in (0, 1):
        trajectory = path(g, commands, profile, q, prior)[0]
        room[0] = min(room[0], g.clearance(trajectory).min())
    if room[0] >= g.limits.margin:
        return *commands[0], float(room[0])
    return None


def select(g, goal, route, guide, profile, prior, *, route_speed=None):
    from tinynav.rl.route_rejoin import route_coordinates

    if g.delay > 0.5 or not np.isfinite(prior).all() or route is None or len(route) < 2:
        return 0.0, 0.0, g.now
    if g.limits.margin <= g.now < g.limits.margin + .5 * g.resolution:
        escape = retreat(g, profile, prior)
        if escape is not None:
            return escape
    projection = route_coordinates(route, g.pose[:2])
    if projection is None:
        return 0.0, 0.0, g.now
    pts, s, at, off = projection
    ahead = np.array([np.interp(min(at + 0.70, s[-1]), s, pts[:, k]) for k in (0, 1)])
    bearing = np.arctan2(*(ahead - g.pose[:2])[::-1])
    cap = np.clip(guide[0] if route_speed is None else route_speed, 0., g.limits.max_forward)
    speeds = sorted({0., -.25, cap, *(v for v in (.25, .35, .45) if v < cap)})
    cmds = g.limits.executable(
        [(min(guide[0], cap), guide[1])]
        + [(v, w) for v in speeds for w in (0., -.25, .3, -.5, .55, -.75, .75)]
    )
    cmds = np.unique(cmds[cmds[:, 0] <= cap], axis=0)
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
