"""Check a low-speed route rejoin before replacing a large policy turn."""
import numpy as np

from tinynav.rl.bev import GRID_RES
from tinynav.rl.rl_planner import footprint_clearance

FORWARD_SPEED = .25  # First consistently executing forward start in the Go2 calibration.
TURN_LIMIT = .20
MARGIN = .10
MAX_AGE = .75


def project(points, segments, lengths, distance, xy):
    fractions = np.clip(((xy - points) * segments).sum(axis=1) / lengths**2, 0, 1)
    nearest = points + fractions[:, None] * segments
    lateral = np.linalg.norm(nearest - xy, axis=1)
    idx = np.argmin(lateral)
    return distance[idx] + lengths[idx] * fractions[idx], lateral[idx]


def arc(pose, linear, angular, time):
    theta = pose[2] + angular * time
    half = angular * time / 2
    chord = linear * time * np.sinc(half / np.pi)
    return np.column_stack((pose[0] + chord * np.cos(pose[2] + half),
                            pose[1] + chord * np.sin(pose[2] + half), theta))


def candidate_path(pose, previous, turn, delay):
    pending = arc(pose, *previous, np.linspace(0., delay, 1 + int(np.ceil(delay * 48))))
    held = arc(pending[-1], FORWARD_SPEED, turn, np.linspace(0., 1., 49))
    # Brake at 0.30 m/s². Check both continuing the steering arc and losing yaw response.
    duration = FORWARD_SPEED / .30
    time = np.linspace(0., duration, 1 + int(np.ceil(duration * 48)))
    mean_speed = FORWARD_SPEED - .15 * time
    braking = arc(held[-1], mean_speed, turn, time)
    braking_straight = arc(held[-1], mean_speed, 0., time)
    return np.concatenate((pending, held, braking, braking_straight))


def has_clearance(poses, esdf, origin, edge):
    radius = np.linalg.norm(edge, axis=1).max()
    lower = np.asarray(origin[:2]) + radius + GRID_RES / 2
    upper = np.asarray(origin[:2]) + GRID_RES*np.asarray(esdf.shape)-radius-GRID_RES / 2
    if np.any(poses[:, :2] < lower) or np.any(poses[:, :2] > upper):
        return False
    room = footprint_clearance(poses, esdf, origin, GRID_RES, edge=edge)
    return np.isfinite(room).all() and room.min() >= MARGIN


def can_link(to, points, segments, lengths, distance, at, offset, esdf, origin, edge):
    along, departure = project(points, segments, lengths, distance, to[:2])
    if along - at >= .04 and departure <= min(.35, offset + .08):
        return True
    # A short detour or loop may put the closest leg behind the robot. Require
    # a swept connection to a later point, rather than treating proximity as a rejoin.
    for index, waypoint in enumerate(points):
        if not .45 <= distance[index]-at <= 2.25:
            continue
        dx, dy = waypoint - to[:2]
        length = np.hypot(dx, dy)
        if not .05 <= length <= 1.05:
            continue
        bearing = np.arctan2(dy, dx)
        turn = np.angle(np.exp(1j * (bearing - to[2])))
        deviation = np.angle(np.exp(1j * (np.arctan2(*segments[index, ::-1]) - bearing)))
        if abs(turn) > .5 or abs(deviation) > .85:
            continue
        rotation = arc(to, 0., turn, np.linspace(0., 1., 13))
        linked = arc(rotation[-1], length, 0., np.linspace(0., 1., 1 + int(np.ceil(length/.025))))
        if has_clearance(np.concatenate((rotation, linked)), esdf, origin, edge):
            return True
    return False


def forward_rejoin(route, center, yaw, goal, current, feedback, velocity, age, esdf, origin, edge, *, proposal=None):
    if route is None or len(route) < 3 or feedback is None or not 0 <= age <= MAX_AGE:
        return None
    delta = goal - center
    remaining = np.linalg.norm(delta)
    heading = np.angle(np.exp(1j * (np.arctan2(delta[1], delta[0]) - yaw)))
    if remaining < .75 or abs(heading) > np.deg2rad(20.) or abs(current[1]) < .35:
        return None
    pose = np.r_[center, yaw]
    if not np.isfinite(np.r_[pose, feedback, velocity, route.ravel()]).all():
        return None
    segments = np.diff(route, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    keep = lengths > 1e-4
    points, segments, lengths = route[:-1][keep], segments[keep], lengths[keep]
    if not len(lengths):
        return None
    distance = np.r_[0., np.cumsum(lengths)]
    along, offset = project(points, segments, lengths, distance, center)
    if offset > .50 or distance[-1] - along < .45:
        return None

    possibilities = []
    priors = (np.asarray(feedback), np.asarray(velocity), np.zeros(2))
    turns = (0., -.10, .10, -TURN_LIMIT, TURN_LIMIT)
    preferred = None
    if proposal is not None and np.isfinite(proposal).all() and proposal[0] >= FORWARD_SPEED:
        preferred = float(np.clip(proposal[1], -TURN_LIMIT, TURN_LIMIT))
        turns = (preferred,) + turns
    for turn in dict.fromkeys(turns):
        end = arc(pose, FORWARD_SPEED, turn, np.array([1.]))[0]
        after, departure = project(points, segments, lengths, distance, end[:2])
        gain = remaining - np.linalg.norm(goal - end[:2])
        if gain < .06 or not can_link(end, points, segments, lengths, distance,
                                      along, offset, esdf, origin, edge):
            continue
        # Safety must hold both if Go2 obeys the small yaw command and if it stalls.
        sweeps = [candidate_path(pose, prior, w, delay) for prior in priors
                  for w in {0.,turn} for delay in (.32,max(.50,age))]
        if not all(has_clearance(path, esdf, origin, edge) for path in sweeps):
            continue
        if preferred is not None and turn == preferred:
            return FORWARD_SPEED, turn
        score = gain + .4 * (offset - departure) - .20 * abs(turn)
        possibilities.append((score, turn))
    if not possibilities:
        return None
    return FORWARD_SPEED, max(possibilities)[1]
