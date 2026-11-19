"""Checked forward merges into a downstream section of the current route."""
import numpy as np

from tinynav.core.navigation_safety import advance, executable_speed


def route_coordinates(route, point):
    delta = np.diff(route, axis=0)
    length = np.linalg.norm(delta, axis=1)
    valid = length > 1e-4
    starts, delta, length = route[:-1][valid], delta[valid], length[valid]
    if not len(length):
        return None
    fraction = np.clip(((point - starts) * delta).sum(axis=1) / length**2, 0., 1.)
    projected = starts + fraction[:, None] * delta
    offset = np.linalg.norm(projected - point, axis=1)
    index = int(np.argmin(offset))
    distance = np.r_[0., np.cumsum(length)]
    points = np.vstack((starts, route[-1]))
    here = distance[index] + fraction[index] * length[index]
    return points, distance, here, float(offset[index])


def curve_command(curve, pose, speed):
    near = int(np.argmin(np.linalg.norm(curve - pose[:2], axis=1)))
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    ahead = min(distance[-1], distance[near] + .30)
    target = np.array([np.interp(ahead, distance, curve[:, axis]) for axis in range(2)])
    delta = target - pose[:2]
    bearing = np.angle(np.exp(1j * (np.arctan2(delta[1], delta[0]) - pose[2])))
    curvature = 2 * np.sin(bearing) / max(.12, np.linalg.norm(delta))
    velocity = min(speed, .55 / max(abs(curvature), 1e-6))
    velocity = max(.10, velocity)
    turn = float(np.clip(velocity * curvature, -.60, .60))
    return velocity, 0. if abs(turn) < .10 else turn


class RouteRejoin:
    def __init__(self):
        self.curve = None
        self.target = None
        self.clearance = None

    def reset(self):
        self.curve = self.target = self.clearance = None

    def checked_curve(self, guard, curve, speed):
        limits = guard.limits
        speed = min(speed, limits.max_forward)
        pose = advance(guard.pose, guard.command[0], guard.command[1], guard.delay)
        first = tuple(limits.executable([curve_command(curve, pose, speed)])[0])
        held, _, _ = guard.evaluate(np.array([first]))
        if not held[0]:
            return None
        prefix = guard.trajectories(np.array([first]), duration=limits.period, braking=False)
        poses = [prefix[0]]
        pose = poses[-1][-1]
        current_speed = float(executable_speed(guard.command[0], first[0], limits.period, limits))
        target_heading = np.arctan2(*(curve[-1] - curve[-2])[::-1])
        reached = False
        for _ in range(180):
            command = limits.executable([curve_command(curve, pose, speed)])[0]
            current_speed = float(executable_speed(current_speed, command[0], limits.period, limits))
            for _ in range(4):
                pose = advance(pose, current_speed, command[1], limits.period / 4)
                poses.append(pose[None])
            alignment = abs(np.angle(np.exp(1j * (pose[2] - target_heading))))
            if np.linalg.norm(pose[:2] - curve[-1]) < .10 and alignment < .40:
                reached = True
                break
        if not reached:
            return None
        # Check a straight continuation after the merge, including the same body outline.
        for _ in range(24):
            pose = advance(pose, speed, 0., limits.period / 4)
            poses.append(pose[None])
        trajectory = np.vstack(poses)
        room = guard.clearance(trajectory)
        if room.min() < guard.minimum:
            return None
        if guard.now < limits.margin and room[-1] < min(limits.margin, guard.now + .015):
            return None
        return first, float(room.min()), trajectory

    def choose(self, guard, route, goal):
        if route is None or len(route) < 3 or guard.now < guard.limits.margin or guard.delay > .50:
            self.reset()
            return None
        execution_pose = advance(guard.pose, guard.command[0], guard.command[1], guard.delay)
        current = route_coordinates(route, execution_pose[:2])
        if current is None:
            self.reset()
            return None
        if self.curve is not None:
            points, distance, here, offset = current
            sample = np.minimum([here, here + .55], distance[-1])
            preview = np.column_stack([np.interp(sample, distance, points[:, k]) for k in range(2)])
            tangent = preview[1] - preview[0]
            tangent_length = np.linalg.norm(tangent)
            angle = np.angle(np.exp(1j * (np.arctan2(tangent[1], tangent[0]) - execution_pose[2])))
            delta = preview[0] - execution_pose[:2]
            lateral = (tangent[0] * delta[1] - tangent[1] * delta[0]) / max(tangent_length, 1e-6)
            if offset < .15 and tangent_length > .10 and abs(angle + np.arctan2(lateral, .65)) < .35:
                self.reset()
                return None
            projected = route_coordinates(route, self.target)
            if projected is not None and projected[3] < .20 and np.linalg.norm(execution_pose[:2] - self.target) > .15:
                near = int(np.argmin(np.linalg.norm(self.curve - execution_pose[:2], axis=1)))
                remaining = self.curve[near:]
                heading = np.arctan2(np.gradient(self.curve[:, 1]), np.gradient(self.curve[:, 0]))[near:]
                room = guard.clearance(np.column_stack((remaining, heading)))
                command = tuple(guard.limits.executable([curve_command(self.curve, execution_pose,
                                          min(.20, guard.limits.max_forward))])[0])
                allowed, immediate, _ = guard.evaluate([command])
                if allowed[0] and room.min() >= guard.minimum:
                    self.clearance = float(min(room.min(), immediate[0]))
                    return command
            self.reset()

        relative = goal - execution_pose[:2]
        bearing = np.angle(np.exp(1j * (np.arctan2(relative[1], relative[0]) - execution_pose[2])))
        if abs(bearing) > np.deg2rad(65):
            return None
        points, distance, here, _ = current
        positions = np.arange(here + .60, min(distance[-1], here + 3.01), .30)
        options = []
        direction = np.array([np.cos(execution_pose[2]), np.sin(execution_pose[2])])
        for along in positions:
            p3 = np.array([np.interp(along, distance, points[:, k]) for k in range(2)])
            end = np.array([np.interp(min(along + .30, distance[-1]), distance, points[:, k]) for k in range(2)])
            tangent = end - p3
            length = np.linalg.norm(p3 - execution_pose[:2])
            gain = np.linalg.norm(relative) - np.linalg.norm(goal - p3)
            if length < .40 or length > 2.5 or (p3 - execution_pose[:2]) @ direction < .25 or gain < .15 or np.linalg.norm(tangent) < .05:
                continue
            tangent /= np.linalg.norm(tangent)
            reach = min(.60, length * .40)
            p0 = execution_pose[:2]
            p1, p2 = p0 + reach * direction, p3 - reach * tangent
            t = np.linspace(0., 1., max(30, int(length / .025)))[:, None]
            curve = (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3
            checked = self.checked_curve(guard, curve, .20)
            if checked is not None:
                command, room, path = checked
                score = gain / max(length, .1) + .3 * min(room, .25) - .04 * abs(command[1])
                options.append((score, curve, p3, command, room))
        if not options:
            return None
        _, self.curve, self.target, command, self.clearance = max(options, key=lambda item: item[0])
        return command
