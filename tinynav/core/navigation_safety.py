"""Shared executable-motion clearance checks for the DWA and PPO entry points."""
from dataclasses import dataclass
import math
import os

import numpy as np

from tinynav.core.robot_specs import RLFootprint, RL_FOOTPRINT, ROBOT_CONFIG
from tinynav.rl.rl_planner import _edge_points, footprint_clearance


@dataclass(frozen=True)
class NavigationLimits:
    front: float = RL_FOOTPRINT.front
    rear: float = RL_FOOTPRINT.rear
    half_width: float = RL_FOOTPRINT.half_width
    control_x: float = 0.0
    control_y: float = 0.0
    max_forward: float = .55
    max_reverse: float = .30
    max_turn: float = .75
    linear_acceleration: float = .60
    margin: float = .10
    period: float = 1 / 12

    def __post_init__(self):
        values = np.asarray(list(vars(self).values()))
        if not np.isfinite(values).all():
            raise ValueError("Navigation limits must be finite")
        positive = [self.front, self.rear, self.half_width, self.max_forward, self.max_reverse,
                    self.max_turn, self.linear_acceleration, self.margin, self.period]
        if min(positive) <= 0:
            raise ValueError("Navigation extents, speeds, and clearance must be positive")
        if min(self.max_forward, self.max_reverse) < ROBOT_CONFIG.min_linear_vel or self.max_turn < ROBOT_CONFIG.min_angular_vel:
            raise ValueError("Navigation limits are below the robot's executable minimum speeds")

    @classmethod
    def from_environment(cls):
        default = cls()
        fields = ("front", "rear", "half_width", "control_x", "control_y")
        geometry = {name: float(os.environ.get(f"TINYNAV_NAV_{name.upper()}_M",
                     os.environ.get(f"TINYNAV_PPO_{name.upper()}_M", getattr(default, name)))) for name in fields}
        speeds = {name: float(os.environ.get(variable, getattr(default, name))) for name, variable in (
            ("max_forward", "TINYNAV_NAV_MAX_FORWARD_MPS"),
            ("max_reverse", "TINYNAV_NAV_MAX_REVERSE_MPS"),
            ("max_turn", "TINYNAV_NAV_MAX_TURN_RPS"))}
        if not np.isfinite(list(speeds.values())).all():
            raise ValueError("Navigation speed limits must be finite")
        speeds["max_forward"] = min(ROBOT_CONFIG.max_linear_vel, speeds["max_forward"])
        speeds["max_reverse"] = min(ROBOT_CONFIG.max_linear_vel, speeds["max_reverse"])
        speeds["max_turn"] = min(ROBOT_CONFIG.max_angular_vel, speeds["max_turn"])
        return cls(**geometry, **speeds)

    def executable(self, commands):
        out = np.asarray(commands, dtype=float).reshape(-1, 2).copy()
        out[~np.isfinite(out).all(axis=1)] = 0.
        out[:, 0] = np.clip(out[:, 0], -self.max_reverse, self.max_forward)
        out[:, 1] = np.clip(out[:, 1], -self.max_turn, self.max_turn)
        out[(out[:, 0] > 0) & (out[:, 0] < ROBOT_CONFIG.min_linear_vel), 0] = ROBOT_CONFIG.min_linear_vel
        out[(out[:, 0] < 0) & (out[:, 0] > -ROBOT_CONFIG.min_linear_vel), 0] = -ROBOT_CONFIG.min_linear_vel
        out[np.abs(out[:, 1]) < ROBOT_CONFIG.min_angular_vel, 1] = 0.
        return out


def advance(pose, speed, turn, duration):
    rotation = turn * duration
    if np.ndim(pose) == 1 and np.ndim(speed) == 0 and np.ndim(turn) == 0:
        chord = speed * duration if abs(rotation) < 1e-10 else speed * duration * math.sin(rotation / 2) / (rotation / 2)
        mid = pose[2] + rotation / 2
        return np.array([pose[0] + chord * math.cos(mid), pose[1] + chord * math.sin(mid), pose[2] + rotation])
    chord = speed * duration * np.sinc(rotation / (2 * np.pi))
    mid = pose[..., 2] + rotation / 2
    return pose + np.stack((chord * np.cos(mid), chord * np.sin(mid),
                            np.broadcast_to(rotation, np.shape(mid))), axis=-1)


def executable_speed(current, requested, dt, limits):
    speed = current + np.clip(requested - current, -limits.linear_acceleration * dt,
                              limits.linear_acceleration * dt)
    tiny = (speed > 0) & (speed < ROBOT_CONFIG.min_linear_vel)
    speed = np.where(tiny, np.where(requested >= ROBOT_CONFIG.min_linear_vel,
                                   ROBOT_CONFIG.min_linear_vel, 0.), speed)
    tiny_reverse = (speed < 0) & (speed > -ROBOT_CONFIG.min_linear_vel)
    return np.where(tiny_reverse, np.where(requested <= -ROBOT_CONFIG.min_linear_vel,
                                          -ROBOT_CONFIG.min_linear_vel, 0.), speed)


class NavigationGuard:
    def __init__(self, limits, pose, command, esdf, origin, resolution, *, delay=0.0, stamp=0.0):
        self.limits = limits
        self.pose = np.asarray(pose, dtype=float)
        self.command = np.asarray(command, dtype=float)
        self.esdf, self.origin, self.resolution = esdf, np.asarray(origin)[:2], resolution
        self.delay = max(0., float(delay))
        self.stamp = stamp
        self.edge = _edge_points(footprint=RLFootprint(limits.front, limits.rear, limits.half_width))
        self.now = float(self.clearance(self.pose[None])[0])
        self.minimum = limits.margin if self.now >= limits.margin else max(0., self.now - .005)
        self.last_clearance = self.now

    def clearance(self, poses):
        c, s = np.cos(poses[:, 2:3]), np.sin(poses[:, 2:3])
        x = poses[:, :1] + c * self.edge[:, 0] - s * self.edge[:, 1]
        y = poses[:, 1:2] + s * self.edge[:, 0] + c * self.edge[:, 1]
        lower = self.origin + self.resolution / 2
        upper = self.origin + (np.array(self.esdf.shape) - .5) * self.resolution
        outside = ((x < lower[0]) | (x > upper[0]) | (y < lower[1]) | (y > upper[1])).any(axis=1)
        result = np.full(len(poses), -np.inf)
        inside = ~outside
        if inside.any():
            result[inside] = footprint_clearance(poses[inside], self.esdf, self.origin, self.resolution, edge=self.edge)
        return result

    def trajectories(self, commands, duration=1.0, braking=True):
        commands = self.limits.executable(commands)
        n = len(commands)
        current = np.broadcast_to(self.pose, (n, 3)).copy()
        speed = np.full(n, self.command[0])
        result = [current.copy()]
        # Delay covers pose age and the next controller tick. Preserve the command
        # already sent to the base before applying the candidate's acceleration ramp.
        prediction_delay = min(self.delay, .50)
        delay_steps = int(np.ceil(prediction_delay / .025))
        for _ in range(delay_steps):
            current = advance(current, speed, self.command[1], prediction_delay / delay_steps)
            result.append(current.copy())
        ticks = int(np.ceil(duration / self.limits.period))
        for _ in range(ticks):
            speed = executable_speed(speed, commands[:, 0], self.limits.period, self.limits)
            for _ in range(4):
                current = advance(current, speed, commands[:, 1], self.limits.period / 4)
                result.append(current.copy())
        # Keep enough room to brake after the checked command interval.
        braking_ticks = int(np.ceil(np.max(np.abs(speed)) / self.limits.linear_acceleration / self.limits.period)) if braking else 0
        for _ in range(braking_ticks):
            speed = executable_speed(speed, np.zeros(n), self.limits.period, self.limits)
            for _ in range(4):
                current = advance(current, speed, 0., self.limits.period / 4)
                result.append(current.copy())
        return np.stack(result, axis=1)

    def evaluate(self, commands, duration=1.0):
        commands = self.limits.executable(commands)
        paths = self.trajectories(commands, duration)
        clear = self.clearance(paths.reshape(-1, 3)).reshape(paths.shape[:2])
        minimum = clear.min(axis=1)
        allowed = (minimum >= self.minimum) & (self.delay <= .50)
        if self.now < self.limits.margin:
            # Existing under-clearance states share the same monotonic escape rule.
            # A moving candidate must recover room instead of skimming the obstacle.
            moving = np.any(np.abs(commands) > 1e-6, axis=1)
            allowed &= ~moving | (clear[:, -1] >= min(self.limits.margin, self.now + .015))
        return allowed, minimum, paths

    def select(self, requested, goal, preferred_turn=0.0):
        alternatives = np.array([requested, (0., requested[1]), (0., -requested[1]),
                                 (.10, 0.), (.10, -.20), (.10, .20), (-.10, 0.),
                                 (-.10, -.30), (-.10, .30), (-.20, 0.), (0., 0.)])
        alternatives = self.limits.executable(alternatives)
        valid, room, paths = self.evaluate(alternatives)
        if valid[0]:
            self.last_clearance = float(room[0])
            return *alternatives[0], "ok"
        gain = np.linalg.norm(goal - self.pose[:2]) - np.linalg.norm(goal - paths[:, -1, :2], axis=1)
        score = gain + .3 * np.minimum(room, .25) - .025 * np.abs(alternatives[:, 1])
        score -= .04 * (alternatives[:, 1] * preferred_turn < 0.)
        if self.now < self.limits.margin:
            score -= .10 * np.all(np.abs(alternatives) < 1e-6, axis=1)
        score[~valid] = -np.inf
        i = int(np.argmax(score))
        if np.isfinite(score[i]):
            self.last_clearance = float(room[i])
            return *alternatives[i], "shared clearance fallback"
        self.last_clearance = float(room[-1])
        return 0., 0., "clearance blocked"


class MotionResponse:
    """Check ordinary commands against measured translation or rotation."""
    def __init__(self):
        self.start = None
        self.command = np.zeros(2)

    def stalled(self, stamp, pose):
        if not np.any(self.command):
            self.start = None
            return False
        if self.start is None:
            self.start = stamp, np.array(pose)
            return False
        elapsed = stamp - self.start[0]
        if elapsed < .8:
            return False
        displacement = np.linalg.norm(pose[:2] - self.start[1][:2])
        rotation = abs(np.angle(np.exp(1j * (pose[2] - self.start[1][2]))))
        moved = abs(self.command[0]) > .01 and displacement >= max(.015, min(.06, abs(self.command[0]) * elapsed * .30))
        turned = abs(self.command[1]) > .01 and rotation >= max(.04, min(.12, abs(self.command[1]) * elapsed * .30))
        self.start = stamp, np.array(pose)
        return not (moved or turned)

    def issued(self, command):
        command = np.array(command)
        if not np.any(command):
            self.start = None
        self.command = command
