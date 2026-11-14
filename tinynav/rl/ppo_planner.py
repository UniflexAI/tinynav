"""RL local planner used by planning_node when TINYNAV_PLANNER=rl.

Builds the same observation the policy saw in NavSim (BEV + own trail + goal/velocity/last
action), runs the recurrent policy and returns (v, w). A guard, independent of what the policy
learned, keeps the RL footprint (robot_specs.RL_FOOTPRINT, the one used in training) at least
GUARD_MARGIN from obstacles over GUARD_HORIZON: it checks the swept area of the commanded motion,
including turning in place, and falls back to the first safe alternative or a stop. A robot that
stays put while being commanded to move is declared stuck and stopped until the next goal.
"""
import numpy as np

from tinynav.core.robot_specs import RL_FOOTPRINT, ROBOT_CONFIG
from tinynav.rl.bev import GRID_RES, TRAIL_LEN, camera_yaw, make_bev, to_robot_frame, trail_channel

# Action scaling shared with NavSim (which imports these): action[0] in [-1, 1] maps to
# [V_MIN, V_MAX] m/s with 0 -> 0, action[1] to [-W_MAX, W_MAX] rad/s. Fixed to the training values,
# not the running robot's limits (cmd_vel_control clamps to those). Kept torch-free for the robot.
V_MAX, V_MIN, W_MAX = 1.0, -0.3, 0.75
ROLLOUT_DT, ROLLOUT_STEPS = 0.1, 20  # 2s path for visualization / path-following controllers
GUARD_HORIZON = 1.0
# m between the footprint edge and the obstacle cell surface. Not relying on the obstacle mask being
# dilated: planning_node dilates by one cell, the Pilot's planner does not.
GUARD_MARGIN = 0.10
CREEP_V = 0.2  # m/s, forward escape when turning in place is blocked (e.g. backed against a wall)
STUCK_WINDOW, STUCK_MOVE, STUCK_TURN = 10.0, 0.2, 0.3  # s, m, rad
MAX_RECOVERY_ATTEMPTS = 5
RECOVERY_RESPONSE_S = 0.8
RECOVERY_MOVE, RECOVERY_TURN = 0.06, 0.12
RECOVERY_PHASES = ((0.2, 0.6, 0.5), (-0.1, 0.6, 1.0), (-0.2, 0.3, 1.0), (-0.2, 0.0, 0.5))
ESCAPE_TOL = 0.01  # m an escape move may lose to ESDF noise; never below 0 (touching the obstacle)


from tinynav.rl.rl_planner import footprint_clearance, rollout, safe_command


def route_guided_command(route, center, yaw, v, w):
    """Use the route's short-range heading and curvature before the ESDF guard."""
    if route is None or len(route) < 3:
        return v, w
    delta = np.diff(route, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    valid = lengths > 1e-4
    starts, delta, lengths = route[:-1][valid], delta[valid], lengths[valid]
    if not len(lengths):
        return v, w
    fraction = np.clip(((center - starts) * delta).sum(axis=1) / lengths**2, 0, 1)
    offsets = np.linalg.norm(starts + fraction[:, None] * delta - center, axis=1)
    index = int(np.argmin(offsets))
    if offsets[index] > .8:
        return v, w
    distance = np.r_[0., np.cumsum(lengths)]
    here = distance[index] + fraction[index] * lengths[index]
    remaining = distance[-1] - here
    if remaining < .2:
        return v, w
    points = np.vstack([starts, route[-1]])
    sample = np.minimum(here + np.array([0., .55, .85, 1.65]), distance[-1])
    preview = np.column_stack([np.interp(sample, distance, points[:, axis]) for axis in range(2)])
    tangent = preview[1] - preview[0]
    tangent_heading = np.arctan2(tangent[1], tangent[0])
    heading_error = np.angle(np.exp(1j * (tangent_heading - yaw)))
    forward = tangent / np.linalg.norm(tangent)
    offset = preview[0] - center
    lateral = forward[0] * offset[1] - forward[1] * offset[0]
    heading_error += np.arctan2(lateral, .65)

    far = preview[3] - preview[2]
    turn_ahead = np.arctan2(far[1], far[0]) - tangent_heading
    curve = abs(np.angle(np.exp(1j * turn_ahead))) if np.linalg.norm(far) > .12 else 0.
    cap = float(np.clip(.55 - .47 * max(curve, abs(heading_error)), .16, .55))
    speed = min(max(v, .16), cap) if abs(heading_error) < .9 else 0.
    turn = np.clip(1.5 * heading_error, -W_MAX, W_MAX)
    return speed, float(turn)


def progress_command(center, yaw, goal, desired_turn, esdf, origin):
    """Find a short forward motion with the same swept-footprint clearance rule."""
    relative = to_robot_frame(goal, center, yaw)
    if relative[0] <= abs(relative[1]) * .5:
        return None
    distance = np.linalg.norm(goal - center)
    now = footprint_clearance(rollout(center, yaw, 0., 0., steps=0), esdf, origin, GRID_RES)[0]
    minimum = GUARD_MARGIN if now >= GUARD_MARGIN else max(0., now - ESCAPE_TOL)
    steps = round(GUARD_HORIZON / ROLLOUT_DT)
    options = [(v, w) for v in (.04, .06, .08, .12, .16, .20) for w in (-.6, -.5, -.4, -.3, -.2, -.1, 0., .1, .2, .3, .4, .5, .6)]
    poses = np.asarray([rollout(center, yaw, v, w, steps=steps) for v, w in options])
    clearance = footprint_clearance(poses.reshape(-1, 3), esdf, origin, GRID_RES)
    clearance = clearance.reshape(len(options), steps + 1)[:, 1:].min(axis=1)
    gain = distance - np.linalg.norm(goal - poses[:, -1, :2], axis=1)
    turn = np.asarray(options)[:, 1]
    # Reward ending the rollout with room to continue, beyond the swept minimum.
    end_clearance = footprint_clearance(poses[:, -1], esdf, origin, GRID_RES)
    score = gain + .65 * np.minimum(end_clearance, .25) - .05 * np.abs(turn - desired_turn)
    score[(clearance < minimum) | (gain < .025)] = -np.inf
    selected = int(np.argmax(score))
    return options[selected] if np.isfinite(score[selected]) else None


def search_forward_command(center, yaw, goal, guided, current, policy_speed, esdf, origin):
    """Search faster arcs with the same one-second swept clearance margin."""
    gv, gw = guided
    if gv < .05:
        return None
    top = min(.55, max(gv, policy_speed))
    speeds = sorted({min(top, v) for v in (.04, .08, .12, .16, .20, .25, .30, .35, .40, .45, .50, .55)})
    turns = sorted({gw, *np.linspace(-W_MAX, W_MAX, 11)})
    commands = np.array([(v, w) for v in speeds for w in turns])
    commands = np.vstack((current, commands))
    v, w = commands.T
    t = ROLLOUT_DT * np.arange(round(GUARD_HORIZON / ROLLOUT_DT) + 1)
    heading = yaw + w[:, None] * t
    direction = np.stack((np.cos(heading[:, 1:]), np.sin(heading[:, 1:])), axis=-1)
    offset = np.cumsum(v[:, None, None] * ROLLOUT_DT * direction, axis=1)
    xy = np.concatenate((np.broadcast_to(center, (len(v), 1, 2)), center + offset), axis=1)
    poses = np.concatenate((xy, heading[..., None]), axis=2)
    clearance = footprint_clearance(poses.reshape(-1, 3), esdf, origin, GRID_RES).reshape(len(v), len(t))
    gain = np.linalg.norm(goal - center) - np.linalg.norm(goal - xy[:, -1], axis=1)
    # Look beyond the immediate distance gain when an arc leaves the body misaligned.
    target_yaw = np.arctan2(goal[1] - xy[:, -1, 1], goal[0] - xy[:, -1, 0])
    alignment = np.abs(np.angle(np.exp(1j * (target_yaw - heading[:, -1]))))
    score = gain - .06 * np.abs(w - gw) - .23 * alignment
    score[(clearance.min(axis=1) < GUARD_MARGIN) | (gain < .025)] = -np.inf
    choice = int(np.argmax(score))
    if choice and np.isfinite(score[choice]) and score[choice] > score[0] + .02:
        return tuple(commands[choice])
    return None


class RLPlanner:
    GUARD_REASONS = ("ok", "turn in place", "turn the other way", "creep forward", "creep forward turning",
                     "creep forward turning the other way", "back off", "stop")

    def __init__(self, policy):
        self.policy = policy
        self.reset()

    def reset(self):
        """New goal: clear the GRU memory, own trail, velocity estimate and stuck state."""
        self.policy.reset()
        self.hist, self.last = [], None
        self.vel = np.zeros(2, np.float32)
        self.last_action = np.zeros(2, np.float32)
        self.motion, self.stuck = [], False  # (stamp, x, y, yaw, commanded) for stuck detection
        self.stuck_count = 0
        self.recovery = None
        self.recovery_start = None
        self.failed_recoveries = set()

    def _checked_recovery(self, center, yaw, stamp, esdf, origin, phases):
        if phases in self.failed_recoveries:
            return None
        now = footprint_clearance(rollout(center, yaw, 0.0, 0.0, steps=0), esdf, origin, GRID_RES)[0]
        if now < 0.0:
            return None
        # Apply the normal escape tolerance once, relative to the initial pose.
        minimum = GUARD_MARGIN if now >= GUARD_MARGIN else max(0.0, now - ESCAPE_TOL)
        pose = np.array([*center, yaw])
        for v, w, duration in phases:
            poses = rollout(pose[:2], pose[2], v, w, steps=round(duration / ROLLOUT_DT))
            if footprint_clearance(poses, esdf, origin, GRID_RES).min() < minimum:
                return None
            pose = poses[-1]
        return stamp, phases, minimum

    def _reverse_recovery(self, center, yaw, stamp, esdf, origin, goal_xy):
        forward, left = to_robot_frame(goal_xy, center, yaw)
        distance = np.hypot(forward, left)
        if distance < 0.6 or forward > -0.7 * distance:
            return None
        # Align the rear with the target, opposite the bearing of the target from the front.
        w = -0.3 if left >= 0.0 else 0.3
        return self._checked_recovery(center, yaw, stamp, esdf, origin, ((-0.2, w, 1.0),) * 2)

    def _forward_turn_recovery(self, center, yaw, stamp, esdf, origin, goal_xy):
        forward, left = to_robot_frame(goal_xy, center, yaw)
        bearing = float(np.arctan2(left, forward - 0.4))
        duration = min(4.0, abs(bearing) / 0.6)
        turn = 0.6 if bearing >= 0.0 else -0.6
        phases = ((0.2, 0.0, 1.0),) * 2
        whole = int(duration)
        phases += ((0.0, turn, 1.0),) * whole
        remainder = round(duration - whole, 1)
        if remainder > 0.0:
            phases += ((0.0, turn, remainder),)
        return self._checked_recovery(center, yaw, stamp, esdf, origin, phases)

    def _start_recovery(self, center, yaw, stamp, esdf, origin, preferred, goal_xy):
        if self.failed_recoveries:
            alternate = self._forward_turn_recovery(center, yaw, stamp, esdf, origin, goal_xy)
            if alternate is not None:
                return alternate
        reverse = self._reverse_recovery(center, yaw, stamp, esdf, origin, goal_xy)
        if reverse is not None:
            return reverse
        for direction in (preferred, -preferred):
            phases = tuple((v, direction * w, duration) for v, w, duration in RECOVERY_PHASES)
            recovery = self._checked_recovery(center, yaw, stamp, esdf, origin, phases)
            if recovery is not None:
                return recovery
        return self._forward_turn_recovery(center, yaw, stamp, esdf, origin, goal_xy)

    def _activate_recovery(self, recovery, center, yaw):
        self.recovery = recovery
        self.recovery_start = (center.copy(), yaw)
        self.stuck_count += 1

    def _recovery_moved(self, center, yaw):
        start, heading = self.recovery_start
        return (np.linalg.norm(center - start) >= RECOVERY_MOVE
                or abs(np.angle(np.exp(1j * (yaw - heading)))) >= RECOVERY_TURN)

    def _recovery_command(self, center, yaw, stamp, esdf, origin):
        started, phases, minimum = self.recovery
        elapsed = stamp - started
        for index, (v, w, duration) in enumerate(phases):
            if elapsed < duration:
                steps = max(1, int(np.ceil((duration - elapsed) / ROLLOUT_DT - 1e-8)))
                poses = rollout(center, yaw, v, w, steps=steps)
                clearance = footprint_clearance(poses, esdf, origin, GRID_RES)
                if clearance.min() >= minimum:
                    return v, w, f"recovery {index + 1}/{len(phases)}"
                return 0.0, 0.0, "recovery blocked"
            elapsed -= duration
        return None

    def _check_stuck(self, stamp, center, yaw, commanded):
        self.motion = [m for m in self.motion if stamp - m[0] <= STUCK_WINDOW] + [(stamp, center[0], center[1], yaw, commanded)]
        if stamp - self.motion[0][0] < 0.9 * STUCK_WINDOW or not all(m[4] for m in self.motion):
            return False
        moved = np.hypot(center[0] - self.motion[0][1], center[1] - self.motion[0][2])
        turned = abs(np.angle(np.exp(1j * (yaw - self.motion[0][3]))))
        return moved < STUCK_MOVE and turned < STUCK_TURN

    def step(self, grid, origin, T, stamp, goal_xy, esdf):
        """grid/origin: planning occupancy grid; T: camera->world; esdf: 2D ESDF on the same grid.
        Returns (v, w, rollout poses, policy BEV, status) with status one of GUARD_REASONS or "stuck"."""
        center = (T[:3, 3] - T[:3, :3] @ ROBOT_CONFIG.cam_offset_3d)[:2]
        yaw = camera_yaw(T)
        if self.last is not None:
            dt = max(stamp - self.last[0], 1e-3)
            raw = np.array([to_robot_frame(center, self.last[1], self.last[2])[0] / dt,
                            np.angle(np.exp(1j * (yaw - self.last[2]))) / dt], np.float32)
            self.vel = 0.5 * self.vel + 0.5 * raw  # odometry differences are noisy at ~7Hz
        self.last = (stamp, center, yaw)
        self.hist = [center] + self.hist[:TRAIL_LEN - 1]

        trail = trail_channel(np.array(self.hist), np.arange(len(self.hist)), center, yaw)
        bev = np.concatenate([make_bev(grid, origin, GRID_RES, center, yaw, ref_z=T[2, 3]), trail[None]])
        vec = np.concatenate([to_robot_frame(goal_xy, center, yaw) / 5.0,
                              self.vel / np.array([V_MAX, W_MAX], np.float32), self.last_action]).astype(np.float32)
        a = np.clip(np.asarray(self.policy.infer(bev, vec), np.float32), -1, 1)
        self.last_action = a
        v = float(a[0] * V_MAX if a[0] >= 0 else -a[0] * V_MIN)
        w = float(a[1] * W_MAX)
        policy_speed = v
        v, w = route_guided_command(getattr(self, "route_xy", None), center, yaw, v, w)
        self.guided_command = (v, w)
        requested = abs(v) > 0.05 or abs(w) > 0.1

        requested_v, requested_w = v, w
        v, w, reason = safe_command(center, yaw, v, w, esdf, origin, GRID_RES)
        status = self.GUARD_REASONS[reason]
        if reason != 0 and requested_v > .05:
            alternate = progress_command(center, yaw, goal_xy, requested_w, esdf, origin)
            if alternate is not None:
                v, w = alternate
                status = "progress fallback"
        alternate = search_forward_command(
            center, yaw, goal_xy, self.guided_command, (v, w), policy_speed, esdf, origin,
        )
        if alternate is not None:
            v, w = alternate
            status = "forward search"
        # Requested motion with no measured response can indicate an unseen obstacle.
        preferred = -1.0 if a[1] >= 0.0 else 1.0
        if self.recovery is None and not self.stuck and self._check_stuck(stamp, center, yaw, requested):
            recovery = self._start_recovery(center, yaw, stamp, esdf, origin, preferred, goal_xy)
            if recovery is None or self.stuck_count >= MAX_RECOVERY_ATTEMPTS:
                self.stuck = True
            else:
                self._activate_recovery(recovery, center, yaw)
        if self.recovery is not None:
            command = self._recovery_command(center, yaw, stamp, esdf, origin)
            moved = self._recovery_moved(center, yaw)
            unresponsive = not moved and (command is None or stamp - self.recovery[0] >= RECOVERY_RESPONSE_S)
            blocked = command is not None and command[2] == "recovery blocked"
            if unresponsive or blocked:
                self.failed_recoveries.add(self.recovery[1])
                alternate = self._start_recovery(center, yaw, stamp, esdf, origin, preferred, goal_xy)
                if alternate is None or self.stuck_count >= MAX_RECOVERY_ATTEMPTS:
                    self.recovery = None
                    self.stuck = True
                else:
                    self._activate_recovery(alternate, center, yaw)
                v, w, status = 0.0, 0.0, "recovery blocked" if blocked else "recovery no motion"
            elif command is not None:
                v, w, status = command
            else:
                phases = self.recovery[1]
                # The original four-phase maneuver ends with straight reverse.
                continuation = None
                if len(phases) == len(RECOVERY_PHASES) and phases[-1] == RECOVERY_PHASES[-1]:
                    continuation = self._reverse_recovery(center, yaw, stamp, esdf, origin, goal_xy)
                if continuation is not None:
                    self.recovery = continuation
                    self.recovery_start = (center.copy(), yaw)
                    v, w, status = self._recovery_command(center, yaw, stamp, esdf, origin)
                else:
                    self.recovery = None
                    self.motion = []
                    self.failed_recoveries.clear()
                    self.policy.reset()
                    self.last_action.fill(0.0)
                    v, w, status = 0.0, 0.0, "recovery complete"
        if self.stuck:
            v, w, status = 0.0, 0.0, "stuck"
        return v, w, rollout(center, yaw, v, w), bev, status
