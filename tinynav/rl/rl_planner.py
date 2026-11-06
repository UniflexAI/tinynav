"""RL local planner used by planning_node when TINYNAV_PLANNER=rl.

Builds the same observation the policy saw in NavSim (BEV + own trail + goal/velocity/last
action), runs the recurrent policy and returns (v, w). A guard, independent of what the policy
learned, keeps the RL footprint (robot_specs.RL_FOOTPRINT, the one used in training) at least
GUARD_MARGIN from obstacles over GUARD_HORIZON: it checks the swept area of the commanded motion,
including turning in place, and falls back to the first safe alternative or a stop. A robot that
stays put while being commanded to move is declared stuck and stopped until the next goal.
"""
import asyncio
import os

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
STUCK_HOLD, STUCK_RETRIES = 3.0, 3  # stop this long, then retry with fresh GRU memory; give up after this many
ESCAPE_TOL = 0.01  # m an escape move may lose to ESDF noise; never below 0 (touching the obstacle)


class TRTPolicy:
    """Synchronous wrapper around models_trt.RLNavPolicyTRT for use inside ROS callbacks.
    TINYNAV_RL_ENGINE overrides the engine path, e.g. to go back to an older policy."""
    def __init__(self):
        from tinynav.core.models_trt import RLNavPolicyTRT
        engine = os.environ.get("TINYNAV_RL_ENGINE")
        self.trt = RLNavPolicyTRT(engine) if engine else RLNavPolicyTRT()
        self.loop = asyncio.new_event_loop()

    def reset(self):
        self.trt.reset()

    def infer(self, bev, vec):
        return self.loop.run_until_complete(self.trt.infer(bev.astype(np.float32), vec.astype(np.float32)))


def rollout(center, yaw, v, w, steps=ROLLOUT_STEPS, dt=ROLLOUT_DT):
    """Unicycle poses [steps + 1, 3] (x, y, yaw) for a constant command."""
    out = [(center[0], center[1], yaw)]
    for _ in range(steps):
        x, y, th = out[-1]
        th = th + w * dt
        out.append((x + v * dt * np.cos(th), y + v * dt * np.sin(th), th))
    return np.array(out)


def _edge_points(step=GRID_RES / 2):
    fl, rl, hw = RL_FOOTPRINT.front, RL_FOOTPRINT.rear, RL_FOOTPRINT.half_width
    f, s = np.arange(-rl, fl + 1e-6, step), np.arange(-hw, hw + 1e-6, step)
    return np.concatenate([np.stack([f, np.full_like(f, hw)], 1), np.stack([f, np.full_like(f, -hw)], 1),
                           np.stack([np.full_like(s, fl), s], 1), np.stack([np.full_like(s, -rl), s], 1)])


EDGE = _edge_points()


def footprint_clearance(poses, esdf, origin, res):
    """Per pose, the smallest distance from the footprint outline to an obstacle cell's surface (m).
    The ESDF holds center-to-center distances, so half a cell is subtracted; bilinear interpolation
    avoids the up-to-half-cell error of reading the 10cm grid at cell centers."""
    c, sn = np.cos(poses[:, 2:3]), np.sin(poses[:, 2:3])
    gx = (poses[:, :1] + c * EDGE[:, 0] - sn * EDGE[:, 1] - origin[0]) / res - 0.5
    gy = (poses[:, 1:2] + sn * EDGE[:, 0] + c * EDGE[:, 1] - origin[1]) / res - 0.5
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - x0, gy - y0
    def at(i, j):
        ok = (i >= 0) & (i < esdf.shape[0]) & (j >= 0) & (j < esdf.shape[1])
        return np.where(ok, esdf[np.clip(i, 0, esdf.shape[0] - 1), np.clip(j, 0, esdf.shape[1] - 1)], np.inf)
    d = ((1 - fx) * (1 - fy) * at(x0, y0) + fx * (1 - fy) * at(x0 + 1, y0)
         + (1 - fx) * fy * at(x0, y0 + 1) + fx * fy * at(x0 + 1, y0 + 1))
    return np.nan_to_num(d, nan=np.inf).min(axis=1) - res / 2


def safe_command(center, yaw, v, w, esdf, origin, res):
    """First command in (policy, turn in place, turn the other way, creep forward, creep forward
    turning either way, straight back, stop) whose swept footprint keeps GUARD_MARGIN over
    GUARD_HORIZON. When the robot is already closer than the margin, a command also passes if it
    never gets closer than now (within ESCAPE_TOL, and never below 0), so it can drive along a wall
    it is hugging instead of freezing. Creeping forward covers a robot backed against a wall, where
    every turn sweeps a rear corner into it."""
    n = int(GUARD_HORIZON / ROLLOUT_DT)
    now = float(footprint_clearance(rollout(center, yaw, 0.0, 0.0, steps=0), esdf, origin, res)[0])
    turn = np.sign(w) * max(abs(w), 0.3) if w != 0.0 else 0.3
    options = ((v, w), (0.0, turn), (0.0, -turn), (CREEP_V, 0.0), (CREEP_V, turn), (CREEP_V, -turn), (V_MIN, 0.0))
    for k, (cv, cw) in enumerate(options):
        clr = footprint_clearance(rollout(center, yaw, cv, cw, steps=n), esdf, origin, res)[1:]
        if clr.min() >= GUARD_MARGIN or (now < GUARD_MARGIN and clr.min() >= max(0.0, now - ESCAPE_TOL)):
            return cv, cw, k
    return 0.0, 0.0, len(options)


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
        self.stuck_count, self.stuck_since = 0, None

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
        requested = abs(v) > 0.05 or abs(w) > 0.1

        v, w, reason = safe_command(center, yaw, v, w, esdf, origin, GRID_RES)
        status = self.GUARD_REASONS[reason]
        # Judged on what the policy asks for, so "keeps asking, guard keeps refusing" counts as stuck too.
        if self.stuck_since is None and not self.stuck and self._check_stuck(stamp, center, yaw, requested):
            self.stuck_count += 1
            self.stuck_since = stamp
            self.stuck = self.stuck_count >= STUCK_RETRIES  # out of retries: stop until the next goal
        if self.stuck_since is not None and not self.stuck and stamp - self.stuck_since >= STUCK_HOLD:
            self.policy.reset()  # retry: let the policy decide afresh instead of repeating itself
            self.motion, self.stuck_since = [], None
        if self.stuck or self.stuck_since is not None:
            v, w, status = 0.0, 0.0, "stuck" if self.stuck else f"stuck, retry {self.stuck_count}/{STUCK_RETRIES - 1}"
        return v, w, rollout(center, yaw, v, w), bev, status
