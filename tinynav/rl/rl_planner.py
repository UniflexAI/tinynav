"""RL local planner used by planning_node when TINYNAV_PLANNER=rl.

Builds the same observation the policy saw in NavSim (BEV + own trail + goal/velocity/last
action), runs the recurrent policy and returns (v, w). An ESDF guard rolls the command out for
GUARD_HORIZON seconds and drops the forward speed (keeping the turn) if the footprint would
enter an obstacle.
"""
import asyncio
import os

import numpy as np

from tinynav.core.robot_specs import ROBOT_CONFIG
from tinynav.rl.bev import GRID_RES, TRAIL_LEN, camera_yaw, make_bev, to_robot_frame, trail_channel

# Action scaling shared with NavSim (which imports these): action[0] in [-1, 1] maps to
# [V_MIN, V_MAX] m/s with 0 -> 0, action[1] to [-W_MAX, W_MAX] rad/s. Kept torch-free for the robot.
V_MAX, V_MIN, W_MAX = ROBOT_CONFIG.max_linear_vel, -0.3, ROBOT_CONFIG.max_angular_vel
ROLLOUT_DT, ROLLOUT_STEPS = 0.1, 20  # 2s path for visualization / path-following controllers
GUARD_HORIZON = 1.0


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


def footprint_hits(poses, esdf, origin, res):
    """True if any footprint edge point of any pose lies in an (inflated) obstacle cell."""
    fl, rl, hw = ROBOT_CONFIG.footprint_from_control()
    f = np.arange(-rl, fl + 1e-6, res / 2)
    s = np.arange(-hw, hw + 1e-6, res / 2)
    pts = np.concatenate([np.stack([f, np.full_like(f, hw)], 1), np.stack([f, np.full_like(f, -hw)], 1),
                          np.stack([np.full_like(s, fl), s], 1), np.stack([np.full_like(s, -rl), s], 1)])
    c, sn = np.cos(poses[:, 2:3]), np.sin(poses[:, 2:3])
    wx = poses[:, :1] + c * pts[:, 0] - sn * pts[:, 1]
    wy = poses[:, 1:2] + sn * pts[:, 0] + c * pts[:, 1]
    ix = np.floor((wx - origin[0]) / res).astype(int)
    iy = np.floor((wy - origin[1]) / res).astype(int)
    ok = (ix >= 0) & (ix < esdf.shape[0]) & (iy >= 0) & (iy < esdf.shape[1])
    return bool((esdf[ix[ok], iy[ok]] < res / 2).any())


class RLPlanner:
    def __init__(self, policy):
        self.policy = policy
        self.reset()

    def reset(self):
        """New goal: clear the GRU memory, own trail and velocity estimate."""
        self.policy.reset()
        self.hist, self.last = [], None
        self.vel = np.zeros(2, np.float32)
        self.last_action = np.zeros(2, np.float32)

    def step(self, grid, origin, T, stamp, goal_xy, esdf):
        """grid/origin: planning occupancy grid; T: camera->world; esdf: 2D ESDF on the same grid.
        Returns (v, w, rollout poses, policy BEV, guard_triggered)."""
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

        poses = rollout(center, yaw, v, w)
        guard = v != 0.0 and footprint_hits(poses[1:int(GUARD_HORIZON / ROLLOUT_DT) + 1], esdf, origin, GRID_RES)
        if guard:
            v = 0.0
            poses = rollout(center, yaw, v, w)
        return v, w, poses, bev, guard
