"""Turn human-driven recordings into (observation, action) demonstrations for the RL policy.

Input: a bag with /slam/depth + /slam/odometry_visual + camera_info (run perception_node over a raw
stereo bag first). For every synced frame (the planner's ~7Hz tick) this rebuilds exactly what
RLPlanner feeds the policy: BEV + own trail, goal / velocity / last action. The action is the
driver's (v, w) from pose differences; the goal is hindsight: the recorded pose GOAL_AHEAD metres
further along the path, like map_node's carrot. Frames are labelled with the RL footprint's side
clearances (narrow passages) and whether the 1.0 x 0.5m footprint stays clear of the obstacles over
the next second, so demos from a smaller robot (go2) that squeeze through gaps can be dropped.

Usage:
  uv run python -m tinynav.rl.demos --bag <processed_bag> --out demos/<name>.npz
"""
import argparse

import numpy as np
from cv_bridge import CvBridge
from scipy.ndimage import distance_transform_edt

from tinynav.core.math_utils import msg2np

from tinynav.core.planning_node import build_obstacle_map, update_occupancy_grid
from tinynav.core.robot_specs import RL_FOOTPRINT, ROBOT_CONFIG
from tinynav.rl.bev import GRID_RES, GRID_SHAPE, RAY_STEP, TRAIL_LEN, camera_yaw, make_bev, to_robot_frame, trail_channel
from tinynav.rl.export_bev import DEPTH, INFO_TOPICS, ODOM, read_bag, stamp_ns
from tinynav.rl.rl_planner import EDGE, GUARD_MARGIN, V_MAX, V_MIN, W_MAX, footprint_clearance

GOAL_AHEAD = 2.2  # m along the recorded path (map_node's carrot is ~2.1m ahead)
SMOOTH = 3  # frames in the centered moving average of the driver's (v, w)
NARROW = 0.4  # m: both sides of the RL footprint closer than this -> narrow passage (< 1.3m wide)


def side_gaps(center, yaw, esdf, origin):
    """Distance (m) from the RL footprint's left / right edge to the nearest obstacle surface."""
    out, c, s = [], np.cos(yaw), np.sin(yaw)
    for sign in (1, -1):
        pts = EDGE[np.isclose(EDGE[:, 1], sign * RL_FOOTPRINT.half_width)]
        side_pts = np.asarray(center) + np.stack([c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]], 1)
        normal = sign * np.array([-s, c])  # probe outward from the edge until an obstacle surface (or 1.5m)
        gaps = []
        for p in side_pts:
            for d in np.arange(0.0, 1.5, GRID_RES / 2):
                q = p + d * normal
                i, j = int((q[0] - origin[0]) / GRID_RES), int((q[1] - origin[1]) / GRID_RES)
                if not (0 <= i < esdf.shape[0] and 0 <= j < esdf.shape[1]):
                    break
                if esdf[i, j] < GRID_RES / 2:
                    gaps.append(d)
                    break
            else:
                gaps.append(1.5)
        out.append(min(gaps) if gaps else 1.5)
    return out


def extract(bag):
    bridge = CvBridge()
    grid, origin = np.zeros(GRID_SHAPE), np.array(GRID_SHAPE) * GRID_RES / -2.
    K, depths, odoms = None, {}, {}
    frames = []  # per synced frame: stamp, control center, yaw, perception BEV, ESDF
    for topic, msg in read_bag(bag):
        if topic in INFO_TOPICS and K is None:
            K = np.array(msg.k).reshape(3, 3)
        elif topic in (DEPTH, ODOM):
            (depths if topic == DEPTH else odoms)[stamp_ns(msg)] = msg
        synced = depths.keys() & odoms.keys()
        if K is None or not synced:
            continue
        t = min(synced)
        depth = bridge.imgmsg_to_cv2(depths.pop(t), desired_encoding="32FC1")
        T, _ = msg2np(odoms.pop(t))
        depths = {k: v for k, v in depths.items() if k > t}
        odoms = {k: v for k, v in odoms.items() if k > t}
        grid, origin = update_occupancy_grid(grid, origin, depth, T, K, GRID_RES, RAY_STEP)
        obstacle = build_obstacle_map(grid, origin, GRID_RES, robot_z=T[2, 3], config=ROBOT_CONFIG.obstacle)
        esdf = (distance_transform_edt(~obstacle) * GRID_RES).astype(np.float32)
        center = (T[:3, 3] - T[:3, :3] @ ROBOT_CONFIG.cam_offset_3d)[:2]
        yaw = camera_yaw(T)
        bev = make_bev(grid, origin, GRID_RES, center, yaw, ref_z=T[2, 3])
        frames.append(dict(t=t * 1e-9, center=center, yaw=yaw, bev=bev, esdf=esdf, origin=origin.copy()))
    return frames


def label(frames):
    n = len(frames)
    t = np.array([f["t"] for f in frames])
    xy = np.array([f["center"] for f in frames])
    yaw = np.unwrap(np.array([f["yaw"] for f in frames]))
    # driver's command: forward speed and yaw rate from pose differences, centered smoothing
    dt = np.gradient(t).clip(1e-3)
    fwd = np.stack([np.cos(yaw), np.sin(yaw)], 1)
    v = (np.gradient(xy, axis=0) * fwd).sum(1) / dt
    w = np.gradient(yaw) / dt
    k = np.ones(SMOOTH) / SMOOTH
    v, w = np.convolve(v, k, "same"), np.convolve(w, k, "same")
    action = np.stack([np.where(v >= 0, v / V_MAX, v / -V_MIN), w / W_MAX], 1).clip(-1, 1).astype(np.float32)
    # runtime velocity estimate (RLPlanner: EMA 0.5 of raw differences) and teacher-forced last action
    vel = np.zeros((n, 2), np.float32)
    for i in range(1, n):
        raw = np.array([to_robot_frame(xy[i], xy[i - 1], yaw[i - 1])[0] / (t[i] - t[i - 1]), (yaw[i] - yaw[i - 1]) / (t[i] - t[i - 1])])
        vel[i] = 0.5 * vel[i - 1] + 0.5 * raw
    last_action = np.vstack([np.zeros((1, 2), np.float32), action[:-1]])
    # hindsight goal: first recorded pose GOAL_AHEAD metres further along the path
    arc = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    ahead = np.searchsorted(arc, arc + GOAL_AHEAD).clip(max=n - 1)
    goal = np.array([to_robot_frame(xy[j], xy[i], yaw[i]) for i, j in enumerate(ahead)], np.float32)
    vec = np.concatenate([goal / 5.0, vel / np.array([V_MAX, W_MAX], np.float32), last_action], 1)
    # own trail and safety / narrow labels
    bev, safe, gaps = [], np.zeros(n, bool), np.zeros((n, 2), np.float32)
    nxt = int(round(1.0 / max(np.median(np.diff(t)), 1e-3)))  # frames in the next second
    for i, f in enumerate(frames):
        lo = max(0, i - TRAIL_LEN + 1)
        trail = trail_channel(xy[lo:i + 1][::-1], np.arange(i + 1 - lo), xy[i], yaw[i])
        bev.append(np.concatenate([f["bev"], trail[None]]))
        poses = np.c_[xy[i:i + nxt + 1], yaw[i:i + nxt + 1]]
        safe[i] = footprint_clearance(poses, f["esdf"], f["origin"], GRID_RES).min() >= GUARD_MARGIN
        gaps[i] = side_gaps(xy[i], yaw[i], f["esdf"], f["origin"])
    narrow = (gaps < NARROW).all(1)
    moving = (np.abs(v) > 0.05) | (np.abs(w) > 0.1)
    return dict(bev=np.stack(bev).astype(np.float16), vec=vec, action=action, t=t, xy=xy.astype(np.float32),
                yaw=yaw.astype(np.float32), safe=safe, narrow=narrow, gaps=gaps, moving=moving)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = label(extract(args.bag))
    np.savez_compressed(args.out, **d)
    n = len(d["t"])
    print(f"{args.out}: {n} frames over {d['t'][-1] - d['t'][0]:.0f}s, path {np.hypot(*np.diff(d['xy'], axis=0).T).sum():.0f} m; "
          f"moving {d['moving'].mean():.0%}, safe for the RL footprint {d['safe'].mean():.0%}, narrow {d['narrow'].mean():.1%} "
          f"(narrow & safe & moving: {(d['narrow'] & d['safe'] & d['moving']).sum()} frames); "
          f"|v| mean {np.abs(d['action'][:, 0]).mean() * V_MAX:.2f} m/s")


if __name__ == "__main__":
    main()
