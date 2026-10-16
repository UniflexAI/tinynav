"""Capture-speed prior from the capture trajectory.

Rationale: with 先建图后导航 the robot at nav time is always near the capture
path, so "how fast is safe here?" can borrow "how fast did the operator drive
through here?". The operator naturally slows in tight / cluttered / descending /
sharply-turning spots and opens up in clear corridors — so a single capture-speed
prior subsumes the would-be narrow-passage, down-slope and curvature priors
(they all show up as a slower capture speed). At nav time this prior becomes the
open-space TARGET of the planner's clearance-scaled speed schedule (see
planning_node._open_target_speed): the online clearance only scales speed *down*
from the operator's speed, so two governors compose — "how fast the operator went
here" (the ceiling) and "how much room is ahead right now" (the reduction).

Pure numpy; no ROS. Rides on `poses.npy` (timestamp -> 4x4) which build_map saves,
using the timestamps to recover speed. Shared lookup scaffolding lives in path_prior.
"""
from __future__ import annotations
import os

import numpy as np
from tinynav.core.path_prior import (
    poses_to_positions, horizontal_arclength, is_stale, PathSampleIndex)

# How much faster than the operator to replay. Multiplies the prior for BOTH of its
# consumers -- planning's open-space target and map_node's carrot horizon -- so they
# stay one speed. Set from the environment because pilot spawns both nodes with no
# `--ros-args`.
CAPTURE_SPEED_GAIN = float(os.environ.get('TINYNAV_CAPTURE_SPEED_GAIN', '1.0'))

# The carrot sits at a TIME horizon along the path, so how far it sits rides the speed
# actually driven: the capture-speed prior times CAPTURE_SPEED_GAIN. map_node's target
# pose is LOOKAHEAD_S ahead; planning may hold a shorter horizon of its own.
_LOOKAHEAD_S = 5.0
# No prior (off-path, or a map with no path_speed.npy): planning falls back to vx_max for
# the speed, so the carrot falls back to the same number.
_NO_CAP_SPEED_MPS = 0.6
# Bounds on the resulting distance: planning's [vx_min, vx_hard_max] x _LOOKAHEAD_S, as
# metres, so neither node needs the other's parameters to stay consistent.
_LOOKAHEAD_MIN_M = 1.0
_LOOKAHEAD_MAX_M = 5.0


def lookahead_distance_m(speed_cap_mps: float, gain: float = CAPTURE_SPEED_GAIN,
                         horizon_s: float = _LOOKAHEAD_S) -> float:
    """How far along the path a carrot `horizon_s` ahead sits, given the capture-speed
    prior. `speed_cap_mps` is what /planning/speed_cap carries: +inf (or NaN) means
    off-path or no prior. `gain` is applied on the prior branch only -- the no-prior
    fallback is planning's ungained vx_max."""
    speed = speed_cap_mps * gain if np.isfinite(speed_cap_mps) else _NO_CAP_SPEED_MPS
    return float(np.clip(speed * horizon_s, _LOOKAHEAD_MIN_M, _LOOKAHEAD_MAX_M))

# Defaults (tunable).
WIN_M = 1.0            # half-window horizontal arclength (m) for local speed aggregation
SPEED_PCT = 75.0       # percentile of in-window segment speeds -> the representative
                       # "cruising" speed here; a high percentile ignores momentary
                       # stops (operator pausing to look) instead of being dragged to ~0
MAX_SPEED = 3.0        # segment speed above this = VIO teleport / bad dt -> dropped
MIN_DT_S = 1e-3        # segments with dt <= this are dropped (avoids div blow-up)


def compute_path_speed(poses, win_m=WIN_M, pct=SPEED_PCT,
                       max_speed=MAX_SPEED, min_dt_s=MIN_DT_S) -> np.ndarray:
    """Label each capture-path sample with the operator's local speed (m/s).

    poses: dict{timestamp_ns:int -> 4x4} (as saved by build_map). Returns (N,4)
    float array [x, y, z, speed]. speed is the `pct`-percentile of valid segment
    speeds within a +/-win_m horizontal-arclength window (NaN where no valid
    segment is in range — treated as "unknown / no cap" at nav time).
    """
    keys = sorted(poses.keys())
    n = len(keys)
    out = np.zeros((n, 4), dtype=np.float32)
    if n == 0:
        return out
    pos = poses_to_positions(poses)                # (N,3), same order as sorted keys
    out[:, :3] = pos
    out[:, 3] = np.nan
    if n < 2:
        return out
    t_s = np.array(keys, dtype=np.float64) * 1e-9  # ns -> s (capture order)
    seg_dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dt = np.diff(t_s)
    with np.errstate(divide='ignore', invalid='ignore'):
        seg_speed = seg_dist / dt
    valid = (dt > min_dt_s) & np.isfinite(seg_speed) & (seg_speed <= max_speed)
    # window segments by the arclength of their midpoints (same horizontal windowing
    # as compute_path_climb); speed itself is from 3D motion above.
    s = horizontal_arclength(pos)
    s_mid = 0.5 * (s[:-1] + s[1:])
    for i in range(n):
        lo = np.searchsorted(s_mid, s[i] - win_m, side='left')
        hi = np.searchsorted(s_mid, s[i] + win_m, side='right')
        w = seg_speed[lo:hi][valid[lo:hi]]
        if w.size:
            out[i, 3] = np.percentile(w, pct)
    return out


class PathSpeedIndex(PathSampleIndex):
    """Nav-time lookup: what forward speed did the operator use near here?"""

    def speed_cap(self, position_xyz) -> float:
        """Capture speed (m/s) at the nearest capture-path sample within assoc_m.
        Returns +inf when off-path (beyond assoc_m) or the nearest sample has no
        valid speed — i.e. "no cap", the fail-safe (never restricts on missing data;
        the online clearance schedule still governs)."""
        v = self.nearest_value(position_xyz)
        return v if (v is not None and np.isfinite(v) and v > 0.0) else float('inf')


def bake(map_path: str, force: bool = False) -> str:
    """Write `path_speed.npy` into `map_path`; returns a one-line summary.

    The counterpart of `path_climb.bake`, with the same contract: rebake when the
    labels are older than the poses, and never raise for an unusable map -- no speed
    prior means planning falls back to vx_max, which is the fail-safe."""
    out_path = os.path.join(map_path, 'path_speed.npy')
    if not force and not is_stale(map_path, 'path_speed.npy'):
        return 'path_speed.npy: already baked'
    poses_path = os.path.join(map_path, 'poses.npy')
    if not os.path.exists(poses_path):
        return 'path_speed.npy: skipped, no poses.npy'
    speeds = compute_path_speed(np.load(poses_path, allow_pickle=True).item())
    np.save(out_path, speeds)
    finite = np.isfinite(speeds[:, 3])
    med = float(np.median(speeds[finite, 3])) if finite.any() else float('nan')
    return f'path_speed.npy: median capture speed {med:.2f} m/s'
