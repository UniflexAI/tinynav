"""Stair hint from the capture trajectory.

Rationale: with 先建图后导航 the global planner stays near the capture path
(SDF seeds = capture trajectory). So "am I heading into stairs?" reduces to
"does the capture path here go up/down a sustained flight?". We label each
capture-path sample offline (at build time) as climbing/flat using a SUSTAINED
NET vertical change over a horizontal window — robust to quadruped gait bob
(near-zero net change, and its up/down steps split the sign vote) and to VIO
teleports (rejected by max_step_dz). A single riser or a small platform must
also qualify, so the sign vote ignores the flat part of the window and the rise
threshold sits below one step. At nav time a tiny node looks
up the nearest labelled sample to the robot's pose-in-map and emits a boolean.

Pure numpy; no ROS, no occupancy grid — rides on `poses.npy` which build_map
already saves. Shared lookup scaffolding lives in path_prior.
"""
from __future__ import annotations
import numpy as np
from tinynav.core.path_prior import poses_to_positions, horizontal_arclength, PathSampleIndex

# Defaults (tunable). A single riser (~0.15 m) must qualify, so min_rise sits below
# one step; gait bob is rejected by net (~0 over the window) and by the sign vote.
WIN_M = 1.0            # half-window horizontal arclength (m)
MIN_RISE = 0.12        # min sustained net |dz| over the window to call it climbing (m)
CONSISTENCY = 0.6      # min fraction of *moving* in-window steps whose dz sign matches the net
NOISE_DZ = 0.02        # |dz| below this is flat: excluded from the sign vote (m)
MAX_STEP_DZ = 0.5      # a single consecutive |dz| above this = VIO teleport -> not climbing


def compute_path_climb(poses, win_m=WIN_M, min_rise=MIN_RISE,
                       consistency=CONSISTENCY, max_step_dz=MAX_STEP_DZ,
                       noise_dz=NOISE_DZ) -> np.ndarray:
    """Label each capture-path sample climbing (1) or flat (0).

    Returns (N,4) float array: columns [x, y, z, is_climbing]. is_climbing is
    1.0 where the path shows a sustained net vertical change over a +/-win_m
    horizontal window with consistent sign (robust to gait bob / ramp crest).
    """
    pos = poses_to_positions(poses)
    n = len(pos)
    out = np.zeros((n, 4), dtype=np.float32)
    if n == 0:
        return out
    out[:, :3] = pos
    if n < 3:
        return out
    s = horizontal_arclength(pos)
    z = pos[:, 2]
    for i in range(n):
        j0 = np.searchsorted(s, s[i] - win_m, side='left')
        j1 = np.searchsorted(s, s[i] + win_m, side='right')
        if j1 - j0 < 3:
            continue
        seg_z = z[j0:j1]
        steps = np.diff(seg_z)
        if steps.size == 0 or np.abs(steps).max() > max_step_dz:
            continue                       # teleport / discontinuity in window
        net = seg_z[-1] - seg_z[0]
        if abs(net) < min_rise:
            continue
        # Vote only over steps that actually move in z: a single riser (or a small
        # platform) climbs in a fraction of the window, and counting the flat
        # remainder as disagreement would veto it.
        moving = np.abs(steps) >= noise_dz
        if not np.any(moving):
            continue
        same = np.mean(np.sign(steps[moving]) == np.sign(net))
        if same >= consistency:
            out[i, 3] = 1.0
    return out


class PathClimbIndex(PathSampleIndex):
    """Nav-time lookup: is the robot on a climbing stretch of the capture path?"""

    def on_stairs(self, position_xyz) -> bool:
        """True if the robot's nearest capture-path sample is within assoc_m and is
        labelled climbing. Off the recorded path (>assoc_m) the label is not trusted
        => flat/strict, the safe default. Lead before the flight comes from the
        +/-WIN_M labelling window, not from the association radius."""
        v = self.nearest_value(position_xyz)
        return v is not None and v >= 0.5
