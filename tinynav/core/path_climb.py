"""Where the capture path climbs — the region in which a riser is a step, not a wall.

Rationale: with 先建图后导航 the global planner stays near the capture path
(SDF seeds = capture trajectory). So "may I step up here?" reduces to "did the
capture path go up/down here?". We label each capture-path sample offline (at
bake time) as climbing/flat using a SUSTAINED NET vertical change over a
horizontal window — robust to quadruped gait bob (near-zero net change, and its
up/down steps split the sign vote) and to VIO teleports (rejected by
max_step_dz). A single riser or a small platform must also qualify, so the sign
vote ignores the flat part of the window and the rise threshold sits below one
step.

The label is consumed as a REGION (PilotMapNode publishes the samples, the planner
relaxes per cell), so a miss costs strictness — the safe direction, which is why
min_rise sits low.

Pure numpy; no ROS — rides on `poses.npy` which build_map already saves. Shared
lookup scaffolding lives in path_prior, alongside the capture-speed prior.
"""
from __future__ import annotations

import os

import numpy as np

from tinynav.core.path_prior import (
    poses_to_positions, horizontal_arclength, is_stale, PathSampleIndex)

# Defaults (tunable). A single riser (~0.15 m) must qualify, so min_rise sits below
# one step; gait bob is rejected by net (~0 over the window) and by the sign vote.
WIN_M = 1.0            # half-window horizontal arclength (m)
MIN_RISE = 0.12        # min sustained net |dz| over the window to call it climbing (m)
CONSISTENCY = 0.6      # min fraction of *moving* in-window steps whose dz sign matches the net
NOISE_DZ = 0.02        # |dz| below this is flat: excluded from the sign vote (m)
MAX_STEP_DZ = 0.5      # a single consecutive |dz| above this = VIO teleport -> not climbing
# Column 3 of a label array is a float; this is the one place that reads it as a flag.
CLIMBING = 0.5


def n_climbing(labels) -> int:
    """How many samples of a (N,4) label array are climbing."""
    return int((np.asarray(labels)[:, 3] >= CLIMBING).sum())


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
    """Nav-time lookup: the region form for the publisher, the point form for the
    app's indicator."""

    def climbing_within(self, position_xyz, radius_m: float) -> np.ndarray:
        """The (M,3) positions of climbing samples within radius_m horizontally."""
        return self.samples_within(position_xyz, radius_m, min_label=CLIMBING)

    def on_stairs(self, position_xyz) -> bool:
        """Whether the nearest sample within assoc_m is climbing. Off the recorded
        path the label is not trusted => flat/strict, the safe default."""
        v = self.nearest_value(position_xyz)
        return v is not None and v >= CLIMBING


def bake(map_path: str, force: bool = False) -> str:
    """Write `path_climb.npy` into `map_path`; returns a one-line summary.

    Rebakes when the labels are older than the poses they came from (see
    `path_prior.is_stale`), so a reloop that rewrites the poses does not leave a
    stale region behind.

    Never raises for an unusable map: callers sit in the map-build and map-load
    paths, where "no climb labels" must degrade to strict everywhere, not fail."""
    out_path = os.path.join(map_path, 'path_climb.npy')
    if not force and not is_stale(map_path, 'path_climb.npy'):
        return 'path_climb.npy: already baked'
    poses_path = os.path.join(map_path, 'poses.npy')
    if not os.path.exists(poses_path):
        return 'path_climb.npy: skipped, no poses.npy'
    labels = compute_path_climb(np.load(poses_path, allow_pickle=True).item())
    np.save(out_path, labels)
    return f'path_climb.npy: {n_climbing(labels)}/{len(labels)} samples climbing'
