"""Shared scaffolding for capture-path priors (stair hint, capture speed, ...).

Every such prior rides on `poses.npy` (timestamp -> 4x4) that build_map saves,
labels each capture-path sample offline, and at nav time looks up the nearest
labelled sample to the robot's pose-in-map within an association radius. This
module holds the parts that are identical across priors; the per-prior labelling
and the *meaning* of the label live in the individual modules (stair_hint, path_speed).

Pure numpy; no ROS.
"""
from __future__ import annotations
import numpy as np

# Nav-time trajectory-association radius: how close the robot must be to the capture
# path for that path point's label to apply. Beyond it the robot is off the recorded
# trajectory, so the label is not trusted (the safe default is left to each caller).
ASSOC_M = 1.5


def poses_to_positions(poses) -> np.ndarray:
    """poses: dict{timestamp:int -> 4x4} (as saved by build_map). Returns (N,3)
    translations ordered by timestamp (== capture order)."""
    keys = sorted(poses.keys())
    return np.array([np.asarray(poses[k])[:3, 3] for k in keys], dtype=np.float64)


def horizontal_arclength(pos) -> np.ndarray:
    """Cumulative horizontal (xy) arclength along an ordered (N,3) path."""
    dxy = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(dxy)])


class PathSampleIndex:
    """Nav-time nearest-capture-sample lookup. pts is (N,>=4) with columns
    [x, y, z, label...]. The nearest sample is found in full 3D so stacked floors
    don't alias; assoc_m is the trajectory-association radius -- beyond it the robot
    is off the recorded path. Subclasses interpret column 3 (see nearest_value)."""

    def __init__(self, pts: np.ndarray, assoc_m: float = ASSOC_M):
        self.pts = np.asarray(pts, dtype=np.float64)
        self.assoc_m = float(assoc_m)

    @classmethod
    def load(cls, npy_path: str, **kw) -> "PathSampleIndex":
        return cls(np.load(npy_path), **kw)

    def nearest_value(self, position_xyz):
        """Column-3 label of the nearest capture-path sample within assoc_m, or None
        when the index is empty or the robot is off the recorded path (>assoc_m)."""
        if self.pts.shape[0] == 0:
            return None
        p = np.asarray(position_xyz, dtype=np.float64)[:3]
        d3 = np.linalg.norm(self.pts[:, :3] - p, axis=1)
        i = int(np.argmin(d3))
        if d3[i] > self.assoc_m:
            return None
        return float(self.pts[i, 3])
